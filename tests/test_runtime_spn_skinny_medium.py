from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_attribution import MODELS
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_medium import (
    adjudicate_runtime_spn_skinny_medium,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_STEM = "i1_rtg2a_skinny64_general_gf2_medium_65536"


def _rows(seed: int, aucs: dict[str, float]) -> list[dict[str, object]]:
    cache_root = (
        "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        f"{RUN_STEM}_seed{seed}_20260724\\cache"
    )
    common_training = {
        "epochs": 5,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "batch_size": 64,
        "train_eval_interval": 1,
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "selected_checkpoint": "best",
        "train_rows": 131072,
        "validation_rows": 65536,
        "model_options": {
            "processor_steps": 2,
            "pair_embedding_dim": 128,
            "dropout": 0.0,
            "sbox_context_mode": "late_pair",
        },
        "dataset_cache_root": cache_root,
        "dataset_cache_chunk_size": 1024,
        "dataset_cache_workers": 1,
        "train_dataset_storage": "disk",
        "validation_dataset_storage": "disk",
    }
    return [
        {
            "cipher": "SKINNY-64/64",
            "model": model,
            "rounds": 7,
            "seed": seed,
            "samples_per_class": 65536,
            "dataset_label_mode": "balanced_per_class",
            "pairs_per_sample": 4,
            "feature_encoding": "ciphertext_pair_bits",
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "independent_pairs",
            "difference_profile": "skinny64_gohr2022_single_key",
            "difference_member": 0,
            "input_difference": 0x2000,
            "train_key": 0,
            "validation_key": 0x1111111111111111,
            "parameter_count": 442466,
            "trainable_parameter_count": 442466,
            "input_bit_order": "project_msb_to_runtime_lsb",
            "metrics": {"auc": aucs[role]},
            "training": common_training.copy(),
        }
        for role, model in MODELS.items()
    ]


def test_medium_seed0_pass_authorizes_only_identical_seed1() -> None:
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="seed0",
        rows=_rows(0, {"true": 0.61, "corrupted": 0.56, "independent": 0.53}),
        expected_seed=0,
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_rtg2a_skinny_medium_seed0_supported"
    assert gate["next_action"] == "run the identical conditional seed1 RTG2-A matrix"
    assert all(gate["protocol_checks"].values())


def test_medium_seed0_hold_blocks_seed1_and_larger_scale() -> None:
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="held",
        rows=_rows(0, {"true": 0.552, "corrupted": 0.550, "independent": 0.51}),
        expected_seed=0,
    )

    assert gate["status"] == "hold"
    assert "launch seed1" in gate["blocked_actions"][-1]
    assert any("262144/class" in action for action in gate["blocked_actions"])


def test_medium_protocol_mismatch_fails_closed() -> None:
    rows = _rows(0, {"true": 0.61, "corrupted": 0.56, "independent": 0.53})
    rows[1]["samples_per_class"] = 65535

    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="invalid",
        rows=rows,
        expected_seed=0,
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_rtg2a_skinny_medium_protocol_invalid"


def test_medium_plans_change_only_scale_seed_and_descriptive_fields() -> None:
    import csv

    frozen_fields = {
        "cipher",
        "structure",
        "model_key",
        "architecture_rank",
        "score",
        "rounds",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "train_key",
        "validation_key",
        "key_rotation_interval",
        "sample_structure",
        "integral_active_nibble",
        "difference_profile",
        "difference_member",
        "loss",
        "learning_rate",
        "optimizer",
        "optimizer_state_transition",
        "weight_decay",
        "lr_scheduler",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "target_epochs",
        "pretrain_epochs",
        "model_options",
    }
    with (ROOT / "configs/experiment/innovation1/innovation1_spn_skinny64_runtime_e4_attribution_t2c_2048_seed0.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        anchor = list(csv.DictReader(handle))

    for seed in (0, 1):
        path = ROOT / (
            "configs/experiment/innovation1/"
            f"innovation1_spn_skinny64_runtime_e4_medium_rtg2a_65536_seed{seed}.csv"
        )
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert {row["seed"] for row in rows} == {str(seed)}
        assert {row["samples_per_class"] for row in rows} == {"65536"}
        assert [row["model_key"] for row in rows] == [
            row["model_key"] for row in anchor
        ]
        for expected, observed in zip(anchor, rows, strict=True):
            assert {field: observed[field] for field in frozen_fields} == {
                field: expected[field] for field in frozen_fields
            }


def test_medium_remote_assets_are_ready_and_windows_safe() -> None:
    run_script = (
        ROOT
        / "configs/remote/generated/run_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.cmd"
    ).read_text(encoding="utf-8")
    launch_script = (
        ROOT
        / "configs/remote/generated/launch_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.cmd"
    ).read_text(encoding="utf-8")
    monitor_script = (
        ROOT
        / "configs/remote/generated/monitor_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.sh"
    ).read_text(encoding="utf-8")

    for seed in (0, 1):
        config_path = ROOT / (
            "configs/remote/"
            f"innovation1_rtg2a_skinny64_general_gf2_medium_65536_seed{seed}_gpu0_20260724.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report = remote_readiness_report(config_path)
        assert report["status"] == "pass", report["errors"]
        assert config["expected_rows"] == 3
        assert config["epochs"] == 5
        assert config["batch_size"] == 64
        assert config["dataset_cache_chunk_size"] == 1024
        assert config["dataset_cache_workers"] == 1
        assert config["dataset_cache_root"].startswith(
            "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        )

    combined = run_script + launch_script
    assert "cmd.exe /c" in launch_script
    assert "cmd.exe /k" not in combined
    assert "EnableDelayedExpansion" not in combined
    assert "!" not in combined
    assert "--dataset-cache-root \"%CACHE_ROOT%\"" in run_script
    assert "--dataset-cache-chunk-size 1024" in run_script
    assert "--dataset-cache-workers 1" in run_script
    assert '--progress "%LOG_DIR%\\progress.jsonl"' in run_script
    assert "--no-plot" in run_script
    assert "visual_qa_pending.marker" in run_script + monitor_script
    assert "seed0 gate did not pass" in launch_script
    assert "if errorlevel 1 (" in launch_script
    assert "scripts/index-results" in monitor_script
    assert "gate-runtime-spn-skinny-medium" in monitor_script
    assert "launch_i1_rtg2a" in monitor_script
    assert "medium_pair_complete.marker" in monitor_script
    assert "C:\\Users" not in combined.replace(
        "C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519",
        "",
    )


def test_medium_gate_cli_import_does_not_require_matplotlib() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import blockcipher_nd.cli.gate_runtime_spn_skinny_medium
print('import=pass')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import=pass"
