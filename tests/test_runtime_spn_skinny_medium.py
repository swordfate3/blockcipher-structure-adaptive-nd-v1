from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.cli.gate_runtime_spn_skinny_medium_joint import (
    main as joint_gate_main,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_attribution import MODELS
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_medium import (
    adjudicate_runtime_spn_skinny_medium,
    adjudicate_runtime_spn_skinny_medium_joint,
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
    rows = []
    for role, model in MODELS.items():
        auc = aucs[role]
        history = [
            {
                "epoch": float(epoch),
                "learning_rate": 0.0001,
                "train_loss": 0.25,
                "train_eval_loss": 0.693,
                "train_accuracy": 0.5,
                "train_auc": auc - 0.03 + 0.01 * epoch,
                "val_loss": 0.693,
                "val_accuracy": 0.5,
                "val_auc": auc - 0.05 + 0.01 * epoch,
            }
            for epoch in range(1, 6)
        ]
        training = {
            **common_training,
            "epochs_ran": 5,
            "best_epoch": 5,
            "best_checkpoint_metric": auc,
            "selected_checkpoint": "best",
            "stopped_epoch": 0,
        }
        rows.append(
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
                "metrics": {"auc": auc},
                "history": history,
                "training": training,
            }
        )
    return rows


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
    assert gate["training_dynamics"]["true"] == {
        "model": MODELS["true"],
        "best_epoch": 5,
        "epochs_ran": 5,
        "first_val_auc": pytest.approx(0.57),
        "best_val_auc": pytest.approx(0.61),
        "final_val_auc": pytest.approx(0.61),
        "best_train_auc": pytest.approx(0.63),
        "train_minus_val_auc_at_best": pytest.approx(0.02),
        "first_to_best_val_auc_gain": pytest.approx(0.04),
        "best_to_final_val_auc_change": pytest.approx(0.0),
    }


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


@pytest.mark.parametrize("malformation", ["missing_history", "wrong_best_epoch"])
def test_medium_checkpoint_history_mismatch_fails_closed(malformation: str) -> None:
    rows = _rows(0, {"true": 0.61, "corrupted": 0.56, "independent": 0.53})
    if malformation == "missing_history":
        rows[0]["history"] = rows[0]["history"][:-1]
    else:
        rows[0]["training"]["best_epoch"] = 4

    gate = adjudicate_runtime_spn_skinny_medium(
        run_id="invalid-history",
        rows=rows,
        expected_seed=0,
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation1_rtg2a_skinny_medium_protocol_invalid"
    assert gate["protocol_checks"]["complete_five_epoch_checkpoint_replay"] is False


def _seed_gate(seed: int, *, supported: bool = True) -> dict[str, object]:
    aucs = (
        {"true": 0.61, "corrupted": 0.56, "independent": 0.53}
        if supported
        else {"true": 0.552, "corrupted": 0.550, "independent": 0.51}
    )
    return adjudicate_runtime_spn_skinny_medium(
        run_id=f"{RUN_STEM}_seed{seed}_20260724",
        rows=_rows(seed, aucs),
        expected_seed=seed,
    )


def test_medium_joint_gate_requires_both_seeds_to_pass() -> None:
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), _seed_gate(1)],
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_rtg2a_skinny_medium_two_seed_supported"
    assert all(gate["protocol_checks"].values())
    assert gate["research_checks"]["both_medium_seeds_supported"] is True
    assert "262144/class" in gate["next_action"]


def test_medium_joint_gate_holds_when_seed1_is_not_supported() -> None:
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), _seed_gate(1, supported=False)],
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_rtg2a_skinny_medium_two_seed_not_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert gate["research_checks"]["both_medium_seeds_supported"] is False


def test_medium_joint_gate_fails_closed_on_duplicate_seed_sources() -> None:
    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), _seed_gate(0)],
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation1_rtg2a_skinny_medium_joint_protocol_invalid"
    )
    assert gate["protocol_checks"]["two_distinct_seed_gates_complete"] is False


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("run_id", "wrong-run", "source_run_ids_match_frozen_rtg2a"),
        (
            "thresholds",
            {"true_auc": 0.54, "control_margin": 0.005},
            "frozen_thresholds_preserved",
        ),
        (
            "aucs",
            {"true": "not-a-number", "corrupted": 0.56, "independent": 0.53},
            "finite_source_metrics",
        ),
        ("protocol_checks", [True], "source_protocol_checks_passed"),
        ("decision", "inconsistent", "source_gate_contracts_consistent"),
    ],
)
def test_medium_joint_gate_fails_closed_on_malformed_source_evidence(
    field: str,
    value: object,
    failed_check: str,
) -> None:
    seed1 = _seed_gate(1)
    seed1[field] = value

    gate = adjudicate_runtime_spn_skinny_medium_joint(
        run_id=f"{RUN_STEM}_joint_seed0_seed1_20260724",
        gates=[_seed_gate(0), seed1],
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation1_rtg2a_skinny_medium_joint_protocol_invalid"
    )
    assert gate["protocol_checks"][failed_check] is False


def test_medium_joint_gate_cli_writes_hashed_source_evidence(tmp_path: Path) -> None:
    source_paths = []
    for seed in (0, 1):
        path = tmp_path / f"seed{seed}" / "gate.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(_seed_gate(seed), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_paths.append(path)
    output_root = tmp_path / "joint"

    status = joint_gate_main(
        [
            "--run-id",
            f"{RUN_STEM}_joint_seed0_seed1_20260724",
            "--seed0-gate",
            str(source_paths[0]),
            "--seed1-gate",
            str(source_paths[1]),
            "--output-root",
            str(output_root),
        ]
    )

    assert status == 0
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (output_root / "validation.json").read_text(encoding="utf-8")
    )
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert validation["status"] == "pass"
    assert len(validation["sources"]) == 2
    assert all(len(source["sha256"]) == 64 for source in validation["sources"])
    assert summary["training_performed"] is False
    assert (output_root / "progress.jsonl").is_file()


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
    with (
        ROOT
        / "configs/experiment/innovation1/innovation1_spn_skinny64_runtime_e4_attribution_t2c_2048_seed0.csv"
    ).open(newline="", encoding="utf-8") as handle:
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
    assert '--dataset-cache-root "%CACHE_ROOT%"' in run_script
    assert "--dataset-cache-chunk-size 1024" in run_script
    assert "--dataset-cache-workers 1" in run_script
    assert '--progress "%LOG_DIR%\\progress.jsonl"' in run_script
    assert "--no-plot" in run_script
    assert "visual_qa_pending.marker" in run_script + monitor_script
    assert "seed0 gate did not pass" in launch_script
    assert "if errorlevel 1 (" in launch_script
    assert "scripts/index-results" in monitor_script
    assert "gate-runtime-spn-skinny-medium" in monitor_script
    assert "gate-runtime-spn-skinny-medium-joint" in monitor_script
    assert f'{RUN_STEM}_joint_seed0_seed1_20260724"' in monitor_script
    assert "adjudicate_joint || exit 5" in monitor_script
    assert "validated_destination_reused" in monitor_script
    assert monitor_script.count('if [[ ! -f "${destination}/gate.json" ]]') == 2
    assert monitor_script.count('return "${gate_exit}"') == 2
    assert "conditional_seed1_launched.marker" in monitor_script
    assert "conditional_seed1_launched_after_resume" in monitor_script
    assert "resumed_medium_pair_joint_adjudicated_and_indexed" in monitor_script
    assert "launch_i1_rtg2a" in monitor_script
    assert "medium_pair_complete.marker" in monitor_script
    subprocess.run(
        [
            "bash",
            "-n",
            str(
                ROOT
                / "configs/remote/generated/monitor_i1_rtg2a_skinny64_general_gf2_medium_65536_20260724.sh"
            ),
        ],
        check=True,
    )
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
