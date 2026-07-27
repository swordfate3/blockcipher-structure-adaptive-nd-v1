from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from blockcipher_nd.cli.gate_runtime_spn_rectangle_medium import main as medium_main
from blockcipher_nd.engine.matrix_runner import parse_args
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.tasks.innovation1.runtime_spn_rectangle_attribution import (
    DESCRIPTOR_SHA256,
    INPUT_DIFFERENCE,
    MEDIUM_SAMPLES_PER_CLASS,
    MEDIUM_TRAIN_ROWS,
    MEDIUM_VALIDATION_ROWS,
    MODELS,
    PROFILE,
    SCALE_SAMPLES_PER_CLASS,
    SCALE_TRAIN_ROWS,
    SCALE_VALIDATION_ROWS,
    adjudicate_runtime_spn_rectangle_attribution,
    adjudicate_runtime_spn_rectangle_medium,
    adjudicate_runtime_spn_rectangle_scale,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT / "configs/experiment/innovation1/innovation1_spn_rectangle80_runtime_e4_"
    "noncontiguous_attribution_rct1_2048_seed0_seed1.csv"
)
MEDIUM_PLAN = (
    ROOT / "configs/experiment/innovation1/innovation1_spn_rectangle80_runtime_e4_"
    "medium_rct2_65536_seed0.csv"
)
SCALE_PLAN = (
    ROOT / "configs/experiment/innovation1/innovation1_spn_rectangle80_runtime_e4_"
    "scale_rct3_262144_seed0.csv"
)


def _rows(
    aucs: dict[int, dict[str, float]],
) -> list[dict[str, object]]:
    model_options = {
        "runtime_structure_path": "configs/runtime/spn/rectangle64.json",
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "late_pair",
    }
    common_training = {
        "epochs": 5,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "selected_checkpoint": "best",
        "train_rows": 4096,
        "validation_rows": 2048,
        "input_bits": 512,
        "model_options": model_options,
        "train_dataset_storage": "disk",
        "validation_dataset_storage": "disk",
    }
    window_sha = {
        "true": "true-window",
        "corrupted": "corrupted-window",
        "independent": "true-window",
    }
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for role, model in MODELS.items():
            rows.append(
                {
                    "cipher": "RECTANGLE-80",
                    "cipher_key": "rectangle80",
                    "model": model,
                    "rounds": 6,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "dataset_label_mode": "balanced_per_class",
                    "pairs_per_sample": 4,
                    "feature_encoding": "ciphertext_pair_bits",
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "difference_profile": PROFILE,
                    "difference_member": 0,
                    "input_difference": INPUT_DIFFERENCE,
                    "train_key": 0,
                    "validation_key": 0x11111111111111111111,
                    "parameter_count": 442466,
                    "trainable_parameter_count": 442466,
                    "input_bit_order": "project_msb_to_runtime_lsb",
                    "runtime_structure_descriptor_name": (
                        "RECTANGLE-80 runtime SPN structure"
                    ),
                    "runtime_structure_descriptor_path": str(
                        ROOT / "configs/runtime/spn/rectangle64.json"
                    ),
                    "runtime_structure_descriptor_sha256": DESCRIPTOR_SHA256,
                    "runtime_structure_loaded_rounds": 2,
                    "runtime_structure_unique_transition_count": 1,
                    "runtime_structure_homogeneous": True,
                    "runtime_structure_mode": role,
                    "runtime_structure_window_sha256": window_sha[role],
                    "metrics": {"auc": aucs[seed][role]},
                    "training": common_training.copy(),
                }
            )
    return rows


def _scale_rows(
    aucs: dict[str, float],
    *,
    samples_per_class: int = MEDIUM_SAMPLES_PER_CLASS,
    train_rows: int = MEDIUM_TRAIN_ROWS,
    validation_rows: int = MEDIUM_VALIDATION_ROWS,
    cache_name: str = "rct2",
) -> list[dict[str, object]]:
    rows = _rows({0: aucs, 1: aucs})[:3]
    for row in rows:
        role = str(row["runtime_structure_mode"])
        auc = aucs[role]
        row["samples_per_class"] = samples_per_class
        row["history"] = [
            {
                "epoch": epoch,
                "train_loss": 0.25 - 0.01 * epoch,
                "train_auc": auc - 0.04 + 0.01 * epoch,
                "val_loss": 0.26 - 0.01 * epoch,
                "val_auc": auc - 0.05 + 0.01 * epoch,
            }
            for epoch in range(1, 6)
        ]
        training = row["training"]
        assert isinstance(training, dict)
        training.update(
            {
                "batch_size": 64,
                "train_eval_interval": 1,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "samples_total": train_rows,
                "positive_rows": samples_per_class,
                "negative_rows": samples_per_class,
                "pair_bits": 128,
                "dataset_cache_root": (
                    "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
                    f"{cache_name}\\cache"
                ),
                "dataset_cache_chunk_size": 1024,
                "dataset_cache_workers": 1,
                "device": "cuda",
                "epochs_ran": 5,
                "best_epoch": 5,
                "best_checkpoint_metric": auc,
                "stopped_epoch": 0,
            }
        )
        row["validation"] = {
            "samples_per_class": validation_rows // 2,
            "samples_total": validation_rows,
            "positive_rows": validation_rows // 2,
            "negative_rows": validation_rows // 2,
            "negative_mode": "encrypted_random_plaintexts",
        }
    return rows


def _medium_rows(aucs: dict[str, float]) -> list[dict[str, object]]:
    return _scale_rows(aucs)


def test_real_rectangle_rct1_plan_parses_exact_two_seed_matrix() -> None:
    tasks = build_tasks(parse_args(["--plan", str(PLAN)]))

    assert len(tasks) == 6
    assert {(task["seed"], task["model_key"]) for task in tasks} == {
        (seed, model) for seed in (0, 1) for model in MODELS.values()
    }
    assert {task["rounds"] for task in tasks} == {6}
    assert {task["train_key"] for task in tasks} == {0}
    assert {task["validation_key"] for task in tasks} == {0x11111111111111111111}
    assert {task["difference_profile"] for task in tasks} == {PROFILE}
    assert {task["model_options"]["runtime_structure_path"] for task in tasks} == {
        "configs/runtime/spn/rectangle64.json"
    }


def test_real_rectangle_rct2_plan_changes_only_medium_scale_and_seed() -> None:
    tasks = build_tasks(parse_args(["--plan", str(MEDIUM_PLAN)]))

    assert len(tasks) == 3
    assert {task["model_key"] for task in tasks} == set(MODELS.values())
    assert {task["rounds"] for task in tasks} == {6}
    assert {task["seed"] for task in tasks} == {0}
    assert {task["samples_per_class"] for task in tasks} == {MEDIUM_SAMPLES_PER_CLASS}
    assert {task["pairs_per_sample"] for task in tasks} == {4}
    assert {task["difference_profile"] for task in tasks} == {PROFILE}
    assert {task["model_options"]["runtime_structure_path"] for task in tasks} == {
        "configs/runtime/spn/rectangle64.json"
    }


def test_real_rectangle_rct3_plan_preserves_protocol_at_scale() -> None:
    tasks = build_tasks(parse_args(["--plan", str(SCALE_PLAN)]))

    assert len(tasks) == 3
    assert {task["model_key"] for task in tasks} == set(MODELS.values())
    assert {task["rounds"] for task in tasks} == {6}
    assert {task["seed"] for task in tasks} == {0}
    assert {task["samples_per_class"] for task in tasks} == {SCALE_SAMPLES_PER_CLASS}
    assert {task["pairs_per_sample"] for task in tasks} == {4}
    assert {task["negative_mode"] for task in tasks} == {"encrypted_random_plaintexts"}
    assert {task["difference_profile"] for task in tasks} == {PROFILE}
    assert {task["model_options"]["runtime_structure_path"] for task in tasks} == {
        "configs/runtime/spn/rectangle64.json"
    }


def test_rectangle_rct1_passes_only_when_both_seeds_beat_controls() -> None:
    gate = adjudicate_runtime_spn_rectangle_attribution(
        run_id="pass",
        rows=_rows(
            {
                0: {"true": 0.62, "corrupted": 0.57, "independent": 0.54},
                1: {"true": 0.60, "corrupted": 0.56, "independent": 0.53},
            }
        ),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_runtime_spn_rectangle_noncontiguous_attribution_supported"
    )
    assert gate["margins"]["0"]["true_minus_corrupted"] > 0.005
    assert gate["margins"]["1"]["true_minus_independent"] > 0.005


def test_rectangle_rct1_holds_when_one_seed_loses_to_control() -> None:
    gate = adjudicate_runtime_spn_rectangle_attribution(
        run_id="hold",
        rows=_rows(
            {
                0: {"true": 0.62, "corrupted": 0.57, "independent": 0.54},
                1: {"true": 0.58, "corrupted": 0.59, "independent": 0.53},
            }
        ),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation1_runtime_spn_rectangle_noncontiguous_attribution_not_supported"
    )


def test_rectangle_rct1_protocol_mismatch_fails_closed() -> None:
    rows = _rows(
        {
            0: {"true": 0.62, "corrupted": 0.57, "independent": 0.54},
            1: {"true": 0.60, "corrupted": 0.56, "independent": 0.53},
        }
    )
    rows[1]["runtime_structure_descriptor_sha256"] = "wrong"
    rows[2]["negative_mode"] = "random_ciphertext"

    gate = adjudicate_runtime_spn_rectangle_attribution(
        run_id="invalid",
        rows=rows,
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation1_runtime_spn_rectangle_attribution_protocol_invalid"
    )
    assert not gate["protocol_checks"]["exact_rectangle_descriptor"]
    assert not gate["protocol_checks"]["strict_encrypted_random_plaintext_negatives"]


def test_rectangle_rct2_medium_passes_only_with_remote_cache_and_history() -> None:
    gate = adjudicate_runtime_spn_rectangle_medium(
        run_id="rct2-pass",
        rows=_medium_rows({"true": 0.71, "corrupted": 0.62, "independent": 0.61}),
        expected_seed=0,
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_rct2_rectangle_medium_seed_passed"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())
    assert "262144/class RCT3" in gate["next_action"]


def test_rectangle_rct3_scale_passes_only_at_exact_scale() -> None:
    rows = _scale_rows(
        {"true": 0.71, "corrupted": 0.62, "independent": 0.61},
        samples_per_class=SCALE_SAMPLES_PER_CLASS,
        train_rows=SCALE_TRAIN_ROWS,
        validation_rows=SCALE_VALIDATION_ROWS,
        cache_name="rct3",
    )

    gate = adjudicate_runtime_spn_rectangle_scale(
        run_id="rct3-pass",
        rows=rows,
        expected_seed=0,
        phase="rct3",
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_rct3_rectangle_scale_seed_passed"
    assert gate["samples_per_class"] == SCALE_SAMPLES_PER_CLASS
    assert gate["validation_rows"] == SCALE_VALIDATION_ROWS
    assert all(gate["protocol_checks"].values())
    assert "1000000/class" in gate["next_action"]


def test_rectangle_scale_gate_rejects_unknown_phase() -> None:
    try:
        adjudicate_runtime_spn_rectangle_scale(
            run_id="invalid-phase",
            rows=[],
            expected_seed=0,
            phase="unknown",
        )
    except ValueError as error:
        assert str(error) == "RECTANGLE scale phase must be rct2 or rct3"
    else:
        raise AssertionError("unknown phase must fail closed")


def test_rectangle_rct2_medium_fails_closed_on_cache_and_history_drift() -> None:
    rows = _medium_rows({"true": 0.71, "corrupted": 0.62, "independent": 0.61})
    for row in rows:
        training = row["training"]
        assert isinstance(training, dict)
        training["dataset_cache_root"] = "/tmp/not-remote"
    rows[0]["history"] = []

    gate = adjudicate_runtime_spn_rectangle_medium(
        run_id="rct2-invalid",
        rows=rows,
        expected_seed=0,
    )

    assert gate["status"] == "fail"
    assert not gate["protocol_checks"]["disk_backed_remote_cache"]
    assert not gate["protocol_checks"]["five_epoch_best_checkpoint_replay"]


def test_rectangle_rct2_cli_writes_fail_closed_evidence_without_plot(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    rows = _medium_rows({"true": 0.71, "corrupted": 0.62, "independent": 0.61})
    (run_root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    exit_code = medium_main(
        [
            "--run-id",
            "rct2-cli",
            "--run-root",
            str(run_root),
            "--seed",
            "0",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    gate = json.loads((run_root / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert (run_root / "validation.json").is_file()
    assert (run_root / "summary.json").is_file()
    assert (run_root / "history.csv").is_file()
    assert not (run_root / "curves.svg").exists()


def test_rectangle_scale_real_cli_no_plot_blocks_matplotlib_import(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    rows = _medium_rows({"true": 0.71, "corrupted": 0.62, "independent": 0.61})
    (run_root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    script = f"""
import builtins
import runpy
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ImportError('matplotlib import blocked by regression test')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = [
    'scripts/gate-runtime-spn-rectangle-medium',
    '--run-id', 'rct2-no-plot-subprocess',
    '--run-root', {str(run_root)!r},
    '--seed', '0',
    '--phase', 'rct2',
    '--no-plot',
]
runpy.run_path('scripts/gate-runtime-spn-rectangle-medium', run_name='__main__')
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (run_root / "gate.json").is_file()
    assert (run_root / "validation.json").is_file()
    assert (run_root / "history.csv").is_file()
    assert not (run_root / "curves.svg").exists()
