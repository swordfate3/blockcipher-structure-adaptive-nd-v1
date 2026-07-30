from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
)
from blockcipher_nd.tasks.innovation1.uknit_r6_last2_neural_scale_k1br import (
    EXPECTED_DESCRIPTOR_SHA256S,
    adjudicate_k1br,
    candidate_protocol_frozen,
)
from blockcipher_nd.tasks.innovation1.uknit_r6_last2_neural_scale_k1br_launch import (
    build_k1br_launch_gate,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_r6_last2_neural_scale_k1br_262144_seed3.csv"
)
CONFIG = (
    ROOT
    / "configs/remote/innovation1_uknit_r6_last2_neural_scale_k1br_262144_seed3_gpu1_20260730.json"
)
RUN = (
    ROOT
    / "configs/remote/generated/run_i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730.cmd"
)
LAUNCH = (
    ROOT
    / "configs/remote/generated/launch_i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730.cmd"
)
MONITOR = (
    ROOT
    / "configs/remote/generated/monitor_i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730.sh"
)


def read_tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def test_k1br_plan_is_exact_three_row_r6_scale_exception() -> None:
    tasks = read_tasks()
    assert len(tasks) == 3
    assert candidate_protocol_frozen(tasks)
    assert {task["rounds"] for task in tasks} == {6}
    assert {task["seed"] for task in tasks} == {3}
    assert {task["samples_per_class"] for task in tasks} == {262144}
    assert {task["validation_samples_total"] for task in tasks} == {131072}
    assert {task["model_options"]["runtime_round_start"] for task in tasks} == {4}
    assert {task["model_options"]["runtime_rounds"] for task in tasks} == {2}


def test_k1br_gate_separates_weak_attributed_and_unattributed() -> None:
    weak = adjudicate_k1br(
        tasks=read_tasks(),
        result_rows=synthetic_rows(exact=0.518, wrong=0.507, invariant=0.509),
        progress_events=synthetic_progress(),
        source_checks={"source": True},
    )
    assert weak["status"] == "pass"
    assert weak["tier"] == "weak_attributed"
    assert weak["attribution_margin"] > 0.005

    unattributed = adjudicate_k1br(
        tasks=read_tasks(),
        result_rows=synthetic_rows(exact=0.518, wrong=0.516, invariant=0.509),
        progress_events=synthetic_progress(),
        source_checks={"source": True},
    )
    assert unattributed["status"] == "hold"
    assert unattributed["tier"] == "weak_unattributed"


def test_k1br_gate_selects_stronger_invariant_candidate() -> None:
    gate = adjudicate_k1br(
        tasks=read_tasks(),
        result_rows=synthetic_rows(exact=0.54, wrong=0.50, invariant=0.57),
        progress_events=synthetic_progress(),
        source_checks={"source": True},
    )
    assert gate["status"] == "pass"
    assert gate["tier"] == "strong_attributed"
    assert gate["best_candidate_condition"] == "invariant_histogram_residual"


def test_k1br_remote_assets_freeze_cache_gpu_and_windows_safety() -> None:
    run = RUN.read_text(encoding="utf-8")
    launch = LAUNCH.read_text(encoding="utf-8")
    monitor = MONITOR.read_text(encoding="utf-8")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "!" not in run + launch
    assert "cmd.exe /k" not in run + launch + config["launch_policy"]
    assert "cmd.exe /c" in launch
    assert "G:\\lxy\\blockcipher-structure-adaptive-nd-runs" in run + launch
    assert "--dataset-cache-chunk-size 1024" in run
    assert "--dataset-cache-workers 1" in run
    assert "--expected-rows 3" in run
    assert 'if not "%PHYSICAL_GPU%"=="1"' in run + launch
    assert config["physical_gpu"] == 1
    assert config["user_requested_data_scarcity_exception"] is True
    assert "sed 's/\\r$//' SHA256SUMS | sha256sum -c -" in monitor


def test_k1br_launch_gate_requires_published_commit_and_source_exception(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "k1bp"
    source_root.mkdir()
    source_gate = {
        "run_id": "i1_uknit_r6_last_round_key_hypothesis_k1bp_seed2_seed3_seed4_20260730",
        "status": "hold",
        "decision": "innovation1_uknit_r6_k1bp_single_cell_sparse_anchor_not_supported",
        "failed_protocol_checks": [],
    }
    gate_path = source_root / "gate.json"
    gate_path.write_text(json.dumps(source_gate), encoding="utf-8")
    result = build_k1br_launch_gate(
        k1bp_root=source_root,
        repository=ROOT,
        source_commit="a" * 40,
        remote_main_sha="b" * 40,
        readiness_status="pass",
    )
    assert result["launch_authorized"] is False
    assert result["ssh_allowed"] is False


def synthetic_rows(
    *, exact: float, wrong: float, invariant: float
) -> list[dict[str, object]]:
    aucs = {
        "exact_position_histogram_residual": exact,
        "wrong_sbox_position_histogram_residual": wrong,
        "invariant_histogram_residual": invariant,
    }
    rows = []
    for task in read_tasks():
        condition = next(
            name for name, model in CONTROL_MODELS.items() if model == task["model_key"]
        )
        auc = aucs[condition]
        rows.append(
            {
                "cipher_key": "uknit64",
                "rounds": 6,
                "seed": 3,
                "model": task["model_key"],
                "samples_per_class": 262144,
                "input_difference": 0x0000400000000000,
                "difference_profile": "uknit64_k1q_cell11_r5",
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                "runtime_structure_descriptor_sha256": next(
                    iter(EXPECTED_DESCRIPTOR_SHA256S)
                ),
                "runtime_structure_round_start": 4,
                "runtime_structure_loaded_rounds": 2,
                "metrics": {"auc": auc, "accuracy": 0.5, "loss": 0.25},
                "history": [{"epoch": epoch} for epoch in range(1, 11)],
                "training": {
                    "train_rows": 524288,
                    "validation_rows": 131072,
                    "train_positive_rows": 262144,
                    "train_negative_rows": 262144,
                    "validation_positive_rows": 65536,
                    "validation_negative_rows": 65536,
                    "train_dataset_storage": "disk",
                    "validation_dataset_storage": "disk",
                    "dataset_cache_root": r"G:\lxy\runs\k1br\cache",
                    "checkpoint_output": r"G:\lxy\runs\k1br\checkpoints\row.pt",
                    "dataset_cache_chunk_size": 1024,
                    "dataset_cache_workers": 1,
                    "device": "cuda",
                    "batch_size": 64,
                    "epochs": 10,
                    "epochs_ran": 10,
                    "checkpoint_metric": "val_auc",
                    "restore_best_checkpoint": True,
                    "selected_checkpoint": "best",
                    "best_checkpoint_metric": auc,
                },
                "validation": {"samples_total": 131072, "samples_per_class": 65536},
            }
        )
    return rows


def synthetic_progress() -> list[dict[str, object]]:
    paths = {
        "train": r"G:\lxy\runs\k1br\cache\train",
        "validation": r"G:\lxy\runs\k1br\cache\validation",
    }
    events = []
    for split, path in paths.items():
        base = {
            "seed": 3,
            "split": split,
            "cache_path": path,
            "chunk_size": 1024,
            "workers": 1,
        }
        for event in (
            "cache_start",
            "cache_flush_start",
            "cache_positive_chunk",
            "cache_negative_chunk",
            "cache_done",
        ):
            events.append({"event": event, **base})
    for model in (
        CONTROL_MODELS["wrong_sbox_position_histogram_residual"],
        CONTROL_MODELS["invariant_histogram_residual"],
    ):
        for split, path in paths.items():
            events.append(
                {
                    "event": "cache_reuse",
                    "seed": 3,
                    "model": model,
                    "split": split,
                    "cache_path": path,
                    "chunk_size": 1024,
                    "workers": 1,
                }
            )
    events.append({"event": "run_done"})
    return deepcopy(events)
