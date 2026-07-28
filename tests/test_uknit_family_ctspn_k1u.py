from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1u import render_k1u_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1r import read_tasks
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import CONTROL_MODELS
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1u import (
    EXPECTED_PARAMETER_COUNT,
    adjudicate_k1u,
    candidate_protocol_frozen,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4.csv"
)


def test_k1u_plan_changes_only_to_frozen_medium_scale() -> None:
    tasks = read_tasks(PLAN)

    assert len(tasks) == 6
    assert candidate_protocol_frozen(tasks)
    assert {task["seed"] for task in tasks} == {3, 4}
    assert {task["samples_per_class"] for task in tasks} == {65536}
    assert {task["validation_samples_total"] for task in tasks} == {65536}


def test_k1u_gate_passes_two_seed_medium_attribution() -> None:
    gate = adjudicate_k1u(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_result_rows(),
        progress_events=synthetic_progress_events(),
        source_checks={"source": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("medium_position_residual_supported")
    assert gate["remote_scale"] == "no_mechanical_scale"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_k1u_gate_holds_one_seed_failure_and_invalidates_memory_cache() -> None:
    rows = synthetic_result_rows()
    for row in rows:
        if row["seed"] == 4 and row["model"] == CONTROL_MODELS[
            "exact_position_histogram_residual"
        ]:
            row["metrics"]["auc"] = 0.55
            row["training"]["best_checkpoint_metric"] = 0.55
    held = adjudicate_k1u(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_events=synthetic_progress_events(),
        source_checks={"source": True},
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("medium_seed_key_instability")

    invalid_rows = synthetic_result_rows()
    invalid_rows[0]["training"]["train_dataset_storage"] = "memory"
    invalid = adjudicate_k1u(
        tasks=read_tasks(PLAN),
        result_rows=invalid_rows,
        progress_events=synthetic_progress_events(),
        source_checks={"source": True},
    )
    assert invalid["status"] == "invalid"
    assert "result_protocol_frozen" in invalid["failed_protocol_checks"]


def test_k1u_gate_selects_compact_invariant_route_when_only_position_fails() -> None:
    rows = synthetic_result_rows()
    invariant_model = CONTROL_MODELS["invariant_histogram_residual"]
    for row in rows:
        if row["model"] == invariant_model:
            row["metrics"]["auc"] = 0.75
            row["training"]["best_checkpoint_metric"] = 0.75

    gate = adjudicate_k1u(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_events=synthetic_progress_events(),
        source_checks={"source": True},
    )

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("medium_signal_without_position_necessity")
    assert gate["descriptive_diagnostics"]["exact_signal_both_seeds"] is True
    assert (
        gate["descriptive_diagnostics"]["wrong_sbox_attribution_both_seeds"]
        is True
    )
    assert gate["descriptive_diagnostics"]["position_necessity_both_seeds"] is False
    assert gate["next_action"].endswith("the simpler invariant branch")


def test_k1u_plot_explains_medium_candidate_and_controls(tmp_path: Path) -> None:
    gate = adjudicate_k1u(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_result_rows(),
        progress_events=synthetic_progress_events(),
        source_checks={"source": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1u_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "uKNIT 第5轮位置残差在中型数据规模是否仍成立" in svg
    assert "65536/class 训练" in svg
    assert "错误 S盒 + 保留位置" in svg
    assert "对比位置抹除" in svg


def synthetic_result_rows() -> list[dict[str, object]]:
    aucs = {
        "exact_position_histogram_residual": 0.74,
        "wrong_sbox_position_histogram_residual": 0.51,
        "invariant_histogram_residual": 0.59,
    }
    rows = []
    for task in read_tasks(PLAN):
        condition = next(
            condition
            for condition, model in CONTROL_MODELS.items()
            if model == task["model_key"]
        )
        auc = aucs[condition]
        rows.append(
            {
                "cipher_key": "uknit64",
                "rounds": 5,
                "seed": task["seed"],
                "model": task["model_key"],
                "samples_per_class": 65536,
                "input_difference": 0x0000400000000000,
                "difference_profile": "uknit64_k1q_cell11_r5",
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "independent_pairs",
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                "runtime_structure_descriptor_sha256": (
                    "b74f9cc28b5fc28637b179f45ded67dec1a3d5dca04ca2eccb176ec790fbefd2"
                ),
                "runtime_structure_round_start": 3,
                "runtime_structure_loaded_rounds": 2,
                "metrics": {"auc": auc, "accuracy": 0.7, "loss": 0.2},
                "history": [{"epoch": epoch} for epoch in range(1, 11)],
                "training": {
                    "train_rows": 131072,
                    "validation_rows": 65536,
                    "train_positive_rows": 65536,
                    "train_negative_rows": 65536,
                    "validation_positive_rows": 32768,
                    "validation_negative_rows": 32768,
                    "train_dataset_storage": "disk",
                    "validation_dataset_storage": "disk",
                    "dataset_cache_root": r"G:\lxy\runs\k1u\cache",
                    "checkpoint_output": r"G:\lxy\runs\k1u\checkpoints\row.pt",
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
                "validation": {"samples_total": 65536, "samples_per_class": 32768},
            }
        )
    return rows


def synthetic_progress_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    exact_model = CONTROL_MODELS["exact_position_histogram_residual"]
    reused_models = (
        CONTROL_MODELS["wrong_sbox_position_histogram_residual"],
        CONTROL_MODELS["invariant_histogram_residual"],
    )
    for seed in (3, 4):
        for split in ("train", "validation"):
            path = rf"G:\lxy\runs\k1u\cache\seed{seed}\{split}"
            context = {
                "seed": seed,
                "model": exact_model,
                "split": split,
                "cache_path": path,
                "chunk_size": 1024,
                "workers": 1,
            }
            for event in (
                "cache_start",
                "cache_positive_chunk",
                "cache_negative_chunk",
                "cache_flush_start",
                "cache_done",
            ):
                events.append({**deepcopy(context), "event": event})
            for model in reused_models:
                events.append(
                    {
                        **deepcopy(context),
                        "event": "cache_reuse",
                        "model": model,
                    }
                )
    events.append({"event": "run_done"})
    return events
