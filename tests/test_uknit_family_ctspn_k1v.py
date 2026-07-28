from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1v import render_k1v_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1v import write_comparison_csv
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1v import (
    RUN_ID,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_pair_count_k1v_16pair_2048_seed3_seed4.csv"
)


def test_k1v_plan_changes_only_to_sixteen_pairs() -> None:
    tasks = read_tasks(PLAN)
    mapped = task_map(tasks)

    assert len(tasks) == 6
    assert candidate_protocol_frozen(tasks)
    assert set(mapped) == {
        (seed, condition)
        for seed in (3, 4)
        for condition in CONTROL_MODELS
    }
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["samples_per_class"] for task in tasks} == {2048}
    assert {task["validation_samples_total"] for task in tasks} == {2048}


def test_k1v_readiness_proves_2048_bit_sixteen_pair_geometry() -> None:
    readiness = build_readiness(
        tasks=read_tasks(PLAN),
        anchor_gate=synthetic_anchor_gate(),
        anchor_gate_sha256=(
            "f122f43f4d895a1b68fb696bd81df4e1d362880a3a12d9883933c932dd7f0dbf"
        ),
    )

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())
    assert readiness["evidence_metrics"]["fixture_shape"] == [8, 2048]
    assert set(readiness["evidence_metrics"]["pairs_per_sample"].values()) == {16}
    assert set(readiness["evidence_metrics"]["parameter_counts"].values()) == {
        EXPECTED_PARAMETER_COUNT
    }


def test_k1v_gate_requires_each_seed_to_keep_semantics_and_add_value(
    tmp_path: Path,
) -> None:
    rows = synthetic_result_rows(tmp_path)
    readiness = synthetic_readiness()
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_rows=synthetic_progress_rows(),
        readiness=readiness,
        anchor_gate=synthetic_anchor_gate(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("16pair_added_value_supported")
    assert all(gate["research_checks"].values())

    no_gain = deepcopy(rows)
    for row in no_gain:
        if row["seed"] == 4 and row["model"] == CONTROL_MODELS[
            "exact_position_histogram_residual"
        ]:
            row["metrics"]["auc"] = 0.749
        if row["seed"] == 4 and row["model"] == CONTROL_MODELS[
            "invariant_histogram_residual"
        ]:
            row["metrics"]["auc"] = 0.745
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=no_gain,
        progress_rows=synthetic_progress_rows(),
        readiness=readiness,
        anchor_gate=synthetic_anchor_gate(),
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("16pair_no_added_value")
    assert held["research_checks"]["seed4_added_value"] is False


def test_k1v_plot_explains_pair_count_and_controls_in_chinese(
    tmp_path: Path,
) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_result_rows(tmp_path),
        progress_rows=synthetic_progress_rows(),
        readiness=synthetic_readiness(),
        anchor_gate=synthetic_anchor_gate(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1v_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "从4对密文提升到16对" in svg
    assert "16对：正确结构" in svg
    assert "4对：正确结构锚点" in svg
    assert "16对 - 4对" in svg
    assert "正确结构 - 错误 S盒" in svg


def test_k1v_comparison_csv_preserves_pair_metrics(tmp_path: Path) -> None:
    output = tmp_path / "pair_comparison.csv"
    rows = [
        {
            "seed": 3,
            "exact_16pair_auc": 0.90,
            "exact_4pair_anchor_auc": 0.71,
            "exact_16pair_minus_exact_4pair": 0.19,
            "wrong_sbox_16pair_auc": 0.51,
            "exact_minus_wrong_sbox": 0.39,
            "invariant_16pair_auc": 0.59,
            "exact_minus_invariant": 0.31,
        }
    ]

    write_comparison_csv(output, rows)

    with output.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["seed"] == "3"
    assert written[0]["exact_16pair_auc"] == "0.9"
    assert written[0]["exact_16pair_minus_exact_4pair"] == "0.19"


def synthetic_anchor_gate() -> dict[str, object]:
    return {
        "run_id": (
            "i1_uknit_family_ctspn_deterministic_position_residual_"
            "k1t_2048_seed3_seed4_20260728"
        ),
        "status": "pass",
        "protocol_checks": {"anchor": True},
        "seed_results": {
            "3": {
                "cross_key_validation": {
                    "exact_position_histogram_residual_auc": 0.713162422
                }
            },
            "4": {
                "cross_key_validation": {
                    "exact_position_histogram_residual_auc": 0.748229027
                }
            },
        },
    }


def synthetic_readiness() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"ready": True},
        "evidence_checks": {"ready": True},
    }


def synthetic_result_rows(tmp_path: Path) -> list[dict[str, object]]:
    aucs = {
        3: {
            "exact_position_histogram_residual": 0.80,
            "wrong_sbox_position_histogram_residual": 0.51,
            "invariant_histogram_residual": 0.74,
        },
        4: {
            "exact_position_histogram_residual": 0.82,
            "wrong_sbox_position_histogram_residual": 0.50,
            "invariant_histogram_residual": 0.76,
        },
    }
    rows = []
    for seed in (3, 4):
        for condition, model in CONTROL_MODELS.items():
            checkpoint = tmp_path / f"seed{seed}_{condition}.pt"
            checkpoint.write_bytes(b"checkpoint")
            rows.append(
                {
                    "model": model,
                    "rounds": 5,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 16,
                    "input_difference": 0x0000400000000000,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                    "metrics": {"auc": aucs[seed][condition]},
                    "training": {
                        "input_bits": 2048,
                        "train_rows": 4096,
                        "validation_rows": 2048,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "selected_checkpoint": "best",
                        "checkpoint_output": str(checkpoint),
                    },
                }
            )
    return rows


def synthetic_progress_rows() -> list[dict[str, object]]:
    rows = []
    for seed in (3, 4):
        for split in ("train", "validation"):
            rows.append({"event": "cache_start", "seed": seed, "split": split})
            for _ in range(2):
                rows.append(
                    {"event": "cache_reuse", "seed": seed, "split": split}
                )
    return rows
