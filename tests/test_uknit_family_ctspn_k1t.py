from __future__ import annotations

from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1t import render_k1t_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1t import parse_args
from blockcipher_nd.cli.run_uknit_family_ctspn_k1r import read_tasks
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1t import (
    ANCHOR_CONDITION,
    CONTROL_MODELS,
    EVALUATION_CONDITIONS,
    EXPECTED_PARAMETER_COUNT,
    RUN_ID,
    adjudicate_k1t,
    build_k1t_readiness,
    candidate_protocol_frozen,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_deterministic_position_residual_k1t_2048_seed3_seed4.csv"
)


def test_k1t_plan_is_frozen_two_seed_three_condition_matrix() -> None:
    tasks = read_tasks(PLAN)
    mapped = task_map(tasks)

    assert len(tasks) == 6
    assert candidate_protocol_frozen(tasks)
    assert set(mapped) == {
        (seed, condition)
        for seed in (3, 4)
        for condition in CONTROL_MODELS
    }
    assert all(task["input_difference"] == 0x0000400000000000 for task in tasks)
    assert all(task["negative_mode"] == "encrypted_random_plaintexts" for task in tasks)


def test_k1t_runner_accepts_zero_step_readiness_only() -> None:
    args = parse_args(
        [
            "--plan",
            "plan.csv",
            "--k1q-root",
            "k1q",
            "--k1r-root",
            "k1r",
            "--k1r-plan",
            "k1r.csv",
            "--k1s-root",
            "k1s",
            "--output-root",
            "output",
            "--readiness-only",
        ]
    )

    assert args.readiness_only is True


def test_k1t_readiness_proves_geometry_semantics_and_gradient() -> None:
    readiness = build_k1t_readiness(
        tasks=read_tasks(PLAN),
        datasets=synthetic_datasets(),
        source_checks={"source": True},
    )

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())
    assert set(readiness["evidence_metrics"]["parameter_counts"].values()) == {
        EXPECTED_PARAMETER_COUNT
    }


def test_k1t_gate_requires_every_fresh_attribution_margin() -> None:
    gate = adjudicate_k1t(
        tasks=read_tasks(PLAN),
        training_rows=synthetic_training_rows(),
        evaluation_rows=synthetic_evaluation_rows(),
        checkpoint_manifest=synthetic_checkpoint_manifest(),
        readiness=synthetic_readiness(),
        source_checks={"source": True},
        cache_checks={"cache": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("deterministic_position_residual_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = synthetic_evaluation_rows()
    for row in failed:
        if (
            row["seed"] == 4
            and row["split"] == "cross_key_validation"
            and row["condition"] == "exact_position_histogram_residual"
        ):
            row["auc"] = 0.55
    held = adjudicate_k1t(
        tasks=read_tasks(PLAN),
        training_rows=synthetic_training_rows(),
        evaluation_rows=failed,
        checkpoint_manifest=synthetic_checkpoint_manifest(),
        readiness=synthetic_readiness(),
        source_checks={"source": True},
        cache_checks={"cache": True},
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("key_specific_position_residual")


def test_k1t_plot_explains_candidate_and_controls_in_chinese(tmp_path: Path) -> None:
    gate = adjudicate_k1t(
        tasks=read_tasks(PLAN),
        training_rows=synthetic_training_rows(),
        evaluation_rows=synthetic_evaluation_rows(),
        checkpoint_manifest=synthetic_checkpoint_manifest(),
        readiness=synthetic_readiness(),
        source_checks={"source": True},
        cache_checks={"cache": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1t_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "保留 uKNIT 原生位置统计后" in svg
    assert "错误 S盒 + 保留位置" in svg
    assert "正确结构 + 抹除位置" in svg
    assert "对比旧 K1-R" in svg
    assert "正值表示候选更好" in svg


def synthetic_datasets() -> dict[tuple[int, str], DifferentialDataset]:
    datasets = {}
    for seed in (3, 4):
        for split_index, split in enumerate(
            ("train_seen", "same_key_fresh", "cross_key_validation")
        ):
            rng = np.random.default_rng(20260728 + seed * 10 + split_index)
            datasets[(seed, split)] = DifferentialDataset(
                features=rng.integers(0, 2, size=(16, 512), dtype=np.uint8),
                labels=np.arange(16, dtype=np.uint8) % 2,
                metadata={},
            )
    return datasets


def synthetic_training_rows() -> list[dict[str, object]]:
    rows = []
    for task in read_tasks(PLAN):
        rows.append(
            {
                "model": task["model_key"],
                "rounds": 5,
                "seed": task["seed"],
                "input_difference": 0x0000400000000000,
                "difference_profile": "uknit64_k1q_cell11_r5",
                "samples_per_class": 2048,
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                "training": {
                    "batch_size": 64,
                    "epochs": 10,
                    "epochs_ran": 10,
                    "checkpoint_metric": "val_auc",
                    "selected_checkpoint": "best",
                },
            }
        )
    return rows


def synthetic_checkpoint_manifest() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "status": "pass",
        "entries": [
            {
                "seed": task["seed"],
                "condition": next(
                    condition
                    for condition, model in CONTROL_MODELS.items()
                    if model == task["model_key"]
                ),
            }
            for task in read_tasks(PLAN)
        ],
    }


def synthetic_evaluation_rows() -> list[dict[str, object]]:
    aucs = {
        "exact_position_histogram_residual": 0.75,
        "wrong_sbox_position_histogram_residual": 0.70,
        "invariant_histogram_residual": 0.68,
        ANCHOR_CONDITION: 0.51,
    }
    rows = []
    for seed in (3, 4):
        for split in ("train_seen", "same_key_fresh", "cross_key_validation"):
            for condition in EVALUATION_CONDITIONS:
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "rows": 4096 if split == "train_seen" else 2048,
                        "auc": aucs[condition],
                        "source_auc": (
                            aucs[condition] if condition == ANCHOR_CONDITION else None
                        ),
                        "dataset_sha256": f"dataset-{seed}-{split}",
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "strict_state_dict_load": True,
                    }
                )
    return rows


def synthetic_readiness() -> dict[str, object]:
    return {
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"ready": True},
        "evidence_checks": {"ready": True},
    }
