from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_midori64_k1ai import render_k1ai_svg
from blockcipher_nd.cli.run_uknit_family_midori64_k1ai import (
    cache_reuse_checks,
    read_tasks,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_difference_position_k1ah import (
    RUN_ID as K1AH_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_neural_attribution_k1ai import (
    CONTROL_CONDITIONS,
    CONTROL_MODELS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SEEDS,
    EXPECTED_SOURCE_DIGESTS,
    EXPECTED_SPLITS,
    INPUT_DIFFERENCE,
    K1AH_DECISION,
    adjudicate_k1ai,
    build_control_checks,
    build_k1ai_control,
    candidate_protocol_frozen,
    source_binding_checks,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_midori64_neural_attribution_"
    "k1ai_2048_seed6_seed7.csv"
)


def test_k1ai_plan_is_frozen_two_seed_four_condition_cell8_matrix() -> None:
    tasks = read_tasks(PLAN)
    mapped = task_map(tasks)

    assert len(tasks) == 8
    assert candidate_protocol_frozen(tasks)
    assert set(mapped) == {
        (seed, condition) for seed in EXPECTED_SEEDS for condition in CONTROL_CONDITIONS
    }
    assert all(task["input_difference"] == INPUT_DIFFERENCE for task in tasks)
    assert all(task["pairs_per_sample"] == 4 for task in tasks)
    assert all(task["negative_mode"] == "encrypted_random_plaintexts" for task in tasks)


def test_k1ai_models_keep_geometry_and_change_only_declared_structure() -> None:
    tasks = read_tasks(PLAN)
    checks = build_control_checks(tasks)

    assert all(checks.values())
    assert checks["all_models_parameter_count_exact"]
    assert checks["all_models_state_dict_geometry_identical"]
    assert checks["wrong_sbox_changes_only_sbox"]
    assert checks["corrupted_linear_changes_only_linear"]
    assert checks["no_structure_is_identity_without_sboxes"]
    assert checks["correct_midori_window_homogeneous"]
    assert checks["reversed_control_unavailable"]

    task = task_map(tasks)[(6, "correct_structure")]
    model = build_k1ai_control(task=task, condition="correct_structure")
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        EXPECTED_PARAMETER_COUNT
    )


def test_k1ai_rejects_a_reversed_control_name() -> None:
    task = task_map(read_tasks(PLAN))[(6, "correct_structure")]

    try:
        build_k1ai_control(task=task, condition="reversed_linear")
    except ValueError as exc:
        assert "unknown K1-AI condition" in str(exc)
    else:
        raise AssertionError("K1-AI must reject the algebraically equivalent control")


def test_k1ai_source_gate_requires_exact_k1ah_cell8_confirmation() -> None:
    checks = source_binding_checks(
        gate={
            "run_id": K1AH_RUN_ID,
            "status": "pass",
            "decision": K1AH_DECISION,
            "confirmed_cells": [0, 8],
            "selection": {"selected_cells": [0, 8]},
            "protocol_checks": {"complete": True},
        },
        validation={"run_id": K1AH_RUN_ID, "status": "pass", "errors": []},
        source_digests=EXPECTED_SOURCE_DIGESTS,
        manifest_rows=source_manifest_rows(),
    )
    assert all(checks.values())

    wrong = source_manifest_rows()
    wrong[-1]["cell"] = 0
    failed = source_binding_checks(
        gate={
            "run_id": K1AH_RUN_ID,
            "status": "pass",
            "decision": K1AH_DECISION,
            "confirmed_cells": [0, 8],
            "selection": {"selected_cells": [0, 8]},
            "protocol_checks": {"complete": True},
        },
        validation={"run_id": K1AH_RUN_ID, "status": "pass", "errors": []},
        source_digests=EXPECTED_SOURCE_DIGESTS,
        manifest_rows=wrong,
    )
    assert failed["six_cell8_confirmation_caches_exact"] is False


def test_k1ai_cache_gate_requires_every_model_split_reuse() -> None:
    events = [
        {
            "event": "cache_reuse",
            "seed": seed,
            "model": CONTROL_MODELS[condition],
            "split": split,
        }
        for seed in EXPECTED_SEEDS
        for condition in CONTROL_CONDITIONS
        for split in ("train", "validation")
    ]
    assert all(cache_reuse_checks(events).values())

    failed = cache_reuse_checks([*events[:-1], {**events[-1], "event": "cache_start"}])
    assert failed["sixteen_training_validation_cache_reuses_exact"] is False
    assert failed["no_training_or_validation_cache_regenerated"] is False


def test_k1ai_gate_requires_every_fresh_structure_margin() -> None:
    gate = adjudicate_k1ai(
        tasks=read_tasks(PLAN),
        training_rows=training_rows(),
        evaluation_rows=evaluation_rows(),
        checkpoint_manifest=checkpoint_manifest(),
        source_checks={"source": True},
        control_checks={"controls": True},
        cache_checks={"cache": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("neural_structure_attribution_supported")
    assert gate["remote_scale"] == "yes"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed_rows = evaluation_rows()
    for row in failed_rows:
        if (
            row["seed"] == 7
            and row["split"] == "cross_key_validation"
            and row["condition"] == "corrupted_linear"
        ):
            row["auc"] = 0.699
    failed_training = training_rows()
    for row in failed_training:
        if row["seed"] == 7 and row["model"] == CONTROL_MODELS["corrupted_linear"]:
            row["metrics"]["auc"] = 0.699
    held = adjudicate_k1ai(
        tasks=read_tasks(PLAN),
        training_rows=failed_training,
        evaluation_rows=failed_rows,
        checkpoint_manifest=checkpoint_manifest(),
        source_checks={"source": True},
        control_checks={"controls": True},
        cache_checks={"cache": True},
    )

    assert held["status"] == "hold"
    assert held["decision"].endswith(
        "signal_learned_structure_attribution_not_supported"
    )
    assert held["remote_scale"] == "no"
    assert (
        held["research_checks"]["seed7_cross_key_validation_beats_corrupted_linear"]
        is False
    )


def test_k1ai_plot_uses_chinese_explanation_and_separated_heatmaps(
    tmp_path: Path,
) -> None:
    gate = adjudicate_k1ai(
        tasks=read_tasks(PLAN),
        training_rows=training_rows(),
        evaluation_rows=evaluation_rows(),
        checkpoint_manifest=checkpoint_manifest(),
        source_checks={"source": True},
        control_checks={"controls": True},
        cache_checks={"cache": True},
    )
    output = tmp_path / "curves.svg"
    report = render_k1ai_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert report["heatmaps_used_instead_of_overlapping_curves"] is True
    assert "Midori64 第4轮" in svg
    assert "正确 S盒 + 正确扩散" in svg
    assert "正确 S盒 + 错误扩散" in svg
    assert "结构归因净优势" in svg


def source_manifest_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": K1AH_RUN_ID,
            "phase": "confirmation",
            "cell": 8,
            "seed": seed,
            "split": split,
            "input_difference": INPUT_DIFFERENCE,
            "rounds": 4,
            "rows": 4096 if split == "train_seen" else 2048,
            "cache_payloads_present": True,
        }
        for seed in EXPECTED_SEEDS
        for split in EXPECTED_SPLITS
    ]


def training_rows() -> list[dict[str, object]]:
    rows = []
    for task in read_tasks(PLAN):
        condition = next(
            condition
            for condition, model in CONTROL_MODELS.items()
            if model == task["model_key"]
        )
        rows.append(
            {
                "cipher_key": "midori64",
                "model": task["model_key"],
                "rounds": 4,
                "seed": task["seed"],
                "input_difference": INPUT_DIFFERENCE,
                "difference_profile": "midori64_k1ah_cell8_r4",
                "samples_per_class": 2048,
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                "metrics": {"auc": condition_aucs()[condition]},
                "training": {
                    "batch_size": 64,
                    "epochs": 10,
                    "epochs_ran": 10,
                    "checkpoint_metric": "val_auc",
                    "selected_checkpoint": "best",
                    "samples_total": 4096,
                },
                "validation": {"samples_total": 2048},
            }
        )
    return rows


def evaluation_rows() -> list[dict[str, object]]:
    rows = []
    aucs = condition_aucs()
    for seed in EXPECTED_SEEDS:
        for split in EXPECTED_SPLITS:
            for condition in CONTROL_CONDITIONS:
                rows.append(
                    {
                        "run_id": "k1ai",
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "rows": 4096 if split == "train_seen" else 2048,
                        "auc": aucs[condition],
                        "dataset_sha256": f"dataset-{seed}-{split}",
                        "composition_sha256": f"composition-{condition}",
                        "residual_gate": 0.05,
                        "histogram_gate": 0.05,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return rows


def checkpoint_manifest() -> dict[str, object]:
    return {
        "run_id": "k1ai",
        "entries": [
            {
                "seed": seed,
                "condition": condition,
                "path": f"checkpoint-{seed}-{condition}.pt",
            }
            for seed in EXPECTED_SEEDS
            for condition in CONTROL_CONDITIONS
        ],
    }


def condition_aucs() -> dict[str, float]:
    return {
        "correct_structure": 0.700,
        "wrong_sbox": 0.680,
        "corrupted_linear": 0.670,
        "no_structure": 0.650,
    }
