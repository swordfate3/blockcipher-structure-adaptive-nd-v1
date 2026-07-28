from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1r import render_k1r_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1r import (
    cache_reuse_checks,
    read_tasks,
)
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import (
    CONTROL_MODELS,
    build_k1n_control,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1q import (
    RUN_ID as K1Q_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import (
    CONTROL_CONDITIONS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SEEDS,
    EXPECTED_SOURCE_DIGESTS,
    EXPECTED_SPLITS,
    INPUT_DIFFERENCE,
    K1Q_DECISION,
    adjudicate_k1r,
    candidate_protocol_frozen,
    source_binding_checks,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_cell11_neural_attribution_k1r_2048_seed3_seed4.csv"
)


def test_k1r_plan_is_frozen_two_seed_four_condition_cell11_matrix() -> None:
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


def test_k1r_models_keep_geometry_and_change_declared_composition() -> None:
    task = task_map(read_tasks(PLAN))[(3, "exact_composition")]
    models = {
        condition: build_k1n_control(
            task={**task, "model_key": CONTROL_MODELS[condition]},
            condition=condition,
            input_bits=512,
        )
        for condition in CONTROL_CONDITIONS
    }
    geometries = {
        condition: [
            (name, tuple(value.shape)) for name, value in model.state_dict().items()
        ]
        for condition, model in models.items()
    }

    assert all(
        model_metadata(model)["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
        for model in models.values()
    )
    assert len({tuple(geometry) for geometry in geometries.values()}) == 1
    assert len({model.composition_sha256 for model in models.values()}) == 4


def test_k1r_source_gate_requires_exact_k1q_cell11_confirmation() -> None:
    checks = source_binding_checks(
        gate={
            "run_id": K1Q_RUN_ID,
            "status": "pass",
            "decision": K1Q_DECISION,
            "confirmed_cells": [11, 0],
            "protocol_checks": {"complete": True},
        },
        validation={"run_id": K1Q_RUN_ID, "status": "pass", "errors": []},
        source_digests=EXPECTED_SOURCE_DIGESTS,
        manifest_rows=source_manifest_rows(),
    )
    assert all(checks.values())

    wrong = source_manifest_rows()
    wrong[-1]["cell"] = 0
    failed = source_binding_checks(
        gate={
            "run_id": K1Q_RUN_ID,
            "status": "pass",
            "decision": K1Q_DECISION,
            "confirmed_cells": [11, 0],
            "protocol_checks": {"complete": True},
        },
        validation={"run_id": K1Q_RUN_ID, "status": "pass", "errors": []},
        source_digests=EXPECTED_SOURCE_DIGESTS,
        manifest_rows=wrong,
    )
    assert failed["six_cell11_confirmation_caches_exact"] is False


def test_k1r_cache_gate_requires_every_model_split_reuse() -> None:
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


def test_k1r_gate_requires_every_fresh_structure_margin() -> None:
    gate = adjudicate_k1r(
        tasks=read_tasks(PLAN),
        training_rows=training_rows(),
        evaluation_rows=evaluation_rows(),
        checkpoint_manifest=checkpoint_manifest(),
        source_checks={"source": True},
        cache_checks={"cache": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("cell11_neural_structure_attribution_supported")
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed_rows = evaluation_rows()
    for row in failed_rows:
        if (
            row["seed"] == 4
            and row["split"] == "cross_key_validation"
            and row["condition"] == "wrong_sbox_semantics"
        ):
            row["auc"] = 0.699
    failed_training = training_rows()
    for row in failed_training:
        if row["seed"] == 4 and row["model"] == CONTROL_MODELS["wrong_sbox_semantics"]:
            row["metrics"]["auc"] = 0.699
    held = adjudicate_k1r(
        tasks=read_tasks(PLAN),
        training_rows=failed_training,
        evaluation_rows=failed_rows,
        checkpoint_manifest=checkpoint_manifest(),
        source_checks={"source": True},
        cache_checks={"cache": True},
    )

    assert held["status"] == "hold"
    assert held["decision"].endswith(
        "cell11_signal_learned_structure_attribution_not_supported"
    )
    assert (
        held["research_checks"]["seed4_cross_key_validation_beats_wrong_sbox"] is False
    )


def test_k1r_plot_uses_chinese_explanation_and_separated_heatmaps(
    tmp_path: Path,
) -> None:
    gate = adjudicate_k1r(
        tasks=read_tasks(PLAN),
        training_rows=training_rows(),
        evaluation_rows=evaluation_rows(),
        checkpoint_manifest=checkpoint_manifest(),
        source_checks={"source": True},
        cache_checks={"cache": True},
    )
    output = tmp_path / "curves.svg"
    report = render_k1r_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert report["heatmaps_used_instead_of_overlapping_curves"] is True
    assert "uKNIT 第5轮换成强差分后" in svg
    assert "正确 S盒 + 正确扩散" in svg
    assert "无 S盒 + 无扩散拓扑" in svg
    assert "结构归因净优势" in svg


def source_manifest_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": K1Q_RUN_ID,
            "phase": "confirmation",
            "cell": 11,
            "seed": seed,
            "split": split,
            "input_difference": INPUT_DIFFERENCE,
            "rounds": 5,
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
        cross_auc = condition_aucs()[condition]
        rows.append(
            {
                "cipher_key": "uknit64",
                "model": task["model_key"],
                "rounds": 5,
                "seed": task["seed"],
                "input_difference": INPUT_DIFFERENCE,
                "difference_profile": "uknit64_k1q_cell11_r5",
                "samples_per_class": 2048,
                "pairs_per_sample": 4,
                "negative_mode": "encrypted_random_plaintexts",
                "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                "metrics": {"auc": cross_auc},
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
                        "run_id": "k1r",
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "rows": 4096 if split == "train_seen" else 2048,
                        "auc": aucs[condition],
                        "dataset_sha256": f"dataset-{seed}-{split}",
                        "composition_sha256": f"composition-{condition}",
                        "effective_gate": 0.05,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return rows


def checkpoint_manifest() -> dict[str, object]:
    return {
        "run_id": "k1r",
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
        "exact_composition": 0.700,
        "wrong_sbox_semantics": 0.680,
        "no_sbox_composition": 0.670,
        "no_topology": 0.650,
    }
