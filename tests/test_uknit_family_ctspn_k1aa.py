from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1aa import render_k1aa_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1aa import prepare_bound_cache_link
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1aa import (
    CONTROL_MODELS,
    EXPECTED_KEYS,
    EXPECTED_PARAMETER_COUNT,
    VIRTUAL_PARAMETER,
    VIRTUAL_SHAPE,
    adjudicate,
    audit_virtual_projection_gradient,
    build_k1aa_control,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_virtual_slot_projection_"
    "k1aa_2048_seed3_seed4.csv"
)


def test_k1aa_plan_freezes_four_virtual_slot_rows() -> None:
    tasks = read_tasks(PLAN)
    mapped = task_map(tasks)

    assert len(tasks) == 4
    assert set(mapped) == EXPECTED_KEYS
    assert candidate_protocol_frozen(tasks)
    assert {task["pairs_per_sample"] for task in tasks} == {4}
    assert {task["samples_per_class"] for task in tasks} == {2048}


def test_k1aa_model_has_fixed_virtual_geometry_and_no_lr_override() -> None:
    task = task_map(read_tasks(PLAN))[(3, "virtual_slot_exact")]
    model = build_k1aa_control(task=task, condition="virtual_slot_exact")
    named = dict(model.named_parameters())
    metadata = model_metadata(model)

    assert tuple(named[VIRTUAL_PARAMETER].shape) == VIRTUAL_SHAPE
    assert metadata["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert metadata["virtual_projection_slots"] == 16
    assert metadata["virtual_projection_parameter"] == VIRTUAL_PARAMETER
    assert not hasattr(model, "optimizer_parameter_lr_multipliers")
    assert model(torch.zeros(3, 512)).shape == (3, 1)


def test_k1aa_virtual_slot_gradient_is_exactly_sixteen_fold() -> None:
    task = task_map(read_tasks(PLAN))[(3, "virtual_slot_exact")]
    model = build_k1aa_control(task=task, condition="virtual_slot_exact")

    audit = audit_virtual_projection_gradient(
        model.backbone.histogram_projection[0]
    )

    assert audit["optimizer_steps"] == 0
    assert audit["slot_gradient_relative_error"] <= 1e-9
    assert audit["effective_gradient_ratio_error"] <= 1e-9
    assert abs(audit["effective_gradient_ratio"] - 16.0) <= 1e-9


def test_k1aa_readiness_proves_source_geometry_forward_and_optimizer() -> None:
    readiness = build_readiness(read_tasks(PLAN))

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert set(readiness["effective_compact_forward_max_errors"]) == {"3", "4"}
    assert set(readiness["gradient_audits"]) == {"3", "4"}


def test_k1aa_gate_requires_each_seed_to_retain_and_attribute(
    tmp_path: Path,
) -> None:
    rows = synthetic_results(tmp_path)
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("virtual_slots_supported")
    assert all(gate["research_checks"].values())

    failed = deepcopy(rows)
    failed[0]["metrics"]["auc"] = 0.54
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=failed,
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("anchor_retention_failed")


def test_k1aa_cache_link_reuses_uknit_source(tmp_path: Path) -> None:
    root = tmp_path / "cache"

    prepare_bound_cache_link(root)

    assert (root / "uknit64").is_symlink()
    assert (root / "uknit64").resolve().is_dir()


def test_k1aa_plot_names_virtual_slots_controls_and_next_action(
    tmp_path: Path,
) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_results(tmp_path),
        progress_rows=synthetic_cache_reuses(),
        readiness=synthetic_readiness(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1aa_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "固定16个虚拟投影槽" in svg
    assert "正确 S盒" in svg
    assert "错误 S盒" in svg
    assert "4对与16对密文" in svg


def synthetic_readiness() -> dict[str, object]:
    return {
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"ready": True},
    }


def synthetic_results(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (3, 4):
        for condition, model in CONTROL_MODELS.items():
            checkpoint = tmp_path / f"{seed}_{condition}.pt"
            checkpoint.write_bytes(b"checkpoint")
            exact = condition == "virtual_slot_exact"
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 4,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                    "virtual_projection_slots": 16,
                    "virtual_projection_parameter": VIRTUAL_PARAMETER,
                    "metrics": {"auc": 0.62 if exact else 0.50},
                    "training": {
                        "train_rows": 4096,
                        "validation_rows": 2048,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "learning_rate": 1e-4,
                        "selected_checkpoint": "best",
                        "checkpoint_output": str(checkpoint),
                    },
                }
            )
    return rows


def synthetic_cache_reuses() -> list[dict[str, object]]:
    return [{"event": "cache_reuse", "index": index} for index in range(8)]
