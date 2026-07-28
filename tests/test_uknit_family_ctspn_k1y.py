from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1y import render_k1y_svg
from blockcipher_nd.cli.run_uknit_family_ctspn_k1y import prepare_bound_cache_link
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1y import (
    BASE_LR,
    CONTROL_MODELS,
    EXPECTED_KEYS,
    EXPECTED_PARAMETER_COUNT,
    PROJECTION_LR,
    PROJECTION_LR_MULTIPLIER,
    PROJECTION_PARAMETER,
    adjudicate,
    build_k1y_control,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    task_map,
)
from blockcipher_nd.training.optim import make_optimizer
from blockcipher_nd.training.types import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_compact_projection_update_k1y_2048_seed3_seed4.csv"
)


def test_k1y_plan_freezes_four_single_variable_rows() -> None:
    tasks = read_tasks(PLAN)

    assert len(tasks) == 4
    assert set(task_map(tasks)) == EXPECTED_KEYS
    assert candidate_protocol_frozen(tasks)


def test_k1y_optimizer_accelerates_only_projection_weight() -> None:
    task = task_map(read_tasks(PLAN))[(3, "projection16x_exact")]
    model = build_k1y_control(task=task, condition="projection16x_exact")
    optimizer = make_optimizer(
        model,
        TrainingConfig(
            learning_rate=BASE_LR,
            optimizer="adam",
            weight_decay=1e-5,
            lr_scheduler="none",
        ),
    )
    groups = {str(group["parameter_group_name"]): group for group in optimizer.param_groups}

    assert set(groups) == {"default", PROJECTION_PARAMETER}
    assert groups["default"]["lr"] == BASE_LR
    assert groups[PROJECTION_PARAMETER]["lr"] == PROJECTION_LR
    assert groups[PROJECTION_PARAMETER]["lr_multiplier"] == PROJECTION_LR_MULTIPLIER
    assert len(groups[PROJECTION_PARAMETER]["params"]) == 1
    assert groups[PROJECTION_PARAMETER]["params"][0] is dict(
        model.named_parameters()
    )[PROJECTION_PARAMETER]
    assert model_metadata(model)["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT


def test_k1y_readiness_proves_sources_geometry_and_groups() -> None:
    readiness = build_readiness(read_tasks(PLAN))

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert set(readiness["k1w_k1y_forward_max_errors"]) == {"3", "4"}


def test_k1y_gate_requires_each_seed_to_retain_and_attribute(tmp_path: Path) -> None:
    rows = synthetic_results(tmp_path)
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_rows=synthetic_cache_reuses(),
        readiness={
            "status": "pass",
            "optimizer_step_authorized": True,
            "protocol_checks": {"ready": True},
        },
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("projection_update_supported")
    failed = deepcopy(rows)
    failed[0]["metrics"]["auc"] = 0.51
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=failed,
        progress_rows=synthetic_cache_reuses(),
        readiness={
            "status": "pass",
            "optimizer_step_authorized": True,
            "protocol_checks": {"ready": True},
        },
    )
    assert held["status"] == "hold"


def test_k1y_cache_link_reuses_uknit_source(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    prepare_bound_cache_link(root)

    assert (root / "uknit64").is_symlink()
    assert (root / "uknit64").resolve().is_dir()


def test_k1y_plot_explains_near_miss_and_single_variable(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_results(tmp_path),
        progress_rows=synthetic_cache_reuses(),
        readiness={
            "status": "pass",
            "optimizer_step_authorized": True,
            "protocol_checks": {"ready": True},
        },
    )
    output = tmp_path / "curves.svg"

    report = render_k1y_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "仅5120个投影权重使用16倍学习率" in svg
    assert "跨密钥验证 AUC" in svg
    assert "净提升与独立门控" in svg


def synthetic_results(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (3, 4):
        for condition, model in CONTROL_MODELS.items():
            checkpoint = tmp_path / f"{seed}_{condition}.pt"
            checkpoint.write_bytes(b"checkpoint")
            exact = condition == "projection16x_exact"
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 4,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                    "histogram_projection_lr_multiplier": PROJECTION_LR_MULTIPLIER,
                    "histogram_projection_lr_parameter": PROJECTION_PARAMETER,
                    "metrics": {"auc": 0.62 if exact else 0.50},
                    "training": {
                        "train_rows": 4096,
                        "validation_rows": 2048,
                        "epochs": 10,
                        "epochs_ran": 10,
                        "learning_rate": BASE_LR,
                        "selected_checkpoint": "best",
                        "checkpoint_output": str(checkpoint),
                    },
                }
            )
    return rows


def synthetic_cache_reuses() -> list[dict[str, object]]:
    return [{"event": "cache_reuse", "index": index} for index in range(8)]
