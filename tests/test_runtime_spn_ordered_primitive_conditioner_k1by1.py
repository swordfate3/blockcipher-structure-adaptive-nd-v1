from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_runtime_spn_ordered_primitive_conditioner_k1by1 import (
    render_k1by1_svg,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_ordered_primitive_conditioner_k1by1 import (
    CONDITIONS,
    EXPECTED_PARAMETER_COUNT,
    PLAN_PATH,
    RUN_ID,
    adjudicate,
    build_condition,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    source_binding_checks,
    task_map,
)


def test_k1by1_plan_binds_compiler_and_changes_only_routing() -> None:
    tasks = read_tasks(PLAN_PATH)
    mapped = task_map(tasks)

    assert len(tasks) == 8
    assert candidate_protocol_frozen(tasks)
    assert all(source_binding_checks().values())
    assert set(mapped) == {
        (seed, condition) for seed in (3, 4) for condition in CONDITIONS
    }
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["samples_per_class"] for task in tasks} == {2048}
    assert {task["rounds"] for task in tasks} == {5}


def test_k1by1_models_share_geometry_but_route_distinct_programs() -> None:
    tasks = task_map(read_tasks(PLAN_PATH))
    fixture = torch.randint(0, 2, (3, 2048), dtype=torch.float32)
    models = {}
    outputs = {}
    for condition in CONDITIONS:
        torch.manual_seed(7)
        model = build_condition(task=tasks[(3, condition)], condition=condition)
        models[condition] = model
        outputs[condition] = model(fixture)

    assert {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    } == {EXPECTED_PARAMETER_COUNT}
    assert {tuple(output.shape) for output in outputs.values()} == {(3, 1)}
    assert models["correct_compiler_routing"].uses_cipher_identity is False
    assert models["correct_compiler_routing"].shared_experts_across_cells_and_stages
    assert models["correct_compiler_routing"].state_width_independent_parameter_shapes
    assert models["no_compiler_conditioner"].primitive_conditioner_enabled is False
    assert models["correct_compiler_routing"].compiled_program_semantic_sha256 != (
        models["wrong_order_routing"].compiled_program_semantic_sha256
    )
    assert models["correct_compiler_routing"].compiled_program_semantic_sha256 != (
        models["wrong_target_binding_routing"].compiled_program_semantic_sha256
    )


def test_k1by1_readiness_authorizes_only_exact_equal_geometry() -> None:
    readiness = build_readiness(tasks=read_tasks(PLAN_PATH))

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())
    assert set(readiness["evidence_metrics"]["parameter_counts"].values()) == {
        EXPECTED_PARAMETER_COUNT
    }


def test_k1by1_gate_requires_signal_and_all_margins_per_seed(
    tmp_path: Path,
) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN_PATH),
        result_rows=_synthetic_result_rows(tmp_path),
        progress_rows=_synthetic_progress_rows(),
        readiness=_synthetic_readiness(),
    )
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("compiler_conditioner_supported")
    assert all(gate["research_checks"].values())

    failed = deepcopy(_synthetic_result_rows(tmp_path))
    for row in failed:
        if row["seed"] == 4 and row["model"] == CONDITIONS["wrong_order_routing"]:
            row["metrics"]["auc"] = 0.819
    held = adjudicate(
        tasks=read_tasks(PLAN_PATH),
        result_rows=failed,
        progress_rows=_synthetic_progress_rows(),
        readiness=_synthetic_readiness(),
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("structure_attribution_not_supported")
    assert held["research_checks"]["seed4_wrong_order_margin"] is False


def test_k1by1_plot_explains_structure_route_and_scale_in_chinese(
    tmp_path: Path,
) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN_PATH),
        result_rows=_synthetic_result_rows(tmp_path),
        progress_rows=_synthetic_progress_rows(),
        readiness=_synthetic_readiness(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1by1_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "可学习密码结构能否帮助 uKNIT 第5轮区分" in svg
    assert "只改变编译后的结构路由" in svg
    assert "错误阶段顺序" in svg
    assert "错误目标绑定" in svg
    assert "不是正式规模或跨密码迁移结果" in svg


def _synthetic_readiness() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"ready": True},
        "evidence_checks": {"ready": True},
    }


def _synthetic_result_rows(tmp_path: Path) -> list[dict[str, object]]:
    aucs = {
        3: {
            "correct_compiler_routing": 0.80,
            "wrong_order_routing": 0.70,
            "wrong_target_binding_routing": 0.65,
            "no_compiler_conditioner": 0.55,
        },
        4: {
            "correct_compiler_routing": 0.82,
            "wrong_order_routing": 0.72,
            "wrong_target_binding_routing": 0.67,
            "no_compiler_conditioner": 0.56,
        },
    }
    rows = []
    for seed in (3, 4):
        for condition, model in CONDITIONS.items():
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


def _synthetic_progress_rows() -> list[dict[str, object]]:
    rows = []
    for seed in (3, 4):
        for split in ("train", "validation"):
            rows.append({"event": "cache_start", "seed": seed, "split": split})
            for _ in range(3):
                rows.append({"event": "cache_reuse", "seed": seed, "split": split})
    return rows
