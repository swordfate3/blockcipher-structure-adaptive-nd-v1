from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_runtime_spn_ordered_primitive_conditioner_k1by2 import (
    render_k1by2_svg,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_ordered_primitive_conditioner_k1by1 import (
    EXPECTED_PARAMETER_COUNT,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_ordered_primitive_conditioner_k1by2 import (
    CONDITIONS,
    EXPECTED_KEYS,
    PLAN_PATH,
    RUN_ID,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    source_binding_checks,
    task_map,
)


def test_k1by2_plan_changes_only_fresh_seed_and_key_pair() -> None:
    tasks = read_tasks(PLAN_PATH)
    mapped = task_map(tasks)

    assert len(tasks) == 6
    assert candidate_protocol_frozen(tasks)
    assert all(source_binding_checks().values())
    assert set(mapped) == {
        (seed, condition) for seed in (5, 6) for condition in CONDITIONS
    }
    assert {
        (task["seed"], task["train_key"], task["validation_key"]) for task in tasks
    } == {(seed, *EXPECTED_KEYS[seed]) for seed in (5, 6)}
    assert {task["samples_per_class"] for task in tasks} == {2048}
    assert {task["pairs_per_sample"] for task in tasks} == {16}


def test_k1by2_readiness_binds_source_and_equal_geometry() -> None:
    readiness = build_readiness(tasks=read_tasks(PLAN_PATH))

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())
    assert set(readiness["evidence_metrics"]["parameter_counts"].values()) == {
        EXPECTED_PARAMETER_COUNT
    }


def test_k1by2_gate_requires_each_fresh_seed_and_control(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN_PATH),
        result_rows=_synthetic_results(tmp_path),
        progress_rows=_synthetic_progress(),
        readiness=_synthetic_readiness(),
    )
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("fresh_seed_confirmed")
    assert all(gate["research_checks"].values())

    failed = deepcopy(_synthetic_results(tmp_path))
    for row in failed:
        if row["seed"] == 6 and row["model"] == CONDITIONS["wrong_order_routing"]:
            row["metrics"]["auc"] = 0.818
    held = adjudicate(
        tasks=read_tasks(PLAN_PATH),
        result_rows=failed,
        progress_rows=_synthetic_progress(),
        readiness=_synthetic_readiness(),
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("fresh_seed_attribution_not_confirmed")
    assert held["research_checks"]["seed6_wrong_order_margin"] is False


def test_k1by2_plot_explains_fresh_seed_scope_in_chinese(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN_PATH),
        result_rows=_synthetic_results(tmp_path),
        progress_rows=_synthetic_progress(),
        readiness=_synthetic_readiness(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1by2_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "uKNIT 第5轮可学习结构的新种子确认" in svg
    assert "seed5/6 和新固定密钥" in svg
    assert "只回答结果能否跨新种子和密钥复现" in svg
    assert "错误阶段顺序" in svg


def _synthetic_readiness() -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"ready": True},
        "evidence_checks": {"ready": True},
    }


def _synthetic_results(tmp_path: Path) -> list[dict[str, object]]:
    aucs = {
        5: {
            "correct_compiler_routing": 0.80,
            "wrong_order_routing": 0.60,
            "no_compiler_conditioner": 0.55,
        },
        6: {
            "correct_compiler_routing": 0.82,
            "wrong_order_routing": 0.62,
            "no_compiler_conditioner": 0.56,
        },
    }
    rows = []
    for seed in (5, 6):
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
                    "train_key": EXPECTED_KEYS[seed][0],
                    "validation_key": EXPECTED_KEYS[seed][1],
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


def _synthetic_progress() -> list[dict[str, object]]:
    rows = []
    for seed in (5, 6):
        for split in ("train", "validation"):
            rows.append({"event": "cache_start", "seed": seed, "split": split})
            for _ in range(2):
                rows.append({"event": "cache_reuse", "seed": seed, "split": split})
    return rows
