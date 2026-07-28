from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1ac import render_k1ac_svg
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1aa import (
    EXPECTED_PARAMETER_COUNT,
    VIRTUAL_PARAMETER,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ac import (
    CONTROL_MODELS,
    EXPECTED_KEYS,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    read_tasks,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_dialga_retention_"
    "k1ac_16pair_2048_seed0_seed1.csv"
)


def test_k1ac_plan_freezes_dialga_sixteen_pair_panel() -> None:
    tasks = read_tasks(PLAN)
    assert len(tasks) == 4
    assert set(task_map(tasks)) == EXPECTED_KEYS
    assert candidate_protocol_frozen(tasks)
    assert {task["cipher_key"] for task in tasks} == {"dialga128"}
    assert {task["rounds"] for task in tasks} == {4}
    assert {task["pairs_per_sample"] for task in tasks} == {16}


def test_k1ac_readiness_proves_source_model_and_control_bindings() -> None:
    readiness = build_readiness(read_tasks(PLAN))
    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())


def test_k1ac_gate_separates_retention_from_semantics(tmp_path: Path) -> None:
    rows = synthetic_results(tmp_path)
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=rows,
        progress_rows=synthetic_cache_progress(),
        readiness=synthetic_readiness(),
    )
    assert gate["status"] == "pass"
    assert gate["decision"].endswith("retention_and_semantics_supported")
    assert all(gate["research_checks"].values())

    semantic_failure = deepcopy(rows)
    semantic_failure[1]["metrics"]["auc"] = 0.965
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=semantic_failure,
        progress_rows=synthetic_cache_progress(),
        readiness=synthetic_readiness(),
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("semantic_attribution_failed")
    assert "same-checkpoint" in held["next_action"]

    retention_failure = deepcopy(rows)
    retention_failure[0]["metrics"]["auc"] = 0.90
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=retention_failure,
        progress_rows=synthetic_cache_progress(),
        readiness=synthetic_readiness(),
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("signal_retention_failed")


def test_k1ac_plot_explains_retention_and_semantic_gates(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_results(tmp_path),
        progress_rows=synthetic_cache_progress(),
        readiness=synthetic_readiness(),
    )
    output = tmp_path / "curves.svg"
    report = render_k1ac_svg(gate, output)
    svg = output.read_text(encoding="utf-8")
    assert report["panels"] == 2
    assert "Dialga 4轮" in svg
    assert "强信号是否保留" in svg
    assert "正确 S盒是否必要" in svg


def synthetic_readiness() -> dict[str, object]:
    return {
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"ready": True},
    }


def synthetic_results(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for condition, model in CONTROL_MODELS.items():
            checkpoint = tmp_path / f"{seed}_{condition}.pt"
            checkpoint.write_bytes(b"checkpoint")
            exact = condition == "virtual_slot_exact"
            rows.append(
                {
                    "cipher_key": "dialga128",
                    "model": model,
                    "seed": seed,
                    "input_difference": 0x40,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 16,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                    "virtual_projection_slots": 16,
                    "virtual_projection_parameter": VIRTUAL_PARAMETER,
                    "metrics": {"auc": 0.97 if exact else 0.94},
                    "training": {
                        "input_bits": 4096,
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


def synthetic_cache_progress() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        for split in ("train", "validation"):
            rows.append(
                {
                    "event": "cache_start",
                    "seed": seed,
                    "split": split,
                    "pairs_per_sample": 16,
                    "input_bits": 4096,
                }
            )
            rows.append(
                {
                    "event": "cache_done",
                    "seed": seed,
                    "split": split,
                    "pairs_per_sample": 16,
                    "input_bits": 4096,
                }
            )
            rows.append(
                {
                    "event": "cache_reuse",
                    "seed": seed,
                    "split": split,
                    "pairs_per_sample": 16,
                    "input_bits": 4096,
                }
            )
    return rows
