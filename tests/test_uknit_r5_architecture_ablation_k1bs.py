from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_r5_architecture_ablation_k1bs import (
    render_k1bs_svg,
)
from blockcipher_nd.cli.run_uknit_r5_architecture_ablation_k1bs import (
    write_comparison_csv,
)
from blockcipher_nd.tasks.innovation1.uknit_r5_architecture_ablation_k1bs import (
    ARCHITECTURES,
    EXPECTED_PARAMETER_COUNTS,
    RUN_ID,
    adjudicate,
    build_readiness,
    candidate_protocol_frozen,
    comparison_rows,
    read_tasks,
    task_map,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/"
    "innovation1_uknit_r5_neural_architecture_ablation_k1bs_"
    "16pair_2048_seed3_seed4.csv"
)


def test_k1bs_plan_changes_only_neural_architecture() -> None:
    tasks = read_tasks(PLAN)
    mapped = task_map(tasks)

    assert len(tasks) == 8
    assert candidate_protocol_frozen(tasks)
    assert set(mapped) == {
        (seed, architecture)
        for seed in (3, 4)
        for architecture in ARCHITECTURES
    }
    assert {task["pairs_per_sample"] for task in tasks} == {16}
    assert {task["samples_per_class"] for task in tasks} == {2048}
    assert {task["validation_samples_total"] for task in tasks} == {2048}
    assert {task["rounds"] for task in tasks} == {5}


def test_k1bs_readiness_proves_all_models_accept_same_16pair_input() -> None:
    readiness = build_readiness(tasks=read_tasks(PLAN))

    assert readiness["status"] == "pass"
    assert readiness["optimizer_step_authorized"] is True
    assert all(readiness["protocol_checks"].values())
    assert all(readiness["evidence_checks"].values())
    assert readiness["evidence_metrics"]["fixture_shape"] == [4, 2048]
    assert readiness["evidence_metrics"]["pair_bits"] == 128
    assert readiness["evidence_metrics"]["pairs_per_sample"] == 16
    assert readiness["evidence_metrics"]["parameter_counts"] == (
        EXPECTED_PARAMETER_COUNTS
    )
    assert set(
        tuple(shape)
        for shape in readiness["evidence_metrics"]["output_shapes"].values()
    ) == {(4, 1)}


def test_k1bs_gate_requires_expert_signal_and_margin_on_each_seed(
    tmp_path: Path,
) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_result_rows(tmp_path),
        progress_rows=synthetic_progress_rows(),
        readiness=synthetic_readiness(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("structure_expert_retained")
    assert gate["remote_scale"] == "candidate"
    assert all(gate["research_checks"].values())

    matched = deepcopy(synthetic_result_rows(tmp_path))
    for row in matched:
        if row["seed"] == 4 and row["model"] == ARCHITECTURES[
            "generic_spn_token_mixer"
        ]:
            row["metrics"]["auc"] = 0.816
    held = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=matched,
        progress_rows=synthetic_progress_rows(),
        readiness=synthetic_readiness(),
    )
    assert held["status"] == "hold"
    assert held["decision"].endswith("structure_expert_not_necessary")
    assert held["research_checks"]["seed4_expert_margin"] is False


def test_k1bs_plot_explains_round_models_and_capacity_in_chinese(
    tmp_path: Path,
) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_result_rows(tmp_path),
        progress_rows=synthetic_progress_rows(),
        readiness=synthetic_readiness(),
    )
    output = tmp_path / "curves.svg"

    report = render_k1bs_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 2
    assert "uKNIT 第5轮神经网络横向对比" in svg
    assert "只替换神经网络" in svg
    assert "uKNIT 结构专家" in svg
    assert "AutoND DBitNet" in svg
    assert "通用 SPN Token Mixer" in svg
    assert "本次并非参数量匹配实验" in svg


def test_k1bs_comparison_csv_preserves_model_metrics(tmp_path: Path) -> None:
    gate = adjudicate(
        tasks=read_tasks(PLAN),
        result_rows=synthetic_result_rows(tmp_path),
        progress_rows=synthetic_progress_rows(),
        readiness=synthetic_readiness(),
    )
    output = tmp_path / "architecture_comparison.csv"

    write_comparison_csv(output, comparison_rows(gate))

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["seed"] == "3"
    assert rows[0]["uknit_structure_expert_auc"] == "0.8"
    assert rows[0]["best_generic_architecture"] == "generic_spn_token_mixer"
    assert rows[0]["expert_minus_best_generic"] == "0.050000000000000044"


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
            "uknit_structure_expert": 0.80,
            "autond_dbitnet": 0.62,
            "generic_spn_cell_pairset": 0.70,
            "generic_spn_token_mixer": 0.75,
        },
        4: {
            "uknit_structure_expert": 0.82,
            "autond_dbitnet": 0.61,
            "generic_spn_cell_pairset": 0.72,
            "generic_spn_token_mixer": 0.78,
        },
    }
    rows = []
    for seed in (3, 4):
        for architecture, model in ARCHITECTURES.items():
            checkpoint = tmp_path / f"seed{seed}_{architecture}.pt"
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
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNTS[
                        architecture
                    ],
                    "metrics": {"auc": aucs[seed][architecture]},
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
            for _ in range(3):
                rows.append(
                    {"event": "cache_reuse", "seed": seed, "split": split}
                )
    return rows
