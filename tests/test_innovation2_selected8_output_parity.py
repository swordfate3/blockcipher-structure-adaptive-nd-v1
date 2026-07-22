from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_innovation2_selected8_parity import (
    render_selected8_parity,
)
from blockcipher_nd.cli.run_innovation2_selected8_parity import main
from blockcipher_nd.evaluation.result_index import display_name_for_run
from blockcipher_nd.tasks.innovation2.selected8_output_parity import (
    MODEL_SPECS,
    SELECTED8_PARITY_MASK,
    SHIFTED_CONTROL_MASK,
    Selected8ParityConfig,
    adjudicate_selected8_parity,
    parameter_counts,
    parity_targets,
)
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputResidualCnn,
)


def test_parity_targets_xor_all_eight_true_output_bits() -> None:
    targets = np.zeros((3, 64), dtype=np.float32)
    targets[0, [0, 2, 8]] = 1
    targets[1, list(SELECTED8_PARITY_MASK)] = 1
    targets[2, [32, 34, 40, 42]] = 1

    parity = parity_targets(targets, SELECTED8_PARITY_MASK)

    assert parity.shape == (3, 1)
    assert parity[:, 0].tolist() == [1.0, 0.0, 0.0]


def test_shifted_control_is_distinct_same_weight_geometry() -> None:
    assert len(SELECTED8_PARITY_MASK) == len(SHIFTED_CONTROL_MASK) == 8
    assert set(SELECTED8_PARITY_MASK).isdisjoint(SHIFTED_CONTROL_MASK)
    assert all(
        control == selected + 1
        for selected, control in zip(
            SELECTED8_PARITY_MASK,
            SHIFTED_CONTROL_MASK,
            strict=True,
        )
    )


def test_local_mlp_and_rescnn_are_parameter_matched() -> None:
    counts = parameter_counts(Selected8ParityConfig())

    assert counts["parity_mlp"] == 82_689
    assert counts["parity_rescnn"] == 82_441
    assert abs(counts["parity_mlp"] - counts["parity_rescnn"]) / counts[
        "parity_mlp"
    ] < 0.005
    model = SelectedOutputResidualCnn(channels=4, blocks=1, output_bits=1)
    assert model(torch.zeros((2, 64))).shape == (2, 1)


def test_r3_and_r4_gates_require_frozen_controls() -> None:
    r3_config = Selected8ParityConfig.diagnostic(rounds=3)
    r3_training = _synthetic_training(
        candidate_auc=0.56,
        mlp_auc=0.53,
        geometry_auc=0.52,
        shuffle_auc=0.50,
        derived_auc=0.51,
        component_auc=0.54,
    )
    r3_gate = adjudicate_selected8_parity(
        r3_config,
        {"valid": True},
        r3_training,
    )

    assert r3_gate["decision"] == "innovation2_selected8_parity_r3_calibrated"

    r4_config = Selected8ParityConfig.diagnostic(rounds=4)
    r4_gate = adjudicate_selected8_parity(
        r4_config,
        {"valid": True},
        r3_training,
        r3_gate=r3_gate,
    )
    assert r4_gate["decision"] == "innovation2_selected8_parity_r4_local_supported"

    blocked = adjudicate_selected8_parity(
        r4_config,
        {"valid": True},
        _synthetic_training(
            candidate_auc=0.505,
            mlp_auc=0.503,
            geometry_auc=0.504,
            shuffle_auc=0.501,
            derived_auc=0.504,
            component_auc=0.506,
        ),
        r3_gate=r3_gate,
    )
    assert blocked["status"] == "hold"
    assert blocked["decision"] == "innovation2_selected8_parity_r4_not_supported"


def test_end_to_end_smoke_emits_twelve_rows_and_five_checkpoints(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "ope1"

    exit_code = main(
        [
            "--mode",
            "smoke",
            "--rounds",
            "3",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    gate = json.loads((output_root / "gate.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (output_root / "metadata.json").read_text(encoding="utf-8")
    )
    results = (output_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    checkpoints = json.loads(
        (output_root / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["execution_checks"].values())
    assert metadata["sample_classification"] is False
    assert metadata["selected8_parity_mask"] == list(SELECTED8_PARITY_MASK)
    assert len(results) == 12
    assert len(checkpoints) == 5
    assert (output_root / "curves.svg").is_file()


def test_plot_explains_true_output_value_and_controls(tmp_path: Path) -> None:
    config = Selected8ParityConfig.diagnostic(rounds=3)
    training = _synthetic_training(
        candidate_auc=0.56,
        mlp_auc=0.53,
        geometry_auc=0.52,
        shuffle_auc=0.50,
        derived_auc=0.51,
        component_auc=0.54,
    )
    gate = adjudicate_selected8_parity(config, {"valid": True}, training)
    summary = {
        "metadata": {"mode": "diagnostic", "config": {"rounds": 3}},
        "result_rows": training["rows"],
        "gate": gate,
    }
    output = tmp_path / "curves.svg"

    render_selected8_parity(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "八个密文bit全异或为一个bit" in svg
    assert "真实输出值，不是样本分类" in svg
    assert "右移一位同重量mask" in svg
    assert "单bit派生parity" in svg


def test_result_index_names_ope1_in_plain_chinese() -> None:
    name = display_name_for_run(
        "i2_output_prediction_ope1_present_r3_r4_selected8_parity_r3_diagnostic_seed1_20260722"
    )

    assert name == "创新2 OPE1：PRESENT三至四轮八个密文bit全异或真实值预测"


def _synthetic_training(
    *,
    candidate_auc: float,
    mlp_auc: float,
    geometry_auc: float,
    shuffle_auc: float,
    derived_auc: float,
    component_auc: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = [
        {
            "model": "selected8_mlp_true_output",
            "msb_index": bit,
            "threshold_accuracy": 0.53,
            "majority_accuracy": 0.50,
            "accuracy_minus_majority": 0.03,
            "auc": component_auc,
            "mse": 0.24,
        }
        for bit in SELECTED8_PARITY_MASK
    ]
    for model, auc in (
        ("selected8_parity_mlp_true_output", mlp_auc),
        ("selected8_parity_rescnn_true_output", candidate_auc),
        ("control8_parity_rescnn_true_output", geometry_auc),
        ("selected8_parity_rescnn_label_shuffle", shuffle_auc),
    ):
        row: dict[str, object] = {
            "model": model,
            "threshold_accuracy": 0.53,
            "majority_accuracy": 0.50,
            "accuracy_minus_majority": 0.03,
            "auc": auc,
            "mse": 0.24,
        }
        if model == "selected8_parity_rescnn_true_output":
            row.update(
                {
                    "mlp_auc": mlp_auc,
                    "geometry_auc": geometry_auc,
                    "shuffle_auc": shuffle_auc,
                    "derived_auc": derived_auc,
                    "best_component_auc": component_auc,
                    "auc_minus_mlp": candidate_auc - mlp_auc,
                    "auc_minus_geometry": candidate_auc - geometry_auc,
                    "auc_minus_shuffle": candidate_auc - shuffle_auc,
                    "auc_minus_derived": candidate_auc - derived_auc,
                    "auc_minus_best_component": candidate_auc - component_auc,
                }
            )
        rows.append(row)
    config = Selected8ParityConfig.diagnostic(rounds=3)
    return {
        "rows": rows,
        "summaries": [{"model": model} for model, *_ in MODEL_SPECS],
        "history": [
            {"model": model, "epoch": epoch}
            for model, *_ in MODEL_SPECS
            for epoch in range(1, config.epochs + 1)
        ],
        "checkpoints": [{"sha256": "hash"} for _ in MODEL_SPECS],
    }
