from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_present_mspn_attribution import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_mspn_attribution import (
    MspnAttributionConfig,
    adjudicate_e47,
)
from blockcipher_nd.tasks.innovation2.present_mspn_readiness import (
    E44_TRIANGLE_AUC,
    E45_PREFIX_AUC,
)


def valid_contract() -> dict[str, object]:
    return {
        "initial_state_shape": [8, 64, 32],
        "cell_relabeling_max_abs_logit_error": 5e-8,
        "true_corrupted_max_abs_logit_difference": 0.05,
        "parameter_ratio_to_e44": 1.65,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
        "precomputed_certificate_feature_buffers_absent": True,
        "parameter_count": 17788,
    }


def make_matrix(
    *, true_auc: float, corrupted_auc: float, shuffle_auc: float
) -> dict[str, object]:
    trained = [
        {
            "row_id": "mspn_true_seed0",
            "topology_mode": "true",
            "label_mode": "true",
            "train_auc": 0.72,
            "validation_auc": true_auc,
            "train_loss": 0.62,
            "validation_loss": 0.64,
            "training_performed": True,
        },
        {
            "row_id": "mspn_corrupted_seed0",
            "topology_mode": "corrupted",
            "label_mode": "true",
            "train_auc": 0.63,
            "validation_auc": corrupted_auc,
            "train_loss": 0.66,
            "validation_loss": 0.67,
            "training_performed": True,
        },
        {
            "row_id": "mspn_label_shuffle_seed0",
            "topology_mode": "true",
            "label_mode": "shuffled",
            "train_auc": 0.51,
            "validation_auc": shuffle_auc,
            "train_loss": 0.69,
            "validation_loss": 0.69,
            "training_performed": True,
        },
    ]
    return {
        "rows": [
            {
                "row_id": "e45_anf_prefix_ridge_anchor",
                "validation_auc": E45_PREFIX_AUC,
                "training_performed": False,
            },
            {
                "row_id": "e44_triangle_anchor",
                "validation_auc": E44_TRIANGLE_AUC,
                "training_performed": False,
            },
            *trained,
        ],
        "history": [
            {"row_id": row["row_id"], "epoch": epoch}
            for row in trained
            for epoch in range(1, 31)
        ],
    }


def test_e47_config_is_frozen() -> None:
    config = MspnAttributionConfig(run_id="e47-test")

    assert config.epochs == 30
    assert config.batch_size == 32
    assert config.hidden_dim == 32
    assert config.degree_channels == 9


def test_e47_gate_passes_candidate_and_topology_margins() -> None:
    matrix = make_matrix(true_auc=0.66, corrupted_auc=0.61, shuffle_auc=0.50)

    gate = adjudicate_e47(
        MspnAttributionConfig(run_id="e47-test"),
        {"source_valid": True},
        valid_contract(),
        matrix,
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_mspn_topology_attributed"
    assert np.isclose(gate["metrics"]["mspn_true_minus_corrupted"], 0.05)
    assert np.isclose(gate["metrics"]["mspn_true_minus_e44"], 0.0980206837115771)


def test_e47_gate_holds_when_topology_margin_is_missing() -> None:
    matrix = make_matrix(true_auc=0.66, corrupted_auc=0.65, shuffle_auc=0.50)

    gate = adjudicate_e47(
        MspnAttributionConfig(run_id="e47-test"),
        {"source_valid": True},
        valid_contract(),
        matrix,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_mspn_topology_not_attributed"


def test_plot_writes_chinese_e47_svg(tmp_path: Path) -> None:
    matrix = make_matrix(true_auc=0.66, corrupted_auc=0.61, shuffle_auc=0.50)
    gate = adjudicate_e47(
        MspnAttributionConfig(run_id="e47-test"),
        {"source_valid": True},
        valid_contract(),
        matrix,
    )
    rows = matrix["rows"]
    summary = {"rows": rows, "gate": gate}
    summary_path = tmp_path / "summary.json"
    history_path = tmp_path / "history.csv"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "row_id",
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_auc",
            ),
        )
        writer.writeheader()
        final_auc = {
            "mspn_true_seed0": 0.66,
            "mspn_corrupted_seed0": 0.61,
            "mspn_label_shuffle_seed0": 0.50,
        }
        for row_id, target in final_auc.items():
            for epoch in range(1, 31):
                writer.writerow(
                    {
                        "row_id": row_id,
                        "epoch": epoch,
                        "train_loss": 0.70 - epoch * 0.002,
                        "validation_loss": 0.70 - epoch * 0.001,
                        "validation_auc": 0.5 + (target - 0.5) * epoch / 30,
                    }
                )

    assert (
        plot_main(
            [
                "--summary",
                str(summary_path),
                "--history",
                str(history_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert "创新2 E47" in output_path.read_text(encoding="utf-8")
