from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.plot_innovation2_present_pair_state_attribution import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_pair_state_neural_attribution import (
    PresentPairStateTrainingConfig,
    adjudicate_e44,
    load_e43_source,
    measure_model_contract,
    train_pair_state_model,
    validate_e43_source,
)


def make_source(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "gate.json").write_text(
        json.dumps(
            {
                "run_id": "i2_present_r4_universal_balance_atlas_20260718",
                "status": "pass",
                "decision": "innovation2_present_universal_balance_atlas_ready",
            }
        ),
        encoding="utf-8",
    )
    structures = [
        {
            "index": index,
            "active_bits": list(range(8 * index, 8 * index + 8)),
        }
        for index in range(4)
    ]
    masks = [
        {"index": index, "bits": [8 * index, 8 * index + 1]}
        for index in range(4)
    ]
    (root / "structures.json").write_text(
        json.dumps({"structures": structures}), encoding="utf-8"
    )
    (root / "masks.json").write_text(
        json.dumps({"masks": masks}), encoding="utf-8"
    )
    rows = [
        ("train", 0, 0, 0, 1),
        ("train", 0, 0, 1, 0),
        ("train", 0, 1, 0, 0),
        ("train", 0, 1, 1, 1),
        ("validation", 0, 2, 2, 1),
        ("validation", 0, 2, 3, 0),
        ("validation", 0, 3, 2, 0),
        ("validation", 0, 3, 3, 1),
    ]
    with (root / "matched_contrast.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "rectangle_index",
                "structure_index",
                "mask_index",
                "label",
            ]
        )
        writer.writerows(rows)
    return root


def smoke_config() -> PresentPairStateTrainingConfig:
    return PresentPairStateTrainingConfig(
        run_id="i2_present_pair_state_smoke_test",
        mode="smoke",
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        path_rank=2,
        dropout=0.0,
    )


def test_source_loader_and_protocol_checks(tmp_path: Path) -> None:
    data = load_e43_source(make_source(tmp_path / "source"))
    checks = validate_e43_source(data, strict=False)

    assert all(checks.values())
    assert data["structure_active"].shape == (4, 64)
    assert data["output_mask_bits"].shape == (4, 64)
    assert np.array_equal(np.sort(data["players"][0]), np.arange(64))


def test_64_bit_pair_state_contract_is_finite(tmp_path: Path) -> None:
    data = load_e43_source(make_source(tmp_path / "source"))
    contract = measure_model_contract(smoke_config(), data)

    assert contract["initial_pair_shape"] == [8, 64, 64, 8]
    assert contract["pair_count"] == 4096
    assert contract["parameter_counts_match"]
    assert contract["logits_finite"]
    assert contract["loss_finite"]
    assert contract["gradients_finite"]
    assert contract["corrupted_player_is_permutation"]
    assert contract["corrupted_player_differs"]
    assert contract["true_corrupted_max_abs_logit_difference"] >= 1e-5
    assert contract["step_schedule"] == [4]


def test_smoke_pair_local_training_produces_finite_metrics(tmp_path: Path) -> None:
    data = load_e43_source(make_source(tmp_path / "source"))
    output = train_pair_state_model(
        smoke_config(), data, processor_mode="local", topology_mode="true"
    )

    assert len(output["history"]) == 1
    assert np.isfinite(output["result"]["train_auc"])
    assert np.isfinite(output["result"]["validation_auc"])


def test_adjudication_requires_candidate_and_topology_margins(tmp_path: Path) -> None:
    data = load_e43_source(make_source(tmp_path / "source"))
    config = smoke_config()
    source_checks = validate_e43_source(data, strict=False)
    contract = measure_model_contract(config, data)
    rows = [
        {
            "row_id": "unary_marginal_baseline",
            "processor_mode": "none",
            "topology_mode": "none",
            "seed": 0,
            "train_auc": 0.5,
            "validation_auc": 0.5,
            "training_performed": False,
        },
        {
            "row_id": "pair_local_true_seed0",
            "processor_mode": "local",
            "topology_mode": "true",
            "seed": 0,
            "train_auc": 0.75,
            "validation_auc": 0.66,
            "training_performed": True,
        },
        {
            "row_id": "pair_triangle_true_seed0",
            "processor_mode": "triangle",
            "topology_mode": "true",
            "seed": 0,
            "train_auc": 0.72,
            "validation_auc": 0.64,
            "training_performed": True,
        },
        {
            "row_id": "pair_local_corrupted_seed0",
            "processor_mode": "local",
            "topology_mode": "corrupted",
            "seed": 0,
            "train_auc": 0.61,
            "validation_auc": 0.60,
            "training_performed": True,
        },
    ]

    gate = adjudicate_e44(
        config,
        source_checks,
        contract,
        {"rows": rows, "history": [], "selected_processor": "local"},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_pair_state_topology_attributed"
    assert np.isclose(gate["metrics"]["best_true_minus_unary"], 0.16)
    assert np.isclose(gate["metrics"]["best_true_minus_corrupted"], 0.06)


def test_plot_writes_chinese_e44_svg(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {
                "row_id": "unary_marginal_baseline",
                "training_performed": False,
                "validation_auc": 0.5,
            },
            {
                "row_id": "pair_local_true_seed0",
                "training_performed": True,
                "validation_auc": 0.66,
            },
            {
                "row_id": "pair_triangle_true_seed0",
                "training_performed": True,
                "validation_auc": 0.64,
            },
            {
                "row_id": "pair_local_corrupted_seed0",
                "training_performed": True,
                "validation_auc": 0.60,
            },
        ],
        "gate": {
            "decision": "innovation2_present_pair_state_topology_attributed",
            "metrics": {
                "selected_processor": "local",
                "best_true_minus_unary": 0.16,
                "best_true_minus_corrupted": 0.06,
            },
        },
    }
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
        for row_id, auc in (
            ("pair_local_true_seed0", 0.66),
            ("pair_triangle_true_seed0", 0.64),
            ("pair_local_corrupted_seed0", 0.60),
        ):
            writer.writerow(
                {
                    "row_id": row_id,
                    "epoch": 1,
                    "train_loss": 0.7,
                    "validation_loss": 0.69,
                    "validation_auc": auc,
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
    assert "创新2 E44" in output_path.read_text(encoding="utf-8")
