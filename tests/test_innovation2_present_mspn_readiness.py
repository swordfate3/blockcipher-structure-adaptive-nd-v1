from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, PRESENT_SBOX_ANF
from blockcipher_nd.cli.plot_innovation2_present_mspn_readiness import (
    main as plot_main,
)
from blockcipher_nd.models.structure.spn.present_monomial_support_propagation import (
    PresentMonomialSupportPropagationNetwork,
    PresentMspnSpec,
)
from blockcipher_nd.tasks.innovation2.present_mspn_readiness import (
    E44_PARAMETER_ANCHOR,
    E44_TRIANGLE_AUC,
    E45_PREFIX_AUC,
    MspnReadinessConfig,
    adjudicate_e46,
    measure_mspn_contract,
    train_readiness_matrix,
)


def make_data() -> dict[str, object]:
    structure_active = np.zeros((4, 64), dtype=np.float32)
    output_mask_bits = np.zeros((4, 64), dtype=np.float32)
    for index in range(4):
        structure_active[index, 8 * index : 8 * index + 8] = 1.0
        output_mask_bits[index, [8 * index, 8 * index + 1]] = 1.0
    rows = [
        {"split": "train", "structure_index": 0, "mask_index": 0, "label": 1},
        {"split": "train", "structure_index": 0, "mask_index": 1, "label": 0},
        {"split": "train", "structure_index": 1, "mask_index": 0, "label": 0},
        {"split": "train", "structure_index": 1, "mask_index": 1, "label": 1},
        {"split": "validation", "structure_index": 2, "mask_index": 2, "label": 1},
        {"split": "validation", "structure_index": 2, "mask_index": 3, "label": 0},
        {"split": "validation", "structure_index": 3, "mask_index": 2, "label": 0},
        {"split": "validation", "structure_index": 3, "mask_index": 3, "label": 1},
    ]
    return {
        "players": np.asarray(
            [[(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)]],
            dtype=np.int64,
        ),
        "structure_active": structure_active,
        "output_mask_bits": output_mask_bits,
        "rows": rows,
    }


def config() -> MspnReadinessConfig:
    return MspnReadinessConfig(run_id="i2_mspn_readiness_test")


def test_present_sbox_anf_fixture_reconstructs_truth_table() -> None:
    reconstructed = []
    for value in range(16):
        output = 0
        for output_bit, terms in enumerate(PRESENT_SBOX_ANF):
            coordinate = 0
            for term in terms:
                coordinate ^= int((value & term) == term)
            output |= coordinate << output_bit
        reconstructed.append(output)

    assert tuple(reconstructed) == PRESENT_SBOX


def test_mspn_forward_uses_64_bit_state_and_is_finite() -> None:
    data = make_data()
    model = PresentMonomialSupportPropagationNetwork(
        PresentMspnSpec(dropout=0.0),
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    )
    variants = torch.zeros(4, dtype=torch.long)
    structures = torch.arange(4, dtype=torch.long)
    masks = torch.arange(4, dtype=torch.long)

    logits = model(variants, structures, masks)
    initial, _, _ = model.build_initial_state(structures, masks)

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    assert initial.shape == (4, 64, 32)
    assert model.spec.rounds == 4


def test_mspn_contract_passes_equivariance_and_leakage_checks() -> None:
    contract = measure_mspn_contract(config(), make_data())

    assert contract["initial_state_shape"] == [8, 64, 32]
    assert contract["execution_rounds"] == 4
    assert contract["shared_step_count"] == 1
    assert contract["logits_finite"]
    assert contract["loss_finite"]
    assert contract["gradients_finite"]
    assert contract["cell_relabeling_max_abs_logit_error"] <= 1e-6
    assert contract["true_corrupted_max_abs_logit_difference"] >= 1e-5
    assert 0.5 <= contract["parameter_ratio_to_e44"] <= 2.0
    assert contract["allowed_buffer_names_only"]
    assert contract["precomputed_certificate_feature_buffers_absent"]


def test_two_epoch_smoke_matrix_completes_all_mspn_controls() -> None:
    matrix = train_readiness_matrix(config(), make_data())

    trained = [row for row in matrix["rows"] if row["training_performed"]]
    assert len(matrix["rows"]) == 5
    assert len(trained) == 3
    assert {row["row_id"] for row in trained} == {
        "mspn_true_seed0",
        "mspn_corrupted_seed0",
        "mspn_label_shuffle_seed0",
    }
    assert len(matrix["history"]) == 6
    assert all(np.isfinite(row["validation_auc"]) for row in trained)


def test_readiness_gate_accepts_valid_synthetic_metrics() -> None:
    contract = measure_mspn_contract(config(), make_data())
    matrix = {
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
            {
                "row_id": "mspn_true_seed0",
                "topology_mode": "true",
                "label_mode": "true",
                "train_auc": 0.55,
                "validation_auc": 0.54,
                "train_loss": 0.69,
                "validation_loss": 0.69,
                "training_performed": True,
            },
            {
                "row_id": "mspn_corrupted_seed0",
                "topology_mode": "corrupted",
                "label_mode": "true",
                "train_auc": 0.52,
                "validation_auc": 0.51,
                "train_loss": 0.69,
                "validation_loss": 0.69,
                "training_performed": True,
            },
            {
                "row_id": "mspn_label_shuffle_seed0",
                "topology_mode": "true",
                "label_mode": "shuffled",
                "train_auc": 0.50,
                "validation_auc": 0.50,
                "train_loss": 0.69,
                "validation_loss": 0.69,
                "training_performed": True,
            },
        ],
        "history": [
            {"row_id": row_id, "epoch": epoch}
            for row_id in (
                "mspn_true_seed0",
                "mspn_corrupted_seed0",
                "mspn_label_shuffle_seed0",
            )
            for epoch in (1, 2)
        ],
    }

    gate = adjudicate_e46(config(), {"source_valid": True}, contract, matrix)

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_mspn_readiness_passed"
    assert gate["metrics"]["parameter_count"] != E44_PARAMETER_ANCHOR


def test_plot_writes_chinese_e46_svg(tmp_path: Path) -> None:
    rows = [
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
        {
            "row_id": "mspn_true_seed0",
            "validation_auc": 0.54,
            "training_performed": True,
        },
        {
            "row_id": "mspn_corrupted_seed0",
            "validation_auc": 0.51,
            "training_performed": True,
        },
        {
            "row_id": "mspn_label_shuffle_seed0",
            "validation_auc": 0.50,
            "training_performed": True,
        },
    ]
    summary = {
        "rows": rows,
        "gate": {
            "decision": "innovation2_present_mspn_readiness_passed",
            "metrics": {
                "parameter_count": 17788,
                "parameter_ratio_to_e44": 17788 / E44_PARAMETER_ANCHOR,
                "cell_relabeling_max_abs_logit_error": 3e-8,
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
        for row_id, start in (
            ("mspn_true_seed0", 0.50),
            ("mspn_corrupted_seed0", 0.49),
            ("mspn_label_shuffle_seed0", 0.50),
        ):
            for epoch in (1, 2):
                writer.writerow(
                    {
                        "row_id": row_id,
                        "epoch": epoch,
                        "train_loss": 0.69,
                        "validation_loss": 0.69,
                        "validation_auc": start + 0.01 * epoch,
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
    assert "创新2 E46" in output_path.read_text(encoding="utf-8")
