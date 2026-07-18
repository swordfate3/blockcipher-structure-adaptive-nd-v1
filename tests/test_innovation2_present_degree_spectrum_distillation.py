from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_innovation2_present_degree_spectrum_distillation import (
    main as plot_main,
)
from blockcipher_nd.models.structure.spn.present_monomial_support_propagation import (
    PresentDegreeSpectrumDistillationNetwork,
    PresentMspnSpec,
)
from blockcipher_nd.tasks.innovation2.present_certificate_complexity_attribution import (
    anf_prefix_features,
)
from blockcipher_nd.tasks.innovation2.present_degree_spectrum_distillation import (
    DegreeSpectrumDistillationConfig,
    adjudicate_e49,
    build_degree_spectrum_targets,
    build_teacher_bundle,
    shuffled_training_targets,
)
from blockcipher_nd.tasks.innovation2.present_universal_balance_atlas import (
    possible_active_monomials,
)


def present_player() -> np.ndarray:
    return np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)],
        dtype=np.int64,
    )


def small_data() -> dict[str, object]:
    active_bits = ((0, 1, 4, 5, 20, 21, 40, 41), tuple(range(8)))
    structure_active = np.zeros((2, 64), dtype=np.float32)
    for index, bits in enumerate(active_bits):
        structure_active[index, list(bits)] = 1.0
    output_mask_bits = np.zeros((2, 64), dtype=np.float32)
    output_mask_bits[0, [0, 1, 4, 5]] = 1.0
    output_mask_bits[1, [3, 19, 35, 51]] = 1.0
    rows = [
        {"split": "train", "structure_index": 0, "mask_index": 0, "label": 0},
        {"split": "train", "structure_index": 0, "mask_index": 1, "label": 1},
        {"split": "train", "structure_index": 1, "mask_index": 0, "label": 1},
        {"split": "train", "structure_index": 1, "mask_index": 1, "label": 0},
        {
            "split": "validation",
            "structure_index": 0,
            "mask_index": 0,
            "label": 0,
        },
        {
            "split": "validation",
            "structure_index": 1,
            "mask_index": 1,
            "label": 1,
        },
    ]
    return {
        "players": present_player()[None, :],
        "structures": [
            {"active_bits": list(bits)} for bits in active_bits
        ],
        "structure_active": structure_active,
        "output_mask_bits": output_mask_bits,
        "rows": rows,
    }


def test_true_teacher_matches_e45_prefix_definition_and_corrupted_differs() -> None:
    data = small_data()
    true_targets = build_degree_spectrum_targets(data, "true")
    corrupted_targets = build_degree_spectrum_targets(data, "corrupted")
    selected = np.flatnonzero(data["output_mask_bits"][0])
    supports = {
        rounds: possible_active_monomials(
            tuple(data["structures"][0]["active_bits"]), rounds
        )
        for rounds in (1, 2, 3)
    }
    expected = anf_prefix_features(selected, supports).reshape(3, 13)

    assert true_targets.shape == (6, 3, 13)
    assert np.allclose(true_targets[0], expected)
    assert not np.array_equal(true_targets, corrupted_targets)


def test_teacher_bundle_uses_train_only_normalization() -> None:
    teachers = build_teacher_bundle(small_data())

    assert teachers["train_mask"].sum() == 4
    assert teachers["validation_mask"].sum() == 2
    assert teachers["normalized"]["true"].shape == (6, 3, 13)
    assert np.isfinite(teachers["normalized"]["true"]).all()
    assert np.allclose(
        teachers["normalized"]["true"][teachers["train_mask"]].mean(axis=0),
        0.0,
        atol=1e-6,
    )


def test_teacher_shuffle_preserves_columns_and_validation_rows() -> None:
    targets = np.arange(10 * 3 * 13, dtype=np.float32).reshape(10, 3, 13)
    train_mask = np.asarray([True] * 8 + [False] * 2)

    shuffled, permutation = shuffled_training_targets(targets, train_mask)

    assert not np.array_equal(permutation, np.arange(8))
    assert np.array_equal(shuffled[~train_mask], targets[~train_mask])
    assert np.array_equal(
        np.sort(shuffled[train_mask].reshape(8, -1), axis=0),
        np.sort(targets[train_mask].reshape(8, -1), axis=0),
    )


def test_auxiliary_head_is_not_a_direct_balance_input() -> None:
    data = small_data()
    model = PresentDegreeSpectrumDistillationNetwork(
        PresentMspnSpec(dropout=0.0),
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    )
    model.eval()
    inputs = (
        torch.zeros(2, dtype=torch.int64),
        torch.asarray([0, 1], dtype=torch.int64),
        torch.asarray([0, 1], dtype=torch.int64),
    )
    with torch.no_grad():
        before = model(*inputs)
        for parameter in model.spectrum_head.parameters():
            parameter.add_(torch.randn_like(parameter))
        after = model(*inputs)
        logits, spectrum = model.forward_with_auxiliary(*inputs)

    assert torch.equal(before, after)
    assert logits.shape == (2,)
    assert spectrum.shape == (2, 3, 13)
    assert torch.isfinite(spectrum).all()


def test_e49_gate_passes_when_true_teacher_is_learned() -> None:
    gate = adjudicate_e49(
        DegreeSpectrumDistillationConfig(run_id="e49-test"),
        {"source_valid": True},
        valid_contract(),
        teacher_fixture(),
        matrix_fixture(true_mse=0.70, shuffle_mse=0.90, true_auc=0.50),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_degree_spectrum_readiness_passed"
    assert gate["next_action"]["formal_seed0"] is True


def test_e49_gate_holds_when_true_teacher_does_not_beat_shuffle() -> None:
    gate = adjudicate_e49(
        DegreeSpectrumDistillationConfig(run_id="e49-test"),
        {"source_valid": True},
        valid_contract(),
        teacher_fixture(),
        matrix_fixture(true_mse=0.92, shuffle_mse=0.95, true_auc=0.51),
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_degree_spectrum_not_learned"
    assert gate["next_action"]["formal_seed0"] is False


def test_plot_writes_chinese_e49_svg(tmp_path: Path) -> None:
    matrix = matrix_fixture(true_mse=0.70, shuffle_mse=0.90, true_auc=0.50)
    gate = adjudicate_e49(
        DegreeSpectrumDistillationConfig(run_id="e49-test"),
        {"source_valid": True},
        valid_contract(),
        teacher_fixture(),
        matrix,
    )
    summary_path = tmp_path / "summary.json"
    history_path = tmp_path / "history.csv"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(
        json.dumps({"rows": matrix["rows"], "gate": gate}), encoding="utf-8"
    )
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix["history"][0]))
        writer.writeheader()
        writer.writerows(matrix["history"])

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
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E49" in svg
    assert "允许另建E50正式计划" in svg


def valid_contract() -> dict[str, object]:
    return {
        "teacher_shape": [10, 3, 13],
        "teacher_values_finite": True,
        "parameter_count": 18281,
        "parameter_ratio_to_e47": 1.03,
        "auxiliary_head_parameter_count": 493,
        "auxiliary_parameter_ratio_to_e47": 0.028,
        "auxiliary_prediction_shape": [8, 3, 13],
        "direct_logit_delta_after_auxiliary_head_change": 0.0,
        "balance_head_width_unchanged": True,
        "logits_finite": True,
        "spectrum_finite": True,
        "losses_finite": True,
        "gradients_finite": True,
        "teacher_buffers_absent": True,
    }


def teacher_fixture() -> dict[str, object]:
    rng = np.random.default_rng(49)
    true = rng.normal(size=(10, 3, 13)).astype(np.float32)
    corrupted = (true + 0.5).astype(np.float32)
    return {
        "raw": {"true": true, "corrupted": corrupted},
        "normalized": {"true": true, "corrupted": corrupted},
        "train_mask": np.asarray([True] * 8 + [False] * 2),
        "validation_mask": np.asarray([False] * 8 + [True] * 2),
    }


def matrix_fixture(
    *, true_mse: float, shuffle_mse: float, true_auc: float
) -> dict[str, object]:
    rows = [
        {
            "row_id": "e47_mspn_true_label_only_anchor",
            "training_performed": False,
            "validation_auc": 0.518673,
        },
        trained_row(
            "mspn_spectrum_true_seed0", "true", true_auc, true_mse
        ),
        trained_row(
            "mspn_spectrum_target_shuffle_seed0",
            "shuffled",
            0.49,
            shuffle_mse,
        ),
        trained_row(
            "mspn_spectrum_corrupted_seed0", "corrupted", 0.48, 0.85
        ),
    ]
    history = [
        {
            "row_id": row["row_id"],
            "epoch": epoch,
            "train_balance_loss": 0.70 - epoch * 0.01,
            "train_auxiliary_normalized_mse": 1.1 - epoch * 0.05,
            "validation_balance_loss": 0.70,
            "validation_auc": row["validation_auc"],
            "validation_teacher_normalized_mse": row[
                "validation_teacher_normalized_mse"
            ],
        }
        for row in rows
        if row["training_performed"]
        for epoch in (1, 2)
    ]
    return {"rows": rows, "history": history}


def trained_row(
    row_id: str, teacher_mode: str, auc: float, mse: float
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "topology_mode": "corrupted" if teacher_mode == "corrupted" else "true",
        "teacher_mode": teacher_mode,
        "train_auc": 0.55,
        "validation_auc": auc,
        "train_balance_loss": 0.69,
        "validation_balance_loss": 0.70,
        "train_teacher_normalized_mse": 0.80,
        "validation_teacher_normalized_mse": mse,
        "training_performed": True,
    }
