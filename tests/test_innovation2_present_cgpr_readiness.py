from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.cli.plot_innovation2_present_cgpr_readiness import (
    main as plot_main,
)
from blockcipher_nd.models.structure.spn.present_certificate_guided_pair_residual import (
    PresentCertificateGuidedPairResidual,
    PresentCgprSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_pair_relation_models import (
    SmallSpnPairRelationReasoner,
    SmallSpnPairRelationSpec,
)
from blockcipher_nd.tasks.innovation2.present_certificate_guided_pair_residual import (
    CgprReadinessConfig,
    adjudicate_e50,
    build_prefix_ridge_bundle,
    measure_cgpr_contract,
    train_cgpr_matrix,
)


def present_player() -> np.ndarray:
    return np.asarray(
        [(16 * bit) % 63 if bit < 63 else 63 for bit in range(64)],
        dtype=np.int64,
    )


def small_data() -> dict[str, object]:
    active_bits = (
        (0, 1, 4, 5, 20, 21, 40, 41),
        tuple(range(8)),
        (8, 9, 12, 13, 28, 29, 48, 49),
        tuple(range(16, 24)),
    )
    structure_active = np.zeros((4, 64), dtype=np.float32)
    for index, bits in enumerate(active_bits):
        structure_active[index, list(bits)] = 1.0
    output_mask_bits = np.zeros((4, 64), dtype=np.float32)
    output_mask_bits[0, [0, 1, 4, 5]] = 1.0
    output_mask_bits[1, [3, 19, 35, 51]] = 1.0
    output_mask_bits[2, [8, 9, 12, 13]] = 1.0
    output_mask_bits[3, [15, 31, 47, 63]] = 1.0
    rows = [
        {"split": "train", "structure_index": 0, "mask_index": 0, "label": 0},
        {"split": "train", "structure_index": 0, "mask_index": 1, "label": 1},
        {"split": "train", "structure_index": 1, "mask_index": 0, "label": 1},
        {"split": "train", "structure_index": 1, "mask_index": 1, "label": 0},
        {
            "split": "validation",
            "structure_index": 2,
            "mask_index": 2,
            "label": 0,
        },
        {
            "split": "validation",
            "structure_index": 2,
            "mask_index": 3,
            "label": 1,
        },
        {
            "split": "validation",
            "structure_index": 3,
            "mask_index": 2,
            "label": 1,
        },
        {
            "split": "validation",
            "structure_index": 3,
            "mask_index": 3,
            "label": 0,
        },
    ]
    return {
        "players": present_player()[None, :],
        "sboxes": np.asarray([PRESENT_SBOX], dtype=np.uint8),
        "structures": [
            {"active_bits": list(bits)} for bits in active_bits
        ],
        "structure_active": structure_active,
        "output_mask_bits": output_mask_bits,
        "rows": rows,
    }


def config() -> CgprReadinessConfig:
    return CgprReadinessConfig(run_id="e50-test")


def test_pair_reasoner_forward_is_head_of_exposed_embedding() -> None:
    data = small_data()
    model = SmallSpnPairRelationReasoner(
        SmallSpnPairRelationSpec(
            state_bits=64,
            round_categories=1,
            round_step_offset=4,
            hidden_dim=8,
            path_rank=2,
            dropout=0.0,
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    )
    model.eval()
    inputs = (
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        torch.asarray([0, 1], dtype=torch.int64),
        torch.asarray([0, 1], dtype=torch.int64),
    )

    with torch.no_grad():
        embedding = model.encode(*inputs)
        direct = model(*inputs)
        reconstructed = model.head(embedding).squeeze(-1)

    assert embedding.shape == (2, 7 * model.spec.hidden_dim)
    assert torch.equal(direct, reconstructed)


def test_prefix_ridge_bundle_uses_39_nonoracle_features() -> None:
    ridge = build_prefix_ridge_bundle(small_data())

    assert ridge["features"].shape == (8, 39)
    assert ridge["weights"].shape == (40,)
    assert ridge["train_standardization_only"]
    assert np.isfinite(ridge["features"]).all()
    assert np.isfinite(ridge["scores"]).all()
    assert 0.0 <= ridge["validation_auc"] <= 1.0


def test_cgpr_zero_residual_matches_ridge_and_parameters_are_fair() -> None:
    data = small_data()
    ridge = build_prefix_ridge_bundle(data)
    contract = measure_cgpr_contract(config(), data, ridge)

    assert contract["prefix_shape"] == [8, 39]
    assert contract["zero_residual_prefix_max_abs_error"] == 0.0
    assert contract["zero_residual_true_max_abs_error"] == 0.0
    assert contract["zero_residual_corrupted_max_abs_error"] == 0.0
    assert contract["true_corrupted_pair_embedding_max_abs_difference"] >= 1e-5
    assert contract["parameter_relative_spread"] <= 0.01
    assert contract["ridge_buffers_require_grad_false"]
    assert contract["logits_finite"]
    assert contract["loss_finite"]
    assert contract["gradients_finite"]


def test_prefix_and_pair_models_have_bounded_zero_initialized_residuals() -> None:
    data = small_data()
    ridge = build_prefix_ridge_bundle(data)
    models = [
        make_model(data, ridge, "prefix", "true"),
        make_model(data, ridge, "pair", "true"),
        make_model(data, ridge, "pair", "corrupted"),
    ]
    inputs = (
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        torch.asarray([0, 1], dtype=torch.int64),
        torch.asarray([0, 1], dtype=torch.int64),
        torch.from_numpy(ridge["features"][:2].astype(np.float32)),
    )

    with torch.no_grad():
        base = models[0].base_score(inputs[-1])
        outputs = [model(*inputs) for model in models]

    assert all(torch.equal(output, base) for output in outputs)
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert (max(counts) - min(counts)) / max(counts) <= 0.01


def test_two_epoch_cgpr_matrix_completes_all_controls() -> None:
    data = small_data()
    ridge = build_prefix_ridge_bundle(data)
    matrix = train_cgpr_matrix(config(), data, ridge)

    trained = [row for row in matrix["rows"] if row["training_performed"]]
    assert len(matrix["rows"]) == 4
    assert len(trained) == 3
    assert len(matrix["history"]) == 6
    assert all(row["ridge_weight_max_delta"] == 0.0 for row in trained)
    assert all(np.isfinite(row["validation_auc"]) for row in trained)


def test_e50_gate_accepts_valid_readiness_contract() -> None:
    matrix = matrix_fixture()
    gate = adjudicate_e50(
        config(), {"source_valid": True}, valid_contract(), matrix
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_cgpr_readiness_passed"
    assert gate["next_action"]["formal_seed0"] is True


def test_plot_writes_chinese_e50_svg(tmp_path: Path) -> None:
    matrix = matrix_fixture()
    gate = adjudicate_e50(
        config(), {"source_valid": True}, valid_contract(), matrix
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(
        json.dumps({"rows": matrix["rows"], "gate": gate}), encoding="utf-8"
    )

    assert (
        plot_main(["--summary", str(summary_path), "--output", str(output_path)])
        == 0
    )
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E50" in svg
    assert "允许另建E51正式计划" in svg


def make_model(
    data: dict[str, object],
    ridge: dict[str, object],
    residual_mode: str,
    topology_mode: str,
) -> PresentCertificateGuidedPairResidual:
    return PresentCertificateGuidedPairResidual(
        PresentCgprSpec(
            residual_mode=residual_mode,
            topology_mode=topology_mode,
            dropout=0.0,
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
        ridge_mean=ridge["mean"],
        ridge_scale=ridge["scale"],
        ridge_weights=ridge["weights"],
    )


def valid_contract() -> dict[str, object]:
    return {
        "prefix_shape": [1036, 39],
        "ridge_validation_auc": 0.6860815857512209,
        "train_standardization_only": True,
        "parameter_counts": {
            "prefix": 10659,
            "pair_true": 10725,
            "pair_corrupted": 10725,
        },
        "parameter_relative_spread": 0.0062,
        "zero_residual_prefix_max_abs_error": 0.0,
        "zero_residual_true_max_abs_error": 0.0,
        "zero_residual_corrupted_max_abs_error": 0.0,
        "true_corrupted_pair_embedding_max_abs_difference": 0.05,
        "ridge_buffers_require_grad_false": True,
        "logits_finite": True,
        "loss_finite": True,
        "gradients_finite": True,
        "forbidden_buffers_absent": True,
    }


def matrix_fixture() -> dict[str, object]:
    rows = [
        {
            "row_id": "e45_anf_prefix_ridge_anchor",
            "residual_mode": "off",
            "topology_mode": "none",
            "train_auc": 0.70,
            "validation_auc": 0.6860815857512209,
            "ridge_weight_max_delta": 0.0,
            "training_performed": False,
        },
        trained_row("cgpr_prefix_only_seed0", "prefix", "true", 0.72, 0.68),
        trained_row("cgpr_pair_true_seed0", "pair", "true", 0.73, 0.69),
        trained_row(
            "cgpr_pair_corrupted_seed0", "pair", "corrupted", 0.71, 0.67
        ),
    ]
    history = [
        {"row_id": row["row_id"], "epoch": epoch}
        for row in rows
        if row["training_performed"]
        for epoch in (1, 2)
    ]
    return {"rows": rows, "history": history}


def trained_row(
    row_id: str,
    residual_mode: str,
    topology_mode: str,
    train_auc: float,
    validation_auc: float,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "residual_mode": residual_mode,
        "topology_mode": topology_mode,
        "train_auc": train_auc,
        "validation_auc": validation_auc,
        "train_loss": 0.68,
        "validation_loss": 0.69,
        "ridge_weight_max_delta": 0.0,
        "training_performed": True,
    }
