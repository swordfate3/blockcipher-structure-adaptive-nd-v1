from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_innovation2_small_spn_topology import (
    render_topology_training_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_cell_equivariance import (
    render_cell_equivariance_svg,
)
from blockcipher_nd.models.structure.spn.small_spn_graph_models import (
    BasisSetEncoder,
    SmallSpnModelSpec,
    SmallSpnTopologyPredictor,
)
from blockcipher_nd.tasks.innovation2 import small_spn_exact_labels as labels
from blockcipher_nd.tasks.innovation2 import small_spn_cell_equivariance as equivariance
from blockcipher_nd.tasks.innovation2 import small_spn_topology_training as training


def _model_data() -> dict[str, np.ndarray]:
    config = labels.SmallSpnAuditConfig(run_id="model")
    variants = labels.make_variants(config)
    structures = labels.make_structures()
    masks = labels.make_output_masks()
    active = np.zeros((14, 16), dtype=np.float32)
    basis = np.zeros((14, 12, 16), dtype=np.float32)
    valid = np.zeros((14, 12), dtype=np.bool_)
    for structure_index, structure in enumerate(structures):
        active[structure_index, list(structure.active_bits)] = 1
        for row, bit in enumerate(structure.active_bits):
            basis[structure_index, row, bit] = 1
            valid[structure_index, row] = True
    mask_bits = np.asarray(
        [[(mask >> bit) & 1 for bit in range(16)] for mask in masks],
        dtype=np.float32,
    )
    return {
        "sboxes": np.asarray([variant.sbox for variant in variants]),
        "players": np.asarray([variant.player for variant in variants]),
        "structure_active": active,
        "structure_basis": basis,
        "structure_basis_valid": valid,
        "output_mask_bits": mask_bits,
    }


def _model(
    model_name: str,
    topology_mode: str = "true",
    position_mode: str = "absolute",
) -> SmallSpnTopologyPredictor:
    data = _model_data()
    return SmallSpnTopologyPredictor(
        SmallSpnModelSpec(
            model_name=model_name,
            topology_mode=topology_mode,
            position_mode=position_mode,
            hidden_dim=32,
            blocks=2,
            heads=4,
            dropout=0.0,
        ),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        structure_basis=data["structure_basis"],
        structure_basis_valid=data["structure_basis_valid"],
        output_mask_bits=data["output_mask_bits"],
    )


def test_graphgps_and_scgt_forward_without_cipher_id_embedding() -> None:
    inputs = (
        torch.tensor([0, 15]),
        torch.tensor([0, 3]),
        torch.tensor([0, 13]),
        torch.tensor([0, 63]),
    )
    for model_name in ("graphgps", "scgt"):
        model = _model(model_name)
        output = model(*inputs)
        assert output.shape == (2,)
        assert torch.isfinite(output).all()
        assert not any("variant_embedding" in name for name, _ in model.named_modules())


def test_basis_set_encoder_is_permutation_invariant_in_eval_mode() -> None:
    encoder = BasisSetEncoder(hidden_dim=16, heads=4, dropout=0.0).eval()
    basis = torch.eye(16)[:6].unsqueeze(0)
    valid = torch.ones((1, 6), dtype=torch.bool)
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    with torch.no_grad():
        expected = encoder(basis, valid)
        actual = encoder(basis[:, permutation], valid[:, permutation])
    assert torch.allclose(expected, actual, atol=1e-6)


def test_cell_equivariant_mode_removes_absolute_ids_and_is_relabeling_invariant() -> None:
    model = _model("graphgps", position_mode="cell_equivariant")
    assert model.bit_embedding is None
    assert model.nibble_embedding is None
    assert model.lane_embedding is not None
    error = equivariance.measure_cell_relabeling_error(_model_data())
    assert error <= 1e-6


def test_training_matrix_and_gate_are_frozen() -> None:
    config = training.TopologyTrainingConfig(run_id="full")
    matrix = training.training_matrix(config)
    assert len(matrix) == 7
    assert sum(row.model_name == "scgt" for row in matrix) == 2
    assert sum(row.topology_mode == "shuffled" for row in matrix) == 2
    assert sum(row.label_mode == "shuffled" for row in matrix) == 1

    def row(model, topology, label, seed, dual, unseen_s=0.80, unseen_p=0.76):
        return {
            "model_name": model,
            "topology_mode": topology,
            "label_mode": label,
            "seed": seed,
            "best_validation_auc": 0.8,
            "train_auc": 0.8,
            "unseen_sbox_auc": unseen_s,
            "unseen_player_auc": unseen_p,
            "dual_unseen_auc": dual,
            "training_performed": True,
        }

    rows = [
        row("graphgps", "true", "true", 0, 0.80),
        row("graphgps", "true", "true", 1, 0.79),
        row("scgt", "true", "true", 0, 0.82),
        row("scgt", "true", "true", 1, 0.81),
        row("graphgps", "shuffled", "true", 0, 0.70),
        row("graphgps", "shuffled", "true", 1, 0.71),
        row("graphgps", "true", "shuffled", 0, 0.50, unseen_s=0.5, unseen_p=0.5),
    ]
    gate = training.adjudicate_topology_training(config, {"protocol": True}, rows)
    assert gate["decision"] == "innovation2_small_spn_topology_predictor_ready"
    assert gate["metrics"]["scgt_basis_branch_keep"] is True


def test_topology_plot_contains_models_controls_and_scope(tmp_path: Path) -> None:
    rows = []
    for model, topology, label, seeds, base in (
        ("graphgps", "true", "true", (0, 1), 0.80),
        ("scgt", "true", "true", (0, 1), 0.82),
        ("graphgps", "shuffled", "true", (0, 1), 0.70),
        ("graphgps", "true", "shuffled", (0,), 0.50),
    ):
        for seed in seeds:
            rows.append(
                {
                    "model_name": model,
                    "topology_mode": topology,
                    "label_mode": label,
                    "seed": seed,
                    "unseen_sbox_auc": base,
                    "unseen_player_auc": base,
                    "dual_unseen_auc": base,
                }
            )
    summary = {
        "rows": rows,
        "gate": {
            "decision": "innovation2_small_spn_topology_predictor_ready",
            "metrics": {"scgt_basis_branch_keep": True},
        },
    }
    output = tmp_path / "curves.svg"
    render_topology_training_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E33" in svg
    assert "错误P-layer" in svg
    assert "不是PRESENT/GIFT/SKINNY高轮结果" in svg


def test_cell_equivariance_matrix_gate_and_plot_are_frozen(tmp_path: Path) -> None:
    config = training.TopologyTrainingConfig(run_id="e33r")
    matrix = equivariance.equivariance_training_matrix(config)
    assert len(matrix) == 5
    assert all(row.position_mode == "cell_equivariant" for row in matrix)

    def row(topology: str, label: str, seed: int, dual: float) -> dict[str, object]:
        return {
            "model_name": "graphgps",
            "position_mode": "cell_equivariant",
            "topology_mode": topology,
            "label_mode": label,
            "seed": seed,
            "best_validation_auc": 0.8,
            "train_auc": 0.8,
            "unseen_sbox_auc": 0.80 if label == "true" else 0.5,
            "unseen_player_auc": 0.76 if label == "true" else 0.5,
            "dual_unseen_auc": dual,
            "training_performed": True,
        }

    rows = [
        row("true", "true", 0, 0.82),
        row("true", "true", 1, 0.81),
        row("shuffled", "true", 0, 0.74),
        row("shuffled", "true", 1, 0.73),
        row("true", "shuffled", 0, 0.50),
    ]
    readiness = {
        "source": True,
    }
    gate = equivariance.adjudicate_cell_equivariance(
        config, readiness, 1e-7, rows
    )
    assert gate["decision"] == (
        "innovation2_small_spn_cell_equivariance_repair_confirmed"
    )

    output = tmp_path / "curves.svg"
    render_cell_equivariance_svg({"rows": rows, "gate": gate}, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E33-R" in svg
    assert "cell重标号" in svg
    assert "不是PRESENT/GIFT/SKINNY真实密码结果" in svg
