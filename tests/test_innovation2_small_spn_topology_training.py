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
from blockcipher_nd.cli.plot_innovation2_small_spn_round_shared_reasoner import (
    render_round_shared_reasoner_svg,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_cipher_edge_token import (
    render_cipher_edge_token_svg,
)
from blockcipher_nd.models.structure.spn.small_spn_graph_models import (
    BasisSetEncoder,
    SmallSpnModelSpec,
    SmallSpnTopologyPredictor,
)
from blockcipher_nd.models.structure.spn.small_spn_edge_token_models import (
    SmallSpnCipherEdgeTokenTransformer,
    SmallSpnEdgeTokenSpec,
)
from blockcipher_nd.models.structure.spn.small_spn_topology_controls import (
    topology_players,
)
from blockcipher_nd.tasks.innovation2 import small_spn_exact_labels as labels
from blockcipher_nd.tasks.innovation2 import small_spn_cell_equivariance as equivariance
from blockcipher_nd.tasks.innovation2 import small_spn_topology_training as training
from blockcipher_nd.tasks.innovation2 import small_spn_round_shared_reasoner as reasoner
from blockcipher_nd.tasks.innovation2 import small_spn_cipher_edge_token as edge_token


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
    processor_mode: str = "stacked",
) -> SmallSpnTopologyPredictor:
    data = _model_data()
    return SmallSpnTopologyPredictor(
        SmallSpnModelSpec(
            model_name=model_name,
            topology_mode=topology_mode,
            position_mode=position_mode,
            processor_mode=processor_mode,
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


def test_round_shared_processor_uses_one_block_and_two_to_five_steps() -> None:
    class AddOne(torch.nn.Module):
        def forward(
            self,
            hidden: torch.Tensor,
            incoming: torch.Tensor,
            outgoing: torch.Tensor,
        ) -> torch.Tensor:
            del incoming, outgoing
            return hidden + 1

    model = _model(
        "graphgps",
        position_mode="cell_equivariant",
        processor_mode="round_shared",
    )
    assert len(model.blocks) == 1
    model.blocks[0] = AddOne()
    hidden = torch.zeros((4, 16, 32))
    identity = torch.arange(16).expand(4, -1)
    output = model._run_graph_processor(
        hidden,
        identity,
        identity,
        torch.tensor([0, 1, 2, 3]),
    )
    expected = torch.tensor([2, 3, 4, 5], dtype=output.dtype)
    assert torch.allclose(output[:, 0, 0], expected)
    assert torch.allclose(output, expected[:, None, None].expand_as(output))
    error = equivariance.measure_cell_relabeling_error(
        _model_data(), processor_mode="round_shared"
    )
    assert error <= 1e-6


def test_cipher_edge_token_contract_is_37_and_cell_relabeling_invariant() -> None:
    data = _model_data()
    model = SmallSpnCipherEdgeTokenTransformer(
        SmallSpnEdgeTokenSpec(hidden_dim=32, layers=2, heads=4, dropout=0.0),
        sboxes=data["sboxes"],
        players=data["players"],
        structure_active_bits=data["structure_active"],
        output_mask_bits=data["output_mask_bits"],
    ).eval()
    inputs = (
        torch.tensor([0, 15]),
        torch.tensor([0, 3]),
        torch.tensor([0, 13]),
        torch.tensor([0, 63]),
    )
    tokens = model.build_tokens(*inputs)
    assert tokens.shape == (2, 37, 32)
    assert model(*inputs).shape == (2,)
    assert not hasattr(model, "bit_embedding")
    assert not hasattr(model, "nibble_embedding")
    contract = edge_token.measure_cipher_edge_token_contract(data)
    assert contract["token_count"] == 37
    assert contract["cell_relabeling_max_abs_logit_error"] <= 1e-6
    assert contract["fair_control_heldout_avoids_train_players"] is True
    assert contract["all_corrupted_players_are_permutations"] is True


def test_family_preserving_topology_control_does_not_replace_heldout_player() -> None:
    players = _model_data()["players"]
    rolled = topology_players(players, "shuffled")
    corrupted = topology_players(players, "corrupted")
    for variant_index in (3, 7, 11, 15):
        assert np.array_equal(rolled[variant_index], players[variant_index - 1])
        assert not any(
            np.array_equal(corrupted[variant_index], players[train_player])
            for train_player in (0, 1, 2)
        )
        assert sorted(corrupted[variant_index].tolist()) == list(range(16))


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


def test_round_shared_matrix_gate_and_plot_are_frozen(tmp_path: Path) -> None:
    config = training.TopologyTrainingConfig(run_id="e34")
    matrix = reasoner.round_shared_training_matrix(config)
    assert len(matrix) == 5
    assert all(row.position_mode == "cell_equivariant" for row in matrix)
    assert all(row.processor_mode == "round_shared" for row in matrix)

    def row(topology: str, label: str, seed: int, dual: float) -> dict[str, object]:
        return {
            "model_name": "graphgps",
            "position_mode": "cell_equivariant",
            "processor_mode": "round_shared",
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
    gate = reasoner.adjudicate_round_shared_reasoner(
        config, {"source": True}, 1e-7, rows
    )
    assert gate["decision"] == (
        "innovation2_small_spn_round_shared_reasoner_confirmed"
    )

    output = tmp_path / "curves.svg"
    render_round_shared_reasoner_svg({"rows": rows, "gate": gate}, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E34" in svg
    assert "按实际轮数" in svg
    assert "不是PRESENT/GIFT/SKINNY真实密码结果" in svg


def test_cipher_edge_token_matrix_gate_and_plot_are_frozen(tmp_path: Path) -> None:
    config = training.TopologyTrainingConfig(run_id="e35")
    matrix = edge_token.cipher_edge_token_training_matrix(config)
    assert len(matrix) == 5
    assert all(row.model_name == "cett" for row in matrix)
    assert all(row.position_mode == "cell_equivariant" for row in matrix)
    assert all(row.processor_mode == "edge_token_transformer" for row in matrix)
    assert sum(row.topology_mode == "corrupted" for row in matrix) == 2

    def row(topology: str, label: str, seed: int, dual: float) -> dict[str, object]:
        return {
            "model_name": "cett",
            "position_mode": "cell_equivariant",
            "processor_mode": "edge_token_transformer",
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
        row("corrupted", "true", 0, 0.74),
        row("corrupted", "true", 1, 0.73),
        row("true", "shuffled", 0, 0.50),
    ]
    gate = edge_token.adjudicate_cipher_edge_token(
        config,
        {"source": True},
        {
            "token_count": 37,
            "cell_relabeling_max_abs_logit_error": 1e-7,
            "fair_control_heldout_avoids_train_players": True,
            "all_corrupted_players_are_permutations": True,
        },
        rows,
    )
    assert gate["decision"] == "innovation2_small_spn_cipher_edge_token_confirmed"

    output = tmp_path / "curves.svg"
    render_cipher_edge_token_svg({"rows": rows, "gate": gate}, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E35" in svg
    assert "37个token" in svg
    assert "不是PRESENT/GIFT/SKINNY真实密码结果" in svg

    fair_output = tmp_path / "fair-curves.svg"
    render_cipher_edge_token_svg(
        {
            "run_id": "i2_small_spn_cipher_edge_token_fair_control_seed0_seed1_20260718",
            "rows": rows,
            "gate": gate,
        },
        fair_output,
    )
    fair_svg = fair_output.read_text(encoding="utf-8")
    assert "创新2 E35b" in fair_svg
    assert "公平P-layer控制重裁决" in fair_svg
