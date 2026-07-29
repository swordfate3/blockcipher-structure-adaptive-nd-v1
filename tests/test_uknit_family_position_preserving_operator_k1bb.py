from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_position_preserving_operator_k1bb import (
    render_k1bb_svg,
)
from blockcipher_nd.models.structure.spn.position_preserving_operator import (
    OPERATOR_TOKEN_DIM,
    PositionPreservingOperatorSpec,
    SharedPositionPreservingOperatorEncoder,
    transported_position_ids,
)
from blockcipher_nd.tasks.innovation1.uknit_family_position_preserving_operator_k1bb import (
    adjudicate_readiness,
    load_and_validate_config,
    load_authority,
)


def test_k1bb_actual_collision_edges_are_visible_with_fixed_geometry() -> None:
    config = load_and_validate_config()
    (
        _readiness,
        _dataset_rows,
        _datasets,
        structures,
        _summaries,
        _checkpoints,
        controls,
        checks,
    ) = load_authority(config)
    torch.manual_seed(40)
    encoder = SharedPositionPreservingOperatorEncoder(
        PositionPreservingOperatorSpec()
    )

    assert all(checks.values()), checks
    assert set(structures) == {"uknit64", "midori64", "dialga128"}
    deltas = []
    for cipher, structure in structures.items():
        corrupted = controls["corrupted_structures"][cipher]
        correct_tokens = encoder.operator_tokens(structure)
        corrupted_tokens = encoder.operator_tokens(corrupted)
        correct_embedding = encoder.structure_embedding(structure)
        corrupted_embedding = encoder.structure_embedding(corrupted)
        deltas.append(
            float(
                torch.max(torch.abs(correct_embedding - corrupted_embedding)).detach()
            )
        )
        assert correct_tokens.values.shape[1] == OPERATOR_TOKEN_DIM
        assert correct_tokens.values.shape == corrupted_tokens.values.shape
    assert min(deltas) >= 1e-4


def test_k1bb_transported_cell_identity_is_relabel_equivariant() -> None:
    config = load_and_validate_config()
    structures = load_authority(config)[3]
    structure = structures["uknit64"]
    permutation = torch.roll(torch.arange(structure.cells), shifts=-1)
    relabeled, _bit_permutation = structure.relabel_cells(permutation.tolist())
    torch.manual_seed(41)
    encoder = SharedPositionPreservingOperatorEncoder(
        PositionPreservingOperatorSpec()
    )

    correct = encoder.structure_embedding(structure)
    transported = encoder.structure_embedding(
        relabeled,
        cell_position_ids=transported_position_ids(permutation),
    )

    assert float(torch.max(torch.abs(correct - transported)).detach()) <= 1e-6


def test_k1bb_gate_separates_pass_hold_and_invalid() -> None:
    config = load_and_validate_config()
    rows = _synthetic_rows()
    geometry = _synthetic_geometry()
    operators = _synthetic_operator_rows()
    checkpoints = {
        0: {"best_epoch": 9},
        1: {"best_epoch": 9},
    }

    passed = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        rows=rows,
        geometry_rows=geometry,
        operator_rows=operators,
        checkpoints=checkpoints,
    )
    assert passed["status"] == "pass"
    assert "readiness_authorized" in passed["decision"]

    held_rows = deepcopy(rows)
    held_rows[0]["correct_vs_corrupted_logit_delta"] = 0.0
    held = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        rows=held_rows,
        geometry_rows=geometry,
        operator_rows=operators,
        checkpoints=checkpoints,
    )
    assert held["status"] == "hold"
    assert "same_summary_enabled_logits_respond" in held["failed_panel_checks"]

    invalid_rows = deepcopy(rows)
    invalid_rows[0]["state_immutable"] = False
    invalid = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        rows=invalid_rows,
        geometry_rows=geometry,
        operator_rows=operators,
        checkpoints=checkpoints,
    )
    assert invalid["status"] == "invalid"
    assert "zero_updates_and_immutable_states" in invalid["failed_protocol_checks"]


def test_k1bb_plot_writes_clear_chinese_svg(tmp_path: Path) -> None:
    gate = adjudicate_readiness(
        config=load_and_validate_config(),
        source_checks={"source": True},
        rows=_synthetic_rows(),
        geometry_rows=_synthetic_geometry(),
        operator_rows=_synthetic_operator_rows(),
        checkpoints={0: {"best_epoch": 9}, 1: {"best_epoch": 9}},
    )
    output = tmp_path / "curves.svg"

    report = render_k1bb_svg(
        gate,
        _synthetic_rows(),
        _synthetic_operator_rows(),
        output,
    )

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert report["result_panels"] == 12
    assert "真实 GF(2) 连线已能进入样本边调制" in text
    assert "还没有证明训练后 AUC 会提高" in text
    assert "不是准确率结果" in text


def _synthetic_rows() -> list[dict[str, object]]:
    rows = []
    seeds = {
        0: {"uknit64": 3, "midori64": 6, "dialga128": 0},
        1: {"uknit64": 4, "midori64": 7, "dialga128": 1},
    }
    for replica in (0, 1):
        for cipher in ("uknit64", "midori64", "dialga128"):
            for split in ("same_key_fresh", "cross_key_validation"):
                rows.append(
                    {
                        "replica": replica,
                        "cipher_key": cipher,
                        "seed": seeds[replica][cipher],
                        "split": split,
                        "correct_vs_corrupted_operator_embedding_delta": 0.2,
                        "correct_vs_cross_cipher_operator_embedding_delta": 0.3,
                        "correct_vs_corrupted_edge_modulation_delta": 0.02,
                        "correct_vs_corrupted_logit_delta": 0.001,
                        "disabled_k1az_logit_replay_delta": 0.0,
                        "joint_relabel_embedding_delta": 2e-7,
                        "joint_relabel_modulation_delta": 2e-6,
                        "joint_relabel_logit_delta": 2e-6,
                        "runtime_structure_held_correct": True,
                        "operator_control_only": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "state_immutable": True,
                    }
                )
    return rows


def _synthetic_geometry() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "operator_token_dim": OPERATOR_TOKEN_DIM,
            "trainable_parameter_count": 100,
            "trainable_parameter_geometry": {"weight": [10, 10]},
            "uses_actual_source_target_connectivity": True,
            "operator_interaction_before_pooling": True,
            "uses_cipher_identity": False,
            "uses_per_cipher_parameters": False,
            "uses_invariant_linear_summary": False,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
    ]


def _synthetic_operator_rows() -> list[dict[str, object]]:
    return [
        {
            "replica": replica,
            "cipher_key": cipher,
            "correct_vs_corrupted_embedding_max_abs_delta": 0.2,
            "correct_vs_cross_cipher_embedding_max_abs_delta": 0.3,
        }
        for replica in (0, 1)
        for cipher in ("uknit64", "midori64", "dialga128")
    ]
