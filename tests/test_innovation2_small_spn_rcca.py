from __future__ import annotations

import numpy as np
import torch

from blockcipher_nd.models.structure.spn.small_spn_relation_cross_attention import (
    SmallSpnRelationModelSpec,
    SmallSpnRelationPredictor,
)
from blockcipher_nd.tasks.innovation2.small_spn_exact_labels import (
    make_output_masks,
    make_sboxes,
    make_structures,
)
from blockcipher_nd.tasks.innovation2.small_spn_expanded_topology_labels import (
    make_player_family,
)
from blockcipher_nd.tasks.innovation2.small_spn_rcca_training import (
    RelationTrainingConfig,
    adjudicate_relation_training,
    readiness_matrix,
)
from blockcipher_nd.cli.plot_innovation2_small_spn_rcca import (
    render_small_spn_rcca,
)


def _arrays() -> dict[str, np.ndarray]:
    structures = make_structures()
    masks = make_output_masks()
    return {
        "sboxes": np.repeat(
            np.asarray(make_sboxes(), dtype=np.uint8), 16, axis=0
        ),
        "players": np.tile(
            np.asarray(make_player_family(16), dtype=np.int64), (4, 1)
        ),
        "active": np.asarray(
            [
                [int(bit in structure.active_bits) for bit in range(16)]
                for structure in structures
            ],
            dtype=np.float32,
        ),
        "masks": np.asarray(
            [[int(mask & (1 << bit) != 0) for bit in range(16)] for mask in masks],
            dtype=np.float32,
        ),
        "pairs": np.asarray([[0, 65], [65, 0], [130, 195]], dtype=np.int64),
        "rounds": np.asarray([1, 1, 2], dtype=np.int64),
    }


def _model(
    model_name: str,
    arrays: dict[str, np.ndarray],
    *,
    topology_mode: str = "true",
) -> SmallSpnRelationPredictor:
    return SmallSpnRelationPredictor(
        SmallSpnRelationModelSpec(
            model_name=model_name,
            topology_mode=topology_mode,
            hidden_dim=32,
            layers=2,
            heads=4,
            dropout=0.0,
        ),
        sboxes=arrays["sboxes"],
        players=arrays["players"],
        structure_active_bits=arrays["active"],
        output_mask_bits=arrays["masks"],
        relation_pairs=arrays["pairs"],
        relation_round_indices=arrays["rounds"],
    )


def _copy_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target_parameters = dict(target.named_parameters())
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            target_parameters[name].copy_(parameter)


def test_relation_models_are_pair_order_invariant_and_finite() -> None:
    arrays = _arrays()
    variants = torch.tensor([0, 5, 15, 48], dtype=torch.long)
    first = torch.zeros(4, dtype=torch.long)
    swapped = torch.ones(4, dtype=torch.long)

    for model_name in ("deepsets", "rcca"):
        torch.manual_seed(63001)
        model = _model(model_name, arrays).eval()
        with torch.no_grad():
            original = model(variants, first)
            reversed_order = model(variants, swapped)
        assert torch.isfinite(original).all()
        assert torch.max(torch.abs(original - reversed_order)).item() <= 1e-6


def test_rcca_is_cell_relabel_invariant() -> None:
    arrays = _arrays()
    cell_permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    node_permutation = np.asarray(
        [4 * cell_permutation[node // 4] + node % 4 for node in range(16)],
        dtype=np.int64,
    )
    inverse = np.argsort(node_permutation)
    relabeled = dict(arrays)
    relabeled["players"] = node_permutation[arrays["players"][:, inverse]]
    relabeled["active"] = arrays["active"][:, inverse]
    relabeled["masks"] = arrays["masks"][:, inverse]

    torch.manual_seed(63002)
    original = _model("rcca", arrays).eval()
    transformed = _model("rcca", relabeled).eval()
    _copy_parameters(original, transformed)
    variants = torch.tensor([0, 5, 15, 48], dtype=torch.long)
    relations = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    with torch.no_grad():
        expected = original(variants, relations)
        actual = transformed(variants, relations)

    assert torch.max(torch.abs(expected - actual)).item() <= 1e-6


def test_rcca_true_and_corrupted_topologies_change_logits() -> None:
    arrays = _arrays()
    torch.manual_seed(63003)
    true_model = _model("rcca", arrays, topology_mode="true").eval()
    corrupted_model = _model("rcca", arrays, topology_mode="corrupted").eval()
    _copy_parameters(true_model, corrupted_model)
    variants = torch.arange(16, dtype=torch.long)
    relations = torch.arange(16, dtype=torch.long) % 3
    with torch.no_grad():
        true_logits = true_model(variants, relations)
        corrupted_logits = corrupted_model(variants, relations)

    assert torch.max(torch.abs(true_logits - corrupted_logits)).item() >= 1e-5


def test_relation_models_stay_within_readiness_parameter_budget() -> None:
    arrays = _arrays()
    models = [_model(name, arrays) for name in ("deepsets", "rcca")]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]

    assert max(counts) <= 300_000
    assert max(counts) / min(counts) <= 1.35


def _gate_data() -> dict[str, object]:
    return {
        "e62_gate": {
            "status": "pass",
            "decision": "innovation2_small_spn_multicoordinate_relation_training_ready",
        },
        "e37_gate": {"status": "pass"},
        "labels": np.zeros((64, 2048), dtype=np.bool_),
        "variant_splits": {
            "train": np.arange(36),
            "unseen_sbox": np.arange(36, 48),
            "unseen_player": np.arange(48, 60),
            "dual_unseen": np.arange(60, 64),
        },
        "fit_relations": np.arange(1638),
        "validation_relations": np.arange(1638, 2048),
    }


def test_readiness_gate_requires_all_four_rows_and_contracts() -> None:
    rows = [
        {
            "row_id": spec.row_id,
            "model_name": spec.model_name,
            "topology_mode": spec.topology_mode,
            "label_mode": spec.label_mode,
            "seed": spec.seed,
            "best_validation_auc": 0.5,
            "train_auc": 0.5,
            "unseen_sbox_auc": 0.5,
            "unseen_player_auc": 0.5,
            "dual_unseen_auc": 0.5,
            "training_performed": True,
        }
        for spec in readiness_matrix()
    ]
    gate = adjudicate_relation_training(
        RelationTrainingConfig(run_id="e63_gate_test"),
        data=_gate_data(),
        contract={
            "relation_swap_max_logit_error": 0.0,
            "cell_relabel_max_logit_error": 0.0,
            "true_corrupted_max_logit_delta": 0.01,
            "parameter_counts": {"deepsets": 100000, "rcca": 120000},
            "parameter_ratio": 1.2,
        },
        rows=rows,
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_small_spn_rcca_readiness_passed"
    assert gate["next_action"]["remote_scale"] is False


def test_readiness_gate_rejects_relation_order_leakage() -> None:
    rows = [
        {
            "best_validation_auc": 0.5,
            "train_auc": 0.5,
            "unseen_sbox_auc": 0.5,
            "unseen_player_auc": 0.5,
            "dual_unseen_auc": 0.5,
            "training_performed": True,
        }
        for _ in readiness_matrix()
    ]
    gate = adjudicate_relation_training(
        RelationTrainingConfig(run_id="e63_invalid_test"),
        data=_gate_data(),
        contract={
            "relation_swap_max_logit_error": 1e-3,
            "cell_relabel_max_logit_error": 0.0,
            "true_corrupted_max_logit_delta": 0.01,
            "parameter_counts": {"deepsets": 100000, "rcca": 120000},
            "parameter_ratio": 1.2,
        },
        rows=rows,
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation2_small_spn_rcca_protocol_invalid"


def test_rcca_plot_marks_smoke_auc_as_readiness_only(tmp_path) -> None:
    rows = [
        {
            "model_name": spec.model_name,
            "topology_mode": spec.topology_mode,
            "label_mode": spec.label_mode,
            "seed": spec.seed,
            "unseen_sbox_auc": 0.7,
            "unseen_player_auc": 0.65,
            "dual_unseen_auc": 0.6,
        }
        for spec in readiness_matrix()
    ]
    summary = {
        "metadata": {
            "mode": "smoke",
            "hidden_dim": 32,
            "epochs": 8,
            "fit_relations": 1638,
            "validation_relations": 410,
        },
        "contract": {
            "parameter_counts": {"deepsets": 70465, "rcca": 79073},
            "relation_swap_max_logit_error": 2.98e-8,
            "cell_relabel_max_logit_error": 7.45e-8,
            "true_corrupted_max_logit_delta": 9.19e-4,
            "parameter_ratio": 1.122,
        },
        "rows": rows,
        "gate": {
            "status": "pass",
            "thresholds": {"dual_marginal_anchor": 0.685895},
        },
    }
    output = tmp_path / "curves.svg"

    render_small_spn_rcca(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E63" in svg
    assert "当前8-epoch AUC只验证流程" in svg
    assert "不是PRESENT/GIFT结果" in svg
