from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.cli.plot_innovation2_present_r9_identity_topology_residual_attribution import (
    render_identity_topology_residual,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_neural_ranking import PRESENT_P
from blockcipher_nd.tasks.innovation2.present_r9_identity_topology_residual_attribution import (
    E99_ANCHORS,
    E99_DECISION,
    E99_GATE_SHA256,
    MODEL_NAMES,
    WRONG_P,
    IdentityTopologyResidual,
    IdentityTopologyResidualConfig,
    _cycle_histogram,
    adjudicate_identity_topology_residual,
)


def test_wrong_p_is_distinct_cycle_matched_bijection() -> None:
    assert sorted(PRESENT_P) == list(range(64))
    assert sorted(WRONG_P) == list(range(64))
    assert PRESENT_P != WRONG_P
    assert _cycle_histogram(PRESENT_P) == _cycle_histogram(WRONG_P)


def test_true_and_wrong_residuals_have_paired_parameters_and_zero_scale_outputs() -> None:
    torch.manual_seed(17)
    true_model = IdentityTopologyResidual(PRESENT_P).eval()
    torch.manual_seed(17)
    wrong_model = IdentityTopologyResidual(WRONG_P).eval()
    true_parameters = dict(true_model.named_parameters())
    wrong_parameters = dict(wrong_model.named_parameters())
    assert true_parameters.keys() == wrong_parameters.keys()
    assert all(
        torch.equal(true_parameters[name], wrong_parameters[name])
        for name in true_parameters
    )
    coordinates = torch.randint(0, 2, (2, 4, 3, 128)).float()
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    with torch.no_grad():
        true_scores = true_model(coordinates, mask)
        wrong_scores = wrong_model(coordinates, mask)
    assert torch.equal(true_scores, wrong_scores)


def test_identity_topology_residual_is_coordinate_set_invariant() -> None:
    model = IdentityTopologyResidual(PRESENT_P).eval()
    coordinates = torch.randint(0, 2, (2, 5, 4, 128)).float()
    mask = torch.tensor([[[True, True, True, False]] * 5] * 2)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        direct = model(coordinates, mask)
        permuted = model(coordinates[:, :, permutation], mask[:, :, permutation])
    assert torch.allclose(direct, permuted, atol=1e-6)


def _fold_audit() -> dict[str, object]:
    return {
        "metrics": {
            "independent_relations": 468,
            "groups": 6,
            "minimum_group_positives": 78,
            "maximum_group_positives": 78,
            "maximum_train_test_relation_overlap": 0,
            "candidate_positive_support_overlap": 0,
            "candidate_known_positive_overlap": 0,
            "minimum_train_unlabeled": 55,
            "minimum_test_unlabeled": 51,
        }
    }


def _rows(*, true_recall_seed0: float = 0.94) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (0, 1):
        expected = E99_ANCHORS[seed]
        values = {
            "coordinate_anchor": (
                expected["recall_at_1"], expected["recall_at_5"], expected["mrr"], 3937
            ),
            "identity_true_p_residual": (
                true_recall_seed0 if seed == 0 else 0.93,
                expected["recall_at_5"],
                0.96,
                6001,
            ),
            "identity_wrong_p_residual": (0.92 if seed == 0 else 0.91, 0.994, 0.95, 6001),
        }
        for model, (recall1, recall5, mrr, parameters) in values.items():
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "parameter_count": parameters,
                    "recall_at_1": recall1,
                    "recall_at_5": recall5,
                    "mean_reciprocal_rank": mrr,
                    "minimum_fold_recall_at_5": 0.97,
                }
            )
    return rows


def test_e100_pass_requires_true_p_to_beat_anchor_and_wrong_p_both_seeds() -> None:
    gate = adjudicate_identity_topology_residual(
        IdentityTopologyResidualConfig(),
        fold_audit=_fold_audit(),
        training={"aggregate_rows": _rows()},
        e99_gate={"status": "hold", "decision": E99_DECISION},
        e99_gate_hash=E99_GATE_SHA256,
    )
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r9_identity_true_p_residual_attributed"
    assert gate["next_action"]["independent_confirmation_design_open"] is True
    assert gate["next_action"]["remote_scale"] is False


def test_e100_holds_when_identity_anchor_is_not_improved() -> None:
    gate = adjudicate_identity_topology_residual(
        IdentityTopologyResidualConfig(),
        fold_audit=_fold_audit(),
        training={"aggregate_rows": _rows(true_recall_seed0=0.90)},
        e99_gate={"status": "hold", "decision": E99_DECISION},
        e99_gate_hash=E99_GATE_SHA256,
    )
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_r9_coordinate_identity_anchor_remains_best"
    assert gate["next_action"]["remote_scale"] is False


def test_e100_plot_explains_true_wrong_p_control_and_scope(tmp_path: Path) -> None:
    aggregates = _rows()
    folds = [
        {
            "model": model,
            "seed": seed,
            "fold": fold,
            "recall_at_1": 0.90 + 0.01 * (model == "identity_true_p_residual"),
        }
        for model in MODEL_NAMES
        for seed in (0, 1)
        for fold in range(6)
    ]
    summary = {
        "aggregate_metrics": aggregates,
        "fold_metrics": folds,
        "gate": {
            "decision": "innovation2_present_r9_identity_true_p_residual_attributed"
        },
    }
    output = tmp_path / "curves.svg"
    render_identity_topology_residual(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "坐标身份保持拓扑残差归因" in svg
    assert "身份+真实P残差" in svg
    assert "身份+错误P残差" in svg
    assert "不是密码学负例" in svg
    assert "真实P残差稳定超过" in svg
