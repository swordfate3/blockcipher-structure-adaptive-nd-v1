from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.cli.plot_innovation2_present_r9_atm_support_component_pu_neural_ranking import (
    render_pu_neural_ranking,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_neural_ranking import (
    E98C_ANCHOR_MRR,
    E98C_ANCHOR_RECALL_AT_5,
    E98C_DECISION,
    E98C_GATE_SHA256,
    MODEL_NAMES,
    PRESENT_P,
    PoolTensors,
    PuNeuralRankingConfig,
    _training_targets,
    adjudicate_neural_ranking,
    make_model,
)


def test_present_permutation_is_bijective_and_matches_fixed_points() -> None:
    assert sorted(PRESENT_P) == list(range(64))
    assert PRESENT_P[0] == 0
    assert PRESENT_P[1] == 16
    assert PRESENT_P[63] == 63


def _model_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    coordinates = torch.randint(0, 2, (2, 5, 4, 128), generator=generator).float()
    mask = torch.tensor(
        [
            [[True, True, False, False]] * 5,
            [[True, True, True, False]] * 5,
        ]
    ).reshape(2, 5, 4)
    return coordinates, mask


def test_all_e99_models_score_pools_and_set_models_ignore_coordinate_order() -> None:
    coordinates, mask = _model_inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    for model_name in ("summary_mlp", "coordinate_deepsets", "present_topology_set"):
        model = make_model(model_name).eval()
        with torch.no_grad():
            scores = model(coordinates, mask)
            permuted = model(coordinates[:, :, permutation], mask[:, :, permutation])
        assert scores.shape == (2, 5)
        assert torch.allclose(scores, permuted, atol=1e-6)


def test_label_shuffle_targets_are_deterministic_and_never_select_positive() -> None:
    tensors = PoolTensors(
        coordinates=torch.zeros(2, 4, 1, 128),
        coordinate_mask=torch.ones(2, 4, 1, dtype=torch.bool),
        item_mask=torch.ones(2, 4, dtype=torch.bool),
        relation_ids=(("p0", "a", "b", "c"), ("p1", "d", "e", "f")),
    )
    first = _training_targets(tensors, seed=0, fold=2, shuffled=True)
    second = _training_targets(tensors, seed=0, fold=2, shuffled=True)
    assert torch.equal(first, second)
    assert torch.all(first > 0)
    assert torch.all(first < 4)
    assert torch.equal(
        _training_targets(tensors, seed=0, fold=2, shuffled=False),
        torch.zeros(2, dtype=torch.long),
    )


def _aggregate_rows(*, topology_recall: float = 0.23) -> list[dict[str, float | int | str]]:
    rows = []
    for seed in (0, 1):
        values = {
            "absolute_position": (E98C_ANCHOR_RECALL_AT_5, E98C_ANCHOR_MRR, 0.09),
            "summary_mlp": (0.17, 0.14, 0.11),
            "coordinate_deepsets": (0.19, 0.15, 0.12),
            "present_topology_set": (topology_recall, 0.18, 0.14),
            "present_topology_set_label_shuffle": (0.09, 0.08, 0.06),
        }
        for model, (recall, mrr, worst) in values.items():
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "recall_at_5": recall,
                    "mean_reciprocal_rank": mrr,
                    "minimum_fold_recall_at_5": worst,
                }
            )
    return rows


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


def test_e99_pass_requires_topology_to_beat_all_controls_on_both_seeds() -> None:
    gate = adjudicate_neural_ranking(
        PuNeuralRankingConfig(),
        fold_audit=_fold_audit(),
        training={"aggregate_rows": _aggregate_rows()},
        e98c_gate={"status": "pass", "decision": E98C_DECISION},
        e98c_gate_hash=E98C_GATE_SHA256,
    )
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r9_pu_topology_neural_signal_confirmed"
    assert gate["next_action"]["remote_design_open"] is True
    assert gate["next_action"]["remote_scale"] is False


def test_e99_holds_when_topology_margin_over_coordinate_is_missing() -> None:
    rows = _aggregate_rows(topology_recall=0.195)
    gate = adjudicate_neural_ranking(
        PuNeuralRankingConfig(),
        fold_audit=_fold_audit(),
        training={"aggregate_rows": rows},
        e98c_gate={"status": "pass", "decision": E98C_DECISION},
        e98c_gate_hash=E98C_GATE_SHA256,
    )
    assert gate["status"] == "hold"
    assert gate["next_action"]["remote_scale"] is False


def test_e99_plot_uses_chinese_model_names_and_states_pu_scope(tmp_path: Path) -> None:
    aggregates = []
    folds = []
    for row in _aggregate_rows():
        aggregates.append(
            {
                **row,
                "recall_at_1": 0.05,
                "top5_enrichment": 1.5,
                "ranking_pools": 468,
            }
        )
    for model in MODEL_NAMES:
        for seed in (0, 1):
            for fold in range(6):
                folds.append(
                    {
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "recall_at_5": 0.20 if model == "present_topology_set" else 0.08,
                    }
                )
    summary = {
        "aggregate_metrics": aggregates,
        "fold_metrics": folds,
        "gate": {
            "decision": "innovation2_present_r9_pu_topology_neural_signal_confirmed"
        },
    }
    output = tmp_path / "curves.svg"
    render_pu_neural_ranking(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "九轮已知正关系/未标注候选神经排序" in svg
    assert "P层拓扑网" in svg
    assert "拓扑网标签打乱" in svg
    assert "不是二分类准确率" in svg
    assert "只开放远程方案设计" in svg
