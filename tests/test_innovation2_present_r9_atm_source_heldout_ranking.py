from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli.plot_innovation2_present_r9_atm_source_heldout_ranking import (
    render_source_heldout_ranking,
)
from blockcipher_nd.cli.run_innovation2_present_r9_atm_source_heldout_ranking import (
    parse_args,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_source_heldout_ranking import (
    E99_GATE_SHA256,
    E99_SUMMARY_SHA256,
    SourceHeldoutRankingConfig,
    evaluate_source_heldout,
    load_relations_json,
    sha256,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_support_component_pu_neural_ranking import (
    make_model,
)


def test_cli_separates_freeze_from_heldout_evaluation() -> None:
    freeze = parse_args(
        [
            "freeze",
            "--results-root",
            "public",
            "--atm-root",
            "atm",
            "--e99-summary",
            "summary.json",
            "--e99-gate",
            "gate.json",
            "--output-root",
            "freeze-output",
        ]
    )
    assert freeze.mode == "freeze"
    assert not hasattr(freeze, "e104_root")

    evaluate = parse_args(
        [
            "evaluate",
            "--results-root",
            "public",
            "--checkpoint-root",
            "checkpoints",
            "--e104-root",
            "heldout",
            "--output-root",
            "evaluation-output",
        ]
    )
    assert evaluate.mode == "evaluate"
    assert not hasattr(evaluate, "e99_summary")


def test_load_relations_json_accepts_unsigned_64_bit_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "relations.json"
    path.write_text(
        json.dumps({"relations": [[[(1 << 63), 7], [3, (1 << 64) - 1]]]}) + "\n",
        encoding="utf-8",
    )
    relations = load_relations_json(path)
    assert relations == {frozenset({((1 << 63), 7), (3, (1 << 64) - 1)})}


def test_load_relations_json_rejects_duplicate_or_out_of_range_data(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps({"relations": [[[1, 2], [1, 2]]]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate coordinates"):
        load_relations_json(duplicate)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"relations": [[[1 << 64, 0]]]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        load_relations_json(invalid)


def _checkpoint_manifest(root: Path) -> dict[str, object]:
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir()
    rows = []
    for seed in (0, 1):
        for fold in range(6):
            torch.manual_seed(seed * 1009 + fold * 37 + 17)
            path = checkpoint_root / f"seed{seed}_fold{fold}.pt"
            torch.save(
                {
                    "model": "coordinate_deepsets",
                    "seed": seed,
                    "fold": fold,
                    "state_dict": make_model("coordinate_deepsets").state_dict(),
                },
                path,
            )
            rows.append(
                {
                    "model": "coordinate_deepsets",
                    "seed": seed,
                    "fold": fold,
                    "path": f"checkpoints/{path.name}",
                    "sha256": sha256(path),
                    "absolute_position_target": 31.5,
                }
            )
    return {
        "status": "pass",
        "decision": "innovation2_present_r9_e99_coordinate_checkpoints_frozen",
        "heldout_source_read": False,
        "e99_summary_sha256": E99_SUMMARY_SHA256,
        "e99_gate_sha256": E99_GATE_SHA256,
        "checkpoints": rows,
    }


def _public_groups() -> dict[str, set[frozenset[tuple[int, int]]]]:
    return {
        "public": {
            frozenset({(1, 1 << relative_distance)})
            for relative_distance in range(1, 7)
        }
    }


def test_source_heldout_evaluation_is_zero_update_and_small_set_is_diagnostic(
    tmp_path: Path,
) -> None:
    manifest = _checkpoint_manifest(tmp_path)
    public_groups = _public_groups()
    heldout = {frozenset({(1 << 9, 1 << 27)})}
    result = evaluate_source_heldout(
        SourceHeldoutRankingConfig(),
        public_groups=public_groups,
        heldout_relations=heldout,
        checkpoint_manifest=manifest,
        checkpoint_root=tmp_path,
        e104_gate={"status": "pass", "decision": "innovation2_present_r9_split333_generation_passed"},
        e104_evidence_checks={"frozen_e104_evidence": True},
        device="cpu",
    )
    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_present_r9_split333_source_heldout_diagnostic_only"
    )
    assert result["audit"]["evaluation_optimizer_steps"] == 0
    assert result["audit"]["evaluation_backward_calls"] == 0
    assert result["audit"]["model_state_unchanged"] is True
    assert result["audit"]["minimum_unlabeled_per_pool"] >= 31
    deterministic = result["gate"]["metrics"]["deterministic_anchors"]
    assert {row["model"] for row in deterministic} == {
        "absolute_position_6fold_ensemble",
        "training_coordinate_frequency_6fold_ensemble",
        "training_support_overlap_6fold_ensemble",
    }
    best = result["gate"]["metrics"]["best_deterministic_anchor"]
    assert best["recall_at_5"] == max(row["recall_at_5"] for row in deterministic)
    assert best["mean_reciprocal_rank"] == max(
        row["mean_reciprocal_rank"] for row in deterministic
    )


def test_source_heldout_gate_rejects_unverified_e104_source(tmp_path: Path) -> None:
    manifest = _checkpoint_manifest(tmp_path)
    result = evaluate_source_heldout(
        SourceHeldoutRankingConfig(),
        public_groups=_public_groups(),
        heldout_relations={frozenset({(2, 8)})},
        checkpoint_manifest=manifest,
        checkpoint_root=tmp_path,
        e104_gate={"status": "hold", "decision": "resource_cap_hit"},
        e104_evidence_checks={"frozen_e104_evidence": True},
        device="cpu",
    )
    assert result["gate"]["status"] == "fail"
    assert result["gate"]["decision"].endswith("protocol_invalid")


def test_source_heldout_gate_rejects_relations_seen_in_fold_training_candidates(
    tmp_path: Path,
) -> None:
    manifest = _checkpoint_manifest(tmp_path)
    result = evaluate_source_heldout(
        SourceHeldoutRankingConfig(),
        public_groups=_public_groups(),
        heldout_relations={frozenset({(2, 4)})},
        checkpoint_manifest=manifest,
        checkpoint_root=tmp_path,
        e104_gate={
            "status": "pass",
            "decision": "innovation2_present_r9_split333_generation_passed",
        },
        e104_evidence_checks={"frozen_e104_evidence": True},
        device="cpu",
    )

    assert result["gate"]["status"] == "fail"
    assert result["gate"]["source_checks"][
        "all_evaluation_relations_absent_from_fold_training_pools"
    ] is False
    assert result["audit"]["maximum_fold_training_overlap"] > 0
    assert result["result_rows"] == []


def test_e105_plot_uses_chinese_explanations_and_zero_adaptation_scope(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "model": "absolute_position_6fold_ensemble",
            "seed": -1,
            "recall_at_1": 0.10,
            "recall_at_5": 0.20,
            "mean_reciprocal_rank": 0.18,
            "top5_enrichment": 2.0,
        },
        {
            "model": "coordinate_deepsets_6fold_ensemble",
            "seed": 0,
            "recall_at_1": 0.62,
            "recall_at_5": 0.82,
            "mean_reciprocal_rank": 0.70,
            "top5_enrichment": 8.2,
        },
        {
            "model": "coordinate_deepsets_6fold_ensemble",
            "seed": 1,
            "recall_at_1": 0.60,
            "recall_at_5": 0.80,
            "mean_reciprocal_rank": 0.68,
            "top5_enrichment": 8.0,
        },
    ]
    summary = {
        "gate": {
            "decision": "innovation2_present_r9_split333_source_heldout_signal_confirmed",
            "metrics": {
                "best_deterministic_anchor": {
                    "recall_at_5": 0.25,
                    "mean_reciprocal_rank": 0.22,
                }
            },
        },
        "result_rows": rows,
        "audit": {"heldout_relations": 48, "minimum_unlabeled_per_pool": 61},
    }
    output = tmp_path / "curves.svg"

    render_source_heldout_ranking(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "PRESENT九轮缺失(3,3,3)来源留出排序" in svg
    assert "不更新权重" in svg
    assert "训练坐标频率和支撑重合规则" in svg
    assert "相对最强确定性规则的绝对增益" in svg
    assert "不是二分类" in svg
