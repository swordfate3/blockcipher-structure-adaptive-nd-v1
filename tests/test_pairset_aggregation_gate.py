from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.planning.pairset_aggregation_gate import (
    gate_pairset_aggregation_control,
)


def test_pairset_aggregation_gate_supports_learned_pairset(tmp_path) -> None:
    learned = write_learned_results(
        tmp_path / "learned.jsonl",
        pairset_auc=0.805,
        anchor_auc=0.800,
    )
    frozen = write_frozen_summary(tmp_path / "frozen.json", auc=0.801)

    report = gate_pairset_aggregation_control(
        learned,
        frozen,
        expected_learned_rows=2,
        margin=0.001,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "support_learned_pairset_consistency"
    assert report["action"] == "repeat_262k_seed1_before_1m_pairset_scale"
    assert abs(report["margin_vs_frozen_auc"] - 0.004) < 1e-12
    assert abs(report["margin_vs_anchor_auc"] - 0.005) < 1e-12


def test_pairset_aggregation_gate_stops_tied_pairset(tmp_path) -> None:
    learned = write_learned_results(
        tmp_path / "learned.jsonl",
        pairset_auc=0.8005,
        anchor_auc=0.801,
    )
    frozen = write_frozen_summary(tmp_path / "frozen.json", auc=0.801)

    report = gate_pairset_aggregation_control(learned, frozen, margin=0.001)

    assert report["status"] == "pass"
    assert report["decision"] == "stop_pairset_consistency_route"
    assert report["action"] == "treat_as_aggregation_or_diagnostic_context"


def test_pairset_aggregation_gate_fails_without_anchor(tmp_path) -> None:
    learned = tmp_path / "learned.jsonl"
    learned.write_text(
        json.dumps(
            {
                "model": "present_nibble_invp_pair_consistency_spn_only",
                "metrics": {"auc": 0.805},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frozen = write_frozen_summary(tmp_path / "frozen.json", auc=0.801)

    report = gate_pairset_aggregation_control(learned, frozen)

    assert report["status"] == "fail"
    assert "missing_anchor_model=present_nibble_invp_only_spn_only and anchor_auc not provided" in report["errors"]


def write_learned_results(path: Path, *, pairset_auc: float, anchor_auc: float) -> Path:
    rows = [
        {
            "model": "present_nibble_invp_only_spn_only",
            "metrics": {
                "auc": anchor_auc,
                "accuracy": 0.72,
                "calibrated_accuracy": 0.721,
            },
        },
        {
            "model": "present_nibble_invp_pair_consistency_spn_only",
            "metrics": {
                "auc": pairset_auc,
                "accuracy": 0.723,
                "calibrated_accuracy": 0.724,
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def write_frozen_summary(path: Path, *, auc: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "claim_scope": "frozen single-pair score aggregation control; not a learned pair-set model",
                "metrics": {
                    "auc": auc,
                    "accuracy": 0.721,
                    "calibrated_accuracy": 0.722,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
