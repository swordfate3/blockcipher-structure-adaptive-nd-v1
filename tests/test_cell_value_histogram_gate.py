from __future__ import annotations

import json

from blockcipher_nd.planning.cell_value_histogram_gate import gate_cell_value_histogram_result


def _row(seed: int, model: str, auc: float) -> dict[str, object]:
    return {
        "seed": seed,
        "model": model,
        "metrics": {
            "auc": auc,
            "accuracy": 0.5,
            "loss": 0.25,
        },
    }


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cell_value_histogram_gate_supports_two_seed_positive(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(
        results,
        [
            _row(0, "present_pairset_global_stats", 0.70),
            _row(0, "present_pairset_histogram_hybrid", 0.73),
            _row(1, "present_pairset_global_stats", 0.71),
            _row(1, "present_pairset_histogram_hybrid", 0.74),
        ],
    )

    payload = gate_cell_value_histogram_result(results)

    assert payload["status"] == "pass"
    assert payload["decision"] == "support_cell_value_histogram_local_weak_positive"
    assert payload["mean_candidate_margin_vs_baseline_auc"] == 0.030000000000000027
    assert payload["min_candidate_auc"] == 0.73
    assert payload["claim_scope"].startswith("PRESENT r8 cell-value histogram local diagnostic only")


def test_cell_value_histogram_gate_holds_when_candidate_loses_seed(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(
        results,
        [
            _row(0, "present_pairset_global_stats", 0.70),
            _row(0, "present_pairset_histogram_hybrid", 0.72),
            _row(1, "present_pairset_global_stats", 0.71),
            _row(1, "present_pairset_histogram_hybrid", 0.70),
        ],
    )

    payload = gate_cell_value_histogram_result(results)

    assert payload["status"] == "pass"
    assert payload["decision"] == "hold_cell_value_histogram_local_screen"
    assert payload["action"] == "do_not_promote_to_diverse_expert_pool"
    assert payload["per_seed"][1]["candidate_clears_baseline"] is False
