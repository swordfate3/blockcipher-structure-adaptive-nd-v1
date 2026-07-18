from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_open_3sdp_glpk_gate import (
    main as audit_main,
)
from blockcipher_nd.cli.plot_innovation2_present_open_3sdp_glpk_gate import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_open_3sdp_glpk_gate import (
    GlpkEnumerationGateConfig,
    evaluate_glpk_enumeration_gate,
    exact_counts_by_output_exponent,
)


def _completed_record(output_exponent: int) -> dict[str, object]:
    counts = exact_counts_by_output_exponent()[output_exponent]
    return {
        "output_exponent": output_exponent,
        "status": "completed",
        "backend": "GLPKBackend",
        "seconds": 0.1,
        "solutions": sum(counts.values()),
        "expected_solutions": sum(counts.values()),
        "complete": True,
        "counts": {str(key): value for key, value in counts.items()},
        "error": None,
    }


def test_exact_representative_solution_counts_are_frozen() -> None:
    counts = exact_counts_by_output_exponent()

    assert [sum(counts[exponent].values()) for exponent in (1, 3, 7, 15)] == [
        4,
        28,
        224,
        1792,
    ]


def test_gate_holds_when_heaviest_glpk_query_times_out() -> None:
    config = GlpkEnumerationGateConfig(run_id="e53b")
    records = [_completed_record(exponent) for exponent in (1, 3, 7)]
    records.append(
        {
            "output_exponent": 15,
            "status": "timeout",
            "seconds": 10.0,
            "timeout_seconds": 10.0,
            "counts": {},
            "error": None,
        }
    )

    result = evaluate_glpk_enumeration_gate(config, records)

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_present_r5_open_3sdp_glpk_blocking_not_scalable"
    )
    assert all(result["gate"]["correctness_checks"].values())
    assert result["gate"]["scalability_checks"][
        "heaviest_query_completes_within_budget"
    ] is False
    assert result["gate"]["next_action"]["five_round_subset"] is False


def test_smoke_cli_enumerates_low_complexity_queries_with_glpk(
    tmp_path: Path,
) -> None:
    output = tmp_path / "e53b"
    exit_code = audit_main(
        [
            "--run-id",
            "e53b_smoke_test",
            "--output-root",
            str(output),
            "--mode",
            "smoke",
            "--output-exponents",
            "1",
            "3",
            "--required-complete",
            "1",
            "3",
            "--timeout-seconds",
            "2",
        ]
    )

    assert exit_code == 0
    expected = {
        "queries.jsonl",
        "results.jsonl",
        "provider_manifest.json",
        "metadata.json",
        "summary.json",
        "gate.json",
        "progress.jsonl",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_present_r5_open_3sdp_glpk_sbox_enumerator_ready"
    )
    assert gate["metrics"]["completed_solutions"] == 32
    assert gate["correctness_checks"]["required_counts_match_exact"]

    assert (
        plot_main(
            [
                "--summary",
                str(output / "summary.json"),
                "--output",
                str(output / "curves.svg"),
            ]
        )
        == 0
    )
    svg = (output / "curves.svg").read_text(encoding="utf-8")
    assert "创新2 E53-B" in svg
    assert "不是全密码provider" in svg
