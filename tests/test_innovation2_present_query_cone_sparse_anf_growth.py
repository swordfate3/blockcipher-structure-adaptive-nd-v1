from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_query_cone_sparse_anf_growth import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_open_3sdp_exact_oracle import (
    build_present_exact_anf_snapshots,
    polynomial_sha256,
)
from blockcipher_nd.tasks.innovation2.present_query_cone_sparse_anf_growth import (
    FROZEN_FIXTURE_IDS,
    SparseAnfGrowthConfig,
    compute_query_output_polynomials,
    evaluate_sparse_growth_gate,
    freeze_query_manifest,
    required_state_cone,
    run_sparse_query,
)


def _smoke_config(**overrides: object) -> SparseAnfGrowthConfig:
    values: dict[str, object] = {
        "run_id": "e55_smoke",
        "mode": "smoke",
        "rounds": 3,
        "maximum_terms": 100_000,
        "maximum_seconds": 10.0,
        "maximum_memory_bytes": 2 * (1 << 30),
    }
    values.update(overrides)
    return SparseAnfGrowthConfig(**values)  # type: ignore[arg-type]


def _fixture(fixture_id: str) -> dict[str, object]:
    is_multi = "multi_mask" in fixture_id
    return {
        "fixture_id": fixture_id,
        "fixture_type": "multi_bit_mask_xor" if is_multi else "strict_unit_mask",
        "rounds": 2,
        "active_bits": [int(fixture_id[-2:]) % 4],
        "output_mask_hex": "0x0000000000020001" if is_multi else "0x0000000000000001",
        "output_bits": [0, 17] if is_multi else [0],
        "status": "positive" if "positive" in fixture_id else "negative",
    }


def _source_summary() -> dict[str, object]:
    return {
        "gate": {
            "status": "pass",
            "decision": "innovation2_present_r5_open_3sdp_exact_oracle_ready",
        },
        "provider_manifest": {"providers": [{"variables": {"total": 144}}]},
    }


def _calibration(passed: bool = True) -> dict[str, object]:
    return {
        "checks": {
            "all_selected_calibration_rows_executed": passed,
            "all_superpolies_match_e53a": passed,
            "all_unit_output_hashes_match_e53a": passed,
            "all_query_assignments_match_scalar_present": passed,
            "wrong_player_control_is_detected": passed,
            "zero_offset_control_rejected": passed,
        }
    }


def test_frozen_manifest_uses_four_positive_four_negative_and_four_multi() -> None:
    fixtures = [_fixture(fixture_id) for fixture_id in FROZEN_FIXTURE_IDS]

    manifest = freeze_query_manifest(fixtures)

    assert len(manifest) == 12
    assert [row["source_fixture_id"] for row in manifest] == list(FROZEN_FIXTURE_IDS)
    assert sum(row["query_type"] == "unit_output_bit" for row in manifest) == 8
    assert sum(row["query_type"] == "multi_bit_mask" for row in manifest) == 4
    assert all(row["rounds"] == 3 for row in manifest)


def test_query_cone_one_round_matches_full_exact_anf() -> None:
    outputs, metrics = compute_query_output_polynomials(
        rounds=1,
        output_bits=(0, 17),
        config=_smoke_config(maximum_terms=10_000),
    )
    full = build_present_exact_anf_snapshots((1,))[1]

    assert polynomial_sha256(outputs[0]) == polynomial_sha256(full[0])
    assert polynomial_sha256(outputs[17]) == polynomial_sha256(full[17])
    assert metrics["cone_widths"] == {"0": 8, "1": 2}
    assert required_state_cone(rounds=1, output_bits=(0, 17))[0] == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )


def test_wrong_player_changes_the_frozen_one_round_query() -> None:
    correct, _ = compute_query_output_polynomials(
        rounds=1,
        output_bits=(16,),
        config=_smoke_config(maximum_terms=10_000),
    )
    wrong, _ = compute_query_output_polynomials(
        rounds=1,
        output_bits=(16,),
        config=_smoke_config(maximum_terms=10_000),
        player_mode="identity",
    )

    assert polynomial_sha256(correct[16]) != polynomial_sha256(wrong[16])


def test_query_aborts_at_term_cap_without_returning_partial_label() -> None:
    query = {
        "query_id": "r3_query_00",
        "source_fixture_id": "r2_negative_00",
        "query_type": "unit_output_bit",
        "rounds": 2,
        "active_bits": [0],
        "output_bits": [20],
        "output_mask_hex": "0x0000000000100000",
    }

    row = run_sparse_query(
        query,
        config=_smoke_config(rounds=2, maximum_terms=100, maximum_seconds=10.0),
    )

    assert row["status"] == "cap_exceeded"
    assert row["cap_reason"] == "term_cap_exceeded"
    assert row["label"] == "unknown"
    assert row["superpoly_sha256"] is None


def test_gate_holds_and_closes_provider_after_any_cap_exceeded() -> None:
    rows = [
        {
            "query_id": f"r3_query_{index:02d}",
            "status": "cap_exceeded" if index == 0 else "skipped",
            "label": "unknown",
            "cap_reason": "term_cap_exceeded",
            "maximum_observed_terms": 5_000_001 if index == 0 else 0,
        }
        for index in range(12)
    ]

    result = evaluate_sparse_growth_gate(
        _smoke_config(),
        source_summary=_source_summary(),
        calibration=_calibration(),
        query_rows=rows,
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_present_r3_query_cone_sparse_anf_hard_cap_exceeded"
    )
    assert result["gate"]["next_action"]["training"] is False
    assert result["metrics"]["completed_queries"] == 0


def test_plot_writes_chinese_scope_and_hard_cap_chart(tmp_path: Path) -> None:
    rows = [
        {
            "query_id": f"r3_query_{index:02d}",
            "status": "cap_exceeded" if index == 0 else "skipped",
            "label": "unknown",
            "maximum_observed_terms": 101 if index == 0 else 0,
            "elapsed_seconds": 0.1 if index == 0 else 0.0,
        }
        for index in range(12)
    ]
    summary = {
        "metadata": {"config": {
            "maximum_terms": 100,
            "maximum_seconds": 10.0,
        }},
        "query_rows": rows,
        "calibration": {
            "checks": _calibration()["checks"],
            "completed_rows": 32,
            "expected_rows": 32,
        },
        "gate": {
            "decision": "innovation2_present_r3_query_cone_sparse_anf_hard_cap_exceeded",
            "next_action": {"action": "close the current exact full-variable sparse provider family"},
        },
    }
    source = tmp_path / "summary.json"
    output = tmp_path / "curves.svg"
    source.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(source), "--output", str(output)]) == 0
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E55" in svg
    assert "不是五轮标签" in svg
    assert "单项式门" in svg
