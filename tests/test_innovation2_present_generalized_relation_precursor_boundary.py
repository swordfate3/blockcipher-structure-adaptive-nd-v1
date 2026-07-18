from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_generalized_relation_precursor_boundary import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import ATM_COMMIT
from blockcipher_nd.tasks.innovation2.present_generalized_relation_precursor_boundary import (
    PrecursorBoundaryConfig,
    audit_relation_costs,
    canonical_relation,
    evaluate_precursor_boundary,
    precursor_plaintext_count,
    wrong_monomial_plaintext_count,
)


def _relations() -> tuple[tuple[tuple[int, int], ...], ...]:
    return (
        canonical_relation([(((1 << 64) - 1) ^ 1, 1)]),
        canonical_relation(
            [
                (((1 << 64) - 1) ^ 3, 2),
                (((1 << 64) - 1) ^ 7, 4),
            ]
        ),
    )


def test_precursor_and_wrong_monomial_support_sizes_point_opposite_directions() -> None:
    input_exponent = ((1 << 64) - 1) ^ 1

    assert input_exponent.bit_count() == 63
    assert precursor_plaintext_count(input_exponent) == 1 << 63
    assert wrong_monomial_plaintext_count(input_exponent) == 2


def test_relation_cost_audit_sums_precursor_sets_per_coordinate() -> None:
    audit = audit_relation_costs(_relations())

    assert audit["metrics"]["relations"] == 2
    assert audit["metrics"]["input_weight_histogram"] == {"61": 1, "62": 1, "63": 1}
    assert audit["rows"][0]["precursor_plaintexts_per_key"] == 1 << 63
    assert audit["rows"][0]["wrong_monomial_plaintexts_per_key"] == 2
    assert audit["rows"][1]["precursor_plaintexts_per_key"] == (1 << 62) + (1 << 61)


def test_gate_holds_before_scalar_enumeration_when_minimum_cost_exceeds_cap() -> None:
    relations = _relations()
    cost_audit = audit_relation_costs(relations)
    config = PrecursorBoundaryConfig(
        run_id="e57_smoke",
        mode="smoke",
        expected_relations=2,
        maximum_scalar_plaintexts=1 << 24,
    )

    result = evaluate_precursor_boundary(
        config,
        actual_commit=ATM_COMMIT,
        relations=relations,
        cost_audit=cost_audit,
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_present_r9_generalized_relation_scalar_witness_infeasible"
    )
    assert result["gate"]["semantic_checks"]["wrong_basis_diagnostic_rejected"]
    assert result["gate"]["next_action"]["training"] is False
    assert result["gate"]["next_action"]["remote_scale"] is False


def test_plot_explains_precursor_cost_without_claiming_relation_values(tmp_path: Path) -> None:
    relations = _relations()
    cost_audit = audit_relation_costs(relations)
    result = evaluate_precursor_boundary(
        PrecursorBoundaryConfig(
            run_id="e57_smoke",
            mode="smoke",
            expected_relations=2,
            maximum_scalar_plaintexts=1 << 24,
        ),
        actual_commit=ATM_COMMIT,
        relations=relations,
        cost_audit=cost_audit,
    )
    summary = {"metrics": result["metrics"], "gate": result["gate"]}
    summary["metrics"]["maximum_precursor_plaintexts_per_relation_key"] = 1 << 65
    source = tmp_path / "summary.json"
    output = tmp_path / "curves.svg"
    source.write_text(json.dumps(summary), encoding="utf-8")

    assert plot_main(["--summary", str(source), "--output", str(output)]) == 0
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E57" in svg
    assert "不是relation常数" in svg
    assert "2^wt(u)" in svg
