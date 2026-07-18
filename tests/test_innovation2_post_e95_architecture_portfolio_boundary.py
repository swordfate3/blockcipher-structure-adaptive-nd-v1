from __future__ import annotations

import json
from pathlib import Path

import pytest

from blockcipher_nd.cli.plot_innovation2_post_e95_architecture_portfolio_boundary import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.post_e95_architecture_portfolio_boundary import (
    SOURCE_SPECS,
    ArchitecturePortfolioConfig,
    adjudicate_architecture_portfolio,
    build_architecture_portfolio,
    validate_portfolio_sources,
)


def _sources() -> dict[str, dict[str, object]]:
    sources = {
        spec.role: {
            "sha256": spec.sha256,
            "gate": {
                "run_id": spec.run_id,
                "status": spec.status,
                "decision": spec.decision,
            },
        }
        for spec in SOURCE_SPECS
    }
    gates = {role: source["gate"] for role, source in sources.items()}
    gates["multibit_mask"].update(
        {"protocol_checks": {"ok": True}, "nontrivial_checks": {"wide": False}}
    )
    gates["active_dimension"].update(
        {"source_checks": {"ok": True}, "label_checks": {"wide": False}}
    )
    gates["formal_method"].update(
        {
            "source_checks": {"ok": True},
            "method_checks": {"ok": True},
            "skinny_readiness_checks": {"ok": True},
            "metrics": {
                "ciphers": [
                    {"cipher": "PRESENT-80", "mean_true_minus_corrupted": 0.096944},
                    {"cipher": "GIFT-64", "mean_true_minus_corrupted": 0.132414},
                ]
            },
        }
    )
    gates["skinny_residual"].update(
        {"protocol_checks": {"ok": True}, "readiness_checks": {"gain": False}}
    )
    gates["shared_operator"].update(
        {
            "protocol_checks": {"ok": True},
            "relation_checks": {"ok": True},
            "candidate_checks": {"quality": False},
        }
    )
    gates["rectangle_untyped"].update(
        {
            "protocol_checks": {"ok": True},
            "candidate_checks": {"quality": True},
            "relation_checks": {"gain": False},
        }
    )
    gates["rectangle_row_operator"].update(
        {"protocol_checks": {"ok": True}, "readiness_checks": {"row": False}}
    )
    gates["e93_boundary"].update(
        {"source_checks": {"ok": True}, "ranking_checks": {"ok": True}}
    )
    gates["nested_labels"].update(
        {
            group: {"ok": True}
            for group in (
                "protocol_checks",
                "monotonic_checks",
                "width_checks",
                "shortcut_checks",
            )
        }
    )
    gates["nested_relation"].update(
        {
            "protocol_checks": {"ok": True},
            "quality_checks": {"absolute": False},
            "attribution_checks": {"margin": False},
            "metrics": {
                "margins": {
                    "true_minus_independent": 0.014837,
                    "true_minus_shuffled": 0.018351,
                }
            },
        }
    )
    return sources


def test_e96_source_contract_and_config_are_frozen() -> None:
    assert len(SOURCE_SPECS) == 10
    assert ArchitecturePortfolioConfig().run_id.endswith("20260719")
    with pytest.raises(ValueError, match="frozen"):
        ArchitecturePortfolioConfig(run_id="other")


def test_e96_replays_sources_and_builds_zero_budget_portfolio() -> None:
    sources = _sources()

    checks = validate_portfolio_sources(sources)
    rows = build_architecture_portfolio(sources)
    gate = adjudicate_architecture_portfolio(
        ArchitecturePortfolioConfig(), checks, rows
    )

    assert all(checks.values())
    assert len(rows) == 8
    assert sum(row["evidence_class"] == "formal_confirmed" for row in rows) == 1
    assert sum(row["evidence_class"] == "provider_missing" for row in rows) == 2
    assert not any(row["training_budget"] for row in rows)
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_architecture_portfolio_converged_no_new_training_budget"
    )


def test_e96_gate_fails_on_source_or_portfolio_mismatch() -> None:
    sources = _sources()
    checks = validate_portfolio_sources(sources)
    rows = build_architecture_portfolio(sources)

    broken_checks = {**checks, "formal_method_hash_matches": False}
    gate = adjudicate_architecture_portfolio(
        ArchitecturePortfolioConfig(), broken_checks, rows
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == "innovation2_architecture_portfolio_protocol_invalid"


def test_plot_writes_clear_chinese_e96_svg(tmp_path: Path) -> None:
    sources = _sources()
    rows = build_architecture_portfolio(sources)
    gate = adjudicate_architecture_portfolio(
        ArchitecturePortfolioConfig(), {"ok": True}, rows
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E96" in svg
    assert "哪些神经结构真正还有实验资格" in svg
    assert "不训练模型" in svg
