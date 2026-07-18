from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_neural_architecture_boundary import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.neural_architecture_boundary_synthesis import (
    EXPECTED_SOURCES,
    NeuralArchitectureBoundaryConfig,
    adjudicate_boundary_synthesis,
    architecture_rows,
    validate_gate_sources,
)


def _sources() -> dict[str, object]:
    sources = {
        role: {
            "root": f"/tmp/{role}",
            "sha256": "a" * 64,
            "gate": {"run_id": run_id, "status": status, "decision": decision},
        }
        for role, (run_id, status, decision) in EXPECTED_SOURCES.items()
    }
    sources["formal_method"]["gate"].update(
        {
            "source_checks": {"valid": True},
            "method_checks": {"valid": True},
            "skinny_readiness_checks": {"valid": True},
            "metrics": {
                "ciphers": [
                    {
                        "cipher": "PRESENT-80",
                        "mean_true_minus_corrupted": 0.096944,
                    },
                    {
                        "cipher": "GIFT-64",
                        "mean_true_minus_corrupted": 0.132414,
                    },
                ]
            },
        }
    )
    sources["skinny_residual"]["gate"].update(
        {
            "protocol_checks": {"valid": True},
            "readiness_checks": {"margin": False},
            "metrics": {
                "true_minus_independent": -0.000457,
                "true_minus_corrupted": 0.000152,
            },
        }
    )
    sources["shared_operator"]["gate"].update(
        {
            "protocol_checks": {"valid": True},
            "relation_checks": {"valid": True},
            "candidate_checks": {"gift_quality": False},
            "metrics": {
                "gift_true_minus_anchor": -0.053590,
                "present_true_minus_anchor": 0.004722,
            },
        }
    )
    sources["rectangle_labels"]["gate"].update(
        {
            name: {"valid": True}
            for name in (
                "protocol_checks",
                "raw_width_checks",
                "matching_width_checks",
                "shortcut_checks",
            )
        }
    )
    sources["rectangle_untyped"]["gate"].update(
        {
            "protocol_checks": {"valid": True},
            "candidate_checks": {"valid": True},
            "relation_checks": {"wrong_margin": False},
            "metrics": {
                "true_minus_corrupted": 0.029646,
                "true_minus_independent": 0.164331,
            },
        }
    )
    sources["rectangle_row_mechanism"]["gate"].update(
        {
            "protocol_checks": {"valid": True},
            "mechanism_checks": {"valid": True},
            "metrics": {
                "typed_true_minus_untyped_true": 0.014282,
                "typed_true_minus_wrong_row_typed": 0.017224,
            },
        }
    )
    sources["rectangle_row_operator"]["gate"].update(
        {
            "protocol_checks": {"valid": True},
            "readiness_checks": {"row_margin": False},
            "metrics": {
                "typed_true_minus_untyped": 0.007792,
                "typed_true_minus_wrong_row": 0.006244,
            },
        }
    )
    return sources


def test_e93_replays_all_sources_and_builds_frozen_ranking() -> None:
    sources = _sources()
    checks = validate_gate_sources(sources)
    rows = architecture_rows(sources)

    assert all(checks.values())
    assert len(rows) == 7
    assert rows[0]["evidence_class"] == "formal_confirmed"
    assert rows[0]["route"] == "separate_r3_only_profile_operator"
    assert sum(row["evidence_class"] == "closed" for row in rows) == 3
    assert rows[-1]["status"] == "no_budget"
    assert all("primary_margin" in row for row in rows)


def test_e93_gate_confirms_method_and_stops_architecture_enumeration() -> None:
    sources = _sources()
    gate = adjudicate_boundary_synthesis(
        NeuralArchitectureBoundaryConfig(run_id="e93-test"),
        validate_gate_sources(sources),
        architecture_rows(sources),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_architecture_boundary_confirmed_third_spn_neural_not_confirmed"
    )
    assert gate["metrics"]["formal_real_spn_count"] == 2
    assert gate["next_action"]["training"] is False


def test_e93_gate_fails_when_a_frozen_source_changes() -> None:
    sources = _sources()
    sources["rectangle_row_operator"]["gate"]["status"] = "pass"
    gate = adjudicate_boundary_synthesis(
        NeuralArchitectureBoundaryConfig(run_id="e93-test"),
        validate_gate_sources(sources),
        architecture_rows(sources),
    )

    assert gate["status"] == "fail"
    assert gate["decision"] == (
        "innovation2_architecture_boundary_synthesis_protocol_invalid"
    )


def test_e93_plot_writes_clear_chinese_svg(tmp_path: Path) -> None:
    sources = _sources()
    gate = adjudicate_boundary_synthesis(
        NeuralArchitectureBoundaryConfig(run_id="e93-test"),
        validate_gate_sources(sources),
        architecture_rows(sources),
    )
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "curves.svg"
    summary_path.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary_path), "--output", str(output_path)]) == 0
    svg = output_path.read_text(encoding="utf-8")
    assert "创新2 E93" in svg
    assert "不用于跨密码比较绝对AUC" in svg
    assert "重新开放训练的条件" in svg
