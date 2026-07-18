from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_transition_tensor_boundary import (
    main as audit_main,
)
from blockcipher_nd.cli.plot_innovation2_present_transition_tensor_boundary import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_transition_tensor_boundary_audit import (
    TensorBoundaryAuditConfig,
    build_boundary_routes,
    evaluate_tensor_boundary_audit,
)


def _exact_summary() -> dict[str, object]:
    return {
        "run_id": "e53a",
        "gate": {
            "status": "pass",
            "decision": "innovation2_present_r5_open_3sdp_exact_oracle_ready",
        },
        "provider_manifest": {
            "providers": [{"variables": {"plaintext": 64, "key": 80, "total": 144}}]
        },
        "metrics": {
            "round_monomial_metrics": {
                "1": {"minimum": 11, "maximum": 42, "total": 1907},
                "2": {"minimum": 151, "maximum": 158490, "total": 4352830},
            }
        },
    }


def _fixtures() -> list[dict[str, int]]:
    return [
        {"rounds": 1, "superpoly_monomials": 0},
        {"rounds": 1, "superpoly_monomials": 13},
        {"rounds": 2, "superpoly_monomials": 0},
        {"rounds": 2, "superpoly_monomials": 53392},
    ]


def test_required_boundary_retains_key_and_all_inactive_variables() -> None:
    config = TensorBoundaryAuditConfig(run_id="e54")
    routes = build_boundary_routes(config)

    assert [row["retained_variables"] for row in routes] == [136, 56, 80, 0]
    assert [row["semantic_match"] for row in routes] == [True, False, False, False]
    assert routes[0]["dense_entries"] == 1 << 136
    assert routes[0]["within_variable_gate"] is False
    assert routes[0]["within_dense_memory_gate"] is False


def test_boundary_gate_stops_before_internal_factor_graph() -> None:
    result = evaluate_tensor_boundary_audit(
        TensorBoundaryAuditConfig(run_id="e54"),
        exact_summary=_exact_summary(),
        fixtures=_fixtures(),
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_present_r5_transition_tensor_boundary_infeasible"
    )
    assert result["gate"]["metrics"]["required_retained_variables"] == 136
    assert result["gate"]["metrics"]["internal_factor_graph_constructed"] is False
    assert result["elimination_rows"][0]["status"] == "skipped"
    assert result["gate"]["next_action"]["training"] is False


def test_cli_writes_boundary_and_skipped_elimination_artifacts(tmp_path: Path) -> None:
    exact_summary = tmp_path / "exact_summary.json"
    exact_fixtures = tmp_path / "fixtures.jsonl"
    output = tmp_path / "e54"
    exact_summary.write_text(json.dumps(_exact_summary()), encoding="utf-8")
    exact_fixtures.write_text(
        "".join(json.dumps(row) + "\n" for row in _fixtures()), encoding="utf-8"
    )

    exit_code = audit_main(
        [
            "--run-id",
            "e54_smoke_test",
            "--output-root",
            str(output),
            "--exact-summary",
            str(exact_summary),
            "--exact-fixtures",
            str(exact_fixtures),
            "--mode",
            "smoke",
        ]
    )

    assert exit_code == 0
    expected = {
        "factor_manifest.json",
        "elimination_orders.jsonl",
        "results.jsonl",
        "metadata.json",
        "summary.json",
        "gate.json",
        "progress.jsonl",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "hold"
    elimination = json.loads(
        (output / "elimination_orders.jsonl").read_text(encoding="utf-8")
    )
    assert elimination["status"] == "skipped"

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
    assert "创新2 E54" in svg
    assert "不是内部treewidth" in svg
