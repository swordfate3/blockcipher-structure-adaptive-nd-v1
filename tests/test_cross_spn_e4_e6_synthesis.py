from __future__ import annotations

from blockcipher_nd.planning.cross_spn_e4_e6_synthesis import (
    build_cross_spn_e4_e6_synthesis,
)


def _e4() -> dict:
    return {
        "status": "pass",
        "errors": [],
        "decision": "e4_typed_topology_attribution_robust_scratch_efficiency_conditional",
        "comparisons": {
            "scratch_margin": {
                "pass_count": 2,
                "cell_count": 4,
                "minimum": 0.0,
                "maximum": 0.01,
                "all_cells_pass": False,
            },
            "source_topology_margin": {
                "pass_count": 4,
                "cell_count": 4,
                "minimum": 0.01,
                "maximum": 0.02,
                "all_cells_pass": True,
            },
            "target_topology_margin": {
                "pass_count": 4,
                "cell_count": 4,
                "minimum": 0.07,
                "maximum": 0.08,
                "all_cells_pass": True,
            },
        },
    }


def _objective(stage: str) -> dict:
    decision = {
        "e5": "e5_r0_source_objective_rejected",
        "e6": "e6_r0_functional_margin_rejected",
    }[stage]
    per_seed = {}
    for seed in (2, 3):
        per_seed[str(seed)] = {
            "aucs": {
                "scratch": 0.53 + seed * 0.001,
                "off_transfer": 0.56 + seed * 0.001,
                "candidate_transfer": 0.559 + seed * 0.001,
                "placebo_transfer": 0.561 + seed * 0.001,
            },
            "margins": {
                "off_transfer": -0.001,
                "placebo_transfer": -0.002,
                "scratch": 0.029,
            },
            "point_pass": {
                "off_transfer": False,
                "placebo_transfer": False,
                "scratch": True,
            },
            "ci_pass": {
                "off_transfer": False,
                "placebo_transfer": False,
                "scratch": True,
            },
            "bootstrap": {
                "comparisons": {
                    "off_transfer": {"ci_lower": -0.003, "ci_upper": 0.001},
                    "placebo_transfer": {"ci_lower": -0.004, "ci_upper": 0.0},
                    "scratch": {"ci_lower": 0.02, "ci_upper": 0.04},
                }
            },
            "gate_pass": False,
        }
    return {
        "status": "pass",
        "errors": [],
        "decision": decision,
        "expected_source_seed": 0,
        "expected_target_seeds": [2, 3],
        "research_decision_applied": True,
        "per_seed": per_seed,
    }


def test_e4_e6_synthesis_retains_topology_and_rejects_source_objectives() -> None:
    report = build_cross_spn_e4_e6_synthesis(
        _e4(),
        _objective("e5"),
        _objective("e6"),
    )

    assert report["status"] == "pass"
    assert report["decision"] == (
        "typed_topology_representation_retained_source_objectives_rejected"
    )
    assert report["e4_representation"]["source_topology_margin"]["pass_count"] == 4
    assert report["objective_summary"] == {
        "cell_count": 4,
        "complete_gate_pass_count": 0,
        "candidate_vs_off_pass_count": 0,
        "candidate_vs_placebo_pass_count": 0,
        "candidate_vs_scratch_pass_count": 4,
        "all_objectives_rejected": True,
        "ordinary_transfer_signal_retained": True,
    }


def test_e4_e6_synthesis_rejects_mismatched_shared_anchors() -> None:
    e6 = _objective("e6")
    e6["per_seed"]["2"]["aucs"]["off_transfer"] += 0.001

    report = build_cross_spn_e4_e6_synthesis(_e4(), _objective("e5"), e6)

    assert report["status"] == "fail"
    assert any("shared anchor" in error for error in report["errors"])
