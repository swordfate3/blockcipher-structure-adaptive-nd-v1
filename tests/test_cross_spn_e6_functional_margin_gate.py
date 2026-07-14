from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from blockcipher_nd.planning.cross_spn_e6_functional_margin_gate import (
    E6_MODEL_ROLES,
    gate_cross_spn_e6_functional_margin_joint,
)
from blockcipher_nd.planning.matrix import build_tasks


def test_e6_joint_gate_stops_when_either_target_seed_fails() -> None:
    reports = [
        {
            "status": "pass",
            "errors": [],
            "expected_source_seed": 0,
            "expected_target_seed": 2,
            "research_decision_applied": True,
            "gate_pass": True,
        },
        {
            "status": "pass",
            "errors": [],
            "expected_source_seed": 0,
            "expected_target_seed": 3,
            "research_decision_applied": True,
            "gate_pass": False,
        },
    ]

    report = gate_cross_spn_e6_functional_margin_joint(reports)

    assert report["decision"] == "e6_r0_functional_margin_rejected"
    assert report["gate_pass"] is False
    assert "65536_per_class_remote_medium" in report["stopped_actions"]


def test_e6_target_plans_freeze_models_budget_and_target_seeds() -> None:
    root = Path("configs/experiment/innovation1")
    for target_seed in (2, 3):
        tasks = build_tasks(
            SimpleNamespace(
                plan=str(
                    root
                    / (
                        "innovation1_spn_gift64_cross_spn_e6_target_8192_"
                        f"source_seed0_target_seed{target_seed}.csv"
                    )
                ),
                feature_encoding="ciphertext_pair_bits",
                pairs_per_sample=1,
                difference_profile=None,
                difference_member=0,
                key_rotation_interval=0,
                sample_structure="independent_pairs",
                integral_active_nibble=0,
            )
        )

        assert [task["model_key"] for task in tasks] == list(E6_MODEL_ROLES.values())
        assert {task["seed"] for task in tasks} == {target_seed}
        assert {task["samples_per_class"] for task in tasks} == {8192}
        assert {task["pairs_per_sample"] for task in tasks} == {4}
