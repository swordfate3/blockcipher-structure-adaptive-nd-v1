from __future__ import annotations

from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_launch_gate import (
    REQUIRED_FALLBACK_FILES,
    SEED0_DECISION,
    SEED0_RUN_ID,
    adjudicate_runtime_spn_skinny_seed1_launch,
)


def _gate(*, published: bool = True, protected_changes: list[str] | None = None):
    return adjudicate_runtime_spn_skinny_seed1_launch(
        source_commit="a" * 40,
        upstream_ref="origin/main",
        artifact_names=set(REQUIRED_FALLBACK_FILES),
        seed0_gate={
            "run_id": SEED0_RUN_ID,
            "seed": 0,
            "status": "pass",
            "decision": SEED0_DECISION,
            "aucs": {"true": 0.64, "corrupted": 0.60, "independent": 0.51},
            "margins": {
                "true_minus_corrupted": 0.04,
                "true_minus_independent": 0.13,
            },
            "protocol_checks": {"protocol": True},
            "research_checks": {"research": True},
        },
        seed0_validation={
            "status": "pass",
            "expected_rows": 3,
            "result_rows": 3,
            "errors": [],
        },
        source_commit_valid=True,
        source_commit_exists=True,
        source_commit_published=published,
        training_commit_exists=True,
        protected_changes=protected_changes or [],
        seed1_plan_identical=True,
        seed1_remote_protocol_identical=True,
    )


def test_launch_gate_authorizes_only_published_equivalent_seed1_source() -> None:
    gate = _gate()

    assert gate["status"] == "pass"
    assert gate["should_ssh"] is True
    assert gate["ssh_allowed"] is True
    assert gate["launch_authorized"] is True
    assert gate["protected_changes"] == []


def test_launch_gate_holds_unpublished_source_without_weakening_research_gate() -> None:
    gate = _gate(published=False)

    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_rtg2a_seed1_source_not_published"
    assert gate["should_ssh"] is True
    assert gate["ssh_allowed"] is False
    assert gate["launch_authorized"] is False
    assert gate["publication_checks"] == {
        "source_commit_published_to_upstream": False
    }


def test_launch_gate_fails_closed_when_training_path_changed() -> None:
    gate = _gate(protected_changes=["src/blockcipher_nd/models/runtime.py"])

    assert gate["status"] == "fail"
    assert gate["should_ssh"] is False
    assert gate["ssh_allowed"] is True
    assert gate["launch_authorized"] is False
    assert gate["equivalence_checks"]["protected_training_paths_unchanged"] is False
