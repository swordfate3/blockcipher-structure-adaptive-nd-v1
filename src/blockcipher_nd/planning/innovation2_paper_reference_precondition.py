from __future__ import annotations

import json
from pathlib import Path
from typing import Any


READY_DECISION = "innovation2_paper_reference_seed1_precondition_ready"
BLOCKED_DECISION = "innovation2_paper_reference_seed1_precondition_blocked"


def paper_reference_seed1_precondition_report(
    *,
    plan_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    load_errors: list[str] = []
    plan = _load_object(plan_path, load_errors)
    gate = _load_object(source_root / "gate.local.json", load_errors)
    validation = _load_object(source_root / "validation.local.json", load_errors)
    dataset = _load_object(source_root / "dataset_summary.json", load_errors)
    cache = _load_object(source_root / "cache_metadata.json", load_errors)
    memory = _load_object(source_root / "memory_preflight.json", load_errors)
    rows = _load_jsonl(source_root / "results.jsonl", load_errors)

    precondition_value = plan.get("required_precondition", {})
    precondition = precondition_value if isinstance(precondition_value, dict) else {}
    allowed_decisions_value = precondition.get("decision_in", [])
    allowed_decisions = (
        allowed_decisions_value if isinstance(allowed_decisions_value, list) else []
    )
    paper_checks_value = gate.get("paper_reference_plan_checks", {})
    paper_checks = paper_checks_value if isinstance(paper_checks_value, dict) else {}
    readiness_checks_value = gate.get("readiness_checks", {})
    readiness_checks = (
        readiness_checks_value if isinstance(readiness_checks_value, dict) else {}
    )
    readjudication_value = gate.get("readjudication", {})
    readjudication = (
        readjudication_value if isinstance(readjudication_value, dict) else {}
    )
    row_roles = {str(row.get("role")) for row in rows}
    try:
        row_seeds = {int(row.get("seed")) for row in rows}
    except (TypeError, ValueError):
        row_seeds = set()
    row_run_ids = {str(row.get("run_id", "")) for row in rows}
    cache_entries = list(cache.values()) if isinstance(cache, dict) else []
    expected_run_id = str(precondition.get("run_id", ""))

    checks = {
        "seed1_plan_id_expected": plan.get("plan_id")
        == (
            "innovation2_present_r8_high_round_integral_"
            "paper_reference_2pow21_seed1"
        ),
        "seed1_plan_is_conditionally_blocked": plan.get("launch_state")
        == "blocked_until_seed0_retrieved_gate_and_visual_qa_pass",
        "remote_package_not_prebuilt": plan.get("remote_package_state")
        == "not_generated_until_precondition_passes",
        "one_variable_is_seed0_to_seed1": plan.get("one_variable_change")
        == {"field": "seed", "anchor_value": 0, "candidate_value": 1},
        "verified_result_branch_retrieved": (
            source_root / "retrieved_from_verified_result_branch.marker"
        ).is_file(),
        "visual_qa_passed": (source_root / "visual_qa_passed.marker").is_file(),
        "visual_qa_not_pending": not (
            source_root / "visual_qa_pending.marker"
        ).exists(),
        "local_gate_status_pass": gate.get("status") == precondition.get("status"),
        "local_gate_decision_allows_seed1": gate.get("decision")
        in allowed_decisions,
        "local_gate_run_id_matches_precondition": gate.get("run_id")
        == expected_run_id,
        "local_gate_mode_is_paper_reference": gate.get("gate_mode")
        == "paper_reference",
        "local_gate_rounds_is_8": gate.get("rounds") == 8,
        "all_paper_reference_plan_checks_pass": bool(paper_checks)
        and all(bool(value) for value in paper_checks.values()),
        "all_readiness_checks_pass": bool(readiness_checks)
        and all(bool(value) for value in readiness_checks.values()),
        "source_revision_matches_expected": bool(
            readjudication.get("source_revision_matches_expected")
        ),
        "local_artifact_validation_pass": validation.get("status") == "pass",
        "dataset_summary_pass": dataset.get("status") == "pass",
        "three_disk_caches_complete": len(cache_entries) == 3
        and all(
            isinstance(entry, dict) and entry.get("status") == "complete"
            for entry in cache_entries
        ),
        "memory_preflight_pass": memory.get("status") == "pass",
        "memory_preflight_batch_is_2000": memory.get("batch_size") == 2000,
        "memory_preflight_has_headroom": _float_at_most(
            memory.get("max_peak_reserved_fraction"),
            0.9,
        ),
        "four_result_roles_present": len(rows) == 4
        and row_roles == {"anchor", "candidate", "linear", "control"},
        "result_rows_are_seed0": row_seeds == {0},
        "result_rows_match_source_run_id": row_run_ids == {expected_run_id},
    }
    errors = [*load_errors, *(name for name, passed in checks.items() if not passed)]
    ready = not errors
    return {
        "status": "pass" if ready else "blocked",
        "decision": READY_DECISION if ready else BLOCKED_DECISION,
        "plan": str(plan_path),
        "source_root": str(source_root),
        "source_run_id": expected_run_id,
        "checks": checks,
        "errors": errors,
        "should_generate_remote_package": ready,
        "should_launch_remote": False,
        "next_action": (
            "Generate the exact seed1 remote config, launcher, and monitor from "
            "the frozen plan; run readiness tests, commit and push, then launch "
            "from that pushed commit."
            if ready
            else "Keep seed1 blocked and repair or wait for the listed seed0 evidence."
        ),
        "claim_scope": (
            "read-only Innovation 2 paper-reference seed1 precondition gate; "
            "no training and no remote launch performed"
        ),
    }


def _load_object(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"load_failed:{path}:{error}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"expected_object:{path}")
        return {}
    return payload


def _load_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"load_failed:{path}:{error}")
        return []
    if not all(isinstance(row, dict) for row in rows):
        errors.append(f"expected_jsonl_objects:{path}")
        return []
    return rows


def _float_at_most(value: Any, maximum: float) -> bool:
    try:
        return float(value) <= maximum
    except (TypeError, ValueError):
        return False


__all__ = [
    "BLOCKED_DECISION",
    "READY_DECISION",
    "paper_reference_seed1_precondition_report",
]
