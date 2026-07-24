from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window_readiness import (
    adjudicate_recurrent_window_readiness,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_medium import (
    CONTROL_MARGIN,
    RTG3A_RUN_STEM,
    RTG3A_SAMPLES_PER_CLASS,
    SIGNAL_FLOOR,
    adjudicate_runtime_spn_skinny_medium_joint,
)


RUN_ID = "i1_uknit64_runtime_e4_recurrent_window_r5_u3_authorization_20260725"
RTG3_SEED0_RUN_ID = f"{RTG3A_RUN_STEM}_seed0_20260725"
RTG3_JOINT_RUN_ID = f"{RTG3A_RUN_STEM}_joint_seed0_seed1_20260725"
U3_READINESS_RUN_ID = (
    "i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_readiness_20260725"
)
U3_PLAN = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv"
)
U3_PLAN_SHA256 = "060805c3e1e6793aa11b3e9758ddef738d646c77df596150032c8486b7bbd87f"
SEED0_DECISIONS = {
    "pass": "innovation1_rtg3a_skinny_formal_seed0_supported",
    "hold": "innovation1_rtg3a_skinny_formal_not_supported",
    "fail": "innovation1_rtg3a_skinny_formal_protocol_invalid",
}
JOINT_DECISIONS = {
    "pass": "innovation1_rtg3a_skinny_formal_two_seed_supported",
    "hold": "innovation1_rtg3a_skinny_formal_two_seed_not_supported",
    "fail": "innovation1_rtg3a_skinny_formal_joint_protocol_invalid",
}


def build_runtime_spn_uknit_u3_launch_gate(
    *,
    seed0_root: Path,
    readiness_root: Path,
    plan: Path,
    repository: Path,
    rtg3_joint_root: Path | None = None,
) -> dict[str, Any]:
    seed0_path = seed0_root / "gate.local.json"
    seed0_gate = _read_json(seed0_path)
    seed0_sha256 = _sha256_file(seed0_path)

    readiness_gate = _read_json(readiness_root / "gate.json")
    readiness_validation = _read_json(readiness_root / "validation.json")
    readiness_manifests = _read_jsonl(readiness_root / "manifest.jsonl")
    readiness_recomputed = adjudicate_recurrent_window_readiness(
        run_id=U3_READINESS_RUN_ID,
        manifests=readiness_manifests,
    )

    joint_present = rtg3_joint_root is not None
    joint_gate: dict[str, Any] = {}
    joint_validation: dict[str, Any] = {}
    joint_recomputed: dict[str, Any] = {}
    joint_sources_verified = False
    seed0_linked_to_joint = False
    if rtg3_joint_root is not None:
        joint_gate = _read_json(rtg3_joint_root / "gate.json")
        joint_validation = _read_json(rtg3_joint_root / "validation.json")
        source_gates, actual_hashes, joint_sources_verified = _joint_sources(
            joint_validation,
            repository=repository,
        )
        if len(source_gates) == 2:
            joint_recomputed = adjudicate_runtime_spn_skinny_medium_joint(
                run_id=RTG3_JOINT_RUN_ID,
                gates=source_gates,
                phase="rtg3a",
            )
        seed0_linked_to_joint = bool(seed0_sha256) and seed0_sha256 in actual_hashes

    expected_plan = repository / U3_PLAN
    return adjudicate_runtime_spn_uknit_u3_launch(
        seed0_gate=seed0_gate,
        joint_present=joint_present,
        joint_gate=joint_gate,
        joint_validation=joint_validation,
        joint_recomputed_exact=bool(joint_recomputed)
        and joint_gate == joint_recomputed,
        joint_sources_verified=joint_sources_verified,
        seed0_linked_to_joint=seed0_linked_to_joint,
        readiness_gate=readiness_gate,
        readiness_validation=readiness_validation,
        readiness_recomputed_exact=readiness_gate == readiness_recomputed,
        readiness_manifest_rows=len(readiness_manifests),
        plan_path_exact=_same_path(plan, expected_plan),
        plan_sha256=_sha256_file(plan),
    )


def adjudicate_runtime_spn_uknit_u3_launch(
    *,
    seed0_gate: dict[str, Any],
    joint_present: bool,
    joint_gate: dict[str, Any],
    joint_validation: dict[str, Any],
    joint_recomputed_exact: bool,
    joint_sources_verified: bool,
    seed0_linked_to_joint: bool,
    readiness_gate: dict[str, Any],
    readiness_validation: dict[str, Any],
    readiness_recomputed_exact: bool,
    readiness_manifest_rows: int,
    plan_path_exact: bool,
    plan_sha256: str,
) -> dict[str, Any]:
    seed0_status = str(seed0_gate.get("status", ""))
    seed0_protocol = seed0_gate.get("protocol_checks")
    seed0_protocol_all_pass = _all_true(seed0_protocol)
    seed0_checks = {
        "seed0_identity_exact": (
            seed0_gate.get("run_id") == RTG3_SEED0_RUN_ID
            and seed0_gate.get("phase") == "rtg3a"
            and seed0_gate.get("seed") == 0
        ),
        "seed0_terminal_decision_consistent": (
            seed0_status in SEED0_DECISIONS
            and seed0_gate.get("decision") == SEED0_DECISIONS.get(seed0_status)
        ),
        "seed0_formal_scale_exact": (
            seed0_gate.get("samples_per_class") == RTG3A_SAMPLES_PER_CLASS
            and seed0_gate.get("train_rows") == 2_000_000
            and seed0_gate.get("validation_rows") == 1_000_000
        ),
        "seed0_thresholds_exact": seed0_gate.get("thresholds")
        == {"true_auc": SIGNAL_FLOOR, "control_margin": CONTROL_MARGIN},
        "seed0_protocol_state_consistent": (
            seed0_protocol_all_pass
            if seed0_status in {"pass", "hold"}
            else isinstance(seed0_protocol, dict)
            and bool(seed0_protocol)
            and not seed0_protocol_all_pass
        ),
        "seed0_research_contract_consistent": (
            _research_contract_matches(seed0_gate)
            if seed0_status in {"pass", "hold"}
            else True
        ),
    }
    seed0_evidence_valid = all(seed0_checks.values())

    joint_status = str(joint_gate.get("status", ""))
    joint_protocol = joint_gate.get("protocol_checks")
    joint_protocol_all_pass = _all_true(joint_protocol)
    joint_checks = {
        "joint_evidence_present": joint_present,
        "joint_identity_exact": joint_present
        and joint_gate.get("run_id") == RTG3_JOINT_RUN_ID
        and joint_gate.get("phase") == "rtg3a"
        and joint_gate.get("samples_per_class") == RTG3A_SAMPLES_PER_CLASS,
        "joint_terminal_decision_consistent": joint_present
        and joint_status in JOINT_DECISIONS
        and joint_gate.get("decision") == JOINT_DECISIONS.get(joint_status),
        "joint_protocol_state_consistent": joint_present
        and (
            joint_protocol_all_pass
            if joint_status in {"pass", "hold"}
            else isinstance(joint_protocol, dict)
            and bool(joint_protocol)
            and not joint_protocol_all_pass
        ),
        "joint_research_state_consistent": joint_present
        and _joint_research_state_matches(joint_gate),
        "joint_validation_matches_gate": joint_present
        and joint_validation.get("run_id") == RTG3_JOINT_RUN_ID
        and joint_validation.get("status")
        == ("pass" if joint_protocol_all_pass else "fail")
        and joint_validation.get("checks") == joint_protocol,
        "joint_sources_verified": joint_present and joint_sources_verified,
        "seed0_gate_linked_by_sha256": joint_present and seed0_linked_to_joint,
        "joint_gate_recomputed_exact": joint_present and joint_recomputed_exact,
    }
    joint_evidence_valid = joint_present and all(joint_checks.values())

    readiness_protocol = readiness_gate.get("protocol_checks")
    readiness_checks = {
        "readiness_identity_exact": (
            readiness_gate.get("run_id") == U3_READINESS_RUN_ID
            and readiness_gate.get("task")
            == "innovation1_runtime_spn_recurrent_window_readiness"
            and readiness_gate.get("status") == "pass"
            and readiness_gate.get("decision")
            == "innovation1_runtime_spn_recurrent_window_readiness_passed"
        ),
        "readiness_protocol_checks_pass": _all_true(readiness_protocol),
        "readiness_gate_recomputed_exact": readiness_recomputed_exact,
        "readiness_manifest_exact_ten_rows": readiness_manifest_rows == 10,
        "readiness_validation_exact": (
            readiness_validation.get("run_id") == U3_READINESS_RUN_ID
            and readiness_validation.get("status") == "pass"
            and readiness_validation.get("checks") == readiness_protocol
            and readiness_validation.get("manifest_rows") == 10
            and readiness_validation.get("expected_rows") == 10
            and readiness_validation.get("training_performed") is False
            and readiness_validation.get("plan") == U3_PLAN.as_posix()
        ),
        "u3_plan_path_exact": plan_path_exact,
        "u3_plan_sha256_exact": plan_sha256 == U3_PLAN_SHA256,
    }
    readiness_valid = all(readiness_checks.values())

    execution_authorized = (
        seed0_evidence_valid
        and seed0_status == "pass"
        and joint_evidence_valid
        and joint_status == "pass"
        and readiness_valid
    )
    if execution_authorized:
        status = "pass"
        decision = "innovation1_runtime_spn_uknit_u3_execution_authorized"
        next_action = (
            "run the unchanged ten-row uKNIT U3 local diagnostic, then apply its "
            "frozen result gate and visual QA"
        )
    elif not seed0_evidence_valid or not readiness_valid:
        status = "fail"
        decision = "innovation1_runtime_spn_uknit_u3_authorization_invalid"
        next_action = (
            "repair only the failed RTG3 seed0, U3 readiness, or frozen-plan "
            "identity check; do not train U3"
        )
    elif seed0_status == "fail":
        status = "fail"
        decision = "innovation1_runtime_spn_uknit_u3_rtg3_protocol_invalid"
        next_action = "repair RTG3 protocol evidence; do not train U3"
    elif seed0_status == "hold":
        status = "hold"
        decision = "innovation1_runtime_spn_uknit_u3_not_authorized"
        next_action = "keep U3 stopped because RTG3 seed0 did not support the runtime-topology route"
    elif not joint_present:
        status = "hold"
        decision = "innovation1_runtime_spn_uknit_u3_waiting_for_rtg3_joint"
        next_action = (
            "wait for the conditional RTG3 seed1 result and verified two-seed joint gate; "
            "do not train U3"
        )
    elif not joint_evidence_valid or joint_status == "fail":
        status = "fail"
        decision = "innovation1_runtime_spn_uknit_u3_rtg3_joint_invalid"
        next_action = "repair the RTG3 joint evidence chain; do not train U3"
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_uknit_u3_not_authorized"
        next_action = (
            "keep U3 stopped because the two-seed RTG3 joint gate did not support "
            "the runtime-topology route"
        )

    return {
        "run_id": RUN_ID,
        "task": "innovation1_runtime_spn_uknit_u3_execution_authorization",
        "status": status,
        "decision": decision,
        "execution_authorized": execution_authorized,
        "rtg3_seed0_status": seed0_status,
        "rtg3_joint_status": joint_status if joint_present else "missing",
        "u3_plan": U3_PLAN.as_posix(),
        "u3_plan_sha256": plan_sha256,
        "expected_u3_plan_sha256": U3_PLAN_SHA256,
        "seed0_checks": seed0_checks,
        "joint_checks": joint_checks,
        "readiness_checks": readiness_checks,
        "next_action": next_action,
        "blocked_actions": [
            "train U3 unless execution_authorized is true",
            "treat a seed0 pass as two-seed RTG3 support",
            "train U3 while RTG3 seed1 or its joint gate is missing",
            "train U3 after an RTG3 hold or protocol failure",
            "change the frozen U3 CSV, model, data, controls, thresholds, or optimizer",
            "launch this 2048/class diagnostic on the remote GPU",
        ],
        "claim_scope": (
            "local execution authorization only; no U3 training result, transfer, "
            "attack, SOTA, breakthrough, or universal-SPN claim"
        ),
    }


def _research_contract_matches(gate: dict[str, Any]) -> bool:
    aucs = gate.get("aucs")
    margins = gate.get("margins")
    checks = gate.get("research_checks")
    if not isinstance(aucs, dict) or not isinstance(margins, dict):
        return False
    if not all(
        _finite(aucs.get(role)) for role in ("true", "corrupted", "independent")
    ):
        return False
    true_auc = float(aucs["true"])
    true_minus_corrupted = true_auc - float(aucs["corrupted"])
    true_minus_independent = true_auc - float(aucs["independent"])
    expected_checks = {
        "true_auc_at_least_0p55": true_auc >= SIGNAL_FLOOR,
        "true_exceeds_corrupted_by_0p005": true_minus_corrupted >= CONTROL_MARGIN,
        "true_exceeds_independent_by_0p005": true_minus_independent >= CONTROL_MARGIN,
    }
    return (
        _close(margins.get("true_minus_corrupted"), true_minus_corrupted)
        and _close(margins.get("true_minus_independent"), true_minus_independent)
        and checks == expected_checks
        and (
            all(expected_checks.values())
            if gate.get("status") == "pass"
            else not all(expected_checks.values())
        )
    )


def _joint_research_state_matches(gate: dict[str, Any]) -> bool:
    checks = gate.get("research_checks")
    if not isinstance(checks, dict) or set(checks) != {"both_formal_seeds_supported"}:
        return False
    supported = checks["both_formal_seeds_supported"] is True
    return supported if gate.get("status") == "pass" else not supported


def _joint_sources(
    validation: dict[str, Any],
    *,
    repository: Path,
) -> tuple[list[dict[str, Any]], set[str], bool]:
    entries = validation.get("sources")
    if not isinstance(entries, list) or len(entries) != 2:
        return [], set(), False
    gates: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return [], set(), False
        raw_path = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            return [], set(), False
        path = Path(raw_path)
        if not path.is_absolute():
            path = repository / path
        actual_hash = _sha256_file(path)
        gate = _read_json(path)
        if not actual_hash or actual_hash != expected_hash or not gate:
            return [], set(), False
        gates.append(gate)
        hashes.add(actual_hash)
    return gates, hashes, len(hashes) == 2


def _all_true(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(item is True for item in value.values())
    )


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _close(left: object, right: float) -> bool:
    return _finite(left) and abs(float(left) - right) <= 1e-7


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return []
    return values if all(isinstance(value, dict) for value in values) else []


__all__ = [
    "RUN_ID",
    "U3_PLAN",
    "adjudicate_runtime_spn_uknit_u3_launch",
    "build_runtime_spn_uknit_u3_launch_gate",
]
