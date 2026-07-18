from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from blockcipher_nd.tasks.innovation2.present_open_3sdp_exact_oracle import (
    audit_sbox_transition_parity,
)


AUDIT_OUTPUT_EXPONENTS = (1, 3, 7, 15)
AUDIT_REQUIRED_COMPLETE = (1, 3, 7)
AUDIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class GlpkEnumerationGateConfig:
    run_id: str
    mode: str = "audit"
    output_exponents: tuple[int, ...] = AUDIT_OUTPUT_EXPONENTS
    required_complete: tuple[int, ...] = AUDIT_REQUIRED_COMPLETE
    timeout_seconds: float = AUDIT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if not self.output_exponents:
            raise ValueError("output_exponents must be non-empty")
        if any(exponent not in range(1, 16) for exponent in self.output_exponents):
            raise ValueError("output exponents must be nonzero four-bit values")
        if not set(self.required_complete).issubset(self.output_exponents):
            raise ValueError("required_complete must be a subset of output_exponents")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.mode == "audit" and (
            self.output_exponents != AUDIT_OUTPUT_EXPONENTS
            or self.required_complete != AUDIT_REQUIRED_COMPLETE
            or self.timeout_seconds != AUDIT_TIMEOUT_SECONDS
        ):
            raise ValueError("E53-B audit protocol is frozen")


def exact_counts_by_output_exponent() -> dict[int, dict[int, int]]:
    transition = audit_sbox_transition_parity()
    counts: dict[int, dict[int, int]] = {
        output_exponent: {} for output_exponent in range(16)
    }
    for row in transition["rows"]:
        if row["trail_count"]:
            counts[int(row["output_exponent"])][int(row["input_exponent"])] = int(
                row["trail_count"]
            )
    return counts


def evaluate_glpk_enumeration_gate(
    config: GlpkEnumerationGateConfig, records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_exponent = {int(record["output_exponent"]): record for record in records}
    exact = exact_counts_by_output_exponent()
    for exponent, record in by_exponent.items():
        record_counts = {
            int(key): int(value) for key, value in record.get("counts", {}).items()
        }
        record["exact_expected_solutions"] = sum(exact[exponent].values())
        record["counts_match_exact"] = (
            record.get("status") == "completed" and record_counts == exact[exponent]
        )
        record["parity_matches_exact"] = (
            record.get("status") == "completed"
            and {
                key: value & 1 for key, value in record_counts.items() if value & 1
            }
            == {key: value & 1 for key, value in exact[exponent].items() if value & 1}
        )
    protocol_checks = {
        "audit_protocol_frozen": config.mode != "audit"
        or (
            config.output_exponents == AUDIT_OUTPUT_EXPONENTS
            and config.required_complete == AUDIT_REQUIRED_COMPLETE
            and config.timeout_seconds == AUDIT_TIMEOUT_SECONDS
        ),
        "all_queries_recorded": set(by_exponent) == set(config.output_exponents),
        "timeouts_separate_from_errors": all(
            record.get("status") in {"completed", "timeout", "error"}
            for record in records
        ),
        "completed_queries_use_glpk": all(
            record.get("backend") == "GLPKBackend"
            for record in records
            if record.get("status") == "completed"
        ),
    }
    correctness_checks = {
        "required_queries_complete": all(
            by_exponent[exponent].get("status") == "completed"
            and by_exponent[exponent].get("complete") is True
            for exponent in config.required_complete
        ),
        "required_counts_match_exact": all(
            by_exponent[exponent]["counts_match_exact"]
            for exponent in config.required_complete
        ),
        "required_parity_matches_exact": all(
            by_exponent[exponent]["parity_matches_exact"]
            for exponent in config.required_complete
        ),
    }
    scalability_checks = {
        "all_representative_queries_complete_within_budget": all(
            record.get("status") == "completed" and record.get("complete") is True
            for record in records
        ),
        "heaviest_query_completes_within_budget": by_exponent[
            max(config.output_exponents)
        ].get("status")
        == "completed",
        "all_representative_counts_match_exact": all(
            record["counts_match_exact"] for record in records
        ),
    }
    if not all(protocol_checks.values()) or not all(correctness_checks.values()):
        status = "fail"
        decision = "innovation2_present_r5_open_3sdp_glpk_enumerator_invalid"
        action = "repair GLPK constraints, blocking, count completeness, or timeout classification"
    elif not all(scalability_checks.values()):
        status = "hold"
        decision = "innovation2_present_r5_open_3sdp_glpk_blocking_not_scalable"
        action = (
            "stop per-solution GLPK blocking; audit exact GF(2) transition-tensor "
            "variable-elimination width before any five-round provider implementation"
        )
    else:
        status = "pass"
        decision = "innovation2_present_r5_open_3sdp_glpk_sbox_enumerator_ready"
        action = "extend the enumerator to the frozen one-round PRESENT circuit fixtures"
    completed = [record for record in records if record["status"] == "completed"]
    timed_out = [record for record in records if record["status"] == "timeout"]
    metrics = {
        "query_timeout_seconds": config.timeout_seconds,
        "queries": len(records),
        "completed_queries": len(completed),
        "timed_out_queries": len(timed_out),
        "error_queries": sum(record["status"] == "error" for record in records),
        "completed_solutions": sum(
            int(record.get("solutions", 0)) for record in completed
        ),
        "representative_expected_solutions": sum(
            int(record["exact_expected_solutions"]) for record in records
        ),
        "records": records,
        "available_alternative_backends": {
            "pysat": False,
            "pycryptosat": False,
            "z3": False,
            "bdd_dd": False,
            "model_counter_cli": False,
        },
    }
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "correctness_checks": correctness_checks,
        "scalability_checks": scalability_checks,
        "metrics": metrics,
        "claim_scope": (
            "PRESENT S-box GLPK per-solution blocking correctness and bounded "
            "scalability gate; not a full-cipher 3SDP provider, five-round label "
            "result, neural training, distinguisher, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
            "five_round_subset": False,
            "closed_routes": [
                "longer GLPK blocking timeout",
                "partial enumeration interpreted as parity",
                "five-round GLPK circuit expansion",
                "neural training before strict labels",
            ],
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_open_3sdp_glpk_enumeration_gate",
            **record,
            "gate_status": status,
            "decision": decision,
            "training_performed": False,
        }
        for record in records
    ]
    return {"gate": gate, "metrics": metrics, "result_rows": result_rows}


def serializable_config(config: GlpkEnumerationGateConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_exponents"] = list(config.output_exponents)
    payload["required_complete"] = list(config.required_complete)
    return payload
