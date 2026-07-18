from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


AUDIT_TARGET_ROUNDS = 5
AUDIT_ACTIVE_DIMENSION = 8
AUDIT_KEY_VARIABLES = 80
AUDIT_PLAINTEXT_VARIABLES = 64
AUDIT_MAX_RETAINED_VARIABLES = 26
AUDIT_MAX_DENSE_BYTES = 4 * (1 << 30)
E53A_DECISION = "innovation2_present_r5_open_3sdp_exact_oracle_ready"


@dataclass(frozen=True)
class TensorBoundaryAuditConfig:
    run_id: str
    mode: str = "audit"
    target_rounds: int = AUDIT_TARGET_ROUNDS
    active_dimension: int = AUDIT_ACTIVE_DIMENSION
    key_variables: int = AUDIT_KEY_VARIABLES
    plaintext_variables: int = AUDIT_PLAINTEXT_VARIABLES
    maximum_retained_variables: int = AUDIT_MAX_RETAINED_VARIABLES
    maximum_dense_bytes: int = AUDIT_MAX_DENSE_BYTES

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if not 0 < self.active_dimension < self.plaintext_variables:
            raise ValueError("active_dimension must be inside the plaintext width")
        if self.key_variables <= 0 or self.target_rounds <= 0:
            raise ValueError("key_variables and target_rounds must be positive")
        if self.maximum_retained_variables <= 0 or self.maximum_dense_bytes <= 0:
            raise ValueError("boundary limits must be positive")
        if self.mode == "audit" and (
            self.target_rounds != AUDIT_TARGET_ROUNDS
            or self.active_dimension != AUDIT_ACTIVE_DIMENSION
            or self.key_variables != AUDIT_KEY_VARIABLES
            or self.plaintext_variables != AUDIT_PLAINTEXT_VARIABLES
            or self.maximum_retained_variables != AUDIT_MAX_RETAINED_VARIABLES
            or self.maximum_dense_bytes != AUDIT_MAX_DENSE_BYTES
        ):
            raise ValueError("E54 boundary audit protocol is frozen")


def build_boundary_routes(config: TensorBoundaryAuditConfig) -> list[dict[str, Any]]:
    inactive = config.plaintext_variables - config.active_dimension
    definitions = (
        (
            "all_key_all_inactive_offset",
            config.key_variables + inactive,
            True,
            "required innovation2 positive-label semantics",
        ),
        (
            "fixed_key_all_inactive_offset",
            inactive,
            False,
            "drops the all-key proof scope",
        ),
        (
            "all_key_zero_inactive_offset",
            config.key_variables,
            False,
            "key coefficient at zero offset only",
        ),
        (
            "fixed_key_fixed_offset",
            0,
            False,
            "finite assignment evaluation, not a universal certificate",
        ),
    )
    rows: list[dict[str, Any]] = []
    for route, retained, semantic_match, note in definitions:
        entries = 1 << retained
        rows.append(
            {
                "route": route,
                "retained_variables": retained,
                "dense_entries": entries,
                "one_byte_dense_bytes": entries,
                "bitpacked_dense_bytes": (entries + 7) // 8,
                "semantic_match": semantic_match,
                "within_variable_gate": retained
                <= config.maximum_retained_variables,
                "within_dense_memory_gate": entries <= config.maximum_dense_bytes,
                "note": note,
            }
        )
    return rows


def summarize_sparse_evidence(
    exact_summary: dict[str, Any], fixtures: list[dict[str, Any]]
) -> dict[str, Any]:
    round_metrics = exact_summary["metrics"]["round_monomial_metrics"]
    one_total = int(round_metrics["1"]["total"])
    two_total = int(round_metrics["2"]["total"])
    maximum_superpoly = {
        str(rounds): max(
            int(row["superpoly_monomials"])
            for row in fixtures
            if int(row["rounds"]) == rounds
        )
        for rounds in (1, 2)
    }
    one_max = maximum_superpoly["1"]
    two_max = maximum_superpoly["2"]
    return {
        "round_output_anf_terms": {"1": one_total, "2": two_total},
        "round_fixture_maximum_superpoly_terms": maximum_superpoly,
        "output_anf_growth_r2_over_r1": two_total / one_total,
        "fixture_maximum_growth_r2_over_r1": two_max / max(one_max, 1),
        "sound_r5_sparse_upper_bound_available": False,
        "sparse_route_ready": False,
    }


def evaluate_tensor_boundary_audit(
    config: TensorBoundaryAuditConfig,
    *,
    exact_summary: dict[str, Any],
    fixtures: list[dict[str, Any]],
) -> dict[str, Any]:
    routes = build_boundary_routes(config)
    route_by_name = {row["route"]: row for row in routes}
    required = route_by_name["all_key_all_inactive_offset"]
    sparse = summarize_sparse_evidence(exact_summary, fixtures)
    source_checks = {
        "e53a_decision_is_exact_oracle_ready": exact_summary["gate"]["decision"]
        == E53A_DECISION,
        "e53a_status_is_pass": exact_summary["gate"]["status"] == "pass",
        "e53a_retains_144_symbolic_inputs": exact_summary["provider_manifest"][
            "providers"
        ][0]["variables"]["total"]
        == 144,
        "e53a_rounds_one_and_two_present": set(
            exact_summary["metrics"]["round_monomial_metrics"]
        )
        == {"1", "2"},
        "fixture_rows_cover_rounds_one_and_two": {int(row["rounds"]) for row in fixtures}
        == {1, 2},
    }
    semantic_checks = {
        "inactive_variable_count_is_56": (
            config.plaintext_variables - config.active_dimension == 56
        ),
        "required_boundary_count_is_136": required["retained_variables"] == 136,
        "required_route_is_only_semantic_match": [
            row["route"] for row in routes if row["semantic_match"]
        ]
        == ["all_key_all_inactive_offset"],
        "zero_offset_control_rejected": not route_by_name[
            "all_key_zero_inactive_offset"
        ]["semantic_match"],
        "fixed_assignment_control_rejected": not route_by_name[
            "fixed_key_fixed_offset"
        ]["semantic_match"],
    }
    feasibility_checks = {
        "required_boundary_within_26_variables": required["within_variable_gate"],
        "required_dense_tensor_within_4_gib": required[
            "within_dense_memory_gate"
        ],
        "sound_five_round_sparse_upper_bound_available": sparse[
            "sound_r5_sparse_upper_bound_available"
        ],
    }
    if not all(source_checks.values()) or not all(semantic_checks.values()):
        status = "fail"
        decision = "innovation2_present_r5_transition_tensor_boundary_protocol_invalid"
        action = "repair E53-A source alignment or all-key/all-offset boundary accounting"
    elif not feasibility_checks["required_boundary_within_26_variables"] or not feasibility_checks[
        "required_dense_tensor_within_4_gib"
    ]:
        status = "hold"
        decision = "innovation2_present_r5_transition_tensor_boundary_infeasible"
        action = (
            "skip internal factor-graph construction; run one capped r3 query-cone "
            "sparse-ANF growth gate before closing the current open provider family"
        )
    else:
        status = "pass"
        decision = "innovation2_present_r5_transition_tensor_internal_width_audit_ready"
        action = "construct the PRESENT factor graph and compare deterministic elimination orders"
    metrics = {
        "target_rounds": config.target_rounds,
        "active_dimension": config.active_dimension,
        "inactive_plaintext_variables": config.plaintext_variables
        - config.active_dimension,
        "key_variables": config.key_variables,
        "required_retained_variables": required["retained_variables"],
        "maximum_retained_variables": config.maximum_retained_variables,
        "required_dense_entries": required["dense_entries"],
        "required_one_byte_dense_gib": required["one_byte_dense_bytes"]
        / (1 << 30),
        "required_bitpacked_dense_gib": required["bitpacked_dense_bytes"]
        / (1 << 30),
        "maximum_dense_gib": config.maximum_dense_bytes / (1 << 30),
        "routes": routes,
        "sparse_evidence": sparse,
        "internal_factor_graph_constructed": False,
        "elimination_orders_executed": 0,
    }
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "semantic_checks": semantic_checks,
        "feasibility_checks": feasibility_checks,
        "metrics": metrics,
        "claim_scope": (
            "PRESENT-80 r5 exact full-superpoly retained-boundary feasibility "
            "audit for dense GF(2) tensor contraction; not an internal treewidth "
            "result, five-round label result, neural training, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
            "five_round_subset": False,
            "internal_factor_graph": status == "pass",
            "closed_routes": [
                "dense 136-variable final tensor",
                "zero-offset key coefficient substituted for all offsets",
                "fixed key or fixed offset substituted for universal labels",
                "internal min-fill audit after a failed semantic boundary gate",
                "neural training before strict labels",
            ],
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_transition_tensor_boundary_audit",
            **row,
            "gate_status": status,
            "decision": decision,
            "training_performed": False,
        }
        for row in routes
    ]
    return {
        "gate": gate,
        "metrics": metrics,
        "result_rows": result_rows,
        "boundary_manifest": {
            "run_id": config.run_id,
            "target": "full cube superpoly over key and inactive plaintext variables",
            "routes": routes,
            "selected_route": "all_key_all_inactive_offset",
            "internal_factor_graph_constructed": False,
        },
        "elimination_rows": [
            {
                "status": "skipped",
                "reason": "required final boundary failed before internal order construction",
                "retained_variables": required["retained_variables"],
                "maximum_retained_variables": config.maximum_retained_variables,
            }
        ],
    }


def serializable_config(config: TensorBoundaryAuditConfig) -> dict[str, Any]:
    return asdict(config)
