from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPECTED_SOURCES = {
    "formal_method": (
        "i2_cross_spn_r3_profile_operator_method_synthesis_20260719",
        "pass",
        "innovation2_cross_spn_r3_profile_method_confirmed_skinny_labels_not_ready",
    ),
    "skinny_residual": (
        "i2_skinny64_r5_true_ridge_sparse_residual_readiness_seed0_20260719",
        "hold",
        "innovation2_skinny64_true_ridge_residual_not_ready",
    ),
    "shared_operator": (
        "i2_present_gift_r4_topology_parameterized_shared_profile_operator_attribution_seed0_20260719",
        "hold",
        "innovation2_shared_profile_operator_quality_not_retained",
    ),
    "rectangle_labels": (
        "i2_rectangle80_r4_unit_balance_profile_192_structures_20260719",
        "pass",
        "innovation2_rectangle80_unit_profile_expansion_ready",
    ),
    "rectangle_untyped": (
        "i2_rectangle80_r4_r3_only_profile_operator_attribution_seed0_20260719",
        "hold",
        "innovation2_rectangle80_r3_only_topology_not_attributed",
    ),
    "rectangle_row_mechanism": (
        "i2_rectangle80_row_typed_shift_representation_audit_20260719",
        "pass",
        "innovation2_rectangle80_row_typed_representation_ready",
    ),
    "rectangle_row_operator": (
        "i2_rectangle80_row_typed_shift_operator_readiness_seed0_20260719",
        "hold",
        "innovation2_rectangle80_row_typed_shift_operator_not_ready",
    ),
}


@dataclass(frozen=True)
class NeuralArchitectureBoundaryConfig:
    run_id: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")


def load_gate_sources(roots: dict[str, Path]) -> dict[str, Any]:
    if set(roots) != set(EXPECTED_SOURCES):
        raise ValueError("E93 requires exactly the seven frozen source roles")
    sources: dict[str, Any] = {}
    for role, root in roots.items():
        gate_path = root / "gate.json"
        sources[role] = {
            "root": str(root),
            "gate": json.loads(gate_path.read_text(encoding="utf-8")),
            "sha256": _sha256(gate_path),
        }
    return sources


def validate_gate_sources(sources: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for role, (run_id, status, decision) in EXPECTED_SOURCES.items():
        source = sources.get(role, {})
        gate = source.get("gate", {})
        checks[f"{role}_run_id_matches"] = gate.get("run_id") == run_id
        checks[f"{role}_status_matches"] = gate.get("status") == status
        checks[f"{role}_decision_matches"] = gate.get("decision") == decision
        checks[f"{role}_hash_present"] = len(source.get("sha256", "")) == 64
    gates = {role: sources[role]["gate"] for role in EXPECTED_SOURCES}
    checks.update(
        {
            "formal_method_internal_checks_pass": all(
                all(gates["formal_method"].get(name, {}).values())
                for name in ("source_checks", "method_checks", "skinny_readiness_checks")
            ),
            "skinny_protocol_passes": all(
                gates["skinny_residual"].get("protocol_checks", {}).values()
            ),
            "skinny_readiness_has_failure": not all(
                gates["skinny_residual"].get("readiness_checks", {}).values()
            ),
            "shared_protocol_and_relation_pass": all(
                gates["shared_operator"].get("protocol_checks", {}).values()
            )
            and all(gates["shared_operator"].get("relation_checks", {}).values()),
            "shared_candidate_has_failure": not all(
                gates["shared_operator"].get("candidate_checks", {}).values()
            ),
            "rectangle_label_all_checks_pass": all(
                all(gates["rectangle_labels"].get(name, {}).values())
                for name in (
                    "protocol_checks",
                    "raw_width_checks",
                    "matching_width_checks",
                    "shortcut_checks",
                )
            ),
            "rectangle_untyped_protocol_candidate_pass": all(
                gates["rectangle_untyped"].get("protocol_checks", {}).values()
            )
            and all(gates["rectangle_untyped"].get("candidate_checks", {}).values()),
            "rectangle_untyped_relation_has_failure": not all(
                gates["rectangle_untyped"].get("relation_checks", {}).values()
            ),
            "rectangle_row_mechanism_all_checks_pass": all(
                gates["rectangle_row_mechanism"].get("protocol_checks", {}).values()
            )
            and all(
                gates["rectangle_row_mechanism"].get("mechanism_checks", {}).values()
            ),
            "rectangle_row_operator_protocol_passes": all(
                gates["rectangle_row_operator"].get("protocol_checks", {}).values()
            ),
            "rectangle_row_operator_readiness_has_failure": not all(
                gates["rectangle_row_operator"].get("readiness_checks", {}).values()
            ),
        }
    )
    return checks


def architecture_rows(sources: dict[str, Any]) -> list[dict[str, Any]]:
    gates = {role: source["gate"] for role, source in sources.items()}
    formal_ciphers = gates["formal_method"]["metrics"]["ciphers"]
    present = next(row for row in formal_ciphers if row["cipher"] == "PRESENT-80")
    gift = next(row for row in formal_ciphers if row["cipher"] == "GIFT-64")
    rows = [
        {
            "rank": 1,
            "route": "separate_r3_only_profile_operator",
            "scope": "PRESENT-80 + GIFT-64",
            "evidence_class": "formal_confirmed",
            "status": "keep",
            "key_metric": (
                f"PRESENT mean true-wrong={present['mean_true_minus_corrupted']:.6f}; "
                f"GIFT={gift['mean_true_minus_corrupted']:.6f}"
            ),
            "next_action": "retain as the formal Innovation 2 neural method",
        },
        {
            "rank": 2,
            "route": "rectangle_untyped_r3_only_operator",
            "scope": "RECTANGLE-80",
            "evidence_class": "mechanism_only",
            "status": "hold",
            "key_metric": (
                "30-epoch true-wrong="
                f"{gates['rectangle_untyped']['metrics']['true_minus_corrupted']:.6f}"
            ),
            "next_action": "retain as a near-boundary diagnostic; no seed1",
        },
        {
            "rank": 3,
            "route": "rectangle_row_typed_representation",
            "scope": "RECTANGLE-80 deterministic representation",
            "evidence_class": "mechanism_only",
            "status": "keep_mechanism",
            "key_metric": (
                "typed-untyped="
                f"{gates['rectangle_row_mechanism']['metrics']['typed_true_minus_untyped_true']:.6f}"
            ),
            "next_action": "retain as an explanatory result, not neural gain",
        },
        {
            "rank": 4,
            "route": "rectangle_row_typed_shift_operator",
            "scope": "RECTANGLE-80",
            "evidence_class": "closed",
            "status": "discard",
            "key_metric": (
                "typed-untyped="
                f"{gates['rectangle_row_operator']['metrics']['typed_true_minus_untyped']:.6f}; "
                "typed-wrong-row="
                f"{gates['rectangle_row_operator']['metrics']['typed_true_minus_wrong_row']:.6f}"
            ),
            "next_action": "no 30-epoch run or row-embedding tuning",
        },
        {
            "rank": 5,
            "route": "shared_topology_parameterized_operator",
            "scope": "PRESENT-80 + GIFT-64 one checkpoint",
            "evidence_class": "closed",
            "status": "discard",
            "key_metric": (
                "GIFT quality delta="
                f"{gates['shared_operator']['metrics']['gift_true_minus_anchor']:.6f}"
            ),
            "next_action": "retain separate cipher models; no adapters or reweighting",
        },
        {
            "rank": 6,
            "route": "skinny_true_ridge_sparse_residual",
            "scope": "SKINNY-64",
            "evidence_class": "closed",
            "status": "discard",
            "key_metric": (
                "true-independent="
                f"{gates['skinny_residual']['metrics']['true_minus_independent']:.6f}; "
                "true-corrupted="
                f"{gates['skinny_residual']['metrics']['true_minus_corrupted']:.6f}"
            ),
            "next_action": "retain strict labels and deterministic ridge only",
        },
        {
            "rank": 7,
            "route": "generic_transformer_graphgps_nbfnet_variants",
            "scope": "current strict unit-profile benchmarks",
            "evidence_class": "deferred",
            "status": "no_budget",
            "key_metric": "no new sound label or independent mechanism gate",
            "next_action": "do not train until a new pre-neural gate passes",
        },
    ]
    plot_margins = {
        "separate_r3_only_profile_operator": (
            present["mean_true_minus_corrupted"],
            gift["mean_true_minus_corrupted"],
        ),
        "rectangle_untyped_r3_only_operator": (
            gates["rectangle_untyped"]["metrics"]["true_minus_corrupted"],
            gates["rectangle_untyped"]["metrics"]["true_minus_independent"],
        ),
        "rectangle_row_typed_representation": (
            gates["rectangle_row_mechanism"]["metrics"][
                "typed_true_minus_wrong_row_typed"
            ],
            gates["rectangle_row_mechanism"]["metrics"][
                "typed_true_minus_untyped_true"
            ],
        ),
        "rectangle_row_typed_shift_operator": (
            gates["rectangle_row_operator"]["metrics"][
                "typed_true_minus_wrong_row"
            ],
            gates["rectangle_row_operator"]["metrics"]["typed_true_minus_untyped"],
        ),
        "shared_topology_parameterized_operator": (
            gates["shared_operator"]["metrics"]["gift_true_minus_anchor"],
            gates["shared_operator"]["metrics"]["present_true_minus_anchor"],
        ),
        "skinny_true_ridge_sparse_residual": (
            gates["skinny_residual"]["metrics"]["true_minus_independent"],
            gates["skinny_residual"]["metrics"]["true_minus_corrupted"],
        ),
        "generic_transformer_graphgps_nbfnet_variants": (None, None),
    }
    for row in rows:
        row["primary_margin"], row["secondary_margin"] = plot_margins[row["route"]]
    return rows


def adjudicate_boundary_synthesis(
    config: NeuralArchitectureBoundaryConfig,
    source_checks: dict[str, bool],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    formal = [row for row in rows if row["evidence_class"] == "formal_confirmed"]
    third_formal = any(
        row["evidence_class"] == "formal_confirmed"
        and "RECTANGLE" in row["scope"]
        for row in rows
    )
    ranking_checks = {
        "one_formal_method_family_present": len(formal) == 1,
        "formal_method_is_present_gift_separate": bool(formal)
        and formal[0]["route"] == "separate_r3_only_profile_operator",
        "third_spn_formal_neural_absent": not third_formal,
        "closed_routes_are_not_budgeted": all(
            row["status"] == "discard"
            for row in rows
            if row["evidence_class"] == "closed"
        ),
        "generic_architectures_deferred": rows[-1]["evidence_class"] == "deferred"
        and rows[-1]["status"] == "no_budget",
    }
    if not all(source_checks.values()) or not all(ranking_checks.values()):
        status = "fail"
        decision = "innovation2_architecture_boundary_synthesis_protocol_invalid"
        action = "repair frozen source replay or evidence-class ranking"
    else:
        status = "pass"
        decision = (
            "innovation2_architecture_boundary_confirmed_"
            "third_spn_neural_not_confirmed"
        )
        action = (
            "prioritize a new sound label/task or independent pre-neural mechanism; "
            "stop current benchmark architecture enumeration"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "ranking_checks": ranking_checks,
        "metrics": {
            "architecture_rows": rows,
            "formal_method_family_count": len(formal),
            "formal_real_spn_count": 2,
            "third_spn_strict_labels_ready": True,
            "third_spn_formal_neural_confirmed": False,
            "closed_route_count": sum(
                row["evidence_class"] == "closed" for row in rows
            ),
        },
        "claim_scope": (
            "no-training Innovation 2 architecture evidence synthesis across "
            "PRESENT, GIFT, SKINNY, and RECTANGLE; no new neural gain or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
            "reopen_condition": (
                "new sound label family, independent task mechanism, or >=0.03 "
                "same-capacity pre-neural topology margin"
            ),
        },
    }


def source_hashes(sources: dict[str, Any]) -> dict[str, Any]:
    return {
        role: {
            "root": source["root"],
            "gate_sha256": source["sha256"],
            "run_id": source["gate"].get("run_id"),
        }
        for role, source in sources.items()
    }


def serializable_config(
    config: NeuralArchitectureBoundaryConfig,
) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "NeuralArchitectureBoundaryConfig",
    "adjudicate_boundary_synthesis",
    "architecture_rows",
    "load_gate_sources",
    "serializable_config",
    "source_hashes",
    "validate_gate_sources",
]
