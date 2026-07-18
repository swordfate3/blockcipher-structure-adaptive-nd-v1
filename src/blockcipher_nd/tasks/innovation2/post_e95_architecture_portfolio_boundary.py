from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RUN_ID = "i2_post_e95_architecture_portfolio_boundary_20260719"


@dataclass(frozen=True)
class SourceSpec:
    role: str
    relative_root: str
    run_id: str
    status: str
    decision: str
    sha256: str


SOURCE_SPECS = (
    SourceSpec(
        "multibit_mask",
        "local_audits/i2_present_r4_multibit_mask_profile_readiness_20260718",
        "i2_present_r4_multibit_mask_profile_readiness_20260718",
        "hold",
        "innovation2_present_multibit_profile_componentwise_dominated",
        "b1a69ab4e6c5c3c335432f3f64b1ad14492d4c8b3185bed9c63d7a9aafdd707d",
    ),
    SourceSpec(
        "active_dimension",
        "local_audits/i2_present_r4_active_dimension_zero_shot_transfer_20260718",
        "i2_present_r4_active_dimension_zero_shot_transfer_20260718",
        "hold",
        "innovation2_present_active_dimension_transfer_labels_not_ready",
        "54d8f5b1b2c409a064f7050b6e4ea30a861f9e75f0816931923b6fde46e5521c",
    ),
    SourceSpec(
        "formal_method",
        "local_audits/i2_cross_spn_r3_profile_operator_method_synthesis_20260719",
        "i2_cross_spn_r3_profile_operator_method_synthesis_20260719",
        "pass",
        "innovation2_cross_spn_r3_profile_method_confirmed_skinny_labels_not_ready",
        "95a949127e0af2721a3b7bdcda25ba5f6dd592473e674c2d8da0225340aaa8f6",
    ),
    SourceSpec(
        "skinny_residual",
        "local_smoke/i2_skinny64_r5_true_ridge_sparse_residual_readiness_seed0_20260719",
        "i2_skinny64_r5_true_ridge_sparse_residual_readiness_seed0_20260719",
        "hold",
        "innovation2_skinny64_true_ridge_residual_not_ready",
        "3b2593c20e13478d2ff570dc0d83e3be1a87791179189a10d91b6ec8456379b2",
    ),
    SourceSpec(
        "shared_operator",
        "local_diagnostic/i2_present_gift_r4_topology_parameterized_shared_profile_operator_attribution_seed0_20260719",
        "i2_present_gift_r4_topology_parameterized_shared_profile_operator_attribution_seed0_20260719",
        "hold",
        "innovation2_shared_profile_operator_quality_not_retained",
        "a39fd0a0c9d9ac0ceb721f64ab189a5fc26bcfcb893b58b284540e6492dfd600",
    ),
    SourceSpec(
        "rectangle_untyped",
        "local_diagnostic/i2_rectangle80_r4_r3_only_profile_operator_attribution_seed0_20260719",
        "i2_rectangle80_r4_r3_only_profile_operator_attribution_seed0_20260719",
        "hold",
        "innovation2_rectangle80_r3_only_topology_not_attributed",
        "3aab7d2f8bc2cb347194ec8cc8bde0e14d98ff669f51c4d65bcc95ad8ba75b2f",
    ),
    SourceSpec(
        "rectangle_row_operator",
        "local_smoke/i2_rectangle80_row_typed_shift_operator_readiness_seed0_20260719",
        "i2_rectangle80_row_typed_shift_operator_readiness_seed0_20260719",
        "hold",
        "innovation2_rectangle80_row_typed_shift_operator_not_ready",
        "72bed4c51600076bf6b7fd2c5ac9e2525fd64a0ed9cb31e0029d163fda3405bd",
    ),
    SourceSpec(
        "e93_boundary",
        "local_audits/i2_neural_architecture_boundary_synthesis_20260719",
        "i2_neural_architecture_boundary_synthesis_20260719",
        "pass",
        "innovation2_architecture_boundary_confirmed_third_spn_neural_not_confirmed",
        "a5fe825b87e0e208596e0b17558283e61c116b2eadaf1a773d3eefd36a757bcf",
    ),
    SourceSpec(
        "nested_labels",
        "local_audits/i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719",
        "i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719",
        "pass",
        "innovation2_rectangle80_nested_cube_monotonic_labels_ready",
        "323c7bada04941e79dc932ab1c562cab977b0288039632557cf1556b41c9f4f9",
    ),
    SourceSpec(
        "nested_relation",
        "local_audits/i2_rectangle80_r4_nested_cube_relation_mechanism_20260719",
        "i2_rectangle80_r4_nested_cube_relation_mechanism_20260719",
        "hold",
        "innovation2_rectangle80_nested_cube_relation_not_attributed",
        "af54d0ab853b8b32032cb77753acc6871c7a56e6e4e90a9d7c3bc80991e07507",
    ),
)


@dataclass(frozen=True)
class ArchitecturePortfolioConfig:
    run_id: str = RUN_ID

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E96 run_id is frozen")


def load_portfolio_sources(outputs_root: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for spec in SOURCE_SPECS:
        gate_path = outputs_root / spec.relative_root / "gate.json"
        if not gate_path.is_file():
            raise FileNotFoundError(f"missing E96 source: {gate_path}")
        sources[spec.role] = {
            "root": str(gate_path.parent),
            "gate": json.loads(gate_path.read_text(encoding="utf-8")),
            "sha256": _sha256(gate_path),
        }
    return sources


def validate_portfolio_sources(sources: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "exactly_ten_sources_present": set(sources)
        == {spec.role for spec in SOURCE_SPECS}
    }
    specs = {spec.role: spec for spec in SOURCE_SPECS}
    for role, spec in specs.items():
        gate = sources.get(role, {}).get("gate", {})
        checks[f"{role}_run_id_matches"] = gate.get("run_id") == spec.run_id
        checks[f"{role}_status_matches"] = gate.get("status") == spec.status
        checks[f"{role}_decision_matches"] = gate.get("decision") == spec.decision
        checks[f"{role}_hash_matches"] = sources.get(role, {}).get("sha256") == spec.sha256

    gates = {role: sources[role]["gate"] for role in specs}
    checks.update(
        {
            "multibit_protocol_passes": _all(gates["multibit_mask"], "protocol_checks"),
            "multibit_nontrivial_gate_fails": not _all(
                gates["multibit_mask"], "nontrivial_checks"
            ),
            "active_dimension_source_passes": _all(
                gates["active_dimension"], "source_checks"
            ),
            "active_dimension_label_gate_fails": not _all(
                gates["active_dimension"], "label_checks"
            ),
            "formal_method_internal_checks_pass": all(
                _all(gates["formal_method"], group)
                for group in ("source_checks", "method_checks", "skinny_readiness_checks")
            ),
            "skinny_protocol_passes": _all(gates["skinny_residual"], "protocol_checks"),
            "skinny_readiness_fails": not _all(
                gates["skinny_residual"], "readiness_checks"
            ),
            "shared_protocol_relation_pass": _all(
                gates["shared_operator"], "protocol_checks"
            )
            and _all(gates["shared_operator"], "relation_checks"),
            "shared_candidate_fails": not _all(
                gates["shared_operator"], "candidate_checks"
            ),
            "rectangle_untyped_protocol_candidate_pass": _all(
                gates["rectangle_untyped"], "protocol_checks"
            )
            and _all(gates["rectangle_untyped"], "candidate_checks"),
            "rectangle_untyped_relation_fails": not _all(
                gates["rectangle_untyped"], "relation_checks"
            ),
            "rectangle_row_protocol_passes": _all(
                gates["rectangle_row_operator"], "protocol_checks"
            ),
            "rectangle_row_readiness_fails": not _all(
                gates["rectangle_row_operator"], "readiness_checks"
            ),
            "e93_source_ranking_pass": _all(gates["e93_boundary"], "source_checks")
            and _all(gates["e93_boundary"], "ranking_checks"),
            "nested_label_all_checks_pass": all(
                _all(gates["nested_labels"], group)
                for group in (
                    "protocol_checks",
                    "monotonic_checks",
                    "width_checks",
                    "shortcut_checks",
                )
            ),
            "nested_relation_protocol_passes": _all(
                gates["nested_relation"], "protocol_checks"
            ),
            "nested_relation_quality_or_attribution_fails": not (
                _all(gates["nested_relation"], "quality_checks")
                and _all(gates["nested_relation"], "attribution_checks")
            ),
        }
    )
    return checks


def build_architecture_portfolio(sources: dict[str, Any]) -> list[dict[str, Any]]:
    formal_ciphers = sources["formal_method"]["gate"]["metrics"]["ciphers"]
    present = next(row for row in formal_ciphers if row["cipher"] == "PRESENT-80")
    gift = next(row for row in formal_ciphers if row["cipher"] == "GIFT-64")
    nested_margin = sources["nested_relation"]["gate"]["metrics"]["margins"]
    return [
        {
            "rank": 1,
            "route": "separate_r3_only_profile_operator",
            "display_name": "PRESENT/GIFT独立r3-only Profile Operator",
            "evidence_class": "formal_confirmed",
            "label_ready": True,
            "mechanism_ready": True,
            "formal_neural": True,
            "training_budget": False,
            "key_evidence": (
                f"PRESENT true-wrong={present['mean_true_minus_corrupted']:.6f}; "
                f"GIFT={gift['mean_true_minus_corrupted']:.6f}"
            ),
            "unlock_condition": "none; retain and write the confirmed method",
        },
        {
            "rank": 2,
            "route": "rectangle_monotone_cube_lattice_operator",
            "display_name": "RECTANGLE单调Cube-Lattice Operator",
            "evidence_class": "label_ready_but_unattributed",
            "label_ready": True,
            "mechanism_ready": False,
            "formal_neural": False,
            "training_budget": False,
            "key_evidence": (
                "true-independent="
                f"{nested_margin['true_minus_independent']:.6f}; "
                "true-corrupted="
                f"{nested_margin['true_minus_shuffled']:.6f}"
            ),
            "unlock_condition": "new independent mechanism; do not tune E95",
        },
        {
            "rank": 3,
            "route": "cancellation_aware_mask_query_hypergraph",
            "display_name": "消去感知Mask-Query Hypergraph",
            "evidence_class": "provider_missing",
            "label_ready": False,
            "mechanism_ready": False,
            "formal_neural": False,
            "training_budget": False,
            "key_evidence": "E69 nontrivial positive=0; componentwise AUC=1.0",
            "unlock_condition": "sound nontrivial GF(2)-cancellation positive provider",
        },
        {
            "rank": 4,
            "route": "active_dimension_conditioned_profile_operator",
            "display_name": "活动维度条件Profile Operator",
            "evidence_class": "provider_missing",
            "label_ready": False,
            "mechanism_ready": False,
            "formal_neural": False,
            "training_budget": False,
            "key_evidence": "E70 d4 all unknown; d12 support cap before labels",
            "unlock_condition": "sound adjacent-dimension labels without raising frozen cap",
        },
        {
            "rank": 5,
            "route": "rectangle_typed_and_untyped_profile_operator",
            "display_name": "RECTANGLE Row-Typed/原r3-only算子",
            "evidence_class": "mechanism_only_closed",
            "label_ready": True,
            "mechanism_ready": True,
            "formal_neural": False,
            "training_budget": False,
            "key_evidence": "E90 true-wrong=0.029646; E92 row controls fail",
            "unlock_condition": "new task semantics, not row/model tuning",
        },
        {
            "rank": 6,
            "route": "skinny_true_ridge_sparse_residual",
            "display_name": "SKINNY true-ridge稀疏残差",
            "evidence_class": "closed",
            "label_ready": True,
            "mechanism_ready": False,
            "formal_neural": False,
            "training_budget": False,
            "key_evidence": "E84 true-independent=-0.000457",
            "unlock_condition": "new sound task; current residual remains closed",
        },
        {
            "rank": 7,
            "route": "shared_topology_parameterized_operator",
            "display_name": "PRESENT/GIFT共享拓扑参数算子",
            "evidence_class": "closed",
            "label_ready": True,
            "mechanism_ready": True,
            "formal_neural": False,
            "training_budget": False,
            "key_evidence": "E86 GIFT anchor delta=-0.053590",
            "unlock_condition": "none; retain separate models",
        },
        {
            "rank": 8,
            "route": "generic_transformer_graphgps_nbfnet",
            "display_name": "Transformer/GraphGPS/NBFNet通用变体",
            "evidence_class": "deferred_no_budget",
            "label_ready": False,
            "mechanism_ready": False,
            "formal_neural": False,
            "training_budget": False,
            "key_evidence": "no new sound label or independent mechanism",
            "unlock_condition": "pass label and pre-neural mechanism gates first",
        },
    ]


def adjudicate_architecture_portfolio(
    config: ArchitecturePortfolioConfig,
    source_checks: dict[str, bool],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {
        evidence_class: sum(row["evidence_class"] == evidence_class for row in rows)
        for evidence_class in (
            "formal_confirmed",
            "label_ready_but_unattributed",
            "provider_missing",
            "mechanism_only_closed",
            "closed",
            "deferred_no_budget",
        )
    }
    trainable = [row for row in rows if row["training_budget"]]
    portfolio_checks = {
        "exactly_eight_candidates": len(rows) == 8,
        "one_formal_method_family": counts["formal_confirmed"] == 1,
        "formal_method_is_present_gift_separate": rows[0]["route"]
        == "separate_r3_only_profile_operator",
        "two_provider_missing_routes": counts["provider_missing"] == 2,
        "one_label_ready_but_unattributed": counts[
            "label_ready_but_unattributed"
        ]
        == 1,
        "training_budget_requires_all_pre_neural_gates": all(
            not row["training_budget"]
            or (
                row["label_ready"]
                and row["mechanism_ready"]
                and not row["formal_neural"]
            )
            for row in rows
        ),
        "closed_provider_and_deferred_routes_have_no_budget": all(
            not row["training_budget"]
            for row in rows
            if row["evidence_class"]
            in {
                "provider_missing",
                "mechanism_only_closed",
                "closed",
                "deferred_no_budget",
            }
        ),
        "generic_variants_remain_deferred": rows[-1]["evidence_class"]
        == "deferred_no_budget",
    }
    if not all(source_checks.values()) or not all(portfolio_checks.values()):
        status = "fail"
        decision = "innovation2_architecture_portfolio_protocol_invalid"
        action = "repair frozen source replay or candidate classification"
    elif trainable:
        status = "pass"
        decision = "innovation2_architecture_portfolio_new_candidate_ready"
        action = "pre-register only the highest-ranked trainable candidate"
    else:
        status = "pass"
        decision = "innovation2_architecture_portfolio_converged_no_new_training_budget"
        action = (
            "stop architecture enumeration; choose provider research or thesis "
            "consolidation"
        )
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "portfolio_checks": portfolio_checks,
        "metrics": {
            "architecture_rows": rows,
            "evidence_class_counts": counts,
            "formal_method_family_count": counts["formal_confirmed"],
            "formal_real_spn_count": 2,
            "immediately_trainable_candidate_count": len(trainable),
            "provider_missing_candidate_count": counts["provider_missing"],
        },
        "claim_scope": (
            "no-training post-E95 Innovation 2 architecture portfolio boundary; "
            "no new neural gain, high-round distinguisher, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
            "allowed_routes": [
                "new sound nontrivial label-provider research",
                "PRESENT/GIFT formal-method thesis consolidation",
            ],
            "forbidden_routes": [
                row["route"]
                for row in rows
                if row["evidence_class"] != "formal_confirmed"
            ],
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


def serializable_config(config: ArchitecturePortfolioConfig) -> dict[str, Any]:
    return asdict(config)


def _all(gate: dict[str, Any], group: str) -> bool:
    values = gate.get(group, {})
    return bool(values) and all(bool(value) for value in values.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ArchitecturePortfolioConfig",
    "RUN_ID",
    "SOURCE_SPECS",
    "SourceSpec",
    "adjudicate_architecture_portfolio",
    "build_architecture_portfolio",
    "load_portfolio_sources",
    "serializable_config",
    "source_hashes",
    "validate_portfolio_sources",
]
