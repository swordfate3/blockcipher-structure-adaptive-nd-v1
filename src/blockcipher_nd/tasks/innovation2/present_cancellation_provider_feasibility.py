from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RUN_ID = "i2_present_r5_cancellation_provider_feasibility_20260719"
PANEL_STRUCTURE_INDICES = (0, 4, 8)
PANEL_MASK_FAMILIES = (
    "nibble",
    "player_pair",
    "same_nibble_pair",
    "adjacent_nibble_pair",
)
EXPECTED_PANEL_KEYS = (
    (0, 64),
    (0, 80),
    (0, 140),
    (0, 238),
    (4, 64),
    (4, 80),
    (4, 140),
    (4, 236),
    (8, 64),
    (8, 80),
    (8, 140),
    (8, 236),
)


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
        "e52_strict_bank",
        "local_audits/i2_present_r5_strict_label_provider_coverage_20260718",
        "i2_present_r5_strict_label_provider_coverage_20260718",
        "hold",
        "innovation2_present_r5_strict_label_bank_not_ready",
        "557a61f1c00cefea994ab56d1f1cab9ab1c6e9762eac77958220de57e9fd03fe",
    ),
    SourceSpec(
        "e53a_exact_oracle",
        "local_audits/i2_present_r5_open_3sdp_exact_anf_phase_a_20260718",
        "i2_present_r5_open_3sdp_exact_anf_phase_a_20260718",
        "pass",
        "innovation2_present_r5_open_3sdp_exact_oracle_ready",
        "779c0107cc77fe66ed4eb3d0a2f7ab45a44a5a7f8d497f3931f7bc9e985c7181",
    ),
    SourceSpec(
        "e53b_glpk",
        "local_audits/i2_present_r5_open_3sdp_glpk_blocking_gate_20260718",
        "i2_present_r5_open_3sdp_glpk_blocking_gate_20260718",
        "hold",
        "innovation2_present_r5_open_3sdp_glpk_blocking_not_scalable",
        "2a0b35cbdb234d9fcb2225beabc0245ddfe571a8c1c5c3dd9596702a86c24f5d",
    ),
    SourceSpec(
        "e54_tensor",
        "local_audits/i2_present_r5_transition_tensor_boundary_audit_20260718",
        "i2_present_r5_transition_tensor_boundary_audit_20260718",
        "hold",
        "innovation2_present_r5_transition_tensor_boundary_infeasible",
        "6d7504b71247adc4b9efec025339a2d219324a11232313758fd974d8d9c52c14",
    ),
    SourceSpec(
        "e55_sparse",
        "local_audits/i2_present_r3_query_cone_sparse_anf_growth_20260718",
        "i2_present_r3_query_cone_sparse_anf_growth_20260718",
        "hold",
        "innovation2_present_r3_query_cone_sparse_anf_hard_cap_exceeded",
        "7aca10e7f9aa9a08ffa500fba30256be57584dd6f2b57d1be85ad48f5ffbff08",
    ),
    SourceSpec(
        "e61_atm",
        "local_audits/i2_present_r2_atm_multicoordinate_support_phase_a_20260718",
        "i2_present_r2_atm_multicoordinate_support_phase_a_20260718",
        "hold",
        "innovation2_atm_r2_multicoordinate_support_runtime_not_ready",
        "ebb25a0ce10072d6facb94c49cb7e5a04e502f0610cf0e9323456f299e7c4a12",
    ),
    SourceSpec(
        "e64_small_spn",
        "local_audits/i2_small_spn_relation_decomposition_20260718",
        "i2_small_spn_relation_decomposition_20260718",
        "hold",
        "innovation2_small_spn_relation_nontrivial_width_not_ready",
        "b0d8ba5c0cd498f20ec85a1ca4567bd7fec11ca8b0511cb5cfb14025c90b859b",
    ),
    SourceSpec(
        "e69_multibit",
        "local_audits/i2_present_r4_multibit_mask_profile_readiness_20260718",
        "i2_present_r4_multibit_mask_profile_readiness_20260718",
        "hold",
        "innovation2_present_multibit_profile_componentwise_dominated",
        "b1a69ab4e6c5c3c335432f3f64b1ad14492d4c8b3185bed9c63d7a9aafdd707d",
    ),
)


@dataclass(frozen=True)
class CancellationProviderConfig:
    run_id: str = RUN_ID

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ValueError("E97 run_id is frozen")


def load_sources(outputs_root: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for spec in SOURCE_SPECS:
        root = outputs_root / spec.relative_root
        gate_path = root / "gate.json"
        if not gate_path.is_file():
            raise FileNotFoundError(f"missing E97 source: {gate_path}")
        sources[spec.role] = {
            "root": str(root),
            "gate": json.loads(gate_path.read_text(encoding="utf-8")),
            "sha256": _sha256(gate_path),
        }
    return sources


def validate_sources(sources: dict[str, dict[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "exactly_eight_sources_present": set(sources)
        == {spec.role for spec in SOURCE_SPECS}
    }
    for spec in SOURCE_SPECS:
        source = sources.get(spec.role, {})
        gate = source.get("gate", {})
        checks[f"{spec.role}_run_id_matches"] = gate.get("run_id") == spec.run_id
        checks[f"{spec.role}_status_matches"] = gate.get("status") == spec.status
        checks[f"{spec.role}_decision_matches"] = gate.get("decision") == spec.decision
        checks[f"{spec.role}_hash_matches"] = source.get("sha256") == spec.sha256
    return checks


def select_frozen_panel(labels_path: Path) -> list[dict[str, Any]]:
    with labels_path.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    panel: list[dict[str, Any]] = []
    for structure_index in PANEL_STRUCTURE_INDICES:
        for family in PANEL_MASK_FAMILIES:
            candidates = sorted(
                (
                    row
                    for row in labels
                    if int(row["structure_index"]) == structure_index
                    and row["mask_family"] == family
                    and row["status"] == "unknown"
                ),
                key=lambda row: int(row["mask_index"]),
            )
            if not candidates:
                raise ValueError(
                    f"no frozen unknown for structure={structure_index}, family={family}"
                )
            selected = candidates[0]
            panel.append(
                {
                    "query_id": f"{selected['structure_id']}__{selected['mask_id']}",
                    "rounds": 5,
                    "structure_index": int(selected["structure_index"]),
                    "structure_id": selected["structure_id"],
                    "structure_role": selected["structure_role"],
                    "active_mask_hex": selected["active_mask_hex"],
                    "mask_index": int(selected["mask_index"]),
                    "mask_id": selected["mask_id"],
                    "mask_family": selected["mask_family"],
                    "mask_hex": selected["mask_hex"],
                    "mask_weight": int(selected["mask_weight"]),
                    "source_status": selected["status"],
                    "e97_status": "unresolved",
                    "strict_positive_certificate": None,
                    "finite_key_voting": False,
                }
            )
    actual = tuple((row["structure_index"], row["mask_index"]) for row in panel)
    if actual != EXPECTED_PANEL_KEYS:
        raise ValueError(f"E97 frozen query panel drifted: {actual}")
    return panel


def build_provider_portfolio(
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gates = {role: source["gate"] for role, source in sources.items()}
    e52 = gates["e52_strict_bank"]
    e53a = gates["e53a_exact_oracle"]
    e53b = gates["e53b_glpk"]
    e54 = gates["e54_tensor"]
    e55 = gates["e55_sparse"]
    e61 = gates["e61_atm"]
    e64 = gates["e64_small_spn"]
    e69 = gates["e69_multibit"]
    small_spn_nontrivial = sum(
        int(metrics["nontrivial_positive"])
        for metrics in e64["metrics"]["split_metrics"].values()
    )
    return [
        {
            "provider_id": "p0_support_absence",
            "display_name": "P0支撑缺失",
            "target_semantics_match": True,
            "sound_certificate": bool(
                e52["correctness_checks"]["all_positive_rows_have_sound_certificate"]
            ),
            "currently_executable": bool(e52["provider_checks"]["p0_completed"]),
            "cancellation_aware": False,
            "within_frozen_cap": True,
            "real_present_r5": True,
            "finite_key_voting": False,
            "nontrivial_positive_certificates": 0,
            "panel_resolved": 0,
            "limitation": "可执行且sound，但支撑饱和；E52正类为0且不能证明trail相消",
        },
        {
            "provider_id": "full_superpoly_sparse_anf",
            "display_name": "全superpoly稀疏ANF",
            "target_semantics_match": bool(
                e54["semantic_checks"]["required_route_is_only_semantic_match"]
            ),
            "sound_certificate": bool(
                e53a["fixture_checks"]["all_multi_mask_component_xors_match"]
                and e55["calibration_checks"]["all_superpolies_match_e53a"]
            ),
            "currently_executable": bool(e55["execution_checks"]["all_12_queries_completed"]),
            "cancellation_aware": True,
            "within_frozen_cap": bool(e55["execution_checks"]["no_query_exceeded_hard_cap"]),
            "real_present_r5": True,
            "finite_key_voting": False,
            "nontrivial_positive_certificates": 0,
            "panel_resolved": 0,
            "limitation": "语义匹配，但三轮第4个query已越过500万项硬cap",
        },
        {
            "provider_id": "glpk_per_solution_3sdp",
            "display_name": "GLPK逐解3SDP",
            "target_semantics_match": bool(e53b["next_action"]["five_round_subset"]),
            "sound_certificate": bool(
                e53b["correctness_checks"]["required_parity_matches_exact"]
            ),
            "currently_executable": bool(
                e53b["scalability_checks"][
                    "all_representative_queries_complete_within_budget"
                ]
            ),
            "cancellation_aware": True,
            "within_frozen_cap": bool(
                e53b["scalability_checks"]["heaviest_query_completes_within_budget"]
            ),
            "real_present_r5": False,
            "finite_key_voting": False,
            "nontrivial_positive_certificates": 0,
            "panel_resolved": 0,
            "limitation": "S-box重查询已超时，尚无五轮full-cipher provider",
        },
        {
            "provider_id": "atm_key_support_cancellation",
            "display_name": "ATM key-support消去",
            "target_semantics_match": False,
            "sound_certificate": bool(
                e61["readiness_checks"]["all_saved_odd_masks_replay_odd"]
            ),
            "currently_executable": e61["metrics"]["worker_status"] == "completed",
            "cancellation_aware": True,
            "within_frozen_cap": e61["metrics"]["worker_status"] == "completed",
            "real_present_r5": False,
            "finite_key_voting": False,
            "nontrivial_positive_certificates": int(
                e61["metrics"]["low_weight_positive_relations"]
            ),
            "panel_resolved": 0,
            "limitation": "仅两轮独立轮密钥；60秒只完成8/240且没有positive relation",
        },
        {
            "provider_id": "small_spn_exact_enumeration",
            "display_name": "小型SPN完整枚举",
            "target_semantics_match": False,
            "sound_certificate": all(e64["source_checks"].values()),
            "currently_executable": True,
            "cancellation_aware": True,
            "within_frozen_cap": True,
            "real_present_r5": False,
            "finite_key_voting": False,
            "nontrivial_positive_certificates": small_spn_nontrivial,
            "panel_resolved": 0,
            "limitation": "存在244条非平凡正类，但对象是16-bit合成SPN且宽度被singleton主导",
        },
        {
            "provider_id": "componentwise_multibit",
            "display_name": "多bit逐分量组合",
            "target_semantics_match": False,
            "sound_certificate": all(e69["protocol_checks"].values()),
            "currently_executable": True,
            "cancellation_aware": False,
            "within_frozen_cap": True,
            "real_present_r5": False,
            "finite_key_voting": False,
            "nontrivial_positive_certificates": int(
                e69["metrics"]["decomposition_reports"]["combined"][
                    "raw_nontrivial_positive"
                ]
            ),
            "panel_resolved": 0,
            "limitation": "PRESENT r4正类全部可由unit状态组合解释；非平凡正类为0",
        },
    ]


def adjudicate_provider_feasibility(
    config: CancellationProviderConfig,
    source_checks: dict[str, bool],
    panel: list[dict[str, Any]],
    providers: list[dict[str, Any]],
) -> dict[str, Any]:
    panel_checks = {
        "exactly_12_queries": len(panel) == 12,
        "three_frozen_structures": {row["structure_index"] for row in panel}
        == set(PANEL_STRUCTURE_INDICES),
        "four_mask_families_per_structure": all(
            {row["mask_family"] for row in panel if row["structure_index"] == index}
            == set(PANEL_MASK_FAMILIES)
            for index in PANEL_STRUCTURE_INDICES
        ),
        "all_queries_are_e52_unknown": all(row["source_status"] == "unknown" for row in panel),
        "all_queries_are_multibit": all(row["mask_weight"] > 1 for row in panel),
        "no_finite_key_voting": all(not row["finite_key_voting"] for row in panel),
        "panel_keys_match_frozen_selection": tuple(
            (row["structure_index"], row["mask_index"]) for row in panel
        )
        == EXPECTED_PANEL_KEYS,
    }
    eligible = [
        row
        for row in providers
        if row["target_semantics_match"]
        and row["sound_certificate"]
        and row["currently_executable"]
        and row["cancellation_aware"]
        and row["within_frozen_cap"]
        and row["real_present_r5"]
        and not row["finite_key_voting"]
    ]
    strict_nontrivial = sum(
        int(row["nontrivial_positive_certificates"])
        for row in eligible
    )
    resolved_panel = sum(row["e97_status"] != "unresolved" for row in panel)
    advance_checks = {
        "sound_executable_cancellation_provider_exists": bool(eligible),
        "nontrivial_present_positive_at_least_one": strict_nontrivial >= 1,
        "frozen_panel_positive_at_least_one": any(
            row["e97_status"] == "strict_nontrivial_positive" for row in panel
        ),
        "eligible_provider_within_frozen_cap": bool(eligible),
    }
    protocol_ok = all(source_checks.values()) and all(panel_checks.values())
    if not protocol_ok:
        status = "fail"
        decision = "innovation2_present_cancellation_provider_protocol_invalid"
        action = "repair frozen source ownership, hashes, or 12-query selection"
    elif all(advance_checks.values()):
        status = "pass"
        decision = "innovation2_present_cancellation_provider_feasible"
        action = "expand strict labels, then run a deterministic mechanism gate before any neural model"
    else:
        status = "hold"
        decision = "innovation2_present_cancellation_provider_not_feasible_under_frozen_caps"
        action = "stop current provider research and consolidate the confirmed thesis contribution"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "panel_checks": panel_checks,
        "advance_checks": advance_checks,
        "metrics": {
            "source_gates": len(source_checks),
            "providers_audited": len(providers),
            "semantics_matching_providers": sum(
                row["target_semantics_match"] for row in providers
            ),
            "cancellation_aware_providers": sum(row["cancellation_aware"] for row in providers),
            "eligible_providers": len(eligible),
            "strict_nontrivial_present_positives": strict_nontrivial,
            "panel_queries": len(panel),
            "panel_resolved": resolved_panel,
            "panel_unresolved": len(panel) - resolved_panel,
            "provider_rows": providers,
        },
        "claim_scope": (
            "PRESENT-80 r5 cancellation-aware strict-label provider feasibility under "
            "frozen local caps; no neural training, r7-r9 label result, distinguisher, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "label_expansion": status == "pass",
            "neural_training": False,
            "remote_scale": False,
            "present_r7_r9_output_prediction": "blocked_until_strict_provider_passes",
            "do_not_pursue": [
                "raise exact-ANF, GLPK, ATM, time, memory, or term caps",
                "use finite-key voting as a universal label",
                "transfer small-SPN positives to PRESENT",
                "train Mask-Query Hypergraph before a provider pass",
                "launch remote PRESENT r7-r9 output prediction",
            ],
        },
    }


def source_hashes(sources: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {role: str(source["sha256"]) for role, source in sorted(sources.items())}


def serializable_config(config: CancellationProviderConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
