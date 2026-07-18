from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRESENT_RUN_ID = "i2_present_r4_r3_only_profile_operator_seed1_20260718"
GIFT_RUN_ID = "i2_gift64_r4_r3_only_profile_operator_seed1_20260719"
SKINNY_R7_RUN_ID = "i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717"
SKINNY_R8_RUN_ID = "i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717"
SKINNY_ADJACENT_RUN_ID = (
    "i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717"
)
SKINNY_BOTTOM_ROW_RUN_ID = (
    "i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717"
)
SKINNY_SINGLE_CELL_RUN_ID = (
    "i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717"
)
REAL_SPN_RUN_ID = "i2_real_spn_pair_state_transfer_readiness_20260718"


@dataclass(frozen=True)
class FrozenSource:
    run_id: str
    status: str
    decision: str
    metadata_required: bool = False


FROZEN_SOURCES = {
    "present": FrozenSource(
        PRESENT_RUN_ID,
        "pass",
        "innovation2_present_r3_only_two_seed_confirmed",
        True,
    ),
    "gift": FrozenSource(
        GIFT_RUN_ID,
        "pass",
        "innovation2_gift64_r3_only_two_seed_confirmed",
        True,
    ),
    "skinny_r7": FrozenSource(
        SKINNY_R7_RUN_ID,
        "pass",
        "innovation2_skinny_r7_hwang_kernel_reproduced",
    ),
    "skinny_r8": FrozenSource(
        SKINNY_R8_RUN_ID,
        "pass",
        "innovation2_skinny_r8_hwang_kernel_reproduced",
    ),
    "skinny_adjacent": FrozenSource(
        SKINNY_ADJACENT_RUN_ID,
        "hold",
        "innovation2_skinny_r8_geometry_kernel_not_diverse",
    ),
    "skinny_bottom_row": FrozenSource(
        SKINNY_BOTTOM_ROW_RUN_ID,
        "hold",
        "innovation2_skinny_r8_bottom_row_pair_family_not_closed",
    ),
    "skinny_single_cell": FrozenSource(
        SKINNY_SINGLE_CELL_RUN_ID,
        "hold",
        "innovation2_skinny_r7_single_cell_kernel_not_diverse",
    ),
    "real_spn": FrozenSource(
        REAL_SPN_RUN_ID,
        "hold",
        "innovation2_real_spn_pair_state_label_bank_not_ready",
    ),
}


def load_source(root: Path, *, metadata_required: bool) -> dict[str, Any]:
    import json

    gate_path = root / "gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "root": str(root),
        "gate": gate,
        "hashes": {"gate.json": _sha256(gate_path)},
    }
    if metadata_required:
        metadata_path = root / "metadata.json"
        payload["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["hashes"]["metadata.json"] = _sha256(metadata_path)
    return payload


def load_frozen_sources(roots: dict[str, Path]) -> dict[str, dict[str, Any]]:
    if set(roots) != set(FROZEN_SOURCES):
        missing = sorted(set(FROZEN_SOURCES) - set(roots))
        extra = sorted(set(roots) - set(FROZEN_SOURCES))
        raise ValueError(f"source roots mismatch: missing={missing}, extra={extra}")
    return {
        name: load_source(root, metadata_required=FROZEN_SOURCES[name].metadata_required)
        for name, root in roots.items()
    }


def validate_sources(sources: dict[str, dict[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "all_frozen_sources_present": set(sources) == set(FROZEN_SOURCES),
    }
    for name, spec in FROZEN_SOURCES.items():
        source = sources.get(name, {})
        gate = source.get("gate", {})
        hashes = source.get("hashes", {})
        checks[f"{name}_run_id_matches"] = gate.get("run_id") == spec.run_id
        checks[f"{name}_status_matches"] = gate.get("status") == spec.status
        checks[f"{name}_decision_matches"] = gate.get("decision") == spec.decision
        checks[f"{name}_hashes_present"] = len(hashes) == (
            2 if spec.metadata_required else 1
        ) and all(len(value) == 64 for value in hashes.values())

    for name in ("present", "gift"):
        gate = sources.get(name, {}).get("gate", {})
        metadata = sources.get(name, {}).get("metadata", {})
        config = metadata.get("config", {})
        protocol_checks = gate.get("protocol_checks", {})
        contract = gate.get("metrics", {}).get("contract", {})
        checks[f"{name}_internal_protocol_passes"] = bool(protocol_checks) and all(
            protocol_checks.values()
        )
        checks[f"{name}_config_is_30epoch_hidden32_steps2"] = (
            config.get("epochs") == 30
            and config.get("hidden_dim") == 32
            and config.get("steps") == 2
        )
        checks[f"{name}_contract_is_13dim_64node_4795params"] = (
            contract.get("input_dim") == 13
            and contract.get("output_shape") == [4, 64]
            and contract.get("parameter_counts_match") is True
            and set(contract.get("parameter_counts", {}).values()) == {4_795}
        )

    present_metadata = sources.get("present", {}).get("metadata", {})
    gift_metadata = sources.get("gift", {}).get("metadata", {})
    checks["independent_cipher_label_sources"] = (
        present_metadata.get("profile_source_run_id")
        != gift_metadata.get("profile_source_run_id")
    )
    checks["gift_checkpoint_transfer_is_false"] = (
        gift_metadata.get("checkpoint_transfer") is False
    )
    return checks


def adjudicate_method_synthesis(
    run_id: str,
    sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_checks = validate_sources(sources)
    present = _cipher_metrics(sources["present"]["gate"], "PRESENT-80", 96, 476)
    gift = _cipher_metrics(sources["gift"]["gate"], "GIFT-64", 192, 620)
    cipher_rows = [present, gift]
    method_checks = {
        "two_real_64bit_spns_present": {row["cipher"] for row in cipher_rows}
        == {"PRESENT-80", "GIFT-64"},
        "same_r3_only_method_contract": all(
            row["input_dim"] == 13
            and row["parameter_count"] == 4_795
            and row["epochs"] == 30
            and row["hidden_dim"] == 32
            and row["message_steps"] == 2
            for row in cipher_rows
        ),
        "both_ciphers_have_two_seeds": all(
            set(row["per_seed"]) == {"seed0", "seed1"} for row in cipher_rows
        ),
        "true_p_beats_independent_each_seed_by_0p03": all(
            seed["true_minus_independent"] >= 0.03
            for row in cipher_rows
            for seed in row["per_seed"].values()
        ),
        "true_p_beats_corrupted_each_seed_by_0p03": all(
            seed["true_minus_corrupted"] >= 0.03
            for row in cipher_rows
            for seed in row["per_seed"].values()
        ),
        "all_reported_aucs_finite": all(
            math.isfinite(value)
            for row in cipher_rows
            for seed in row["per_seed"].values()
            for value in (
                seed["true_auc"],
                seed["independent_auc"],
                seed["corrupted_auc"],
            )
        ),
        "different_structure_libraries_recorded": (
            present["structures"] != gift["structures"]
            and present["observed_matched_edges"] != gift["observed_matched_edges"]
        ),
        "direct_cross_cipher_auc_ranking_prohibited": True,
    }

    skinny_metrics = _skinny_metrics(sources)
    skinny_labels_ready = bool(skinny_metrics["strict_profile_labels_ready"])
    skinny_checks = {
        "r7_paper_kernel_reproduced": skinny_metrics["r7_kernel_reproduced"],
        "r8_paper_kernel_reproduced": skinny_metrics["r8_kernel_reproduced"],
        "r8_adjacent_pair_width_not_ready": not skinny_metrics[
            "r8_adjacent_pair_ready"
        ],
        "r8_bottom_row_family_not_ready": not skinny_metrics[
            "r8_bottom_row_ready"
        ],
        "r7_single_cell_width_not_ready": not skinny_metrics[
            "r7_single_cell_ready"
        ],
        "real_spn_ready_label_family_count_is_zero": skinny_metrics[
            "ready_label_family_count"
        ]
        == 0,
        "training_closed_until_strict_profile_ready": not skinny_labels_ready,
    }

    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_cross_spn_method_synthesis_protocol_invalid"
        action = "repair or restore the frozen E73/E79/E20-E24/E42 sources"
    elif not all(method_checks.values()):
        status = "hold"
        decision = "innovation2_cross_spn_r3_profile_method_not_confirmed"
        action = "retain per-cipher evidence and stop cross-SPN method promotion"
    elif not skinny_labels_ready:
        status = "pass"
        decision = (
            "innovation2_cross_spn_r3_profile_method_confirmed_"
            "skinny_labels_not_ready"
        )
        action = "run E81 SKINNY-64 r4 strict unit-profile label readiness"
    else:
        status = "pass"
        decision = (
            "innovation2_cross_spn_r3_profile_method_confirmed_third_spn_ready"
        )
        action = "run the frozen 4,795-parameter three-row SKINNY readiness matrix"

    skinny_row = {
        "cipher": "SKINNY-64/64",
        "rounds": None,
        "evidence_role": "third_spn_label_readiness",
        **skinny_metrics,
        "training_performed": False,
        "recommended_action": action,
    }
    results = [
        _result_row(row, decision, status) for row in cipher_rows
    ] + [_result_row(skinny_row, decision, status)]
    gate = {
        "run_id": run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "method_checks": method_checks,
        "skinny_readiness_checks": skinny_checks,
        "metrics": {
            "ciphers": cipher_rows,
            "skinny": skinny_metrics,
            "confirmed_real_spn_count": sum(
                row["two_seed_neural_attribution_confirmed"] for row in cipher_rows
            ),
            "shared_input_dim": 13,
            "shared_parameter_count": 4_795,
            "shared_message_steps": 2,
        },
        "claim_scope": (
            "method-level synthesis of separately trained two-seed PRESENT-80 r4 "
            "and GIFT-64 r4 strict unit-profile attribution; no direct cross-cipher "
            "AUC ranking, checkpoint transfer, zero-shot generalization, high-round, "
            "attack, remote-scale, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": "E81 SKINNY-64 r4 strict unit-profile label readiness",
            "training": skinny_labels_ready,
            "remote_scale": False,
            "stop_present_gift_same_benchmark_model_search": status == "pass",
        },
    }
    return gate, results


def source_hashes(sources: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {name: dict(source["hashes"]) for name, source in sources.items()}


def _cipher_metrics(
    gate: dict[str, Any],
    cipher: str,
    structures: int,
    observed_matched_edges: int,
) -> dict[str, Any]:
    per_seed: dict[str, dict[str, float]] = {}
    for seed in (0, 1):
        rows = gate.get("metrics", {}).get(f"seed{seed}_rows", [])
        by_mode = {row.get("relation_mode"): row for row in rows}
        if set(by_mode) != {"independent", "true", "corrupted"}:
            per_seed[f"seed{seed}"] = {}
            continue
        true_auc = float(by_mode["true"]["validation_auc"])
        independent_auc = float(by_mode["independent"]["validation_auc"])
        corrupted_auc = float(by_mode["corrupted"]["validation_auc"])
        per_seed[f"seed{seed}"] = {
            "true_auc": true_auc,
            "independent_auc": independent_auc,
            "corrupted_auc": corrupted_auc,
            "true_minus_independent": true_auc - independent_auc,
            "true_minus_corrupted": true_auc - corrupted_auc,
        }
    complete = all(per_seed.get(f"seed{seed}") for seed in (0, 1))
    means = {
        key: sum(per_seed[f"seed{seed}"][key] for seed in (0, 1)) / 2
        if complete
        else float("nan")
        for key in (
            "true_auc",
            "true_minus_independent",
            "true_minus_corrupted",
        )
    }
    return {
        "cipher": cipher,
        "rounds": 4,
        "evidence_role": "confirmed_method_attribution",
        "structures": structures,
        "observed_matched_edges": observed_matched_edges,
        "input_dim": 13,
        "parameter_count": 4_795,
        "epochs": 30,
        "hidden_dim": 32,
        "message_steps": 2,
        "per_seed": per_seed,
        "mean_true_auc": means["true_auc"],
        "mean_true_minus_independent": means["true_minus_independent"],
        "mean_true_minus_corrupted": means["true_minus_corrupted"],
        "two_seed_neural_attribution_confirmed": gate.get("status") == "pass",
        "strict_profile_labels_ready": True,
        "training_performed": True,
        "direct_cross_cipher_auc_ranking_allowed": False,
    }


def _skinny_metrics(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gates = {name: source["gate"] for name, source in sources.items()}
    ready_count = int(
        gates["real_spn"].get("metrics", {}).get("ready_label_family_count", -1)
    )
    return {
        "r7_kernel_reproduced": gates["skinny_r7"].get("status") == "pass",
        "r8_kernel_reproduced": gates["skinny_r8"].get("status") == "pass",
        "r8_adjacent_pair_ready": gates["skinny_adjacent"].get("status") == "pass",
        "r8_bottom_row_ready": gates["skinny_bottom_row"].get("status") == "pass",
        "r7_single_cell_ready": gates["skinny_single_cell"].get("status") == "pass",
        "ready_label_family_count": ready_count,
        "strict_profile_labels_ready": ready_count > 0
        and any(
            gates[name].get("status") == "pass"
            for name in ("skinny_adjacent", "skinny_bottom_row", "skinny_single_cell")
        ),
        "paper_kernel_reproduction_is_not_training_readiness": True,
    }


def _result_row(row: dict[str, Any], decision: str, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        **row,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FROZEN_SOURCES",
    "adjudicate_method_synthesis",
    "load_frozen_sources",
    "source_hashes",
    "validate_sources",
]
