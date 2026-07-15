from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


CIPHER_PROTOCOLS = {
    "simon64": {
        "cipher": "SIMON64/128",
        "rounds": 12,
        "profile": "simon64_lu2024_ordinary",
        "models": {
            "candidate": "simon_lu_round_relation_true",
            "shuffled": "simon_lu_round_relation_shuffled",
            "generic": "multiscale_dense_resnet",
        },
    },
    "simeck64": {
        "cipher": "Simeck64/128",
        "rounds": 15,
        "profile": "simeck64_lu2024_ordinary",
        "models": {
            "candidate": "simeck_lu_round_relation_true",
            "shuffled": "simeck_lu_round_relation_shuffled",
            "generic": "multiscale_dense_resnet",
        },
    },
}

CALIBRATION_PROTOCOLS = {
    cipher_key: {
        **protocol,
        "rounds": 11 if cipher_key == "simon64" else 14,
    }
    for cipher_key, protocol in CIPHER_PROTOCOLS.items()
}

LU_LAYOUT_PROTOCOLS = {
    "simon64": {
        **CALIBRATION_PROTOCOLS["simon64"],
        "models": {
            "candidate": "simon_lu_senet_layout_true",
            "shuffled": "simon_lu_senet_layout_shuffled",
            "generic": "simon_lu_round_relation_true",
        },
    },
    "simeck64": {
        **CALIBRATION_PROTOCOLS["simeck64"],
        "models": {
            "candidate": "simeck_lu_senet_layout_true",
            "shuffled": "simeck_lu_senet_layout_shuffled",
            "generic": "simeck_lu_round_relation_true",
        },
    },
}

SCALE_PROBE_PROTOCOLS = {
    cipher_key: {
        **protocol,
        "models": {
            "candidate": protocol["models"]["candidate"],
            "shuffled": protocol["models"]["shuffled"],
        },
    }
    for cipher_key, protocol in CALIBRATION_PROTOCOLS.items()
}

SCALE_PROBE_ANCHOR_AUC = {
    "simon64": 0.5496445496877035,
    "simeck64": 0.5740933318932852,
}

HIGH_ROUND_SCALE_PROTOCOLS = {
    cipher_key: {
        **protocol,
        "models": {
            "candidate": protocol["models"]["candidate"],
            "shuffled": protocol["models"]["shuffled"],
        },
    }
    for cipher_key, protocol in CIPHER_PROTOCOLS.items()
}

HIGH_ROUND_ANCHOR_AUC = {
    "simon64": 0.503118654092153,
    "simeck64": 0.49672653277715045,
}


def feistel_balanced_relation_decision(
    scores_by_cipher: dict[str, dict[int, dict[str, float]]],
    *,
    minimum_signal_auc: float = 0.55,
    topology_margin: float = 0.01,
    competitiveness_tolerance: float = 0.005,
) -> dict[str, Any]:
    cipher_gates: dict[str, dict[str, bool]] = {}
    topology_margins: dict[str, dict[int, float]] = {}
    generic_margins: dict[str, dict[int, float]] = {}
    passing_ciphers: list[str] = []

    for cipher_key, scores_by_seed in scores_by_cipher.items():
        topology_margins[cipher_key] = {
            seed: scores["candidate"] - scores["shuffled"]
            for seed, scores in scores_by_seed.items()
        }
        generic_margins[cipher_key] = {
            seed: scores["candidate"] - scores["generic"]
            for seed, scores in scores_by_seed.items()
        }
        gates = {
            "signal_present": all(
                scores["candidate"] >= minimum_signal_auc
                for scores in scores_by_seed.values()
            ),
            "relation_attributed": all(
                margin >= topology_margin
                for margin in topology_margins[cipher_key].values()
            ),
            "generic_competitive": all(
                margin >= -competitiveness_tolerance
                for margin in generic_margins[cipher_key].values()
            ),
        }
        gates["passed"] = all(gates.values())
        cipher_gates[cipher_key] = gates
        if gates["passed"]:
            passing_ciphers.append(cipher_key)

    if len(passing_ciphers) == len(scores_by_cipher):
        decision = "feistel_balanced_relation_two_cipher_seed0_pass"
        next_action = "run_same_budget_simon_simeck_seed1_confirmation"
    elif len(passing_ciphers) == 1:
        decision = "feistel_balanced_relation_cipher_conditional"
        next_action = (
            "audit_wrong_cipher_relation_on_nonpassing_cipher_before_another_seed"
        )
    elif any(gates["signal_present"] for gates in cipher_gates.values()):
        decision = "feistel_balanced_signal_without_relation_attribution"
        next_action = "retain_strongest_raw_baseline_and_reject_relation_scaleup"
    else:
        decision = "feistel_balanced_relation_not_ready"
        next_action = "run_easier_round_formula_calibration_before_redesign"

    return {
        "decision": decision,
        "next_action": next_action,
        "scores_by_cipher": scores_by_cipher,
        "topology_margins": topology_margins,
        "generic_margins": generic_margins,
        "cipher_gates": cipher_gates,
        "passing_ciphers": passing_ciphers,
        "thresholds": {
            "minimum_signal_auc": minimum_signal_auc,
            "topology_margin": topology_margin,
            "competitiveness_tolerance": competitiveness_tolerance,
        },
    }


def feistel_balanced_calibration_decision(
    scores_by_cipher: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    report = feistel_balanced_relation_decision(
        scores_by_cipher,
        minimum_signal_auc=0.60,
        topology_margin=0.01,
        competitiveness_tolerance=0.005,
    )
    passing = report["passing_ciphers"]
    if len(passing) == len(scores_by_cipher):
        report["decision"] = "feistel_balanced_easier_round_calibrated"
        report["next_action"] = (
            "implement_closer_lu_senet_high_round_protocol_comparison_locally"
        )
    elif len(passing) == 1:
        report["decision"] = "feistel_balanced_easier_round_cipher_conditional"
        report["next_action"] = (
            "audit_nonpassing_cipher_round_function_with_wrong_cipher_control"
        )
    elif any(gates["signal_present"] for gates in report["cipher_gates"].values()):
        report["decision"] = "feistel_balanced_easier_round_signal_without_attribution"
        report["next_action"] = (
            "retain_raw_baseline_and_stop_round_relation_architecture_scaleup"
        )
    else:
        report["decision"] = "feistel_balanced_easier_round_not_calibrated"
        report["next_action"] = "run_author_code_row_level_data_and_layout_parity_audit"
    return report


def feistel_lu_layout_decision(
    scores_by_cipher: dict[str, dict[int, dict[str, float]]],
    *,
    minimum_signal_auc: float = 0.60,
    topology_margin: float = 0.01,
    anchor_improvement_margin: float = 0.02,
) -> dict[str, Any]:
    topology_margins: dict[str, dict[int, float]] = {}
    anchor_margins: dict[str, dict[int, float]] = {}
    cipher_gates: dict[str, dict[str, bool]] = {}
    passing_ciphers: list[str] = []
    for cipher_key, scores_by_seed in scores_by_cipher.items():
        topology_margins[cipher_key] = {
            seed: scores["candidate"] - scores["shuffled"]
            for seed, scores in scores_by_seed.items()
        }
        anchor_margins[cipher_key] = {
            seed: scores["candidate"] - scores["generic"]
            for seed, scores in scores_by_seed.items()
        }
        gates = {
            "signal_present": all(
                scores["candidate"] >= minimum_signal_auc
                for scores in scores_by_seed.values()
            ),
            "relation_attributed": all(
                margin >= topology_margin
                for margin in topology_margins[cipher_key].values()
            ),
            "architecture_improved": all(
                margin >= anchor_improvement_margin
                for margin in anchor_margins[cipher_key].values()
            ),
        }
        gates["passed"] = all(gates.values())
        cipher_gates[cipher_key] = gates
        if gates["passed"]:
            passing_ciphers.append(cipher_key)

    if len(passing_ciphers) == len(scores_by_cipher):
        decision = "feistel_lu_layout_two_cipher_calibrated"
        next_action = "run_same_layout_comparison_at_simon_r12_simeck_r15"
    elif len(passing_ciphers) == 1:
        decision = "feistel_lu_layout_cipher_conditional"
        next_action = "retain_passing_cipher_layout_and_audit_nonpassing_cipher"
    elif any(gates["signal_present"] for gates in cipher_gates.values()):
        decision = "feistel_lu_layout_signal_without_architecture_gain"
        next_action = "retain_pair_pool_anchor_and_stop_layout_scaleup"
    else:
        decision = "feistel_lu_layout_not_calibrated"
        next_action = "quantify_data_scale_gap_before_any_remote_request"

    return {
        "decision": decision,
        "next_action": next_action,
        "scores_by_cipher": scores_by_cipher,
        "topology_margins": topology_margins,
        "anchor_margins": anchor_margins,
        "cipher_gates": cipher_gates,
        "passing_ciphers": passing_ciphers,
        "thresholds": {
            "minimum_signal_auc": minimum_signal_auc,
            "topology_margin": topology_margin,
            "anchor_improvement_margin": anchor_improvement_margin,
        },
    }


def feistel_relation_scale_probe_decision(
    scores_by_cipher: dict[str, dict[int, dict[str, float]]],
    *,
    minimum_signal_auc: float = 0.57,
    topology_margin: float = 0.02,
    scale_gain_margin: float = 0.02,
    anchor_auc_2048: dict[str, float] | None = None,
) -> dict[str, Any]:
    anchors = anchor_auc_2048 or SCALE_PROBE_ANCHOR_AUC
    topology_margins: dict[str, dict[int, float]] = {}
    scale_gains: dict[str, dict[int, float]] = {}
    cipher_gates: dict[str, dict[str, bool]] = {}
    passing_ciphers: list[str] = []
    for cipher_key, scores_by_seed in scores_by_cipher.items():
        topology_margins[cipher_key] = {
            seed: scores["candidate"] - scores["shuffled"]
            for seed, scores in scores_by_seed.items()
        }
        scale_gains[cipher_key] = {
            seed: scores["candidate"] - anchors[cipher_key]
            for seed, scores in scores_by_seed.items()
        }
        gates = {
            "signal_present": all(
                scores["candidate"] >= minimum_signal_auc
                for scores in scores_by_seed.values()
            ),
            "relation_attributed": all(
                margin >= topology_margin
                for margin in topology_margins[cipher_key].values()
            ),
            "positive_scale_slope": all(
                margin >= scale_gain_margin
                for margin in scale_gains[cipher_key].values()
            ),
        }
        gates["passed"] = all(gates.values())
        cipher_gates[cipher_key] = gates
        if gates["passed"]:
            passing_ciphers.append(cipher_key)

    if len(passing_ciphers) == len(scores_by_cipher):
        decision = "feistel_relation_scale_slope_two_cipher_pass"
        next_action = "run_same_8192_class_probe_with_independent_seed1"
    elif len(passing_ciphers) == 1:
        decision = "feistel_relation_scale_slope_cipher_conditional"
        next_action = "confirm_only_passing_cipher_with_independent_seed1"
    elif any(gates["signal_present"] for gates in cipher_gates.values()):
        decision = "feistel_relation_signal_without_scale_slope"
        next_action = "stop_mechanical_scaleup_and_redesign_representation"
    else:
        decision = "feistel_relation_scale_probe_not_ready"
        next_action = "stop_scaleup_and_reassess_feistel_route_priority"

    return {
        "decision": decision,
        "next_action": next_action,
        "scores_by_cipher": scores_by_cipher,
        "topology_margins": topology_margins,
        "scale_gains": scale_gains,
        "anchor_auc_2048": anchors,
        "cipher_gates": cipher_gates,
        "passing_ciphers": passing_ciphers,
        "thresholds": {
            "minimum_signal_auc": minimum_signal_auc,
            "topology_margin": topology_margin,
            "scale_gain_margin": scale_gain_margin,
        },
    }


def feistel_target_round_scale_probe_decision(
    scores_by_cipher: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    report = feistel_relation_scale_probe_decision(
        scores_by_cipher,
        minimum_signal_auc=0.55,
        topology_margin=0.02,
        scale_gain_margin=0.02,
        anchor_auc_2048=HIGH_ROUND_ANCHOR_AUC,
    )
    if report["decision"] == "feistel_relation_scale_slope_two_cipher_pass":
        report["decision"] = "feistel_target_round_8192_two_cipher_pass"
        report["next_action"] = "run_same_target_round_8192_matrix_with_seed1"
    elif report["decision"] == "feistel_relation_scale_slope_cipher_conditional":
        report["decision"] = "feistel_target_round_8192_cipher_conditional"
        report["next_action"] = "confirm_only_passing_target_cipher_with_seed1"
    elif report["decision"] == "feistel_relation_signal_without_scale_slope":
        report["decision"] = "feistel_target_round_signal_without_scale_slope"
        report["next_action"] = "hold_target_round_scale_and_retain_low_round_evidence"
    else:
        report["decision"] = "feistel_target_round_8192_not_ready"
        report["next_action"] = "hold_high_round_route_without_remote_scale"
    return report


def feistel_relation_scale_confirmation_decision(
    scores_by_cipher: dict[str, dict[int, dict[str, float]]],
    *,
    minimum_signal_auc: float = 0.57,
    topology_margin: float = 0.02,
) -> dict[str, Any]:
    topology_margins = {
        cipher_key: {
            seed: scores["candidate"] - scores["shuffled"]
            for seed, scores in scores_by_seed.items()
        }
        for cipher_key, scores_by_seed in scores_by_cipher.items()
    }
    cipher_gates: dict[str, dict[str, bool]] = {}
    passing_ciphers: list[str] = []
    for cipher_key, scores_by_seed in scores_by_cipher.items():
        gates = {
            "signal_present": all(
                scores["candidate"] >= minimum_signal_auc
                for scores in scores_by_seed.values()
            ),
            "relation_attributed": all(
                margin >= topology_margin
                for margin in topology_margins[cipher_key].values()
            ),
        }
        gates["passed"] = all(gates.values())
        cipher_gates[cipher_key] = gates
        if gates["passed"]:
            passing_ciphers.append(cipher_key)

    if len(passing_ciphers) == len(scores_by_cipher):
        decision = "feistel_relation_8192_seed1_confirmation_pass"
        next_action = "synthesize_two_seed_8192_evidence_before_high_round_probe"
    elif len(passing_ciphers) == 1:
        decision = "feistel_relation_8192_seed1_cipher_conditional"
        next_action = "retain_only_two_run_confirmed_cipher_route"
    else:
        decision = "feistel_relation_8192_seed1_confirmation_failed"
        next_action = "stop_scaleup_and_reassess_feistel_route_priority"

    return {
        "decision": decision,
        "next_action": next_action,
        "scores_by_cipher": scores_by_cipher,
        "topology_margins": topology_margins,
        "cipher_gates": cipher_gates,
        "passing_ciphers": passing_ciphers,
        "thresholds": {
            "minimum_signal_auc": minimum_signal_auc,
            "topology_margin": topology_margin,
        },
    }


def gate_feistel_balanced_results(
    *,
    plan_path: Path,
    results_path: Path,
    expected_samples_per_class: int,
    expected_seeds: tuple[int, ...],
    expected_epochs: int,
    expected_final_repeats: int,
    readiness: bool = False,
    calibration: bool = False,
    layout_repair: bool = False,
    scale_probe: bool = False,
    scale_confirmation: bool = False,
    target_round_probe: bool = False,
) -> dict[str, Any]:
    if (
        sum(
            (
                readiness,
                calibration,
                layout_repair,
                scale_probe,
                scale_confirmation,
                target_round_probe,
            )
        )
        > 1
    ):
        raise ValueError("gate modes are mutually exclusive")
    if target_round_probe:
        protocols = HIGH_ROUND_SCALE_PROTOCOLS
    elif scale_confirmation:
        protocols = SCALE_PROBE_PROTOCOLS
    elif scale_probe:
        protocols = SCALE_PROBE_PROTOCOLS
    elif layout_repair:
        protocols = LU_LAYOUT_PROTOCOLS
    elif calibration:
        protocols = CALIBRATION_PROTOCOLS
    else:
        protocols = CIPHER_PROTOCOLS
    roles_per_cipher = len(next(iter(protocols.values()))["models"])
    expected_rows = len(protocols) * roles_per_cipher * len(expected_seeds)
    alignment = validate_result_plan_alignment(
        plan_path, results_path, expected_rows=expected_rows
    )
    errors = list(alignment["errors"])
    plan_rows = _read_plan_rows(plan_path, errors)
    rows = _read_result_rows(results_path, errors)
    plan_by_identity = {
        (
            row.get("cipher", ""),
            row.get("model_key", ""),
            int(row.get("seed") or 0),
        ): row
        for row in plan_rows
    }

    rows_by_cipher: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        cipher_key: {seed: {} for seed in expected_seeds} for cipher_key in protocols
    }
    for row in rows:
        cipher_key = str(row.get("cipher_key"))
        seed = row.get("seed")
        if cipher_key not in protocols:
            errors.append(f"unexpected cipher_key: {cipher_key}")
            continue
        if seed not in rows_by_cipher[cipher_key]:
            errors.append(f"unexpected seed for {cipher_key}: {seed}")
            continue
        models = protocols[cipher_key]["models"]
        role_by_model = {model: role for role, model in models.items()}
        role = role_by_model.get(str(row.get("model")))
        if role is None:
            errors.append(f"unexpected model for {cipher_key}: {row.get('model')}")
            continue
        if role in rows_by_cipher[cipher_key][int(seed)]:
            errors.append(f"duplicate row: cipher={cipher_key} seed={seed} role={role}")
            continue
        rows_by_cipher[cipher_key][int(seed)][role] = row

    scores_by_cipher: dict[str, dict[int, dict[str, float]]] = {}
    parameter_counts: dict[str, dict[int, dict[str, int]]] = {}
    for cipher_key, protocol in protocols.items():
        scores_by_cipher[cipher_key] = {}
        parameter_counts[cipher_key] = {}
        expected_roles = set(protocol["models"])
        for seed in expected_seeds:
            role_rows = rows_by_cipher[cipher_key][seed]
            missing = expected_roles - set(role_rows)
            if missing:
                errors.append(
                    f"{cipher_key} seed{seed} missing roles: {sorted(missing)}"
                )
                continue
            scores_by_cipher[cipher_key][seed] = {}
            parameter_counts[cipher_key][seed] = {}
            for role, row in role_rows.items():
                _validate_protocol_row(
                    row,
                    protocol=protocol,
                    expected_samples_per_class=expected_samples_per_class,
                    expected_epochs=expected_epochs,
                    expected_final_repeats=expected_final_repeats,
                    plan_row=plan_by_identity.get(
                        (str(row.get("cipher")), str(row.get("model")), seed)
                    ),
                    errors=errors,
                )
                auc = _fresh_auc(row, errors, cipher_key, seed, role)
                if auc is not None:
                    scores_by_cipher[cipher_key][seed][role] = auc
                parameter_counts[cipher_key][seed][role] = int(
                    row.get("parameter_count") or -1
                )
            if parameter_counts[cipher_key][seed].get("candidate") != parameter_counts[
                cipher_key
            ][seed].get("shuffled"):
                errors.append(
                    f"{cipher_key} seed{seed} candidate/shuffled parameter_count mismatch"
                )

    if errors:
        return {
            "status": "fail",
            "decision": "invalid_feistel_balanced_protocol",
            "next_action": "repair_protocol_or_artifacts_before_interpretation",
            "research_decision_applied": False,
            "alignment": alignment,
            "errors": errors,
            "parameter_counts": parameter_counts,
            "claim_scope": "invalid artifacts; no research interpretation allowed",
        }

    if readiness:
        return {
            "status": "pass",
            "decision": "feistel_balanced_relation_readiness_passed",
            "next_action": "run_frozen_2048_class_seed0_local_diagnostic",
            "research_decision_applied": False,
            "alignment": alignment,
            "errors": [],
            "scores_by_cipher": scores_by_cipher,
            "parameter_counts": parameter_counts,
            "claim_scope": "mechanics readiness only; AUC is not research evidence",
            "stopped_actions": [
                "remote_launch",
                "paper_scale_claim",
                "readiness_auc_interpretation",
            ],
        }

    if target_round_probe:
        decision = feistel_target_round_scale_probe_decision(scores_by_cipher)
    elif scale_confirmation:
        decision = feistel_relation_scale_confirmation_decision(scores_by_cipher)
    elif scale_probe:
        decision = feistel_relation_scale_probe_decision(scores_by_cipher)
    elif layout_repair:
        decision = feistel_lu_layout_decision(scores_by_cipher)
    elif calibration:
        decision = feistel_balanced_calibration_decision(scores_by_cipher)
    else:
        decision = feistel_balanced_relation_decision(scores_by_cipher)
    return {
        "status": "pass",
        **decision,
        "research_decision_applied": True,
        "alignment": alignment,
        "errors": [],
        "parameter_counts": parameter_counts,
        "claim_scope": (
            "single-seed local SIMON64-r12/SIMECK64-r15 8192/class "
            "target-round signal and scale probe; not formal or paper-scale evidence"
            if target_round_probe
            else (
                "independent seed1 local SIMON64-r11/SIMECK64-r14 8192/class "
                "signal and relation-attribution confirmation; not a second "
                "within-seed scale slope or paper-scale evidence"
                if scale_confirmation
                else (
                    "single-seed local SIMON64-r11/SIMECK64-r14 8192/class "
                    "data-scarcity probe; not formal or paper-scale evidence"
                    if scale_probe
                    else (
                        "single-seed local SIMON64-r11/SIMECK64-r14 Lu-layout repair "
                        "calibration; not an exact or paper-scale reproduction"
                        if layout_repair
                        else (
                            "single-seed local SIMON64-r11/SIMECK64-r14 easier-round "
                            "implementation calibration; not formal or paper-scale evidence"
                            if calibration
                            else "single-seed local SIMON64-r12/SIMECK64-r15 architecture and "
                            "relation-attribution diagnostic; not formal or paper-scale evidence"
                        )
                    )
                )
            )
        ),
        "stopped_actions": [
            "remote_launch",
            "paper_scale_claim",
            "related_key_or_rx_protocol_change",
            "cross_feistel_generalization_claim",
        ],
    }


def _read_plan_rows(path: Path, errors: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        errors.append(f"cannot read plan: {exc}")
        return []


def _read_result_rows(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read result rows: {exc}")
        return []


def _validate_protocol_row(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    expected_samples_per_class: int,
    expected_epochs: int,
    expected_final_repeats: int,
    plan_row: dict[str, str] | None,
    errors: list[str],
) -> None:
    identity = f"{row.get('cipher_key')} seed{row.get('seed')} {row.get('model')}"
    expected = {
        "cipher": protocol["cipher"],
        "structure": "Feistel-like",
        "rounds": protocol["rounds"],
        "samples_per_class": expected_samples_per_class,
        "pairs_per_sample": 8,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "key_rotation_interval": 1,
        "sample_structure": "independent_pairs",
        "difference_profile": protocol["profile"],
        "difference_member": 0,
        "final_test_repeats": expected_final_repeats,
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            errors.append(
                f"{identity} {field}={row.get(field)!r} expected={expected_value!r}"
            )

    history = row.get("history")
    if not isinstance(history, list) or len(history) != expected_epochs:
        errors.append(
            f"{identity} history epochs={len(history) if isinstance(history, list) else None} "
            f"expected={expected_epochs}"
        )
    training = row.get("training")
    if not isinstance(training, dict):
        errors.append(f"{identity} missing training metadata")
    else:
        training_expected = {
            "epochs": expected_epochs,
            "epochs_ran": expected_epochs,
            "samples_total": expected_samples_per_class * 2,
            "key_rotation_interval": 1,
            "pairs_per_sample": 8,
            "sample_structure": "independent_pairs",
            "feature_encoding": "ciphertext_pair_bits",
            "key_rotation_row_indexing": "global_dataset_row",
        }
        for field, expected_value in training_expected.items():
            if training.get(field) != expected_value:
                errors.append(
                    f"{identity} training.{field}={training.get(field)!r} "
                    f"expected={expected_value!r}"
                )

    validation = row.get("validation")
    if not isinstance(validation, dict):
        errors.append(f"{identity} missing validation metadata")
    elif plan_row is not None:
        if validation.get("key_rotation_row_indexing") != "global_dataset_row":
            errors.append(
                f"{identity} validation.key_rotation_row_indexing="
                f"{validation.get('key_rotation_row_indexing')!r} "
                "expected='global_dataset_row'"
            )
        expected_validation_total = _optional_int(
            plan_row.get("validation_samples_total")
        )
        if validation.get("samples_total") != expected_validation_total:
            errors.append(
                f"{identity} validation.samples_total={validation.get('samples_total')!r} "
                f"expected={expected_validation_total!r}"
            )

    if plan_row is None:
        errors.append(f"{identity} has no matching plan row for total-size validation")
        return
    for field in (
        "train_samples_total",
        "validation_samples_total",
        "final_test_samples_total",
    ):
        expected_total = _optional_int(plan_row.get(field))
        if row.get(field) != expected_total:
            errors.append(
                f"{identity} {field}={row.get(field)!r} expected={expected_total!r}"
            )
    final_total = _optional_int(plan_row.get("final_test_samples_total"))
    final_evaluation = row.get("final_evaluation")
    if not isinstance(final_evaluation, dict):
        errors.append(f"{identity} missing final_evaluation")
    else:
        if final_evaluation.get("repeats") != expected_final_repeats:
            errors.append(
                f"{identity} final_evaluation.repeats="
                f"{final_evaluation.get('repeats')!r} expected={expected_final_repeats}"
            )
        if final_evaluation.get("samples_total_per_repeat") != final_total:
            errors.append(
                f"{identity} final_evaluation.samples_total_per_repeat="
                f"{final_evaluation.get('samples_total_per_repeat')!r} "
                f"expected={final_total!r}"
            )
        metrics = final_evaluation.get("metrics_by_repeat")
        if not isinstance(metrics, list) or len(metrics) != expected_final_repeats:
            errors.append(f"{identity} final repeat metrics are incomplete")


def _fresh_auc(
    row: dict[str, Any],
    errors: list[str],
    cipher_key: str,
    seed: int,
    role: str,
) -> float | None:
    try:
        auc = float(row["final_evaluation"]["auc_mean"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"{cipher_key} seed{seed} {role} missing fresh auc_mean")
        return None
    if not math.isfinite(auc):
        errors.append(f"{cipher_key} seed{seed} {role} fresh auc_mean is not finite")
        return None
    return auc


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value, 0)


__all__ = [
    "CALIBRATION_PROTOCOLS",
    "CIPHER_PROTOCOLS",
    "LU_LAYOUT_PROTOCOLS",
    "HIGH_ROUND_ANCHOR_AUC",
    "HIGH_ROUND_SCALE_PROTOCOLS",
    "SCALE_PROBE_ANCHOR_AUC",
    "SCALE_PROBE_PROTOCOLS",
    "feistel_balanced_calibration_decision",
    "feistel_lu_layout_decision",
    "feistel_relation_scale_probe_decision",
    "feistel_relation_scale_confirmation_decision",
    "feistel_target_round_scale_probe_decision",
    "feistel_balanced_relation_decision",
    "gate_feistel_balanced_results",
]
