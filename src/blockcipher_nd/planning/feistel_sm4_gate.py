from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


MODEL_ROLES = {
    "candidate": "sm4_word_recurrence_true",
    "shuffled": "sm4_word_recurrence_shuffled",
    "baseline": "multiscale_dense_resnet",
}


def gate_feistel_sm4_results(
    *,
    plan_path: Path,
    results_path: Path,
    expected_samples_per_class: int,
    expected_seeds: tuple[int, ...],
    expected_epochs: int,
    expected_final_repeats: int,
    topology_margin: float = 0.005,
    competitiveness_tolerance: float = 0.002,
    minimum_signal_auc: float = 0.55,
) -> dict[str, Any]:
    expected_rows = len(MODEL_ROLES) * len(expected_seeds)
    alignment = validate_result_plan_alignment(
        plan_path, results_path, expected_rows=expected_rows
    )
    errors = list(alignment["errors"])
    try:
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        rows = []
        errors.append(f"cannot read result rows: {exc}")

    role_by_model = {model: role for role, model in MODEL_ROLES.items()}
    rows_by_seed: dict[int, dict[str, dict[str, Any]]] = {
        seed: {} for seed in expected_seeds
    }
    for row in rows:
        seed = row.get("seed")
        role = role_by_model.get(str(row.get("model")))
        if seed not in rows_by_seed:
            errors.append(f"unexpected seed: {seed}")
            continue
        if role is None:
            errors.append(f"unexpected model: {row.get('model')}")
            continue
        if role in rows_by_seed[int(seed)]:
            errors.append(f"duplicate seed/model row: seed={seed} role={role}")
            continue
        rows_by_seed[int(seed)][role] = row

    scores_by_seed: dict[int, dict[str, float]] = {}
    parameter_counts: dict[int, dict[str, dict[str, int]]] = {}
    for seed, rows_by_role in rows_by_seed.items():
        missing = set(MODEL_ROLES) - set(rows_by_role)
        if missing:
            errors.append(f"seed{seed} missing roles: {sorted(missing)}")
            continue
        scores_by_seed[seed] = {}
        parameter_counts[seed] = {}
        for role, row in rows_by_role.items():
            expected_fields = {
                "cipher_key": "sm4",
                "rounds": 5,
                "samples_per_class": expected_samples_per_class,
                "pairs_per_sample": 1,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "key_rotation_interval": 1,
                "sample_structure": "independent_pairs",
                "difference_profile": "sm4_yu2023_conv_resnet",
                "difference_member": 0,
                "final_test_repeats": expected_final_repeats,
            }
            for field, expected in expected_fields.items():
                if row.get(field) != expected:
                    errors.append(
                        f"seed{seed} {role} {field}={row.get(field)!r} "
                        f"expected={expected!r}"
                    )
            history = row.get("history")
            if not isinstance(history, list) or len(history) != expected_epochs:
                errors.append(
                    f"seed{seed} {role} history must contain {expected_epochs} epochs"
                )
            for split in ("training", "validation"):
                metadata = row.get(split)
                if not isinstance(metadata, dict):
                    errors.append(f"seed{seed} {role} missing {split} metadata")
                elif metadata.get("key_schedule") != "rotating":
                    errors.append(
                        f"seed{seed} {role} {split} key_schedule="
                        f"{metadata.get('key_schedule')!r} expected='rotating'"
                    )
            final = row.get("final_evaluation")
            if not isinstance(final, dict):
                errors.append(f"seed{seed} {role} missing final_evaluation")
            else:
                auc = final.get("auc_mean")
                if not isinstance(auc, (int, float)):
                    errors.append(f"seed{seed} {role} missing final auc_mean")
                else:
                    scores_by_seed[seed][role] = float(auc)
                metrics = final.get("metrics_by_repeat")
                if final.get("repeats") != expected_final_repeats:
                    errors.append(
                        f"seed{seed} {role} final repeats={final.get('repeats')!r} "
                        f"expected={expected_final_repeats}"
                    )
                if not isinstance(metrics, list) or len(metrics) != expected_final_repeats:
                    errors.append(
                        f"seed{seed} {role} final metrics_by_repeat must contain "
                        f"{expected_final_repeats} rows"
                    )
            parameter_counts[seed][role] = {}
            for field in ("parameter_count", "trainable_parameter_count"):
                count = row.get(field)
                if not isinstance(count, int):
                    errors.append(f"seed{seed} {role} missing {field}")
                else:
                    parameter_counts[seed][role][field] = count
        if {"candidate", "shuffled"}.issubset(parameter_counts[seed]):
            if parameter_counts[seed]["candidate"] != parameter_counts[seed]["shuffled"]:
                errors.append(
                    f"seed{seed} SM4 topology control capacity mismatch: "
                    f"candidate={parameter_counts[seed]['candidate']} "
                    f"shuffled={parameter_counts[seed]['shuffled']}"
                )

    if errors:
        return {
            "status": "fail",
            "decision": "invalid_feistel_sm4_protocol",
            "errors": errors,
            "alignment": alignment,
            "research_decision_applied": False,
        }

    common = {
        "status": "pass",
        "errors": [],
        "alignment": alignment,
        "samples_per_class": expected_samples_per_class,
        "seeds": list(expected_seeds),
        "epochs": expected_epochs,
        "models": MODEL_ROLES,
        "parameter_counts": parameter_counts,
        "scores_by_seed": scores_by_seed,
    }
    if expected_samples_per_class < 2048:
        return {
            **common,
            "decision": "feistel_sm4_word_recurrence_readiness_passed",
            "next_action": "run_sm4_r5_word_recurrence_attribution_2048",
            "research_decision_applied": False,
            "claim_scope": "readiness only; metrics are not attribution evidence",
        }

    topology_margins = {
        seed: scores["candidate"] - scores["shuffled"]
        for seed, scores in scores_by_seed.items()
    }
    baseline_margins = {
        seed: scores["candidate"] - scores["baseline"]
        for seed, scores in scores_by_seed.items()
    }
    topology_attributed = all(
        margin >= topology_margin for margin in topology_margins.values()
    )
    baseline_competitive = all(
        margin >= -competitiveness_tolerance for margin in baseline_margins.values()
    )
    signal_present = all(
        scores["candidate"] >= minimum_signal_auc
        for scores in scores_by_seed.values()
    )

    if topology_attributed and baseline_competitive and signal_present:
        decision = "feistel_sm4_r5_word_recurrence_attributed"
        next_action = "freeze_sm4_r5_65536_class_two_seed_remote_diagnostic"
    elif signal_present and not topology_attributed:
        decision = "feistel_sm4_signal_without_recurrence_attribution"
        next_action = "retain_strongest_baseline_and_redesign_sm4_control_locally"
    elif signal_present and topology_attributed:
        decision = "feistel_sm4_recurrence_attributed_but_not_competitive"
        next_action = "retain_attribution_only_and_reject_candidate_architecture"
    else:
        decision = "feistel_sm4_word_recurrence_not_ready"
        next_action = "stop_scale_and_audit_sm4_data_model_semantics"

    return {
        **common,
        "decision": decision,
        "next_action": next_action,
        "research_decision_applied": True,
        "topology_margins": topology_margins,
        "baseline_margins": baseline_margins,
        "gates": {
            "topology_attributed": topology_attributed,
            "baseline_competitive": baseline_competitive,
            "signal_present": signal_present,
        },
        "thresholds": {
            "topology_margin": topology_margin,
            "competitiveness_tolerance": competitiveness_tolerance,
            "minimum_signal_auc": minimum_signal_auc,
        },
        "claim_scope": (
            "local SM4-r5 word-recurrence architecture/attribution diagnostic; "
            "not paper-scale or a cross-Feistel architecture rule"
        ),
        "stopped_actions": [
            "sm4_r6_r8_round_sweep",
            "paper_scale_training",
            "dense_ddt_route",
            "cross_feistel_generalization_claim",
        ],
    }


__all__ = ["MODEL_ROLES", "gate_feistel_sm4_results"]
