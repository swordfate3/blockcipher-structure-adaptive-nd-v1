from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


MODEL_ROLES = {
    "typed_true": "des_feistel_branch_inception_true",
    "typed_shuffled": "des_feistel_branch_inception_shuffled",
    "paper_inception": "des_zhang_wang_inception_pairset",
    "lstm": "des_lstm_pairset",
}


def feistel_des_decision(
    scores_by_seed: dict[int, dict[str, float]],
    *,
    topology_margin: float = 0.005,
    competitiveness_tolerance: float = 0.002,
    minimum_signal_auc: float = 0.55,
) -> dict[str, Any]:
    topology_margins: dict[int, float] = {}
    mainstream_margins: dict[int, float] = {}
    strongest_baselines: dict[int, str] = {}
    for seed, scores in sorted(scores_by_seed.items()):
        topology_margins[seed] = scores["typed_true"] - scores["typed_shuffled"]
        strongest_baseline = max(
            ("paper_inception", "lstm"), key=lambda role: scores[role]
        )
        strongest_baselines[seed] = strongest_baseline
        mainstream_margins[seed] = scores["typed_true"] - scores[strongest_baseline]

    topology_attributed = all(
        margin >= topology_margin for margin in topology_margins.values()
    )
    mainstream_competitive = all(
        margin >= -competitiveness_tolerance
        for margin in mainstream_margins.values()
    )
    signal_present = all(
        scores["typed_true"] >= minimum_signal_auc
        for scores in scores_by_seed.values()
    )
    if topology_attributed and mainstream_competitive and signal_present:
        decision = "feistel_branch_candidate_ready_for_medium_diagnostic"
        next_action = "run_des_r6_65536_class_two_seed_remote_diagnostic"
    elif signal_present and not topology_attributed:
        decision = "feistel_signal_without_branch_topology_attribution"
        next_action = "retain_best_baseline_and_redesign_branch_control_before_scale"
    else:
        decision = "feistel_branch_candidate_not_ready"
        next_action = "stop_scale_and_redesign_locally"
    return {
        "decision": decision,
        "next_action": next_action,
        "scores_by_seed": scores_by_seed,
        "topology_margins": topology_margins,
        "mainstream_margins": mainstream_margins,
        "strongest_baselines": strongest_baselines,
        "gates": {
            "topology_attributed": topology_attributed,
            "mainstream_competitive": mainstream_competitive,
            "signal_present": signal_present,
        },
    }


def gate_feistel_des_results(
    *,
    plan_path: Path,
    results_path: Path,
    expected_samples_per_class: int,
    expected_seeds: tuple[int, ...],
    expected_epochs: int,
    expected_final_repeats: int,
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

    rows_by_seed: dict[int, dict[str, dict[str, Any]]] = {
        seed: {} for seed in expected_seeds
    }
    role_by_model = {model: role for role, model in MODEL_ROLES.items()}
    for row in rows:
        seed = row.get("seed")
        model = row.get("model")
        if seed not in rows_by_seed:
            errors.append(f"unexpected seed: {seed}")
            continue
        role = role_by_model.get(str(model))
        if role is None:
            errors.append(f"unexpected model: {model}")
            continue
        if role in rows_by_seed[int(seed)]:
            errors.append(f"duplicate seed/model row: seed={seed} model={model}")
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
                "cipher_key": "des",
                "rounds": 6,
                "samples_per_class": expected_samples_per_class,
                "pairs_per_sample": 16,
                "feature_encoding": "ciphertext_pair_bits",
                "negative_mode": "encrypted_random_plaintexts",
                "sample_structure": "zhang_wang_case2_official_mcnd",
                "difference_profile": "des_zhang_wang2022_mcnd",
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
            training = row.get("training")
            if not isinstance(training, dict):
                errors.append(f"seed{seed} {role} missing training metadata")
            elif training.get("key_schedule") != "per_pair_random":
                errors.append(
                    f"seed{seed} {role} training key_schedule="
                    f"{training.get('key_schedule')!r} expected='per_pair_random'"
                )
            validation = row.get("validation")
            if not isinstance(validation, dict):
                errors.append(f"seed{seed} {role} missing validation metadata")
            elif validation.get("key_schedule") != "per_pair_random":
                errors.append(
                    f"seed{seed} {role} validation key_schedule="
                    f"{validation.get('key_schedule')!r} expected='per_pair_random'"
                )
            final = row.get("final_evaluation")
            if not isinstance(final, dict):
                errors.append(f"seed{seed} {role} missing final_evaluation")
            elif not isinstance(final.get("auc_mean"), (int, float)):
                errors.append(f"seed{seed} {role} missing final auc_mean")
            else:
                scores_by_seed[seed][role] = float(final["auc_mean"])
                metrics_by_repeat = final.get("metrics_by_repeat")
                if final.get("repeats") != expected_final_repeats:
                    errors.append(
                        f"seed{seed} {role} final repeats={final.get('repeats')!r} "
                        f"expected={expected_final_repeats}"
                    )
                if not isinstance(metrics_by_repeat, list) or len(
                    metrics_by_repeat
                ) != expected_final_repeats:
                    errors.append(
                        f"seed{seed} {role} final metrics_by_repeat must contain "
                        f"{expected_final_repeats} rows"
                    )
            for count_field in ("parameter_count", "trainable_parameter_count"):
                count = row.get(count_field)
                if not isinstance(count, int):
                    errors.append(f"seed{seed} {role} missing {count_field}")
                    continue
                parameter_counts[seed].setdefault(role, {})[count_field] = count

        if {"typed_true", "typed_shuffled"}.issubset(parameter_counts[seed]):
            for count_field in ("parameter_count", "trainable_parameter_count"):
                true_count = parameter_counts[seed]["typed_true"][count_field]
                shuffled_count = parameter_counts[seed]["typed_shuffled"][count_field]
                if true_count != shuffled_count:
                    errors.append(
                        f"seed{seed} Feistel topology control {count_field} mismatch: "
                        f"true={true_count} shuffled={shuffled_count}"
                    )

    if errors:
        return {
            "status": "fail",
            "decision": "invalid_feistel_des_protocol",
            "errors": errors,
            "alignment": alignment,
            "research_decision_applied": False,
        }
    if expected_samples_per_class < 2048:
        return {
            "status": "pass",
            "decision": "feistel_des_readiness_passed",
            "next_action": "run_frozen_2048_class_two_seed_local_gate",
            "errors": [],
            "alignment": alignment,
            "research_decision_applied": False,
            "samples_per_class": expected_samples_per_class,
            "seeds": list(expected_seeds),
            "epochs": expected_epochs,
            "models": MODEL_ROLES,
            "parameter_counts": parameter_counts,
            "scores_by_seed": scores_by_seed,
            "claim_scope": (
                "readiness-only DES pipeline validation; metrics are not research "
                "evidence and no architecture or topology decision is applied"
            ),
        }
    decision = feistel_des_decision(scores_by_seed)
    return {
        "status": "pass",
        "errors": [],
        "alignment": alignment,
        "research_decision_applied": True,
        "samples_per_class": expected_samples_per_class,
        "seeds": list(expected_seeds),
        "epochs": expected_epochs,
        "models": MODEL_ROLES,
        "parameter_counts": parameter_counts,
        "claim_scope": (
            "local DES r6 Feistel architecture/attribution diagnostic under the "
            "project's strict negative protocol; not paper-scale, SOTA, or a "
            "cross-Feistel conclusion"
        ),
        "stopped_actions": [
            "des_r7_staged_training",
            "paper_scale_5000000_per_class",
            "simon_or_sm4_generalization_claim",
        ],
        **decision,
    }


__all__ = [
    "MODEL_ROLES",
    "feistel_des_decision",
    "gate_feistel_des_results",
]
