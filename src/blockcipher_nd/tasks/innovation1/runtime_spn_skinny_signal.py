from __future__ import annotations

import math
from typing import Any


MODEL = "skinny64_runtime_e4_equivariant_true"
PROFILE = "skinny64_gohr2022_single_key"
SIGNAL_FLOOR = 0.55


def adjudicate_runtime_spn_skinny_signal(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    expected_rounds: tuple[int, ...],
    expected_seed: int,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"screen", "confirmation"}:
        raise ValueError("phase must be screen or confirmation")
    by_round = {int(row.get("rounds", -1)): row for row in rows}
    reference = rows[0] if rows else {}
    static_fields = (
        "cipher",
        "seed",
        "samples_per_class",
        "dataset_label_mode",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "input_difference",
        "train_key",
        "validation_key",
        "parameter_count",
        "trainable_parameter_count",
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
        "model_options",
    )
    protocol_checks = {
        "expected_round_rows_complete": (
            len(rows) == len(expected_rounds)
            and set(by_round) == set(expected_rounds)
        ),
        "correct_runtime_model_only": all(row.get("model") == MODEL for row in rows),
        "same_protocol_except_rounds": all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in rows
        ),
        "same_training_budget": all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in rows
        ),
        "frozen_fixed_key_signal_protocol": all(
            row.get("cipher") == "SKINNY-64/64"
            and row.get("seed") == expected_seed
            and row.get("samples_per_class") == 512
            and row.get("pairs_per_sample") == 4
            and row.get("difference_profile") == PROFILE
            and row.get("input_difference") == 0x2000
            and row.get("train_key") == 0
            and row.get("validation_key") == 0x1111111111111111
            and row.get("training", {}).get("train_rows") == 1024
            and row.get("training", {}).get("validation_rows") == 512
            and row.get("training", {}).get("epochs") == 5
            for row in rows
        ),
        "strict_encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in rows
        ),
        "raw_independent_ciphertext_pairs": all(
            row.get("feature_encoding") == "ciphertext_pair_bits"
            and row.get("sample_structure") == "independent_pairs"
            for row in rows
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in rows
        ),
        "finite_auc_metrics": bool(rows)
        and all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in rows
        ),
    }
    aucs = {
        str(rounds): float(by_round[rounds].get("metrics", {}).get("auc", math.nan))
        for rounds in expected_rounds
        if rounds in by_round
    }
    passing_rounds = sorted(
        rounds for rounds in expected_rounds if aucs.get(str(rounds), 0.0) >= SIGNAL_FLOOR
    )
    selected_round = max(passing_rounds) if passing_rounds else None
    research_checks = {
        f"r{rounds}_auc_at_least_0p55": aucs.get(str(rounds), 0.0) >= SIGNAL_FLOOR
        for rounds in expected_rounds
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_runtime_spn_skinny_signal_protocol_invalid"
        next_action = "repair only the failed T2-B protocol check before interpretation"
    elif selected_round is None:
        status = "hold"
        decision = "innovation1_runtime_spn_skinny_signal_anchor_not_supported"
        next_action = "stop SKINNY topology training; redesign the signal protocol or representation"
    elif phase == "screen":
        status = "pass"
        decision = "innovation1_runtime_spn_skinny_signal_anchor_selected"
        next_action = f"repeat only r{selected_round} at the same budget with fresh seed1"
    else:
        status = "pass"
        decision = "innovation1_runtime_spn_skinny_signal_anchor_confirmed"
        next_action = (
            f"open the frozen r{selected_round} 2048/class two-seed correct/corrupted/"
            "no-topology attribution gate"
        )
    return {
        "run_id": run_id,
        "cipher": "SKINNY-64/64",
        "phase": phase,
        "seed": expected_seed,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs_by_round": aucs,
        "signal_floor": SIGNAL_FLOOR,
        "passing_rounds": passing_rounds,
        "selected_round": selected_round,
        "claim_scope": (
            "local 512/class-train, 256/class-validation fixed-key SKINNY signal "
            "diagnostic only; no topology attribution, formal scale, paper reproduction, "
            "attack, or SOTA claim"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "claim general-GF(2) topology superiority",
            "remote launch or mechanical scale-up",
            "change difference, features, pairs, epochs, or negative definition",
        ],
    }


__all__ = [
    "MODEL",
    "PROFILE",
    "SIGNAL_FLOOR",
    "adjudicate_runtime_spn_skinny_signal",
]
