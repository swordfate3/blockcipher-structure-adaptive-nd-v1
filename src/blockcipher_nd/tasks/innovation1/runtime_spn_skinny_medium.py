from __future__ import annotations

import math
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_attribution import MODELS


SAMPLES_PER_CLASS = 65_536
TRAIN_ROWS = 131_072
VALIDATION_ROWS = 65_536
SIGNAL_FLOOR = 0.55
CONTROL_MARGIN = 0.005
RUN_STEM = "i1_rtg2a_skinny64_general_gf2_medium_65536"
RTG2B_RUN_STEM = "i1_rtg2b_skinny64_general_gf2_scale_262144"
HISTORY_EPOCHS = 5
RTG2B_SAMPLES_PER_CLASS = 262_144
RTG2B_TRAIN_ROWS = 524_288
RTG2B_VALIDATION_ROWS = 262_144
RTG3A_RUN_STEM = "i1_rtg3a_skinny64_general_gf2_formal_1000000"
RTG3A_SAMPLES_PER_CLASS = 1_000_000
RTG3A_TRAIN_ROWS = 2_000_000
RTG3A_VALIDATION_ROWS = 1_000_000


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _has_finite_metric_map(
    gate: dict[str, Any],
    field: str,
    keys: tuple[str, ...],
) -> bool:
    values = gate.get(field)
    return isinstance(values, dict) and all(
        _is_finite_number(values.get(key)) for key in keys
    )


def _tightly_equal(left: object, right: object, *, tolerance: float = 1e-7) -> bool:
    return (
        _is_finite_number(left)
        and _is_finite_number(right)
        and abs(float(left) - float(right)) <= tolerance
    )


def _checkpoint_dynamics(row: dict[str, Any]) -> dict[str, Any] | None:
    history = row.get("history")
    training = row.get("training")
    metrics = row.get("metrics")
    if (
        not isinstance(history, list)
        or len(history) != HISTORY_EPOCHS
        or not isinstance(training, dict)
        or not isinstance(metrics, dict)
    ):
        return None

    required_metrics = (
        "learning_rate",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
    )
    for expected_epoch, item in enumerate(history, start=1):
        if not isinstance(item, dict) or item.get("epoch") != expected_epoch:
            return None
        if any(not _is_finite_number(item.get(field)) for field in required_metrics):
            return None

    best_index = max(
        range(HISTORY_EPOCHS),
        key=lambda index: float(history[index]["val_auc"]),
    )
    best_epoch = best_index + 1
    best_row = history[best_index]
    if (
        type(training.get("epochs_ran")) is not int
        or training["epochs_ran"] != HISTORY_EPOCHS
        or type(training.get("best_epoch")) is not int
        or training["best_epoch"] != best_epoch
        or training.get("checkpoint_metric") != "val_auc"
        or training.get("restore_best_checkpoint") is not True
        or training.get("selected_checkpoint") != "best"
        or training.get("stopped_epoch") != 0
        or not _tightly_equal(
            training.get("best_checkpoint_metric"), best_row["val_auc"]
        )
        or not _tightly_equal(metrics.get("auc"), best_row["val_auc"])
    ):
        return None

    first_val_auc = float(history[0]["val_auc"])
    best_val_auc = float(best_row["val_auc"])
    final_val_auc = float(history[-1]["val_auc"])
    best_train_auc = float(best_row["train_auc"])
    return {
        "model": row.get("model"),
        "best_epoch": best_epoch,
        "epochs_ran": HISTORY_EPOCHS,
        "first_val_auc": first_val_auc,
        "best_val_auc": best_val_auc,
        "final_val_auc": final_val_auc,
        "best_train_auc": best_train_auc,
        "train_minus_val_auc_at_best": best_train_auc - best_val_auc,
        "first_to_best_val_auc_gain": best_val_auc - first_val_auc,
        "best_to_final_val_auc_change": final_val_auc - best_val_auc,
    }


def adjudicate_runtime_spn_skinny_medium(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    expected_seed: int,
    phase: str = "rtg2a",
) -> dict[str, Any]:
    if phase not in {"rtg2a", "rtg2b", "rtg3a"}:
        raise ValueError("SKINNY scale phase must be rtg2a, rtg2b, or rtg3a")
    if expected_seed not in {0, 1}:
        raise ValueError(f"{phase.upper()} supports only seed0 and conditional seed1")
    if phase == "rtg2a":
        samples_per_class = SAMPLES_PER_CLASS
        train_rows = TRAIN_ROWS
        validation_rows = VALIDATION_ROWS
        scale_check = "frozen_rtg2a_scale_and_task"
    elif phase == "rtg2b":
        samples_per_class = RTG2B_SAMPLES_PER_CLASS
        train_rows = RTG2B_TRAIN_ROWS
        validation_rows = RTG2B_VALIDATION_ROWS
        scale_check = "frozen_rtg2b_scale_and_task"
    else:
        samples_per_class = RTG3A_SAMPLES_PER_CLASS
        train_rows = RTG3A_TRAIN_ROWS
        validation_rows = RTG3A_VALIDATION_ROWS
        scale_check = "frozen_rtg3a_formal_scale_and_task"

    by_model = {str(row.get("model")): row for row in rows}
    by_role = {
        role: by_model[model] for role, model in MODELS.items() if model in by_model
    }
    reference = by_role.get("true", {})
    static_fields = (
        "cipher",
        "rounds",
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
    )
    training_fields = (
        "epochs",
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "train_eval_interval",
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
        "model_options",
        "dataset_cache_root",
        "dataset_cache_chunk_size",
        "dataset_cache_workers",
    )
    cache_roots = {
        str(row.get("training", {}).get("dataset_cache_root") or "")
        for row in by_role.values()
    }
    training_dynamics = {
        role: _checkpoint_dynamics(row) for role, row in by_role.items()
    }
    protocol_checks = {
        "three_runtime_controls_complete": (
            len(rows) == 3 and set(by_model) == set(MODELS.values())
        ),
        "same_data_protocol": len(by_role) == 3
        and all(
            all(row.get(field) == reference.get(field) for field in static_fields)
            for row in by_role.values()
        ),
        "same_training_protocol": len(by_role) == 3
        and all(
            all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in by_role.values()
        ),
        scale_check: len(by_role) == 3
        and all(
            row.get("cipher") == "SKINNY-64/64"
            and row.get("rounds") == 7
            and row.get("seed") == expected_seed
            and row.get("samples_per_class") == samples_per_class
            and row.get("pairs_per_sample") == 4
            and row.get("difference_profile") == "skinny64_gohr2022_single_key"
            and row.get("input_difference") == 0x2000
            and row.get("training", {}).get("train_rows") == train_rows
            and row.get("training", {}).get("validation_rows") == validation_rows
            and row.get("training", {}).get("epochs") == 5
            and row.get("training", {}).get("batch_size") == 64
            and row.get("training", {}).get("train_eval_interval") == 1
            for row in by_role.values()
        ),
        "strict_encrypted_random_plaintext_negatives": len(by_role) == 3
        and all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in by_role.values()
        ),
        "raw_independent_ciphertext_pairs": len(by_role) == 3
        and all(
            row.get("feature_encoding") == "ciphertext_pair_bits"
            and row.get("sample_structure") == "independent_pairs"
            for row in by_role.values()
        ),
        "equal_parameter_geometry": len(by_role) == 3
        and len(
            {
                (
                    int(row.get("parameter_count", -1)),
                    int(row.get("trainable_parameter_count", -1)),
                )
                for row in by_role.values()
            }
        )
        == 1,
        "runtime_bit_order_adapter_recorded": len(by_role) == 3
        and all(
            row.get("input_bit_order") == "project_msb_to_runtime_lsb"
            for row in by_role.values()
        ),
        "disk_backed_parameter_matched_cache": len(by_role) == 3
        and len(cache_roots) == 1
        and next(iter(cache_roots), "").startswith(
            "G:\\lxy\\blockcipher-structure-adaptive-nd-runs\\"
        )
        and all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            and row.get("training", {}).get("dataset_cache_chunk_size") == 1024
            and row.get("training", {}).get("dataset_cache_workers") == 1
            for row in by_role.values()
        ),
        "finite_auc_metrics": len(by_role) == 3
        and all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in by_role.values()
        ),
        "complete_five_epoch_checkpoint_replay": len(training_dynamics) == 3
        and all(summary is not None for summary in training_dynamics.values()),
    }
    aucs = {
        role: float(row.get("metrics", {}).get("auc", math.nan))
        for role, row in by_role.items()
    }
    margins = {
        "true_minus_corrupted": aucs.get("true", math.nan)
        - aucs.get("corrupted", math.nan),
        "true_minus_independent": aucs.get("true", math.nan)
        - aucs.get("independent", math.nan),
    }
    research_checks = {
        "true_auc_at_least_0p55": aucs.get("true", 0.0) >= SIGNAL_FLOOR,
        "true_exceeds_corrupted_by_0p005": (
            margins["true_minus_corrupted"] >= CONTROL_MARGIN
        ),
        "true_exceeds_independent_by_0p005": (
            margins["true_minus_independent"] >= CONTROL_MARGIN
        ),
    }
    if not all(protocol_checks.values()):
        status = "fail"
        decision = {
            "rtg2a": "innovation1_rtg2a_skinny_medium_protocol_invalid",
            "rtg2b": "innovation1_rtg2b_skinny_scale_protocol_invalid",
            "rtg3a": "innovation1_rtg3a_skinny_formal_protocol_invalid",
        }[phase]
        next_action = (
            f"repair only the failed {phase.upper()} protocol check; do not "
            "interpret or rerun at larger scale"
        )
    elif all(research_checks.values()):
        status = "pass"
        if phase == "rtg2a":
            decision = f"innovation1_rtg2a_skinny_medium_seed{expected_seed}_supported"
            next_action = (
                "run the identical conditional seed1 RTG2-A matrix"
                if expected_seed == 0
                else "synthesize the two-seed medium evidence before considering 262144/class"
            )
        elif phase == "rtg2b":
            decision = f"innovation1_rtg2b_skinny_scale_seed{expected_seed}_supported"
            next_action = (
                "prepare an identical 262144/class seed1 RTG2-B confirmation without launching it automatically"
                if expected_seed == 0
                else "synthesize the two-seed 262144/class evidence before any formal-scale decision"
            )
        else:
            decision = f"innovation1_rtg3a_skinny_formal_seed{expected_seed}_supported"
            next_action = (
                "prepare the identical conditional 1000000/class seed1 RTG3-A confirmation; do not make a two-seed formal claim yet"
                if expected_seed == 0
                else "synthesize the two-seed project-formal evidence before any cross-cipher generalization claim"
            )
    else:
        status = "hold"
        decision = {
            "rtg2a": "innovation1_rtg2a_skinny_medium_not_supported",
            "rtg2b": "innovation1_rtg2b_skinny_scale_not_supported",
            "rtg3a": "innovation1_rtg3a_skinny_formal_not_supported",
        }[phase]
        next_action = (
            f"stop {phase.upper()} scale-up and audit whether the prior margin was sample variance or a "
            "training-dynamics mismatch; do not rescue it by changing the frozen protocol"
        )

    blocked_actions = [
        "claim formal, paper-scale, attack, SOTA, breakthrough, or universal-SPN evidence",
        "change difference, pairs, epochs, negatives, topology corruption, or model geometry",
        "add DDT or trail features, switch to related keys, or launch a broad cipher matrix",
        {
            "rtg2a": "launch 262144/class before both RTG2-A medium seeds pass",
            "rtg2b": "launch seed1 or formal scale before RTG2-B seed0 passes",
            "rtg3a": "launch RTG3-A seed1 before the project-formal seed0 gate passes",
        }[phase],
    ]
    if expected_seed == 0 and status != "pass":
        blocked_actions.append("launch seed1 after a seed0 hold or protocol failure")

    return {
        "run_id": run_id,
        "cipher": "SKINNY-64/64",
        "phase": phase,
        "seed": expected_seed,
        "samples_per_class": samples_per_class,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "training_dynamics": training_dynamics,
        "thresholds": {
            "true_auc": SIGNAL_FLOOR,
            "control_margin": CONTROL_MARGIN,
        },
        "claim_scope": (
            f"SKINNY-64/64 r7 seed{expected_seed} remote project-formal "
            f"{samples_per_class}/class general-GF(2) runtime-topology replication; "
            "single-seed evidence only, not a paper reproduction, attack, SOTA, "
            "breakthrough, universal-SPN, or multi-seed formal conclusion"
            if phase == "rtg3a"
            else f"SKINNY-64/64 r7 seed{expected_seed} remote medium "
            f"{samples_per_class}/class general-GF(2) runtime-topology replication; "
            "architecture/protocol diagnostic only, not formal scale, paper "
            "reproduction, attack, SOTA, breakthrough, or universal-SPN evidence"
        ),
        "next_action": next_action,
        "blocked_actions": blocked_actions,
    }


def adjudicate_runtime_spn_skinny_medium_joint(
    *,
    run_id: str,
    gates: list[dict[str, Any]],
    phase: str = "rtg2a",
) -> dict[str, Any]:
    if phase not in {"rtg2a", "rtg2b", "rtg3a"}:
        raise ValueError("SKINNY joint scale phase must be rtg2a, rtg2b, or rtg3a")
    if phase == "rtg2a":
        run_stem = RUN_STEM
        samples_per_class = SAMPLES_PER_CLASS
        decision_stem = "innovation1_rtg2a_skinny_medium"
        run_id_check = "source_run_ids_match_frozen_rtg2a"
        research_check = "both_medium_seeds_supported"
    elif phase == "rtg2b":
        run_stem = RTG2B_RUN_STEM
        samples_per_class = RTG2B_SAMPLES_PER_CLASS
        decision_stem = "innovation1_rtg2b_skinny_scale"
        run_id_check = "source_run_ids_match_frozen_rtg2b"
        research_check = "both_medium_seeds_supported"
    else:
        run_stem = RTG3A_RUN_STEM
        samples_per_class = RTG3A_SAMPLES_PER_CLASS
        decision_stem = "innovation1_rtg3a_skinny_formal"
        run_id_check = "source_run_ids_match_frozen_rtg3a"
        research_check = "both_formal_seeds_supported"

    by_seed = {
        int(gate.get("seed", -1)): gate
        for gate in gates
        if isinstance(gate.get("seed"), int)
    }
    complete = len(gates) == 2 and set(by_seed) == {0, 1}

    expected_decisions = {
        seed: {
            "pass": f"{decision_stem}_seed{seed}_supported",
            "hold": f"{decision_stem}_not_supported",
            "fail": f"{decision_stem}_protocol_invalid",
        }
        for seed in (0, 1)
    }
    source_statuses = {
        seed: str(by_seed[seed].get("status", "")) for seed in (0, 1) if seed in by_seed
    }
    protocol_checks = {
        "two_distinct_seed_gates_complete": complete,
        run_id_check: complete
        and all(
            str(by_seed[seed].get("run_id", "")).startswith(f"{run_stem}_seed{seed}_")
            for seed in (0, 1)
        ),
        "source_phases_match_joint_phase": complete
        and all(by_seed[seed].get("phase") == phase for seed in (0, 1)),
        "source_protocol_checks_passed": complete
        and all(
            isinstance(by_seed[seed].get("protocol_checks"), dict)
            and bool(by_seed[seed]["protocol_checks"])
            and all(bool(value) for value in by_seed[seed]["protocol_checks"].values())
            for seed in (0, 1)
        ),
        "source_gate_contracts_consistent": complete
        and all(
            source_statuses[seed] in expected_decisions[seed]
            and str(by_seed[seed].get("decision", ""))
            == expected_decisions[seed][source_statuses[seed]]
            for seed in (0, 1)
        ),
        "frozen_thresholds_preserved": complete
        and all(
            by_seed[seed].get("thresholds")
            == {"true_auc": SIGNAL_FLOOR, "control_margin": CONTROL_MARGIN}
            for seed in (0, 1)
        ),
        "finite_source_metrics": complete
        and all(
            _has_finite_metric_map(
                by_seed[seed], "aucs", ("true", "corrupted", "independent")
            )
            and _has_finite_metric_map(
                by_seed[seed],
                "margins",
                ("true_minus_corrupted", "true_minus_independent"),
            )
            for seed in (0, 1)
        ),
    }
    research_checks = {
        research_check: complete
        and all(source_statuses[seed] == "pass" for seed in (0, 1))
    }

    if not all(protocol_checks.values()):
        status = "fail"
        decision = f"{decision_stem}_joint_protocol_invalid"
        next_action = (
            "repair only the missing or inconsistent seed gate evidence; do not "
            "interpret the pair or prepare a larger run"
        )
    elif all(research_checks.values()):
        status = "pass"
        decision = f"{decision_stem}_two_seed_supported"
        if phase == "rtg2a":
            next_action = (
                "freeze an RTG2-B 262144/class seed0 plan that changes only sample scale, "
                "then pass remote disk-cache readiness before launch"
            )
        elif phase == "rtg2b":
            next_action = (
                "run a separate route-decision audit comparing formal SKINNY scale with "
                "frozen-backbone target-head cross-cipher adaptation; do not launch either "
                "route automatically"
            )
        else:
            next_action = (
                "record the completed two-seed project-formal SKINNY evidence and compare "
                "its controlled topology margins with the existing GIFT/PRESENT branches"
            )
    else:
        status = "hold"
        decision = f"{decision_stem}_two_seed_not_supported"
        next_action = (
            f"stop {phase.upper()} scale-up and audit sample variance versus training dynamics "
            "without changing the frozen model or data protocol"
        )

    source_summaries = [
        {
            "seed": seed,
            "run_id": by_seed[seed].get("run_id"),
            "status": by_seed[seed].get("status"),
            "decision": by_seed[seed].get("decision"),
            "aucs": by_seed[seed].get("aucs"),
            "margins": by_seed[seed].get("margins"),
        }
        for seed in (0, 1)
        if seed in by_seed
    ]
    return {
        "run_id": run_id,
        "phase": phase,
        "samples_per_class": samples_per_class,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "thresholds": {
            "true_auc": SIGNAL_FLOOR,
            "control_margin": CONTROL_MARGIN,
        },
        "sources": source_summaries,
        "claim_scope": (
            f"SKINNY-64/64 r7 two-seed remote project-formal "
            f"{samples_per_class}/class general-GF(2) runtime-topology replication; "
            "not a paper reproduction, attack, SOTA, breakthrough, or universal-SPN claim"
            if phase == "rtg3a"
            else f"SKINNY-64/64 r7 two-seed remote medium {samples_per_class}/class "
            "general-GF(2) runtime-topology replication synthesis; architecture/protocol "
            "diagnostic only, not formal scale, paper reproduction, attack, SOTA, "
            "breakthrough, or universal-SPN evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "claim formal, paper-scale, attack, SOTA, breakthrough, or universal-SPN evidence",
            {
                "rtg2a": "launch 262144/class unless this joint gate passes",
                "rtg2b": "launch 1000000/class without a separate formal-scale plan and gate",
                "rtg3a": "claim two-seed project-formal support unless this joint gate passes",
            }[phase],
            "change difference, pairs, epochs, negatives, topology corruption, or model geometry",
            "add DDT or trail features, switch to related keys, or launch a broad cipher matrix",
        ],
    }


__all__ = [
    "CONTROL_MARGIN",
    "HISTORY_EPOCHS",
    "RUN_STEM",
    "RTG2B_RUN_STEM",
    "RTG3A_RUN_STEM",
    "RTG3A_SAMPLES_PER_CLASS",
    "SAMPLES_PER_CLASS",
    "SIGNAL_FLOOR",
    "TRAIN_ROWS",
    "VALIDATION_ROWS",
    "adjudicate_runtime_spn_skinny_medium",
    "adjudicate_runtime_spn_skinny_medium_joint",
]
