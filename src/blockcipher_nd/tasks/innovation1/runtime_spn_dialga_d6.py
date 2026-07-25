from __future__ import annotations

import math
from pathlib import Path
from typing import Any


D6_MODELS = {
    "correct": "runtime_spn_e5_gated_residual_true",
    "corrupted": "runtime_spn_e5_gated_residual_corrupted",
    "no_topology": "runtime_spn_e5_gated_residual_independent",
}
D3_MODEL = "runtime_spn_e4_equivariant_true"
D3_RUN_ID = "i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725"
D3_HOLD_DECISION = "innovation1_dialga_runtime_e4_d3_adjacent_window_not_replicated"
SEEDS = (0, 1)
EXPECTED_PARAMETER_COUNT = 492_644
CORRECT_AUC_FLOOR = 0.520
CONTROL_MARGIN = 0.005
D3_IMPROVEMENT_MARGIN = 0.010
TOPOLOGY_RESIDUAL_MODE = (
    "independent_base_plus_bounded_topology_logit_residual"
)


def adjudicate_runtime_spn_dialga_d6(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    d3_rows: list[dict[str, Any]],
    persisted_d3_gate: dict[str, Any],
    replayed_d3_gate: dict[str, Any],
    d3_validation: dict[str, Any],
    expected_cache_root: str | Path,
    cache_audit: dict[str, Any],
) -> dict[str, Any]:
    by_key = {
        (int(row.get("seed", -1)), str(row.get("model", ""))): row
        for row in rows
    }
    expected_keys = {
        (seed, model) for seed in SEEDS for model in D6_MODELS.values()
    }
    groups = {
        seed: {
            role: by_key.get((seed, model), {})
            for role, model in D6_MODELS.items()
        }
        for seed in SEEDS
    }
    flat_rows = [groups[seed][role] for seed in SEEDS for role in D6_MODELS]
    reference = groups[0]["correct"]
    d3_anchors = _d3_correct_anchors(d3_rows)

    static_fields = (
        "cipher",
        "cipher_key",
        "rounds",
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
        "checkpoint_metric",
        "restore_best_checkpoint",
        "selected_checkpoint",
        "train_rows",
        "validation_rows",
        "input_bits",
        "pair_bits",
    )
    protocol_checks = {
        "six_rows_two_seeds_complete": len(rows) == 6
        and set(by_key) == expected_keys,
        "same_data_protocol": all(
            row
            and all(row.get(field) == reference.get(field) for field in static_fields)
            for row in flat_rows
        ),
        "same_training_protocol": all(
            row
            and all(
                row.get("training", {}).get(field)
                == reference.get("training", {}).get(field)
                for field in training_fields
            )
            for row in flat_rows
        ),
        "frozen_d6_task_and_scale": all(
            row.get("cipher") == "Dialga-128"
            and row.get("cipher_key") == "dialga128"
            and row.get("rounds") == 5
            and row.get("samples_per_class") == 2048
            and row.get("pairs_per_sample") == 4
            and row.get("input_difference") == 0x40
            and row.get("train_key") == 0
            and row.get("validation_key") == int("11" * 32, 16)
            and row.get("training", {}).get("train_rows") == 4096
            and row.get("training", {}).get("validation_rows") == 2048
            and row.get("training", {}).get("epochs") == 10
            and row.get("training", {}).get("input_bits") == 1024
            and row.get("training", {}).get("pair_bits") == 256
            for row in flat_rows
        ),
        "strict_encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts"
            for row in flat_rows
        ),
        "raw_independent_ciphertext_pairs": all(
            row.get("feature_encoding") == "ciphertext_pair_bits"
            and row.get("sample_structure") == "independent_pairs"
            for row in flat_rows
        ),
        "exact_d6_model_options": _model_options_exact(groups),
        "equal_expected_parameter_geometry": all(
            row.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and row.get("trainable_parameter_count") == EXPECTED_PARAMETER_COUNT
            for row in flat_rows
        ),
        "runtime_descriptor_contract": _runtime_descriptor_contract(groups),
        "runtime_topology_controls_distinct": _topology_controls_distinct(groups),
        "gated_residual_metadata_complete": all(
            _valid_gate_metadata(row) for row in flat_rows
        ),
        "best_validation_checkpoints_restored": all(
            row.get("training", {}).get("checkpoint_metric") == "val_auc"
            and row.get("training", {}).get("restore_best_checkpoint") is True
            and row.get("training", {}).get("selected_checkpoint") == "best"
            for row in flat_rows
        ),
        "complete_best_checkpoint_histories": all(
            _history_matches_result(row) for row in flat_rows
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            for row in flat_rows
        ),
        "exact_d3_cache_root_recorded": all(
            _same_path(
                row.get("training", {}).get("dataset_cache_root"),
                expected_cache_root,
            )
            for row in flat_rows
        ),
        "d3_cache_reused_without_generation": cache_audit.get("status") == "pass",
        "finite_auc_metrics": all(
            math.isfinite(float(row.get("metrics", {}).get("auc", math.nan)))
            for row in flat_rows
        ),
        "d3_source_rows_complete": len(d3_rows) == 6
        and set(d3_anchors) == set(SEEDS),
        "d3_source_gate_recomputed_exactly": persisted_d3_gate == replayed_d3_gate,
        "d3_source_validation_passed": (
            d3_validation.get("run_id") == D3_RUN_ID
            and d3_validation.get("status") == "pass"
            and d3_validation.get("checks")
            == persisted_d3_gate.get("protocol_checks")
        ),
        "d3_source_is_exact_completed_hold": (
            persisted_d3_gate.get("run_id") == D3_RUN_ID
            and persisted_d3_gate.get("status") == "hold"
            and persisted_d3_gate.get("decision") == D3_HOLD_DECISION
            and all(persisted_d3_gate.get("protocol_checks", {}).values())
        ),
    }

    aucs = {
        f"seed{seed}": {
            role: float(groups[seed][role].get("metrics", {}).get("auc", math.nan))
            for role in D6_MODELS
        }
        for seed in SEEDS
    }
    margins = {
        f"seed{seed}": {
            "correct_minus_corrupted": aucs[f"seed{seed}"]["correct"]
            - aucs[f"seed{seed}"]["corrupted"],
            "correct_minus_no_topology": aucs[f"seed{seed}"]["correct"]
            - aucs[f"seed{seed}"]["no_topology"],
            "correct_minus_d3": aucs[f"seed{seed}"]["correct"]
            - d3_anchors.get(seed, math.nan),
        }
        for seed in SEEDS
    }
    research_checks: dict[str, bool] = {}
    for seed in SEEDS:
        seed_key = f"seed{seed}"
        research_checks[f"{seed_key}_correct_auc_at_least_0p520"] = (
            aucs[seed_key]["correct"] >= CORRECT_AUC_FLOOR
        )
        research_checks[f"{seed_key}_correct_exceeds_corrupted_by_0p005"] = (
            margins[seed_key]["correct_minus_corrupted"] >= CONTROL_MARGIN
        )
        research_checks[f"{seed_key}_correct_exceeds_no_topology_by_0p005"] = (
            margins[seed_key]["correct_minus_no_topology"] >= CONTROL_MARGIN
        )
        research_checks[f"{seed_key}_correct_improves_d3_by_0p010"] = (
            margins[seed_key]["correct_minus_d3"] >= D3_IMPROVEMENT_MARGIN
        )

    architecture_improves_both_seeds = all(
        research_checks[f"seed{seed}_correct_improves_d3_by_0p010"]
        for seed in SEEDS
    )
    if not all(protocol_checks.values()):
        status = "fail"
        decision = "innovation1_dialga_runtime_e5_d6_protocol_invalid"
        next_action = "repair only the failed frozen D6 protocol check and rerun unchanged"
    elif all(research_checks.values()):
        status = "pass"
        decision = "innovation1_dialga_runtime_e5_d6_gated_residual_supported"
        next_action = (
            "freeze both correct D6 best checkpoints and run a training-free same-"
            "checkpoint correct/corrupted/no-topology swap before any scale increase"
        )
    elif architecture_improves_both_seeds:
        status = "hold"
        decision = (
            "innovation1_dialga_runtime_e5_d6_base_improvement_without_topology_attribution"
        )
        next_action = (
            "retain only the independent-base improvement and audit learned gates with "
            "same-checkpoint topology interventions; do not scale or claim topology use"
        )
    else:
        status = "hold"
        decision = "innovation1_dialga_runtime_e5_d6_not_supported"
        next_action = (
            "stop the Dialga prefix-r5 E5 branch and run the unchanged E5 architecture "
            "once on the strong D1 r4 mechanism as a regression before retain/discard"
        )

    learned_gates = {
        f"seed{seed}": {
            role: {
                "raw": float(groups[seed][role].get("topology_gate_final_raw", math.nan)),
                "bounded": float(
                    groups[seed][role].get("topology_gate_final_bounded", math.nan)
                ),
            }
            for role in D6_MODELS
        }
        for seed in SEEDS
    }
    return {
        "run_id": run_id,
        "task": "innovation1_dialga128_runtime_e5_d6_gated_residual",
        "cipher": "Dialga-128",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "aucs": aucs,
        "margins": margins,
        "d3_correct_anchors": {
            f"seed{seed}": d3_anchors.get(seed, math.nan) for seed in SEEDS
        },
        "learned_topology_gates": learned_gates,
        "cache_audit": cache_audit,
        "thresholds": {
            "correct_auc": CORRECT_AUC_FLOOR,
            "control_margin": CONTROL_MARGIN,
            "d3_improvement_margin": D3_IMPROVEMENT_MARGIN,
        },
        "claim_scope": (
            "Dialga-128 prefix-r5 two-seed local 2048/class gated-residual "
            "architecture diagnostic only; not formal scale, attack, paper "
            "reproduction, SOTA, or universal-SPN evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "remote GPU or sample scale-up",
            "change the input difference, pairs, keys, epochs, negatives, or metric",
            "claim topology attribution without both per-seed control margins",
            "resume mechanical single-bit difference searching",
        ],
    }


def _d3_correct_anchors(rows: list[dict[str, Any]]) -> dict[int, float]:
    anchors: dict[int, float] = {}
    for row in rows:
        if row.get("model") != D3_MODEL:
            continue
        seed = int(row.get("seed", -1))
        if seed not in SEEDS or seed in anchors:
            return {}
        try:
            auc = float(row["metrics"]["auc"])
        except (KeyError, TypeError, ValueError):
            return {}
        if not math.isfinite(auc):
            return {}
        anchors[seed] = auc
    return anchors


def _model_options_exact(
    groups: dict[int, dict[str, dict[str, Any]]],
) -> bool:
    common = {
        "runtime_structure_path": "configs/runtime/spn/dialga128.json",
        "runtime_round_start": 3,
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet",
        "round_window_mode": "recurrent_window",
        "runtime_structure_window_control": "full",
    }
    for seed in SEEDS:
        for role in D6_MODELS:
            expected = dict(common)
            if role == "corrupted":
                expected["topology_corruption_seed"] = 20260725
            if groups[seed][role].get("training", {}).get("model_options") != expected:
                return False
    return True


def _runtime_descriptor_contract(
    groups: dict[int, dict[str, dict[str, Any]]],
) -> bool:
    expected_modes = {
        "correct": "true",
        "corrupted": "corrupted",
        "no_topology": "independent",
    }
    descriptor_hashes: set[str] = set()
    for seed in SEEDS:
        for role in D6_MODELS:
            row = groups[seed][role]
            descriptor_hashes.add(str(row.get("runtime_structure_descriptor_sha256")))
            if not (
                row.get("runtime_structure_descriptor_name")
                == "Dialga-128 20-round heterogeneous runtime SPN structure"
                and str(row.get("runtime_structure_descriptor_path", "")).endswith(
                    "configs/runtime/spn/dialga128.json"
                )
                and row.get("runtime_structure_round_start") == 3
                and row.get("runtime_structure_available_rounds") == 20
                and row.get("runtime_structure_loaded_rounds") == 2
                and row.get("runtime_structure_unique_transition_count") == 2
                and row.get("runtime_structure_homogeneous") is False
                and row.get("runtime_structure_mode") == expected_modes[role]
                and row.get("runtime_structure_window_control") == "full"
                and len(row.get("runtime_structure_transition_sha256s", [])) == 2
                and len(str(row.get("runtime_structure_window_sha256", ""))) == 64
            ):
                return False
    return len(descriptor_hashes) == 1 and len(next(iter(descriptor_hashes), "")) == 64


def _topology_controls_distinct(
    groups: dict[int, dict[str, dict[str, Any]]],
) -> bool:
    hashes: dict[str, set[str]] = {role: set() for role in D6_MODELS}
    for seed in SEEDS:
        for role in D6_MODELS:
            hashes[role].add(
                str(groups[seed][role].get("runtime_structure_window_sha256", ""))
            )
    return (
        all(len(values) == 1 for values in hashes.values())
        and hashes["correct"] == hashes["no_topology"]
        and hashes["correct"] != hashes["corrupted"]
    )


def _valid_gate_metadata(row: dict[str, Any]) -> bool:
    try:
        initial = float(row.get("topology_gate_initial", math.nan))
        raw = float(row.get("topology_gate_final_raw", math.nan))
        bounded = float(row.get("topology_gate_final_bounded", math.nan))
    except (TypeError, ValueError):
        return False
    return (
        row.get("topology_residual_mode") == TOPOLOGY_RESIDUAL_MODE
        and initial == 0.0
        and math.isfinite(raw)
        and math.isfinite(bounded)
        and abs(math.tanh(raw) - bounded) <= 1e-7
        and abs(bounded) <= 1.0
    )


def _history_matches_result(row: dict[str, Any]) -> bool:
    history = row.get("history")
    if not isinstance(history, list) or len(history) != 10:
        return False
    try:
        best_auc = max(float(epoch["val_auc"]) for epoch in history)
        result_auc = float(row["metrics"]["auc"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(best_auc) and abs(best_auc - result_auc) <= 1e-12


def _same_path(left: Any, right: str | Path) -> bool:
    if not isinstance(left, str) or not left:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


__all__ = [
    "CONTROL_MARGIN",
    "CORRECT_AUC_FLOOR",
    "D3_HOLD_DECISION",
    "D3_IMPROVEMENT_MARGIN",
    "D3_RUN_ID",
    "D6_MODELS",
    "EXPECTED_PARAMETER_COUNT",
    "SEEDS",
    "TOPOLOGY_RESIDUAL_MODE",
    "adjudicate_runtime_spn_dialga_d6",
]
