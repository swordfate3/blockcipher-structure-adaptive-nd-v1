from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


EXPECTED_MODELS = {
    "candidate": "runtime_spn_e4_equivariant_true",
    "anchor": "runtime_spn_e4_equivariant_true",
    "shuffled": "runtime_spn_e4_equivariant_sbox_shuffled",
}
EXPECTED_SEEDS = (0, 1)
AUC_FLOOR = 0.520
MARGIN_FLOOR = 0.005


def adjudicate_uknit_sbox_assignment(
    *,
    run_id: str,
    rows: Iterable[dict[str, Any]],
    candidate_context: str = "late_cell",
    candidate_cell_input_mode: str | None = None,
    anchor_context: str = "late_pair",
    anchor_cell_input_mode: str | None = None,
) -> dict[str, Any]:
    if candidate_context not in {"late_cell", "edge_gate"}:
        raise ValueError("candidate_context must be late_cell or edge_gate")
    if anchor_context not in {"late_pair", "edge_gate"}:
        raise ValueError("anchor_context must be late_pair or edge_gate")
    for value in (candidate_cell_input_mode, anchor_cell_input_mode):
        if value not in {
            None,
            "difference_only",
            "state_triplet",
            "inverse_sbox_triplet",
            "dual_view_triplet",
        }:
            raise ValueError(
                "cell input mode must be difference_only, state_triplet, "
                "inverse_sbox_triplet, dual_view_triplet, or None"
            )
    rows = list(rows)
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row.get("seed", -1))][
            _role(
                row,
                candidate_context=candidate_context,
                candidate_cell_input_mode=candidate_cell_input_mode,
                anchor_context=anchor_context,
                anchor_cell_input_mode=anchor_cell_input_mode,
            )
        ].append(row)

    complete_roles = all(
        len(grouped[seed].get(role, ())) == 1
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_MODELS
    )
    seed_results = {str(seed): _seed_result(grouped[seed]) for seed in EXPECTED_SEEDS}
    protocol_checks = {
        "six_rows_complete": len(rows) == 6,
        "two_expected_seeds": set(grouped) == set(EXPECTED_SEEDS),
        "three_unique_roles_per_seed": complete_roles,
        "uknit_prefix_r4_protocol": _all_rows_match_uknit_protocol(rows),
        "strict_encrypted_random_plaintext_negatives": all(
            row.get("negative_mode") == "encrypted_random_plaintexts" for row in rows
        ),
        "same_data_protocol_within_seed": _same_within_seed(
            grouped,
            _data_protocol,
        ),
        "same_training_protocol_within_seed": _same_within_seed(
            grouped,
            _training_protocol,
        ),
        "disk_backed_datasets": all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            and bool(row.get("training", {}).get("dataset_cache_root"))
            for row in rows
        ),
        "equal_parameter_geometry": _equal_parameter_geometry(rows),
        "exact_descriptor_window": _exact_descriptor_window(rows),
        "role_contracts_match_plan": complete_roles
        and all(
            _role_contract(
                grouped[seed][role][0],
                role,
                candidate_context=candidate_context,
                candidate_cell_input_mode=candidate_cell_input_mode,
                anchor_context=anchor_context,
                anchor_cell_input_mode=anchor_cell_input_mode,
            )
            for seed in EXPECTED_SEEDS
            for role in EXPECTED_MODELS
        ),
        "finite_auc_metrics": all(
            _finite(row.get("metrics", {}).get("auc")) for row in rows
        ),
    }
    research_checks = {
        f"seed{seed}_candidate_auc_at_least_0p520": bool(
            seed_results[str(seed)]["candidate_auc"] is not None
            and seed_results[str(seed)]["candidate_auc"] >= AUC_FLOOR
        )
        for seed in EXPECTED_SEEDS
    }
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        research_checks[f"seed{seed}_candidate_beats_anchor_by_0p005"] = bool(
            result["candidate_minus_anchor"] is not None
            and result["candidate_minus_anchor"] >= MARGIN_FLOOR
        )
        research_checks[f"seed{seed}_candidate_beats_shuffle_by_0p005"] = bool(
            result["candidate_minus_shuffled"] is not None
            and result["candidate_minus_shuffled"] >= MARGIN_FLOOR
        )

    protocol_passed = all(protocol_checks.values())
    research_passed = all(research_checks.values())
    dual_view_triplet = candidate_cell_input_mode == "dual_view_triplet"
    inverse_sbox_triplet = candidate_cell_input_mode == "inverse_sbox_triplet"
    state_triplet = candidate_cell_input_mode == "state_triplet"
    edge_gate = (
        candidate_context == "edge_gate"
        and not state_triplet
        and not inverse_sbox_triplet
        and not dual_view_triplet
    )
    if not protocol_passed:
        status = "fail"
        decision = (
            "innovation1_uknit_dual_view_triplet_protocol_invalid"
            if dual_view_triplet
            else (
                "innovation1_uknit_inverse_sbox_triplet_protocol_invalid"
                if inverse_sbox_triplet
                else (
                    "innovation1_uknit_state_triplet_protocol_invalid"
                    if state_triplet
                    else (
                        "innovation1_uknit_sbox_edge_gate_protocol_invalid"
                        if edge_gate
                        else "innovation1_uknit_sbox_assignment_protocol_invalid"
                    )
                )
            )
        )
        next_action = (
            "repair the protocol mismatch and rerun the unchanged local U1 matrix; "
            "do not interpret the AUCs"
        )
    elif research_passed:
        status = "pass"
        if dual_view_triplet:
            decision = "innovation1_uknit_dual_view_triplet_two_seed_supported"
            next_action = (
                "audit both best dual-view checkpoints with a same-checkpoint "
                "correct-versus-shuffled view swap before any scale increase"
            )
        elif inverse_sbox_triplet:
            decision = "innovation1_uknit_inverse_sbox_triplet_two_seed_supported"
            next_action = (
                "audit both best inverse-S-box-triplet checkpoints with a "
                "same-checkpoint correct-versus-shuffled operator swap before any "
                "scale increase"
            )
        elif state_triplet:
            decision = "innovation1_uknit_state_triplet_two_seed_supported"
            next_action = (
                "audit both best state-triplet checkpoints with a same-checkpoint "
                "correct-versus-shuffled ownership swap before any scale increase"
            )
        elif edge_gate:
            decision = "innovation1_uknit_sbox_edge_gate_two_seed_supported"
            next_action = (
                "audit both best candidate checkpoints with a same-checkpoint "
                "correct-versus-shuffled ownership swap before any scale increase"
            )
        else:
            decision = "innovation1_uknit_sbox_assignment_two_seed_supported"
            next_action = (
                "preregister one unchanged-budget replication on a different valid "
                "uKNIT transition window; do not increase scale yet"
            )
    else:
        status = "hold"
        decision = (
            "innovation1_uknit_dual_view_triplet_hold"
            if dual_view_triplet
            else (
                "innovation1_uknit_inverse_sbox_triplet_hold"
                if inverse_sbox_triplet
                else (
                    "innovation1_uknit_state_triplet_hold"
                    if state_triplet
                    else (
                        "innovation1_uknit_sbox_edge_gate_hold"
                        if edge_gate
                        else "innovation1_uknit_sbox_assignment_hold_redesign_local"
                    )
                )
            )
        )
        passing_seeds = sum(
            all(
                research_checks[key]
                for key in (
                    f"seed{seed}_candidate_auc_at_least_0p520",
                    f"seed{seed}_candidate_beats_anchor_by_0p005",
                    f"seed{seed}_candidate_beats_shuffle_by_0p005",
                )
            )
            for seed in EXPECTED_SEEDS
        )
        if dual_view_triplet:
            next_action = (
                "close this parameter-free dual-view fusion and review a distinct "
                "runtime representation hypothesis without scale-up"
            )
        elif inverse_sbox_triplet:
            next_action = (
                "close this inverse-S-box-triplet operator design and review a "
                "distinct runtime representation hypothesis without scale-up"
            )
        elif state_triplet:
            next_action = (
                "close this state-triplet-plus-edge-gate design and review a "
                "distinct runtime representation hypothesis without scale-up"
            )
        elif edge_gate:
            next_action = (
                "close this parameter-free edge-gate design and review a distinct "
                "runtime representation hypothesis without scale-up"
            )
        else:
            next_action = (
                "run a deterministic activation/gradient attribution audit without "
                "more training data"
                if passing_seeds == 1
                else "redesign the local S-box/topology interaction without scale-up"
            )

    return {
        "run_id": run_id,
        "task": (
            "innovation1_uknit_runtime_e4_dual_view_triplet_u2e"
            if dual_view_triplet
            else (
                "innovation1_uknit_runtime_e4_inverse_sbox_triplet_u2d"
                if inverse_sbox_triplet
                else (
                    "innovation1_uknit_runtime_e4_state_triplet_u2c"
                    if state_triplet
                    else (
                        "innovation1_uknit_runtime_e4_sbox_edge_gate_u2b"
                        if edge_gate
                        else "innovation1_uknit_runtime_e4_sbox_assignment_u1"
                    )
                )
            )
        ),
        "cipher": "uKNIT-BC",
        "status": status,
        "decision": decision,
        "thresholds": {
            "candidate_auc": AUC_FLOOR,
            "candidate_minus_anchor": MARGIN_FLOOR,
            "candidate_minus_shuffled": MARGIN_FLOOR,
        },
        "seed_results": seed_results,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "claim_scope": (
            f"uKNIT-BC prefix-r4 two-seed 2048/class local "
            f"{candidate_cell_input_mode or candidate_context} "
            "S-box-assignment diagnostic only; not formal, paper-scale, attack, "
            "cross-cipher, or breakthrough evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "increase samples or epochs before the prescribed next action",
            "launch remote scale",
            "add DDT, trail, or partial-decryption features to U1",
            "claim a uKNIT attack or stable cross-cipher superiority",
        ],
    }


def _role(
    row: dict[str, Any],
    *,
    candidate_context: str,
    candidate_cell_input_mode: str | None,
    anchor_context: str,
    anchor_cell_input_mode: str | None,
) -> str:
    mode = row.get("runtime_structure_mode")
    options = row.get("training", {}).get("model_options", {})
    context = options.get("sbox_context_mode")
    cell_input = options.get("cell_input_mode", "difference_only")
    candidate_input_matches = (
        candidate_cell_input_mode is None or cell_input == candidate_cell_input_mode
    )
    anchor_input_matches = (
        anchor_cell_input_mode is None or cell_input == anchor_cell_input_mode
    )
    if mode == "true" and context == candidate_context and candidate_input_matches:
        return "candidate"
    if mode == "true" and context == anchor_context and anchor_input_matches:
        return "anchor"
    if (
        mode == "sbox_shuffled"
        and context == candidate_context
        and candidate_input_matches
    ):
        return "shuffled"
    return "unknown"


def _seed_result(group: dict[str, list[dict[str, Any]]]) -> dict[str, float | None]:
    aucs: dict[str, float | None] = {}
    for role in EXPECTED_MODELS:
        role_rows = group.get(role, ())
        value = (
            role_rows[0].get("metrics", {}).get("auc") if len(role_rows) == 1 else None
        )
        aucs[role] = float(value) if _finite(value) else None
    candidate = aucs["candidate"]
    anchor = aucs["anchor"]
    shuffled = aucs["shuffled"]
    return {
        "candidate_auc": candidate,
        "anchor_auc": anchor,
        "shuffled_auc": shuffled,
        "candidate_minus_anchor": (
            candidate - anchor if candidate is not None and anchor is not None else None
        ),
        "candidate_minus_shuffled": (
            candidate - shuffled
            if candidate is not None and shuffled is not None
            else None
        ),
    }


def _all_rows_match_uknit_protocol(rows: list[dict[str, Any]]) -> bool:
    return len(rows) == 6 and all(
        row.get("cipher") == "uKNIT-BC"
        and row.get("cipher_key") == "uknit64"
        and row.get("structure") == "SPN"
        and row.get("rounds") == 4
        and row.get("samples_per_class") == 2048
        and row.get("validation", {}).get("samples_per_class") == 1024
        and row.get("pairs_per_sample") == 4
        and row.get("input_difference") == 0x40
        and row.get("target_epochs") == 10
        for row in rows
    )


def _same_within_seed(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
    projector,
) -> bool:
    for seed in EXPECTED_SEEDS:
        rows = [row for role_rows in grouped[seed].values() for row in role_rows]
        if len(rows) != 3 or len({projector(row) for row in rows}) != 1:
            return False
    return True


def _data_protocol(row: dict[str, Any]) -> tuple[Any, ...]:
    validation = row.get("validation", {})
    training = row.get("training", {})
    return (
        row.get("cipher_key"),
        row.get("rounds"),
        row.get("seed"),
        row.get("train_key"),
        row.get("validation_key"),
        row.get("input_difference"),
        row.get("samples_per_class"),
        validation.get("samples_per_class"),
        row.get("pairs_per_sample"),
        row.get("feature_encoding"),
        row.get("negative_mode"),
        row.get("sample_structure"),
        training.get("dataset_cache_root"),
    )


def _training_protocol(row: dict[str, Any]) -> tuple[Any, ...]:
    training = row.get("training", {})
    return (
        row.get("seed"),
        row.get("target_epochs"),
        training.get("batch_size"),
        training.get("learning_rate"),
        training.get("optimizer"),
        training.get("optimizer_state_transition"),
        training.get("weight_decay"),
        training.get("loss"),
        training.get("lr_scheduler"),
        training.get("checkpoint_metric"),
        training.get("selected_checkpoint"),
    )


def _equal_parameter_geometry(rows: list[dict[str, Any]]) -> bool:
    counts = {
        (row.get("parameter_count"), row.get("trainable_parameter_count"))
        for row in rows
    }
    return len(rows) == 6 and len(counts) == 1 and None not in next(iter(counts), ())


def _exact_descriptor_window(rows: list[dict[str, Any]]) -> bool:
    hashes = {row.get("runtime_structure_descriptor_sha256") for row in rows}
    return (
        len(rows) == 6
        and len(hashes) == 1
        and None not in hashes
        and all(
            row.get("runtime_structure_round_start") == 2
            and row.get("runtime_structure_available_rounds") == 11
            and row.get("runtime_structure_loaded_rounds") == 2
            and row.get("training", {})
            .get("model_options", {})
            .get("runtime_structure_path")
            == "configs/runtime/spn/uknit64.json"
            for row in rows
        )
    )


def _role_contract(
    row: dict[str, Any],
    role: str,
    *,
    candidate_context: str,
    candidate_cell_input_mode: str | None,
    anchor_context: str,
    anchor_cell_input_mode: str | None,
) -> bool:
    options = row.get("training", {}).get("model_options", {})
    cell_input = options.get("cell_input_mode", "difference_only")
    if row.get("model") != EXPECTED_MODELS[role]:
        return False
    if role == "candidate":
        return (
            row.get("runtime_structure_mode") == "true"
            and options.get("sbox_context_mode") == candidate_context
            and (
                candidate_cell_input_mode is None
                or cell_input == candidate_cell_input_mode
            )
        )
    if role == "anchor":
        return (
            row.get("runtime_structure_mode") == "true"
            and options.get("sbox_context_mode") == anchor_context
            and (
                anchor_cell_input_mode is None or cell_input == anchor_cell_input_mode
            )
        )
    return (
        row.get("runtime_structure_mode") == "sbox_shuffled"
        and options.get("sbox_context_mode") == candidate_context
        and (
            candidate_cell_input_mode is None
            or cell_input == candidate_cell_input_mode
        )
        and options.get("sbox_assignment_shuffle_seed") == 20260724
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


__all__ = ["adjudicate_uknit_sbox_assignment"]
