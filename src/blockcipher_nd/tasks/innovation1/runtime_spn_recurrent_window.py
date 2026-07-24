from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any


EXPECTED_SEEDS = (0, 1)
EXPECTED_ROLES = (
    "anchor",
    "candidate",
    "repeat_last",
    "corrupted",
    "no_topology",
)
AUC_FLOOR = 0.520
MARGIN_FLOOR = 0.005
VALIDATION_KEY = int("11" * 16, 16)


def adjudicate_runtime_spn_recurrent_window(
    *,
    run_id: str,
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row.get("seed", -1))][_role(row)].append(row)

    complete = all(
        len(grouped[seed].get(role, ())) == 1
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    )
    seed_results = {str(seed): _seed_result(grouped[seed]) for seed in EXPECTED_SEEDS}
    protocol_checks = {
        "ten_rows_complete": len(rows) == 10,
        "two_expected_seeds": set(grouped) == set(EXPECTED_SEEDS),
        "five_unique_roles_per_seed": complete,
        "frozen_uknit_r5_protocol": bool(rows)
        and all(_frozen_protocol(row) for row in rows),
        "strict_encrypted_random_plaintext_negatives": bool(rows)
        and all(
            row.get("negative_mode") == "encrypted_random_plaintexts" for row in rows
        ),
        "same_data_protocol_within_seed": complete
        and _same_within_seed(grouped, _data_protocol),
        "same_training_protocol_within_seed": complete
        and _same_within_seed(grouped, _training_protocol),
        "disk_backed_datasets": bool(rows)
        and all(
            row.get("training", {}).get("train_dataset_storage") == "disk"
            and row.get("training", {}).get("validation_dataset_storage") == "disk"
            and bool(row.get("training", {}).get("dataset_cache_root"))
            for row in rows
        ),
        "equal_parameter_geometry": bool(rows)
        and len(
            {
                (
                    row.get("parameter_count"),
                    row.get("trainable_parameter_count"),
                )
                for row in rows
            }
        )
        == 1,
        "runtime_bit_order_adapter_recorded": bool(rows)
        and all(
            row.get("input_bit_order") == "project_msb_to_runtime_lsb" for row in rows
        ),
        "exact_two_transition_runtime_window": bool(rows)
        and all(
            row.get("runtime_structure_round_start") == 3
            and row.get("runtime_structure_loaded_rounds") == 2
            and _integer_at_least(row.get("runtime_structure_available_rounds"), 5)
            for row in rows
        ),
        "source_descriptor_identity_preserved": bool(rows)
        and len(
            {
                (
                    row.get("runtime_structure_descriptor_sha256"),
                    row.get("runtime_structure_descriptor_name"),
                )
                for row in rows
            }
        )
        == 1
        and all(
            _is_sha256(row.get("runtime_structure_descriptor_sha256")) for row in rows
        ),
        "structure_fingerprints_well_formed": bool(rows)
        and all(_well_formed_structure_fingerprints(row) for row in rows),
        "role_contracts_match_frozen_plan": complete
        and all(
            _role_contract(grouped[seed][role][0], role)
            for seed in EXPECTED_SEEDS
            for role in EXPECTED_ROLES
        ),
        "candidate_window_is_heterogeneous": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: (
                _integer_at_least(
                    seed_rows["candidate"].get(
                        "runtime_structure_unique_transition_count"
                    ),
                    2,
                )
                and seed_rows["candidate"].get("runtime_structure_homogeneous") is False
            ),
        ),
        "repeat_last_window_is_homogeneous": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: (
                seed_rows["repeat_last"].get(
                    "runtime_structure_unique_transition_count"
                )
                == 1
                and seed_rows["repeat_last"].get("runtime_structure_homogeneous")
                is True
            ),
        ),
        "candidate_repeat_last_final_transition_equal": complete
        and _all_seed_rows(grouped, _candidate_repeat_last_final_equal),
        "candidate_repeat_last_window_distinct": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: (
                _window_hash(seed_rows["candidate"])
                != _window_hash(seed_rows["repeat_last"])
            ),
        ),
        "anchor_candidate_structure_equal": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: (
                _structure_identity(seed_rows["anchor"])
                == _structure_identity(seed_rows["candidate"])
            ),
        ),
        "candidate_no_topology_structure_equal": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: (
                _structure_identity(seed_rows["candidate"])
                == _structure_identity(seed_rows["no_topology"])
            ),
        ),
        "candidate_corrupted_structure_distinct": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: (
                _window_hash(seed_rows["candidate"])
                != _window_hash(seed_rows["corrupted"])
            ),
        ),
        "structure_evidence_seed_invariant": complete
        and all(
            _structure_identity(grouped[0][role][0])
            == _structure_identity(grouped[1][role][0])
            for role in EXPECTED_ROLES
        ),
        "ten_epoch_best_checkpoint_integrity": bool(rows)
        and all(_checkpoint_integrity(row) for row in rows),
        "finite_auc_metrics": bool(rows)
        and all(_finite(row.get("metrics", {}).get("auc")) for row in rows),
    }

    research_checks: dict[str, bool] = {}
    for seed in EXPECTED_SEEDS:
        result = seed_results[str(seed)]
        research_checks[f"seed{seed}_candidate_auc_at_least_0p520"] = _at_least(
            result["candidate_auc"], AUC_FLOOR
        )
        for control in ("anchor", "repeat_last", "corrupted", "no_topology"):
            research_checks[f"seed{seed}_candidate_beats_{control}_by_0p005"] = (
                _at_least(result[f"candidate_minus_{control}"], MARGIN_FLOOR)
            )

    protocol_passed = all(protocol_checks.values())
    research_passed = all(research_checks.values())
    if not protocol_passed:
        status = "fail"
        decision = "innovation1_runtime_spn_recurrent_window_protocol_invalid"
        next_action = (
            "repair only the failed frozen-protocol or checkpoint-evidence check; "
            "do not interpret the AUCs or rerun with changed thresholds"
        )
    elif research_passed:
        status = "pass"
        decision = "innovation1_runtime_spn_recurrent_window_two_seed_supported"
        next_action = (
            "freeze both candidate checkpoints and run one same-checkpoint full-window "
            "versus repeat-last/corrupted window-swap attribution audit before any "
            "scale or cross-cipher transfer"
        )
    else:
        status = "hold"
        decision = "innovation1_runtime_spn_recurrent_window_not_supported"
        next_action = (
            "stop recurrent-window scale-up and inspect the frozen histories once; "
            "redesign locally if either seed lacks signal or any required control "
            "matches the candidate"
        )

    return {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_recurrent_window_u3",
        "cipher": "uKNIT-BC",
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "seed_results": seed_results,
        "thresholds": {
            "candidate_auc": AUC_FLOOR,
            "candidate_minus_each_control": MARGIN_FLOOR,
        },
        "threshold_provenance": (
            "frozen before U3 training from the completed same-budget uKNIT U2-C "
            "state-triplet signal floor and the existing project topology-attribution "
            "margin; RTG3-A authorizes the route but does not tune uKNIT thresholds"
        ),
        "claim_scope": (
            "uKNIT-BC prefix-r5 two-seed 2048/class local recurrent heterogeneous-"
            "window attribution diagnostic only; not formal scale, attack, SOTA, "
            "universal-SPN, or cross-cipher transfer evidence"
        ),
        "next_action": next_action,
        "blocked_actions": [
            "change thresholds after reading U3 AUCs",
            "launch remote scale directly from this local diagnostic",
            "drop the repeat-last equal-depth control",
            "claim earlier-round topology use from homogeneous SPN windows",
            "add DDT, trail, related-key, or partial-decryption features",
        ],
    }


def _role(row: dict[str, Any]) -> str:
    model = str(row.get("model", ""))
    options = row.get("training", {}).get("model_options", {})
    if not isinstance(options, dict):
        return "invalid"
    mode = options.get("round_window_mode")
    control = options.get("runtime_structure_window_control")
    if model == "runtime_spn_e4_equivariant_true":
        if mode == "last_transition" and control == "full":
            return "anchor"
        if mode == "recurrent_window" and control == "full":
            return "candidate"
        if mode == "recurrent_window" and control == "repeat_last":
            return "repeat_last"
    if (
        model == "runtime_spn_e4_equivariant_corrupted"
        and mode == "recurrent_window"
        and control == "full"
    ):
        return "corrupted"
    if (
        model == "runtime_spn_e4_equivariant_independent"
        and mode == "recurrent_window"
        and control == "full"
    ):
        return "no_topology"
    return "invalid"


def _role_contract(row: dict[str, Any], role: str) -> bool:
    options = row.get("training", {}).get("model_options", {})
    if not isinstance(options, dict):
        return False
    common = {
        "runtime_structure_path": "configs/runtime/spn/uknit64.json",
        "runtime_round_start": 3,
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "dropout": 0.0,
        "sbox_context_mode": "edge_gate",
        "cell_input_mode": "state_triplet",
    }
    if any(options.get(field) != value for field, value in common.items()):
        return False
    expected = {
        "anchor": (
            "runtime_spn_e4_equivariant_true",
            "last_transition",
            "full",
        ),
        "candidate": (
            "runtime_spn_e4_equivariant_true",
            "recurrent_window",
            "full",
        ),
        "repeat_last": (
            "runtime_spn_e4_equivariant_true",
            "recurrent_window",
            "repeat_last",
        ),
        "corrupted": (
            "runtime_spn_e4_equivariant_corrupted",
            "recurrent_window",
            "full",
        ),
        "no_topology": (
            "runtime_spn_e4_equivariant_independent",
            "recurrent_window",
            "full",
        ),
    }
    model, mode, control = expected[role]
    if (
        row.get("model") != model
        or options.get("round_window_mode") != mode
        or options.get("runtime_structure_window_control") != control
    ):
        return False
    if role == "corrupted":
        return options.get("topology_corruption_seed") == 20260724
    return "topology_corruption_seed" not in options


def _frozen_protocol(row: dict[str, Any]) -> bool:
    training = row.get("training", {})
    validation = row.get("validation", {})
    return bool(
        row.get("cipher") == "uKNIT-BC"
        and row.get("cipher_key") == "uknit64"
        and row.get("structure") == "SPN"
        and row.get("rounds") == 5
        and row.get("seed") in EXPECTED_SEEDS
        and row.get("samples_per_class") == 2048
        and row.get("pairs_per_sample") == 4
        and row.get("input_difference") == 0x40
        and row.get("feature_encoding") == "ciphertext_pair_bits"
        and row.get("sample_structure") == "independent_pairs"
        and row.get("train_key") == 0
        and row.get("validation_key") == VALIDATION_KEY
        and row.get("target_epochs") == 10
        and validation.get("samples_per_class") == 1024
        and training.get("epochs") == 10
        and training.get("batch_size") == 256
        and training.get("optimizer") == "adam"
        and training.get("optimizer_state_transition") == "reset_each_stage"
        and training.get("learning_rate") == 0.0001
        and training.get("weight_decay") == 0.00001
        and training.get("loss") == "mse"
        and training.get("lr_scheduler") == "none"
        and training.get("checkpoint_metric") == "val_auc"
        and training.get("restore_best_checkpoint") is True
        and training.get("selected_checkpoint") == "best"
    )


def _data_protocol(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row.get(field)
        for field in (
            "cipher_key",
            "rounds",
            "seed",
            "samples_per_class",
            "dataset_label_mode",
            "pairs_per_sample",
            "feature_encoding",
            "negative_mode",
            "key_rotation_interval",
            "sample_structure",
            "input_difference",
            "train_key",
            "validation_key",
        )
    )


def _training_protocol(row: dict[str, Any]) -> tuple[Any, ...]:
    training = row.get("training", {})
    return tuple(
        training.get(field)
        for field in (
            "batch_size",
            "learning_rate",
            "optimizer",
            "optimizer_state_transition",
            "weight_decay",
            "loss",
            "lr_scheduler",
            "checkpoint_metric",
            "restore_best_checkpoint",
            "selected_checkpoint",
            "epochs",
            "train_rows",
            "validation_rows",
            "dataset_cache_root",
            "dataset_cache_chunk_size",
            "dataset_cache_workers",
        )
    )


def _same_within_seed(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
    projection: Any,
) -> bool:
    for seed in EXPECTED_SEEDS:
        values = {projection(grouped[seed][role][0]) for role in EXPECTED_ROLES}
        if len(values) != 1:
            return False
    return True


def _all_seed_rows(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
    predicate: Callable[[dict[str, dict[str, Any]]], bool],
) -> bool:
    return all(
        predicate({role: grouped[seed][role][0] for role in EXPECTED_ROLES})
        for seed in EXPECTED_SEEDS
    )


def _candidate_repeat_last_final_equal(
    seed_rows: dict[str, dict[str, Any]],
) -> bool:
    candidate = seed_rows["candidate"].get("runtime_structure_transition_sha256s")
    repeated = seed_rows["repeat_last"].get("runtime_structure_transition_sha256s")
    return bool(
        isinstance(candidate, (list, tuple))
        and isinstance(repeated, (list, tuple))
        and candidate
        and repeated
        and candidate[-1] == repeated[-1]
    )


def _structure_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    transitions = row.get("runtime_structure_transition_sha256s")
    return (
        tuple(transitions) if isinstance(transitions, (list, tuple)) else (),
        _window_hash(row),
        row.get("runtime_structure_unique_transition_count"),
        row.get("runtime_structure_homogeneous"),
    )


def _window_hash(row: dict[str, Any]) -> object:
    return row.get("runtime_structure_window_sha256")


def _well_formed_structure_fingerprints(row: dict[str, Any]) -> bool:
    transitions = row.get("runtime_structure_transition_sha256s")
    return bool(
        isinstance(transitions, (list, tuple))
        and len(transitions) == 2
        and all(_is_sha256(value) for value in transitions)
        and _is_sha256(row.get("runtime_structure_window_sha256"))
        and row.get("runtime_structure_unique_transition_count")
        == len(set(transitions))
        and row.get("runtime_structure_homogeneous") is (len(set(transitions)) == 1)
    )


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _checkpoint_integrity(row: dict[str, Any]) -> bool:
    history = row.get("history")
    training = row.get("training", {})
    metrics = row.get("metrics", {})
    if not isinstance(history, list) or len(history) != 10:
        return False
    if any(
        not isinstance(item, dict)
        or item.get("epoch") != float(index)
        or not _finite(item.get("val_auc"))
        for index, item in enumerate(history, start=1)
    ):
        return False
    best_index = max(range(10), key=lambda index: float(history[index]["val_auc"]))
    best_epoch = best_index + 1
    best_auc = float(history[best_index]["val_auc"])
    return bool(
        training.get("epochs_ran") == 10
        and training.get("best_epoch") == best_epoch
        and training.get("stopped_epoch") == 0
        and _close(training.get("best_checkpoint_metric"), best_auc)
        and _close(metrics.get("auc"), best_auc)
    )


def _seed_result(
    seed_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, float | None]:
    aucs = {role: _auc(seed_rows.get(role, ())) for role in EXPECTED_ROLES}
    candidate = aucs["candidate"]
    result: dict[str, float | None] = {
        f"{role}_auc": value for role, value in aucs.items()
    }
    for control in ("anchor", "repeat_last", "corrupted", "no_topology"):
        value = aucs[control]
        result[f"candidate_minus_{control}"] = (
            candidate - value if candidate is not None and value is not None else None
        )
    return result


def _auc(rows: Iterable[dict[str, Any]]) -> float | None:
    rows = list(rows)
    if len(rows) != 1:
        return None
    value = rows[0].get("metrics", {}).get("auc")
    return float(value) if _finite(value) else None


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _close(left: object, right: float, tolerance: float = 1e-7) -> bool:
    return _finite(left) and abs(float(left) - right) <= tolerance


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _integer_at_least(value: object, threshold: int) -> bool:
    return type(value) is int and value >= threshold


__all__ = [
    "AUC_FLOOR",
    "EXPECTED_ROLES",
    "EXPECTED_SEEDS",
    "MARGIN_FLOOR",
    "adjudicate_runtime_spn_recurrent_window",
]
