from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from torch import nn

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.registry.model_factory import build_model


EXPECTED_SEEDS = (0, 1)
EXPECTED_ROLES = (
    "anchor",
    "candidate",
    "repeat_last",
    "corrupted",
    "no_topology",
)
BASE_MODEL_OPTIONS = {
    "runtime_structure_path": "configs/runtime/spn/uknit64.json",
    "runtime_round_start": 3,
    "runtime_rounds": 2,
    "processor_steps": 2,
    "pair_embedding_dim": 128,
    "dropout": 0.0,
    "sbox_context_mode": "edge_gate",
    "cell_input_mode": "state_triplet",
}
ROLE_MODEL_OPTIONS = {
    "anchor": {
        **BASE_MODEL_OPTIONS,
        "round_window_mode": "last_transition",
        "runtime_structure_window_control": "full",
    },
    "candidate": {
        **BASE_MODEL_OPTIONS,
        "round_window_mode": "recurrent_window",
        "runtime_structure_window_control": "full",
    },
    "repeat_last": {
        **BASE_MODEL_OPTIONS,
        "round_window_mode": "recurrent_window",
        "runtime_structure_window_control": "repeat_last",
    },
    "corrupted": {
        **BASE_MODEL_OPTIONS,
        "round_window_mode": "recurrent_window",
        "runtime_structure_window_control": "full",
        "topology_corruption_seed": 20260724,
    },
    "no_topology": {
        **BASE_MODEL_OPTIONS,
        "round_window_mode": "recurrent_window",
        "runtime_structure_window_control": "full",
    },
}
ROLE_MODELS = {
    "anchor": "runtime_spn_e4_equivariant_true",
    "candidate": "runtime_spn_e4_equivariant_true",
    "repeat_last": "runtime_spn_e4_equivariant_true",
    "corrupted": "runtime_spn_e4_equivariant_corrupted",
    "no_topology": "runtime_spn_e4_equivariant_independent",
}


def build_recurrent_window_readiness(
    *,
    run_id: str,
    tasks: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    build_errors: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        try:
            manifests.append(_build_manifest(task))
        except Exception as exc:
            build_errors.append(
                {
                    "index": index,
                    "architecture": str(task.get("architecture", "")),
                    "model": str(task.get("model_key", "")),
                    "seed": task.get("seed"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    gate = adjudicate_recurrent_window_readiness(
        run_id=run_id,
        manifests=manifests,
        build_errors=build_errors,
    )
    return manifests, gate


def adjudicate_recurrent_window_readiness(
    *,
    run_id: str,
    manifests: Iterable[dict[str, Any]],
    build_errors: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    rows = list(manifests)
    errors = list(build_errors)
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[int(row.get("seed", -1))][str(row.get("role", "invalid"))].append(
            row
        )

    complete = _complete_panel(grouped)
    protocol_checks = {
        "all_models_constructed": not errors,
        "ten_manifest_rows": len(rows) == len(EXPECTED_SEEDS) * len(EXPECTED_ROLES),
        "two_seed_five_role_panel": complete
        and set(grouped) == set(EXPECTED_SEEDS),
        "role_model_and_option_contracts": complete
        and _role_contracts_match(grouped),
        "frozen_uknit_r5_local_protocol": bool(rows)
        and all(_frozen_protocol(row) for row in rows),
        "same_data_protocol": bool(rows)
        and len({_canonical(row.get("data_protocol")) for row in rows}) == 1,
        "same_training_protocol": bool(rows)
        and len({_canonical(row.get("training_protocol")) for row in rows}) == 1,
        "strict_encrypted_random_plaintext_negatives": bool(rows)
        and all(
            row.get("data_protocol", {}).get("negative_mode")
            == "encrypted_random_plaintexts"
            for row in rows
        ),
        "runtime_rounds_equal_two": bool(rows)
        and all(row.get("runtime_structure_loaded_rounds") == 2 for row in rows),
        "equal_parameter_count_and_shapes": bool(rows)
        and len({row.get("parameter_count") for row in rows}) == 1
        and len({row.get("parameter_shape_sha256") for row in rows}) == 1,
        "candidate_window_is_heterogeneous": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: seed_rows["candidate"].get(
                "runtime_structure_unique_transition_count"
            )
            > 1
            and seed_rows["candidate"].get("runtime_structure_homogeneous") is False,
        ),
        "repeat_last_window_is_homogeneous": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: seed_rows["repeat_last"].get(
                "runtime_structure_unique_transition_count"
            )
            == 1
            and seed_rows["repeat_last"].get("runtime_structure_homogeneous") is True,
        ),
        "candidate_repeat_last_final_transition_equal": complete
        and _all_seed_rows(grouped, _same_final_transition),
        "candidate_repeat_last_window_distinct": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: seed_rows["candidate"].get(
                "runtime_structure_window_sha256"
            )
            != seed_rows["repeat_last"].get("runtime_structure_window_sha256"),
        ),
        "anchor_and_candidate_structure_equal": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: _structure_identity(seed_rows["anchor"])
            == _structure_identity(seed_rows["candidate"]),
        ),
        "corrupted_structure_is_distinct": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: seed_rows["candidate"].get(
                "runtime_structure_window_sha256"
            )
            != seed_rows["corrupted"].get("runtime_structure_window_sha256"),
        ),
        "no_topology_preserves_candidate_structure": complete
        and _all_seed_rows(
            grouped,
            lambda seed_rows: _structure_identity(seed_rows["candidate"])
            == _structure_identity(seed_rows["no_topology"]),
        ),
        "structure_evidence_seed_invariant": complete
        and _structure_evidence_seed_invariant(grouped),
        "structure_hashes_well_formed": bool(rows)
        and all(_well_formed_structure_hashes(row) for row in rows),
    }
    passed = all(protocol_checks.values())
    return {
        "run_id": run_id,
        "task": "innovation1_runtime_spn_recurrent_window_readiness",
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_runtime_spn_recurrent_window_readiness_passed"
            if passed
            else "innovation1_runtime_spn_recurrent_window_protocol_invalid"
        ),
        "protocol_checks": protocol_checks,
        "build_errors": errors,
        "manifest_rows": len(rows),
        "expected_rows": len(EXPECTED_SEEDS) * len(EXPECTED_ROLES),
        "claim_scope": (
            "deterministic model-construction and protocol readiness only; no data "
            "generation, training, AUC, transfer, attack, scale, or breakthrough claim"
        ),
        "next_action": (
            "keep this frozen uKNIT r5 panel ready, but wait for the RTG3-A joint "
            "decision before running the local recurrent-window diagnostic"
            if passed
            else "repair the failed structure or protocol checks before generating data"
        ),
        "blocked_actions": [
            "set an AUC threshold before the empirical phase is preregistered",
            "train a homogeneous PRESENT, GIFT, or SKINNY window as evidence of distinct earlier topology use",
            "launch remote scale before the local two-seed uKNIT gate",
            "reuse the stopped final-transition delta-U query in recurrent-window mode",
        ],
    }


def _build_manifest(task: dict[str, Any]) -> dict[str, Any]:
    options = task.get("model_options")
    if not isinstance(options, dict):
        raise ValueError("model_options must be a dictionary")
    role = _task_role(task)
    pairs_per_sample = int(task.get("pairs_per_sample", 0))
    model = build_model(
        str(task.get("model_key", "")),
        input_bits=pairs_per_sample * 128,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    )
    return {
        "architecture": str(task.get("architecture", "")),
        "model": str(task.get("model_key", "")),
        "role": role,
        "seed": int(task.get("seed", -1)),
        "model_options": dict(options),
        "data_protocol": _data_protocol(task),
        "training_protocol": _training_protocol(task),
        "parameter_shape_sha256": _parameter_shape_sha256(model),
        **model_metadata(model),
    }


def _task_role(task: dict[str, Any]) -> str:
    model = str(task.get("model_key", ""))
    options = task.get("model_options")
    if not isinstance(options, dict):
        return "invalid"
    mode = str(options.get("round_window_mode", "last_transition"))
    control = str(options.get("runtime_structure_window_control", "full"))
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


def _data_protocol(task: dict[str, Any]) -> dict[str, Any]:
    return {
        field: task.get(field)
        for field in (
            "cipher_key",
            "rounds",
            "samples_per_class",
            "train_samples_total",
            "validation_samples_total",
            "dataset_label_mode",
            "pairs_per_sample",
            "feature_encoding",
            "negative_mode",
            "train_key",
            "validation_key",
            "key_rotation_interval",
            "sample_structure",
            "input_difference",
            "difference_profile",
            "difference_member",
        )
    }


def _training_protocol(task: dict[str, Any]) -> dict[str, Any]:
    return {
        field: task.get(field)
        for field in (
            "loss",
            "learning_rate",
            "optimizer",
            "optimizer_state_transition",
            "weight_decay",
            "lr_scheduler",
            "max_learning_rate",
            "checkpoint_metric",
            "restore_best_checkpoint",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "target_epochs",
            "pretrain_rounds",
            "pretrain_round_sequence",
            "pretrain_epochs",
        )
    }


def _frozen_protocol(row: dict[str, Any]) -> bool:
    data = row.get("data_protocol", {})
    training = row.get("training_protocol", {})
    return bool(
        data.get("cipher_key") == "uknit64"
        and data.get("rounds") == 5
        and data.get("samples_per_class") == 2048
        and data.get("train_samples_total") is None
        and data.get("validation_samples_total") is None
        and data.get("dataset_label_mode") == "balanced_per_class"
        and data.get("pairs_per_sample") == 4
        and data.get("feature_encoding") == "ciphertext_pair_bits"
        and data.get("negative_mode") == "encrypted_random_plaintexts"
        and data.get("train_key") == 0
        and data.get("validation_key") == 0x1111111111111111
        and data.get("key_rotation_interval") == 0
        and data.get("sample_structure") == "independent_pairs"
        and data.get("input_difference") == 0x40
        and not data.get("difference_profile")
        and data.get("difference_member") in {"", 0}
        and training.get("loss") == "mse"
        and training.get("learning_rate") == 0.0001
        and training.get("optimizer") == "adam"
        and training.get("optimizer_state_transition") == "reset_each_stage"
        and training.get("weight_decay") == 0.00001
        and training.get("lr_scheduler") == "none"
        and training.get("max_learning_rate") is None
        and training.get("checkpoint_metric") == "val_auc"
        and training.get("restore_best_checkpoint") is True
        and training.get("early_stopping_patience") == 0
        and training.get("early_stopping_min_delta") == 0.0
        and training.get("target_epochs") == 10
        and training.get("pretrain_rounds") is None
        and training.get("pretrain_round_sequence") in {None, ()}
        and training.get("pretrain_epochs") == 0
    )


def _complete_panel(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
) -> bool:
    return all(
        len(grouped[seed].get(role, ())) == 1
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    )


def _role_contracts_match(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
) -> bool:
    return all(
        grouped[seed][role][0].get("model") == ROLE_MODELS[role]
        and grouped[seed][role][0].get("model_options") == ROLE_MODEL_OPTIONS[role]
        and grouped[seed][role][0].get("runtime_structure_mode")
        == {
            "anchor": "true",
            "candidate": "true",
            "repeat_last": "true",
            "corrupted": "corrupted",
            "no_topology": "independent",
        }[role]
        for seed in EXPECTED_SEEDS
        for role in EXPECTED_ROLES
    )


def _all_seed_rows(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
    check: Callable[[dict[str, dict[str, Any]]], bool],
) -> bool:
    return all(
        check({role: grouped[seed][role][0] for role in EXPECTED_ROLES})
        for seed in EXPECTED_SEEDS
    )


def _same_final_transition(seed_rows: dict[str, dict[str, Any]]) -> bool:
    candidate = seed_rows["candidate"].get("runtime_structure_transition_sha256s")
    repeated = seed_rows["repeat_last"].get(
        "runtime_structure_transition_sha256s"
    )
    return bool(
        isinstance(candidate, list)
        and isinstance(repeated, list)
        and candidate
        and repeated
        and candidate[-1] == repeated[-1]
    )


def _structure_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    transitions = row.get("runtime_structure_transition_sha256s")
    return (
        tuple(transitions) if isinstance(transitions, list) else None,
        row.get("runtime_structure_window_sha256"),
        row.get("runtime_structure_unique_transition_count"),
        row.get("runtime_structure_homogeneous"),
    )


def _structure_evidence_seed_invariant(
    grouped: dict[int, dict[str, list[dict[str, Any]]]],
) -> bool:
    return all(
        _structure_identity(grouped[0][role][0])
        == _structure_identity(grouped[1][role][0])
        for role in EXPECTED_ROLES
    )


def _well_formed_structure_hashes(row: dict[str, Any]) -> bool:
    transitions = row.get("runtime_structure_transition_sha256s")
    window = row.get("runtime_structure_window_sha256")
    return bool(
        isinstance(transitions, list)
        and len(transitions) == 2
        and all(_is_sha256(value) for value in transitions)
        and _is_sha256(window)
    )


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parameter_shape_sha256(model: nn.Module) -> str:
    geometry = [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
    ]
    return hashlib.sha256(
        json.dumps(geometry, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


__all__ = [
    "BASE_MODEL_OPTIONS",
    "EXPECTED_ROLES",
    "EXPECTED_SEEDS",
    "ROLE_MODEL_OPTIONS",
    "ROLE_MODELS",
    "adjudicate_recurrent_window_readiness",
    "build_recurrent_window_readiness",
]
