from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from blockcipher_nd.planning.four_role_attribution_gate import (
    FourRoleGateSpec,
    FourRoleProtocolSpec,
    evaluate_four_role_attribution,
)
from blockcipher_nd.planning.present_case2_attribution_protocol import (
    PRESENT_CASE2_ATTRIBUTION_PROTOCOL,
)


PRESENT_CROSS_SPN_MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_cross_spn_typed_cell_true",
    "shuffled_p": "present_cross_spn_typed_cell_shuffled",
    "raw_delta": "present_cross_spn_typed_cell_raw",
}

GIFT_CROSS_SPN_MODEL_ROLES = {
    "anchor": "gift_cross_spn_aligned_token_mixer_raw_anchor",
    "candidate": "gift_cross_spn_typed_cell_true",
    "shuffled_p": "gift_cross_spn_typed_cell_shuffled",
    "raw_delta": "gift_cross_spn_typed_cell_raw",
}

_PRESENT_ANCHOR_OPTIONS = {
    "spn_mixer_depth": 2,
    "activation": "relu",
    "norm": "layernorm",
}
_TYPED_OPTIONS = {
    "mixer_depth": 2,
    "token_mlp_ratio": 2,
    "activation": "relu",
    "norm": "layernorm",
    "pooling": "attention_mean_max",
    "dropout": 0.0,
}
_GIFT_ANCHOR_OPTIONS = {
    "mixer_depth": 1,
    "token_mlp_ratio": 2,
    "activation": "relu",
    "norm": "layernorm",
    "pooling": "topk_logsumexp",
    "dropout": 0.0,
    "top_k": 2,
    "lse_temperature": 1.0,
}


def _constant_training_fields(
    base: dict[str, Any],
) -> dict[str, Any]:
    return {
        **base,
        "lr_scheduler": "none",
        "max_learning_rate": None,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
    }


_PRESENT_E4_PROTOCOL = replace(
    PRESENT_CASE2_ATTRIBUTION_PROTOCOL,
    claim_prefix="strict PRESENT r7 E4 source",
    readiness_identity_label="E4-R0",
    invalid_claim_scope="invalid E4 PRESENT source protocol",
    readiness_claim_scope="E4-R0 implementation readiness only; metrics not interpreted",
    research_claim_suffix="local diagnostic only; not formal, paper-scale, or breakthrough evidence",
    training_static_fields=_constant_training_fields(
        dict(PRESENT_CASE2_ATTRIBUTION_PROTOCOL.training_static_fields)
    ),
)

_GIFT_E4_PROTOCOL = FourRoleProtocolSpec(
    claim_prefix="strict GIFT-64 r6 E4 scratch",
    readiness_identity_label="E4-R0",
    invalid_claim_scope="invalid E4 GIFT scratch protocol",
    readiness_claim_scope="E4-R0 implementation readiness only; metrics not interpreted",
    research_claim_suffix="local diagnostic only; not formal, paper-scale, or breakthrough evidence",
    row_static_fields={
        "cipher": "GIFT-64",
        "cipher_key": "gift64",
        "structure": "SPN",
        "rounds": 6,
        "dataset_label_mode": "balanced_per_class",
        "input_difference": 0x40,
        "train_samples_total": None,
        "validation_samples_total": None,
        "final_test_samples_total": None,
        "final_test_repeats": 0,
        "pairs_per_sample": 4,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": 0,
        "validation_key": int("11" * 16, 16),
        "key_rotation_interval": 0,
        "sample_structure": "independent_pairs",
        "difference_profile": "gift64_shen2024_spn_screen",
        "difference_member": 0,
        "integral_active_nibble": 0,
        "integral_active_nibbles": [],
        "validation_integral_active_nibbles": [],
    },
    training_static_fields={
        "amsgrad": False,
        "dataset_label_mode": "balanced_per_class",
        "device": "cpu",
        "feature_encoding": "ciphertext_pair_bits",
        "key_schedule": "fixed",
        "input_bits": 512,
        "integral_active_nibble": 0,
        "integral_active_nibbles": [],
        "validation_integral_active_nibbles": [],
        "pair_bits": 128,
        "pairs_per_sample": 4,
        "key_rotation_interval": 0,
        "sample_structure": "independent_pairs",
        "selected_bit_indices": [],
        "train_dataset_storage": "disk",
        "validation_dataset_storage": "disk",
        "optimizer_state_transition": "reset_each_stage",
        "optimizer_state_reused": False,
        "optimizer_state_step_before": 0,
        "optimizer_session_call": 1,
        "train_eval_interval": 1,
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "lr_scheduler": "none",
        "max_learning_rate": None,
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "selected_checkpoint": "best",
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
    },
    validation_static_fields={
        "cipher": "GIFT-64",
        "structure": "SPN",
        "rounds": 6,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "pairs_per_sample": 4,
        "dataset_label_mode": "balanced_per_class",
        "key_rotation_interval": 0,
        "sample_structure": "independent_pairs",
        "key_schedule": "fixed",
        "integral_active_nibble": 0,
        "integral_active_nibbles": [],
    },
    cache_terminal_static_fields={
        "cipher_key": "gift64",
        "rounds": 6,
        "dataset_label_mode": "balanced_per_class",
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "pairs_per_sample": 4,
        "key_rotation_interval": 0,
        "sample_structure": "independent_pairs",
        "difference_profile": "gift64_shen2024_spn_screen",
        "difference_member": 0,
        "input_bits": 512,
        "optimizer_state_transition": "reset_each_stage",
        "loss": "mse",
    },
    readiness_training_fields={
        "batch_size": 32,
        "dataset_cache_chunk_size": 64,
        "dataset_cache_workers": 1,
    },
    standard_training_fields={
        "batch_size": 256,
        "dataset_cache_chunk_size": 512,
        "dataset_cache_workers": 4,
    },
    readiness_seed_layout=(0,),
    readiness_samples_per_class=64,
    readiness_epochs=1,
    class_count=2,
)


def _decide_present_source(
    seed_reports: dict[str, dict[str, Any]],
    **_: Any,
) -> tuple[str, str]:
    report = seed_reports["0"]
    candidate_auc = report["aucs"]["candidate"]
    observed = {
        "absolute_auc": candidate_auc,
        "architecture_margin": report["architecture_margin"],
        "topology_margin": report["topology_margin"],
        "representation_margin": report["representation_margin"],
    }
    thresholds = {
        "absolute_auc": 0.65,
        "architecture_margin": -0.01,
        "topology_margin": 0.003,
        "representation_margin": 0.003,
    }
    failed = [
        name for name, threshold in thresholds.items() if observed[name] < threshold
    ]
    if not failed:
        return "promote_e4_r2", "freeze_and_implement_e4_r2_checkpoint_transfer"

    ordered_controls = (
        report["topology_margin"] > 0.0
        and report["representation_margin"] > 0.0
    )
    if (
        ordered_controls
        and len(failed) == 1
        and thresholds[failed[0]] - observed[failed[0]] <= 0.002
    ):
        return (
            "run_present_seed1_fragility",
            "run_only_same_budget_present_seed1_fragility_gate",
        )
    return (
        "reject_e4_shared_operator",
        "stop_e4_transfer_and_consolidate_invp_method_evidence",
    )


def _decide_gift_scratch(
    seed_reports: dict[str, dict[str, Any]],
    **_: Any,
) -> tuple[str, str]:
    return "gift_scratch_recorded", "defer_to_present_source_gate"


def _decide_present_source_seed_robustness(
    seed_reports: dict[str, dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    **_: Any,
) -> tuple[str, str]:
    report = seed_reports[str(expected_seeds[0])]
    observed = {
        "absolute_auc": report["aucs"]["candidate"],
        "architecture_margin": report["architecture_margin"],
        "topology_margin": report["topology_margin"],
        "representation_margin": report["representation_margin"],
    }
    thresholds = {
        "absolute_auc": 0.65,
        "architecture_margin": -0.01,
        "topology_margin": 0.003,
        "representation_margin": 0.003,
    }
    if all(observed[name] >= threshold for name, threshold in thresholds.items()):
        return (
            "e4_r5_source_seed_gate_pass",
            "freeze_source_seed1_hashes_and_prepare_target_seed4_seed5",
        )
    return (
        "e4_r5_source_seed_gate_fail",
        "stop_formal_scale_retain_conditional_e4_r4_result",
    )


def _source_seed_stopped_actions(decision: str) -> list[dict[str, str]]:
    if decision == "implementation_ready":
        actions = ("interpret_readiness_metrics", "remote_target_launch")
    elif decision == "e4_r5_source_seed_gate_pass":
        actions = ("local_65536_per_class", "262144_per_class", "formal_scale")
    else:
        actions = (
            "remote_target_launch",
            "sample_scale",
            "formal_scale",
            "cross_spn_transfer_claim",
        )
    return [
        {"action": action, "reason": f"stopped_by_decision:{decision}"}
        for action in actions
    ]


def _stopped_actions(decision: str) -> list[dict[str, str]]:
    if decision == "implementation_ready":
        actions = ("interpret_r0_metrics", "remote_scale", "e4_r2")
    elif decision == "promote_e4_r2":
        actions = ("remote_scale", "65536_per_class", "262144_per_class")
    elif decision == "run_present_seed1_fragility":
        actions = ("repeat_gift", "remote_scale", "sample_scale", "e4_r2")
    else:
        actions = ("present_seed1", "gift_repeat", "remote_scale", "e4_r2")
    return [
        {"action": action, "reason": f"stopped_by_decision:{decision}"}
        for action in actions
    ]


_PRESENT_GATE_SPEC = FourRoleGateSpec(
    protocol=_PRESENT_E4_PROTOCOL,
    model_roles=PRESENT_CROSS_SPN_MODEL_ROLES,
    anchor_options=_PRESENT_ANCHOR_OPTIONS,
    hybrid_options=_TYPED_OPTIONS,
    capacity_label="present_cross_spn_typed_cell",
    semantic_checks={
        "raw_input_bits": 2048,
        "pair_bits": 128,
        "pairs_per_sample": 16,
        "typed_cell_shape": ["batch", "pair", 16, 4],
        "mapping_modes": ["true", "shuffled", "raw"],
        "effective_key_schedule": "per_pair_random",
        "ddt_or_trail_input": False,
        "model_test_reference": "tests/test_cross_spn_typed_cell.py",
        "status": "pass",
    },
    allowed_seed_layouts=((0,),),
    readiness_next_action="run_frozen_e4_r1_local_diagnostic",
    claim_label="typed-cell source attribution diagnostic",
    decide=_decide_present_source,
    stopped_actions=_stopped_actions,
    representation_role="raw_delta",
)

_GIFT_GATE_SPEC = FourRoleGateSpec(
    protocol=_GIFT_E4_PROTOCOL,
    model_roles=GIFT_CROSS_SPN_MODEL_ROLES,
    anchor_options=_GIFT_ANCHOR_OPTIONS,
    hybrid_options=_TYPED_OPTIONS,
    capacity_label="gift_cross_spn_typed_cell",
    semantic_checks={
        "raw_input_bits": 512,
        "pair_bits": 128,
        "pairs_per_sample": 4,
        "typed_cell_shape": ["batch", "pair", 16, 4],
        "mapping_modes": ["true", "shuffled", "raw"],
        "effective_key_schedule": "fixed",
        "anchor_equivalence_test": "tests/test_cross_spn_typed_cell.py",
        "ddt_or_trail_input": False,
        "status": "pass",
    },
    allowed_seed_layouts=((0,),),
    readiness_next_action="run_frozen_e4_r1_local_diagnostic",
    claim_label="typed-cell scratch attribution diagnostic",
    decide=_decide_gift_scratch,
    stopped_actions=_stopped_actions,
    representation_role="raw_delta",
)

_E4_R5_PRESENT_PROTOCOL = replace(
    _PRESENT_E4_PROTOCOL,
    claim_prefix="strict PRESENT r7 E4-R5 independent source seed",
    readiness_identity_label="E4-R5 source readiness",
    readiness_claim_scope="E4-R5 source-seed1 readiness only; metrics not interpreted",
    research_claim_suffix="local source-seed diagnostic only; not formal, paper-scale, remote, or breakthrough evidence",
    readiness_seed_layout=(1,),
)

_E4_R5_PRESENT_GATE_SPEC = replace(
    _PRESENT_GATE_SPEC,
    protocol=_E4_R5_PRESENT_PROTOCOL,
    allowed_seed_layouts=((1,),),
    readiness_next_action="run_e4_r5_source_seed1_8192_gate",
    claim_label="E4-R5 independent source-seed attribution diagnostic",
    decide=_decide_present_source_seed_robustness,
    stopped_actions=_source_seed_stopped_actions,
)


def _evaluate(
    results_paths: list[Path],
    progress_paths: list[Path],
    *,
    expected_seeds: tuple[int, ...],
    samples_per_class: int,
    epochs: int,
    readiness_only: bool,
    spec: FourRoleGateSpec,
) -> dict[str, Any]:
    return evaluate_four_role_attribution(
        results_paths,
        progress_paths=progress_paths,
        expected_seeds=expected_seeds,
        samples_per_class=samples_per_class,
        epochs=epochs,
        readiness_only=readiness_only,
        seed0_architecture_margin=0.0,
        seed0_topology_margin=0.0,
        seed0_representation_margin=0.0,
        joint_architecture_margin=0.0,
        joint_control_margin=0.0,
        spec=spec,
    )


def gate_cross_spn_typed_cell(
    present_results_paths: list[Path],
    *,
    present_progress_paths: list[Path],
    gift_results_paths: list[Path],
    gift_progress_paths: list[Path],
    expected_seeds: tuple[int, ...] = (0,),
    samples_per_class: int = 8192,
    epochs: int = 10,
    readiness_only: bool = False,
) -> dict[str, Any]:
    present = _evaluate(
        present_results_paths,
        present_progress_paths,
        expected_seeds=expected_seeds,
        samples_per_class=samples_per_class,
        epochs=epochs,
        readiness_only=readiness_only,
        spec=_PRESENT_GATE_SPEC,
    )
    gift = _evaluate(
        gift_results_paths,
        gift_progress_paths,
        expected_seeds=expected_seeds,
        samples_per_class=samples_per_class,
        epochs=epochs,
        readiness_only=readiness_only,
        spec=_GIFT_GATE_SPEC,
    )
    errors = [
        *[f"present:{error}" for error in present.get("errors", [])],
        *[f"gift:{error}" for error in gift.get("errors", [])],
    ]
    if not errors and present["parameter_counts"]["candidate"] != gift[
        "parameter_counts"
    ]["candidate"]:
        errors.append(
            "cross_cipher:typed_candidate_parameter_count_mismatch "
            f"present={present['parameter_counts']['candidate']} "
            f"gift={gift['parameter_counts']['candidate']}"
        )
    if errors:
        return {
            "status": "fail",
            "decision": "invalid_e4_protocol",
            "errors": errors,
            "present": present,
            "gift": gift,
            "stopped_actions": _stopped_actions("invalid_e4_protocol"),
            "next_action": "repair_e4_protocol_and_replay_same_matrix",
            "claim_scope": "invalid E4 cross-SPN evidence",
            "research_decision_applied": False,
        }
    if readiness_only:
        decision = "implementation_ready"
        next_action = "run_frozen_e4_r1_local_diagnostic"
        claim_scope = "E4-R0 implementation readiness only; metrics not interpreted"
    else:
        decision = present["decision"]
        next_action = present["next_action"]
        claim_scope = (
            f"{samples_per_class}/class cross-SPN typed-cell local diagnostic; "
            "not formal, paper-scale, remote, or breakthrough evidence"
        )
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_seeds": list(expected_seeds),
        "samples_per_class": samples_per_class,
        "epochs": epochs,
        "present": present,
        "gift": gift,
        "cross_cipher_typed_parameter_count": present["parameter_counts"][
            "candidate"
        ],
        "stopped_actions": _stopped_actions(decision),
        "next_action": next_action,
        "claim_scope": claim_scope,
        "research_decision_applied": not readiness_only,
    }


def gate_cross_spn_present_source_seed(
    results_paths: list[Path],
    *,
    progress_paths: list[Path],
    expected_seed: int = 1,
    samples_per_class: int = 8192,
    epochs: int = 10,
    readiness_only: bool = False,
) -> dict[str, Any]:
    report = _evaluate(
        results_paths,
        progress_paths,
        expected_seeds=(expected_seed,),
        samples_per_class=samples_per_class,
        epochs=epochs,
        readiness_only=readiness_only,
        spec=_E4_R5_PRESENT_GATE_SPEC,
    )
    if report["status"] != "pass":
        return report
    report["experiment_stage"] = "e4_r5_source_seed"
    report["research_question"] = "independent PRESENT source checkpoint seed"
    report["claim_scope"] = (
        "E4-R5 source-seed1 readiness only; metrics not interpreted"
        if readiness_only
        else f"{samples_per_class}/class independent PRESENT source-seed diagnostic; "
        "not formal, paper-scale, remote, target-adaptation, or breakthrough evidence"
    )
    return report


__all__ = [
    "GIFT_CROSS_SPN_MODEL_ROLES",
    "PRESENT_CROSS_SPN_MODEL_ROLES",
    "gate_cross_spn_present_source_seed",
    "gate_cross_spn_typed_cell",
]
