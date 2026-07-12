from __future__ import annotations

from pathlib import Path
from typing import Any

from blockcipher_nd.planning.four_role_attribution_gate import (
    FourRoleGateSpec,
    evaluate_four_role_attribution,
)


TOPOLOGY_RESIDUAL_MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_invp_topology_residual_spn_only",
    "shuffled_p": "present_nibble_shuffled_p_topology_residual_spn_only",
    "delta_only": "present_nibble_delta_topology_residual_spn_only",
}


def gate_invp_topology_residual(
    results_paths: list[Path],
    *,
    progress_paths: list[Path],
    expected_seeds: tuple[int, ...] = (0,),
    samples_per_class: int = 8192,
    epochs: int = 10,
    readiness_only: bool = False,
    seed0_architecture_margin: float = 0.003,
    seed0_topology_margin: float = 0.003,
    seed0_representation_margin: float = 0.003,
    joint_architecture_margin: float = 0.001,
    joint_control_margin: float = 0.002,
) -> dict[str, Any]:
    return evaluate_four_role_attribution(
        results_paths,
        progress_paths=progress_paths,
        expected_seeds=expected_seeds,
        samples_per_class=samples_per_class,
        epochs=epochs,
        readiness_only=readiness_only,
        seed0_architecture_margin=seed0_architecture_margin,
        seed0_topology_margin=seed0_topology_margin,
        seed0_representation_margin=seed0_representation_margin,
        joint_architecture_margin=joint_architecture_margin,
        joint_control_margin=joint_control_margin,
        spec=_TOPOLOGY_RESIDUAL_GATE_SPEC,
    )


def _decide_topology_residual(
    seed_reports: dict[str, dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    seed0_architecture_margin: float | None,
    seed0_topology_margin: float,
    seed0_representation_margin: float,
    joint_architecture_margin: float,
    joint_control_margin: float,
) -> tuple[str, str]:
    reports = [seed_reports[str(seed)] for seed in expected_seeds]
    if any(report["architecture_margin"] <= 0.0 for report in reports):
        return (
            "stop_topology_residual",
            "keep_token_mixer_anchor_and_stop_adapter",
        )
    if any(report["topology_margin"] <= 0.0 for report in reports):
        return (
            "stop_true_topology_attribution",
            "stop_adapter_without_true_topology_attribution",
        )
    if any(report["representation_margin"] <= 0.0 for report in reports):
        return (
            "stop_invp_adapter_attribution",
            "stop_adapter_without_invp_attribution",
        )

    if len(reports) == 1:
        architecture_threshold = (
            0.003 if seed0_architecture_margin is None else seed0_architecture_margin
        )
        report = reports[0]
        if (
            report["architecture_margin"] >= architecture_threshold
            and report["topology_margin"] >= seed0_topology_margin
            and report["representation_margin"] >= seed0_representation_margin
        ):
            return "promote_seed1", "run_identical_seed1_local_gate"
        return (
            "weak_or_fragile_no_scale",
            "inspect_histories_once_and_do_not_scale",
        )

    minimum_architecture_margin = min(
        report["architecture_margin"] for report in reports
    )
    minimum_topology_margin = min(report["topology_margin"] for report in reports)
    minimum_representation_margin = min(
        report["representation_margin"] for report in reports
    )
    if (
        all(report["candidate_above_all"] for report in reports)
        and minimum_architecture_margin >= joint_architecture_margin
        and minimum_topology_margin >= joint_control_margin
        and minimum_representation_margin >= joint_control_margin
    ):
        return (
            "promote_medium_65536",
            "run_65536_per_class_two_seed_medium_confirmation",
        )
    return (
        "unstable_no_remote_scale",
        "do_not_launch_remote_scale_inspect_two_seed_variance",
    )


def _stopped_actions(decision: str) -> list[dict[str, str]]:
    if decision == "implementation_ready":
        actions = ("interpret_smoke_metrics", "remote_scale")
    elif decision == "promote_seed1":
        actions = ("65536_per_class", "262144_per_class", "remote_scale")
    elif decision == "promote_medium_65536":
        actions = ("262144_per_class", "remote_scale")
    else:
        actions = ("seed1", "65536_per_class", "262144_per_class", "remote_scale")
    return [
        {"action": action, "reason": f"stopped_by_decision:{decision}"}
        for action in actions
    ]


_TOPOLOGY_RESIDUAL_GATE_SPEC = FourRoleGateSpec(
    model_roles=TOPOLOGY_RESIDUAL_MODEL_ROLES,
    anchor_options={
        "spn_mixer_depth": 2,
        "activation": "relu",
        "norm": "layernorm",
    },
    hybrid_options={
        "spn_mixer_depth": 2,
        "token_mlp_ratio": 2,
        "local_channels": 16,
        "local_depth": 1,
        "local_kernel_size": 3,
        "local_residual_scale_init": 0.1,
        "activation": "relu",
        "norm": "layernorm",
        "local_norm": "batchnorm2d",
        "dropout": 0.0,
    },
    capacity_label="topology_residual",
    semantic_checks={
        "raw_input_bits": 2048,
        "pair_bits": 128,
        "pairs_per_sample": 16,
        "common_token_backbone": {
            "view_mode": "inv_p",
            "p_alignment": "true",
            "roles": ["candidate", "shuffled_p", "delta_only"],
        },
        "adapter_mapping_identities": {
            "candidate": "true_inv_p",
            "shuffled_p": "deterministic_shuffled_p",
            "delta_only": "raw_delta",
        },
        "pair_fusion": "token + alpha * local",
        "hybrid_capacity_constraint": "equal total and trainable counts",
        "negative_mode": "encrypted_random_plaintexts",
        "effective_key_schedule": "per_pair_random",
        "model_test_reference": "tests/test_invp_topology_residual.py",
        "evidence_kind": (
            "frozen/model-tested semantic contract; result gate verifies identities, "
            "options, capacity, and protocol, not runtime tensor equality"
        ),
        "status": "pass",
    },
    allowed_seed_layouts=((0,), (0, 1)),
    readiness_next_action="run_frozen_h1_seed0_local_diagnostic",
    claim_label="topology-residual architecture-attribution diagnostic",
    decide=_decide_topology_residual,
    stopped_actions=_stopped_actions,
)


__all__ = ["TOPOLOGY_RESIDUAL_MODEL_ROLES", "gate_invp_topology_residual"]
