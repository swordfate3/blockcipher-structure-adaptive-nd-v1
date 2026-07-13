from __future__ import annotations

from pathlib import Path
from typing import Any

from blockcipher_nd.planning.four_role_attribution_gate import (
    FourRoleGateSpec,
    evaluate_four_role_attribution,
)
from blockcipher_nd.planning.present_case2_attribution_protocol import (
    PRESENT_CASE2_ATTRIBUTION_PROTOCOL,
)


CASE3_MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_case3_invp_topology_residual_spn_only",
    "shuffled_p": "present_nibble_case3_shuffled_p_topology_residual_spn_only",
    "raw_triple": "present_nibble_case3_raw_topology_residual_spn_only",
}


def gate_present_case3_topology_residual(
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
        spec=_CASE3_GATE_SPEC,
    )


def _decide_case3(
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
    if any(
        report[margin] <= 0.0
        for report in reports
        for margin in (
            "architecture_margin",
            "topology_margin",
            "representation_margin",
        )
    ):
        return "reject_h2", "stop_h2_and_keep_token_mixer_anchor"

    if len(reports) == 1:
        report = reports[0]
        architecture_threshold = (
            0.003 if seed0_architecture_margin is None else seed0_architecture_margin
        )
        if (
            report["architecture_margin"] >= architecture_threshold
            and report["topology_margin"] >= seed0_topology_margin
            and report["representation_margin"] >= seed0_representation_margin
        ):
            return "promote_seed1", "run_identical_h2_seed1_local_gate"
        return (
            "weak_or_fragile_no_scale",
            "inspect_histories_once_and_stop_h2_scaling",
        )

    if (
        all(report["candidate_above_all"] for report in reports)
        and min(report["architecture_margin"] for report in reports)
        >= joint_architecture_margin
        and min(report["topology_margin"] for report in reports)
        >= joint_control_margin
        and min(report["representation_margin"] for report in reports)
        >= joint_control_margin
    ):
        return "two_seed_supported_no_scale", "design_next_method_before_scale"
    return "unstable_no_scale", "stop_h2_after_two_seed_variance"


def _stopped_actions(decision: str) -> list[dict[str, str]]:
    if decision == "implementation_ready":
        actions = ("interpret_smoke_metrics", "seed1", "remote_scale")
    elif decision == "promote_seed1":
        actions = ("65536_per_class", "262144_per_class", "remote_scale")
    else:
        actions = ("seed1", "65536_per_class", "262144_per_class", "remote_scale")
    return [
        {"action": action, "reason": f"stopped_by_decision:{decision}"}
        for action in actions
    ]


_CASE3_GATE_SPEC = FourRoleGateSpec(
    protocol=PRESENT_CASE2_ATTRIBUTION_PROTOCOL,
    model_roles=CASE3_MODEL_ROLES,
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
    capacity_label="case3_topology_residual",
    semantic_checks={
        "raw_input_bits": 2048,
        "pair_bits": 128,
        "pairs_per_sample": 16,
        "case3_tensor_shape": ["batch", "pair", 3, 4, 16],
        "case3_channels": ["C0", "C1", "mapped_delta"],
        "unmapped_ciphertext_channels": ["C0", "C1"],
        "adapter_mapping_identities": {
            "candidate": "true_inv_p_on_delta_only",
            "shuffled_p": "deterministic_shuffled_p_on_delta_only",
            "raw_triple": "identity_on_delta_only",
        },
        "pair_fusion": "token + alpha * local",
        "hybrid_capacity_constraint": "equal total and trainable counts",
        "negative_mode": "encrypted_random_plaintexts",
        "effective_key_schedule": "per_pair_random",
        "model_test_reference": "tests/test_present_case3_topology_residual.py",
        "status": "pass",
    },
    allowed_seed_layouts=((0,), (0, 1)),
    readiness_next_action="run_frozen_h2_seed0_local_diagnostic",
    claim_label="Case3 three-channel topology-residual attribution diagnostic",
    decide=_decide_case3,
    stopped_actions=_stopped_actions,
    representation_role="raw_triple",
)


__all__ = ["CASE3_MODEL_ROLES", "gate_present_case3_topology_residual"]
