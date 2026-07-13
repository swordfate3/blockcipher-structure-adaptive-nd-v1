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


SAME_INPUT_DBITNET_MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_invp_dbitnet2023",
    "shuffled_p": "present_shuffled_p_dbitnet2023",
    "raw_delta": "present_raw_delta_dbitnet2023",
}


def gate_present_same_input_dbitnet(
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
        spec=_SAME_INPUT_DBITNET_GATE_SPEC,
    )


def _decide_same_input_dbitnet(
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
        return (
            "reject_e3_r1",
            "stop_dbitnet_component_gate_and_keep_token_mixer_anchor",
        )

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
            return "promote_seed1", "run_identical_e3_r1_seed1_local_gate"
        return "weak_or_fragile_no_scale", "inspect_histories_once_and_stop_e3_r1"

    if (
        all(report["candidate_above_all"] for report in reports)
        and min(report["architecture_margin"] for report in reports)
        >= joint_architecture_margin
        and min(report["topology_margin"] for report in reports)
        >= joint_control_margin
        and min(report["representation_margin"] for report in reports)
        >= joint_control_margin
    ):
        return (
            "two_seed_supported_medium_candidate",
            "prepare_65536_per_class_diagnostic_plan",
        )
    return "unstable_no_scale", "stop_e3_r1_after_two_seed_variance"


def _stopped_actions(decision: str) -> list[dict[str, str]]:
    if decision == "implementation_ready":
        actions = ("interpret_smoke_metrics", "seed1", "remote_scale")
    elif decision == "promote_seed1":
        actions = ("65536_per_class", "262144_per_class", "remote_scale")
    elif decision == "two_seed_supported_medium_candidate":
        actions = ("262144_per_class", "1000000_per_class", "formal_claim")
    else:
        actions = ("seed1", "65536_per_class", "262144_per_class", "remote_scale")
    return [
        {"action": action, "reason": f"stopped_by_decision:{decision}"}
        for action in actions
    ]


_SAME_INPUT_DBITNET_GATE_SPEC = FourRoleGateSpec(
    protocol=PRESENT_CASE2_ATTRIBUTION_PROTOCOL,
    model_roles=SAME_INPUT_DBITNET_MODEL_ROLES,
    anchor_options={
        "spn_mixer_depth": 2,
        "activation": "relu",
        "norm": "layernorm",
    },
    hybrid_options={},
    capacity_label="same_input_dbitnet",
    semantic_checks={
        "raw_input_bits": 2048,
        "mapped_input_bits": 1024,
        "pair_bits": 128,
        "pairs_per_sample": 16,
        "mapped_word_bits": 64,
        "mapped_words_per_sample": 16,
        "adapter_mapping_identities": {
            "candidate": "true_inverse_p",
            "shuffled_p": "deterministic_shuffled_p",
            "raw_delta": "identity",
        },
        "learner": "AutoNDDBitNet2023Distinguisher",
        "dbitnet_capacity_constraint": "equal total and trainable counts",
        "negative_mode": "encrypted_random_plaintexts",
        "effective_key_schedule": "per_pair_random",
        "ddt_or_trail_input": False,
        "model_test_reference": "tests/test_present_same_input_dbitnet.py",
        "status": "pass",
    },
    allowed_seed_layouts=((0,), (1,), (0, 1)),
    readiness_next_action="run_frozen_e3_r1_seed0_local_diagnostic",
    claim_label="same-input DBitNet component-attribution diagnostic",
    decide=_decide_same_input_dbitnet,
    stopped_actions=_stopped_actions,
    representation_role="raw_delta",
)


__all__ = [
    "SAME_INPUT_DBITNET_MODEL_ROLES",
    "gate_present_same_input_dbitnet",
]
