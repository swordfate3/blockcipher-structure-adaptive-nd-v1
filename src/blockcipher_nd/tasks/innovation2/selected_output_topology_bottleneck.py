from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    KimuraOutputPredictionConfig,
    prepare_disk_output_prediction_data,
)
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    SelectedOutputTopologyBottleneckSpn,
    _build_model,
    _present_topology_mapping,
    _train_one_model,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    validate_selected_output_contract,
)


OPA3_REMOTE_RUN_ID = (
    "i2_output_prediction_opa3_present_r3_selected8_topology_attribution_key3_gpu0_20260722"
)
OPA3_DECISION = "innovation2_selected8_present_topology_not_attributed"
RUN_ID_PREFIX = "i2_output_prediction_opb1_present_r3_topology_bottleneck"
MODEL_SPECS = (
    (
        "selected8_present_spn_anchor_exact_p_true_output",
        "present_spn_exact_p",
        False,
    ),
    (
        "selected8_topology_bottleneck_exact_p_true_output",
        "topology_bottleneck_exact_p",
        False,
    ),
    (
        "selected8_topology_bottleneck_wrong_p_true_output",
        "topology_bottleneck_wrong_p",
        False,
    ),
    (
        "selected8_topology_bottleneck_exact_p_label_shuffle",
        "topology_bottleneck_exact_p",
        True,
    ),
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class TopologyBottleneckConfig:
    run_id: str = f"{RUN_ID_PREFIX}_smoke_20260722"
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 4
    train_rows: int = 64
    test_rows: int = 64
    mlp_hidden_dim: int = 1936
    lstm_hidden_dim: int = 300
    lstm_layers: int = 6
    present_spn_dim: int = 189
    present_spn_blocks: int = 3
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    data_chunk_rows: int = 32
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES
    maximum_parameter_gap: float = 0.03
    minimum_anchor_mean_auc: float = 0.900
    minimum_candidate_mean_auc: float = 0.900
    minimum_candidate_vs_anchor_mean_margin: float = -0.050
    minimum_mean_control_margin: float = 0.030
    minimum_per_bit_auc: float = 0.550
    minimum_per_bit_control_margin: float = 0.020
    minimum_per_bit_anchor_margin: float = -0.100
    minimum_accuracy_margin: float = 0.005
    minimum_attributed_bits: int = 4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "topology_bottleneck"}:
            raise ValueError("invalid OPB1 mode")
        if self.rounds != 3 or self.seed != 4:
            raise ValueError("OPB1 is frozen to PRESENT round three and key seed4")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPB1 positions must match OP10 through OPA3")
        integers = (
            self.train_rows,
            self.test_rows,
            self.present_spn_dim,
            self.present_spn_blocks,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
            self.minimum_attributed_bits,
        )
        if min(integers) <= 0 or self.learning_rate <= 0:
            raise ValueError("OPB1 row, model, training, and gate values must be positive")

    @classmethod
    def formal(
        cls,
        *,
        run_id: str | None = None,
        device: str = "cuda",
    ) -> TopologyBottleneckConfig:
        return cls(
            run_id=run_id or f"{RUN_ID_PREFIX}_key4_gpu0_20260722",
            mode="topology_bottleneck",
            train_rows=1 << 17,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )


def authorize_from_opa3_gate(gate: dict[str, Any]) -> dict[str, float]:
    if gate.get("run_id") != OPA3_REMOTE_RUN_ID:
        raise ValueError("OPB1 requires the frozen OPA3 formal gate")
    if gate.get("status") != "hold" or gate.get("decision") != OPA3_DECISION:
        raise ValueError("OPB1 requires OPA3's verified topology hold")
    for group in ("protocol_checks", "execution_checks"):
        checks = gate.get(group)
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError(f"OPA3 {group} did not fully pass")
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("priority_passed") is not False:
        raise ValueError("OPA3 must be a valid negative attribution result")
    means = metrics.get("mean_auc_by_model")
    if not isinstance(means, dict):
        raise ValueError("OPA3 model means are missing")
    exact = float(means["present_spn_exact_p_true_output"])
    wrong = float(means["present_spn_wrong_p_true_output"])
    if not math.isclose(exact, wrong, abs_tol=1e-12):
        raise ValueError("OPA3 hold is not the frozen exact-versus-wrong tie")
    if int(metrics.get("attributed_bit_count", -1)) != 0:
        raise ValueError("OPA3 attributed-bit count must remain zero")
    return {"exact_mean_auc": exact, "wrong_mean_auc": wrong}


def prepare_bottleneck_data(
    config: TopologyBottleneckConfig,
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    data_config = KimuraOutputPredictionConfig(
        run_id=config.run_id,
        mode="smoke",
        rounds=config.rounds,
        seed=config.seed,
        train_rows=config.train_rows,
        test_rows=config.test_rows,
        hidden_dim=config.lstm_hidden_dim,
        layers=config.lstm_layers,
        mlp_hidden_dim=config.mlp_hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        data_chunk_rows=config.data_chunk_rows,
        device=config.device,
    )
    return prepare_disk_output_prediction_data(
        data_config,
        output_root,
        progress=progress,
    )


def bottleneck_parameter_counts(config: TopologyBottleneckConfig) -> dict[str, int]:
    return {
        architecture: sum(
            parameter.numel()
            for parameter in _build_model(config, architecture).parameters()  # type: ignore[arg-type]
        )
        for architecture in {architecture for _, architecture, _ in MODEL_SPECS}
    }


def validate_bottleneck_contract(
    config: TopologyBottleneckConfig,
    data: dict[str, Any],
    opa3_gate: dict[str, Any],
) -> dict[str, bool]:
    authorize_from_opa3_gate(opa3_gate)
    selected_config = SelectedOutputBitHeadConfig(
        run_id=config.run_id,
        mode="smoke",
        rounds=config.rounds,
        seed=config.seed,
        train_rows=config.train_rows,
        test_rows=config.test_rows,
        hidden_dim=config.mlp_hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        data_chunk_rows=config.data_chunk_rows,
        selected_msb_indices=config.selected_msb_indices,
        device=config.device,
    )
    checks = validate_selected_output_contract(selected_config, data)
    checks.pop("independent_key_seed_is_one")
    counts = bottleneck_parameter_counts(config)
    anchor_count = counts["present_spn_exact_p"]
    candidate_count = counts["topology_bottleneck_exact_p"]
    candidate_gap = abs(candidate_count - anchor_count) / anchor_count
    with torch.random.fork_rng():
        torch.manual_seed(1_640_000 + config.seed)
        exact = _build_model(config, "topology_bottleneck_exact_p")  # type: ignore[arg-type]
        torch.manual_seed(1_640_000 + config.seed)
        wrong = _build_model(config, "topology_bottleneck_wrong_p")  # type: ignore[arg-type]
    exact_state = exact.state_dict()
    wrong_state = wrong.state_dict()
    keys = {
        seed: random.Random(910_000 + seed).getrandbits(80)
        for seed in range(5)
    }
    checks.update(
        {
            "fifth_fixed_key_seed_is_four": config.seed == 4,
            "fifth_fixed_key_differs_from_seed0_through_seed3": len(
                set(keys.values())
            )
            == 5
            and int(data["secret_key"]) == keys[4],
            "opa3_hold_authorizes_new_method_hypothesis": True,
            "four_frozen_matrix_rows": len(MODEL_SPECS) == 4,
            "candidate_variants_have_identical_parameter_counts": counts[
                "topology_bottleneck_exact_p"
            ]
            == counts["topology_bottleneck_wrong_p"],
            "candidate_within_three_percent_of_anchor": candidate_gap
            <= config.maximum_parameter_gap,
            "candidate_has_no_full_position_embedding": isinstance(
                exact, SelectedOutputTopologyBottleneckSpn
            )
            and not hasattr(exact, "position_embedding"),
            "candidate_uses_one_scalar_per_position_per_block": isinstance(
                exact, SelectedOutputTopologyBottleneckSpn
            )
            and all(tuple(item.shape) == (1, 64, 1) for item in exact.key_strengths),
            "exact_and_wrong_trainable_states_match_at_initialization": exact_state.keys()
            == wrong_state.keys()
            and all(
                torch.equal(exact_state[name], wrong_state[name])
                for name in exact_state
            ),
            "exact_and_wrong_topologies_differ_at_all_positions": bool(
                np.all(
                    _present_topology_mapping("exact").numpy()
                    != _present_topology_mapping("wrong").numpy()
                )
            ),
            "labels_are_true_outputs_not_sample_classes": True,
        }
    )
    return checks


def train_bottleneck_matrix(
    config: TopologyBottleneckConfig,
    data: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    train_features = np.array(data["features"][: config.train_rows], copy=True)
    test_features = np.array(data["features"][config.train_rows :], copy=True)
    selected = np.asarray(config.selected_msb_indices, dtype=np.int64)
    train_targets = np.array(
        data["full_targets"][: config.train_rows, selected], copy=True
    )
    test_targets = np.array(
        data["full_targets"][config.train_rows :, selected], copy=True
    )
    shuffle = np.random.default_rng(1_620_000 + config.seed).permutation(
        config.train_rows
    )
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for model_name, architecture, shuffle_labels in MODEL_SPECS:
        model_targets = np.array(train_targets, copy=True)
        if shuffle_labels:
            model_targets = model_targets[shuffle]
        result = _train_one_model(
            config,  # type: ignore[arg-type]
            model_name=model_name,
            architecture=architecture,
            train_features=train_features,
            train_targets=model_targets,
            test_features=test_features,
            test_targets=test_targets,
            output_root=output_root,
            progress=progress,
        )
        rows.extend(result["rows"])
        summaries.append(result["summary"])
        history.extend(result["history"])
        checkpoints.append(result["checkpoint"])
    return {
        "rows": rows,
        "summaries": summaries,
        "history": history,
        "checkpoints": checkpoints,
    }


def adjudicate_bottleneck(
    config: TopologyBottleneckConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in training["rows"]
    }
    model_names = [model for model, _, _ in MODEL_SPECS]
    means = {
        model: float(
            np.mean(
                [
                    float(indexed[(model, bit)]["auc"])
                    for bit in config.selected_msb_indices
                ]
            )
        )
        for model in model_names
    }
    anchor_name, exact_name, wrong_name, shuffle_name = model_names
    bit_gates: list[dict[str, Any]] = []
    for bit in config.selected_msb_indices:
        anchor = indexed[(anchor_name, bit)]
        exact = indexed[(exact_name, bit)]
        wrong = indexed[(wrong_name, bit)]
        shuffled = indexed[(shuffle_name, bit)]
        attribution_bit_checks = {
            "candidate_auc_at_least_0_550": float(exact["auc"])
            >= config.minimum_per_bit_auc,
            "candidate_minus_wrong_at_least_0_020": float(exact["auc"])
            - float(wrong["auc"])
            >= config.minimum_per_bit_control_margin,
            "candidate_minus_shuffle_at_least_0_020": float(exact["auc"])
            - float(shuffled["auc"])
            >= config.minimum_per_bit_control_margin,
        }
        utility_bit_checks = {
            "candidate_minus_anchor_at_least_minus_0_100": float(exact["auc"])
            - float(anchor["auc"])
            >= config.minimum_per_bit_anchor_margin,
            "accuracy_margin_at_least_0_005": float(
                exact["accuracy_minus_majority"]
            )
            >= config.minimum_accuracy_margin,
        }
        bit_gates.append(
            {
                "msb_index": bit,
                "anchor_auc": float(anchor["auc"]),
                "candidate_auc": float(exact["auc"]),
                "wrong_auc": float(wrong["auc"]),
                "shuffle_auc": float(shuffled["auc"]),
                "candidate_minus_anchor_auc": float(exact["auc"])
                - float(anchor["auc"]),
                "candidate_minus_wrong_auc": float(exact["auc"])
                - float(wrong["auc"]),
                "candidate_minus_shuffle_auc": float(exact["auc"])
                - float(shuffled["auc"]),
                "accuracy_minus_majority": float(
                    exact["accuracy_minus_majority"]
                ),
                "attribution_checks": attribution_bit_checks,
                "utility_checks": utility_bit_checks,
                "attributed": all(attribution_bit_checks.values()),
                "passed": all(attribution_bit_checks.values())
                and all(utility_bit_checks.values()),
            }
        )
    attributed_bits = sum(row["attributed"] for row in bit_gates)
    joint_passed_bits = sum(row["passed"] for row in bit_gates)
    attribution_checks = {
        "candidate_minus_wrong_mean_auc_at_least_0_030": means[exact_name]
        - means[wrong_name]
        >= config.minimum_mean_control_margin,
        "candidate_minus_shuffle_mean_auc_at_least_0_030": means[exact_name]
        - means[shuffle_name]
        >= config.minimum_mean_control_margin,
        "at_least_four_bits_attributed": attributed_bits
        >= config.minimum_attributed_bits,
    }
    utility_checks = {
        "anchor_mean_auc_at_least_0_900": means[anchor_name]
        >= config.minimum_anchor_mean_auc,
        "candidate_mean_auc_at_least_0_900": means[exact_name]
        >= config.minimum_candidate_mean_auc,
        "candidate_minus_anchor_mean_auc_at_least_minus_0_050": means[exact_name]
        - means[anchor_name]
        >= config.minimum_candidate_vs_anchor_mean_margin,
        "at_least_four_bits_pass_joint_gate": joint_passed_bits
        >= config.minimum_attributed_bits,
    }
    execution_checks = {
        "four_models_complete": len(training["summaries"]) == 4,
        "thirty_two_result_rows_complete": len(training["rows"]) == 32,
        "history_rows_complete": len(training["history"]) == config.epochs * 4,
        "four_checkpoint_hashes_present": len(training["checkpoints"]) == 4
        and all(row.get("sha256") for row in training["checkpoints"]),
        "all_metrics_are_finite": all(
            math.isfinite(float(row[field]))
            for row in training["rows"]
            for field in (
                "threshold_accuracy",
                "majority_accuracy",
                "accuracy_minus_majority",
                "auc",
                "mse",
                "invalid_numpy_rint_rate",
            )
        ),
        "matched_shuffle_uses_true_test_targets": indexed[
            (shuffle_name, config.selected_msb_indices[0])
        ]["test_target_identity"]
        == "true_selected_ciphertext_targets",
    }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    execution_valid = all(execution_checks.values())
    attribution_passed = all(attribution_checks.values())
    utility_passed = all(utility_checks.values())
    if not protocol_valid or not execution_valid:
        status = "fail"
        decision = "innovation2_topology_bottleneck_protocol_invalid"
        action = "repair only the frozen OPA3 authority, data, model, control, or artifact protocol"
        next_adjudication = "opb1_protocol_repair"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_topology_bottleneck_local_smoke_passed"
        action = "launch the frozen seed4 four-row OPB1 topology-bottleneck matrix"
        next_adjudication = "opb1_remote_seed4_topology_bottleneck"
    elif attribution_passed and utility_passed:
        status = "pass"
        decision = "innovation2_topology_bottleneck_ready_for_independent_confirmation"
        action = "repeat the unchanged four-row matrix under the sixth fixed key seed5"
        next_adjudication = "opb2_seed5_independent_confirmation"
    elif attribution_passed:
        status = "hold"
        decision = "innovation2_topology_bottleneck_attributed_with_performance_cost"
        action = "preregister at most one training-isolated low-rank conditioning repair; do not extend rounds"
        next_adjudication = "opb1_low_rank_capacity_repair_or_thesis_boundary"
    else:
        status = "hold"
        decision = "innovation2_topology_bottleneck_not_attributed"
        action = "stop this bottleneck route and rank an SPN-ResCNN hybrid as a new hypothesis"
        next_adjudication = "innovation2_spn_rescnn_hybrid_design_or_thesis_boundary"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "bit_gates": bit_gates,
        "metrics": {
            "mean_auc_by_model": means,
            "candidate_minus_anchor_mean_auc": means[exact_name]
            - means[anchor_name],
            "candidate_minus_wrong_mean_auc": means[exact_name]
            - means[wrong_name],
            "candidate_minus_shuffle_mean_auc": means[exact_name]
            - means[shuffle_name],
            "attributed_bit_count": attributed_bits,
            "joint_passed_bit_count": joint_passed_bits,
            "attribution_checks": attribution_checks,
            "utility_checks": utility_checks,
            "attribution_passed": attribution_passed,
            "utility_passed": utility_passed,
            "priority_passed": attribution_passed and utility_passed,
        },
        "claim_scope": (
            "local implementation smoke"
            if config.mode == "smoke"
            else "fifth fixed secret key PRESENT r3 selected-eight-output low-rank topology-bottleneck attribution"
        )
        + "; not r4 evidence, full-ciphertext recovery, sample classification, or SOTA",
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "sample_classification": False,
            "target": "eight_preregistered_true_ciphertext_output_bits",
            "reopens_r4": False,
        },
    }


def serializable_bottleneck_config(
    config: TopologyBottleneckConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    return payload
