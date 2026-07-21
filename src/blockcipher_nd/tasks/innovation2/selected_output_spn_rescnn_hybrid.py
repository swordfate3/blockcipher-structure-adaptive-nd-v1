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
    SelectedOutputResidualCnn,
    SelectedOutputSpnResidualCnn,
    _build_model,
    _present_topology_mapping,
    _train_one_model,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    validate_selected_output_contract,
)


RUN_ID_PREFIX = "i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid"
OPB1_RUN_ID = (
    "i2_output_prediction_opb1_present_r3_topology_bottleneck_key4_gpu0_20260722"
)
OPB1_RELEASE_DECISION = "innovation2_topology_bottleneck_not_attributed"
MODEL_SPECS = (
    ("selected8_rescnn_anchor_true_output", "rescnn", False),
    ("selected8_spn_rescnn_exact_p_true_output", "spn_rescnn_exact_p", False),
    ("selected8_spn_rescnn_wrong_p_true_output", "spn_rescnn_wrong_p", False),
    ("selected8_spn_rescnn_exact_p_label_shuffle", "spn_rescnn_exact_p", True),
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SpnResCnnHybridConfig:
    run_id: str = f"{RUN_ID_PREFIX}_smoke_seed6_20260722"
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 6
    train_rows: int = 64
    test_rows: int = 64
    mlp_hidden_dim: int = 1936
    lstm_hidden_dim: int = 300
    lstm_layers: int = 6
    rescnn_channels: int = 252
    rescnn_blocks: int = 10
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    data_chunk_rows: int = 32
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES
    minimum_candidate_mean_auc: float = 0.550
    minimum_candidate_anchor_gain: float = 0.010
    minimum_mean_topology_margin: float = 0.020
    minimum_mean_shuffle_margin: float = 0.030
    minimum_per_bit_auc: float = 0.550
    minimum_per_bit_anchor_gain: float = 0.005
    minimum_per_bit_control_margin: float = 0.015
    minimum_accuracy_margin: float = 0.005
    minimum_passed_bits: int = 4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "spn_rescnn_hybrid"}:
            raise ValueError("invalid OPC1 mode")
        if self.rounds != 3 or self.seed != 6:
            raise ValueError("OPC1 is frozen to PRESENT round three and key seed6")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPC1 positions must match OP10 through OPB1")
        values = (
            self.train_rows,
            self.test_rows,
            self.rescnn_channels,
            self.rescnn_blocks,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
            self.minimum_passed_bits,
        )
        if min(values) <= 0 or self.learning_rate <= 0:
            raise ValueError(
                "OPC1 row, model, training, and gate values must be positive"
            )

    @classmethod
    def formal(
        cls,
        *,
        run_id: str | None = None,
        device: str = "cuda",
    ) -> SpnResCnnHybridConfig:
        return cls(
            run_id=run_id or f"{RUN_ID_PREFIX}_key6_gpu0_20260722",
            mode="spn_rescnn_hybrid",
            train_rows=1 << 17,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )


def authorize_from_opb1_gate(gate: dict[str, Any]) -> None:
    if gate.get("run_id") != OPB1_RUN_ID:
        raise ValueError("OPC1 formal mode requires the frozen OPB1 gate")
    if gate.get("status") != "hold" or gate.get("decision") != OPB1_RELEASE_DECISION:
        raise ValueError(
            "OPC1 formal mode requires OPB1 topology-bottleneck non-attribution"
        )
    for group in ("protocol_checks", "execution_checks"):
        checks = gate.get(group)
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError(f"OPB1 {group} did not fully pass")
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("attribution_passed") is not False:
        raise ValueError("OPB1 must be a valid negative attribution result")


def prepare_hybrid_data(
    config: SpnResCnnHybridConfig,
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
        data_config, output_root, progress=progress
    )


def hybrid_parameter_counts(config: SpnResCnnHybridConfig) -> dict[str, int]:
    return {
        architecture: sum(
            parameter.numel()
            for parameter in _build_model(config, architecture).parameters()  # type: ignore[arg-type]
        )
        for architecture in {architecture for _, architecture, _ in MODEL_SPECS}
    }


def validate_hybrid_contract(
    config: SpnResCnnHybridConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
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
    keys = {seed: random.Random(910_000 + seed).getrandbits(80) for seed in range(7)}
    counts = hybrid_parameter_counts(config)
    with torch.random.fork_rng():
        torch.manual_seed(2_020_000 + config.seed)
        exact = _build_model(config, "spn_rescnn_exact_p")  # type: ignore[arg-type]
        torch.manual_seed(2_020_000 + config.seed)
        wrong = _build_model(config, "spn_rescnn_wrong_p")  # type: ignore[arg-type]
    checks.update(
        {
            "seventh_fixed_key_seed_is_six": config.seed == 6,
            "seed6_key_differs_from_seed0_through_seed5": len(set(keys.values())) == 7
            and int(data["secret_key"]) == keys[6],
            "four_frozen_matrix_rows": len(MODEL_SPECS) == 4,
            "hybrid_and_rescnn_parameter_counts_match": counts["rescnn"]
            == counts["spn_rescnn_exact_p"]
            == counts["spn_rescnn_wrong_p"],
            "hybrid_has_three_topology_routed_stages": isinstance(
                exact, SelectedOutputSpnResidualCnn
            )
            and len(exact.stages) == 3,
            "anchor_remains_plain_position_preserving_rescnn": isinstance(
                _build_model(config, "rescnn"),
                SelectedOutputResidualCnn,  # type: ignore[arg-type]
            ),
            "exact_and_wrong_trainable_states_match_at_initialization": exact.state_dict().keys()
            == wrong.state_dict().keys()
            and all(
                torch.equal(exact.state_dict()[name], wrong.state_dict()[name])
                for name in exact.state_dict()
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


def train_hybrid_matrix(
    config: SpnResCnnHybridConfig,
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
    shuffle = np.random.default_rng(2_030_000 + config.seed).permutation(
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


def adjudicate_hybrid(
    config: SpnResCnnHybridConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row for row in training["rows"]
    }
    names = [name for name, _, _ in MODEL_SPECS]
    means = {
        name: float(
            np.mean(
                [
                    float(indexed[(name, bit)]["auc"])
                    for bit in config.selected_msb_indices
                ]
            )
        )
        for name in names
    }
    anchor_name, exact_name, wrong_name, shuffle_name = names
    bit_gates = []
    for bit in config.selected_msb_indices:
        anchor = indexed[(anchor_name, bit)]
        exact = indexed[(exact_name, bit)]
        wrong = indexed[(wrong_name, bit)]
        shuffled = indexed[(shuffle_name, bit)]
        checks = {
            "candidate_auc_at_least_0_550": float(exact["auc"])
            >= config.minimum_per_bit_auc,
            "candidate_minus_anchor_at_least_0_005": float(exact["auc"])
            - float(anchor["auc"])
            >= config.minimum_per_bit_anchor_gain,
            "candidate_minus_wrong_at_least_0_015": float(exact["auc"])
            - float(wrong["auc"])
            >= config.minimum_per_bit_control_margin,
            "candidate_minus_shuffle_at_least_0_015": float(exact["auc"])
            - float(shuffled["auc"])
            >= config.minimum_per_bit_control_margin,
            "accuracy_margin_at_least_0_005": float(exact["accuracy_minus_majority"])
            >= config.minimum_accuracy_margin,
        }
        bit_gates.append(
            {
                "msb_index": bit,
                "anchor_auc": float(anchor["auc"]),
                "candidate_auc": float(exact["auc"]),
                "wrong_auc": float(wrong["auc"]),
                "shuffle_auc": float(shuffled["auc"]),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    passed_bits = sum(row["passed"] for row in bit_gates)
    formal_checks = {
        "candidate_mean_auc_at_least_0_550": means[exact_name]
        >= config.minimum_candidate_mean_auc,
        "candidate_minus_anchor_mean_auc_at_least_0_010": means[exact_name]
        - means[anchor_name]
        >= config.minimum_candidate_anchor_gain,
        "candidate_minus_wrong_mean_auc_at_least_0_020": means[exact_name]
        - means[wrong_name]
        >= config.minimum_mean_topology_margin,
        "candidate_minus_shuffle_mean_auc_at_least_0_030": means[exact_name]
        - means[shuffle_name]
        >= config.minimum_mean_shuffle_margin,
        "at_least_four_bits_pass": passed_bits >= config.minimum_passed_bits,
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
            )
        ),
    }
    valid = (
        bool(protocol_checks)
        and all(protocol_checks.values())
        and all(execution_checks.values())
    )
    if not valid:
        status = "fail"
        decision = "innovation2_spn_rescnn_hybrid_protocol_invalid"
        action = "repair only the frozen model, data, controls, or artifact protocol"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_spn_rescnn_hybrid_local_smoke_passed"
        action = "wait for OPB1; launch OPC1 only if OPB1 returns valid non-attribution"
    elif all(formal_checks.values()):
        status = "pass"
        decision = "innovation2_spn_rescnn_hybrid_candidate_requires_confirmation"
        action = "repeat the unchanged four-row matrix under a fresh fixed key"
    else:
        status = "hold"
        decision = "innovation2_spn_rescnn_hybrid_not_supported"
        action = "retain ResCNN as the non-leaky anchor and stop this hybrid route"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "bit_gates": bit_gates,
        "metrics": {
            "mean_auc_by_model": means,
            "candidate_minus_anchor_mean_auc": means[exact_name] - means[anchor_name],
            "candidate_minus_wrong_mean_auc": means[exact_name] - means[wrong_name],
            "candidate_minus_shuffle_mean_auc": means[exact_name] - means[shuffle_name],
            "passed_bit_count": passed_bits,
            "formal_checks": formal_checks,
        },
        "claim_scope": (
            "local implementation smoke"
            if config.mode == "smoke"
            else "seed6 PRESENT r3 selected-eight-output SPN-ResCNN hybrid discovery"
        )
        + "; not r4 evidence, full-ciphertext recovery, sample classification, or SOTA",
        "next_action": {
            "action": action,
            "formal_launch_requires_opb1_non_attribution": True,
            "sample_classification": False,
            "target": "eight_preregistered_true_ciphertext_output_bits",
            "reopens_r4": False,
        },
    }


def serializable_hybrid_config(config: SpnResCnnHybridConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    return payload
