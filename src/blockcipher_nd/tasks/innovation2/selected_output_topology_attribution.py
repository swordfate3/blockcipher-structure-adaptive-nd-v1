from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    KimuraOutputPredictionConfig,
    prepare_disk_output_prediction_data,
)
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    _build_model,
    _present_topology_mapping,
    _train_one_model,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    validate_selected_output_contract,
)


OPA2_REMOTE_RUN_ID = (
    "i2_output_prediction_opa2_present_r3_selected8_present_spn_key3_gpu0_20260722"
)
RUN_ID_PREFIX = "i2_output_prediction_opa3_present_r3_selected8_topology_attribution"
TOPOLOGY_SPECS = (
    ("present_spn_exact_p_true_output", "present_spn_exact_p"),
    ("present_spn_identity_p_true_output", "present_spn_identity_p"),
    ("present_spn_wrong_p_true_output", "present_spn_wrong_p"),
)
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class TopologyAttributionConfig:
    run_id: str = f"{RUN_ID_PREFIX}_smoke_20260722"
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 3
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
    minimum_exact_auc: float = 0.510
    minimum_mean_control_margin: float = 0.030
    minimum_per_bit_control_margin: float = 0.020
    minimum_attributed_bits: int = 4
    maximum_opa2_reproduction_delta: float = 0.005
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "topology_attribution"}:
            raise ValueError("invalid OPA3 mode")
        if self.rounds != 3 or self.seed != 3:
            raise ValueError("OPA3 is frozen to PRESENT round three and key seed3")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPA3 positions must match OP10 through OPA2")
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
            raise ValueError("OPA3 row, model, training, and gate values must be positive")

    @classmethod
    def formal(
        cls,
        *,
        run_id: str | None = None,
        device: str = "cuda",
    ) -> TopologyAttributionConfig:
        return cls(
            run_id=run_id or f"{RUN_ID_PREFIX}_key3_gpu0_20260722",
            mode="topology_attribution",
            train_rows=1 << 17,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )


def authorize_from_opa2_gate(gate: dict[str, Any]) -> float:
    if gate.get("run_id") != OPA2_REMOTE_RUN_ID:
        raise ValueError("OPA3 requires the frozen OPA2 formal gate")
    if gate.get("status") != "pass" or gate.get("decision") != (
        "innovation2_selected8_architecture_priority_independently_confirmed"
    ):
        raise ValueError("OPA2 did not authorize topology attribution")
    if gate.get("candidate_architecture") != "present_spn":
        raise ValueError("OPA2 candidate is not PRESENT-SPN-aware")
    for group in ("protocol_checks", "execution_checks"):
        checks = gate.get(group)
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError(f"OPA2 {group} did not fully pass")
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("priority_passed") is not True:
        raise ValueError("OPA2 priority gate did not pass")
    architectures = metrics.get("architectures")
    if not isinstance(architectures, dict):
        raise ValueError("OPA2 architecture metrics are missing")
    candidate = architectures.get("present_spn")
    if not isinstance(candidate, dict):
        raise ValueError("OPA2 PRESENT-SPN metrics are missing")
    return float(candidate["mean_true_auc"])


def prepare_topology_data(
    config: TopologyAttributionConfig,
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


def topology_parameter_counts(config: TopologyAttributionConfig) -> dict[str, int]:
    return {
        architecture: sum(
            parameter.numel()
            for parameter in _build_model(config, architecture).parameters()  # type: ignore[arg-type]
        )
        for _, architecture in TOPOLOGY_SPECS
    }


def validate_topology_contract(
    config: TopologyAttributionConfig,
    data: dict[str, Any],
    opa2_gate: dict[str, Any],
) -> dict[str, bool]:
    authorize_from_opa2_gate(opa2_gate)
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
    exact = _present_topology_mapping("exact")
    identity = _present_topology_mapping("identity")
    wrong = _present_topology_mapping("wrong")
    counts = topology_parameter_counts(config)
    checks.update(
        {
            "fourth_fixed_key_seed_is_three": config.seed == 3,
            "opa2_gate_authorizes_topology_attribution": True,
            "three_frozen_topology_rows": len(TOPOLOGY_SPECS) == 3,
            "all_topologies_are_permutations": all(
                sorted(mapping.tolist()) == list(range(64))
                for mapping in (exact, identity, wrong)
            ),
            "identity_mapping_is_exact_identity": identity.tolist() == list(range(64)),
            "wrong_mapping_differs_at_all_positions": bool(np.all(wrong.numpy() != exact.numpy())),
            "three_models_have_identical_parameter_counts": len(set(counts.values())) == 1,
            "labels_are_true_outputs_not_sample_classes": True,
        }
    )
    return checks


def train_topology_matrix(
    config: TopologyAttributionConfig,
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
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for model_name, architecture in TOPOLOGY_SPECS:
        result = _train_one_model(
            config,  # type: ignore[arg-type]
            model_name=model_name,
            architecture=architecture,
            train_features=train_features,
            train_targets=np.array(train_targets, copy=True),
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


def adjudicate_topology(
    config: TopologyAttributionConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
    opa2_gate: dict[str, Any],
) -> dict[str, Any]:
    opa2_exact_auc = authorize_from_opa2_gate(opa2_gate)
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in training["rows"]
    }
    model_names = [model for model, _ in TOPOLOGY_SPECS]
    means = {
        model: float(
            np.mean(
                [float(indexed[(model, bit)]["auc"]) for bit in config.selected_msb_indices]
            )
        )
        for model in model_names
    }
    exact_name, identity_name, wrong_name = model_names
    control_best = max(means[identity_name], means[wrong_name])
    bit_gates: list[dict[str, Any]] = []
    for bit in config.selected_msb_indices:
        exact_auc = float(indexed[(exact_name, bit)]["auc"])
        identity_auc = float(indexed[(identity_name, bit)]["auc"])
        wrong_auc = float(indexed[(wrong_name, bit)]["auc"])
        checks = {
            "exact_auc_at_least_0_510": exact_auc >= config.minimum_exact_auc,
            "exact_minus_identity_at_least_0_020": exact_auc - identity_auc
            >= config.minimum_per_bit_control_margin,
            "exact_minus_wrong_at_least_0_020": exact_auc - wrong_auc
            >= config.minimum_per_bit_control_margin,
        }
        bit_gates.append(
            {
                "msb_index": bit,
                "exact_auc": exact_auc,
                "identity_auc": identity_auc,
                "wrong_auc": wrong_auc,
                "exact_minus_identity_auc": exact_auc - identity_auc,
                "exact_minus_wrong_auc": exact_auc - wrong_auc,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    attributed_bits = sum(row["passed"] for row in bit_gates)
    reproduction_delta = abs(means[exact_name] - opa2_exact_auc)
    priority_checks = {
        "exact_mean_auc_at_least_0_510": means[exact_name]
        >= config.minimum_exact_auc,
        "exact_mean_minus_best_control_at_least_0_030": means[exact_name]
        - control_best
        >= config.minimum_mean_control_margin,
        "at_least_four_bits_attributed": attributed_bits
        >= config.minimum_attributed_bits,
        "exact_reproduces_opa2_within_0_005": reproduction_delta
        <= config.maximum_opa2_reproduction_delta,
    }
    execution_checks = {
        "three_models_complete": len(training["summaries"]) == 3,
        "twenty_four_result_rows_complete": len(training["rows"]) == 24,
        "history_rows_complete": len(training["history"]) == config.epochs * 3,
        "three_checkpoint_hashes_present": len(training["checkpoints"]) == 3
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
    }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    execution_valid = all(execution_checks.values())
    if not protocol_valid or not execution_valid:
        status = "fail"
        decision = "innovation2_selected8_topology_attribution_protocol_invalid"
        action = "repair only the frozen topology mappings, OPA2 authority, data, or artifacts"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_selected8_topology_attribution_local_smoke_passed"
        action = "launch the exact three-row topology attribution after binding the OPA2 gate"
    elif all(priority_checks.values()):
        status = "pass"
        decision = "innovation2_selected8_present_topology_independently_attributed"
        action = "preregister the controlled PRESENT r4 selected-output architecture gate"
    else:
        status = "hold"
        decision = "innovation2_selected8_present_topology_not_attributed"
        action = "retain only the whole-architecture result and stop topology and round expansion"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "bit_gates": bit_gates,
        "metrics": {
            "mean_auc_by_model": means,
            "opa2_exact_mean_auc": opa2_exact_auc,
            "exact_reproduction_delta": reproduction_delta,
            "exact_minus_best_control_mean_auc": means[exact_name] - control_best,
            "attributed_bit_count": attributed_bits,
            "priority_checks": priority_checks,
            "priority_passed": all(priority_checks.values()),
        },
        "claim_scope": (
            "local implementation smoke"
            if config.mode == "smoke"
            else "fourth fixed secret key PRESENT r3 selected-eight-output P-layer topology attribution"
        )
        + "; not r4 evidence, full-ciphertext recovery, sample classification, or SOTA",
        "next_action": {
            "action": action,
            "sample_classification": False,
            "target": "eight_preregistered_true_ciphertext_output_bits",
            "reopens_controlled_r4_gate": config.mode != "smoke"
            and all(priority_checks.values()),
        },
    }


def serializable_topology_config(config: TopologyAttributionConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    return payload
