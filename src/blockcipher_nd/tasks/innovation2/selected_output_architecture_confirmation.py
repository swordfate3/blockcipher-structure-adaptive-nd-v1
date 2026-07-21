from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    KimuraOutputPredictionConfig,
    prepare_disk_output_prediction_data,
)
from blockcipher_nd.tasks.innovation2.selected_output_architecture_gate import (
    NON_MLP_ARCHITECTURES,
    REMOTE_RUN_ID as PHASE_A_REMOTE_RUN_ID,
    _build_model,
    _train_one_model,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    validate_selected_output_contract,
)


RUN_ID_PREFIX = "i2_output_prediction_opa2_present_r3_selected8"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ArchitectureConfirmationConfig:
    candidate_architecture: str
    run_id: str
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 3
    train_rows: int = 64
    test_rows: int = 64
    mlp_hidden_dim: int = 1936
    lstm_hidden_dim: int = 300
    lstm_layers: int = 6
    rescnn_channels: int = 252
    rescnn_blocks: int = 10
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 7
    transformer_ff_dim: int = 576
    present_spn_dim: int = 189
    present_spn_blocks: int = 3
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    data_chunk_rows: int = 32
    selected_msb_indices: tuple[int, ...] = SELECTED_MSB_INDICES
    minimum_auc: float = 0.510
    minimum_accuracy_margin: float = 0.005
    minimum_shuffle_auc_margin: float = 0.005
    minimum_viable_bits: int = 4
    minimum_candidate_mean_auc_gain: float = 0.003
    minimum_candidate_adjusted_gain: float = 0.003
    minimum_candidate_bit_gain: float = 0.002
    minimum_candidate_gain_bits: int = 4
    maximum_parameter_gap: float = 0.03
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.candidate_architecture not in NON_MLP_ARCHITECTURES:
            raise ValueError("OPA2 candidate must be an OPA1 non-MLP architecture")
        if self.mode not in {"smoke", "phase_b_confirmation"}:
            raise ValueError("invalid OPA2 mode")
        if self.rounds != 3 or self.seed != 3:
            raise ValueError("OPA2 is frozen to PRESENT round three and key seed3")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPA2 positions must match OP10, OP11, and OPA1")
        integer_fields = (
            self.train_rows,
            self.test_rows,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
        )
        if min(integer_fields) <= 0 or self.learning_rate <= 0:
            raise ValueError("row, epoch, batch, chunk, and learning-rate values must be positive")

    @classmethod
    def smoke(cls, candidate_architecture: str) -> ArchitectureConfirmationConfig:
        return cls(
            candidate_architecture=candidate_architecture,
            run_id=f"{RUN_ID_PREFIX}_{candidate_architecture}_smoke_20260721",
        )

    @classmethod
    def phase_b_confirmation(
        cls,
        candidate_architecture: str,
        *,
        run_id: str | None = None,
        device: str = "cuda",
    ) -> ArchitectureConfirmationConfig:
        return cls(
            candidate_architecture=candidate_architecture,
            run_id=run_id
            or f"{RUN_ID_PREFIX}_{candidate_architecture}_key3_gpu0_20260721",
            mode="phase_b_confirmation",
            train_rows=1 << 17,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )


def candidate_from_phase_a_gate(gate: dict[str, Any]) -> str:
    if gate.get("run_id") != PHASE_A_REMOTE_RUN_ID:
        raise ValueError("OPA2 requires the frozen OPA1 formal run gate")
    if gate.get("status") != "pass" or gate.get("decision") != (
        "innovation2_selected8_architecture_candidate_requires_confirmation"
    ):
        raise ValueError("OPA1 did not authorize Phase B confirmation")
    for check_group in ("protocol_checks", "execution_checks"):
        checks = gate.get(check_group)
        if not isinstance(checks, dict) or not checks or not all(checks.values()):
            raise ValueError(f"OPA1 {check_group} did not fully pass")
    metrics = gate.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("OPA1 gate is missing architecture metrics")
    candidate = metrics.get("selected_candidate_for_phase_b")
    if candidate not in NON_MLP_ARCHITECTURES:
        raise ValueError("OPA1 gate does not contain a valid non-MLP candidate")
    candidate_gates = metrics.get("candidate_gates")
    if not isinstance(candidate_gates, dict):
        raise ValueError("OPA1 gate is missing candidate gates")
    candidate_gate = candidate_gates.get(candidate)
    if not isinstance(candidate_gate, dict) or candidate_gate.get("passed") is not True:
        raise ValueError("OPA1 selected candidate did not pass its preregistered gate")
    return str(candidate)


def confirmation_model_specs(
    config: ArchitectureConfirmationConfig,
) -> tuple[tuple[str, str, bool], ...]:
    candidate = config.candidate_architecture
    return (
        ("selected8_mlp_true_output", "mlp", False),
        ("selected8_mlp_label_shuffle", "mlp", True),
        (f"selected8_{candidate}_true_output", candidate, False),
        (f"selected8_{candidate}_label_shuffle", candidate, True),
    )


def prepare_confirmation_data(
    config: ArchitectureConfirmationConfig,
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


def validate_confirmation_contract(
    config: ArchitectureConfirmationConfig,
    data: dict[str, Any],
    phase_a_gate: dict[str, Any],
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
    expected_candidate = candidate_from_phase_a_gate(phase_a_gate)
    keys = {
        seed: random.Random(910_000 + seed).getrandbits(80)
        for seed in (0, 1, 2, 3)
    }
    counts = confirmation_parameter_counts(config)
    specs = confirmation_model_specs(config)
    checks.update(
        {
            "fourth_fixed_key_seed_is_three": config.seed == 3,
            "fourth_key_differs_from_first_three": len(set(keys.values())) == 4
            and int(data["secret_key"]) == keys[3],
            "candidate_is_exactly_phase_a_selection": config.candidate_architecture
            == expected_candidate,
            "four_row_true_and_matched_shuffle_matrix": len(specs) == 4
            and sum(shuffled for _, _, shuffled in specs) == 2,
            "candidate_parameter_gap_within_three_percent": abs(
                counts[config.candidate_architecture] - counts["mlp"]
            )
            / counts["mlp"]
            <= config.maximum_parameter_gap,
            "labels_are_true_outputs_not_sample_classes": True,
        }
    )
    return checks


def train_confirmation_matrix(
    config: ArchitectureConfirmationConfig,
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
    permutation = np.random.default_rng(1_410_000 + config.seed).permutation(
        config.train_rows
    )
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for model_name, architecture, shuffle_labels in confirmation_model_specs(config):
        model_targets = train_targets[permutation] if shuffle_labels else train_targets
        result = _train_one_model(
            config,  # type: ignore[arg-type]
            model_name=model_name,
            architecture=architecture,
            train_features=train_features,
            train_targets=np.array(model_targets, copy=True),
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


def adjudicate_confirmation(
    config: ArchitectureConfirmationConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    candidate = config.candidate_architecture
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in training["rows"]
    }
    bit_gates: list[dict[str, Any]] = []
    architecture_metrics: dict[str, Any] = {}
    for architecture in ("mlp", candidate):
        true_name = f"selected8_{architecture}_true_output"
        shuffle_name = f"selected8_{architecture}_label_shuffle"
        for bit in config.selected_msb_indices:
            true_row = indexed[(true_name, bit)]
            shuffle_row = indexed[(shuffle_name, bit)]
            shuffle_margin = float(true_row["auc"]) - float(shuffle_row["auc"])
            checks = {
                "auc_at_least_0_510": float(true_row["auc"]) >= config.minimum_auc,
                "accuracy_margin_at_least_0_005": float(
                    true_row["accuracy_minus_majority"]
                )
                >= config.minimum_accuracy_margin,
                "auc_minus_matched_shuffle_at_least_0_005": shuffle_margin
                >= config.minimum_shuffle_auc_margin,
            }
            bit_gates.append(
                {
                    "architecture": architecture,
                    "msb_index": bit,
                    "true_auc": float(true_row["auc"]),
                    "shuffle_auc": float(shuffle_row["auc"]),
                    "auc_minus_shuffle": shuffle_margin,
                    "accuracy_minus_majority": float(
                        true_row["accuracy_minus_majority"]
                    ),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        true_rows = [indexed[(true_name, bit)] for bit in config.selected_msb_indices]
        shuffle_rows = [
            indexed[(shuffle_name, bit)] for bit in config.selected_msb_indices
        ]
        passed = [
            row
            for row in bit_gates
            if row["architecture"] == architecture and row["passed"]
        ]
        mean_true = float(np.mean([row["auc"] for row in true_rows]))
        mean_shuffle = float(np.mean([row["auc"] for row in shuffle_rows]))
        architecture_metrics[architecture] = {
            "passed_bits": len(passed),
            "passed_msb_indices": [row["msb_index"] for row in passed],
            "viability_passed": len(passed) >= config.minimum_viable_bits,
            "mean_true_auc": mean_true,
            "mean_shuffle_auc": mean_shuffle,
            "mean_true_minus_shuffle_auc": mean_true - mean_shuffle,
            "mean_accuracy_minus_majority": float(
                np.mean([row["accuracy_minus_majority"] for row in true_rows])
            ),
        }

    per_bit_gain = {
        str(bit): float(indexed[(f"selected8_{candidate}_true_output", bit)]["auc"])
        - float(indexed[("selected8_mlp_true_output", bit)]["auc"])
        for bit in config.selected_msb_indices
    }
    mean_gain = (
        architecture_metrics[candidate]["mean_true_auc"]
        - architecture_metrics["mlp"]["mean_true_auc"]
    )
    adjusted_gain = (
        architecture_metrics[candidate]["mean_true_minus_shuffle_auc"]
        - architecture_metrics["mlp"]["mean_true_minus_shuffle_auc"]
    )
    gain_bits = sum(
        value >= config.minimum_candidate_bit_gain for value in per_bit_gain.values()
    )
    priority_checks = {
        "candidate_viability_passed": architecture_metrics[candidate][
            "viability_passed"
        ],
        "mean_candidate_minus_mlp_at_least_0_003": mean_gain
        >= config.minimum_candidate_mean_auc_gain,
        "adjusted_candidate_gain_at_least_0_003": adjusted_gain
        >= config.minimum_candidate_adjusted_gain,
        "at_least_four_bits_gain_0_002": gain_bits
        >= config.minimum_candidate_gain_bits,
    }
    priority_passed = all(priority_checks.values())
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
    }
    protocol_valid = bool(protocol_checks) and all(protocol_checks.values())
    execution_valid = all(execution_checks.values())
    if not protocol_valid or not execution_valid:
        status = "fail"
        decision = "innovation2_selected8_architecture_confirmation_protocol_invalid"
        action = "repair only the frozen seed3 data, gate ownership, controls, or artifacts"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_selected8_architecture_confirmation_local_smoke_passed"
        action = "launch the exact gate-selected candidate seed3 four-row confirmation"
    elif priority_passed:
        status = "pass"
        decision = "innovation2_selected8_architecture_priority_independently_confirmed"
        action = "retain the confirmed architecture for the Innovation 2 thesis claim"
    else:
        status = "hold"
        decision = "innovation2_selected8_mlp_retained_after_architecture_confirmation"
        action = "retain MLP and stop architecture, seed, data, epoch, and round expansion"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "candidate_architecture": candidate,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "bit_gates": bit_gates,
        "metrics": {
            "architectures": architecture_metrics,
            "mean_candidate_minus_mlp_auc": mean_gain,
            "adjusted_candidate_gain_auc": adjusted_gain,
            "candidate_gain_bit_count": gain_bits,
            "per_bit_candidate_minus_mlp_auc": per_bit_gain,
            "priority_checks": priority_checks,
            "priority_passed": priority_passed,
        },
        "claim_scope": (
            "local implementation smoke"
            if config.mode == "smoke"
            else "fourth fixed secret key PRESENT r3 selected-eight-output matched-control architecture confirmation"
        )
        + "; not output reselection, r4 evidence, full-ciphertext recovery, sample classification, or SOTA",
        "next_action": {
            "action": action,
            "sample_classification": False,
            "target": "eight_preregistered_true_ciphertext_output_bits",
            "reopens_op12": False,
        },
    }


def confirmation_parameter_counts(
    config: ArchitectureConfirmationConfig,
) -> dict[str, int]:
    return {
        architecture: sum(
            parameter.numel()
            for parameter in _build_model(config, architecture).parameters()  # type: ignore[arg-type]
        )
        for architecture in ("mlp", config.candidate_architecture)
    }


def serializable_confirmation_config(
    config: ArchitectureConfirmationConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    return payload
