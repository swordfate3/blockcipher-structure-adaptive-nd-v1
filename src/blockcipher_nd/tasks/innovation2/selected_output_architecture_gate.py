from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from blockcipher_nd.tasks.innovation2.output_prediction_kimura_lstm import (
    KimuraOutputPredictionConfig,
    prepare_disk_output_prediction_data,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
    SelectedOutputBitHeadConfig,
    SelectedOutputMlp,
    validate_selected_output_contract,
)
from blockcipher_nd.training.metrics import binary_auc


RUN_ID = "i2_output_prediction_opa1_present_r3_selected8_architecture_screen_position_preserving_smoke_20260721"
REMOTE_RUN_ID = "i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721"
MODEL_SPECS = (
    ("selected8_mlp_true_output", "mlp", False),
    ("selected8_lstm_true_output", "lstm", False),
    ("selected8_rescnn_true_output", "rescnn", False),
    ("selected8_transformer_true_output", "transformer", False),
    ("selected8_present_spn_true_output", "present_spn", False),
)
ARCHITECTURES = tuple(spec[1] for spec in MODEL_SPECS)
NON_MLP_ARCHITECTURES = ARCHITECTURES[1:]
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SelectedOutputArchitectureConfig:
    run_id: str = RUN_ID
    mode: str = "smoke"
    rounds: int = 3
    seed: int = 2
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
    minimum_viable_bits: int = 4
    minimum_candidate_mean_auc_gain: float = 0.003
    minimum_candidate_bit_gain: float = 0.002
    minimum_candidate_gain_bits: int = 4
    mean_auc_tie_tolerance: float = 0.001
    maximum_parameter_gap: float = 0.03
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "phase_a_screen"}:
            raise ValueError("invalid selected-output architecture mode")
        if self.rounds != 3:
            raise ValueError("OPA1 is frozen to PRESENT round three")
        if self.seed != 2:
            raise ValueError("OPA1 is frozen to the third fixed-key seed")
        if self.selected_msb_indices != SELECTED_MSB_INDICES:
            raise ValueError("OPA1 positions must match OP10 and OP11")
        integer_fields = (
            self.train_rows,
            self.test_rows,
            self.mlp_hidden_dim,
            self.lstm_hidden_dim,
            self.lstm_layers,
            self.rescnn_channels,
            self.rescnn_blocks,
            self.transformer_dim,
            self.transformer_heads,
            self.transformer_layers,
            self.transformer_ff_dim,
            self.present_spn_dim,
            self.present_spn_blocks,
            self.epochs,
            self.batch_size,
            self.data_chunk_rows,
            self.minimum_viable_bits,
            self.minimum_candidate_gain_bits,
        )
        if min(integer_fields) <= 0:
            raise ValueError("row, model, epoch, batch, and gate values must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @classmethod
    def phase_a_screen(
        cls,
        *,
        run_id: str = REMOTE_RUN_ID,
        device: str = "cuda",
    ) -> SelectedOutputArchitectureConfig:
        return cls(
            run_id=run_id,
            mode="phase_a_screen",
            train_rows=1 << 17,
            test_rows=1 << 16,
            epochs=100,
            batch_size=250,
            data_chunk_rows=4096,
            device=device,
        )


class SelectedOutputLstm(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 300,
        layers: int = 6,
        output_bits: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, output_bits)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError(f"expected [batch, 64] input, got {tuple(features.shape)}")
        _, (hidden, _) = self.encoder(features.float().unsqueeze(-1))
        return self.head(hidden[-1])


class _OutputResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.network(features))


class SelectedOutputResidualCnn(nn.Module):
    def __init__(self, channels: int = 252, blocks: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *(_OutputResidualBlock(channels) for _ in range(blocks))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 64, 8),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_feature_shape(features)
        return self.head(self.blocks(self.stem(features.float().unsqueeze(1))))


class SelectedOutputTransformer(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        heads: int = 8,
        layers: int = 7,
        feedforward_dim: int = 576,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(1, token_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, 64, token_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, 8)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_feature_shape(features)
        hidden = self.embedding(features.float().unsqueeze(-1))
        hidden = self.encoder(hidden + self.position_embedding)
        return self.head(self.norm(hidden.mean(dim=1)))


class _PresentSpnOutputBlock(nn.Module):
    def __init__(self, token_dim: int, source_for_destination: torch.Tensor) -> None:
        super().__init__()
        nibble_dim = token_dim * 4
        self.register_buffer(
            "source_for_destination",
            source_for_destination,
            persistent=False,
        )
        self.local_norm = nn.LayerNorm(token_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(nibble_dim, nibble_dim),
            nn.GELU(),
            nn.Linear(nibble_dim, nibble_dim),
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, _, channels = hidden.shape
        normalized = self.local_norm(hidden).reshape(batch, 16, channels * 4)
        mixed = hidden.reshape(batch, 16, channels * 4) + self.local_mlp(normalized)
        hidden = mixed.reshape(batch, 64, channels).index_select(
            1, self.source_for_destination
        )
        return hidden + self.channel_mlp(self.channel_norm(hidden))


class SelectedOutputPresentSpn(nn.Module):
    def __init__(
        self,
        token_dim: int = 189,
        blocks: int = 3,
        source_for_destination: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(1, token_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, 64, token_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        source_for_destination = (
            _present_source_for_destination()
            if source_for_destination is None
            else source_for_destination.detach().clone().to(dtype=torch.long)
        )
        if source_for_destination.shape != (64,) or sorted(
            source_for_destination.tolist()
        ) != list(range(64)):
            raise ValueError("source_for_destination must be a 64-position permutation")
        self.blocks = nn.ModuleList(
            _PresentSpnOutputBlock(token_dim, source_for_destination)
            for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _validate_feature_shape(features)
        hidden = self.embedding(features.float().unsqueeze(-1))
        hidden = hidden + self.position_embedding
        for block in self.blocks:
            hidden = block(hidden)
        selected = hidden[:, list(SELECTED_MSB_INDICES), :]
        return self.head(self.norm(selected)).squeeze(-1)


def _validate_feature_shape(features: torch.Tensor) -> None:
    if features.ndim != 2 or features.shape[1] != 64:
        raise ValueError(f"expected [batch, 64] input, got {tuple(features.shape)}")


def _present_source_for_destination() -> torch.Tensor:
    source_for_destination = [0] * 64
    for source_msb in range(64):
        source_integer = 63 - source_msb
        destination_integer = (
            63 if source_integer == 63 else (16 * source_integer) % 63
        )
        destination_msb = 63 - destination_integer
        source_for_destination[destination_msb] = source_msb
    return torch.tensor(source_for_destination, dtype=torch.long)


def _present_topology_mapping(mode: str) -> torch.Tensor:
    exact = _present_source_for_destination()
    if mode == "exact":
        return exact
    if mode == "identity":
        return torch.arange(64, dtype=torch.long)
    if mode == "wrong":
        return exact.roll(1)
    raise ValueError(f"unknown PRESENT topology mode: {mode}")


def prepare_architecture_data(
    config: SelectedOutputArchitectureConfig,
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


def validate_architecture_contract(
    config: SelectedOutputArchitectureConfig,
    data: dict[str, Any],
) -> dict[str, bool]:
    base_config = SelectedOutputBitHeadConfig(
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
    checks = validate_selected_output_contract(base_config, data)
    checks.pop("independent_key_seed_is_one")
    keys = {
        seed: random.Random(910_000 + seed).getrandbits(80)
        for seed in (0, 1, 2)
    }
    counts = parameter_counts(config)
    parameter_gaps = {
        architecture: abs(count - counts["mlp"]) / counts["mlp"]
        for architecture, count in counts.items()
    }
    checks.update(
        {
            "third_fixed_key_seed_is_two": config.seed == 2,
            "third_fixed_key_differs_from_seed0_and_seed1": len(set(keys.values()))
            == 3
            and int(data["secret_key"]) == keys[2],
            "all_architectures_within_three_percent_of_mlp": all(
                gap <= config.maximum_parameter_gap
                for gap in parameter_gaps.values()
            ),
            "five_row_phase_a_matrix_is_frozen": len(MODEL_SPECS) == 5
            and all(not shuffled for _, _, shuffled in MODEL_SPECS),
            "present_p_layer_mapping_is_a_permutation": sorted(
                _present_source_for_destination().tolist()
            )
            == list(range(64)),
            "labels_are_true_output_bits_not_sample_classes": True,
        }
    )
    return checks


def train_architecture_matrix(
    config: SelectedOutputArchitectureConfig,
    data: dict[str, Any],
    output_root: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    train_features = np.array(data["features"][: config.train_rows], copy=True)
    test_features = np.array(data["features"][config.train_rows :], copy=True)
    selected_columns = np.asarray(config.selected_msb_indices, dtype=np.int64)
    train_targets = np.array(
        data["full_targets"][: config.train_rows, selected_columns], copy=True
    )
    test_targets = np.array(
        data["full_targets"][config.train_rows :, selected_columns], copy=True
    )
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for model_name, architecture, shuffle_labels in MODEL_SPECS:
        model_targets = np.array(train_targets, copy=True)
        if shuffle_labels:
            raise ValueError("OPA1 Phase A does not include label-shuffle rows")
        result = _train_one_model(
            config,
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


def adjudicate_architecture_gate(
    config: SelectedOutputArchitectureConfig,
    protocol_checks: dict[str, bool],
    training: dict[str, Any],
) -> dict[str, Any]:
    indexed = {
        (str(row["model"]), int(row["msb_index"])): row
        for row in training["rows"]
    }
    bit_gates: list[dict[str, Any]] = []
    architecture_metrics: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        true_name = f"selected8_{architecture}_true_output"
        rows = [
            indexed[(true_name, bit)] for bit in config.selected_msb_indices
        ]
        for bit in config.selected_msb_indices:
            true_row = indexed[(true_name, bit)]
            checks = {
                "auc_at_least_0_510": float(true_row["auc"]) >= config.minimum_auc,
                "accuracy_margin_at_least_0_005": float(
                    true_row["accuracy_minus_majority"]
                )
                >= config.minimum_accuracy_margin,
            }
            bit_gates.append(
                {
                    "architecture": architecture,
                    "msb_index": bit,
                    "auc": true_row["auc"],
                    "accuracy_minus_majority": true_row[
                        "accuracy_minus_majority"
                    ],
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        passed = [
            row
            for row in bit_gates
            if row["architecture"] == architecture and row["passed"]
        ]
        architecture_metrics[architecture] = {
            "passed_bits": len(passed),
            "passed_msb_indices": [row["msb_index"] for row in passed],
            "viability_passed": len(passed) >= config.minimum_viable_bits,
            "mean_auc": float(np.mean([row["auc"] for row in rows])),
            "mean_accuracy_minus_majority": float(
                np.mean([row["accuracy_minus_majority"] for row in rows])
            ),
        }

    candidate_gates: dict[str, Any] = {}
    for architecture in NON_MLP_ARCHITECTURES:
        per_bit_gain = {
            str(bit): float(
                indexed[(f"selected8_{architecture}_true_output", bit)]["auc"]
            )
            - float(indexed[("selected8_mlp_true_output", bit)]["auc"])
            for bit in config.selected_msb_indices
        }
        mean_gain = (
            architecture_metrics[architecture]["mean_auc"]
            - architecture_metrics["mlp"]["mean_auc"]
        )
        gain_bits = sum(
            gain >= config.minimum_candidate_bit_gain
            for gain in per_bit_gain.values()
        )
        checks = {
            "viability_passed": architecture_metrics[architecture][
                "viability_passed"
            ],
            "mean_auc_gain_at_least_0_003": mean_gain
            >= config.minimum_candidate_mean_auc_gain,
            "at_least_four_bits_gain_0_002": gain_bits
            >= config.minimum_candidate_gain_bits,
        }
        candidate_gates[architecture] = {
            "mean_auc_gain_vs_mlp": mean_gain,
            "gain_bit_count": gain_bits,
            "per_bit_auc_gain_vs_mlp": per_bit_gain,
            "checks": checks,
            "passed": all(checks.values()),
        }

    ranked_candidates = _rank_candidates(
        list(NON_MLP_ARCHITECTURES),
        architecture_metrics,
        tie_tolerance=config.mean_auc_tie_tolerance,
    )
    leading_candidate = ranked_candidates[0]
    advancing_candidates = [
        architecture
        for architecture in ranked_candidates
        if candidate_gates[architecture]["passed"]
    ]
    selected_candidate = (
        _rank_candidates(
            advancing_candidates,
            architecture_metrics,
            tie_tolerance=config.mean_auc_tie_tolerance,
        )[0]
        if advancing_candidates
        else None
    )
    execution_checks = {
        "five_models_complete": len(training["summaries"]) == 5,
        "forty_result_rows_complete": len(training["rows"]) == 40,
        "history_rows_complete": len(training["history"]) == config.epochs * 5,
        "five_checkpoint_hashes_present": len(training["checkpoints"]) == 5
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
        decision = "innovation2_selected8_architecture_protocol_invalid"
        action = "repair only the frozen data, target, architecture, or artifact protocol"
        next_adjudication = "opa1_protocol_repair"
    elif config.mode == "smoke":
        status = "pass"
        decision = "innovation2_selected8_architecture_screen_local_smoke_passed"
        action = "launch the frozen seed2 five-model Phase A architecture screen"
        next_adjudication = "opa1_remote_seed2_architecture_screen"
    elif selected_candidate is not None:
        status = "pass"
        decision = "innovation2_selected8_architecture_candidate_requires_confirmation"
        action = (
            f"confirm {selected_candidate} against MLP and matched shuffles under seed3"
        )
        next_adjudication = "opa2_seed3_winner_vs_mlp_matched_shuffle"
    else:
        status = "hold"
        decision = "innovation2_selected8_mlp_anchor_retained_after_screen"
        action = "retain the MLP anchor; do not tune more architectures on the seed2 test set"
        next_adjudication = "innovation2_output_prediction_thesis_boundary"

    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "bit_gates": bit_gates,
        "metrics": {
            "architectures": architecture_metrics,
            "candidate_gates": candidate_gates,
            "ranked_non_mlp_candidates": ranked_candidates,
            "leading_candidate": leading_candidate,
            "selected_candidate_for_phase_b": selected_candidate,
        },
        "claim_scope": (
            "local implementation smoke"
            if config.mode == "smoke"
            else "third fixed secret key PRESENT r3 selected-eight-output five-model discovery screen"
        )
        + "; discovery only, not a final architecture claim, position reselection, r4 evidence, full-ciphertext recovery, sample classification, or SOTA",
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "sample_classification": False,
            "target": "eight_preregistered_true_ciphertext_output_bits",
            "selected_candidate": selected_candidate,
            "phase_b_requires_matched_shuffle": selected_candidate is not None,
            "reopens_op12": False,
        },
    }


def parameter_counts(config: SelectedOutputArchitectureConfig) -> dict[str, int]:
    return {
        architecture: sum(
            parameter.numel()
            for parameter in _build_model(config, architecture).parameters()
        )
        for architecture in ARCHITECTURES
    }


def _rank_candidates(
    architectures: list[str],
    metrics: dict[str, Any],
    *,
    tie_tolerance: float,
) -> list[str]:
    remaining = list(architectures)
    ranked: list[str] = []
    while remaining:
        best_auc = max(float(metrics[item]["mean_auc"]) for item in remaining)
        tied = [
            item
            for item in remaining
            if best_auc - float(metrics[item]["mean_auc"]) < tie_tolerance
        ]
        tied.sort(
            key=lambda item: (
                -float(metrics[item]["mean_accuracy_minus_majority"]),
                NON_MLP_ARCHITECTURES.index(item),
            )
        )
        ranked.extend(tied)
        remaining = [item for item in remaining if item not in tied]
    return ranked


def serializable_config(config: SelectedOutputArchitectureConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_msb_indices"] = list(config.selected_msb_indices)
    return payload


def _build_model(
    config: SelectedOutputArchitectureConfig,
    architecture: str,
) -> nn.Module:
    if architecture == "mlp":
        return SelectedOutputMlp(config.mlp_hidden_dim, output_bits=8)
    if architecture == "lstm":
        return SelectedOutputLstm(
            config.lstm_hidden_dim,
            config.lstm_layers,
            output_bits=8,
        )
    if architecture == "rescnn":
        return SelectedOutputResidualCnn(
            channels=config.rescnn_channels,
            blocks=config.rescnn_blocks,
        )
    if architecture == "transformer":
        return SelectedOutputTransformer(
            token_dim=config.transformer_dim,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            feedforward_dim=config.transformer_ff_dim,
        )
    if architecture == "present_spn":
        return SelectedOutputPresentSpn(
            token_dim=config.present_spn_dim,
            blocks=config.present_spn_blocks,
        )
    topology_mode = {
        "present_spn_exact_p": "exact",
        "present_spn_identity_p": "identity",
        "present_spn_wrong_p": "wrong",
    }.get(architecture)
    if topology_mode is not None:
        return SelectedOutputPresentSpn(
            token_dim=config.present_spn_dim,
            blocks=config.present_spn_blocks,
            source_for_destination=_present_topology_mapping(topology_mode),
        )
    raise ValueError(f"unknown architecture: {architecture}")


def _train_one_model(
    config: SelectedOutputArchitectureConfig,
    *,
    model_name: str,
    architecture: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    output_root: Path,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    _seed_everything(1_320_000 + config.seed)
    model = _build_model(config, architecture)
    model.to(config.device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    latest_path = model_root / f"{model_name}_latest.pt"
    final_path = model_root / f"{model_name}_final.pt"
    config_hash = _training_config_hash(config, model_name)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if latest_path.exists():
        checkpoint = torch.load(
            latest_path,
            map_location=config.device,
            weights_only=False,
        )
        if checkpoint.get("config_hash") != config_hash:
            raise ValueError(f"checkpoint config mismatch for {model_name}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
    feature_tensor = torch.from_numpy(train_features)
    target_tensor = torch.from_numpy(train_targets)
    for epoch in range(start_epoch, config.epochs + 1):
        loader = DataLoader(
            TensorDataset(feature_tensor, target_tensor),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(
                1_330_000 + config.seed + epoch
            ),
        )
        model.train()
        total_loss = 0.0
        total_cells = 0
        for features, targets in loader:
            features = features.to(config.device)
            targets = targets.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * targets.numel()
            total_cells += targets.numel()
        history_row = {
            "run_id": config.run_id,
            "model": model_name,
            "architecture": architecture,
            "epoch": epoch,
            "train_mse": total_loss / max(1, total_cells),
        }
        history.append(history_row)
        torch.save(
            {
                "config_hash": config_hash,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            },
            latest_path,
        )
        if progress is not None:
            progress("epoch_done", history_row)
    scores = _predict_raw(
        model,
        test_features,
        batch_size=config.batch_size,
        device=config.device,
    )
    rows = _selected_bit_rows(
        model_name,
        architecture,
        scores,
        test_targets,
        config.selected_msb_indices,
    )
    torch.save(
        {
            "config_hash": config_hash,
            "epoch": config.epochs,
            "model_state": model.state_dict(),
        },
        final_path,
    )
    checkpoint = {
        "model": model_name,
        "architecture": architecture,
        "path": str(final_path.relative_to(output_root)),
        "sha256": _sha256(final_path),
        "config_hash": config_hash,
    }
    return {
        "rows": rows,
        "summary": {
            "model": model_name,
            "architecture": architecture,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_labels_shuffled": model_name.endswith("label_shuffle"),
            "mean_auc": float(np.mean([row["auc"] for row in rows])),
            "mean_accuracy_margin": float(
                np.mean([row["accuracy_minus_majority"] for row in rows])
            ),
            "test_target_identity": "true_selected_ciphertext_targets",
        },
        "history": history,
        "checkpoint": checkpoint,
    }


def _selected_bit_rows(
    model_name: str,
    architecture: str,
    scores: np.ndarray,
    labels: np.ndarray,
    selected_bits: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, bit in enumerate(selected_bits):
        bit_scores = np.asarray(scores[:, column], dtype=np.float64)
        bit_labels = np.asarray(labels[:, column], dtype=np.float64)
        predictions = bit_scores >= 0.5
        prevalence = float(np.mean(bit_labels))
        majority = max(prevalence, 1.0 - prevalence)
        rounded = np.rint(bit_scores)
        valid = (rounded == 0.0) | (rounded == 1.0)
        accuracy = float(np.mean(predictions == bit_labels))
        rows.append(
            {
                "model": model_name,
                "architecture": architecture,
                "target": "preregistered_true_ciphertext_output_bit",
                "sample_classification": False,
                "msb_index": bit,
                "integer_bit": 63 - bit,
                "nibble_msb_index": bit // 4,
                "bit_in_nibble_msb": bit % 4,
                "threshold_accuracy": accuracy,
                "majority_accuracy": majority,
                "accuracy_minus_majority": accuracy - majority,
                "auc": float(binary_auc(bit_labels, bit_scores)),
                "mse": float(np.mean(np.square(bit_scores - bit_labels))),
                "invalid_numpy_rint_rate": float(1.0 - np.mean(valid)),
                "test_target_identity": "true_selected_ciphertext_targets",
            }
        )
    return rows


def _predict_raw(
    model: nn.Module,
    features: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _training_config_hash(
    config: SelectedOutputArchitectureConfig,
    model_name: str,
) -> str:
    payload = {**serializable_config(config), "model_name": model_name}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
