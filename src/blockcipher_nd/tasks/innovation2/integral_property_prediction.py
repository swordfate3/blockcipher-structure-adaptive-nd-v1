from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.baseline.mlp import MlpDistinguisher
from blockcipher_nd.training import (
    TrainingConfig,
    evaluate_binary_classifier,
    predict_binary_probabilities,
    train_binary_classifier,
)


POSITION_BITS = 16
MASK_BITS = 15
CONTEXT_BITS = 64
INPUT_BITS = POSITION_BITS + POSITION_BITS + MASK_BITS + CONTEXT_BITS
INTEGRAL_SET_SIZE = 16

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class IntegralStructure:
    structure_id: str
    active_nibble: int
    output_nibble: int
    output_mask: int
    fixed_plaintext: int

    def __post_init__(self) -> None:
        if not 0 <= self.active_nibble < 16:
            raise ValueError("active_nibble must be in [0, 15]")
        if not 0 <= self.output_nibble < 16:
            raise ValueError("output_nibble must be in [0, 15]")
        if not 1 <= self.output_mask < 16:
            raise ValueError("output_mask must be a nonzero 4-bit mask")
        if self.fixed_plaintext >> 64:
            raise ValueError("fixed_plaintext must fit in 64 bits")
        active_mask = 0xF << (4 * self.active_nibble)
        if self.fixed_plaintext & active_mask:
            raise ValueError("fixed_plaintext must clear the active nibble")

    @property
    def signature(self) -> str:
        return (
            f"a{self.active_nibble:02d}-o{self.output_nibble:02d}-"
            f"m{self.output_mask:X}-p{self.fixed_plaintext:016X}"
        )

    def feature_vector(self) -> np.ndarray:
        features = np.zeros(INPUT_BITS, dtype=np.uint8)
        features[self.active_nibble] = 1
        output_offset = POSITION_BITS
        features[output_offset + self.output_nibble] = 1
        mask_offset = POSITION_BITS * 2
        features[mask_offset + self.output_mask - 1] = 1
        context_offset = mask_offset + MASK_BITS
        for bit_index in range(CONTEXT_BITS):
            features[context_offset + bit_index] = (
                self.fixed_plaintext >> bit_index
            ) & 1
        return features

    def plaintext(self, active_value: int) -> int:
        if not 0 <= active_value < INTEGRAL_SET_SIZE:
            raise ValueError("active_value must be in [0, 15]")
        return self.fixed_plaintext | (
            active_value << (4 * self.active_nibble)
        )


@dataclass(frozen=True)
class IntegralParitySplit:
    name: str
    dataset: DifferentialDataset
    structures: tuple[IntegralStructure, ...]
    keys: tuple[int, ...]
    structure_indices: np.ndarray
    observed_q1_rates: np.ndarray


@dataclass(frozen=True)
class IntegralExperimentConfig:
    run_id: str
    train_structures: int
    validation_structures: int
    test_structures: int
    train_keys: int
    validation_keys: int
    test_keys: int
    epochs: int
    seed: int = 0
    rounds: int = 5
    batch_size: int = 256
    hidden_bits: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    gate_mode: str = "diagnostic"

    def __post_init__(self) -> None:
        for name in (
            "train_structures",
            "validation_structures",
            "test_structures",
            "train_keys",
            "validation_keys",
            "test_keys",
            "epochs",
            "batch_size",
            "hidden_bits",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rounds != 5:
            raise ValueError("the frozen feasibility protocol requires PRESENT r5")
        if self.gate_mode not in {"smoke", "diagnostic"}:
            raise ValueError("gate_mode must be smoke or diagnostic")


class LinearParityPredictor(nn.Module):
    def __init__(self, input_bits: int) -> None:
        super().__init__()
        self.input_bits = input_bits
        self.linear = nn.Linear(input_bits, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(
                f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
            )
        return self.linear(features.float())


def integral_mask_parity(cipher: Present80, structure: IntegralStructure) -> int:
    parity = 0
    shift = 4 * structure.output_nibble
    for active_value in range(INTEGRAL_SET_SIZE):
        ciphertext = cipher.encrypt(structure.plaintext(active_value))
        output_value = (ciphertext >> shift) & 0xF
        parity ^= (output_value & structure.output_mask).bit_count() & 1
    return parity


def build_integral_split(
    *,
    name: str,
    rounds: int,
    structure_count: int,
    key_count: int,
    structure_seed: int,
    key_seed: int,
    progress_callback: ProgressCallback | None = None,
) -> IntegralParitySplit:
    structures = make_structures(
        count=structure_count,
        seed=structure_seed,
        prefix=name,
    )
    keys = make_keys(count=key_count, seed=key_seed)
    return build_integral_split_from_structures(
        name=name,
        rounds=rounds,
        structures=structures,
        keys=keys,
        progress_callback=progress_callback,
    )


def build_integral_split_from_structures(
    *,
    name: str,
    rounds: int,
    structures: tuple[IntegralStructure, ...],
    keys: tuple[int, ...],
    progress_callback: ProgressCallback | None = None,
) -> IntegralParitySplit:
    structure_count = len(structures)
    key_count = len(keys)
    if structure_count <= 0:
        raise ValueError("structures must not be empty")
    if key_count <= 0:
        raise ValueError("keys must not be empty")
    rows = structure_count * key_count
    features = np.empty((rows, INPUT_BITS), dtype=np.uint8)
    labels = np.empty(rows, dtype=np.uint8)
    structure_indices = np.empty(rows, dtype=np.int32)
    observed_rates = np.empty(structure_count, dtype=np.float32)
    _emit(
        progress_callback,
        "dataset_split_start",
        split=name,
        structures=structure_count,
        keys=key_count,
        rows=rows,
    )
    row_index = 0
    ciphers = tuple(Present80(rounds=rounds, key=key) for key in keys)
    report_interval = max(1, structure_count // 8)
    for structure_index, structure in enumerate(structures):
        feature = structure.feature_vector()
        q_sum = 0
        for cipher in ciphers:
            q_value = integral_mask_parity(cipher, structure)
            features[row_index] = feature
            labels[row_index] = q_value
            structure_indices[row_index] = structure_index
            q_sum += q_value
            row_index += 1
        observed_rates[structure_index] = q_sum / float(key_count)
        if (
            (structure_index + 1) % report_interval == 0
            or structure_index + 1 == structure_count
        ):
            _emit(
                progress_callback,
                "dataset_split_progress",
                split=name,
                structures_done=structure_index + 1,
                structures=structure_count,
            )
    dataset = DifferentialDataset(
        features=features,
        labels=labels,
        metadata={
            "task": "innovation2_integral_property_prediction",
            "split": name,
            "cipher": "PRESENT-80",
            "rounds": rounds,
            "structures": structure_count,
            "keys_per_structure": key_count,
            "rows": rows,
            "input_bits": INPUT_BITS,
            "integral_set_size": INTEGRAL_SET_SIZE,
            "key_omitted_from_features": True,
        },
    )
    _emit(
        progress_callback,
        "dataset_split_done",
        split=name,
        rows=rows,
        q1_rate=float(labels.mean()),
        all_zero_structures=int(np.sum(observed_rates == 0.0)),
    )
    return IntegralParitySplit(
        name=name,
        dataset=dataset,
        structures=structures,
        keys=keys,
        structure_indices=structure_indices,
        observed_q1_rates=observed_rates,
    )


def make_structures(
    *,
    count: int,
    seed: int,
    prefix: str,
) -> tuple[IntegralStructure, ...]:
    rng = np.random.default_rng(seed)
    structures: list[IntegralStructure] = []
    signatures: set[tuple[int, int, int, int]] = set()
    while len(structures) < count:
        active_nibble = int(rng.integers(0, 16))
        output_nibble = int(rng.integers(0, 16))
        output_mask = int(rng.integers(1, 16))
        fixed_plaintext = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        fixed_plaintext &= ~(0xF << (4 * active_nibble))
        signature = (
            active_nibble,
            output_nibble,
            output_mask,
            fixed_plaintext,
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        structures.append(
            IntegralStructure(
                structure_id=f"{prefix}-{len(structures):06d}",
                active_nibble=active_nibble,
                output_nibble=output_nibble,
                output_mask=output_mask,
                fixed_plaintext=fixed_plaintext,
            )
        )
    return tuple(structures)


def make_keys(*, count: int, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    keys: list[int] = []
    seen: set[int] = set()
    while len(keys) < count:
        high = int(rng.integers(0, 1 << 16, dtype=np.uint64))
        low = int(rng.integers(0, 1 << 64, dtype=np.uint64))
        key = (high << 64) | low
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return tuple(keys)


def run_integral_property_experiment(
    config: IntegralExperimentConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    split_specs = (
        ("train", config.train_structures, config.train_keys, 101, 201),
        (
            "validation",
            config.validation_structures,
            config.validation_keys,
            301,
            401,
        ),
        ("test", config.test_structures, config.test_keys, 501, 601),
    )
    splits: dict[str, IntegralParitySplit] = {}
    for name, structures, keys, structure_offset, key_offset in split_specs:
        splits[name] = build_integral_split(
            name=name,
            rounds=config.rounds,
            structure_count=structures,
            key_count=keys,
            structure_seed=config.seed + structure_offset,
            key_seed=config.seed + key_offset,
            progress_callback=progress_callback,
        )
    dataset_summary = summarize_splits(splits)
    models = (
        ("anchor", "linear_same_input", False),
        ("candidate", "structure_mlp", False),
        ("control", "structure_mlp_shuffled_labels", True),
    )
    rows: list[dict[str, Any]] = []
    test_rate_records: dict[str, np.ndarray] = {}
    for role, model_name, shuffle_labels in models:
        model_seed = config.seed if role == "anchor" else config.seed + 1000
        row, test_structure_probabilities = _fit_model_row(
            config,
            role=role,
            model_name=model_name,
            shuffle_labels=shuffle_labels,
            train_split=splits["train"],
            validation_split=splits["validation"],
            test_split=splits["test"],
            model_seed=model_seed,
            progress_callback=progress_callback,
        )
        rows.append(row)
        test_rate_records[model_name] = test_structure_probabilities
    gate = adjudicate(config, rows, dataset_summary)
    structure_rate_rows = _structure_rate_rows(
        splits["test"],
        test_rate_records,
    )
    return {
        "rows": rows,
        "gate": gate,
        "dataset_summary": dataset_summary,
        "structure_rate_rows": structure_rate_rows,
    }


def summarize_splits(splits: dict[str, IntegralParitySplit]) -> dict[str, Any]:
    structure_sets = {
        name: {structure.signature for structure in split.structures}
        for name, split in splits.items()
    }
    key_sets = {name: set(split.keys) for name, split in splits.items()}
    structure_disjoint = _three_way_disjoint(structure_sets)
    key_disjoint = _three_way_disjoint(key_sets)
    summary: dict[str, Any] = {
        "status": "pass" if structure_disjoint and key_disjoint else "fail",
        "task": "innovation2_integral_property_prediction",
        "cipher": "PRESENT-80",
        "rounds": 5,
        "input_bits": INPUT_BITS,
        "integral_set_size": INTEGRAL_SET_SIZE,
        "structure_splits_disjoint": structure_disjoint,
        "key_splits_disjoint": key_disjoint,
        "splits": {},
    }
    for name, split in splits.items():
        labels = split.dataset.labels
        summary["splits"][name] = {
            "structures": len(split.structures),
            "keys_per_structure": len(split.keys),
            "rows": int(len(labels)),
            "q0_rows": int(np.sum(labels == 0)),
            "q1_rows": int(np.sum(labels == 1)),
            "q1_rate": float(labels.mean()),
            "all_zero_structures": int(np.sum(split.observed_q1_rates == 0.0)),
            "all_one_structures": int(np.sum(split.observed_q1_rates == 1.0)),
            "key_variable_structures": int(
                np.sum(
                    (split.observed_q1_rates > 0.0)
                    & (split.observed_q1_rates < 1.0)
                )
            ),
            "keys": [f"{key:020X}" for key in split.keys],
        }
    return summary


def _fit_model_row(
    config: IntegralExperimentConfig,
    *,
    role: str,
    model_name: str,
    shuffle_labels: bool,
    train_split: IntegralParitySplit,
    validation_split: IntegralParitySplit,
    test_split: IntegralParitySplit,
    model_seed: int,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, Any], np.ndarray]:
    fit_dataset = train_split.dataset
    if shuffle_labels:
        shuffled_labels = np.random.default_rng(model_seed + 77).permutation(
            train_split.dataset.labels
        )
        fit_dataset = DifferentialDataset(
            features=train_split.dataset.features,
            labels=shuffled_labels.astype(np.uint8, copy=False),
            metadata={
                **train_split.dataset.metadata,
                "fit_train_labels_shuffled": True,
            },
        )
    torch.manual_seed(model_seed)
    model: nn.Module
    if model_name == "linear_same_input":
        model = LinearParityPredictor(INPUT_BITS)
    else:
        model = MlpDistinguisher(INPUT_BITS, hidden_bits=config.hidden_bits)
    _emit(
        progress_callback,
        "model_start",
        model=model_name,
        role=role,
        fit_train_labels_shuffled=shuffle_labels,
    )
    training_config = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=model_seed,
        device=config.device,
        optimizer="adam",
        weight_decay=config.weight_decay,
        checkpoint_metric="val_auc",
        restore_best_checkpoint=True,
        loss="bce",
        train_eval_interval=1,
    )

    def model_progress(event: str, payload: dict[str, Any]) -> None:
        _emit(
            progress_callback,
            event,
            model=model_name,
            role=role,
            **payload,
        )

    training_result = train_binary_classifier(
        model,
        fit_dataset,
        validation_split.dataset,
        training_config,
        progress_callback=model_progress,
    )
    true_train_metrics = evaluate_binary_classifier(
        model,
        train_split.dataset,
        batch_size=config.batch_size,
        device=config.device,
    )
    test_metrics = evaluate_binary_classifier(
        model,
        test_split.dataset,
        batch_size=config.batch_size,
        device=config.device,
    )
    test_probabilities = predict_binary_probabilities(
        model,
        test_split.dataset,
        batch_size=config.batch_size,
        device=config.device,
    )
    rate_metrics, structure_probabilities = structure_rate_metrics(
        test_split,
        test_probabilities,
        global_prior=float(train_split.dataset.labels.mean()),
    )
    row: dict[str, Any] = {
        "run_id": config.run_id,
        "cipher": "PRESENT-80",
        "rounds": config.rounds,
        "seed": config.seed,
        "role": role,
        "task": "innovation2_integral_property_prediction",
        "model": model_name,
        "selected_model": model_name,
        "samples_per_class": None,
        "pairs_per_sample": INTEGRAL_SET_SIZE,
        "integral_set_size": INTEGRAL_SET_SIZE,
        "input_bits": INPUT_BITS,
        "input_view": "active_output_mask_fixed_context",
        "key_omitted_from_features": True,
        "fit_train_labels_shuffled": shuffle_labels,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "train_structures": len(train_split.structures),
        "train_keys_per_structure": len(train_split.keys),
        "train_rows": int(len(train_split.dataset.labels)),
        "validation": {
            "structures": len(validation_split.structures),
            "keys_per_structure": len(validation_split.keys),
            "rows": int(len(validation_split.dataset.labels)),
        },
        "test": {
            "structures": len(test_split.structures),
            "keys_per_structure": len(test_split.keys),
            "rows": int(len(test_split.dataset.labels)),
        },
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "history": training_result.history,
        "train_loss": true_train_metrics["loss"],
        "train_accuracy": true_train_metrics["accuracy"],
        "train_auc": true_train_metrics["auc"],
        "val_loss": training_result.final_metrics["loss"],
        "val_accuracy": training_result.final_metrics["accuracy"],
        "val_auc": training_result.final_metrics["auc"],
        "val_calibrated_accuracy": training_result.final_metrics[
            "calibrated_accuracy"
        ],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_calibrated_accuracy": test_metrics["calibrated_accuracy"],
        **rate_metrics,
        "training": training_result.metadata,
        "claim_scope": (
            "PRESENT r5 local structure-conditioned omitted-key integral parity "
            "probability diagnostic only; not an exact integral proof, not remote, "
            "and not formal output-prediction evidence"
        ),
    }
    _emit(
        progress_callback,
        "model_done",
        model=model_name,
        role=role,
        test_auc=row["test_auc"],
        test_structure_rate_mae=row["test_structure_rate_mae"],
    )
    return row, structure_probabilities


def structure_rate_metrics(
    split: IntegralParitySplit,
    probabilities: np.ndarray,
    *,
    global_prior: float,
) -> tuple[dict[str, float], np.ndarray]:
    predicted_rates = np.empty(len(split.structures), dtype=np.float64)
    for structure_index in range(len(split.structures)):
        predicted_rates[structure_index] = float(
            probabilities[split.structure_indices == structure_index].mean()
        )
    observed_rates = split.observed_q1_rates.astype(np.float64, copy=False)
    errors = predicted_rates - observed_rates
    prior_errors = np.full_like(observed_rates, global_prior) - observed_rates
    correlation = 0.0
    if np.std(predicted_rates) > 0.0 and np.std(observed_rates) > 0.0:
        correlation = float(np.corrcoef(predicted_rates, observed_rates)[0, 1])
    return (
        {
            "test_structure_rate_mae": float(np.mean(np.abs(errors))),
            "test_structure_rate_brier": float(np.mean(errors**2)),
            "test_structure_rate_correlation": correlation,
            "train_global_prior_rate_mae": float(np.mean(np.abs(prior_errors))),
        },
        predicted_rates,
    )


def adjudicate(
    config: IntegralExperimentConfig,
    rows: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    by_role = {str(row["role"]): row for row in rows}
    candidate = by_role["candidate"]
    linear = by_role["anchor"]
    shuffled = by_role["control"]
    split_has_both_labels = all(
        split["q0_rows"] > 0 and split["q1_rows"] > 0
        for split in dataset_summary["splits"].values()
    )
    finite_metrics = all(
        np.isfinite(float(row[key]))
        for row in rows
        for key in ("test_auc", "test_structure_rate_mae")
    )
    readiness_checks = {
        "structure_splits_disjoint": bool(
            dataset_summary["structure_splits_disjoint"]
        ),
        "key_splits_disjoint": bool(dataset_summary["key_splits_disjoint"]),
        "all_splits_have_both_labels": split_has_both_labels,
        "all_metrics_finite": finite_metrics,
        "three_model_rows_present": len(rows) == 3,
    }
    candidate_auc = float(candidate["test_auc"])
    linear_auc = float(linear["test_auc"])
    shuffled_auc = float(shuffled["test_auc"])
    candidate_rate_mae = float(candidate["test_structure_rate_mae"])
    linear_rate_mae = float(linear["test_structure_rate_mae"])
    diagnostic_checks = {
        "candidate_auc_at_least_0_60": candidate_auc >= 0.60,
        "candidate_linear_auc_margin_at_least_0_02": (
            candidate_auc - linear_auc >= 0.02
        ),
        "candidate_shuffled_auc_margin_at_least_0_05": (
            candidate_auc - shuffled_auc >= 0.05
        ),
        "candidate_rate_mae_improves_by_0_02": (
            linear_rate_mae - candidate_rate_mae >= 0.02
        ),
        "shuffled_auc_near_chance": abs(shuffled_auc - 0.5) <= 0.05,
    }
    if config.gate_mode == "smoke":
        passed = all(readiness_checks.values())
        status = "pass" if passed else "fail"
        decision = (
            "innovation2_integral_property_implementation_ready"
            if passed
            else "innovation2_integral_property_smoke_invalid"
        )
        next_action = (
            "Run the frozen 512/128/128-structure local diagnostic."
            if passed
            else "Repair dataset split, label balance, or artifact generation before training scale-up."
        )
    else:
        readiness_passed = all(readiness_checks.values())
        diagnostic_passed = all(diagnostic_checks.values())
        if readiness_passed and diagnostic_passed:
            status = "pass"
            decision = "innovation2_integral_property_advance_multiseed"
            next_action = (
                "Run seed1 with held-out active/output/mask geometry combinations and add a "
                "classical deterministic integral baseline; do not launch remote scale yet."
            )
        elif not readiness_passed or not diagnostic_checks["shuffled_auc_near_chance"]:
            status = "fail"
            decision = "innovation2_integral_property_invalid_control"
            next_action = (
                "Audit split leakage and shuffled-label behavior; the current result is invalid."
            )
        elif candidate_auc >= 0.60 and candidate_auc - linear_auc < 0.02:
            status = "hold"
            decision = "innovation2_integral_property_linear_signal_only"
            next_action = (
                "Add an explicit active/output/mask deterministic lookup baseline before any "
                "larger neural run; do not scale the MLP mechanically."
            )
        else:
            status = "hold"
            decision = "innovation2_integral_property_redesign_before_scale"
            next_action = (
                "Redesign the structure representation or target-noise treatment locally; do "
                "not increase structures, epochs, or remote GPU budget."
            )
    return {
        "status": status,
        "decision": decision,
        "gate_mode": config.gate_mode,
        "run_id": config.run_id,
        "readiness_checks": readiness_checks,
        "diagnostic_checks": diagnostic_checks,
        "metrics": {
            "candidate_test_auc": candidate_auc,
            "linear_test_auc": linear_auc,
            "shuffled_test_auc": shuffled_auc,
            "candidate_linear_auc_margin": candidate_auc - linear_auc,
            "candidate_shuffled_auc_margin": candidate_auc - shuffled_auc,
            "candidate_structure_rate_mae": candidate_rate_mae,
            "linear_structure_rate_mae": linear_rate_mae,
            "linear_candidate_rate_mae_margin": linear_rate_mae
            - candidate_rate_mae,
        },
        "next_action": next_action,
        "claim_scope": (
            "local PRESENT r5 structure-conditioned omitted-key integral parity "
            "probability diagnostic only"
        ),
    }


def _structure_rate_rows(
    split: IntegralParitySplit,
    model_probabilities: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, structure in enumerate(split.structures):
        row: dict[str, Any] = {
            "split": split.name,
            "structure_id": structure.structure_id,
            "signature": structure.signature,
            "active_nibble": structure.active_nibble,
            "output_nibble": structure.output_nibble,
            "output_mask": f"{structure.output_mask:04b}",
            "fixed_plaintext": f"0x{structure.fixed_plaintext:016X}",
            "observed_q1_rate": float(split.observed_q1_rates[index]),
            "observed_balance_rate": float(1.0 - split.observed_q1_rates[index]),
        }
        for model_name, probabilities in model_probabilities.items():
            row[f"{model_name}_predicted_q1_rate"] = float(probabilities[index])
            row[f"{model_name}_predicted_balance_rate"] = float(
                1.0 - probabilities[index]
            )
        rows.append(row)
    return rows


def _three_way_disjoint(values: dict[str, set[Any]]) -> bool:
    names = tuple(values)
    return all(
        values[names[left]].isdisjoint(values[names[right]])
        for left in range(len(names))
        for right in range(left + 1, len(names))
    )


def _emit(
    callback: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(event, payload)
