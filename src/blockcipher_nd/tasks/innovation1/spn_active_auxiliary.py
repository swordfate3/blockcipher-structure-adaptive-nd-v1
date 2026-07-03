from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.features.spn_active_auxiliary import (
    present_invp_active_mask_targets,
    shuffled_active_mask_targets,
)
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.protocols import OFFICIAL_ZHANG_WANG_CASE2_MCND
from blockcipher_nd.tasks.innovation1.spn_candidate.baseline import (
    binary_accuracy,
    binary_auc,
    calibrated_binary_accuracy,
)


DEFAULT_DIFFERENCE_PROFILE = "present_zhang_wang2022_mcnd"
DEFAULT_VALIDATION_KEY = 0x11111111111111111111
ACTIVE_AUX_MODEL = "present_nibble_invp_active_aux_spn_only"
SHUFFLED_ACTIVE_AUX_MODEL = "present_nibble_invp_active_aux_shuffled_targets"
DEFAULTS = {
    "output": None,
    "rounds": 7,
    "seed": 0,
    "samples_per_class": 4096,
    "pairs_per_sample": 16,
    "negative_mode": "encrypted_random_plaintexts",
    "sample_structure": OFFICIAL_ZHANG_WANG_CASE2_MCND,
    "difference_profile": DEFAULT_DIFFERENCE_PROFILE,
    "difference_member": 0,
    "train_key": 0,
    "validation_key": DEFAULT_VALIDATION_KEY,
    "key_rotation_interval": 0,
    "epochs": 5,
    "learning_rate": 1e-3,
    "lambda_aux": 0.1,
    "batch_size": 512,
    "hidden_bits": 32,
    "spn_mixer_depth": 2,
    "device": "cpu",
    "model": ACTIVE_AUX_MODEL,
}
PATH_FIELDS = {"output"}
INT_FIELDS = {
    "rounds",
    "seed",
    "samples_per_class",
    "pairs_per_sample",
    "difference_member",
    "train_key",
    "validation_key",
    "key_rotation_interval",
    "epochs",
    "batch_size",
    "hidden_bits",
    "spn_mixer_depth",
}
FLOAT_FIELDS = {"learning_rate", "lambda_aux"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--pairs-per-sample", type=int, default=None)
    parser.add_argument("--negative-mode", default=None)
    parser.add_argument("--sample-structure", default=None)
    parser.add_argument("--difference-profile", default=None)
    parser.add_argument("--difference-member", type=int, default=None)
    parser.add_argument("--train-key", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--validation-key", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--key-rotation-interval", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lambda-aux", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-bits", type=int, default=None)
    parser.add_argument("--spn-mixer-depth", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", choices=[ACTIVE_AUX_MODEL, SHUFFLED_ACTIVE_AUX_MODEL], default=None)
    args = parser.parse_args(argv)
    return _apply_config_defaults(args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_active_auxiliary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


def run_active_auxiliary(args: argparse.Namespace) -> dict[str, object]:
    input_difference = difference_for_profile(args.difference_profile, args.difference_member)
    train_features, train_labels = _make_raw_dataset(
        rounds=args.rounds,
        key=args.train_key,
        input_difference=input_difference,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
        key_rotation_interval=args.key_rotation_interval,
    )
    validation_samples_per_class = max(2, args.samples_per_class // 4)
    validation_features, validation_labels = _make_raw_dataset(
        rounds=args.rounds,
        key=args.validation_key,
        input_difference=input_difference,
        seed=args.seed + 10_000,
        samples_per_class=validation_samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
        key_rotation_interval=args.key_rotation_interval,
    )
    device = torch.device(args.device)
    shuffled_targets = args.model == SHUFFLED_ACTIVE_AUX_MODEL
    model, train_aux_loss = _train_active_aux_model(
        train_features,
        train_labels,
        model_name=ACTIVE_AUX_MODEL,
        hidden_bits=args.hidden_bits,
        spn_mixer_depth=args.spn_mixer_depth,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_aux=args.lambda_aux,
        batch_size=args.batch_size,
        shuffled_targets=shuffled_targets,
        seed=args.seed,
        device=device,
    )
    logits, validation_aux_loss = _evaluate_active_aux_model(
        model,
        validation_features,
        validation_labels,
        lambda_aux=args.lambda_aux,
        shuffled_targets=shuffled_targets,
        seed=args.seed + 10_000,
        device=device,
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    val_accuracy = binary_accuracy(validation_labels, probabilities)
    val_auc = binary_auc(validation_labels, probabilities)
    calibrated_accuracy, calibrated_threshold = calibrated_binary_accuracy(validation_labels, probabilities)
    aux_target = "shuffled_present_invp_active_mask" if shuffled_targets else "present_invp_active_mask"
    metrics = {
        "accuracy": val_accuracy,
        "auc": val_auc,
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": calibrated_threshold,
        "auxiliary_loss": validation_aux_loss,
        "train_auxiliary_loss": train_aux_loss,
    }
    return {
        "route": args.model,
        "model": args.model,
        "selected_model": args.model,
        "rounds": args.rounds,
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "validation_samples_per_class": validation_samples_per_class,
        "pairs_per_sample": args.pairs_per_sample,
        "negative_mode": args.negative_mode,
        "sample_structure": args.sample_structure,
        "difference_profile": args.difference_profile,
        "difference_member": args.difference_member,
        "input_difference": input_difference,
        "key_rotation_interval": args.key_rotation_interval,
        "device": args.device,
        "feature_route": "active_pattern_auxiliary_head",
        "auxiliary_target": aux_target,
        "lambda_aux": args.lambda_aux,
        "shuffled_auxiliary_targets": shuffled_targets,
        "input_bits": int(train_features.shape[1]),
        "metrics": metrics,
        "accuracy": val_accuracy,
        "auc": val_auc,
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": calibrated_threshold,
        "val_accuracy": val_accuracy,
        "val_auc": val_auc,
        "val_best_accuracy": calibrated_accuracy,
        "val_calibrated_accuracy": calibrated_accuracy,
    }


def args_from_config(config: dict[str, object]) -> argparse.Namespace:
    args = argparse.Namespace(config=None)
    for field in DEFAULTS:
        setattr(args, field, config.get(field))
    return _apply_config_defaults(args, require_output=False)


def _make_raw_dataset(
    *,
    rounds: int,
    key: int,
    input_difference: int,
    seed: int,
    samples_per_class: int,
    pairs_per_sample: int,
    negative_mode: str,
    sample_structure: str,
    key_rotation_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    cipher = build_cipher("present80", rounds, key=key)
    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=input_difference,
            samples_per_class=samples_per_class,
            seed=seed,
            pairs_per_sample=pairs_per_sample,
            negative_mode=negative_mode,
            sample_structure=sample_structure,
            key_rotation_interval=key_rotation_interval,
        )
    )
    return dataset.features.astype(np.float32), dataset.labels.astype(np.uint8)


def _train_active_aux_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    model_name: str,
    hidden_bits: int,
    spn_mixer_depth: int,
    epochs: int,
    learning_rate: float,
    lambda_aux: float,
    batch_size: int,
    shuffled_targets: bool,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, float]:
    torch.manual_seed(seed)
    model = build_model(
        model_name,
        input_bits=features.shape[1],
        hidden_bits=hidden_bits,
        pair_bits=128,
        structure="spn",
        model_options={"spn_mixer_depth": spn_mixer_depth, "activation": "relu", "norm": "layernorm"},
    ).to(device)
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1).to(device)
    aux_targets = _active_targets(x, shuffled=shuffled_targets, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    main_loss_fn = torch.nn.BCEWithLogitsLoss()
    aux_loss_fn = torch.nn.BCEWithLogitsLoss()
    last_aux_loss = 0.0
    for _epoch in range(epochs):
        for batch_start in range(0, x.shape[0], max(1, batch_size)):
            batch_end = min(x.shape[0], batch_start + max(1, batch_size))
            batch_x = x[batch_start:batch_end]
            batch_y = y[batch_start:batch_end]
            batch_aux = aux_targets[batch_start:batch_end]
            optimizer.zero_grad()
            main_loss = main_loss_fn(model(batch_x), batch_y)
            aux_loss = aux_loss_fn(model.active_mask_logits(batch_x), batch_aux)
            loss = main_loss + float(lambda_aux) * aux_loss
            loss.backward()
            optimizer.step()
            last_aux_loss = float(aux_loss.detach().cpu().item())
    return model, last_aux_loss


def _evaluate_active_aux_model(
    model: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    lambda_aux: float,
    shuffled_targets: bool,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    del lambda_aux
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    aux_targets = _active_targets(x, shuffled=shuffled_targets, seed=seed)
    aux_loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy().reshape(-1)
        aux_loss = aux_loss_fn(model.active_mask_logits(x), aux_targets)
    return logits, float(aux_loss.detach().cpu().item())


def _active_targets(features: torch.Tensor, *, shuffled: bool, seed: int) -> torch.Tensor:
    targets = present_invp_active_mask_targets(features, pair_bits=128)
    if shuffled:
        targets = shuffled_active_mask_targets(targets, seed=seed)
    return targets


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"active-auxiliary config must be a JSON object: {path}")
    return data


def _apply_config_defaults(args: argparse.Namespace, *, require_output: bool = True) -> argparse.Namespace:
    config = _load_config(args.config)
    for field, default in DEFAULTS.items():
        value = getattr(args, field)
        if value is None:
            value = config.get(field, default)
        if field in PATH_FIELDS and value is not None:
            value = Path(str(value))
        elif field in INT_FIELDS:
            value = int(str(value), 0) if isinstance(value, str) else int(value)
        elif field in FLOAT_FIELDS:
            value = float(value)
        setattr(args, field, value)
    if require_output and args.output is None:
        raise SystemExit("active-auxiliary output path is required")
    if args.model not in {ACTIVE_AUX_MODEL, SHUFFLED_ACTIVE_AUX_MODEL}:
        raise SystemExit(f"unsupported active-auxiliary model: {args.model}")
    return args


__all__ = ["args_from_config", "run_active_auxiliary"]
