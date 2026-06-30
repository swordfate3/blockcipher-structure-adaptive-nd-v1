from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.tasks.innovation1.spn_candidate.baseline import (
    binary_accuracy,
    binary_auc,
    train_model,
)
from blockcipher_nd.tasks.innovation1.spn_candidate.dataset import make_candidate_dataset
from blockcipher_nd.tasks.innovation1.protocols import OFFICIAL_ZHANG_WANG_CASE2_MCND


DEFAULT_DIFFERENCE_PROFILE = "present_zhang_wang2022_mcnd"
DEFAULT_VALIDATION_KEY = 0x11111111111111111111
MODEL_ROUTE_KEYS = {
    "linear": "candidate_trail_consistency_linear",
    "mlp": "candidate_trail_consistency_mlp",
}


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
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--feature-cache-root", type=Path, default=None)
    parser.add_argument("--feature-cache-chunk-size", type=int, default=None)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--model", choices=["linear", "mlp"], default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    return _apply_config_defaults(args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_difference = difference_for_profile(args.difference_profile, args.difference_member)
    train_features, train_labels = make_candidate_dataset(
        rounds=args.rounds,
        key=args.train_key,
        input_difference=input_difference,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
        key_rotation_interval=args.key_rotation_interval,
        beam_width=args.beam_width,
        depth=args.depth,
        feature_cache_root=args.feature_cache_root,
        feature_cache_chunk_size=args.feature_cache_chunk_size,
        progress_output=args.progress_output,
        split="train",
    )
    validation_samples_per_class = max(1024, args.samples_per_class // 4)
    validation_features, validation_labels = make_candidate_dataset(
        rounds=args.rounds,
        key=args.validation_key,
        input_difference=input_difference,
        seed=args.seed + 10_000,
        samples_per_class=validation_samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
        key_rotation_interval=args.key_rotation_interval,
        beam_width=args.beam_width,
        depth=args.depth,
        feature_cache_root=args.feature_cache_root,
        feature_cache_chunk_size=args.feature_cache_chunk_size,
        progress_output=args.progress_output,
        split="validation",
    )
    device = torch.device(args.device)
    model = train_model(
        train_features,
        train_labels,
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    with torch.no_grad():
        logits = (
            model(torch.from_numpy(validation_features.astype(np.float32)).to(device))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    route = MODEL_ROUTE_KEYS[args.model]
    result = {
        "route": route,
        "model": route,
        "training_model": args.model,
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
        "beam_width": args.beam_width,
        "depth": args.depth,
        "device": args.device,
        "feature_cache_enabled": args.feature_cache_root is not None,
        "feature_cache_root": str(args.feature_cache_root) if args.feature_cache_root is not None else None,
        "feature_cache_chunk_size": args.feature_cache_chunk_size,
        "progress_output": str(args.progress_output) if args.progress_output is not None else None,
        "feature_dim": int(train_features.shape[1]),
        "val_accuracy": binary_accuracy(validation_labels, probabilities),
        "val_auc": binary_auc(validation_labels, probabilities),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


def _apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(args.config)
    defaults = {
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
        "beam_width": 4,
        "depth": 3,
        "feature_cache_root": None,
        "feature_cache_chunk_size": 4096,
        "progress_output": None,
        "epochs": 30,
        "learning_rate": 1e-2,
        "model": "linear",
        "device": "cpu",
    }
    path_fields = {"output", "feature_cache_root", "progress_output"}
    int_fields = {
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "difference_member",
        "train_key",
        "validation_key",
        "key_rotation_interval",
        "beam_width",
        "depth",
        "feature_cache_chunk_size",
        "epochs",
    }
    float_fields = {"learning_rate"}
    for field, default in defaults.items():
        value = getattr(args, field)
        if value is None:
            value = config.get(field, default)
        if field in path_fields and value is not None:
            value = Path(value)
        elif field in int_fields and value is not None:
            value = int(value, 0) if isinstance(value, str) else int(value)
        elif field in float_fields and value is not None:
            value = float(value)
        setattr(args, field, value)
    if args.output is None:
        raise SystemExit("--output is required unless provided by --config")
    if args.model not in MODEL_ROUTE_KEYS:
        raise SystemExit(f"unsupported model in config: {args.model}")
    return args


def _load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"candidate evidence config must be a JSON object: {path}")
    return data


if __name__ == "__main__":
    main()
