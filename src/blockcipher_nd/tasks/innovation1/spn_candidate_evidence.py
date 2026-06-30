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
from blockcipher_nd.tasks.innovation1.protocols import LEGACY_MASKED_ZHANG_WANG_CASE2_MCND


DEFAULT_DIFFERENCE_PROFILE = "present_zhang_wang2022_mcnd"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=4096)
    parser.add_argument("--pairs-per-sample", type=int, default=16)
    parser.add_argument("--negative-mode", default="encrypted_random_plaintexts")
    parser.add_argument("--sample-structure", default=LEGACY_MASKED_ZHANG_WANG_CASE2_MCND)
    parser.add_argument("--difference-profile", default=DEFAULT_DIFFERENCE_PROFILE)
    parser.add_argument("--difference-member", type=int, default=0)
    parser.add_argument("--train-key", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--validation-key", type=lambda value: int(value, 0), default=(1 << 80) - 1)
    parser.add_argument("--key-rotation-interval", type=int, default=1024)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--feature-cache-root", type=Path, default=None)
    parser.add_argument("--feature-cache-chunk-size", type=int, default=4096)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--model", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


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
    result = {
        "route": "spn_candidate_evidence_baseline",
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
        "model": args.model,
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


if __name__ == "__main__":
    main()
