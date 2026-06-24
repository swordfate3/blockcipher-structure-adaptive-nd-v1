from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.features.pair_features import pair_bits_for_encoding
from blockcipher_nd.features.spn_active_pattern import extract_active_pattern_features


DEFAULT_FEATURE_ENCODING = "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=4096)
    parser.add_argument("--pairs-per-sample", type=int, default=16)
    parser.add_argument("--feature-encoding", default=DEFAULT_FEATURE_ENCODING)
    parser.add_argument("--negative-mode", default="encrypted_random_plaintexts")
    parser.add_argument("--sample-structure", default="zhang_wang_case2_mcnd")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def train_linear(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> torch.nn.Module:
    torch.manual_seed(0)
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1).to(device)
    model = torch.nn.Linear(features.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return model


def binary_accuracy(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5) -> float:
    predictions = (probabilities >= threshold).astype(np.uint8)
    return float((predictions == labels.astype(np.uint8)).mean())


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    label_array = labels.astype(np.uint8)
    positive_count = int(label_array.sum())
    negative_count = int(label_array.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = float(original_ranks[label_array == 1].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def make_dataset(
    *,
    rounds: int,
    seed: int,
    samples_per_class: int,
    pairs_per_sample: int,
    feature_encoding: str,
    negative_mode: str,
    sample_structure: str,
):
    cipher = build_cipher("present80", rounds)
    config = DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=0x0000000000000040,
        samples_per_class=samples_per_class,
        seed=seed,
        feature_encoding=feature_encoding,
        pairs_per_sample=pairs_per_sample,
        negative_mode=negative_mode,
        sample_structure=sample_structure,
    )
    return make_differential_dataset(config), cipher


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    train_dataset, cipher = make_dataset(
        rounds=args.rounds,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        pairs_per_sample=args.pairs_per_sample,
        feature_encoding=args.feature_encoding,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
    )
    validation_dataset, _cipher = make_dataset(
        rounds=args.rounds,
        seed=args.seed + 1000,
        samples_per_class=max(1024, args.samples_per_class // 4),
        pairs_per_sample=args.pairs_per_sample,
        feature_encoding=args.feature_encoding,
        negative_mode=args.negative_mode,
        sample_structure=args.sample_structure,
    )
    pair_bits = pair_bits_for_encoding(cipher.block_bits, args.feature_encoding)
    expected_bits = pair_bits * args.pairs_per_sample
    if train_dataset.features.shape[1] != expected_bits:
        raise ValueError("dataset feature width does not match pair_bits * pairs_per_sample")
    words_per_row = train_dataset.features.shape[1] // 64
    train_features = extract_active_pattern_features(train_dataset.features, words_per_row=words_per_row)
    validation_features = extract_active_pattern_features(
        validation_dataset.features,
        words_per_row=words_per_row,
    )
    model = train_linear(
        train_features,
        train_dataset.labels,
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
        "route": "spn_active_pattern_baseline",
        "rounds": args.rounds,
        "seed": args.seed,
        "samples_per_class": args.samples_per_class,
        "pairs_per_sample": args.pairs_per_sample,
        "feature_encoding": args.feature_encoding,
        "negative_mode": args.negative_mode,
        "sample_structure": args.sample_structure,
        "device": args.device,
        "feature_dim": int(train_features.shape[1]),
        "val_accuracy": binary_accuracy(validation_dataset.labels, probabilities),
        "val_auc": binary_auc(validation_dataset.labels, probabilities),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
