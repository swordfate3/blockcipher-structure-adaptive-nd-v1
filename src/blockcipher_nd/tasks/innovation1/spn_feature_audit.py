from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.features.registry import pair_bits_for_encoding
from blockcipher_nd.tasks.innovation1.protocols import OFFICIAL_ZHANG_WANG_CASE2_MCND
from blockcipher_nd.training.metrics import binary_auc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SPN feature-level positive/negative separation before expensive training."
    )
    parser.add_argument("--cipher", default="present80")
    parser.add_argument("--rounds", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--samples-per-class", type=int, default=2048)
    parser.add_argument("--pairs-per-sample", type=int, default=16)
    parser.add_argument(
        "--feature-encodings",
        nargs="+",
        default=[
            "present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits",
            "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits",
            "present_pair_xor_paligned_sboxddt_beam4deep3_cell_matrix_bits",
            "present_pair_xor_paligned_sboxddt_beam8deep4_cell_matrix_bits",
        ],
    )
    parser.add_argument("--difference-profile", default="present_zhang_wang2022_mcnd")
    parser.add_argument("--difference-member", type=int, default=0)
    parser.add_argument("--negative-mode", default="encrypted_random_plaintexts")
    parser.add_argument("--key-rotation-interval", type=int, default=1024)
    parser.add_argument("--sample-structure", default=OFFICIAL_ZHANG_WANG_CASE2_MCND)
    parser.add_argument("--train-key", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--output", default="outputs/innovation1/spn_feature_separation_audit.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = run_audit(args)
    payload = {
        "kind": "spn_feature_separation_audit",
        "config": {
            "cipher": args.cipher,
            "rounds": args.rounds,
            "seeds": args.seeds,
            "samples_per_class": args.samples_per_class,
            "pairs_per_sample": args.pairs_per_sample,
            "feature_encodings": args.feature_encodings,
            "difference_profile": args.difference_profile,
            "difference_member": args.difference_member,
            "negative_mode": args.negative_mode,
            "key_rotation_interval": args.key_rotation_interval,
            "sample_structure": args.sample_structure,
            "train_key": args.train_key,
            "top_k": args.top_k,
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} audit rows to {output}")


def run_audit(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_difference = difference_for_profile(args.difference_profile, args.difference_member)
    for rounds in args.rounds:
        for feature_encoding in args.feature_encodings:
            for seed in args.seeds:
                cipher = build_cipher(args.cipher, rounds, key=args.train_key)
                dataset = make_differential_dataset(
                    DifferentialDatasetConfig(
                        cipher=cipher,
                        input_difference=input_difference,
                        samples_per_class=args.samples_per_class,
                        seed=seed,
                        shuffle=True,
                        feature_encoding=feature_encoding,
                        pairs_per_sample=args.pairs_per_sample,
                        negative_mode=args.negative_mode,
                        key_rotation_interval=args.key_rotation_interval,
                        sample_structure=args.sample_structure,
                    )
                )
                rows.append(
                    audit_dataset(
                        dataset.features.astype(np.float32),
                        dataset.labels.astype(np.uint8),
                        pair_bits=pair_bits_for_encoding(cipher.block_bits, feature_encoding),
                        block_bits=cipher.block_bits,
                        cipher_name=cipher.name,
                        rounds=rounds,
                        seed=seed,
                        feature_encoding=feature_encoding,
                        top_k=args.top_k,
                    )
                )
    return rows


def audit_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pair_bits: int,
    block_bits: int,
    cipher_name: str,
    rounds: int,
    seed: int,
    feature_encoding: str,
    top_k: int,
) -> dict[str, Any]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if features.shape[1] % pair_bits != 0:
        raise ValueError("feature width must be a multiple of pair_bits")
    pairs_per_sample = features.shape[1] // pair_bits
    words_per_pair = pair_bits // block_bits
    cells_per_word = block_bits // 4

    bit_scores = _feature_axis_scores(features, labels)
    word_activity = features.reshape(features.shape[0], pairs_per_sample, words_per_pair, block_bits).mean(axis=3)
    word_scores = _feature_axis_scores(word_activity.reshape(features.shape[0], -1), labels)
    cell_activity = features.reshape(features.shape[0], pairs_per_sample, words_per_pair, cells_per_word, 4).mean(axis=4)
    cell_scores = _feature_axis_scores(cell_activity.reshape(features.shape[0], -1), labels)
    pair_activity = features.reshape(features.shape[0], pairs_per_sample, pair_bits).mean(axis=2)
    pair_scores = _feature_axis_scores(pair_activity, labels)
    global_scores = _named_scalar_scores(
        {
            "global_bit_density": features.mean(axis=1),
            "pair_activity_mean": pair_activity.mean(axis=1),
            "pair_activity_std": pair_activity.std(axis=1),
            "word_activity_mean": word_activity.mean(axis=(1, 2)),
            "word_activity_std": word_activity.std(axis=(1, 2)),
            "cell_activity_mean": cell_activity.mean(axis=(1, 2, 3)),
            "cell_activity_std": cell_activity.std(axis=(1, 2, 3)),
            "first_last_pair_activity_delta": pair_activity[:, -1] - pair_activity[:, 0],
            "first_last_word_activity_delta": word_activity[:, -1, -1] - word_activity[:, 0, 0],
        },
        labels,
    )

    top_bit_indices = _top_indices(bit_scores["auc_advantage"], top_k)
    top_word_indices = _top_indices(word_scores["auc_advantage"], top_k)
    top_cell_indices = _top_indices(cell_scores["auc_advantage"], top_k)
    return {
        "cipher": cipher_name,
        "rounds": rounds,
        "seed": seed,
        "feature_encoding": feature_encoding,
        "samples": int(features.shape[0]),
        "samples_per_class": int(min((labels == 0).sum(), (labels == 1).sum())),
        "input_bits": int(features.shape[1]),
        "pair_bits": int(pair_bits),
        "pairs_per_sample": int(pairs_per_sample),
        "words_per_pair": int(words_per_pair),
        "best_bit_auc_advantage": float(bit_scores["auc_advantage"][top_bit_indices[0]]) if top_bit_indices else 0.0,
        "best_word_auc_advantage": float(word_scores["auc_advantage"][top_word_indices[0]]) if top_word_indices else 0.0,
        "best_cell_auc_advantage": float(cell_scores["auc_advantage"][top_cell_indices[0]]) if top_cell_indices else 0.0,
        "best_pair_auc_advantage": float(pair_scores["auc_advantage"].max()) if pair_scores["auc_advantage"].size else 0.0,
        "global_scores": global_scores,
        "top_bits": _top_feature_rows(bit_scores, top_bit_indices),
        "top_words": _top_feature_rows(word_scores, top_word_indices),
        "top_cells": _top_feature_rows(cell_scores, top_cell_indices),
    }


def _feature_axis_scores(features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    positive = features[labels == 1]
    negative = features[labels == 0]
    pos_mean = positive.mean(axis=0)
    neg_mean = negative.mean(axis=0)
    pos_var = positive.var(axis=0)
    neg_var = negative.var(axis=0)
    pooled_std = np.sqrt((pos_var + neg_var) / 2.0)
    cohen_d = np.divide(pos_mean - neg_mean, pooled_std, out=np.zeros_like(pos_mean), where=pooled_std > 0)
    auc = np.array([binary_auc(labels, features[:, index]) for index in range(features.shape[1])], dtype=np.float64)
    return {
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "mean_delta": pos_mean - neg_mean,
        "cohen_d": cohen_d,
        "auc": auc,
        "auc_advantage": np.abs(auc - 0.5),
    }


def _named_scalar_scores(named_scores: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, dict[str, float]]:
    rows = {}
    for name, scores in named_scores.items():
        scores = scores.astype(np.float64)
        positive = scores[labels == 1]
        negative = scores[labels == 0]
        pooled_std = np.sqrt((positive.var() + negative.var()) / 2.0)
        cohen_d = float((positive.mean() - negative.mean()) / pooled_std) if pooled_std > 0 else 0.0
        auc = binary_auc(labels, scores)
        rows[name] = {
            "positive_mean": float(positive.mean()),
            "negative_mean": float(negative.mean()),
            "mean_delta": float(positive.mean() - negative.mean()),
            "cohen_d": cohen_d,
            "auc": float(auc),
            "auc_advantage": float(abs(auc - 0.5)),
        }
    return rows


def _top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if top_k <= 0 or scores.size == 0:
        return []
    count = min(top_k, scores.size)
    return np.argsort(scores, kind="mergesort")[-count:][::-1].astype(int).tolist()


def _top_feature_rows(scores: dict[str, np.ndarray], indices: list[int]) -> list[dict[str, float | int]]:
    rows = []
    for index in indices:
        rows.append(
            {
                "index": int(index),
                "positive_mean": float(scores["positive_mean"][index]),
                "negative_mean": float(scores["negative_mean"][index]),
                "mean_delta": float(scores["mean_delta"][index]),
                "cohen_d": float(scores["cohen_d"][index]),
                "auc": float(scores["auc"][index]),
                "auc_advantage": float(scores["auc_advantage"][index]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
