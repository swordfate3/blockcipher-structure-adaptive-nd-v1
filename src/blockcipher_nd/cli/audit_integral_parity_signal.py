from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.features.pair_features import pair_bits_for_encoding
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.cipher_factory import build_cipher


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether plaintext_integral_nibble samples are already separable "
            "by a deterministic pair-xor parity statistic."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--key-split", choices=["train", "validation"], default="validation")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def integral_parity_audit_from_task(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None = None,
    key_split: str = "validation",
) -> dict[str, Any]:
    if task["sample_structure"] not in {
        "plaintext_integral_nibble",
        "plaintext_integral_nibble_matched_negative",
    }:
        raise ValueError("integral parity audit requires a plaintext_integral_nibble sample structure")
    if task["feature_encoding"] != "ciphertext_pair_bits":
        raise ValueError("integral parity audit currently requires feature_encoding=ciphertext_pair_bits")

    key = task["validation_key"] if key_split == "validation" else task["train_key"]
    cipher = build_cipher(task["cipher_key"], task["rounds"], key=key)
    pair_bits = pair_bits_for_encoding(cipher.block_bits, task["feature_encoding"])
    if pair_bits != cipher.block_bits * 2:
        raise ValueError("ciphertext_pair_bits must encode exactly left and right ciphertext blocks")

    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=task["input_difference"],
            samples_per_class=samples_per_class,
            seed=task["seed"] if seed is None else seed,
            shuffle=False,
            feature_encoding=task["feature_encoding"],
            pairs_per_sample=task["pairs_per_sample"],
            negative_mode=task["negative_mode"],
            key_rotation_interval=task["key_rotation_interval"],
            sample_structure=task["sample_structure"],
            integral_active_nibble=task["integral_active_nibble"],
            selected_bit_indices=task["selected_bit_indices"],
        )
    )

    features = dataset.features.astype(np.int16).reshape(
        (-1, task["pairs_per_sample"], pair_bits)
    )
    labels = dataset.labels.astype(bool)
    left = features[:, :, : cipher.block_bits]
    right = features[:, :, cipher.block_bits :]
    pair_xor = left ^ right
    parity_hw = (pair_xor.sum(axis=1) % 2).sum(axis=1)
    positive = parity_hw[labels]
    negative = parity_hw[~labels]
    best = _best_threshold(parity_hw, labels)

    return {
        "status": "pass",
        "audit": "integral_pair_xor_parity",
        "cipher_key": task["cipher_key"],
        "rounds": task["rounds"],
        "samples_per_class": samples_per_class,
        "seed": task["seed"] if seed is None else seed,
        "key_split": key_split,
        "sample_structure": task["sample_structure"],
        "negative_mode": task["negative_mode"],
        "feature_encoding": task["feature_encoding"],
        "pairs_per_sample": task["pairs_per_sample"],
        "input_difference": task["input_difference"],
        "positive_pair_xor_parity_hw": _distribution_summary(positive),
        "negative_pair_xor_parity_hw": _distribution_summary(negative),
        "best_threshold": best,
        "interpretation": _interpretation(best["accuracy"]),
        "claim_scope": (
            "Deterministic local data-structure audit only; this is not a neural "
            "training result, not Zhang/Wang same-protocol evidence, and not a "
            "formal route claim."
        ),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "min": int(values.min()) if values.size else None,
        "max": int(values.max()) if values.size else None,
        "zero_count": int((values == 0).sum()),
        "zero_rate": float((values == 0).mean()) if values.size else None,
    }


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    best = {"accuracy": -1.0, "threshold": None, "operator": None}
    for threshold in np.unique(scores):
        for operator in ("<=", ">="):
            predicted = scores <= threshold if operator == "<=" else scores >= threshold
            accuracy = float((predicted == labels).mean())
            if accuracy > best["accuracy"]:
                best = {
                    "accuracy": accuracy,
                    "threshold": int(threshold),
                    "operator": operator,
                }
    return best


def _interpretation(accuracy: float) -> str:
    if accuracy >= 0.99:
        return "parity_statistic_alone_nearly_separates_classes"
    if accuracy >= 0.75:
        return "parity_statistic_is_strong_control_signal"
    if accuracy >= 0.6:
        return "parity_statistic_is_weak_control_signal"
    return "parity_statistic_does_not_explain_result_by_itself"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = tasks_from_plan(
        args.plan,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=1,
        difference_profile=None,
        difference_member=0,
    )
    if args.row_index < 0 or args.row_index >= len(tasks):
        raise ValueError(f"row-index {args.row_index} outside plan rows 0..{len(tasks) - 1}")
    report = integral_parity_audit_from_task(
        tasks[args.row_index],
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        key_split=args.key_split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
