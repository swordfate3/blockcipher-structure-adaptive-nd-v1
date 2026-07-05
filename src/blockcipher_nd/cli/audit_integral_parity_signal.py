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
    parser.add_argument(
        "--audit",
        choices=["parity", "alignment", "feature-bank", "deterministic-baseline"],
        default="parity",
        help="deterministic statistic family to audit",
    )
    parser.add_argument(
        "--statistic",
        default="pair_xor_column_sum_variance",
        help="fixed statistic to evaluate when --audit deterministic-baseline is used",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def integral_parity_audit_from_task(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None = None,
    key_split: str = "validation",
) -> dict[str, Any]:
    cipher, features, labels, pair_bits = _ciphertext_pair_feature_arrays(
        task,
        samples_per_class=samples_per_class,
        seed=seed,
        key_split=key_split,
    )
    left = features[:, :, : cipher.block_bits]
    right = features[:, :, cipher.block_bits :]
    if pair_bits != cipher.block_bits * 2:
        raise ValueError("ciphertext_pair_bits must encode exactly left and right ciphertext blocks")
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
        "claim_scope": _claim_scope(),
    }


def integral_alignment_audit_from_task(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None = None,
    key_split: str = "validation",
) -> dict[str, Any]:
    cipher, features, labels, pair_bits = _ciphertext_pair_feature_arrays(
        task,
        samples_per_class=samples_per_class,
        seed=seed,
        key_split=key_split,
    )
    if pair_bits != cipher.block_bits * 2:
        raise ValueError("ciphertext_pair_bits must encode exactly left and right ciphertext blocks")

    left = features[:, :, : cipher.block_bits]
    right = features[:, :, cipher.block_bits :]
    shifted_right = np.roll(right, shift=-1, axis=1)

    same_index_xor_hw_mean = (left ^ right).sum(axis=2).mean(axis=1)
    shifted_index_xor_hw_mean = (left ^ shifted_right).sum(axis=2).mean(axis=1)
    same_minus_shifted_xor_hw_mean = same_index_xor_hw_mean - shifted_index_xor_hw_mean

    statistics = {
        "same_index_xor_hw_mean": _statistic_report(same_index_xor_hw_mean, labels),
        "shifted_index_xor_hw_mean": _statistic_report(shifted_index_xor_hw_mean, labels),
        "same_minus_shifted_xor_hw_mean": _statistic_report(
            same_minus_shifted_xor_hw_mean,
            labels,
        ),
    }
    best_name, best_report = max(
        statistics.items(),
        key=lambda item: item[1]["best_threshold"]["accuracy"],
    )

    return {
        "status": "pass",
        "audit": "integral_pair_alignment",
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
        "statistics": statistics,
        "best_statistic": {
            "name": best_name,
            "best_threshold": best_report["best_threshold"],
            "interpretation": _deterministic_statistic_interpretation(
                best_report["best_threshold"]["accuracy"]
            ),
        },
        "claim_scope": _claim_scope(),
    }


def integral_feature_bank_audit_from_task(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None = None,
    key_split: str = "validation",
) -> dict[str, Any]:
    cipher, features, labels, pair_bits = _ciphertext_pair_feature_arrays(
        task,
        samples_per_class=samples_per_class,
        seed=seed,
        key_split=key_split,
    )
    if pair_bits != cipher.block_bits * 2:
        raise ValueError("ciphertext_pair_bits must encode exactly left and right ciphertext blocks")

    scores = _integral_feature_bank_scores(features, cipher.block_bits)
    statistics = {name: _statistic_report(score, labels) for name, score in scores.items()}
    best_name, best_report = max(
        statistics.items(),
        key=lambda item: item[1]["best_threshold"]["accuracy"],
    )

    return {
        "status": "pass",
        "audit": "integral_feature_bank",
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
        "statistics": statistics,
        "best_statistic": {
            "name": best_name,
            "best_threshold": best_report["best_threshold"],
            "interpretation": _deterministic_statistic_interpretation(
                best_report["best_threshold"]["accuracy"]
            ),
        },
        "claim_scope": _claim_scope(),
    }


def integral_deterministic_baseline_from_task(
    task: dict[str, Any],
    *,
    statistic: str,
    samples_per_class: int,
    seed: int | None = None,
    key_split: str = "validation",
) -> dict[str, Any]:
    cipher, features, labels, pair_bits = _ciphertext_pair_feature_arrays(
        task,
        samples_per_class=samples_per_class,
        seed=seed,
        key_split=key_split,
    )
    if pair_bits != cipher.block_bits * 2:
        raise ValueError("ciphertext_pair_bits must encode exactly left and right ciphertext blocks")

    scores = _integral_feature_bank_scores(features, cipher.block_bits)
    if statistic not in scores:
        available = ", ".join(sorted(scores))
        raise ValueError(f"unknown deterministic statistic {statistic!r}; available: {available}")

    statistic_scores = scores[statistic]
    baseline = _statistic_report(statistic_scores, labels)
    baseline["auc"] = _binary_auc(statistic_scores, labels)
    threshold = baseline["best_threshold"]
    return {
        "status": "pass",
        "audit": "integral_deterministic_baseline",
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
        "statistic_name": statistic,
        "baseline": baseline,
        "interpretation": _deterministic_statistic_interpretation(threshold["accuracy"]),
        "claim_scope": _claim_scope(),
    }


def _ciphertext_pair_feature_arrays(
    task: dict[str, Any],
    *,
    samples_per_class: int,
    seed: int | None,
    key_split: str,
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    if task["sample_structure"] not in {
        "plaintext_integral_nibble",
        "plaintext_integral_nibble_difference_matched_negative",
        "plaintext_integral_nibble_matched_negative",
        "plaintext_integral_multi_nibble_difference_matched_negative",
        "plaintext_integral_nibble_scrambled_positive",
    }:
        raise ValueError("integral audit requires a plaintext_integral_nibble sample structure")
    if task["feature_encoding"] != "ciphertext_pair_bits":
        raise ValueError("integral audit currently requires feature_encoding=ciphertext_pair_bits")

    key = task["validation_key"] if key_split == "validation" else task["train_key"]
    cipher = build_cipher(task["cipher_key"], task["rounds"], key=key)
    pair_bits = pair_bits_for_encoding(cipher.block_bits, task["feature_encoding"])

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
    return cipher, features, labels, pair_bits


def _integral_feature_bank_scores(features: np.ndarray, block_bits: int) -> dict[str, np.ndarray]:
    left = features[:, :, :block_bits]
    right = features[:, :, block_bits:]
    pair_xor = left ^ right

    left_hw = left.sum(axis=2)
    right_hw = right.sum(axis=2)
    pair_xor_hw = pair_xor.sum(axis=2)
    left_column_sum = left.sum(axis=1)
    right_column_sum = right.sum(axis=1)
    pair_xor_column_sum = pair_xor.sum(axis=1)

    return {
        "left_hw_mean": left_hw.mean(axis=1),
        "right_hw_mean": right_hw.mean(axis=1),
        "pair_xor_hw_mean": pair_xor_hw.mean(axis=1),
        "left_hw_std": left_hw.std(axis=1),
        "right_hw_std": right_hw.std(axis=1),
        "pair_xor_hw_std": pair_xor_hw.std(axis=1),
        "left_column_sum_variance": left_column_sum.var(axis=1),
        "right_column_sum_variance": right_column_sum.var(axis=1),
        "pair_xor_column_sum_variance": pair_xor_column_sum.var(axis=1),
        "left_right_column_sum_l1_mean": np.abs(left_column_sum - right_column_sum).mean(axis=1),
        "left_right_column_sum_l2_mean": ((left_column_sum - right_column_sum) ** 2).mean(axis=1),
    }


def _statistic_report(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    return {
        "positive": _score_distribution_summary(scores[labels]),
        "negative": _score_distribution_summary(scores[~labels]),
        "best_threshold": _best_threshold(scores, labels),
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


def _score_distribution_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "std": float(values.std()) if values.size else None,
    }


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    best = {"accuracy": -1.0, "threshold": None, "operator": None}
    for threshold in np.unique(scores):
        for operator in ("<=", ">="):
            predicted = scores <= threshold if operator == "<=" else scores >= threshold
            accuracy = float((predicted == labels).mean())
            if accuracy > best["accuracy"]:
                threshold_value = float(threshold)
                if threshold_value.is_integer():
                    threshold_value = int(threshold_value)
                best = {
                    "accuracy": accuracy,
                    "threshold": threshold_value,
                    "operator": operator,
                }
    return best


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[labels].sum())
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )
    return float(auc)


def _interpretation(accuracy: float) -> str:
    if accuracy >= 0.99:
        return "parity_statistic_alone_nearly_separates_classes"
    if accuracy >= 0.75:
        return "parity_statistic_is_strong_control_signal"
    if accuracy >= 0.6:
        return "parity_statistic_is_weak_control_signal"
    return "parity_statistic_does_not_explain_result_by_itself"


def _deterministic_statistic_interpretation(accuracy: float) -> str:
    if accuracy >= 0.99:
        return "deterministic_statistic_alone_nearly_separates_classes"
    if accuracy >= 0.75:
        return "deterministic_statistic_is_strong_control_signal"
    if accuracy >= 0.6:
        return "deterministic_statistic_is_weak_control_signal"
    return "deterministic_statistic_does_not_explain_result_by_itself"


def _claim_scope() -> str:
    return (
        "Deterministic local data-structure audit only; this is not a neural "
        "training result, not Zhang/Wang same-protocol evidence, and not a "
        "formal route claim."
    )


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
    audit_fns = {
        "alignment": integral_alignment_audit_from_task,
        "deterministic-baseline": integral_deterministic_baseline_from_task,
        "feature-bank": integral_feature_bank_audit_from_task,
        "parity": integral_parity_audit_from_task,
    }
    audit_fn = audit_fns[args.audit]
    kwargs: dict[str, Any] = {}
    if args.audit == "deterministic-baseline":
        kwargs["statistic"] = args.statistic
    report = audit_fn(
        tasks[args.row_index],
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        key_split=args.key_split,
        **kwargs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
