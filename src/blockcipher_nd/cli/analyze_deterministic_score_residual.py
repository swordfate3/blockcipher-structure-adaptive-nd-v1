from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.audit_integral_parity_signal import _integral_feature_bank_scores
from blockcipher_nd.data.cache import make_chunked_differential_dataset
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.evaluation.neural_ensemble import load_score_artifact
from blockcipher_nd.registry.cipher_factory import build_cipher, default_difference
from blockcipher_nd.registry.difference_profiles import difference_for_profile
from blockcipher_nd.training.metrics import best_threshold_accuracy_and_threshold, binary_auc


DEFAULT_STATISTIC = "pair_xor_column_sum_variance"
DEFAULT_BUCKETS = 16
DEFAULT_CACHE_ROOT = Path("/tmp/blockcipher_nd_deterministic_score_residual")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether a neural score artifact has residual signal after "
            "conditioning on a deterministic integral statistic."
        )
    )
    parser.add_argument("--score-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--statistic", default=DEFAULT_STATISTIC)
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument(
        "--dataset-cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=(
            "Local cache root used when the score artifact was generated from "
            "a disk-backed dataset cache."
        ),
    )
    return parser.parse_args(argv)


def analyze_deterministic_score_residual(
    *,
    score_artifact_dir: Path,
    statistic: str = DEFAULT_STATISTIC,
    buckets: int = DEFAULT_BUCKETS,
    dataset_cache_root: Path | None = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    artifact = load_score_artifact(score_artifact_dir)
    metadata = artifact.metadata
    deterministic_scores, deterministic_labels, deterministic_dataset = deterministic_scores_from_metadata(
        metadata,
        statistic=statistic,
        score_artifact_dir=score_artifact_dir,
        dataset_cache_root=dataset_cache_root,
    )
    labels = artifact.labels.astype(np.float32, copy=False)
    if labels.shape != deterministic_labels.shape:
        raise ValueError(
            "score artifact rows do not match deterministic dataset rows: "
            f"{labels.shape[0]} != {deterministic_labels.shape[0]}"
        )
    if not np.array_equal(labels, deterministic_labels.astype(np.float32, copy=False)):
        raise ValueError("score artifact labels do not match regenerated deterministic labels")

    neural_scores = artifact.probabilities.astype(np.float64, copy=False)
    deterministic_scores = deterministic_scores.astype(np.float64, copy=False)
    deterministic_report = _score_report(labels, deterministic_scores)
    neural_report = _score_report(labels, neural_scores)
    bucket_report = bucketed_residual_report(
        labels=labels,
        conditioning_scores=deterministic_scores,
        residual_scores=neural_scores,
        buckets=buckets,
    )
    delta_auc = float(neural_report["auc"] - deterministic_report["auc"])
    decision = (
        "support_neural_residual_after_deterministic_conditioning"
        if bucket_report["weighted_auc"] is not None
        and bucket_report["weighted_auc"] >= 0.75
        and delta_auc >= 0.01
        else "hold_neural_residual_after_deterministic_conditioning"
    )
    return {
        "status": "pass",
        "score_artifact": str(score_artifact_dir),
        "rows": int(labels.shape[0]),
        "statistic": statistic,
        "buckets": int(buckets),
        "metadata": {
            key: metadata.get(key)
            for key in (
                "cipher",
                "cipher_key",
                "rounds",
                "seed",
                "score_split",
                "score_samples_per_class",
                "validation_samples_per_class",
                "pairs_per_sample",
                "negative_mode",
                "sample_structure",
                "difference_profile",
                "difference_member",
                "validation_key",
                "model_key",
                "dataset_cache_enabled",
                "dataset_cache_chunk_size",
                "dataset_cache_workers",
            )
        },
        "deterministic_dataset": deterministic_dataset,
        "deterministic": deterministic_report,
        "neural": neural_report,
        "delta_neural_vs_deterministic_auc": delta_auc,
        "bucketed_residual": bucket_report,
        "decision": decision,
        "claim_scope": (
            "Frozen-score deterministic residual diagnostic only; this is not "
            "new neural training, not a formal PRESENT result, and not a remote "
            "launch gate by itself."
        ),
    }


def deterministic_scores_from_metadata(
    metadata: dict[str, Any],
    *,
    statistic: str,
    score_artifact_dir: Path | None = None,
    dataset_cache_root: Path | None = DEFAULT_CACHE_ROOT,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cipher_key = str(metadata["cipher_key"])
    rounds = int(metadata["rounds"])
    split = _score_split(metadata)
    key = metadata.get("validation_key") if split == "validation" else metadata.get("train_key")
    cipher = build_cipher(cipher_key, rounds, key=_optional_int(key))
    samples_per_class = _score_samples_per_class(metadata, split)
    seed = _score_seed(metadata, split)
    input_difference = _input_difference(metadata, cipher_key)
    config = DifferentialDatasetConfig(
        cipher=cipher,
        input_difference=input_difference,
        samples_per_class=samples_per_class,
        seed=seed,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=int(metadata["pairs_per_sample"]),
        negative_mode=str(metadata["negative_mode"]),
        key_rotation_interval=int(metadata.get("key_rotation_interval") or 0),
        sample_structure=str(metadata["sample_structure"]),
        integral_active_nibble=int(metadata.get("integral_active_nibble") or 0),
        selected_bit_indices=(),
    )
    dataset_info = {
        "split": split,
        "seed": seed,
        "samples_per_class": samples_per_class,
        "feature_encoding": config.feature_encoding,
        "label_order": "shuffled",
        "generation_mode": "in_memory",
    }
    if _metadata_bool(metadata.get("dataset_cache_enabled")):
        if dataset_cache_root is None:
            raise ValueError("dataset cache reconstruction requires --dataset-cache-root")
        chunk_size = int(metadata.get("dataset_cache_chunk_size") or 8192)
        workers = int(metadata.get("dataset_cache_workers") or 1)
        cache_dir = _local_cache_dir(
            dataset_cache_root,
            metadata=metadata,
            score_artifact_dir=score_artifact_dir,
            statistic=statistic,
            split=split,
        )
        dataset = make_chunked_differential_dataset(
            config,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
            workers=workers,
        )
        dataset_info.update(
            {
                "label_order": "positive_then_negative",
                "generation_mode": "disk_cache_semantics",
                "cache_dir": str(cache_dir),
                "cache_status": dataset.metadata.get("cache_status"),
                "chunk_size": chunk_size,
                "workers": workers,
                "physical_shuffle": dataset.metadata.get("physical_shuffle"),
            }
        )
    else:
        dataset = make_differential_dataset(config)
    features = dataset.features.reshape((-1, config.pairs_per_sample, cipher.block_bits * 2))
    scores = _integral_feature_bank_scores(features.astype(np.int16, copy=False), cipher.block_bits)
    if statistic not in scores:
        available = ", ".join(sorted(scores))
        raise ValueError(f"unknown deterministic statistic {statistic!r}; available: {available}")
    return (
        scores[statistic].astype(np.float64, copy=False),
        dataset.labels.astype(np.float32, copy=False),
        dataset_info,
    )


def bucketed_residual_report(
    *,
    labels: np.ndarray,
    conditioning_scores: np.ndarray,
    residual_scores: np.ndarray,
    buckets: int,
) -> dict[str, Any]:
    if buckets < 1:
        raise ValueError("buckets must be >= 1")
    order = np.argsort(conditioning_scores, kind="mergesort")
    bucket_rows: list[dict[str, Any]] = []
    weighted_auc_numerator = 0.0
    weighted_auc_denominator = 0.0
    for bucket_index, indices in enumerate(np.array_split(order, buckets)):
        if indices.size == 0:
            continue
        bucket_labels = labels[indices]
        positives = int((bucket_labels == 1).sum())
        negatives = int((bucket_labels == 0).sum())
        auc = binary_auc(bucket_labels, residual_scores[indices]) if positives and negatives else None
        pair_weight = positives * negatives
        if auc is not None and pair_weight > 0:
            weighted_auc_numerator += float(auc) * pair_weight
            weighted_auc_denominator += pair_weight
        bucket_rows.append(
            {
                "bucket": int(bucket_index),
                "rows": int(indices.size),
                "positives": positives,
                "negatives": negatives,
                "conditioning_min": float(conditioning_scores[indices].min()),
                "conditioning_max": float(conditioning_scores[indices].max()),
                "conditioning_mean": float(conditioning_scores[indices].mean()),
                "residual_auc": auc,
            }
        )
    weighted_auc = (
        float(weighted_auc_numerator / weighted_auc_denominator)
        if weighted_auc_denominator > 0.0
        else None
    )
    valid_bucket_aucs = [row["residual_auc"] for row in bucket_rows if row["residual_auc"] is not None]
    return {
        "mode": "equal_frequency_conditioning_score_buckets",
        "bucket_count": int(buckets),
        "valid_bucket_count": len(valid_bucket_aucs),
        "weighted_auc": weighted_auc,
        "mean_bucket_auc": float(np.mean(valid_bucket_aucs)) if valid_bucket_aucs else None,
        "min_bucket_auc": float(np.min(valid_bucket_aucs)) if valid_bucket_aucs else None,
        "max_bucket_auc": float(np.max(valid_bucket_aucs)) if valid_bucket_aucs else None,
        "buckets": bucket_rows,
    }


def _score_report(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    auc = binary_auc(labels, scores)
    best_accuracy, threshold = best_threshold_accuracy_and_threshold(labels, scores)
    return {
        "auc": float(auc),
        "oriented_auc": float(max(auc, 1.0 - auc)),
        "best_accuracy": float(best_accuracy),
        "best_threshold": float(threshold),
        "positive": _distribution(scores[labels == 1]),
        "negative": _distribution(scores[labels == 0]),
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else None,
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
    }


def _score_samples_per_class(metadata: dict[str, Any], split: str) -> int:
    if metadata.get("score_samples_per_class") not in {None, ""}:
        return int(metadata["score_samples_per_class"])
    if split == "validation" and metadata.get("validation_samples_per_class") not in {None, ""}:
        return int(metadata["validation_samples_per_class"])
    if split == "train" and metadata.get("train_samples_per_class") not in {None, ""}:
        return int(metadata["train_samples_per_class"])
    if split == "validation":
        return max(8, int(metadata["samples_per_class"]) // 2)
    return int(metadata["samples_per_class"])


def _score_split(metadata: dict[str, Any]) -> str:
    split = metadata.get("score_split")
    if split in {None, ""}:
        return "validation"
    return str(split)


def _score_seed(metadata: dict[str, Any], split: str) -> int:
    base_seed = int(metadata["seed"])
    return base_seed + 10_000 if split == "validation" else base_seed


def _input_difference(metadata: dict[str, Any], cipher_key: str) -> int:
    profile = str(metadata.get("difference_profile") or "")
    if not profile:
        return default_difference(cipher_key)
    return difference_for_profile(profile, int(metadata.get("difference_member") or 0))


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _local_cache_dir(
    root: Path,
    *,
    metadata: dict[str, Any],
    score_artifact_dir: Path | None,
    statistic: str,
    split: str,
) -> Path:
    identity = {
        "score_artifact_dir": str(score_artifact_dir) if score_artifact_dir is not None else "",
        "cipher_key": metadata.get("cipher_key"),
        "rounds": metadata.get("rounds"),
        "seed": metadata.get("seed"),
        "split": split,
        "samples_per_class": _score_samples_per_class(metadata, split),
        "pairs_per_sample": metadata.get("pairs_per_sample"),
        "negative_mode": metadata.get("negative_mode"),
        "sample_structure": metadata.get("sample_structure"),
        "difference_profile": metadata.get("difference_profile"),
        "difference_member": metadata.get("difference_member"),
        "validation_key": metadata.get("validation_key"),
        "train_key": metadata.get("train_key"),
        "statistic": statistic,
        "dataset_cache_chunk_size": metadata.get("dataset_cache_chunk_size"),
        "dataset_cache_workers": metadata.get("dataset_cache_workers"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]
    return root / digest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_deterministic_score_residual(
        score_artifact_dir=args.score_artifact,
        statistic=args.statistic,
        buckets=args.buckets,
        dataset_cache_root=args.dataset_cache_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
