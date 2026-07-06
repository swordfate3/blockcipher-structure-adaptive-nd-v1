from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from blockcipher_nd.cli.export_checkpoint_scores import eval_task_from_plan, validation_samples_per_class
from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.task_config import build_dataset_config, resolve_task_keys
from blockcipher_nd.evaluation.neural_ensemble import load_score_artifact
from blockcipher_nd.features.pair_features import pair_bits_for_encoding
from blockcipher_nd.models.structure.spn.present_trail_position_stats import (
    PresentTrailPositionStatsPairSetDistinguisher,
)
from blockcipher_nd.registry.cipher_factory import build_cipher

DEFAULT_FEATURE_VIEW = "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a plan-aligned feature matrix for bit-sensitivity projection screens."
    )
    parser.add_argument("--eval-plan", required=True, type=Path)
    parser.add_argument("--eval-row-index", type=int, default=0)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--reference-artifact", type=Path, default=None)
    parser.add_argument("--dataset-cache-root", type=Path, default=None)
    parser.add_argument("--dataset-cache-chunk-size", type=int, default=8192)
    parser.add_argument("--dataset-cache-workers", type=int, default=1)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--feature-view", choices=[DEFAULT_FEATURE_VIEW, "trail_position_stats"], default=DEFAULT_FEATURE_VIEW)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def export_bit_sensitivity_features(
    *,
    eval_plan: Path,
    eval_row_index: int,
    split: str,
    samples_per_class: int | None,
    output_dir: Path,
    reference_artifact_dir: Path | None = None,
    dataset_cache_root: Path | None = None,
    dataset_cache_chunk_size: int = 8192,
    dataset_cache_workers: int = 1,
    progress_output: Path | None = None,
    feature_view: str = DEFAULT_FEATURE_VIEW,
) -> dict[str, Any]:
    task = eval_task_from_plan(
        SimpleNamespace(
            eval_plan=eval_plan,
            eval_row_index=eval_row_index,
        )
    )
    train_key, validation_key = resolve_task_keys(task)
    if split == "train":
        cipher_key = train_key
        seed = int(task["seed"])
        resolved_samples_per_class = int(samples_per_class or task["samples_per_class"])
    elif split == "validation":
        cipher_key = validation_key
        seed = int(task["seed"]) + 10_000
        resolved_samples_per_class = validation_samples_per_class(task, samples_per_class)
    else:
        raise ValueError(f"unsupported split: {split}")

    cipher = build_cipher(task["cipher_key"], task["rounds"], key=cipher_key)
    dataset_args = SimpleNamespace(
        dataset_cache_root=dataset_cache_root,
        dataset_cache_chunk_size=dataset_cache_chunk_size,
        dataset_cache_workers=dataset_cache_workers,
    )
    dataset = make_task_dataset(
        build_dataset_config(
            task,
            cipher=cipher,
            samples_per_class=resolved_samples_per_class,
            seed=seed,
        ),
        dataset_args,
        task,
        split=split,
        progress_path=str(progress_output) if progress_output is not None else None,
        index=eval_row_index + 1,
        total=None,
    )
    output_features, view_metadata = _feature_view_matrix(
        dataset.features.astype(np.float32, copy=False),
        task=task,
        cipher_block_bits=cipher.block_bits,
        feature_view=feature_view,
    )
    sample_ids = np.array([str(index) for index in range(len(dataset.labels))], dtype=str)
    metadata = {
        "status": "pass",
        "kind": "bit_sensitivity_feature_matrix",
        "split": split,
        "feature_view": feature_view,
        "feature_view_metadata": view_metadata,
        "eval_plan": str(eval_plan),
        "eval_row_index": int(eval_row_index),
        "cipher": cipher.name,
        "cipher_key": task["cipher_key"],
        "rounds": int(task["rounds"]),
        "seed": int(seed),
        "task_seed": int(task["seed"]),
        "samples_per_class": int(resolved_samples_per_class),
        "total_rows": int(len(dataset.labels)),
        "input_bits": int(dataset.features.shape[1]),
        "output_feature_bits": int(output_features.shape[1]),
        "pairs_per_sample": int(task["pairs_per_sample"]),
        "feature_encoding": task["feature_encoding"],
        "negative_mode": task["negative_mode"],
        "sample_structure": task["sample_structure"],
        "difference_profile": task.get("difference_profile", ""),
        "difference_member": task.get("difference_member", ""),
        "train_key": task.get("train_key"),
        "validation_key": task.get("validation_key"),
        "dataset_cache_enabled": bool(dataset_cache_root),
        "dataset_cache_root": str(dataset_cache_root) if dataset_cache_root is not None else None,
        "alignment": {"reference_checked": False},
        "claim_scope": (
            "feature matrix export for local bit-sensitivity projection screens only; "
            "not a model result, not remote evidence, and not formal SPN/PRESENT evidence"
        ),
    }
    if reference_artifact_dir is not None:
        reference = load_score_artifact(reference_artifact_dir)
        alignment = _reference_alignment(reference, dataset.labels, sample_ids, metadata)
        metadata["alignment"] = alignment
        if alignment["status"] != "pass":
            return {
                "status": "fail",
                "decision": "feature_reference_alignment_failed",
                "output_dir": str(output_dir),
                "errors": alignment["errors"],
                "metadata": metadata,
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", output_features.astype(np.float32, copy=False))
    np.save(output_dir / "labels.npy", dataset.labels.astype(np.float32, copy=False))
    np.save(output_dir / "sample_ids.npy", sample_ids)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "decision": "bit_sensitivity_features_ready",
        "output_dir": str(output_dir),
        "features": str(output_dir / "features.npy"),
        "labels": str(output_dir / "labels.npy"),
        "sample_ids": str(output_dir / "sample_ids.npy"),
        "metadata": str(output_dir / "metadata.json"),
        "rows": int(len(dataset.labels)),
        "input_bits": int(dataset.features.shape[1]),
        "output_feature_bits": int(output_features.shape[1]),
        "split": split,
        "alignment": metadata["alignment"],
        "claim_scope": metadata["claim_scope"],
    }


def _feature_view_matrix(
    features: np.ndarray,
    *,
    task: dict[str, Any],
    cipher_block_bits: int,
    feature_view: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if feature_view == DEFAULT_FEATURE_VIEW:
        return features, {
            "view": DEFAULT_FEATURE_VIEW,
            "input_bits": int(features.shape[1]),
            "output_feature_bits": int(features.shape[1]),
        }
    if feature_view != "trail_position_stats":
        raise ValueError(f"unsupported feature_view: {feature_view}")

    pair_bits = pair_bits_for_encoding(cipher_block_bits, task["feature_encoding"])
    if features.shape[1] % pair_bits != 0:
        raise ValueError("trail_position_stats feature view requires pair-aligned features")
    options = _model_options(task)
    model = PresentTrailPositionStatsPairSetDistinguisher(
        input_bits=int(features.shape[1]),
        pair_bits=pair_bits,
        base_channels=int(task.get("hidden_bits", 32)),
        trail_depth=int(options.get("trail_depth", 4)),
        trail_words_per_depth=int(options.get("trail_words_per_depth", 9)),
        stats_hidden_bits=int(options["stats_hidden_bits"]) if options.get("stats_hidden_bits") is not None else None,
    )
    with torch.no_grad():
        stat_matrix = model._position_statistics(torch.from_numpy(features)).cpu().numpy()
    stat_matrix = stat_matrix.astype(np.float32, copy=False)
    return stat_matrix, {
        "view": "trail_position_stats",
        "source_input_bits": int(features.shape[1]),
        "pair_bits": int(pair_bits),
        "pairs_per_sample": int(features.shape[1] // pair_bits),
        "words_per_pair": int(pair_bits // cipher_block_bits),
        "trail_depth": int(model.trail_depth),
        "trail_words_per_depth": int(model.trail_words_per_depth),
        "prefix_words": int(model.prefix_words),
        "output_feature_bits": int(stat_matrix.shape[1]),
        "claim_scope": (
            "deterministic trail-position structural statistics for local residual projection screens; "
            "not trained features and not remote evidence"
        ),
    }


def _model_options(task: dict[str, Any]) -> dict[str, Any]:
    raw = task.get("model_options") or {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _reference_alignment(
    reference: Any,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    labels_match = np.array_equal(reference.labels.astype(np.float32, copy=False), labels.astype(np.float32))
    sample_ids_match = np.array_equal(reference.sample_ids.astype(str, copy=False), sample_ids.astype(str))
    if not labels_match:
        errors.append("labels_mismatch")
    if not sample_ids_match:
        errors.append("sample_ids_mismatch")
    for field in (
        "cipher_key",
        "rounds",
        "pairs_per_sample",
        "feature_encoding",
        "negative_mode",
        "sample_structure",
        "difference_profile",
        "difference_member",
        "validation_key",
    ):
        if field in reference.metadata and reference.metadata.get(field) != metadata.get(field):
            errors.append(f"metadata_mismatch:{field}")
    if "validation_samples_per_class" in reference.metadata and int(
        reference.metadata["validation_samples_per_class"]
    ) != int(metadata["samples_per_class"]):
        errors.append("metadata_mismatch:validation_samples_per_class")
    return {
        "status": "pass" if not errors else "fail",
        "reference_checked": True,
        "labels": labels_match,
        "sample_ids": sample_ids_match,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = export_bit_sensitivity_features(
        eval_plan=args.eval_plan,
        eval_row_index=args.eval_row_index,
        split=args.split,
        samples_per_class=args.samples_per_class,
        output_dir=args.output_dir,
        reference_artifact_dir=args.reference_artifact,
        dataset_cache_root=args.dataset_cache_root,
        dataset_cache_chunk_size=args.dataset_cache_chunk_size,
        dataset_cache_workers=args.dataset_cache_workers,
        progress_output=args.progress_output,
        feature_view=args.feature_view,
    )
    if report["status"] != "pass":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metadata.json").write_text(
            json.dumps(report["metadata"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
