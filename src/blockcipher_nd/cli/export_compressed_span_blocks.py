from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.cli.decode_compressed_feature_sparsity import _family_for_name, _feature_names_from_metadata


SPAN_FAMILY_SHAPES = {
    "depth_word_cell_span": ("trail_depth", "trail_words_per_depth", "cells_per_word"),
    "depth_cell_span": ("trail_depth", "cells_per_word"),
    "word_span": ("words_per_pair",),
    "depth_word_span": ("trail_depth", "trail_words_per_depth"),
    "cell_span": ("cells_per_word",),
}
DEFAULT_SPAN_FAMILIES = tuple(SPAN_FAMILY_SHAPES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compressed PRESENT trail-position span families as structured SPN blocks."
    )
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        choices=DEFAULT_SPAN_FAMILIES,
        help="Span feature family to export; repeat for multiple families. Defaults to all span families.",
    )
    parser.add_argument(
        "--output-summary-feature-dir",
        type=Path,
        default=None,
        help="Optional feature artifact containing compact pooled SPN span-block summary features.",
    )
    return parser.parse_args(argv)


def export_compressed_span_blocks(
    *,
    feature_dir: Path,
    output_dir: Path,
    output_summary_feature_dir: Path | None = None,
    families: list[str] | None = None,
) -> dict[str, Any]:
    feature_artifact = _load_feature_dir(feature_dir)
    metadata = feature_artifact["metadata"]
    view_metadata = metadata.get("feature_view_metadata", metadata)
    names = _feature_names_from_metadata(view_metadata)
    features = feature_artifact["features"]
    if len(names) != features.shape[1]:
        raise ValueError(f"expected {features.shape[1]} feature names, got {len(names)}")

    requested_families = tuple(families or DEFAULT_SPAN_FAMILIES)
    unknown = sorted(set(requested_families) - set(DEFAULT_SPAN_FAMILIES))
    if unknown:
        raise ValueError(f"unsupported span families: {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "labels.npy", feature_artifact["labels"].astype(np.float32, copy=False))
    np.save(output_dir / "sample_ids.npy", feature_artifact["sample_ids"])

    blocks: dict[str, Any] = {}
    block_arrays: dict[str, np.ndarray] = {}
    for family in requested_families:
        indices = [index for index, name in enumerate(names) if _family_for_name(name) == family]
        block_shape = _block_shape(family, view_metadata)
        expected_count = int(np.prod(block_shape, dtype=np.int64))
        if len(indices) != expected_count:
            raise ValueError(f"{family} expected {expected_count} features, got {len(indices)}")
        values = features[:, indices].reshape((features.shape[0], *block_shape)).astype(np.float32, copy=False)
        np.save(output_dir / f"{family}.npy", values)
        block_arrays[family] = values
        blocks[family] = {
            "path": f"{family}.npy",
            "shape": [int(dim) for dim in values.shape],
            "feature_count": int(len(indices)),
            "source_feature_indices": [int(index) for index in indices],
            "axes": ["sample", *_axis_names(family)],
            "role": "primary_backbone" if family == "depth_word_cell_span" else "auxiliary_context",
        }

    manifest = {
        "status": "pass",
        "kind": "compressed_spn_span_blocks",
        "feature_dir": str(feature_dir),
        "split": metadata.get("split"),
        "source_feature_count": int(features.shape[1]),
        "row_count": int(features.shape[0]),
        "families": list(requested_families),
        "blocks": blocks,
        "metadata": metadata,
        "claim_scope": (
            "structure-preserving export of already-generated compressed SPN span features; "
            "does not train, score, alter labels, alter negatives, or provide formal SPN/PRESENT evidence"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if output_summary_feature_dir is not None:
        summary_path = _write_summary_feature_dir(
            output_summary_feature_dir,
            source_metadata=metadata,
            labels=feature_artifact["labels"],
            sample_ids=feature_artifact["sample_ids"],
            blocks=block_arrays,
        )
        manifest["summary_feature_dir"] = str(summary_path)
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def _load_feature_dir(path: Path) -> dict[str, Any]:
    features = np.load(path / "features.npy")
    labels = np.load(path / "labels.npy")
    sample_ids = np.load(path / "sample_ids.npy").astype(str, copy=False)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if features.ndim < 2:
        raise ValueError("features must have at least two dimensions: rows x features")
    feature_matrix = features.reshape(features.shape[0], -1).astype(np.float32, copy=False)
    if len({len(feature_matrix), len(labels), len(sample_ids)}) != 1:
        raise ValueError("feature rows, labels, and sample_ids must have equal length")
    return {
        "features": feature_matrix,
        "labels": labels,
        "sample_ids": sample_ids,
        "metadata": metadata,
    }


def _block_shape(family: str, view_metadata: dict[str, Any]) -> tuple[int, ...]:
    dimensions = {
        "words_per_pair": int(view_metadata["words_per_pair"]),
        "trail_depth": int(view_metadata["trail_depth"]),
        "trail_words_per_depth": int(view_metadata["trail_words_per_depth"]),
        "cells_per_word": int(view_metadata.get("cells_per_word", 16)),
    }
    return tuple(dimensions[name] for name in SPAN_FAMILY_SHAPES[family])


def _axis_names(family: str) -> list[str]:
    names = []
    for name in SPAN_FAMILY_SHAPES[family]:
        if name == "words_per_pair":
            names.append("word")
        elif name == "trail_words_per_depth":
            names.append("trailword")
        elif name == "cells_per_word":
            names.append("cell")
        else:
            names.append(name.removeprefix("trail_"))
    return names


def _write_summary_feature_dir(
    path: Path,
    *,
    source_metadata: dict[str, Any],
    labels: np.ndarray,
    sample_ids: np.ndarray,
    blocks: dict[str, np.ndarray],
) -> Path:
    features, feature_names = _summary_features(blocks)
    metadata = {
        **source_metadata,
        "kind": "bit_sensitivity_feature_matrix",
        "feature_view": "compressed_span_summary",
        "output_feature_bits": int(features.shape[1]),
        "feature_view_metadata": {
            "view": "compressed_span_summary",
            "source_kind": "compressed_spn_span_blocks",
            "source_feature_view": source_metadata.get("feature_view"),
            "source_feature_view_metadata": source_metadata.get("feature_view_metadata", {}),
            "source_families": list(blocks),
            "output_feature_bits": int(features.shape[1]),
            "feature_names": feature_names,
            "claim_scope": (
                "compact pooled SPN span-block summary features for local diagnostics; "
                "not trained features, not remote evidence, and not formal SPN/PRESENT evidence"
            ),
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "features.npy", features.astype(np.float32, copy=False))
    np.save(path / "labels.npy", labels.astype(np.float32, copy=False))
    np.save(path / "sample_ids.npy", sample_ids)
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _summary_features(blocks: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    names: list[str] = []

    if "depth_word_cell_span" in blocks:
        block = blocks["depth_word_cell_span"]
        _append_axis_pool(arrays, names, block.mean(axis=(2, 3)), "primary_depth_mean", "depth")
        _append_axis_pool(arrays, names, block.mean(axis=(1, 3)), "primary_trailword_mean", "trailword")
        _append_axis_pool(arrays, names, block.mean(axis=(1, 2)), "primary_cell_mean", "cell")
        _append_matrix_pool(arrays, names, block.mean(axis=2), "primary_depth_cell_mean", "depth", "cell")
        _append_matrix_pool(
            arrays,
            names,
            block.mean(axis=3),
            "primary_depth_trailword_mean",
            "depth",
            "trailword",
        )
        _append_global_stats(arrays, names, block, "primary")

    if "depth_cell_span" in blocks:
        block = blocks["depth_cell_span"]
        _append_axis_pool(arrays, names, block.mean(axis=2), "aux_depth_cell_depth_mean", "depth")
        _append_axis_pool(arrays, names, block.mean(axis=1), "aux_depth_cell_cell_mean", "cell")
        _append_global_stats(arrays, names, block, "aux_depth_cell")

    if "word_span" in blocks:
        block = blocks["word_span"]
        _append_axis_pool(arrays, names, block, "aux_word_mean", "word")
        _append_global_stats(arrays, names, block, "aux_word")

    if "depth_word_span" in blocks:
        block = blocks["depth_word_span"]
        _append_axis_pool(arrays, names, block.mean(axis=2), "aux_depth_word_depth_mean", "depth")
        _append_axis_pool(arrays, names, block.mean(axis=1), "aux_depth_word_trailword_mean", "trailword")
        _append_matrix_pool(
            arrays,
            names,
            block,
            "aux_depth_word_depth_trailword_mean",
            "depth",
            "trailword",
        )
        _append_global_stats(arrays, names, block, "aux_depth_word")

    if "cell_span" in blocks:
        block = blocks["cell_span"]
        _append_axis_pool(arrays, names, block, "aux_cell_mean", "cell")
        _append_global_stats(arrays, names, block, "aux_cell")

    if not arrays:
        raise ValueError("no span blocks available for summary features")
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False), names


def _append_axis_pool(
    arrays: list[np.ndarray],
    names: list[str],
    values: np.ndarray,
    prefix: str,
    axis_name: str,
) -> None:
    matrix = values.reshape(values.shape[0], -1)
    arrays.append(matrix.astype(np.float32, copy=False))
    names.extend(f"{prefix}_{axis_name}{index}" for index in range(matrix.shape[1]))


def _append_matrix_pool(
    arrays: list[np.ndarray],
    names: list[str],
    values: np.ndarray,
    prefix: str,
    axis0_name: str,
    axis1_name: str,
) -> None:
    matrix = values.reshape(values.shape[0], -1)
    arrays.append(matrix.astype(np.float32, copy=False))
    for index0 in range(values.shape[1]):
        for index1 in range(values.shape[2]):
            names.append(f"{prefix}_{axis0_name}{index0}_{axis1_name}{index1}")


def _append_global_stats(
    arrays: list[np.ndarray],
    names: list[str],
    values: np.ndarray,
    prefix: str,
) -> None:
    matrix = values.reshape(values.shape[0], -1)
    stats = np.stack(
        [
            matrix.mean(axis=1),
            matrix.std(axis=1),
            matrix.min(axis=1),
            matrix.max(axis=1),
        ],
        axis=1,
    )
    arrays.append(stats.astype(np.float32, copy=False))
    names.extend([f"{prefix}_global_mean", f"{prefix}_global_std", f"{prefix}_global_min", f"{prefix}_global_max"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_compressed_span_blocks(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        output_summary_feature_dir=args.output_summary_feature_dir,
        families=args.family or None,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
