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
    return parser.parse_args(argv)


def export_compressed_span_blocks(
    *,
    feature_dir: Path,
    output_dir: Path,
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
    for family in requested_families:
        indices = [index for index, name in enumerate(names) if _family_for_name(name) == family]
        block_shape = _block_shape(family, view_metadata)
        expected_count = int(np.prod(block_shape, dtype=np.int64))
        if len(indices) != expected_count:
            raise ValueError(f"{family} expected {expected_count} features, got {len(indices)}")
        values = features[:, indices].reshape((features.shape[0], *block_shape)).astype(np.float32, copy=False)
        np.save(output_dir / f"{family}.npy", values)
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_compressed_span_blocks(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        families=args.family or None,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
