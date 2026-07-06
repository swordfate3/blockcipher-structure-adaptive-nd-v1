from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.spn_feature_audit import _trail_position_stat_names


FAMILY_PREFIXES = (
    "depth_word_cell_mean",
    "depth_word_cell_std",
    "depth_word_cell_span",
    "depth_cell_first_last",
    "depth_cell_mean",
    "depth_cell_span",
    "depth_cell_std",
    "depth_word_first_last",
    "depth_word_mean",
    "depth_word_span",
    "depth_word_std",
    "word_cell_mean",
    "word_cell_std",
    "word_first_last",
    "word_mean",
    "word_span",
    "word_std",
    "cell_first_last",
    "cell_mean",
    "cell_span",
    "cell_std",
    "prefix_mean",
    "prefix_std",
    "global_pair_density",
    "global_trail_density",
    "global_low_cells",
    "global_mid_cells",
    "global_high_cells",
    "global_even_odd",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode sparse compressed SPN feature indices into trail-position statistic names."
    )
    parser.add_argument("--sparse-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def decode_compressed_feature_sparsity(sparse_report: dict[str, Any]) -> dict[str, Any]:
    metadata = _load_feature_metadata(Path(sparse_report["train_feature_dir"]))
    view_metadata = metadata.get("feature_view_metadata", metadata)
    names = _feature_names_from_metadata(view_metadata)
    expected_count = int(sparse_report.get("feature_count", view_metadata.get("output_feature_bits", 0)))
    if len(names) != expected_count:
        raise ValueError(f"expected {expected_count} feature names, got {len(names)}")

    rows = [_decode_row(row, names) for row in sparse_report.get("rows", [])]
    return {
        "status": "pass",
        "feature_count": len(names),
        "sparse_report_train_feature_dir": str(sparse_report["train_feature_dir"]),
        "rows": rows,
        "claim_scope": (
            "interpretation-only decoder for compressed SPN sparse feature audits; "
            "does not train, score, alter labels, or provide formal SPN/PRESENT evidence"
        ),
    }


def _load_feature_metadata(feature_dir: Path) -> dict[str, Any]:
    return json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))


def _feature_names_from_metadata(view_metadata: dict[str, Any]) -> list[str]:
    return _trail_position_stat_names(
        words_per_pair=int(view_metadata["words_per_pair"]),
        cells_per_word=16,
        trail_depth=int(view_metadata["trail_depth"]),
        trail_words_per_depth=int(view_metadata["trail_words_per_depth"]),
    )


def _decode_row(row: dict[str, Any], names: list[str]) -> dict[str, Any]:
    decoded = [_decode_feature(index, names[index]) for index in row["selected_feature_indices"]]
    family_counts = Counter(feature["family"] for feature in decoded)
    depth_counts = Counter(str(feature["depth"]) for feature in decoded if feature["depth"] is not None)
    cell_counts = Counter(str(feature["cell"]) for feature in decoded if feature["cell"] is not None)
    return {
        "top_k": int(row["top_k"]),
        "validation_auc": float(row.get("validation_metrics", {}).get("auc", 0.0)),
        "family_counts": dict(sorted(family_counts.items())),
        "depth_counts": dict(sorted(depth_counts.items())),
        "cell_counts": dict(sorted(cell_counts.items())),
        "decoded_features": decoded,
    }


def _decode_feature(index: int, name: str) -> dict[str, Any]:
    return {
        "index": int(index),
        "name": name,
        "family": _family_for_name(name),
        "word": _extract_int(name, "word"),
        "trailword": _extract_int(name, "trailword"),
        "depth": _extract_int(name, "depth"),
        "cell": _extract_int(name, "cell"),
    }


def _family_for_name(name: str) -> str:
    for prefix in FAMILY_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return name.split("_word", maxsplit=1)[0].split("_cell", maxsplit=1)[0]


def _extract_int(name: str, label: str) -> int | None:
    match = re.search(rf"{label}(\d+)", name)
    return int(match.group(1)) if match else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sparse_report = json.loads(args.sparse_report.read_text(encoding="utf-8"))
    decoded = decode_compressed_feature_sparsity(sparse_report)
    decoded["sparse_report"] = str(args.sparse_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decoded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decoded, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
