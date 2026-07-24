from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.evaluation.plots import DEFAULT_METRICS, plot_jsonl_training_curves, write_history_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from experiment JSONL results.")
    parser.add_argument("--results", required=True, type=Path, help="Input JSONL produced by scripts/train.")
    parser.add_argument("--output", required=True, type=Path, help="Output SVG path.")
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=None,
        help="Optional CSV path for flattened per-epoch history records.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=["accuracy", "auc", "loss"],
        help="Metrics to plot. Each metric uses train_* and val_* history fields when present.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Plot validation curves only so close control results remain readable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plot_jsonl_training_curves(
        args.results,
        args.output,
        metrics=tuple(args.metrics),
        title=args.title,
        validation_only=args.validation_only,
    )
    if args.history_csv is not None:
        report["history_csv"] = write_history_csv(args.results, args.history_csv)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0
