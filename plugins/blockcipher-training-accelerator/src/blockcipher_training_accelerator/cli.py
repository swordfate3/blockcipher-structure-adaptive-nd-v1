from __future__ import annotations

import argparse
from pathlib import Path

from blockcipher_training_accelerator.benchmark import run_benchmark
from blockcipher_training_accelerator.matrix import split_matrix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Opt-in training speed utilities for blockcipher experiments."
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    bench = subparsers.add_parser(
        "bench-command",
        help="Run an existing command and write a JSON timing report.",
    )
    bench.add_argument("--label", required=True, help="Human-readable timing label.")
    bench.add_argument("--report", required=True, help="Output JSON timing report path.")
    bench.add_argument(
        "--cwd",
        default=".",
        help="Working directory for the measured command. Defaults to current directory.",
    )
    bench.add_argument(
        "--fail-on-nonzero",
        action="store_true",
        help="Return a failure if the measured command exits non-zero.",
    )
    bench.add_argument(
        "measured_command",
        nargs=argparse.REMAINDER,
        help="Command to measure. Put it after --.",
    )

    split = subparsers.add_parser(
        "split-matrix",
        help="Split a CSV experiment matrix into shard CSVs for independent GPU launches.",
    )
    split.add_argument("--plan", required=True, help="Input CSV experiment matrix.")
    split.add_argument("--shards", required=True, type=int, help="Number of output shards.")
    split.add_argument("--output-dir", required=True, help="Directory for shard CSVs.")
    split.add_argument(
        "--strategy",
        default="round-robin",
        choices=["round-robin", "contiguous"],
        help="Row assignment strategy.",
    )
    split.add_argument(
        "--prefix",
        default=None,
        help="Optional shard filename prefix. Defaults to the plan stem.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command_name == "bench-command":
        measured = list(args.measured_command)
        if measured and measured[0] == "--":
            measured = measured[1:]
        report = run_benchmark(
            measured,
            label=args.label,
            report_path=Path(args.report),
            cwd=Path(args.cwd),
            fail_on_nonzero=args.fail_on_nonzero,
        )
        print(
            f"{report.status}: {report.label} "
            f"duration_seconds={report.duration_seconds:.6f} returncode={report.returncode}",
            flush=True,
        )
        return
    if args.command_name == "split-matrix":
        result = split_matrix(
            plan_path=Path(args.plan),
            shards=args.shards,
            output_dir=Path(args.output_dir),
            strategy=args.strategy,
            prefix=args.prefix,
        )
        print(
            f"split {result.input_rows} rows into {len(result.shards)} shards; "
            f"manifest={result.manifest_path}",
            flush=True,
        )
        return
    raise AssertionError(f"unsupported command: {args.command_name}")
