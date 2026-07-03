from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_training_accelerator.benchmark import run_benchmark
from blockcipher_training_accelerator.dataset_cache import (
    DatasetCacheBenchConfig,
    TrailFamilyCacheBenchConfig,
    parse_int,
    run_dataset_cache_benchmark,
    run_trail_family_cache_benchmark,
)
from blockcipher_training_accelerator.launcher import build_shard_commands
from blockcipher_training_accelerator.matrix import split_matrix
from blockcipher_training_accelerator.quality_gate import compare_result_files


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
    dataset_cache = subparsers.add_parser(
        "bench-dataset-cache",
        help="Benchmark chunked disk dataset cache generation for one protocol.",
    )
    dataset_cache.add_argument("--cipher", required=True)
    dataset_cache.add_argument("--rounds", required=True, type=int)
    dataset_cache.add_argument("--samples-per-class", required=True, type=int)
    dataset_cache.add_argument("--pairs-per-sample", required=True, type=int)
    dataset_cache.add_argument("--sample-structure", required=True)
    dataset_cache.add_argument("--negative-mode", required=True)
    dataset_cache.add_argument("--feature-encoding", default="ciphertext_pair_bits")
    dataset_cache.add_argument("--seed", required=True, type=int)
    dataset_cache.add_argument("--chunk-size", required=True, type=int)
    dataset_cache.add_argument("--workers", required=True, nargs="+", type=int)
    dataset_cache.add_argument("--output-root", required=True)
    dataset_cache.add_argument("--difference-profile", default=None)
    dataset_cache.add_argument("--difference-member", default=0, type=int)
    dataset_cache.add_argument("--input-difference", default=None, type=parse_int)
    dataset_cache.add_argument("--key", default=None, type=parse_int)
    dataset_cache.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse matching caches instead of deleting each worker cache before timing.",
    )
    trail_family_cache = subparsers.add_parser(
        "bench-trail-family-cache",
        help="Benchmark trail-family feature-cache generation without training.",
    )
    trail_family_cache.add_argument("--samples-per-class", required=True, type=int)
    trail_family_cache.add_argument("--pairs-per-sample", required=True, type=int)
    trail_family_cache.add_argument("--seed", required=True, type=int)
    trail_family_cache.add_argument("--chunk-size", required=True, type=int)
    trail_family_cache.add_argument("--workers", required=True, nargs="+", type=int)
    trail_family_cache.add_argument("--output-root", required=True)
    trail_family_cache.add_argument("--rounds", default=7, type=int)
    trail_family_cache.add_argument("--difference-profile", default="present_zhang_wang2022_mcnd")
    trail_family_cache.add_argument("--difference-member", default=0, type=int)
    trail_family_cache.add_argument("--input-difference", default=None, type=parse_int)
    trail_family_cache.add_argument("--key", default=0, type=parse_int)
    trail_family_cache.add_argument("--negative-mode", default="encrypted_random_plaintexts")
    trail_family_cache.add_argument("--sample-structure", default="zhang_wang_case2_official_mcnd")
    trail_family_cache.add_argument("--key-rotation-interval", default=0, type=int)
    trail_family_cache.add_argument("--beam-width", default=4, type=int)
    trail_family_cache.add_argument("--depth", default=3, type=int)
    trail_family_cache.add_argument("--false-family", action="store_true")
    trail_family_cache.add_argument("--reuse", action="store_true")

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
    launch = subparsers.add_parser(
        "build-launch-plan",
        help="Build per-shard train commands from a split-matrix manifest.",
    )
    launch.add_argument("--manifest", required=True, help="Manifest JSON from split-matrix.")
    launch.add_argument("--output-dir", required=True, help="Directory for shard outputs.")
    launch.add_argument(
        "--devices",
        nargs="+",
        required=True,
        help="Devices assigned round-robin, e.g. cuda:0 cuda:1.",
    )
    launch.add_argument("--python", default="python", help="Python executable.")
    launch.add_argument("--train-script", default="scripts/train", help="Train CLI script path.")
    launch.add_argument("--output", required=True, help="Launch plan JSON output path.")
    launch.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra train args appended after --.",
    )

    accelerated = subparsers.add_parser(
        "run-accelerated",
        help="Run the existing experiment matrix through the plugin accelerated trainer.",
    )
    accelerated.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to scripts/train plus --speed-profile.",
    )
    gate = subparsers.add_parser(
        "quality-gate",
        help="Compare baseline and accelerated JSONL files for protocol alignment and metric drift.",
    )
    gate.add_argument("--baseline", required=True, help="Baseline JSONL result file.")
    gate.add_argument("--candidate", required=True, help="Accelerated candidate JSONL result file.")
    gate.add_argument("--output", required=True, help="Quality gate report JSON path.")
    gate.add_argument("--max-auc-drop", type=float, default=0.002)
    gate.add_argument("--max-calibrated-accuracy-drop", type=float, default=0.002)
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
    if args.command_name == "bench-dataset-cache":
        report = run_dataset_cache_benchmark(
            DatasetCacheBenchConfig(
                cipher=args.cipher,
                rounds=args.rounds,
                samples_per_class=args.samples_per_class,
                pairs_per_sample=args.pairs_per_sample,
                sample_structure=args.sample_structure,
                negative_mode=args.negative_mode,
                feature_encoding=args.feature_encoding,
                seed=args.seed,
                chunk_size=args.chunk_size,
                workers=tuple(args.workers),
                output_root=args.output_root,
                difference_profile=args.difference_profile,
                difference_member=args.difference_member,
                input_difference=args.input_difference,
                key=args.key,
                reuse=args.reuse,
            )
        )
        for row in report.rows:
            print(
                f"workers={row.workers} duration_seconds={row.duration_seconds:.6f} "
                f"rows_per_second={row.rows_per_second:.3f} cache_status={row.cache_status}",
                flush=True,
            )
        print(f"dataset cache benchmark written: {report.summary_path}", flush=True)
        return
    if args.command_name == "bench-trail-family-cache":
        report = run_trail_family_cache_benchmark(
            TrailFamilyCacheBenchConfig(
                samples_per_class=args.samples_per_class,
                pairs_per_sample=args.pairs_per_sample,
                seed=args.seed,
                chunk_size=args.chunk_size,
                workers=tuple(args.workers),
                output_root=args.output_root,
                rounds=args.rounds,
                difference_profile=args.difference_profile,
                difference_member=args.difference_member,
                input_difference=args.input_difference,
                key=args.key,
                negative_mode=args.negative_mode,
                sample_structure=args.sample_structure,
                key_rotation_interval=args.key_rotation_interval,
                beam_width=args.beam_width,
                depth=args.depth,
                false_family=args.false_family,
                reuse=args.reuse,
            )
        )
        for row in report.rows:
            print(
                f"workers={row.workers} duration_seconds={row.duration_seconds:.6f} "
                f"rows_per_second={row.rows_per_second:.3f} cache_status={row.cache_status}",
                flush=True,
            )
        print(f"trail-family cache benchmark written: {report.summary_path}", flush=True)
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
    if args.command_name == "build-launch-plan":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        split_result = split_matrix_result_from_json(manifest)
        extra_args = list(args.extra_args)
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        launch_plan = build_shard_commands(
            split_result,
            output_dir=Path(args.output_dir),
            python_executable=args.python,
            train_script=args.train_script,
            devices=args.devices,
            extra_args=extra_args,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(launch_plan.to_json_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"launch plan written: {output}", flush=True)
        return
    if args.command_name == "run-accelerated":
        from blockcipher_training_accelerator.runner import run_accelerated_matrix

        train_args = list(args.train_args)
        if train_args and train_args[0] == "--":
            train_args = train_args[1:]
        rows = run_accelerated_matrix(train_args)
        print(f"accelerated rows written: {len(rows)}", flush=True)
        return
    if args.command_name == "quality-gate":
        report = compare_result_files(
            Path(args.baseline),
            Path(args.candidate),
            max_auc_drop=args.max_auc_drop,
            max_calibrated_accuracy_drop=args.max_calibrated_accuracy_drop,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.to_json_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{report.status}: rows_compared={report.rows_compared}", flush=True)
        return
    raise AssertionError(f"unsupported command: {args.command_name}")


def split_matrix_result_from_json(payload: dict[str, object]):
    from blockcipher_training_accelerator.matrix import MatrixShard, MatrixSplitResult

    return MatrixSplitResult(
        plan_path=str(payload["plan_path"]),
        output_dir=str(payload["output_dir"]),
        strategy=str(payload["strategy"]),
        input_rows=int(payload["input_rows"]),
        shards=[
            MatrixShard(
                index=int(shard["index"]),
                path=str(shard["path"]),
                rows=int(shard["rows"]),
            )
            for shard in payload["shards"]
        ],
        manifest_path=str(payload["manifest_path"]),
    )
