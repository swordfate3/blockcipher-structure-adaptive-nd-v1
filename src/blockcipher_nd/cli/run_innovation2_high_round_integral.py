from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.high_round_integral_experiment import (
    HighRoundIntegralExperimentConfig,
    run_cuda_memory_preflight,
    run_high_round_integral_experiment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Innovation 2 Wu/Guo-family PRESENT high-round integral "
            "multiset benchmark."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, choices=(5, 7, 8, 9), required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--validation-rows", type=int, required=True)
    parser.add_argument("--test-rows", type=int, required=True)
    parser.add_argument("--multiset-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--head-bits", type=int, default=64)
    parser.add_argument("--block-count", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--cache-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cuda-memory-preflight",
        action="store_true",
        help=(
            "Run one full-batch forward/backward/Adam step for each distinct "
            "model architecture before generating the dataset cache."
        ),
    )
    parser.add_argument(
        "--gate-mode",
        choices=("readiness", "diagnostic", "bridge", "paper_reference"),
        default="readiness",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    config = HighRoundIntegralExperimentConfig(
        run_id=args.run_id,
        output_root=output_root,
        cache_root=args.cache_root,
        rounds=args.rounds,
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        test_rows=args.test_rows,
        multiset_count=args.multiset_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        base_channels=args.base_channels,
        head_bits=args.head_bits,
        block_count=args.block_count,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        cache_chunk_size=args.cache_chunk_size,
        gate_mode=args.gate_mode,
    )
    progress_callback(
        "run_start",
        {
            "run_id": config.run_id,
            "rounds": config.rounds,
            "gate_mode": config.gate_mode,
            "train_total_rows": config.train_rows,
            "multisets_per_sample": config.multiset_count,
        },
    )
    if args.cuda_memory_preflight:
        memory_preflight = run_cuda_memory_preflight(config)
        _write_json(output_root / "memory_preflight.json", memory_preflight)
        progress_callback(
            "cuda_memory_preflight_done",
            {
                "status": memory_preflight["status"],
                "device": memory_preflight["device"],
                "batch_size": memory_preflight["batch_size"],
                "max_peak_reserved_bytes": memory_preflight[
                    "max_peak_reserved_bytes"
                ],
            },
        )
    result = run_high_round_integral_experiment(
        config,
        progress_callback=progress_callback,
    )
    results_path = output_root / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["rows"]),
        encoding="utf-8",
    )
    _write_json(output_root / "dataset_summary.json", result["dataset_summary"])
    _write_json(output_root / "fixed_baselines.json", result["fixed_baselines"])
    plot_report = write_training_artifacts(
        results_path,
        output_root,
        title=(
            f"创新2 H0：PRESENT {config.rounds}轮高轮积分神经锚点"
            f"（{config.gate_mode}，seed {config.seed}）"
        ),
    )
    _write_json(output_root / "gate.json", result["gate"])
    validation = validate_artifacts(output_root, expected_rows=4)
    _write_json(output_root / "validation.json", validation)
    gate = {
        **result["gate"],
        "artifact_validation": validation,
    }
    if validation["status"] != "pass":
        gate["status"] = "fail"
        gate["decision"] = "innovation2_high_round_integral_artifacts_invalid"
        gate["next_action"] = (
            "Repair the missing or malformed artifact before any diagnostic run."
        )
    _write_json(output_root / "gate.json", gate)
    progress_callback(
        "run_done",
        {
            "run_id": config.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    report = {
        "status": gate["status"],
        "decision": gate["decision"],
        "run_id": config.run_id,
        "output_root": str(output_root),
        "results": str(results_path),
        "gate": str(output_root / "gate.json"),
        "validation": str(output_root / "validation.json"),
        "plot_status": plot_report["status"],
        "next_action": gate["next_action"],
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if gate["status"] != "fail" else 1


def write_training_artifacts(
    results_path: Path,
    output_root: Path,
    *,
    title: str,
) -> dict[str, Any]:
    try:
        from blockcipher_nd.evaluation.plots import (
            plot_jsonl_training_curves,
            write_history_csv,
        )
    except ModuleNotFoundError as error:
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _write_deferred_svg(output_root / "curves.svg", missing_module=error.name)
        _write_deferred_history_csv(output_root / "history.csv", rows)
        marker = {
            "status": "deferred_to_local_retrieval",
            "reason": "optional_plot_dependency_missing",
            "missing_module": error.name,
        }
        _write_json(output_root / "plot_deferred.marker", marker)
        return marker

    plot_jsonl_training_curves(
        results_path,
        output_root / "curves.svg",
        title=title,
    )
    history_report = write_history_csv(results_path, output_root / "history.csv")
    return {
        "status": "rendered",
        "history_rows": history_report["rows"],
    }


def _write_deferred_svg(path: Path, *, missing_module: str | None) -> None:
    module = missing_module or "unknown"
    path.write_text(
        "\n".join(
            (
                (
                    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" '
                    'height="320" viewBox="0 0 1200 320">'
                ),
                '  <rect width="1200" height="320" fill="#ffffff"/>',
                (
                    '  <text x="60" y="120" font-family="sans-serif" '
                    'font-size="28" fill="#111827">Remote plotting deferred</text>'
                ),
                (
                    f'  <text x="60" y="175" font-family="sans-serif" '
                    f'font-size="20" fill="#475569">Missing optional module: '
                    f"{module}</text>"
                ),
                (
                    '  <text x="60" y="220" font-family="sans-serif" '
                    'font-size="20" fill="#475569">The local result watcher '
                    "will regenerate the full training curves.</text>"
                ),
                "</svg>",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_deferred_history_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_id", "role", "model", "history_points", "status"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row.get("run_id"),
                    "role": row.get("role"),
                    "model": row.get("model"),
                    "history_points": len(row.get("history", [])),
                    "status": "plot_deferred_to_local_retrieval",
                }
            )


def validate_artifacts(output_root: Path, *, expected_rows: int) -> dict[str, Any]:
    required = (
        "results.jsonl",
        "progress.jsonl",
        "dataset_summary.json",
        "fixed_baselines.json",
        "curves.svg",
        "history.csv",
        "gate.json",
    )
    missing = [name for name in required if not (output_root / name).is_file()]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not missing:
        try:
            rows = [
                json.loads(line)
                for line in (output_root / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"results_jsonl_invalid:{error}")
    if len(rows) != expected_rows:
        errors.append(f"result_rows:{len(rows)}!=expected:{expected_rows}")
    roles = {str(row.get("role")) for row in rows}
    if rows and roles != {"anchor", "candidate", "linear", "control"}:
        errors.append(f"unexpected_roles:{sorted(roles)}")
    if (output_root / "curves.svg").is_file():
        svg = (output_root / "curves.svg").read_text(encoding="utf-8")
        if "<svg" not in svg:
            errors.append("curves_svg_invalid")
    return {
        "status": "pass" if not missing and not errors else "fail",
        "expected_rows": expected_rows,
        "result_rows": len(rows),
        "missing": missing,
        "errors": errors,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
