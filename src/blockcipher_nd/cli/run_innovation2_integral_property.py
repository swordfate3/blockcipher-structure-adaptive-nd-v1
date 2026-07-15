from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.plots import (
    plot_jsonl_training_curves,
    write_history_csv,
)
from blockcipher_nd.tasks.innovation2.integral_property_calibration import (
    IntegralCalibrationConfig,
    run_integral_calibration_experiment,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import (
    IntegralExperimentConfig,
    run_integral_property_experiment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Innovation 2 PRESENT r5 structure-conditioned integral "
            "parity feasibility matrix."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-structures", type=int, required=True)
    parser.add_argument("--validation-structures", type=int, required=True)
    parser.add_argument("--test-structures", type=int, required=True)
    parser.add_argument("--calibration-structures", type=int)
    parser.add_argument("--train-keys", type=int, required=True)
    parser.add_argument("--validation-keys", type=int, required=True)
    parser.add_argument("--test-keys", type=int, required=True)
    parser.add_argument("--calibration-keys", type=int)
    parser.add_argument("--stability-test-keys", type=int)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-bits", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--gate-mode",
        choices=("smoke", "diagnostic", "calibration-smoke", "calibration"),
        default="diagnostic",
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
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    calibration_mode = args.gate_mode.startswith("calibration")
    if calibration_mode:
        missing = [
            name
            for name in (
                "calibration_structures",
                "calibration_keys",
                "stability_test_keys",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "calibration modes require "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        config = IntegralCalibrationConfig(
            run_id=args.run_id,
            train_structures=args.train_structures,
            validation_structures=args.validation_structures,
            calibration_structures=args.calibration_structures,
            test_structures=args.test_structures,
            train_keys=args.train_keys,
            validation_keys=args.validation_keys,
            calibration_keys=args.calibration_keys,
            test_keys=args.test_keys,
            stability_test_keys=args.stability_test_keys,
            epochs=args.epochs,
            seed=args.seed,
            rounds=args.rounds,
            batch_size=args.batch_size,
            hidden_bits=args.hidden_bits,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            gate_mode=args.gate_mode,
        )
    else:
        config = IntegralExperimentConfig(
            run_id=args.run_id,
            train_structures=args.train_structures,
            validation_structures=args.validation_structures,
            test_structures=args.test_structures,
            train_keys=args.train_keys,
            validation_keys=args.validation_keys,
            test_keys=args.test_keys,
            epochs=args.epochs,
            seed=args.seed,
            rounds=args.rounds,
            batch_size=args.batch_size,
            hidden_bits=args.hidden_bits,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            gate_mode=args.gate_mode,
        )
    progress_callback("run_start", {"run_id": args.run_id, "gate_mode": args.gate_mode})
    if calibration_mode:
        result = run_integral_calibration_experiment(
            config,
            progress_callback=progress_callback,
        )
    else:
        result = run_integral_property_experiment(
            config,
            progress_callback=progress_callback,
        )
    results_path = output_root / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_json(output_root / "dataset_summary.json", result["dataset_summary"])
    _write_json(output_root / "gate.json", result["gate"])
    _write_structure_rates(
        output_root / "structure_rates.csv",
        result["structure_rate_rows"],
    )
    if calibration_mode:
        _write_csv_rows(
            output_root / "observation_predictions.csv",
            result["observation_prediction_rows"],
        )
    plot_jsonl_training_curves(
        results_path,
        output_root / "curves.svg",
        title=_plot_title(args.gate_mode, args.seed),
    )
    write_history_csv(results_path, output_root / "history.csv")
    progress_callback(
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
        },
    )
    report = {
        "status": result["gate"]["status"],
        "decision": result["gate"]["decision"],
        "run_id": args.run_id,
        "output_root": str(output_root),
        "results": str(results_path),
        "gate": str(output_root / "gate.json"),
        "next_action": result["gate"]["next_action"],
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if result["gate"]["status"] != "fail" else 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_structure_rates(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv_rows(path, rows)


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires at least one row: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_title(gate_mode: str, seed: int) -> str:
    if gate_mode.startswith("calibration"):
        return (
            "创新2 E1：PRESENT 5轮积分平衡概率独立校准与标签稳定性"
            f"（{gate_mode}，seed {seed}）"
        )
    return (
        "创新2：PRESENT 5轮结构条件积分平衡概率预测"
        f"（{gate_mode}，seed {seed}）"
    )


if __name__ == "__main__":
    raise SystemExit(main())
