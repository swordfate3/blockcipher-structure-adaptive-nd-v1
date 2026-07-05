from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.evaluate_projection_ensemble import main as projection_ensemble_main
from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.projection_feature_postprocess import postprocess_projection_feature_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Advance a retrieved PRESENT truncated/projection feature result through "
            "postprocess, plotting, and optional gated ensemble diagnostics."
        )
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=4)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--run-ensemble", action="store_true")
    parser.add_argument("--ensemble-device", default="auto")
    parser.add_argument("--ensemble-epochs", type=int, default=1)
    parser.add_argument("--ensemble-batch-size", type=int, default=2048)
    parser.add_argument("--ensemble-hidden-bits", type=int, default=128)
    parser.add_argument("--ensemble-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args(argv)


def advance_projection_feature_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path,
    expected_rows: int,
    plan_doc_paths: list[Path] | None = None,
    skip_plot: bool = False,
    run_ensemble: bool = False,
    ensemble_device: str = "auto",
    ensemble_epochs: int = 1,
    ensemble_batch_size: int = 2048,
    ensemble_hidden_bits: int = 128,
    ensemble_output: Path | None = None,
    summary_output: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    postprocess = postprocess_projection_feature_result(
        results_path=results_path,
        output_dir=output_dir,
        run_id=run_id,
        plan_path=plan_path,
        expected_rows=expected_rows,
        plan_doc_paths=plan_doc_paths or [],
    )
    plot_report: dict[str, Any] | None = None
    if postprocess["status"] == "pass" and not skip_plot:
        curves_path = output_dir / f"{run_id}_curves.svg"
        history_path = output_dir / f"{run_id}_history.csv"
        plot_report = plot_jsonl_training_curves(results_path, curves_path, title=run_id)
        plot_report["history_csv"] = write_history_csv(results_path, history_path)

    ensemble_report_path: Path | None = None
    if (
        postprocess["status"] == "pass"
        and run_ensemble
        and postprocess["decision"] == "run_projection_ensemble_diagnostic"
    ):
        ensemble_report_path = ensemble_output or output_dir / f"{run_id}_projection_ensemble.json"
        projection_ensemble_main(
            [
                "--plan",
                str(plan_path),
                "--source-results",
                str(results_path),
                "--device",
                ensemble_device,
                "--epochs",
                str(ensemble_epochs),
                "--batch-size",
                str(ensemble_batch_size),
                "--hidden-bits",
                str(ensemble_hidden_bits),
                "--output",
                str(ensemble_report_path),
            ]
        )

    report = {
        "status": postprocess["status"],
        "run_id": run_id,
        "results": str(results_path),
        "output_dir": str(output_dir),
        "postprocess_summary": postprocess["summary"],
        "decision": postprocess["decision"],
        "next_action": postprocess["next_action"],
        "plot": plot_report,
        "ensemble_requested": run_ensemble,
        "ensemble_ran": ensemble_report_path is not None,
        "ensemble_output": str(ensemble_report_path) if ensemble_report_path is not None else None,
        "claim_scope": postprocess["claim_scope"],
    }
    summary_path = summary_output or output_dir / f"{run_id}_advance_summary.json"
    report["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = advance_projection_feature_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        plan_doc_paths=args.update_plan_doc,
        skip_plot=args.skip_plot,
        run_ensemble=args.run_ensemble,
        ensemble_device=args.ensemble_device,
        ensemble_epochs=args.ensemble_epochs,
        ensemble_batch_size=args.ensemble_batch_size,
        ensemble_hidden_bits=args.ensemble_hidden_bits,
        ensemble_output=args.ensemble_output,
        summary_output=args.summary_output,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
