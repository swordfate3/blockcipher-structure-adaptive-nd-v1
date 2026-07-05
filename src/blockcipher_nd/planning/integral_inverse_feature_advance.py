from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.integral_inverse_feature_postprocess import (
    postprocess_integral_inverse_feature_result,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Advance a retrieved PRESENT r8 integral/inverse feature screen "
            "through postprocess, plotting, and summary generation."
        )
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=3)
    parser.add_argument("--update-plan-doc", type=Path, action="append", default=[])
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args(argv)


def advance_integral_inverse_feature_result(
    *,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    plan_path: Path | None,
    expected_rows: int,
    plan_doc_paths: list[Path] | None = None,
    skip_plot: bool = False,
    summary_output: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    postprocess = postprocess_integral_inverse_feature_result(
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

    report = {
        "status": postprocess["status"],
        "run_id": run_id,
        "results": str(results_path),
        "output_dir": str(output_dir),
        "postprocess_summary": postprocess["summary"],
        "decision": postprocess["decision"],
        "next_action": postprocess["next_action"],
        "plot": plot_report,
        "claim_scope": postprocess["claim_scope"],
    }
    summary_path = summary_output or output_dir / f"{run_id}_advance_summary.json"
    report["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = advance_integral_inverse_feature_result(
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        plan_path=args.plan,
        expected_rows=args.expected_rows,
        plan_doc_paths=args.update_plan_doc,
        skip_plot=args.skip_plot,
        summary_output=args.summary_output,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
