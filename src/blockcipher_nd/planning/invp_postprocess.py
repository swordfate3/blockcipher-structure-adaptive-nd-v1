from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv
from blockcipher_nd.planning.invp_gate import gate_invp_only_result
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local post-retrieval validation, plotting, and branch gating for InvP-only 1M results."
    )
    parser.add_argument("--plan", required=True, type=Path, help="Plan CSV path.")
    parser.add_argument("--results", required=True, type=Path, help="Retrieved result JSONL path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated reports.")
    parser.add_argument("--run-id", required=True, help="Run id used in output filenames and plot title.")
    parser.add_argument("--expected-rows", type=int, default=1)
    parser.add_argument("--reference-auc", type=float, default=0.793897025948)
    return parser.parse_args(argv)


def postprocess_invp_only_result(
    *,
    plan_path: Path,
    results_path: Path,
    output_dir: Path,
    run_id: str,
    expected_rows: int = 1,
    reference_auc: float = 0.793897025948,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report = validate_result_plan_alignment(
        plan_path,
        results_path,
        expected_rows=expected_rows,
    )
    validation_path = output_dir / f"{run_id}_local_result_gate.json"
    _write_json(validation_path, validation_report)

    curves_path = output_dir / f"{run_id}_curves.svg"
    history_path = output_dir / f"{run_id}_history.csv"
    plot_report = plot_jsonl_training_curves(results_path, curves_path, title=run_id)
    plot_report["history_csv"] = write_history_csv(results_path, history_path)

    branch_report = gate_invp_only_result(
        results_path,
        reference_auc=reference_auc,
        expected_rows=expected_rows,
    )
    branch_path = output_dir / f"{run_id}_branch_gate.json"
    _write_json(branch_path, branch_report)

    status = "pass" if validation_report["status"] == "pass" and branch_report["status"] == "pass" else "fail"
    report = {
        "status": status,
        "run_id": run_id,
        "plan": str(plan_path),
        "results": str(results_path),
        "output_dir": str(output_dir),
        "validation_report": str(validation_path),
        "curves": str(curves_path),
        "history_csv": str(history_path),
        "branch_gate": str(branch_path),
        "validation_status": validation_report["status"],
        "branch_status": branch_report["status"],
        "decision": branch_report["decision"],
        "action": branch_report["action"],
        "reference_auc": branch_report["reference_auc"],
        "paligned_mcnd_1m_auc": branch_report["paligned_mcnd_1m_auc"],
        "auc": branch_report["auc"],
        "auc_delta": branch_report["auc_delta"],
        "auc_delta_vs_paligned_mcnd_1m": branch_report["auc_delta_vs_paligned_mcnd_1m"],
        "claim_scope": branch_report["claim_scope"],
    }
    summary_path = output_dir / f"{run_id}_postprocess_summary.json"
    _write_json(summary_path, report)
    report["summary"] = str(summary_path)
    return report


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = postprocess_invp_only_result(
        plan_path=args.plan,
        results_path=args.results,
        output_dir=args.output_dir,
        run_id=args.run_id,
        expected_rows=args.expected_rows,
        reference_auc=args.reference_auc,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
