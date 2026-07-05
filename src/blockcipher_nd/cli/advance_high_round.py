from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.arbitrate_next_actions import arbitrate_next_actions
from blockcipher_nd.cli.summarize_spn_evidence import HIGH_ROUND_RUNS, summarize_spn_evidence
from blockcipher_nd.planning.r8_pairset_1m_postprocess import postprocess_r8_pairset_1m_result
from blockcipher_nd.planning.r9_weak_probe_postprocess import postprocess_r9_weak_probe_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Advance the local high-round SPN/PRESENT evidence loop from retrieved "
            "artifacts only. This never SSH-polls or launches remote training."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/remote_results"),
        help="Local remote-result artifact root.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--arbitration-output",
        type=Path,
        default=Path("outputs/remote_results/high_round_next_action_arbitration.json"),
        help="Where to write arbitration when enough high-round summaries exist.",
    )
    parser.add_argument(
        "--no-update-plan-doc",
        action="store_true",
        help="Do not update docs/experiments during postprocess. Intended for tests/dry runs.",
    )
    return parser.parse_args(argv)


def advance_high_round(
    *,
    root: Path = Path("outputs/remote_results"),
    arbitration_output: Path = Path("outputs/remote_results/high_round_next_action_arbitration.json"),
    update_plan_docs: bool = True,
) -> dict[str, Any]:
    before = summarize_spn_evidence(root)
    active = before["active_recommendation"]
    postprocessed = _postprocess_ready_runs(active, root=root, update_plan_docs=update_plan_docs)
    after_postprocess = summarize_spn_evidence(root) if postprocessed else before
    arbitration = _arbitrate_if_ready(after_postprocess, arbitration_output=arbitration_output)
    status = _status(active, postprocessed=postprocessed, arbitration=arbitration)
    return {
        "status": status,
        "root": str(root),
        "initial_branch": active.get("branch"),
        "initial_status": active.get("status"),
        "postprocessed": postprocessed,
        "arbitration": arbitration,
        "active_recommendation": after_postprocess["active_recommendation"],
        "main_thread_policy": after_postprocess["active_recommendation"].get("main_thread_policy"),
        "remote_policy": "local_artifacts_only_no_ssh_no_remote_launch",
    }


def _postprocess_ready_runs(
    active: dict[str, Any],
    *,
    root: Path,
    update_plan_docs: bool,
) -> list[dict[str, Any]]:
    if active.get("branch") != "postprocess_high_round_result":
        return []
    reports = []
    for entry in active.get("ready_runs", []):
        if not isinstance(entry, dict):
            continue
        spec = _spec_for_run(str(entry.get("run_id") or ""))
        if spec is None:
            reports.append(
                {
                    "status": "fail",
                    "run_id": entry.get("run_id"),
                    "reason": "unknown_high_round_run",
                }
            )
            continue
        reports.append(_postprocess_run(spec, root=root, update_plan_docs=update_plan_docs))
    return reports


def _spec_for_run(run_id: str) -> dict[str, Any] | None:
    for spec in HIGH_ROUND_RUNS:
        if spec["run_id"] == run_id:
            return spec
    return None


def _postprocess_run(spec: dict[str, Any], *, root: Path, update_plan_docs: bool) -> dict[str, Any]:
    run_id = str(spec["run_id"])
    run_root = root / run_id
    results_path = run_root / "results" / f"{run_id}.jsonl"
    plan_path = Path(str(spec["plan"]))
    plan_doc_paths = [Path(str(spec["plan_doc"]))] if update_plan_docs else []
    kind = str(spec["postprocess_kind"])
    if kind == "r9_weak_probe":
        return postprocess_r9_weak_probe_result(
            results_path=results_path,
            output_dir=run_root,
            run_id=run_id,
            plan_path=plan_path,
            expected_rows=int(spec["expected_rows"]),
            plan_doc_paths=plan_doc_paths,
        )
    if kind == "r8_pairset_1m":
        return postprocess_r8_pairset_1m_result(
            results_path=results_path,
            output_dir=run_root,
            run_id=run_id,
            plan_path=plan_path,
            expected_rows=int(spec["expected_rows"]),
            plan_doc_paths=plan_doc_paths,
        )
    return {
        "status": "fail",
        "run_id": run_id,
        "reason": f"unsupported_high_round_postprocess_kind:{kind}",
    }


def _arbitrate_if_ready(report: dict[str, Any], *, arbitration_output: Path) -> dict[str, Any] | None:
    active = report["active_recommendation"]
    if active.get("branch") != "arbitrate_high_round_next_actions":
        return None
    summaries = [Path(str(path)) for path in active.get("summaries", [])]
    arbitration = arbitrate_next_actions(summaries)
    arbitration_output.parent.mkdir(parents=True, exist_ok=True)
    arbitration_output.write_text(json.dumps(arbitration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "written",
        "output": str(arbitration_output),
        "summary_count": len(summaries),
        "report": arbitration,
    }


def _status(
    active: dict[str, Any],
    *,
    postprocessed: list[dict[str, Any]],
    arbitration: dict[str, Any] | None,
) -> str:
    if arbitration is not None:
        return "arbitrated"
    if postprocessed:
        if any(report.get("status") != "pass" for report in postprocessed):
            return "postprocess_failed"
        return "postprocessed"
    if active.get("branch") == "wait_for_high_round_results":
        return "waiting"
    if active.get("branch") == "postprocess_high_round_result":
        return "postprocess_ready_but_no_runs_processed"
    return "no_high_round_action"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = advance_high_round(
        root=args.root,
        arbitration_output=args.arbitration_output,
        update_plan_docs=not args.no_update_plan_doc,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
