from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_present_rtg3b_launch import (
    build_runtime_spn_present_rtg3b_launch_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fail-closed PRESENT RTG3-B seed0 remote launch gate."
    )
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--t1-seed0-root", required=True, type=Path)
    parser.add_argument("--t1-seed1-root", required=True, type=Path)
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    gate = build_runtime_spn_present_rtg3b_launch_gate(
        c2_root=args.c2_root,
        t1_seed0_root=args.t1_seed0_root,
        t1_seed1_root=args.t1_seed1_root,
        readiness_report=args.readiness_report,
        repository=args.repository.resolve(),
        source_commit=args.source_commit,
        remote=args.remote,
        branch=args.branch,
    )
    validation = {
        "run_id": gate["run_id"],
        "status": "pass" if gate["should_ssh"] else "fail",
        "checks": {**gate["evidence_checks"], **gate["readiness_checks"]},
        "errors": [
            key
            for group in (gate["evidence_checks"], gate["readiness_checks"])
            for key, value in group.items()
            if value is not True
        ],
        "publication_checks": gate["publication_checks"],
    }
    summary = {
        "run_id": gate["run_id"],
        "status": gate["status"],
        "decision": gate["decision"],
        "source_commit": gate["source_commit"],
        "live_remote_sha": gate["live_remote_sha"],
        "should_ssh": gate["should_ssh"],
        "ssh_allowed": gate["ssh_allowed"],
        "launch_authorized": gate["launch_authorized"],
        "next_action": gate["next_action"],
        "claim_scope": gate["claim_scope"],
    }
    progress = [
        {
            "event": "run_start",
            "run_id": gate["run_id"],
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        {
            "event": "run_done",
            "run_id": gate["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    ]
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "summary.json", summary)
    (args.output_root / "results.jsonl").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "progress.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in progress
        ),
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["status"] in {"pass", "hold"} else 4


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
