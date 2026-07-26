from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.tasks.innovation1.runtime_spn_present_rtg3b_seed1_launch import (
    RUN_ID,
    SEED1_REMOTE_CONFIG,
    build_runtime_spn_present_rtg3b_seed1_launch_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fail-closed PRESENT RTG3-B seed1 launch gate."
    )
    parser.add_argument("--seed0-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repository", default=Path("."), type=Path)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repository = args.repository.resolve()
    readiness = remote_readiness_report(repository / SEED1_REMOTE_CONFIG)
    gate = build_runtime_spn_present_rtg3b_seed1_launch_gate(
        seed0_root=args.seed0_root,
        repository=repository,
        source_commit=args.source_commit,
        readiness_status=str(readiness["status"]),
        remote=args.remote,
        branch=args.branch,
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if gate["should_ssh"] else "fail",
        "checks": {**gate["evidence_checks"], **gate["readiness_checks"]},
        "errors": [
            name
            for group in (gate["evidence_checks"], gate["readiness_checks"])
            for name, passed in group.items()
            if passed is not True
        ],
        "publication_checks": gate["publication_checks"],
    }
    summary = {
        "run_id": RUN_ID,
        "task": gate["task"],
        "training_performed": False,
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
            "run_id": RUN_ID,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        {
            "event": "run_done",
            "run_id": RUN_ID,
            "status": gate["status"],
            "decision": gate["decision"],
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "readiness.json", readiness)
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
    return 0 if gate["launch_authorized"] else 4


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
