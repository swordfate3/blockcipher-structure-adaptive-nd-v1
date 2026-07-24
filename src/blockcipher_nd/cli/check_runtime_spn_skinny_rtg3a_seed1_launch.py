from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_rtg3a_seed1_launch import (
    RUN_ID,
    SEED1_REMOTE_CONFIG,
    build_runtime_spn_skinny_rtg3a_seed1_launch_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed RTG3-A seed1 publication gate."
    )
    parser.add_argument("--seed0-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--upstream-ref", default="origin/main")
    parser.add_argument("--repository", default=Path("."), type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    readiness = remote_readiness_report(args.repository / SEED1_REMOTE_CONFIG)
    gate = build_runtime_spn_skinny_rtg3a_seed1_launch_gate(
        seed0_root=args.seed0_root,
        repository=args.repository,
        source_commit=args.source_commit,
        readiness_status=str(readiness["status"]),
        upstream_ref=args.upstream_ref,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "readiness.json", readiness)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "task": gate["task"],
            "training_performed": False,
            "status": gate["status"],
            "decision": gate["decision"],
            "should_ssh": gate["should_ssh"],
            "ssh_allowed": gate["ssh_allowed"],
            "launch_authorized": gate["launch_authorized"],
            "claim_scope": gate["claim_scope"],
        },
    )
    with (args.output_root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "launch_gate_done",
                    "run_id": RUN_ID,
                    "status": gate["status"],
                    "decision": gate["decision"],
                    "should_ssh": gate["should_ssh"],
                    "ssh_allowed": gate["ssh_allowed"],
                    "time": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["launch_authorized"] else 4


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
