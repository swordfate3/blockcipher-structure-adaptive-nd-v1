from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.tasks.innovation1.dialga_r4_dmc1_launch import (
    REMOTE_CONFIG,
    build_launch_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fail-closed Dialga DMC1 launch gate.")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--remote-main-sha", required=True)
    parser.add_argument("--repository", default=Path("."), type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    readiness = remote_readiness_report(args.repository / REMOTE_CONFIG)
    gate = build_launch_gate(
        repository=args.repository,
        source_commit=args.source_commit,
        remote_main_sha=args.remote_main_sha,
        readiness_status=str(readiness["status"]),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    payloads = (
        ("readiness.json", readiness),
        ("gate.json", gate),
        (
            "summary.json",
            {
                "status": gate["status"],
                "decision": gate["decision"],
                "should_ssh": gate["should_ssh"],
                "ssh_allowed": gate["ssh_allowed"],
                "launch_authorized": gate["launch_authorized"],
                "next_action": gate["next_action"],
                "claim_scope": gate["claim_scope"],
            },
        ),
    )
    for name, payload in payloads:
        (args.output_root / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["launch_authorized"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
