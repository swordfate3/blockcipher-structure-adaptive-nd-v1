from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report
from blockcipher_nd.tasks.innovation1.uknit_r6_pair_amplification_k1bv_launch import (
    REMOTE_CONFIG,
    build_launch_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fail-closed K1-BV launch gate.")
    parser.add_argument("--model-readiness", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--remote-main-sha", required=True)
    parser.add_argument("--repository", default=Path("."), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    model_readiness = json.loads(args.model_readiness.read_text(encoding="utf-8"))
    remote_readiness = remote_readiness_report(args.repository / REMOTE_CONFIG)
    gate = build_launch_gate(
        repository=args.repository, source_commit=args.source_commit,
        remote_main_sha=args.remote_main_sha, model_readiness=model_readiness,
        remote_readiness_status=str(remote_readiness["status"]),
    )
    gate["remote_readiness"] = remote_readiness
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["launch_authorized"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
