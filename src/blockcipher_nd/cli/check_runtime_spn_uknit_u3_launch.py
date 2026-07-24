from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_uknit_u3_launch import (
    RUN_ID,
    U3_PLAN,
    build_runtime_spn_uknit_u3_launch_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed uKNIT U3 execution-authorization gate."
    )
    parser.add_argument("--seed0-root", required=True, type=Path)
    parser.add_argument("--rtg3-joint-root", type=Path, default=None)
    parser.add_argument("--readiness-root", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=U3_PLAN)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = build_runtime_spn_uknit_u3_launch_gate(
        seed0_root=args.seed0_root,
        rtg3_joint_root=args.rtg3_joint_root,
        readiness_root=args.readiness_root,
        plan=args.plan,
        repository=args.repository,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": RUN_ID,
            "task": gate["task"],
            "training_performed": False,
            "status": gate["status"],
            "decision": gate["decision"],
            "execution_authorized": gate["execution_authorized"],
            "claim_scope": gate["claim_scope"],
            "next_action": gate["next_action"],
        },
    )
    with (args.output_root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "u3_authorization_gate_done",
                    "run_id": RUN_ID,
                    "status": gate["status"],
                    "decision": gate["decision"],
                    "execution_authorized": gate["execution_authorized"],
                    "time": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["execution_authorized"] else 4


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
