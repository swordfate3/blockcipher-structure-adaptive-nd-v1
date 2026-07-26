from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

from blockcipher_nd.tasks.innovation1.runtime_spn_topology_only import (
    adjudicate_topology_only,
    load_and_validate_topology_only_config,
    run_topology_only,
    run_topology_only_readiness,
    write_topology_only_artifacts,
    write_topology_only_readiness_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run C1 topology-only Runtime-SPN readiness or its frozen "
            "two-seed Dialga holdout diagnostic."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    config = load_and_validate_topology_only_config(
        args.config,
        project_root=project_root,
        require_readiness=not args.readiness_only,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress(event: str, payload: dict[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": event, **payload, "time": time.time()},
                    sort_keys=True,
                )
                + "\n"
            )

    if args.readiness_only:
        progress("c1_readiness_start", {})
        readiness = run_topology_only_readiness(
            config=config,
            project_root=project_root,
        )
        write_topology_only_readiness_artifacts(
            readiness,
            output_root=args.output_root,
        )
        progress(
            "c1_readiness_done",
            {"status": readiness["status"], "decision": readiness["decision"]},
        )
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 0 if readiness["status"] == "pass" else 4

    progress("c1_start", {"run_id": config["run_id"]})
    payload = run_topology_only(
        config=config,
        config_path=args.config,
        output_root=args.output_root,
        project_root=project_root,
        progress_callback=progress,
    )
    gate = adjudicate_topology_only(payload)
    write_topology_only_artifacts(
        payload=payload,
        gate=gate,
        output_root=args.output_root,
    )
    progress(
        "c1_done",
        {
            "run_id": config["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if payload["validation"]["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
