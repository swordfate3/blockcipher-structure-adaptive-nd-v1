from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_multiscale_orbit_alignment import (
    load_and_validate_alignment_config,
    run_multiscale_orbit_alignment_audit,
    write_audit_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the fixed multiscale GF(2) orbit against the exact two-transition "
            "Runtime-E4 protocol windows without cipher data or neural training."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    config = load_and_validate_alignment_config(args.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    def progress(event: str, payload: dict[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"event": event, **payload, "time": time.time()},
                    sort_keys=True,
                )
                + "\n"
            )

    progress("run_start", {"run_id": config["run_id"]})
    payload = run_multiscale_orbit_alignment_audit(
        config,
        project_root=project_root,
        progress_callback=progress,
    )
    write_audit_artifacts(payload, args.output_root)
    gate = payload["gate"]
    progress(
        "run_done",
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
