from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_multiscale_orbit import (
    load_and_validate_config,
    run_multiscale_orbit_audit,
    write_audit_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a fixed multiscale bank of exact runtime GF(2) inverse views "
            "without generating cipher data or training a network."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_and_validate_config(args.config)
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
    payload = run_multiscale_orbit_audit(config, progress_callback=progress)
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
