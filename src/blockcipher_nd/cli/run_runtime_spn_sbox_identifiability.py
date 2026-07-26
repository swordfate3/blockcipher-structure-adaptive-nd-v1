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

from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_identifiability import (
    adjudicate_sbox_identifiability,
    load_and_validate_sbox_identifiability_config,
    run_sbox_identifiability,
    write_sbox_identifiability_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Runtime-E4 S-box identifiability using frozen A8 checkpoints."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    config = load_and_validate_sbox_identifiability_config(
        args.config,
        project_root=project_root,
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

    progress("sbox_identifiability_start", {"run_id": config["run_id"]})
    payload = run_sbox_identifiability(
        config=config,
        config_path=args.config,
        project_root=project_root,
        progress_callback=progress,
    )
    gate = adjudicate_sbox_identifiability(
        config=config,
        rows=payload["rows"],
        validation=payload["validation"],
    )
    write_sbox_identifiability_artifacts(
        payload=payload,
        gate=gate,
        output_root=args.output_root,
    )
    progress(
        "sbox_identifiability_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if payload["validation"]["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
