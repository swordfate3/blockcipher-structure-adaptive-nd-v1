from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_uknit_heterogeneous_holdout import (
    adjudicate_uknit_heterogeneous_holdout,
    load_and_validate_uknit_heterogeneous_holdout_config,
    run_uknit_heterogeneous_holdout,
    run_uknit_heterogeneous_holdout_readiness,
    write_uknit_heterogeneous_holdout_artifacts,
    write_uknit_heterogeneous_holdout_readiness_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the A6 uKNIT heterogeneous-GF(2) whole-cipher holdout "
            "readiness gate or frozen two-seed diagnostic."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    config = load_and_validate_uknit_heterogeneous_holdout_config(
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
        progress("uknit_heterogeneous_holdout_readiness_start", {})
        readiness = run_uknit_heterogeneous_holdout_readiness(
            config=config,
            project_root=project_root,
        )
        write_uknit_heterogeneous_holdout_readiness_artifacts(
            readiness,
            output_root=args.output_root,
        )
        progress(
            "uknit_heterogeneous_holdout_readiness_done",
            {"status": readiness["status"], "decision": readiness["decision"]},
        )
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 0 if readiness["status"] == "pass" else 4

    progress("uknit_heterogeneous_holdout_start", {"run_id": config["run_id"]})
    payload = run_uknit_heterogeneous_holdout(
        config=config,
        config_path=args.config,
        output_root=args.output_root,
        project_root=project_root,
        progress_callback=progress,
    )
    gate = adjudicate_uknit_heterogeneous_holdout(payload)
    write_uknit_heterogeneous_holdout_artifacts(
        payload=payload,
        gate=gate,
        output_root=args.output_root,
    )
    progress(
        "uknit_heterogeneous_holdout_done",
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
