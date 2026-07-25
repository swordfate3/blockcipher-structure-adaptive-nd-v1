from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_h1_relation_activity_pooling import (
    adjudicate_h1_relation_activity_pooling,
    load_and_validate_h1_relation_activity_pooling_config,
    revalidate_existing_h1_relation_activity_pooling,
    run_h1_relation_activity_pooling,
    run_h1_relation_activity_pooling_readiness,
    write_h1_relation_activity_pooling_artifacts,
    write_h1_relation_activity_pooling_readiness_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run H1-A5 parameter-free GF(2)-relation activity pooling readiness "
            "or its frozen RECTANGLE holdout diagnostic."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--readiness-only", action="store_true")
    mode.add_argument("--revalidate-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    config = load_and_validate_h1_relation_activity_pooling_config(
        args.config,
        project_root=project_root,
        require_readiness=not (args.readiness_only or args.revalidate_existing),
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

    if args.revalidate_existing:
        progress("h1_relation_activity_pooling_revalidation_start", {})
        gate = revalidate_existing_h1_relation_activity_pooling(
            config=config,
            output_root=args.output_root,
            project_root=project_root,
        )
        progress(
            "h1_relation_activity_pooling_revalidation_done",
            {"status": gate["status"], "decision": gate["decision"]},
        )
        print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
        return 0

    if args.readiness_only:
        progress("h1_relation_activity_pooling_readiness_start", {})
        readiness = run_h1_relation_activity_pooling_readiness(
            config=config,
            project_root=project_root,
        )
        write_h1_relation_activity_pooling_readiness_artifacts(
            readiness,
            output_root=args.output_root,
        )
        progress(
            "h1_relation_activity_pooling_readiness_done",
            {"status": readiness["status"], "decision": readiness["decision"]},
        )
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 0 if readiness["status"] == "pass" else 4

    progress("h1_relation_activity_pooling_start", {"run_id": config["run_id"]})
    payload = run_h1_relation_activity_pooling(
        config=config,
        config_path=args.config,
        output_root=args.output_root,
        project_root=project_root,
        progress_callback=progress,
    )
    gate = adjudicate_h1_relation_activity_pooling(payload)
    write_h1_relation_activity_pooling_artifacts(
        payload=payload,
        gate=gate,
        output_root=args.output_root,
    )
    progress(
        "h1_relation_activity_pooling_done",
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
