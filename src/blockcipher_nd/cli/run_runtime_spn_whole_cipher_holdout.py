from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    adjudicate_holdout_experiment,
    config_sha256,
    load_and_validate_holdout_config,
    run_holdout_experiment,
    verify_readiness,
    write_holdout_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train conditioner-free Runtime-E4 on four SPNs and evaluate a zero-"
            "fine-tune RECTANGLE whole-cipher holdout."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    config = load_and_validate_holdout_config(args.config)
    readiness = verify_readiness(config, project_root)
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

    progress(
        "holdout_experiment_start",
        {
            "run_id": config["run_id"],
            "holdout_cipher": config["holdout_cipher"],
            "source_ciphers": config["source_ciphers"],
            "readiness_decision": readiness["decision"],
        },
    )
    payload = run_holdout_experiment(
        config=config,
        config_sha256=config_sha256(args.config),
        output_root=args.output_root,
        progress_callback=progress,
    )
    gate = adjudicate_holdout_experiment(payload)
    write_holdout_artifacts(payload=payload, gate=gate, output_root=args.output_root)
    progress(
        "holdout_experiment_done",
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
