from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_experiment import (
    config_sha256,
    run_joint_experiment,
    verify_readiness,
    write_joint_artifacts,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_film_experiment import (
    adjudicate_true_film_experiment,
    load_and_validate_true_film_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and adjudicate one shared Runtime-E4 True-FiLM checkpoint "
            "across the frozen five-SPN panel."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    config = load_and_validate_true_film_config(args.config)
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
        "true_film_joint_experiment_start",
        {
            "run_id": config["run_id"],
            "config": str(args.config),
            "readiness_decision": readiness["decision"],
        },
    )
    payload = run_joint_experiment(
        config=config,
        config_sha256=config_sha256(args.config),
        output_root=args.output_root,
        progress_callback=progress,
    )
    gate = adjudicate_true_film_experiment(payload, project_root=project_root)
    write_joint_artifacts(payload=payload, gate=gate, output_root=args.output_root)
    progress(
        "true_film_joint_experiment_done",
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
