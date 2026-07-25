from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    build_holdout_readiness,
    load_and_validate_holdout_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check zero-leakage readiness for the RECTANGLE whole-cipher holdout."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    config = load_and_validate_holdout_config(args.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest, gate = build_holdout_readiness(config, project_root=project_root)
    _write_json(args.output_root / "manifest.json", manifest)
    _write_json(args.output_root / "validation.json", {"status": gate["status"], "checks": gate["checks"]})
    _write_json(args.output_root / "gate.json", gate)
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": gate["run_id"],
            "status": gate["status"],
            "decision": gate["decision"],
            "claim_scope": gate["claim_scope"],
            "next_action": gate["next_action"],
        },
    )
    (args.output_root / "progress.jsonl").write_text(
        json.dumps(
            {
                "event": "readiness_done",
                "run_id": gate["run_id"],
                "status": gate["status"],
                "decision": gate["decision"],
                "time": time.time(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["status"] == "pass" else 4


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
