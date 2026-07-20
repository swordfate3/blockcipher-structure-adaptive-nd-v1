from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.present_r9_atm_split333_retrieval import (
    Split333RetrievalConfig,
    result_rows,
    serializable_config,
    validate_split333_retrieval,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and promote a retrieved E104 PRESENT r9 split333 result."
    )
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--verified-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Split333RetrievalConfig()
    validation = validate_split333_retrieval(config, raw_root=args.raw_root)
    if validation["status"] != "pass":
        _write_json(args.raw_root / "validation.local.json", validation)
        _write_json(
            args.raw_root / "artifact_manifest.local.json",
            validation["artifact_manifest"],
        )
        print(
            json.dumps(
                {
                    "status": "fail",
                    "decision": validation["decision"],
                    "report": str(args.raw_root / "validation.local.json"),
                },
                sort_keys=True,
            )
        )
        return 1

    args.verified_root.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "results"):
        shutil.copytree(
            args.raw_root / name,
            args.verified_root / name,
            dirs_exist_ok=True,
        )
    _write_json(args.verified_root / "validation.json", validation)
    _write_json(
        args.verified_root / "artifact_manifest.json",
        validation["artifact_manifest"],
    )
    _write_json(
        args.verified_root / "metadata.json",
        {
            "run_id": config.run_id,
            "source_run_id": config.source_run_id,
            "task": "innovation2_present_r9_atm_split333_retrieval_validation",
            "config": serializable_config(config),
            "training_performed": False,
            "raw_fallback_promoted_after_validation": True,
            "claim_scope": validation["claim_scope"],
        },
    )
    _write_jsonl(args.verified_root / "results.jsonl", result_rows(validation))
    print(
        json.dumps(
            {
                "status": "pass",
                "decision": validation["decision"],
                "verified_root": str(args.verified_root),
            },
            sort_keys=True,
        )
    )
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
