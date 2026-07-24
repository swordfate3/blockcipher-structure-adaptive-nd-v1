from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window import (
    adjudicate_runtime_spn_recurrent_window,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adjudicate the frozen two-seed uKNIT Runtime-E4 recurrent-window gate."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = args.run_root / "results.jsonl"
    rows = _read_jsonl(results_path)
    gate = adjudicate_runtime_spn_recurrent_window(
        run_id=args.run_id,
        rows=rows,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(results_path),
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": gate["cipher"],
        "training_performed": True,
        "samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 10,
        "seeds": [0, 1],
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    _append_progress(
        args.run_root / "progress.jsonl",
        {
            "event": "recurrent_window_gate_done",
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
