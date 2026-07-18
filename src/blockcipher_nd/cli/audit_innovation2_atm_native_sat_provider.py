from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.atm_native_sat_witness_provider import (
    NativeSatProviderConfig,
    evaluate_phase_a,
    inspect_author_source,
    run_phase_a_calibration,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ATM native projected-SAT witness provider Phase A."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--paper-text", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument("--projected-key-cap", type=int, default=1 << 16)
    parser.add_argument("--trail-model-cap", type=int, default=1 << 20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = NativeSatProviderConfig(
        run_id=args.run_id,
        mode=args.mode,
        projected_key_cap=args.projected_key_cap,
        trail_model_cap=args.trail_model_cap,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    _write_progress(progress, "run_start", {"training": False, "remote": False})
    actual_commit = _git_head(args.atm_root)
    source = inspect_author_source(args.atm_root, args.paper_text)
    environment = _environment_contract(args.atm_root)
    _write_progress(progress, "environment_ready", environment)
    calibration = run_phase_a_calibration(
        args.atm_root, trail_model_cap=config.trail_model_cap
    )
    evaluation = evaluate_phase_a(
        config,
        actual_commit=actual_commit,
        source=source,
        calibration=calibration,
        environment=environment,
    )
    gate = evaluation["gate"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_atm_native_sat_witness_provider_phase_a",
        "config": serializable_config(config),
        "actual_atm_commit": actual_commit,
        "external_source_root": str(args.atm_root),
        "paper_text": str(args.paper_text),
        "python": sys.version,
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "source": source,
        "environment": environment,
        "calibration": calibration,
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }
    _write_jsonl(args.output_root / "sbox_coefficients.jsonl", calibration["sbox_rows"])
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate}, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _environment_contract(atm_root: Path) -> dict[str, Any]:
    python_sat_available = importlib.util.find_spec("pysat") is not None
    glucose4_available = False
    if python_sat_available:
        from pysat.solvers import Glucose4

        glucose4_available = Glucose4 is not None
    extension_paths = tuple((atm_root / "bitarrays").glob("bitset*.so"))
    return {
        "python_sat_available": python_sat_available,
        "glucose4_available": glucose4_available,
        "bitarrays_extension_available": bool(extension_paths),
        "bitarrays_extensions": [str(path) for path in extension_paths],
    }


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "event": event,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
