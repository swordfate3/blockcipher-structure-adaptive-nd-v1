from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation2.atm_native_sat_witness_provider import (
    NativeSatProviderConfig,
    evaluate_low_round_panel,
    run_present_relation_panel,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E59 PRESENT r2 ATM label panel.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--phase-a-gate", required=True, type=Path)
    parser.add_argument("--wall-clock-cap-seconds", type=int, default=60)
    parser.add_argument("--projected-key-cap", type=int, default=1 << 12)
    parser.add_argument("--trail-model-cap", type=int, default=1 << 16)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--panel-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return _worker_main(args)
    config = NativeSatProviderConfig(
        run_id=args.run_id,
        projected_key_cap=args.projected_key_cap,
        trail_model_cap=args.trail_model_cap,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress.write_text("", encoding="utf-8")
    model_output = args.output_root / "model.json"
    panel_output = args.output_root / "panel.jsonl"
    model_output.unlink(missing_ok=True)
    panel_output.unlink(missing_ok=True)
    _write_progress(
        progress,
        "worker_start",
        {"planned_queries": 16, "wall_clock_cap_seconds": args.wall_clock_cap_seconds},
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-id",
        args.run_id,
        "--output-root",
        str(args.output_root),
        "--atm-root",
        str(args.atm_root),
        "--phase-a-gate",
        str(args.phase_a_gate),
        "--projected-key-cap",
        str(config.projected_key_cap),
        "--trail-model-cap",
        str(config.trail_model_cap),
        "--worker",
        "--model-output",
        str(model_output),
        "--panel-output",
        str(panel_output),
    ]
    worker_status = "failed"
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
            timeout=args.wall_clock_cap_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        if completed.returncode == 0:
            worker_status = "completed"
    except subprocess.TimeoutExpired as exc:
        worker_status = "timeout"
        stdout = _decode_timeout(exc.stdout)
        stderr = _decode_timeout(exc.stderr)
    (args.output_root / "worker.stdout.log").write_text(stdout, encoding="utf-8")
    (args.output_root / "worker.stderr.log").write_text(stderr, encoding="utf-8")
    model = _load_json(model_output)
    rows = _load_jsonl(panel_output)
    phase_a_gate = _load_json(args.phase_a_gate) or {}
    evaluation = evaluate_low_round_panel(
        config,
        phase_a_gate=phase_a_gate,
        model=model,
        rows=rows,
        planned_queries=16,
        rounds=2,
        worker_status=worker_status,
        wall_clock_cap_seconds=args.wall_clock_cap_seconds,
    )
    gate = evaluation["gate"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_atm_r2_strict_relation_panel",
        "config": serializable_config(config),
        "wall_clock_cap_seconds": args.wall_clock_cap_seconds,
        "external_source_root": str(args.atm_root),
        "source_phase_a_gate": str(args.phase_a_gate),
        "rounds": 2,
        "input_exponent_hex": "0xFFFFFFFFFFFFFFF0",
        "output_exponents": [f"0x{1 << bit:016X}" for bit in range(16)],
        "key_model": "independent_round_keys",
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "model": model,
        "worker_status": worker_status,
        "rows": rows,
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {
            "status": gate["status"],
            "decision": gate["decision"],
            "completed_queries": len(rows),
        },
    )
    print(json.dumps({"gate": gate}, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _worker_main(args: argparse.Namespace) -> int:
    if args.model_output is None or args.panel_output is None:
        raise ValueError("worker artifact paths are required")
    run_present_relation_panel(
        args.atm_root,
        rounds=2,
        input_exponent=0xFFFFFFFFFFFFFFF0,
        output_exponents=tuple(1 << bit for bit in range(16)),
        projected_key_cap=args.projected_key_cap,
        trail_model_cap=args.trail_model_cap,
        model_output=args.model_output,
        panel_output=args.panel_output,
    )
    return 0


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _decode_timeout(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


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
