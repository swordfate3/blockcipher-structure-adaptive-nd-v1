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
    evaluate_r9_probe,
    run_r9_singleton_probe,
    select_singleton_relation_mutation,
    serializable_config,
)
from blockcipher_nd.tasks.innovation2.present_generalized_relation_precursor_boundary import (
    load_relations,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E58-B frozen ATM r9 SAT probe.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--atm-root", required=True, type=Path)
    parser.add_argument("--wall-clock-cap-seconds", type=int, default=60)
    parser.add_argument("--projected-key-cap", type=int, default=1 << 16)
    parser.add_argument("--trail-model-cap", type=int, default=1 << 20)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input-exponent", type=int)
    parser.add_argument("--output-exponent", type=int)
    parser.add_argument("--worker-output", type=Path)
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
    relations = load_relations(args.atm_root / "Ciphers/PRESENT/Results")
    candidate = select_singleton_relation_mutation(relations)
    _write_json(args.output_root / "candidate.json", candidate)
    _write_progress(
        progress,
        "worker_start",
        {
            "wall_clock_cap_seconds": args.wall_clock_cap_seconds,
            "candidate": candidate["candidate_relation"],
        },
    )
    worker_output = args.output_root / "worker_result.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-id",
        args.run_id,
        "--output-root",
        str(args.output_root),
        "--atm-root",
        str(args.atm_root),
        "--projected-key-cap",
        str(config.projected_key_cap),
        "--trail-model-cap",
        str(config.trail_model_cap),
        "--worker",
        "--input-exponent",
        str(candidate["candidate_relation"][0]["input_exponent"]),
        "--output-exponent",
        str(candidate["candidate_relation"][0]["output_exponent"]),
        "--worker-output",
        str(worker_output),
    ]
    worker_status = "failed"
    worker_result: dict[str, Any] | None = None
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
        if completed.returncode == 0 and worker_output.is_file():
            worker_status = "completed"
            worker_result = json.loads(worker_output.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        worker_status = "timeout"
        stdout = _decode_timeout(exc.stdout)
        stderr = _decode_timeout(exc.stderr)
    (args.output_root / "worker.stdout.log").write_text(stdout, encoding="utf-8")
    (args.output_root / "worker.stderr.log").write_text(stderr, encoding="utf-8")
    evaluation = evaluate_r9_probe(
        config,
        candidate=candidate,
        worker_status=worker_status,
        worker_result=worker_result,
        wall_clock_cap_seconds=args.wall_clock_cap_seconds,
    )
    gate = evaluation["gate"]
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_atm_native_sat_r9_singleton_probe",
        "config": serializable_config(config),
        "wall_clock_cap_seconds": args.wall_clock_cap_seconds,
        "external_source_root": str(args.atm_root),
        "key_model": "independent_round_keys",
        "training_performed": False,
        "remote_scale": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "candidate": candidate,
        "worker_status": worker_status,
        "worker_result": worker_result,
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
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(json.dumps({"gate": gate}, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _worker_main(args: argparse.Namespace) -> int:
    if args.input_exponent is None or args.output_exponent is None:
        raise ValueError("worker exponents are required")
    if args.worker_output is None:
        raise ValueError("worker output is required")
    result = run_r9_singleton_probe(
        args.atm_root,
        input_exponent=args.input_exponent,
        output_exponent=args.output_exponent,
        projected_key_cap=args.projected_key_cap,
        trail_model_cap=args.trail_model_cap,
    )
    _write_json(args.worker_output, result)
    return 0


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
