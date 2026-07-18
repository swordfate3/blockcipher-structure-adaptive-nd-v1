from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from blockcipher_nd.tasks.innovation2.present_open_3sdp_glpk_gate import (
    GlpkEnumerationGateConfig,
    evaluate_glpk_enumeration_gate,
    serializable_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit E53-B Sage/GLPK raw-trail blocking scalability."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "audit"), default="audit")
    parser.add_argument(
        "--output-exponents", type=int, nargs="+", default=[1, 3, 7, 15]
    )
    parser.add_argument(
        "--required-complete", type=int, nargs="+", default=[1, 3, 7]
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = GlpkEnumerationGateConfig(
        run_id=args.run_id,
        mode=args.mode,
        output_exponents=tuple(args.output_exponents),
        required_complete=tuple(args.required_complete),
        timeout_seconds=args.timeout_seconds,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    _write_progress(
        progress,
        "run_start",
        {
            "run_id": config.run_id,
            "mode": config.mode,
            "training": False,
            "five_round_subset": False,
        },
    )
    sage = shutil.which("sage")
    records: list[dict[str, Any]] = []
    for output_exponent in config.output_exponents:
        record = _run_glpk_query(
            sage=sage,
            output_exponent=output_exponent,
            timeout_seconds=config.timeout_seconds,
        )
        records.append(record)
        _write_progress(
            progress,
            "query_complete",
            {
                "output_exponent": output_exponent,
                "status": record["status"],
                "seconds": record["seconds"],
                "solutions": record.get("solutions"),
            },
        )
    evaluation = evaluate_glpk_enumeration_gate(config, records)
    gate = evaluation["gate"]
    provider_manifest = {
        "run_id": config.run_id,
        "provider_id": "sage_glpk_per_solution_blocking",
        "backend": "GLPKBackend",
        "status": (
            "not_scalable"
            if gate["decision"]
            == "innovation2_present_r5_open_3sdp_glpk_blocking_not_scalable"
            else gate["status"]
        ),
        "enumeration_semantics": (
            "one ANF term is selected per active output coordinate; input "
            "exponent is their Boolean union; each solution is excluded by an "
            "exact binary no-good constraint"
        ),
        "completeness_rule": (
            "enumerated solution count must equal the Cartesian product of "
            "selected coordinate ANF term counts"
        ),
        "query_timeout_seconds": config.timeout_seconds,
        "representative_output_exponents": list(config.output_exponents),
        "alternative_runtime_audit": evaluation["metrics"][
            "available_alternative_backends"
        ],
        "full_cipher_provider_implemented": False,
        "five_round_subset_executed": False,
        "training_performed": False,
    }
    metadata = {
        "run_id": config.run_id,
        "task": "innovation2_present_open_3sdp_glpk_enumeration_gate",
        "config": serializable_config(config),
        "cipher_component": "PRESENT 4-bit S-box",
        "target_cipher": "PRESENT-80",
        "target_rounds": 5,
        "training_performed": False,
        "remote_scale": False,
        "five_round_subset_executed": False,
        "claim_scope": gate["claim_scope"],
    }
    summary = {
        "run_id": config.run_id,
        "metadata": metadata,
        "provider_manifest": provider_manifest,
        "metrics": evaluation["metrics"],
        "gate": gate,
        "result_rows": evaluation["result_rows"],
    }
    _write_jsonl(args.output_root / "queries.jsonl", records)
    _write_jsonl(args.output_root / "results.jsonl", evaluation["result_rows"])
    _write_json(args.output_root / "provider_manifest.json", provider_manifest)
    _write_json(args.output_root / "metadata.json", metadata)
    _write_json(args.output_root / "summary.json", summary)
    _write_json(args.output_root / "gate.json", gate)
    _write_progress(
        progress,
        "run_done",
        {"status": gate["status"], "decision": gate["decision"]},
    )
    print(
        json.dumps(
            {"gate": gate, "output_root": str(args.output_root)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if gate["status"] == "fail" else 0


def _run_glpk_query(
    *, sage: str | None, output_exponent: int, timeout_seconds: float
) -> dict[str, Any]:
    if sage is None:
        return {
            "output_exponent": output_exponent,
            "status": "error",
            "seconds": 0.0,
            "error": "sage executable unavailable",
        }
    source_root = Path(__file__).resolve().parents[2]
    python_path = str(source_root)
    if os.environ.get("PYTHONPATH"):
        python_path += os.pathsep + os.environ["PYTHONPATH"]
    code = (
        "from blockcipher_nd.sage_tools.present_sbox_glpk "
        "import enumerate_sbox_output_exponent_glpk_json; "
        f"print(enumerate_sbox_output_exponent_glpk_json(int({output_exponent})))"
    )
    started = perf_counter()
    try:
        completed = subprocess.run(
            [sage, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env={**os.environ, "HOME": "/tmp", "PYTHONPATH": python_path},
        )
    except subprocess.TimeoutExpired:
        return {
            "output_exponent": output_exponent,
            "status": "timeout",
            "seconds": float(perf_counter() - started),
            "timeout_seconds": timeout_seconds,
            "counts": {},
            "error": None,
        }
    seconds = float(perf_counter() - started)
    if completed.returncode:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        return {
            "output_exponent": output_exponent,
            "status": "error",
            "seconds": seconds,
            "counts": {},
            "error": lines[-1] if lines else "unknown Sage/GLPK error",
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "output_exponent": output_exponent,
            "status": "error",
            "seconds": seconds,
            "counts": {},
            "error": f"invalid Sage/GLPK JSON: {exc}",
        }
    return {**payload, "status": "completed", "process_seconds": seconds, "error": None}


def _write_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


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
