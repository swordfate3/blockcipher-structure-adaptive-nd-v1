from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_film_readiness import (
    build_primitive_true_film_readiness,
)


REGRESSION_TESTS = (
    "tests/test_runtime_parameterized_spn_distinguisher.py",
    "tests/test_runtime_spn_recurrent_window_readiness.py",
    "tests/test_runtime_spn_recurrent_window_gate.py",
    "tests/test_runtime_spn_dialga_d1.py",
    "tests/test_dialga128.py",
    "tests/test_runtime_spn_primitive_adapter.py",
    "tests/test_runtime_spn_primitive_film.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed five-cipher Runtime-SPN True-FiLM readiness gate."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[3]
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    regression_command = [sys.executable, "-m", "pytest", *REGRESSION_TESTS, "-q"]
    regression = subprocess.run(
        regression_command,
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (args.output_root / "regression-tests.log").write_text(
        regression.stdout,
        encoding="utf-8",
    )
    _append_progress(
        progress_path,
        {
            "event": "regression_tests_done",
            "returncode": regression.returncode,
            "time": time.time(),
        },
    )

    def progress(event: str, payload: dict[str, Any]) -> None:
        _append_progress(
            progress_path,
            {"event": event, **payload, "time": time.time()},
        )

    manifest, gate, smoke = build_primitive_true_film_readiness(
        run_id=args.run_id,
        cache_root=args.output_root / "cache",
        regression_tests_passed=regression.returncode == 0,
        regression_test_command=regression_command,
        progress_callback=progress,
    )
    _write_jsonl(args.output_root / "manifest.jsonl", manifest)
    _write_json(args.output_root / "smoke-results.json", smoke)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(
        args.output_root / "validation.json",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "checks": gate["checks"],
            "manifest_rows": len(manifest),
            "expected_manifest_rows": 5,
            "regression_test_returncode": regression.returncode,
            "training_scope": gate["training_scope"],
        },
    )
    _write_json(
        args.output_root / "summary.json",
        {
            "run_id": args.run_id,
            "task": gate["task"],
            "status": gate["status"],
            "decision": gate["decision"],
            "claim_scope": gate["claim_scope"],
            "next_action": gate["next_action"],
        },
    )
    _append_progress(
        progress_path,
        {
            "event": "readiness_gate_done",
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
            "time": time.time(),
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if gate["status"] == "pass" else 4


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


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
