from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one E104 subprocess stage with a durable timeout marker."
    )
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--marker-root", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a child command is required after --")
    args.marker_root.mkdir(parents=True, exist_ok=True)
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    started_path = args.marker_root / f"{args.stage_id}_started.marker"
    done_path = args.marker_root / f"{args.stage_id}_done.marker"
    timeout_path = args.marker_root / f"{args.stage_id}_timeout.marker"
    failure_path = args.marker_root / f"{args.stage_id}_failed.marker"
    started = time.time()
    _write_marker(
        started_path,
        {
            "stage_id": args.stage_id,
            "status": "started",
            "started_at": started,
            "timeout_seconds": args.timeout_seconds,
            "command": command,
        },
    )
    creationflags = 0
    start_new_session = os.name != "nt"
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    with args.stdout.open("ab") as stdout, args.stderr.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            elapsed = time.time() - started
            _write_marker(
                timeout_path,
                {
                    "stage_id": args.stage_id,
                    "status": "timeout",
                    "elapsed_seconds": elapsed,
                    "timeout_seconds": args.timeout_seconds,
                    "pid": process.pid,
                },
            )
            return 124
    elapsed = time.time() - started
    marker = {
        "stage_id": args.stage_id,
        "status": "done" if return_code == 0 else "failed",
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "pid": process.pid,
    }
    if return_code == 0:
        _write_marker(done_path, marker)
    else:
        _write_marker(failure_path, marker)
    return return_code


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
