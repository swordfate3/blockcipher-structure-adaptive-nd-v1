from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded local health check for a remote-result monitor directory."
    )
    parser.add_argument("--run-id", required=True, help="Run id under the remote results root.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/remote_results"),
        help="Local remote-result root directory.",
    )
    parser.add_argument(
        "--tmux-session",
        default=None,
        help="Optional local tmux session name to check once with tmux has-session.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Optional plan CSV path used to build a postprocess command when result_ready.",
    )
    parser.add_argument(
        "--plan-doc",
        type=Path,
        default=None,
        help="Optional experiment plan Markdown path for postprocess --update-plan-doc.",
    )
    parser.add_argument(
        "--recent-lines",
        type=int,
        default=8,
        help="Number of recent monitor log lines to include.",
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=int,
        default=1800,
        help="Mark a running monitor stale when its newest timestamp is older than this many seconds.",
    )
    return parser.parse_args(argv)


def monitor_health_report(
    *,
    run_id: str,
    root: Path = Path("outputs/remote_results"),
    tmux_session: str | None = None,
    plan_path: Path | None = None,
    plan_doc_path: Path | None = None,
    recent_lines: int = 8,
    stale_after_seconds: int = 1800,
    now: datetime | None = None,
) -> dict[str, Any]:
    run_root = root / run_id
    monitor_dir = run_root / "monitor"
    monitor_log = monitor_dir / "monitor.log"
    ssh_stderr = monitor_dir / "monitor_ssh_stderr.log"
    results_jsonl = run_root / "results" / f"{run_id}.jsonl"
    done_markers = _relative_paths(run_root, sorted(run_root.glob("**/*done*")))
    failed_markers = _relative_paths(run_root, sorted(run_root.glob("**/*failed*")))
    artifact_files = _relative_paths(run_root, sorted(path for path in run_root.glob("**/*") if path.is_file()))
    recent_monitor_lines = _tail_lines(monitor_log, recent_lines)
    heartbeat = _heartbeat_status(recent_monitor_lines, stale_after_seconds, now=now)
    stderr_text = _read_text(ssh_stderr).strip()
    tmux = _tmux_status(tmux_session)
    status = _health_status(
        run_root_exists=run_root.exists(),
        results_jsonl_exists=results_jsonl.exists(),
        done_markers=done_markers,
        failed_markers=failed_markers,
        stderr_text=stderr_text,
        recent_monitor_lines=recent_monitor_lines,
        heartbeat=heartbeat,
        tmux=tmux,
    )
    postprocess_command = _postprocess_command(
        status=status,
        run_id=run_id,
        results_jsonl=results_jsonl,
        run_root=run_root,
        plan_path=plan_path,
        plan_doc_path=plan_doc_path,
    )
    return {
        "status": status,
        "run_id": run_id,
        "run_root": str(run_root),
        "run_root_exists": run_root.exists(),
        "tmux": tmux,
        "monitor_log": str(monitor_log),
        "monitor_log_exists": monitor_log.exists(),
        "recent_monitor_lines": recent_monitor_lines,
        "heartbeat": heartbeat,
        "ssh_stderr_log": str(ssh_stderr),
        "ssh_stderr_exists": ssh_stderr.exists(),
        "ssh_stderr_empty": stderr_text == "",
        "ssh_stderr_tail": _tail_text(stderr_text, recent_lines),
        "results_jsonl": str(results_jsonl),
        "results_jsonl_exists": results_jsonl.exists(),
        "done_markers": done_markers,
        "failed_markers": failed_markers,
        "artifact_files": artifact_files,
        "needs_main_thread_intervention": status
        in {"failed", "unhealthy", "missing_monitor", "stale_monitor", "completed_missing_results"},
        "postprocess_allowed": status == "result_ready",
        "postprocess_command": postprocess_command,
    }


def _health_status(
    *,
    run_root_exists: bool,
    results_jsonl_exists: bool,
    done_markers: list[str],
    failed_markers: list[str],
    stderr_text: str,
    recent_monitor_lines: list[str],
    heartbeat: dict[str, Any],
    tmux: dict[str, Any],
) -> str:
    if failed_markers:
        return "failed"
    if results_jsonl_exists:
        return "result_ready"
    if done_markers:
        return "completed_missing_results"
    if not run_root_exists or not recent_monitor_lines:
        return "missing_monitor"
    if stderr_text:
        return "unhealthy"
    if tmux["checked"] and tmux["exists"] is False:
        return "unhealthy"
    if heartbeat["is_stale"]:
        return "stale_monitor"
    if any("running" in line for line in recent_monitor_lines):
        return "running"
    return "unknown"


def _heartbeat_status(
    recent_monitor_lines: list[str],
    stale_after_seconds: int,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    newest = _newest_monitor_timestamp(recent_monitor_lines)
    if newest is None:
        return {
            "newest_timestamp": None,
            "age_seconds": None,
            "stale_after_seconds": stale_after_seconds,
            "is_stale": False,
        }
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age_seconds = max(0, int((current.astimezone(timezone.utc) - newest.astimezone(timezone.utc)).total_seconds()))
    return {
        "newest_timestamp": newest.isoformat(),
        "age_seconds": age_seconds,
        "stale_after_seconds": stale_after_seconds,
        "is_stale": age_seconds > stale_after_seconds,
    }


def _newest_monitor_timestamp(lines: list[str]) -> datetime | None:
    newest: datetime | None = None
    for line in lines:
        token = line.split(maxsplit=1)[0] if line.strip() else ""
        if "T" not in token:
            continue
        try:
            timestamp = datetime.fromisoformat(token)
        except ValueError:
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if newest is None or timestamp > newest:
            newest = timestamp
    return newest


def _tmux_status(session: str | None) -> dict[str, Any]:
    if not session:
        return {
            "checked": False,
            "session": None,
            "exists": None,
            "returncode": None,
            "check_error": False,
        }
    process = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr = process.stderr.strip()
    check_error = process.returncode != 0 and bool(stderr) and "can't find session" not in stderr.lower()
    return {
        "checked": True,
        "session": session,
        "exists": None if check_error else process.returncode == 0,
        "returncode": process.returncode,
        "stderr": stderr,
        "check_error": check_error,
    }


def _postprocess_command(
    *,
    status: str,
    run_id: str,
    results_jsonl: Path,
    run_root: Path,
    plan_path: Path | None,
    plan_doc_path: Path | None,
) -> list[str]:
    if status != "result_ready" or plan_path is None:
        return []
    command = [
        "env",
        "UV_CACHE_DIR=/tmp/uv-cache",
        "MPLCONFIGDIR=/tmp/mplconfig",
        "uv",
        "run",
        "python",
        "scripts/postprocess-invp-result",
        "--plan",
        str(plan_path),
        "--results",
        str(results_jsonl),
        "--output-dir",
        str(run_root),
        "--run-id",
        run_id,
        "--expected-rows",
        "1",
    ]
    if plan_doc_path is not None:
        command.extend(["--update-plan-doc", str(plan_doc_path)])
    return command


def _tail_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max(0, count) :]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _tail_text(text: str, count: int) -> list[str]:
    return text.splitlines()[-max(0, count) :]


def _relative_paths(root: Path, paths: list[Path]) -> list[str]:
    output: list[str] = []
    for path in paths:
        try:
            output.append(str(path.relative_to(root)))
        except ValueError:
            output.append(str(path))
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = monitor_health_report(
        run_id=args.run_id,
        root=args.root,
        tmux_session=args.tmux_session,
        plan_path=args.plan,
        plan_doc_path=args.plan_doc,
        recent_lines=args.recent_lines,
        stale_after_seconds=args.stale_after_seconds,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return (
        0
        if report["status"]
        not in {"failed", "unhealthy", "missing_monitor", "stale_monitor", "completed_missing_results"}
        else 4
    )


if __name__ == "__main__":
    raise SystemExit(main())
