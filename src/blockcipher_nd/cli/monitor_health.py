from __future__ import annotations

import argparse
import json
import subprocess
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
        "--recent-lines",
        type=int,
        default=8,
        help="Number of recent monitor log lines to include.",
    )
    return parser.parse_args(argv)


def monitor_health_report(
    *,
    run_id: str,
    root: Path = Path("outputs/remote_results"),
    tmux_session: str | None = None,
    recent_lines: int = 8,
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
    stderr_text = _read_text(ssh_stderr).strip()
    tmux = _tmux_status(tmux_session)
    status = _health_status(
        run_root_exists=run_root.exists(),
        results_jsonl_exists=results_jsonl.exists(),
        done_markers=done_markers,
        failed_markers=failed_markers,
        stderr_text=stderr_text,
        recent_monitor_lines=recent_monitor_lines,
        tmux=tmux,
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
        "ssh_stderr_log": str(ssh_stderr),
        "ssh_stderr_exists": ssh_stderr.exists(),
        "ssh_stderr_empty": stderr_text == "",
        "ssh_stderr_tail": _tail_text(stderr_text, recent_lines),
        "results_jsonl": str(results_jsonl),
        "results_jsonl_exists": results_jsonl.exists(),
        "done_markers": done_markers,
        "failed_markers": failed_markers,
        "artifact_files": artifact_files,
        "needs_main_thread_intervention": status in {"failed", "unhealthy", "missing_monitor"},
        "postprocess_allowed": status == "result_ready",
    }


def _health_status(
    *,
    run_root_exists: bool,
    results_jsonl_exists: bool,
    done_markers: list[str],
    failed_markers: list[str],
    stderr_text: str,
    recent_monitor_lines: list[str],
    tmux: dict[str, Any],
) -> str:
    if failed_markers:
        return "failed"
    if results_jsonl_exists or done_markers:
        return "result_ready"
    if not run_root_exists or not recent_monitor_lines:
        return "missing_monitor"
    if stderr_text:
        return "unhealthy"
    if tmux["checked"] and not tmux["exists"]:
        return "unhealthy"
    if any("running" in line for line in recent_monitor_lines):
        return "running"
    return "unknown"


def _tmux_status(session: str | None) -> dict[str, Any]:
    if not session:
        return {"checked": False, "session": None, "exists": None, "returncode": None}
    process = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "checked": True,
        "session": session,
        "exists": process.returncode == 0,
        "returncode": process.returncode,
        "stderr": process.stderr.strip(),
    }


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
        recent_lines=args.recent_lines,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] not in {"failed", "unhealthy", "missing_monitor"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
