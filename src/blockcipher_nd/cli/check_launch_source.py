from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


_BRANCH_RE = re.compile(
    r"^## (?P<branch>.+?)(?:\.\.\.(?P<upstream>[^\s\[]+))?(?: \[(?P<state>[^\]]+)\])?$"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the current Git source is safe to use for a remote launch."
    )
    parser.add_argument("--worktree", type=Path, default=Path("."), help="Git worktree to inspect.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args(argv)


def launch_source_report(worktree: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=worktree,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "status": "fail",
            "worktree": str(worktree),
            "errors": [f"git_status_failed:{exc}"],
            "claim_scope": "local source publication gate only; does not contact remote hosts",
        }
    report = launch_source_report_from_status(result.stdout)
    report["worktree"] = str(worktree)
    return report


def launch_source_report_from_status(status_text: str) -> dict[str, Any]:
    lines = [line for line in status_text.splitlines() if line.strip()]
    branch_line = lines[0] if lines else ""
    match = _BRANCH_RE.match(branch_line)
    branch = match.group("branch") if match else None
    upstream = match.group("upstream") if match else None
    state = (match.group("state") if match else "") or ""
    ahead, behind = _ahead_behind(state)
    dirty = any(not line.startswith("## ") for line in lines)
    errors: list[str] = []
    if not branch:
        errors.append("missing_branch")
    if not upstream:
        errors.append("missing_upstream")
    if ahead > 0:
        errors.append("unpushed_commits")
    if behind > 0:
        errors.append("local_branch_behind_upstream")
    if dirty:
        errors.append("dirty_worktree")
    return {
        "status": "pass" if not errors else "fail",
        "branch": branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "dirty": dirty,
        "should_push": ahead > 0,
        "errors": errors,
        "raw_status": status_text,
        "claim_scope": (
            "local source publication gate only; remote launch still requires pushed commit, "
            "readiness pass, generated artifacts, GPU gate, and monitor handoff"
        ),
    }


def _ahead_behind(state: str) -> tuple[int, int]:
    ahead = 0
    behind = 0
    for part in state.split(","):
        stripped = part.strip()
        if stripped.startswith("ahead "):
            ahead = _int_or_zero(stripped.removeprefix("ahead "))
        elif stripped.startswith("behind "):
            behind = _int_or_zero(stripped.removeprefix("behind "))
    return ahead, behind


def _int_or_zero(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = launch_source_report(args.worktree)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
