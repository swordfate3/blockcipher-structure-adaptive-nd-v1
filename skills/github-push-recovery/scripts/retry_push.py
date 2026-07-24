#!/usr/bin/env python3
"""Safely retry a Git push only when failures are transient."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


TRANSIENT_MARKERS = (
    "couldn't connect to server",
    "could not connect to server",
    "connection timed out",
    "connection timeout",
    "operation timed out",
    "connection was reset",
    "connection reset by peer",
    "recv failure",
    "send failure",
    "remote end hung up unexpectedly",
    "the requested url returned error: 500",
    "the requested url returned error: 502",
    "the requested url returned error: 503",
    "the requested url returned error: 504",
    "the requested url returned error: 408",
    "the requested url returned error: 429",
    "empty reply from server",
    "http/2 stream",
    "rpc failed; curl",
    "gnutls_recv error",
    "ssl_connect",
    "tls connection was non-properly terminated",
    "temporary failure in name resolution",
    "could not resolve host",
)

AUTH_MARKERS = (
    "authentication failed",
    "permission denied (publickey)",
    "could not read username",
    "invalid username or password",
    "support for password authentication was removed",
    "access denied",
)

NON_FAST_FORWARD_MARKERS = (
    "non-fast-forward",
    "fetch first",
    "failed to push some refs",
)

REMOTE_REJECTED_MARKERS = (
    "remote rejected",
    "protected branch hook declined",
    "protected branch",
    "gh006",
    "gh013",
)

REPOSITORY_MARKERS = (
    "repository not found",
    "could not read from remote repository",
    "does not appear to be a git repository",
    "no such remote",
)

POLICY_MARKERS = (
    "codex-auto-review",
    "unacceptable risk",
    "sandbox reviewer",
    "approval policy",
)

URL_WITH_CREDENTIALS = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)
TOKEN_QUERY = re.compile(r"([?&](?:token|access_token|auth)=)[^&\s]+", re.IGNORECASE)


@dataclass(frozen=True)
class Attempt:
    number: int
    phase: str
    category: str
    returncode: int
    timestamp: str
    message: str


class RecoveryError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(value: str, limit: int = 4000) -> str:
    value = URL_WITH_CREDENTIALS.sub(r"\1<redacted>@", value)
    value = TOKEN_QUERY.sub(r"\1<redacted>", value)
    return value[-limit:].strip()


def classify(output: str) -> str:
    lowered = output.lower()
    if any(marker in lowered for marker in POLICY_MARKERS):
        return "platform_policy"
    if any(marker in lowered for marker in AUTH_MARKERS):
        return "authentication"
    if any(marker in lowered for marker in REPOSITORY_MARKERS):
        return "repository"
    if any(marker in lowered for marker in REMOTE_REJECTED_MARKERS):
        return "remote_rejected"
    if any(marker in lowered for marker in NON_FAST_FORWARD_MARKERS):
        return "non_fast_forward"
    if any(marker in lowered for marker in TRANSIENT_MARKERS):
        return "transient_network"
    return "unknown"


def run_git(
    git_bin: str,
    repo: Path,
    args: Sequence[str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = "true"
    env["SSH_ASKPASS"] = "true"
    env["GCM_INTERACTIVE"] = "Never"
    try:
        return subprocess.run(
            [git_bin, *args],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return subprocess.CompletedProcess(
            [git_bin, *args],
            124,
            stdout=f"{output}\nconnection timed out after {timeout:g} seconds",
        )


def require_git(
    git_bin: str,
    repo: Path,
    args: Sequence[str],
    timeout: float,
    label: str,
) -> str:
    completed = run_git(git_bin, repo, args, timeout)
    if completed.returncode != 0:
        raise RecoveryError(f"{label}: {sanitize(completed.stdout)}")
    return completed.stdout.strip()


def write_json(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def result_payload(
    *,
    status: str,
    category: str,
    repo: Path,
    remote: str,
    branch: str,
    local_sha: str,
    remote_sha: str | None,
    attempts: list[Attempt],
    dirty: bool,
) -> dict[str, object]:
    return {
        "status": status,
        "category": category,
        "repository": str(repo),
        "remote": remote,
        "branch": branch,
        "local_sha": local_sha,
        "remote_sha": remote_sha,
        "dirty_worktree": dirty,
        "attempt_count": len(attempts),
        "attempts": [asdict(item) for item in attempts],
        "finished_at": now_iso(),
    }


def parse_remote_sha(output: str, expected_ref: str) -> str | None:
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == expected_ref:
            return fields[0]
    return None


def sleep_with_deadline(delay: float, deadline: float) -> bool:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    time.sleep(min(delay, remaining))
    return time.monotonic() < deadline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--initial-delay", type=float, default=5.0)
    parser.add_argument("--max-delay", type=float, default=60.0)
    parser.add_argument("--max-elapsed", type=float, default=600.0)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--git-bin", default="git", help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_attempts < 1:
        raise RecoveryError("--max-attempts must be at least 1")
    for name in ("initial_delay", "max_delay", "max_elapsed", "command_timeout"):
        if getattr(args, name) <= 0:
            raise RecoveryError(f"--{name.replace('_', '-')} must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    attempts: list[Attempt] = []

    try:
        validate_args(args)
        repo = args.repo.resolve()
        top = Path(
            require_git(
                args.git_bin,
                repo,
                ["rev-parse", "--show-toplevel"],
                args.command_timeout,
                "not a Git repository",
            )
        ).resolve()
        local_sha = require_git(
            args.git_bin,
            top,
            ["rev-parse", "HEAD"],
            args.command_timeout,
            "cannot resolve HEAD",
        )
        branch = args.branch or require_git(
            args.git_bin,
            top,
            ["symbolic-ref", "--quiet", "--short", "HEAD"],
            args.command_timeout,
            "detached HEAD requires --branch",
        )
        require_git(
            args.git_bin,
            top,
            ["check-ref-format", "--branch", branch],
            args.command_timeout,
            "invalid destination branch",
        )
        require_git(
            args.git_bin,
            top,
            ["remote", "get-url", args.remote],
            args.command_timeout,
            "configured remote is unavailable",
        )
        dirty = bool(
            require_git(
                args.git_bin,
                top,
                ["status", "--porcelain"],
                args.command_timeout,
                "cannot inspect worktree",
            )
        )
    except (OSError, RecoveryError) as exc:
        print(f"preflight failed: {sanitize(str(exc))}", file=sys.stderr)
        return 2

    print(f"repository: {top}")
    print(f"payload: {local_sha} -> {args.remote}/{branch}")
    print(f"dirty worktree: {'yes (left untouched)' if dirty else 'no'}")
    if args.dry_run:
        print("dry-run complete: no network write attempted")
        write_json(
            args.json_out,
            result_payload(
                status="dry_run",
                category="none",
                repo=top,
                remote=args.remote,
                branch=branch,
                local_sha=local_sha,
                remote_sha=None,
                attempts=attempts,
                dirty=dirty,
            ),
        )
        return 0

    expected_ref = f"refs/heads/{branch}"
    refspec = f"HEAD:{expected_ref}"
    deadline = time.monotonic() + args.max_elapsed
    delay = args.initial_delay
    push_reported_success = False
    last_category = "unknown"

    for number in range(1, args.max_attempts + 1):
        current_sha = require_git(
            args.git_bin,
            top,
            ["rev-parse", "HEAD"],
            args.command_timeout,
            "cannot re-check HEAD",
        )
        if current_sha != local_sha:
            print("stopped: local HEAD changed during recovery", file=sys.stderr)
            payload = result_payload(
                status="not_published",
                category="local_head_changed",
                repo=top,
                remote=args.remote,
                branch=branch,
                local_sha=local_sha,
                remote_sha=None,
                attempts=attempts,
                dirty=dirty,
            )
            write_json(args.json_out, payload)
            return 3

        if push_reported_success:
            phase = "verify"
            command = ["ls-remote", "--exit-code", args.remote, expected_ref]
        else:
            phase = "push"
            command = ["push", "--porcelain", args.remote, refspec]

        completed = run_git(
            args.git_bin,
            top,
            command,
            min(args.command_timeout, max(1.0, deadline - time.monotonic())),
        )
        message = sanitize(completed.stdout)

        if phase == "push" and completed.returncode == 0:
            push_reported_success = True
            category = "push_reported_success"
        elif phase == "verify" and completed.returncode == 0:
            remote_sha = parse_remote_sha(completed.stdout, expected_ref)
            category = "verified" if remote_sha == local_sha else "remote_sha_mismatch"
            attempts.append(
                Attempt(
                    number, phase, category, completed.returncode, now_iso(), message
                )
            )
            if remote_sha == local_sha:
                payload = result_payload(
                    status="published_and_verified",
                    category="verified",
                    repo=top,
                    remote=args.remote,
                    branch=branch,
                    local_sha=local_sha,
                    remote_sha=remote_sha,
                    attempts=attempts,
                    dirty=dirty,
                )
                write_json(args.json_out, payload)
                print(f"published and verified: {remote_sha}")
                return 0
            write_json(
                args.json_out,
                result_payload(
                    status="not_published",
                    category=category,
                    repo=top,
                    remote=args.remote,
                    branch=branch,
                    local_sha=local_sha,
                    remote_sha=remote_sha,
                    attempts=attempts,
                    dirty=dirty,
                ),
            )
            print(f"stopped: {category}", file=sys.stderr)
            return 3
        else:
            category = classify(completed.stdout)
            if phase == "verify" and category == "transient_network":
                category = "verification_transient"

        attempts.append(
            Attempt(number, phase, category, completed.returncode, now_iso(), message)
        )
        last_category = category
        print(f"attempt {number}/{args.max_attempts} {phase}: {category}")

        retryable = category in {
            "transient_network",
            "verification_transient",
            "push_reported_success",
        }
        if not retryable:
            status = (
                "blocked_by_platform_policy"
                if category == "platform_policy"
                else "not_published"
            )
            write_json(
                args.json_out,
                result_payload(
                    status=status,
                    category=category,
                    repo=top,
                    remote=args.remote,
                    branch=branch,
                    local_sha=local_sha,
                    remote_sha=None,
                    attempts=attempts,
                    dirty=dirty,
                ),
            )
            print(
                f"stopped: permanent or unknown failure ({category})", file=sys.stderr
            )
            return 5 if category == "platform_policy" else 3

        if category == "push_reported_success":
            continue

        if number == args.max_attempts or not sleep_with_deadline(delay, deadline):
            break
        delay = min(delay * 2.0, args.max_delay)

    status = (
        "publication_uncertain" if push_reported_success else "temporarily_unavailable"
    )
    write_json(
        args.json_out,
        result_payload(
            status=status,
            category=last_category,
            repo=top,
            remote=args.remote,
            branch=branch,
            local_sha=local_sha,
            remote_sha=None,
            attempts=attempts,
            dirty=dirty,
        ),
    )
    print(f"stopped after bounded retries: {status} ({last_category})", file=sys.stderr)
    return 6 if push_reported_success else 4


if __name__ == "__main__":
    raise SystemExit(main())
