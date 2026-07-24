from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/github-push-recovery/scripts/retry_push.py"


def load_module():
    spec = importlib.util.spec_from_file_location("github_push_recovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        [*args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def test_failure_classification_prioritizes_permanent_failures() -> None:
    module = load_module()

    assert module.classify("fatal: Couldn't connect to server") == "transient_network"
    assert module.classify("Permission denied (publickey)") == "authentication"
    assert (
        module.classify("remote rejected: protected branch; failed to push some refs")
        == "remote_rejected"
    )
    assert (
        module.classify("rejected (non-fast-forward); failed to push some refs")
        == "non_fast_forward"
    )
    assert (
        module.classify('Model "codex-auto-review" is not supported')
        == "platform_policy"
    )


def test_failure_classification_covers_common_github_transport_failures() -> None:
    module = load_module()

    failures = (
        "fatal: unable to access 'https://github.com/o/r': Failed to connect to "
        "github.com port 443 after 21067 ms: Couldn't connect to server",
        "fatal: unable to access: getaddrinfo() thread failed to start",
        "error: RPC failed; curl 92 HTTP/2 stream 5 was not closed cleanly",
        "fetch-pack: unexpected disconnect while reading sideband packet",
        "fatal: unable to access: OpenSSL SSL_connect: SSL_ERROR_SYSCALL",
        "fatal: unable to access: Proxy CONNECT aborted",
    )

    assert {module.classify(message) for message in failures} == {
        "transient_network"
    }


def test_sanitize_redacts_url_credentials_and_tokens() -> None:
    module = load_module()

    sanitized = module.sanitize(
        "https://alice:secret@github.com/org/repo.git?access_token=abc123"
    )

    assert "secret" not in sanitized
    assert "abc123" not in sanitized
    assert (
        sanitized
        == "https://<redacted>@github.com/org/repo.git?access_token=<redacted>"
    )


def test_pushes_to_local_remote_and_verifies_exact_sha(tmp_path: Path) -> None:
    module = load_module()
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    result_path = tmp_path / "result.json"

    run("git", "init", "--bare", str(bare), cwd=tmp_path)
    run("git", "init", "--initial-branch", "main", str(work), cwd=tmp_path)
    run("git", "config", "user.name", "Skill Test", cwd=work)
    run("git", "config", "user.email", "skill-test@example.invalid", cwd=work)
    run("git", "commit", "--allow-empty", "-m", "test commit", cwd=work)
    run("git", "remote", "add", "origin", str(bare), cwd=work)

    exit_code = module.main(
        [
            "--repo",
            str(work),
            "--remote",
            "origin",
            "--execute",
            "--max-attempts",
            "3",
            "--initial-delay",
            "0.01",
            "--max-delay",
            "0.01",
            "--max-elapsed",
            "5",
            "--command-timeout",
            "2",
            "--json-out",
            str(result_path),
        ]
    )

    local_sha = run("git", "rev-parse", "HEAD", cwd=work)
    remote_sha = run("git", "rev-parse", "refs/heads/main", cwd=bare)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert remote_sha == local_sha
    assert result["status"] == "published_and_verified"
    assert result["local_sha"] == local_sha
    assert result["remote_sha"] == local_sha
    assert [attempt["phase"] for attempt in result["attempts"]] == ["push", "verify"]


def fake_git_runner(
    module,
    repo: Path,
    push_outputs: list[tuple[int, str]],
) -> tuple[Callable[..., subprocess.CompletedProcess[str]], str]:
    sha = "1" * 40

    def runner(git_bin, command_repo, args, timeout):
        del git_bin, command_repo, timeout
        if args == ["rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{repo}\n")
        if args == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{sha}\n")
        if args == ["symbolic-ref", "--quiet", "--short", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout="main\n")
        if args == ["check-ref-format", "--branch", "main"]:
            return subprocess.CompletedProcess(args, 0, stdout="main\n")
        if args == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="https://github.com/example/repo.git\n"
            )
        if args == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(args, 0, stdout="")
        if args[:2] == ["push", "--porcelain"]:
            returncode, output = push_outputs.pop(0)
            return subprocess.CompletedProcess(args, returncode, stdout=output)
        if args == ["ls-remote", "--exit-code", "origin", "refs/heads/main"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{sha}\trefs/heads/main\n"
            )
        raise AssertionError(f"unexpected git arguments: {args}")

    return runner, sha


def test_retries_transient_network_then_verifies(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    runner, sha = fake_git_runner(
        module,
        tmp_path,
        [
            (128, "fatal: Couldn't connect to server"),
            (128, "fatal: Recv failure: Connection was reset"),
            (0, "To https://github.com/example/repo.git\n"),
        ],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(module, "run_git", runner)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    result_path = tmp_path / "retry-result.json"

    exit_code = module.main(
        [
            "--repo",
            str(tmp_path),
            "--remote",
            "origin",
            "--execute",
            "--max-attempts",
            "4",
            "--initial-delay",
            "1",
            "--max-delay",
            "10",
            "--max-elapsed",
            "30",
            "--json-out",
            str(result_path),
        ]
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["remote_sha"] == sha
    assert [attempt["category"] for attempt in result["attempts"]] == [
        "transient_network",
        "transient_network",
        "push_reported_success",
        "verified",
    ]
    assert sleeps == [1.0, 2.0]


def test_authentication_failure_stops_without_retry(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_module()
    runner, _ = fake_git_runner(
        module,
        tmp_path,
        [(128, "fatal: Authentication failed")],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(module, "run_git", runner)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    exit_code = module.main(
        [
            "--repo",
            str(tmp_path),
            "--remote",
            "origin",
            "--execute",
            "--max-attempts",
            "8",
        ]
    )

    assert exit_code == 3
    assert sleeps == []


def test_existing_lock_prevents_concurrent_recovery(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_module()
    runner, _ = fake_git_runner(
        module,
        tmp_path,
        [(0, "push must not be reached")],
    )
    monkeypatch.setattr(module, "run_git", runner)
    lock_path = tmp_path / "recovery.lock"
    lock_path.write_text('{"pid": 123, "started_at": "test"}\n', encoding="utf-8")

    exit_code = module.main(
        [
            "--repo",
            str(tmp_path),
            "--remote",
            "origin",
            "--execute",
            "--lock-file",
            str(lock_path),
        ]
    )

    assert exit_code == 7
    assert lock_path.exists()
