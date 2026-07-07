from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ACTION_PLAN = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_action_plan.json")
DEFAULT_SOURCE_GATE = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_source_gate.json")
DEFAULT_OUTPUT_DIR = Path("configs/remote/generated")
DEFAULT_OUTPUT = Path("outputs/local_audits/i1_present_r8_residual_focus_262k_remote_package.json")
RUN_ID = "i1_present_r8_residual_focus_262k"
REMOTE_RUNS_ROOT = "G:\\lxy\\blockcipher-structure-adaptive-nd-runs"
REMOTE_RUNS_ROOT_SCP = "G:/lxy/blockcipher-structure-adaptive-nd-runs"
REPO_URL = "git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a remote execution package for the PRESENT r8 residual-focus 262k gate."
    )
    parser.add_argument("--action-plan", type=Path, default=DEFAULT_ACTION_PLAN)
    parser.add_argument("--source-gate", type=Path, default=DEFAULT_SOURCE_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def plan_residual_focus_remote_package(
    *,
    action_plan: Path,
    source_gate: Path,
    output_dir: Path,
) -> dict[str, Any]:
    plan = _read_json(action_plan)
    gate = _read_json(source_gate)
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = output_dir / f"run_{RUN_ID}_20260707.cmd"
    monitor = output_dir / f"monitor_{RUN_ID}_20260707.sh"
    local_artifact_root = str(plan.get("artifact_root", "outputs/local_audits/i1_present_r8_residual_focus_262k"))
    planned_outputs = _planned_outputs(plan)
    commands = [str(command) for command in plan.get("commands", [])]
    control_commands = [str(command) for command in plan.get("control_commands", [])]
    all_commands = commands + control_commands
    windows_commands = [
        _windows_command_from_action(command, local_artifact_root=local_artifact_root)
        for command in all_commands
    ]
    blockers: list[str] = []
    if plan.get("status") != "pass":
        blockers.append("action_plan_not_pass")
    if gate.get("status") != "pass":
        blockers.append("source_gate_not_pass")
    if _contains_unsafe_command(windows_commands):
        blockers.append("unsafe_generated_command")
    launcher.write_text(_launcher_text(windows_commands), encoding="utf-8")
    monitor.write_text(
        _monitor_text(local_artifact_root=local_artifact_root, planned_outputs=planned_outputs),
        encoding="utf-8",
    )
    launch_allowed = not blockers
    return {
        "status": "pass" if launch_allowed else "pending",
        "decision": "residual_focus_remote_package_ready" if launch_allowed else "residual_focus_remote_package_blocked",
        "run_id": RUN_ID,
        "action_plan": str(action_plan),
        "source_gate": str(source_gate),
        "source_gate_status": str(gate.get("status", "")),
        "source_gate_errors": [str(error) for error in gate.get("errors", [])],
        "launcher": str(launcher),
        "monitor": str(monitor),
        "command_count": len(commands),
        "control_command_count": len(control_commands),
        "planned_output_count": len(planned_outputs),
        "blockers": blockers,
        "launch_allowed": launch_allowed,
        "launch_policy": (
            "Prepared only. Launch with cmd.exe /c from pushed source after "
            "check-launch-source passes; no cmd.exe /k, no dirty overlay, no main-thread SSH polling."
        ),
        "claim_scope": (
            "remote package generation only; does not launch remote jobs, does not prove residual-focus "
            "262k outputs, and does not make a medium or formal SPN/PRESENT claim"
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _planned_outputs(plan: dict[str, Any]) -> list[str]:
    outputs: list[str] = []
    for seed in plan.get("seeds", []):
        if not isinstance(seed, dict):
            continue
        planned = seed.get("planned_outputs", {})
        if isinstance(planned, dict):
            outputs.extend(str(value) for value in planned.values())
    return sorted(set(outputs))


def _windows_command_from_action(command: str, *, local_artifact_root: str) -> str:
    tokens = shlex.split(command)
    if len(tokens) >= 3 and tokens[0].startswith("UV_CACHE_DIR=") and tokens[1:3] == ["uv", "run"]:
        tokens = ["%PYTHON_EXE%"] + tokens[3:]
    else:
        tokens = [_translate_path_token(token, local_artifact_root=local_artifact_root) for token in tokens]
        return subprocess.list2cmdline(tokens)
    translated = [tokens[0]] + [
        _translate_path_token(token, local_artifact_root=local_artifact_root)
        for token in tokens[1:]
    ]
    return subprocess.list2cmdline(translated)


def _translate_path_token(token: str, *, local_artifact_root: str) -> str:
    if token.startswith(local_artifact_root):
        suffix = token[len(local_artifact_root) :].lstrip("/")
        return _join_windows("%ARTIFACT_ROOT%", suffix)
    if token.startswith("outputs/remote_results/"):
        suffix = token[len("outputs/remote_results/") :]
        parts = suffix.split("/", 1)
        run_id = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        return _join_windows(f"{REMOTE_RUNS_ROOT}\\{run_id}", rest)
    if token.startswith(("scripts/", "configs/")):
        return token.replace("/", "\\")
    return token


def _join_windows(prefix: str, suffix: str) -> str:
    if not suffix:
        return prefix
    return f"{prefix}\\{suffix.replace('/', '\\')}"


def _contains_unsafe_command(commands: list[str]) -> bool:
    return any("cmd.exe /k" in command.lower() or " ssh " in f" {command.lower()} " for command in commands)


def _launcher_text(commands: list[str]) -> str:
    lines = [
        "@echo off",
        "setlocal EnableExtensions EnableDelayedExpansion",
        "",
        f"set RUN_ID={RUN_ID}",
        f"set REPO_URL={REPO_URL}",
        "set BRANCH=main",
        f"set RUN_ROOT={REMOTE_RUNS_ROOT}\\%RUN_ID%",
        "set SOURCE_ROOT=%RUN_ROOT%\\source",
        "set ARTIFACT_ROOT=%RUN_ROOT%\\artifacts",
        "set LOG_DIR=%RUN_ROOT%\\logs",
        "set PYTHON_EXE=F:\\Anaconda\\envs\\DWT\\torch310\\python.exe",
        "set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519",
        "set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new",
        "",
        "if not exist \"%RUN_ROOT%\" mkdir \"%RUN_ROOT%\"",
        "if not exist \"%LOG_DIR%\" mkdir \"%LOG_DIR%\"",
        "if not exist \"%ARTIFACT_ROOT%\" mkdir \"%ARTIFACT_ROOT%\"",
        "echo run_id=%RUN_ID%>\"%LOG_DIR%\\%RUN_ID%_launch_env.txt\"",
        "echo source_root=%SOURCE_ROOT%>>\"%LOG_DIR%\\%RUN_ID%_launch_env.txt\"",
        "echo artifact_root=%ARTIFACT_ROOT%>>\"%LOG_DIR%\\%RUN_ID%_launch_env.txt\"",
        "",
        "if exist \"%SOURCE_ROOT%\\.git\" (",
        "  cd /d \"%SOURCE_ROOT%\" || goto failed",
        "  git status --short --branch > \"%LOG_DIR%\\%RUN_ID%_git_status_before_run.txt\" 2>&1",
        "  git fetch origin %BRANCH% > \"%LOG_DIR%\\%RUN_ID%_git_fetch_stdout.txt\" 2> \"%LOG_DIR%\\%RUN_ID%_git_fetch_stderr.txt\" || goto failed",
        "  git checkout %BRANCH% > \"%LOG_DIR%\\%RUN_ID%_git_checkout_stdout.txt\" 2> \"%LOG_DIR%\\%RUN_ID%_git_checkout_stderr.txt\" || goto failed",
        "  git pull --ff-only origin %BRANCH% >> \"%LOG_DIR%\\%RUN_ID%_git_fetch_stdout.txt\" 2>> \"%LOG_DIR%\\%RUN_ID%_git_fetch_stderr.txt\" || goto failed",
        ") else (",
        "  if exist \"%SOURCE_ROOT%\" rmdir /s /q \"%SOURCE_ROOT%\"",
        "  git clone --branch %BRANCH% \"%REPO_URL%\" \"%SOURCE_ROOT%\" > \"%LOG_DIR%\\%RUN_ID%_git_clone_stdout.txt\" 2> \"%LOG_DIR%\\%RUN_ID%_git_clone_stderr.txt\" || goto failed",
        "  cd /d \"%SOURCE_ROOT%\" || goto failed",
        ")",
        "",
        "git rev-parse HEAD > \"%LOG_DIR%\\%RUN_ID%_git_revision.txt\" 2>&1",
        "echo started>\"%LOG_DIR%\\%RUN_ID%_started.marker\"",
        "",
    ]
    for index, command in enumerate(commands):
        lines.extend(
            [
                f"echo command_{index}>\"%LOG_DIR%\\%RUN_ID%_command_{index}.marker\"",
                f"{command} > \"%LOG_DIR%\\%RUN_ID%_command_{index}_stdout.txt\" 2> \"%LOG_DIR%\\%RUN_ID%_command_{index}_stderr.txt\"",
                "if errorlevel 1 goto failed",
                "",
            ]
        )
    lines.extend(
        [
            "echo done>\"%LOG_DIR%\\%RUN_ID%_done.marker\"",
            "exit /b 0",
            "",
            ":failed",
            "echo failed>\"%LOG_DIR%\\%RUN_ID%_failed.marker\"",
            "exit /b 1",
            "",
        ]
    )
    return "\n".join(lines)


def _monitor_text(*, local_artifact_root: str, planned_outputs: list[str]) -> str:
    expected_checks = "\n".join(
        f"  [[ -f \"{path}\" ]] || missing=$((missing + 1))"
        for path in planned_outputs
    )
    if not expected_checks:
        expected_checks = "  missing=0"
    return f"""#!/usr/bin/env bash
set -u

RUN_ID="{RUN_ID}"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="{REMOTE_RUNS_ROOT_SCP}/${{RUN_ID}}"
REMOTE_ARTIFACT_ROOT="{REMOTE_RUNS_ROOT_SCP}/{RUN_ID}/artifacts"
LOCAL_ARTIFACT_ROOT="{local_artifact_root}"
LOCAL_ROOT="outputs/remote_results/${{RUN_ID}}"
MONITOR_DIR="${{LOCAL_ROOT}}/monitor"

mkdir -p "${{LOCAL_ARTIFACT_ROOT}}" "${{LOCAL_ROOT}}" "${{MONITOR_DIR}}" "${{LOCAL_ROOT}}/logs"
touch "${{MONITOR_DIR}}/monitor.log"

timestamp() {{
  date --iso-8601=seconds
}}

sync_artifacts() {{
  scp -r "${{REMOTE}}:${{REMOTE_RUN_ROOT}}/logs" "${{LOCAL_ROOT}}/" >> "${{MONITOR_DIR}}/scp.log" 2>> "${{MONITOR_DIR}}/scp_stderr.log" || true
  scp -r "${{REMOTE}}:${{REMOTE_ARTIFACT_ROOT}}/"* "${{LOCAL_ARTIFACT_ROOT}}/" >> "${{MONITOR_DIR}}/scp.log" 2>> "${{MONITOR_DIR}}/scp_stderr.log" || true
}}

while true; do
  echo "$(timestamp) sync" >> "${{MONITOR_DIR}}/monitor.log"
  sync_artifacts

  if compgen -G "${{LOCAL_ROOT}}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) failed" >> "${{MONITOR_DIR}}/monitor.log"
    exit 1
  fi

  missing=0
{expected_checks}
  if [[ "${{missing}}" -eq 0 ]]; then
    echo "$(timestamp) result_ready" >> "${{MONITOR_DIR}}/monitor.log"
    exit 0
  fi

  if compgen -G "${{LOCAL_ROOT}}/logs/*done.marker" > /dev/null; then
    echo "$(timestamp) completed_missing_outputs missing=${{missing}}" >> "${{MONITOR_DIR}}/monitor.log"
    exit 2
  fi

  echo "$(timestamp) running missing=${{missing}}" >> "${{MONITOR_DIR}}/monitor.log"
  sleep 840
done
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan_residual_focus_remote_package(
        action_plan=args.action_plan,
        source_gate=args.source_gate,
        output_dir=args.output_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
