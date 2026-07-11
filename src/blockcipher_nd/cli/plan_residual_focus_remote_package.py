from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ACTION_PLAN = Path(
    "outputs/local_audits/i1_present_r8_residual_focus_262k_action_plan.json"
)
DEFAULT_SOURCE_GATE = Path(
    "outputs/local_audits/i1_present_r8_residual_focus_262k_source_gate.json"
)
DEFAULT_OUTPUT_DIR = Path("configs/remote/generated")
DEFAULT_OUTPUT = Path(
    "outputs/local_audits/i1_present_r8_residual_focus_262k_remote_package.json"
)
DEFAULT_RUN_ID = "i1_present_r8_residual_focus_262k"
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
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args(argv)


def plan_residual_focus_remote_package(
    *,
    action_plan: Path,
    source_gate: Path,
    output_dir: Path,
    package_report: Path = DEFAULT_OUTPUT,
    run_id: str = DEFAULT_RUN_ID,
) -> dict[str, Any]:
    plan = _read_json(action_plan)
    gate = _read_json(source_gate)
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = output_dir / f"run_{run_id}_20260707.cmd"
    monitor = output_dir / f"monitor_{run_id}_20260707.sh"
    launch_wrapper = output_dir / f"launch_{run_id}_20260707.sh"
    local_artifact_root = str(
        plan.get(
            "artifact_root", "outputs/local_audits/i1_present_r8_residual_focus_262k"
        )
    )
    planned_outputs = _planned_outputs(plan)
    commands = [str(command) for command in plan.get("commands", [])]
    control_commands = [str(command) for command in plan.get("control_commands", [])]
    source_selection_commands = [
        str(command) for command in plan.get("source_selection_commands", [])
    ]
    source_selection_summary_command = str(
        plan.get("source_selection_summary_command", "")
    )
    source_selected_commands = [
        str(command) for command in plan.get("source_selected_commands", [])
    ]
    all_commands = [
        *commands,
        *control_commands,
        *source_selection_commands,
        *(
            [source_selection_summary_command]
            if source_selection_summary_command
            else []
        ),
        *source_selected_commands,
    ]
    windows_commands = [
        _windows_command_from_action(command, local_artifact_root=local_artifact_root)
        for command in all_commands
    ]
    progress_outputs = _command_option_paths(
        all_commands,
        local_artifact_root=local_artifact_root,
        option_names={"--progress-output"},
    )
    final_sync_outputs = _final_sync_outputs(
        all_commands,
        planned_outputs=planned_outputs,
        progress_outputs=progress_outputs,
        local_artifact_root=local_artifact_root,
    )
    blockers: list[str] = []
    if plan.get("status") != "pass":
        blockers.append("action_plan_not_pass")
    if gate.get("status") != "pass":
        blockers.append("source_gate_not_pass")
    if _has_medium_cache_command_without_workers(plan, all_commands):
        blockers.append("medium_cache_workers_not_configured")
    if _contains_unsafe_command(windows_commands):
        blockers.append("unsafe_generated_command")
    launcher.write_text(
        _launcher_text(windows_commands, run_id=run_id), encoding="utf-8"
    )
    monitor.write_text(
        _monitor_text(
            run_id=run_id,
            local_artifact_root=local_artifact_root,
            planned_outputs=planned_outputs,
            progress_outputs=progress_outputs,
            final_sync_outputs=final_sync_outputs,
        ),
        encoding="utf-8",
    )
    monitor.chmod(0o755)
    launch_allowed = not blockers
    launch_wrapper.write_text(
        _launch_wrapper_text(
            package_report=package_report,
            launcher=launcher,
            monitor=monitor,
            run_id=run_id,
        ),
        encoding="utf-8",
    )
    launch_wrapper.chmod(0o755)
    return {
        "status": "pass" if launch_allowed else "pending",
        "decision": "residual_focus_remote_package_ready"
        if launch_allowed
        else "residual_focus_remote_package_blocked",
        "run_id": run_id,
        "action_plan": str(action_plan),
        "source_gate": str(source_gate),
        "source_gate_status": str(gate.get("status", "")),
        "source_gate_errors": [str(error) for error in gate.get("errors", [])],
        "launcher": str(launcher),
        "launch_wrapper": str(launch_wrapper),
        "monitor": str(monitor),
        "command_count": len(commands),
        "control_command_count": len(control_commands),
        "source_selection_command_count": len(source_selection_commands),
        "source_selected_command_count": len(source_selected_commands),
        "planned_output_count": len(planned_outputs),
        "progress_sync_count": len(progress_outputs),
        "final_sync_count": len(final_sync_outputs),
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


def _command_option_paths(
    commands: list[str],
    *,
    local_artifact_root: str,
    option_names: set[str],
) -> list[str]:
    paths: list[str] = []
    for command in commands:
        tokens = shlex.split(command)
        for index, token in enumerate(tokens[:-1]):
            if token in option_names and tokens[index + 1].startswith(
                local_artifact_root
            ):
                paths.append(tokens[index + 1])
    return sorted(set(paths))


def _has_medium_cache_command_without_workers(
    plan: dict[str, Any], commands: list[str]
) -> bool:
    if not _is_medium_or_larger_plan(plan):
        return False
    for command in commands:
        tokens = shlex.split(command)
        if not any(
            token.endswith("scripts/export-bit-sensitivity-features")
            or token.endswith("scripts/export-checkpoint-scores")
            for token in tokens
        ):
            continue
        if "--dataset-cache-root" not in tokens:
            continue
        worker_count = _int_option(tokens, "--dataset-cache-workers")
        if worker_count is None or worker_count < 2:
            return True
    return False


def _is_medium_or_larger_plan(plan: dict[str, Any]) -> bool:
    expected_score_rows = plan.get("expected_score_rows")
    if isinstance(expected_score_rows, int) and expected_score_rows >= 65_536:
        return True
    if (
        isinstance(expected_score_rows, str)
        and expected_score_rows.isdigit()
        and int(expected_score_rows) >= 65_536
    ):
        return True
    text = " ".join(
        str(value)
        for value in (
            plan.get("artifact_root", ""),
            plan.get("claim_scope", ""),
            plan.get("decision", ""),
        )
    )
    return "262k" in text or "262144" in text


def _int_option(tokens: list[str], option_name: str) -> int | None:
    for index, token in enumerate(tokens[:-1]):
        if token != option_name:
            continue
        try:
            return int(tokens[index + 1])
        except ValueError:
            return None
    return None


def _final_sync_outputs(
    commands: list[str],
    *,
    planned_outputs: list[str],
    progress_outputs: list[str],
    local_artifact_root: str,
) -> list[str]:
    command_outputs = _command_option_paths(
        commands,
        local_artifact_root=local_artifact_root,
        option_names={
            "--output",
            "--output-dir",
            "--output-report",
            "--output-summary-feature-dir",
            "--output-train-dir",
            "--output-validation-dir",
        },
    )
    outputs = [*planned_outputs, *progress_outputs]
    for path in command_outputs:
        if _is_large_intermediate_artifact(path):
            continue
        outputs.append(path)
    return sorted(set(outputs))


def _is_large_intermediate_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(
        marker in normalized
        for marker in (
            "/dataset_cache/",
            "trail_position_stats_features",
            "span_blocks",
        )
    )


def _windows_command_from_action(command: str, *, local_artifact_root: str) -> str:
    tokens = shlex.split(command)
    if (
        len(tokens) >= 3
        and tokens[0].startswith("UV_CACHE_DIR=")
        and tokens[1:3] == ["uv", "run"]
    ):
        tokens = ["%PYTHON_EXE%"] + tokens[3:]
    else:
        tokens = [
            _translate_path_token(token, local_artifact_root=local_artifact_root)
            for token in tokens
        ]
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
    normalized_suffix = suffix.replace("/", "\\")
    return f"{prefix}\\{normalized_suffix}"


def _contains_unsafe_command(commands: list[str]) -> bool:
    return any(
        "cmd.exe /k" in command.lower() or " ssh " in f" {command.lower()} "
        for command in commands
    )


def _launcher_text(commands: list[str], *, run_id: str) -> str:
    lines = [
        "@echo off",
        "setlocal EnableExtensions EnableDelayedExpansion",
        "",
        f"set RUN_ID={run_id}",
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
        'if not exist "%RUN_ROOT%" mkdir "%RUN_ROOT%"',
        'if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"',
        'if not exist "%ARTIFACT_ROOT%" mkdir "%ARTIFACT_ROOT%"',
        'echo run_id=%RUN_ID%>"%LOG_DIR%\\%RUN_ID%_launch_env.txt"',
        'echo source_root=%SOURCE_ROOT%>>"%LOG_DIR%\\%RUN_ID%_launch_env.txt"',
        'echo artifact_root=%ARTIFACT_ROOT%>>"%LOG_DIR%\\%RUN_ID%_launch_env.txt"',
        "",
        'if exist "%SOURCE_ROOT%\\.git" (',
        '  cd /d "%SOURCE_ROOT%" || goto failed',
        '  git status --short --branch > "%LOG_DIR%\\%RUN_ID%_git_status_before_run.txt" 2>&1',
        '  git fetch origin %BRANCH% > "%LOG_DIR%\\%RUN_ID%_git_fetch_stdout.txt" 2> "%LOG_DIR%\\%RUN_ID%_git_fetch_stderr.txt" || goto failed',
        '  git checkout %BRANCH% > "%LOG_DIR%\\%RUN_ID%_git_checkout_stdout.txt" 2> "%LOG_DIR%\\%RUN_ID%_git_checkout_stderr.txt" || goto failed',
        '  git pull --ff-only origin %BRANCH% >> "%LOG_DIR%\\%RUN_ID%_git_fetch_stdout.txt" 2>> "%LOG_DIR%\\%RUN_ID%_git_fetch_stderr.txt" || goto failed',
        ") else (",
        '  if exist "%SOURCE_ROOT%" rmdir /s /q "%SOURCE_ROOT%"',
        '  git clone --branch %BRANCH% "%REPO_URL%" "%SOURCE_ROOT%" > "%LOG_DIR%\\%RUN_ID%_git_clone_stdout.txt" 2> "%LOG_DIR%\\%RUN_ID%_git_clone_stderr.txt" || goto failed',
        '  cd /d "%SOURCE_ROOT%" || goto failed',
        ")",
        "",
        'git rev-parse HEAD > "%LOG_DIR%\\%RUN_ID%_git_revision.txt" 2>&1',
        "set PYTHONPATH=%SOURCE_ROOT%\\src;%PYTHONPATH%",
        'echo pythonpath=%PYTHONPATH%>>"%LOG_DIR%\\%RUN_ID%_launch_env.txt"',
        'echo started>"%LOG_DIR%\\%RUN_ID%_started.marker"',
        "",
    ]
    for index, command in enumerate(commands):
        lines.extend(
            [
                f'echo command_{index}>"%LOG_DIR%\\%RUN_ID%_command_{index}.marker"',
                f'{command} > "%LOG_DIR%\\%RUN_ID%_command_{index}_stdout.txt" 2> "%LOG_DIR%\\%RUN_ID%_command_{index}_stderr.txt"',
                "if errorlevel 1 goto failed",
                "",
            ]
        )
    lines.extend(
        [
            'echo done>"%LOG_DIR%\\%RUN_ID%_done.marker"',
            "exit /b 0",
            "",
            ":failed",
            'echo failed>"%LOG_DIR%\\%RUN_ID%_failed.marker"',
            "exit /b 1",
            "",
        ]
    )
    return "\n".join(lines)


def _monitor_text(
    *,
    run_id: str,
    local_artifact_root: str,
    planned_outputs: list[str],
    progress_outputs: list[str],
    final_sync_outputs: list[str],
) -> str:
    expected_checks = "\n".join(_missing_check_line(path) for path in planned_outputs)
    if not expected_checks:
        expected_checks = "  missing=0"
    immediate_sync_pairs = _sync_pair_lines(
        [*progress_outputs, *_file_like_outputs(planned_outputs)],
        local_artifact_root=local_artifact_root,
    )
    final_sync_pairs = _sync_pair_lines(
        final_sync_outputs, local_artifact_root=local_artifact_root
    )
    return f"""#!/usr/bin/env bash
set -u

RUN_ID="{run_id}"
REMOTE="lxy-a6000"
REMOTE_RUN_ROOT="{REMOTE_RUNS_ROOT_SCP}/${{RUN_ID}}"
REMOTE_ARTIFACT_ROOT="{REMOTE_RUNS_ROOT_SCP}/{run_id}/artifacts"
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
  sync_path_list <<SYNC_IMMEDIATE_ARTIFACTS
{immediate_sync_pairs}
SYNC_IMMEDIATE_ARTIFACTS
}}

sync_final_artifacts() {{
  sync_path_list <<SYNC_FINAL_ARTIFACTS
{final_sync_pairs}
SYNC_FINAL_ARTIFACTS
}}

sync_path_list() {{
  while IFS='|' read -r remote_path local_path; do
    if [[ -z "${{remote_path}}" || -z "${{local_path}}" ]]; then
      continue
    fi
    mkdir -p "$(dirname "${{local_path}}")"
    scp -r "${{REMOTE}}:${{remote_path}}" "${{local_path}}" >> "${{MONITOR_DIR}}/scp.log" 2>> "${{MONITOR_DIR}}/scp_stderr.log" || true
  done
}}

count_missing_outputs() {{
  missing=0
{expected_checks}
}}

while true; do
  echo "$(timestamp) sync" >> "${{MONITOR_DIR}}/monitor.log"
  sync_artifacts

  if compgen -G "${{LOCAL_ROOT}}/logs/*failed.marker" > /dev/null; then
    echo "$(timestamp) failed" >> "${{MONITOR_DIR}}/monitor.log"
    exit 1
  fi

  count_missing_outputs

  if compgen -G "${{LOCAL_ROOT}}/logs/*done.marker" > /dev/null; then
    sync_final_artifacts
    count_missing_outputs
    if [[ "${{missing}}" -eq 0 ]]; then
      echo "$(timestamp) result_ready" >> "${{MONITOR_DIR}}/monitor.log"
      exit 0
    fi
    echo "$(timestamp) completed_missing_outputs missing=${{missing}}" >> "${{MONITOR_DIR}}/monitor.log"
    exit 2
  fi

  if [[ "${{missing}}" -eq 0 ]]; then
    echo "$(timestamp) outputs_ready_waiting_done" >> "${{MONITOR_DIR}}/monitor.log"
  else
    echo "$(timestamp) running missing=${{missing}}" >> "${{MONITOR_DIR}}/monitor.log"
  fi
  sleep 840
done
"""


def _missing_check_line(path: str) -> str:
    test_operator = "-f" if _is_file_like_output(path) else "-d"
    return f'  [[ {test_operator} "{path}" ]] || missing=$((missing + 1))'


def _file_like_outputs(paths: list[str]) -> list[str]:
    return [path for path in paths if _is_file_like_output(path)]


def _is_file_like_output(path: str) -> bool:
    suffix = Path(path.replace("\\", "/")).suffix
    return bool(suffix)


def _sync_pair_lines(paths: list[str], *, local_artifact_root: str) -> str:
    lines = []
    for path in sorted(set(paths)):
        if not path.startswith(local_artifact_root):
            continue
        suffix = path[len(local_artifact_root) :].lstrip("/")
        lines.append(f"${{REMOTE_ARTIFACT_ROOT}}/{suffix}|{path}")
    return "\n".join(lines)


def _launch_wrapper_text(
    *, package_report: Path, launcher: Path, monitor: Path, run_id: str
) -> str:
    launcher_name = launcher.name
    monitor_session = f"monitor_{run_id}_20260707"
    remote_run_root = f"{REMOTE_RUNS_ROOT}\\{run_id}"
    remote_source_root = f"{remote_run_root}\\source"
    remote_cmd = (
        f"{remote_run_root}\\source\\configs\\remote\\generated\\{launcher_name}"
    )
    local_monitor_dir = f"outputs/remote_results/{run_id}/monitor"
    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_ID="{run_id}"
PACKAGE_REPORT="{package_report}"
LOCAL_MONITOR_DIR="{local_monitor_dir}"
REMOTE_RUN_ROOT="{remote_run_root}"
REMOTE_SOURCE_ROOT="{remote_source_root}"
REMOTE_RUN_CMD="{remote_cmd}"
REPO_URL="{REPO_URL}"
MONITOR_SCRIPT="{monitor}"
MONITOR_SESSION="{monitor_session}"

mkdir -p "${{LOCAL_MONITOR_DIR}}"
echo "$(date -Is) launch_start" >> "${{LOCAL_MONITOR_DIR}}/launch.log"

launch_allowed="$(UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import json; print(json.load(open('${{PACKAGE_REPORT}}')).get('launch_allowed'))")"
if [[ "${{launch_allowed}}" != "True" ]]; then
  echo "$(date -Is) launch_blocked source_gate_not_pass launch_allowed=${{launch_allowed}}" >> "${{LOCAL_MONITOR_DIR}}/launch.log"
  echo "blocked" > "${{LOCAL_MONITOR_DIR}}/launch_blocked.marker"
  exit 3
fi

if tmux has-session -t "${{MONITOR_SESSION}}" >/dev/null 2>&1; then
  echo "$(date -Is) monitor_already_running monitor=${{MONITOR_SESSION}}" >> "${{LOCAL_MONITOR_DIR}}/launch.log"
else
  tmux new-session -d -s {monitor_session} "${{MONITOR_SCRIPT}}"
  echo "$(date -Is) monitor_started monitor=${{MONITOR_SESSION}}" >> "${{LOCAL_MONITOR_DIR}}/launch.log"
fi

set +e
ssh lxy-a6000 "cmd.exe /c if not exist \\"${{REMOTE_RUN_ROOT}}\\" mkdir \\"${{REMOTE_RUN_ROOT}}\\" && if exist \\"${{REMOTE_SOURCE_ROOT}}\\.git\\" (cd /d \\"${{REMOTE_SOURCE_ROOT}}\\" && git fetch origin main && git checkout main && git pull --ff-only origin main) else (git clone --branch main \\"${{REPO_URL}}\\" \\"${{REMOTE_SOURCE_ROOT}}\\") && call \\"${{REMOTE_RUN_CMD}}\\"" \\
  >> "${{LOCAL_MONITOR_DIR}}/launch.log" \\
  2>> "${{LOCAL_MONITOR_DIR}}/launch_stderr.log"
status=$?
set -e

if [[ ${{status}} -ne 0 ]]; then
  echo "$(date -Is) launch_failed status=${{status}}" >> "${{LOCAL_MONITOR_DIR}}/launch.log"
  echo "failed" > "${{LOCAL_MONITOR_DIR}}/launch_failed.marker"
  exit "${{status}}"
fi

echo "$(date -Is) launch_done monitor=${{MONITOR_SESSION}}" >> "${{LOCAL_MONITOR_DIR}}/launch.log"
echo "done" > "${{LOCAL_MONITOR_DIR}}/launch_done.marker"
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = plan_residual_focus_remote_package(
        action_plan=args.action_plan,
        source_gate=args.source_gate,
        output_dir=args.output_dir,
        package_report=args.output,
        run_id=args.run_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
