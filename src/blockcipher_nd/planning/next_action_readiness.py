from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blockcipher_nd.cli.check_remote_readiness import remote_readiness_report


DEFAULT_CONFIG_KEYS: tuple[tuple[str, str], ...] = (("primary", "launch_remote_config"),)
STAGED_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
    ("stage_a", "stage_a_remote_config"),
    ("primary", "launch_remote_config"),
)


def next_action_readiness_report(
    *,
    summary_path: Path,
    report: dict[str, Any],
    config_keys: tuple[tuple[str, str], ...] = DEFAULT_CONFIG_KEYS,
    implementation_checklist: list[str] | None = None,
) -> dict[str, Any]:
    next_action = report.get("next_action", {})
    if not isinstance(next_action, dict):
        next_action = {}

    readiness_reports = readiness_reports_for_next_action(next_action, config_keys=config_keys)
    should_launch_remote = bool(next_action.get("should_launch_remote"))
    requires_implementation = bool(next_action.get("requires_implementation"))
    remote_readiness_pass = bool(readiness_reports) and all(
        item["readiness"]["status"] == "pass" for item in readiness_reports
    )
    launch_artifacts_pass = bool(readiness_reports) and all(
        launch_artifacts_passed(item.get("launch_artifacts", {})) for item in readiness_reports
    )
    readiness_pass = remote_readiness_pass and launch_artifacts_pass
    return {
        "status": "pass" if (not should_launch_remote or readiness_pass) else "fail",
        "summary": str(summary_path),
        "run_id": report.get("run_id"),
        "decision": report.get("decision"),
        "action": report.get("action"),
        "branch": next_action.get("branch"),
        "should_launch_remote": should_launch_remote,
        "requires_implementation": requires_implementation,
        "readiness_pass": readiness_pass,
        "remote_readiness_pass": remote_readiness_pass,
        "launch_artifacts_pass": launch_artifacts_pass,
        "readiness_reports": readiness_reports,
        "implementation_checklist": implementation_checklist or [],
        "next_action": next_action,
        "claim_scope": report.get("claim_scope"),
        "errors": next_action_readiness_errors(
            should_launch_remote=should_launch_remote,
            readiness_reports=readiness_reports,
        ),
    }


def readiness_reports_for_next_action(
    next_action: dict[str, Any],
    *,
    config_keys: tuple[tuple[str, str], ...] = DEFAULT_CONFIG_KEYS,
) -> list[dict[str, Any]]:
    readiness_reports: list[dict[str, Any]] = []
    for role, key in config_keys:
        config = next_action.get(key)
        if not isinstance(config, str) or not config:
            continue
        config_path = Path(config)
        readiness_reports.append(
            {
                "role": role,
                "config": config,
                "launch_artifacts": launch_artifacts(config_path),
                "readiness": readiness_report(config_path),
            }
        )
    _attach_shared_stage_launch_artifacts(readiness_reports)
    return readiness_reports


def next_action_readiness_errors(
    *,
    should_launch_remote: bool,
    readiness_reports: list[dict[str, Any]],
) -> list[str]:
    if not should_launch_remote:
        return []
    if not readiness_reports:
        return ["next_action requests remote launch but no launch_remote_config was provided"]
    errors: list[str] = []
    for item in readiness_reports:
        readiness = item["readiness"]
        if readiness["status"] != "pass":
            errors.append(f"{item['role']} readiness failed: {readiness.get('errors', [])}")
        artifacts = item.get("launch_artifacts", {})
        if isinstance(artifacts, dict) and artifacts.get("status") != "pass":
            errors.append(f"{item['role']} launch artifacts failed: {artifacts.get('errors', [])}")
    return errors


def readiness_report(config_path: Path) -> dict[str, Any]:
    try:
        return remote_readiness_report(config_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "config": str(config_path),
            "errors": [str(exc)],
            "warnings": [],
        }


def launch_artifacts(config_path: Path) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "unknown",
            "config": str(config_path),
            "errors": [str(exc)],
        }

    monitor_name = config.get("monitor_script_name")
    monitor_path = Path("configs/remote/generated") / str(monitor_name) if isinstance(monitor_name, str) else None
    launcher_path = generated_launcher_for_monitor(monitor_path) if monitor_path is not None else None
    errors: list[str] = []
    if monitor_path is None:
        errors.append("remote config does not declare monitor_script_name")
    elif not monitor_path.exists():
        errors.append(f"generated monitor script missing: {monitor_path}")
    if launcher_path is None:
        errors.append("could not infer generated launcher script from monitor_script_name")
    elif not launcher_path.exists():
        errors.append(f"generated launcher script missing: {launcher_path}")
    return {
        "status": "pass" if not errors else "fail",
        "config": str(config_path),
        "launcher": str(launcher_path) if launcher_path is not None else None,
        "launcher_exists": bool(launcher_path is not None and launcher_path.exists()),
        "monitor": str(monitor_path) if monitor_path is not None else None,
        "monitor_exists": bool(monitor_path is not None and monitor_path.exists()),
        "errors": errors,
    }


def _attach_shared_stage_launch_artifacts(readiness_reports: list[dict[str, Any]]) -> None:
    primary = next((item for item in readiness_reports if item.get("role") == "primary"), None)
    if not isinstance(primary, dict):
        return
    primary_artifacts = primary.get("launch_artifacts", {})
    if not launch_artifacts_passed(primary_artifacts):
        return

    for item in readiness_reports:
        if item is primary or item.get("role") == "primary":
            continue
        artifacts = item.get("launch_artifacts", {})
        if launch_artifacts_passed(artifacts):
            continue
        shared = _shared_stage_launch_artifacts(
            stage_config_path=Path(str(item.get("config", ""))),
            primary_artifacts=primary_artifacts,
        )
        if shared is not None:
            item["launch_artifacts"] = shared


def _shared_stage_launch_artifacts(
    *,
    stage_config_path: Path,
    primary_artifacts: dict[str, Any],
) -> dict[str, Any] | None:
    launcher = primary_artifacts.get("launcher")
    monitor = primary_artifacts.get("monitor")
    if not isinstance(launcher, str) or not isinstance(monitor, str):
        return None
    launcher_path = Path(launcher)
    monitor_path = Path(monitor)
    if not launcher_path.exists() or not monitor_path.exists():
        return None

    try:
        stage_config = json.loads(stage_config_path.read_text(encoding="utf-8"))
        launcher_text = launcher_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, json.JSONDecodeError):
        return None

    stage_run_id = stage_config.get("run_id")
    stage_plan = stage_config.get("plan")
    if not isinstance(stage_run_id, str) or not isinstance(stage_plan, str):
        return None

    normalized_stage_plan = stage_plan.replace("/", "\\")
    errors: list[str] = []
    if stage_run_id not in launcher_text:
        errors.append(f"shared launcher does not reference stage run_id: {stage_run_id}")
    if normalized_stage_plan not in launcher_text:
        errors.append(f"shared launcher does not reference stage plan: {normalized_stage_plan}")
    if errors:
        return None

    shared = dict(primary_artifacts)
    shared.update(
        {
            "config": str(stage_config_path),
            "shared_with_primary_launcher": True,
            "stage_run_id": stage_run_id,
            "stage_plan": stage_plan,
            "errors": [],
            "status": "pass",
        }
    )
    return shared


def generated_launcher_for_monitor(monitor_path: Path | None) -> Path | None:
    if monitor_path is None:
        return None
    name = monitor_path.name
    if not name.startswith("monitor_") or not name.endswith(".sh"):
        return None
    return monitor_path.with_name("run_" + name.removeprefix("monitor_").removesuffix(".sh") + ".cmd")


def launch_artifact_checklist(readiness_reports: list[dict[str, Any]]) -> list[str]:
    steps: list[str] = []
    for report in readiness_reports:
        artifacts = report.get("launch_artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        launcher = artifacts.get("launcher")
        monitor = artifacts.get("monitor")
        if artifacts.get("status") == "pass":
            steps.append(f"Use generated launcher `{launcher}` and monitor `{monitor}`.")
        else:
            steps.append(f"Fix generated launch artifacts before launch: {artifacts.get('errors', [])}.")
    return steps


def launch_artifacts_passed(artifacts: Any) -> bool:
    return isinstance(artifacts, dict) and artifacts.get("status") == "pass"
