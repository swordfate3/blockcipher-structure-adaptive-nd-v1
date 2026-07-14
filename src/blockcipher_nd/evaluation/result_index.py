from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


DEFAULT_RESULT_ROOTS = (
    "local_smoke",
    "local_diagnostic",
    "smoke",
    "remote_results",
    "remote_results_incomplete",
)

ARTIFACT_LABELS = {
    "curves": "曲线",
    "gate": "门控",
    "validation": "验证",
    "results": "结果",
    "history": "历史",
    "progress": "进度",
}

DECISION_LABELS = {
    "e4_r3_two_seed_medium_signal_confirmed": "双 seed 中等规模迁移信号已确认",
    "e4_r3_two_seed_medium_signal_unstable": "双 seed 中等规模迁移信号不稳定",
    "e4_r3_seed_signal_preserved": "中等规模迁移信号保持",
    "e4_r3_seed_margin_miss": "中等规模迁移差值未过门槛",
    "two_seed_transfer_signal_confirmed": "双 seed 迁移信号已确认",
    "promote_e4_r2": "进入 E4-R2 检查点迁移实验",
    "promote_e4_transfer_joint_gate": "进入双 seed 联合门控",
    "promote_e4_transfer_seed1": "进入独立 seed1 复验",
    "implementation_ready": "实现就绪",
}

STATUS_LABELS = {
    "pass": "通过",
    "fail": "失败",
    "hold": "暂缓",
    "running": "运行中",
    "results_available": "结果已生成",
    "unknown": "状态未知",
}


def build_result_index(
    outputs_root: Path,
    *,
    roots: tuple[str, ...] = DEFAULT_RESULT_ROOTS,
    limit: int = 30,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    entries: list[dict[str, Any]] = []
    for scope in roots:
        scope_root = outputs_root / scope
        if not scope_root.is_dir():
            continue
        for run_root in sorted(path for path in scope_root.iterdir() if path.is_dir()):
            entry = _index_run(outputs_root, scope, run_root)
            if entry is not None:
                entries.append(entry)
    entries.sort(
        key=lambda entry: (
            -float(entry["completed_timestamp"]),
            str(entry["scope"]),
            str(entry["run_id"]),
        )
    )
    ranked = entries[:limit]
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank
        entry["rank_label"] = f"{rank:03d}"
    return ranked


def write_result_index(
    outputs_root: Path,
    *,
    markdown_output: Path | None = None,
    json_output: Path | None = None,
    roots: tuple[str, ...] = DEFAULT_RESULT_ROOTS,
    limit: int = 30,
) -> dict[str, Any]:
    markdown_path = markdown_output or outputs_root / "00_RECENT_RESULTS.md"
    json_path = json_output or outputs_root / "00_RECENT_RESULTS.json"
    entries = build_result_index(outputs_root, roots=roots, limit=limit)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _render_markdown(
            entries,
            outputs_root=outputs_root,
            markdown_output=markdown_path,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "sort_rule": "gate > validation > results; descending completion time",
                "roots": list(roots),
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "entries": len(entries),
        "markdown": str(markdown_path),
        "json": str(json_path),
    }


def _index_run(
    outputs_root: Path,
    scope: str,
    run_root: Path,
) -> dict[str, Any] | None:
    artifacts = _find_artifacts(outputs_root, run_root)
    completion_key = next(
        (key for key in ("gate", "validation", "results") if key in artifacts),
        None,
    )
    if completion_key is None:
        return None
    completion_path = outputs_root / artifacts[completion_key]
    completed_timestamp = completion_path.stat().st_mtime
    decision_payload = _load_first_json(
        outputs_root,
        artifacts,
        keys=("gate", "validation"),
    )
    status = str(decision_payload.get("status") or "results_available")
    decision = str(decision_payload.get("decision") or "")
    claim_scope = str(decision_payload.get("claim_scope") or "")
    return {
        "run_id": run_root.name,
        "display_name": display_name_for_run(run_root.name),
        "scope": scope,
        "status": status,
        "status_display": STATUS_LABELS.get(status, status),
        "decision": decision,
        "decision_display": DECISION_LABELS.get(decision, decision),
        "claim_scope": claim_scope,
        "completed_at": datetime.fromtimestamp(completed_timestamp)
        .astimezone()
        .isoformat(timespec="seconds"),
        "completed_timestamp": completed_timestamp,
        "completion_source": completion_path.name,
        "path": run_root.relative_to(outputs_root).as_posix(),
        "artifacts": artifacts,
    }


def _find_artifacts(outputs_root: Path, run_root: Path) -> dict[str, str]:
    selectors: tuple[tuple[str, Callable[[Path], bool], tuple[str, ...]], ...] = (
        ("gate", lambda path: path.name == "gate.json", ("gate.json",)),
        (
            "validation",
            lambda path: path.name == "validation.json",
            ("validation.json",),
        ),
        (
            "results",
            lambda path: (
                path.suffix == ".jsonl"
                and path.name != "progress.jsonl"
                and (path.name == "results.jsonl" or path.parent.name == "results")
            ),
            ("results.jsonl",),
        ),
        (
            "curves",
            lambda path: path.suffix == ".svg" and "curves" in path.stem,
            ("curves.svg",),
        ),
        (
            "history",
            lambda path: path.suffix == ".csv" and "history" in path.stem,
            ("history.csv",),
        ),
        (
            "progress",
            lambda path: path.name == "progress.jsonl",
            ("progress.jsonl",),
        ),
    )
    artifacts: dict[str, str] = {}
    files = [path for path in run_root.rglob("*") if path.is_file()]
    for key, predicate, preferred_names in selectors:
        candidates = [path for path in files if predicate(path)]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda path: (
                0 if path.name in preferred_names else 1,
                len(path.relative_to(run_root).parts),
                path.as_posix(),
            ),
        )
        artifacts[key] = selected.relative_to(outputs_root).as_posix()
    return artifacts


def _load_first_json(
    outputs_root: Path,
    artifacts: dict[str, str],
    *,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    for key in keys:
        relative = artifacts.get(key)
        if relative is None:
            continue
        try:
            payload = json.loads((outputs_root / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def display_name_for_run(run_id: str) -> str:
    if run_id == (
        "i1_gift64_cross_spn_typed_transfer_r3_65536_joint_seed0_seed1"
    ):
        return "创新1 E4-R3：PRESENT → GIFT-64 跨 SPN 双 seed 中等规模联合裁决"
    r3_medium = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r3_65536_seed(?P<seed>\d+)",
        run_id,
    )
    if r3_medium:
        return (
            "创新1 E4-R3：PRESENT → GIFT-64 跨 SPN 中等规模迁移，"
            f"目标 seed {r3_medium.group('seed')}"
        )
    r3_readiness = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r3_readiness_seed(?P<seed>\d+)",
        run_id,
    )
    if r3_readiness:
        return (
            "创新1 E4-R3：跨 SPN 中等规模实验就绪检查，"
            f"目标 seed {r3_readiness.group('seed')}"
        )
    if run_id == "i1_gift64_cross_spn_typed_transfer_r2_joint_seed0_seed1":
        return "创新1 E4-R2：PRESENT → GIFT-64 跨 SPN 双 seed 联合裁决"
    transfer = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r2_seed(?P<seed>\d+)",
        run_id,
    )
    if transfer:
        return (
            "创新1 E4-R2：PRESENT → GIFT-64 跨 SPN 结构迁移，"
            f"目标 seed {transfer.group('seed')}"
        )
    readiness = re.fullmatch(
        r"i1_gift64_cross_spn_typed_transfer_r0_seed(?P<seed>\d+)",
        run_id,
    )
    if readiness:
        return f"创新1 E4-R0：GIFT-64 迁移实现就绪检查，seed {readiness.group('seed')}"
    if run_id == "i1_cross_spn_typed_cell_r1_seed0":
        return "创新1 E4-R1：PRESENT/GIFT-64 共享类型单元联合门控"
    if run_id == "i1_cross_spn_typed_cell_r0_seed0":
        return "创新1 E4-R0：PRESENT/GIFT-64 共享类型单元实现就绪门控"
    source = re.fullmatch(
        r"i1_(?P<cipher>present|gift64)_cross_spn_typed_cell_r1_seed(?P<seed>\d+)",
        run_id,
    )
    if source:
        cipher = "PRESENT" if source.group("cipher") == "present" else "GIFT-64"
        return f"创新1 E4-R1：{cipher} 共享类型单元训练，seed {source.group('seed')}"
    return run_id


def _render_markdown(
    entries: list[dict[str, Any]],
    *,
    outputs_root: Path,
    markdown_output: Path,
    generated_at: str,
) -> str:
    lines = [
        "# 最近实验结果",
        "",
        "> `001` 永远表示最新完成的结果。排序优先使用门控、验证、结果文件的完成时间；重新生成曲线不会改变实验先后顺序。",
        "",
        f"更新时间：`{generated_at}`",
        "",
        "| 序号 | 完成时间 | 实验说明 | 位置 | 状态 / 裁决 | 快速查看 |",
        "|---:|---|---|---|---|---|",
    ]
    for entry in entries:
        completed = str(entry["completed_at"]).replace("T", " ")
        status = str(entry["status_display"])
        if entry["decision_display"]:
            status = f"{status} / {entry['decision_display']}"
        links = _artifact_links(
            entry["artifacts"],
            run_path=str(entry["path"]),
            outputs_root=outputs_root,
            markdown_output=markdown_output,
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(entry["rank_label"]),
                    _escape_table(completed),
                    _escape_table(str(entry["display_name"])),
                    _escape_table(str(entry["scope"])),
                    _escape_table(status),
                    links or "-",
                )
            )
            + " |"
        )
    if not entries:
        lines.append("| - | - | 暂无可索引的结果 | - | - | - |")
    lines.extend(
        (
            "",
            "说明：原始实验目录名保持不变，避免破坏配置、文档和门控中的证据路径。",
            "",
        )
    )
    return "\n".join(lines)


def _artifact_links(
    artifacts: dict[str, str],
    *,
    run_path: str,
    outputs_root: Path,
    markdown_output: Path,
) -> str:
    directory_target = os.path.relpath(outputs_root / run_path, markdown_output.parent)
    links = [f"[目录]({Path(directory_target).as_posix()})"]
    for key in ("curves", "gate", "validation", "results", "history", "progress"):
        relative = artifacts.get(key)
        if relative is None:
            continue
        target = os.path.relpath(outputs_root / relative, markdown_output.parent)
        links.append(f"[{ARTIFACT_LABELS[key]}]({Path(target).as_posix()})")
    return " ".join(links)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
