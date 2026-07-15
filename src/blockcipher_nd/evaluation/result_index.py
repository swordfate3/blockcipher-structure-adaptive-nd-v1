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

_SCOPE_PRIORITY = {
    "remote_results": 0,
    "remote_results_incomplete": 1,
    "local_diagnostic": 2,
    "local_smoke": 3,
    "smoke": 4,
}

ARTIFACT_LABELS = {
    "curves": "曲线",
    "gate": "门控",
    "validation": "验证",
    "results": "结果",
    "history": "历史",
    "progress": "进度",
}

DECISION_LABELS = {
    "feistel_balanced_relation_readiness_passed": (
        "SIMON/SIMECK 轮关系实现就绪，进入 2048/类本地诊断"
    ),
    "feistel_balanced_relation_two_cipher_seed0_pass": (
        "两种密码均通过轮关系归因门，进入 seed1 同预算确认"
    ),
    "feistel_balanced_relation_cipher_conditional": (
        "轮关系收益仅在一种密码成立，先审计错误轮函数对照"
    ),
    "feistel_balanced_signal_without_relation_attribution": (
        "存在区分信号但无法归因于真实轮关系，停止扩规模"
    ),
    "feistel_balanced_relation_not_ready": ("轮关系候选未就绪，先做更低轮公式校准"),
    "feistel_balanced_easier_round_calibrated": (
        "低一轮公式与网络已校准，下一步贴近 Lu SE-ResNet 高轮协议"
    ),
    "feistel_balanced_easier_round_cipher_conditional": (
        "低轮校准仅一种密码通过，先审计另一种轮函数"
    ),
    "feistel_balanced_easier_round_signal_without_attribution": (
        "低轮有信号但真实轮关系无独立贡献，停止该架构扩展"
    ),
    "feistel_balanced_easier_round_not_calibrated": (
        "低一轮仍未校准，转作者代码逐样本数据与布局对拍"
    ),
    "feistel_lu_layout_two_cipher_calibrated": (
        "Lu-SE 布局两种密码均校准，进入高一轮同布局比较"
    ),
    "feistel_lu_layout_cipher_conditional": ("Lu-SE 布局仅一种密码通过，保留条件路线"),
    "feistel_lu_layout_signal_without_architecture_gain": (
        "Lu-SE 布局有信号但未优于旧锚点，停止布局扩展"
    ),
    "feistel_lu_layout_not_calibrated": ("Lu-SE 布局未校准，先量化数据规模缺口"),
    "feistel_relation_scale_slope_two_cipher_pass": (
        "两种密码均有正数据斜率，进入独立 seed1 同规模确认"
    ),
    "feistel_relation_scale_slope_cipher_conditional": (
        "数据斜率仅一种密码成立，只确认通过的密码"
    ),
    "feistel_relation_signal_without_scale_slope": (
        "仍有低轮信号但样本斜率不足，停止机械扩规模"
    ),
    "feistel_relation_scale_probe_not_ready": (
        "8192/类规模探针未就绪，重新评估 Feistel 路线优先级"
    ),
    "feistel_relation_8192_seed1_confirmation_pass": (
        "独立 seed1 信号与轮关系归因通过，进入双 seed 综合"
    ),
    "feistel_relation_8192_seed1_cipher_conditional": (
        "独立 seed1 只确认一种密码，保留条件路线"
    ),
    "feistel_relation_8192_seed1_confirmation_failed": (
        "独立 seed1 未确认，停止扩规模"
    ),
    "feistel_target_round_8192_two_cipher_pass": (
        "论文目标轮两种密码均过门，进入独立 seed1 确认"
    ),
    "feistel_target_round_8192_cipher_conditional": (
        "论文目标轮信号仅一种密码成立，保留条件路线"
    ),
    "feistel_target_round_signal_without_scale_slope": (
        "论文目标轮有信号但规模增益不足，保留低轮证据"
    ),
    "feistel_target_round_8192_not_ready": ("论文目标轮 8192/类未就绪，停止远程扩规模"),
    "feistel_curriculum_readiness_passed": (
        "同总轮次课程训练机制就绪，进入 8192/类本地诊断"
    ),
    "feistel_curriculum_two_cipher_pass": (
        "两种密码均通过课程迁移门，进入独立 seed1 确认"
    ),
    "feistel_curriculum_cipher_conditional": ("课程迁移仅一种密码通过，只确认条件路线"),
    "feistel_curriculum_without_scratch_gain": (
        "课程模型有信号但未胜同轮次从零训练，停止课程路线"
    ),
    "feistel_curriculum_without_relation_attribution": (
        "课程信号无法归因于真实轮关系，保留目标轮对照"
    ),
    "feistel_curriculum_target_signal_not_ready": (
        "课程训练未恢复目标轮信号，转表示或差分重设计"
    ),
    "feistel_curriculum_seed1_confirmation_pass": (
        "SIMECK 独立 seed1 课程迁移通过，进入双 seed 综合"
    ),
    "feistel_curriculum_seed1_confirmation_failed": (
        "SIMECK 独立 seed1 未确认，停止课程路线"
    ),
    "feistel_simeck_curriculum_65k_scale_pass": (
        "SIMECK 65536/类课程信号与规模保持通过，进入同规模 seed1"
    ),
    "feistel_simeck_curriculum_65k_scale_regressed": (
        "SIMECK 课程控制过门但规模性能回退，保留本地双 seed 证据"
    ),
    "feistel_simeck_curriculum_65k_not_ready": (
        "SIMECK 65536/类课程规模门未通过，停止远程扩展"
    ),
    "innovation2_integral_property_implementation_ready": (
        "创新2积分性质预测实现就绪，可进入本地诊断"
    ),
    "innovation2_integral_property_smoke_invalid": "创新2积分性质预测 Smoke 无效，先修协议",
    "innovation2_integral_property_advance_multiseed": (
        "结构条件积分概率门控通过，进入多 seed 与经典基线"
    ),
    "innovation2_integral_property_invalid_control": "控制无效，审计拆分与标签泄漏",
    "innovation2_integral_property_linear_signal_only": (
        "信号可由线性基线解释，先补确定性基线"
    ),
    "innovation2_integral_property_redesign_before_scale": (
        "结构排序有信号但概率误差未过门，校准后再扩展"
    ),
    "innovation2_integral_calibration_implementation_ready": (
        "创新2 E1 校准与标签稳定性实现就绪，可进入本地诊断"
    ),
    "innovation2_integral_calibration_smoke_invalid": (
        "创新2 E1 Smoke 无效，先修校准协议"
    ),
    "innovation2_integral_calibration_invalid_control": (
        "创新2 E1 控制无效，审计拆分与校准"
    ),
    "innovation2_integral_calibration_advance_seed1_geometry": (
        "校准概率门控通过，进入 seed1 与几何组合留出"
    ),
    "innovation2_integral_rate_target_unstable": (
        "32-key 概率标签不稳定，改用区间或排序目标"
    ),
    "innovation2_integral_calibration_insufficient": (
        "标签稳定但校准仍不足，下一步仅加入 P-layer 可达性特征"
    ),
    "innovation2_integral_ranking_utility_advance_independent_confirmation": (
        "排序与 top-16 筛选效用过门，进入独立 seed1 确认"
    ),
    "innovation2_integral_ranking_utility_independent_confirmation_passed": (
        "独立 seed1 排序与 top-16 效用通过，进入双 seed 联合裁决"
    ),
    "innovation2_integral_ranking_utility_two_seed_confirmed": (
        "双 seed 排序与 top-16 效用确认，进入几何组合留出"
    ),
    "innovation2_integral_ranking_utility_two_seed_not_confirmed": (
        "双 seed 排序效用未确认，停止几何留出与扩规模"
    ),
    "innovation2_integral_ranking_control_not_attributed": (
        "打乱控制也有 top-16 优势，当前筛选效用不可归因"
    ),
    "innovation2_integral_ranking_explanatory_only": (
        "只有排序相关性过门，保留为解释性证据"
    ),
    "innovation2_integral_ranking_redesign_representation": (
        "当前结构表示无稳定排序效用，先加入 P-layer 可达性特征"
    ),
    "e4_typed_topology_attribution_robust_scratch_efficiency_conditional": (
        "类型拓扑归因稳健，短期 scratch 优势仅条件成立"
    ),
    "e4_r5_source_seed_signal_unstable": "独立 source-seed 稳健性未确认，停止正式扩展",
    "e4_r5_target_adaptation_signal_unstable": "单 seed 目标适配信号不稳定",
    "e4_r5_source_seed_gate_pass": "独立 PRESENT source checkpoint 门控通过",
    "e4_r4_two_seed_target_adaptation_efficiency_confirmed": "双 seed 目标适配效率已确认",
    "e4_r4_two_seed_target_adaptation_signal_unstable": "双 seed 目标适配信号不稳定",
    "e4_r4_two_seed_target_adaptation_rejected": "双 seed 目标适配假设未通过",
    "e4_r4_target_adaptation_efficiency_confirmed": "目标适配效率已确认",
    "e4_r4_target_adaptation_signal_unstable": "目标适配信号不稳定",
    "e4_r4_target_adaptation_rejected": "目标适配假设未通过",
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
    deduplicated: dict[str, dict[str, Any]] = {}
    for entry in entries:
        run_id = str(entry["run_id"])
        current = deduplicated.get(run_id)
        if current is None or _scope_priority(entry) < _scope_priority(current):
            deduplicated[run_id] = entry
    entries = list(deduplicated.values())
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


def _scope_priority(entry: dict[str, Any]) -> tuple[int, float]:
    scope = str(entry["scope"])
    return (
        _SCOPE_PRIORITY.get(scope, len(_SCOPE_PRIORITY)),
        -float(entry["completed_timestamp"]),
    )


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
                and path.stat().st_size > 0
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
    if run_id == "i1_feistel_balanced_round_relation_readiness_seed0":
        return "创新1 Feistel：SIMON/SIMECK 真实轮关系模型就绪检查"
    if run_id == "i1_feistel_balanced_round_relation_2048_seed0":
        return "创新1 Feistel：SIMON/SIMECK 真实轮关系 2048/类归因诊断"
    if run_id == "i1_feistel_balanced_round_relation_calibration_2048_seed0":
        return "创新1 Feistel：SIMON r11 / SIMECK r14 低一轮公式校准"
    if run_id == "i1_feistel_lu_senet_layout_calibration_2048_seed0":
        return "创新1 Feistel：Lu 源码 SE-ResNet pair 轴布局校准"
    if run_id == "i1_feistel_round_relation_scale_probe_8192_seed0":
        return "创新1 Feistel：SIMON/SIMECK 轮关系 8192/类数据斜率探针"
    if run_id == "i1_feistel_round_relation_scale_probe_8192_seed1":
        return "创新1 Feistel：SIMON/SIMECK 轮关系 8192/类独立 seed1 确认"
    if run_id == "i1_feistel_round_relation_target_round_8192_seed0":
        return "创新1 Feistel：SIMON r12 / SIMECK r15 论文目标轮 8192/类探针"
    if run_id == "i1_feistel_low_to_target_curriculum_readiness_seed0":
        return "创新1 Feistel：低轮到目标轮同总轮次课程训练就绪检查"
    if run_id == "i1_feistel_low_to_target_curriculum_8192_seed0":
        return "创新1 Feistel：SIMON/SIMECK 低轮到目标轮课程迁移 8192/类裁决"
    if run_id == "i1_feistel_low_to_target_curriculum_8192_seed1_simeck":
        return "创新1 Feistel：SIMECK 低轮到目标轮课程迁移 seed1 确认"
    if run_id == "i1_feistel_simeck_curriculum_65k_seed0":
        return "创新1 Feistel：SIMECK 低轮到目标轮课程迁移 65536/类规模裁决"
    if run_id == "i2_present_r5_structure_integral_parity_smoke_seed0":
        return "创新2 E0：PRESENT 5轮结构条件积分平衡概率预测 Smoke"
    if run_id == "i2_present_r5_structure_integral_parity_feasibility_seed0":
        return "创新2 E0：PRESENT 5轮结构条件积分平衡概率可行性诊断"
    if run_id == "i2_present_r5_integral_parity_calibration_smoke_seed0":
        return "创新2 E1：PRESENT 5轮积分平衡概率独立校准 Smoke"
    if run_id == "i2_present_r5_integral_parity_calibration_seed0":
        return "创新2 E1：PRESENT 5轮积分平衡概率校准与标签稳定性诊断"
    if run_id == "i2_present_r5_integral_parity_ranking_utility_seed0":
        return "创新2 E2：PRESENT 5轮积分输出平衡候选排序与 top-16 效用审判"
    if run_id == "i2_present_r5_integral_parity_ranking_utility_joint_seed0_seed1":
        return "创新2 E3：PRESENT 5轮积分输出候选排序双 seed 联合裁决"
    ranking_seed = re.fullmatch(
        r"i2_present_r5_integral_parity_ranking_utility_seed(?P<seed>\d+)",
        run_id,
    )
    if ranking_seed:
        return (
            "创新2 E3：PRESENT 5轮积分输出平衡候选排序独立确认，"
            f"seed {ranking_seed.group('seed')}"
        )
    calibration_seed = re.fullmatch(
        r"i2_present_r5_integral_parity_calibration_seed(?P<seed>\d+)",
        run_id,
    )
    if calibration_seed:
        return (
            "创新2 E3：PRESENT 5轮积分平衡概率与标签稳定性独立诊断，"
            f"seed {calibration_seed.group('seed')}"
        )
    if run_id == "i1_cross_spn_e4_final_synthesis_20260715":
        return "创新1 E4：跨 SPN 类型拓扑四个目标 cell 最终证据综合"
    if run_id == "i1_gift64_cross_spn_source_seed_r5_65536_joint_seed4_seed5":
        return "创新1 E4-R5：独立 PRESENT source-seed 稳健性联合裁决"
    r5_medium = re.fullmatch(
        r"i1_gift64_cross_spn_source_seed_r5_65536_seed(?P<seed>\d+)",
        run_id,
    )
    if r5_medium:
        return (
            "创新1 E4-R5：独立 source checkpoint 的一轮 GIFT-64 适配，"
            f"目标 seed {r5_medium.group('seed')}"
        )
    r5_readiness = re.fullmatch(
        r"i1_gift64_cross_spn_source_seed_r5_readiness_seed(?P<seed>\d+)",
        run_id,
    )
    if r5_readiness:
        return (
            "创新1 E4-R5：独立 source checkpoint 目标适配就绪检查，"
            f"目标 seed {r5_readiness.group('seed')}"
        )
    if run_id == "i1_present_cross_spn_source_seed_r5_8192_seed1":
        return "创新1 E4-R5 Phase A：独立 PRESENT source seed1 门控"
    if run_id == ("i1_gift64_cross_spn_target_adaptation_r4_65536_joint_seed2_seed3"):
        return "创新1 E4-R4：PRESENT → GIFT-64 双 seed 目标适配效率联合裁决"
    r4_medium = re.fullmatch(
        r"i1_gift64_cross_spn_target_adaptation_r4_65536_seed(?P<seed>\d+)",
        run_id,
    )
    if r4_medium:
        return (
            "创新1 E4-R4：PRESENT → GIFT-64 一轮目标适配效率，"
            f"目标 seed {r4_medium.group('seed')}"
        )
    r4_readiness = re.fullmatch(
        r"i1_gift64_cross_spn_target_adaptation_r4_readiness_seed(?P<seed>\d+)",
        run_id,
    )
    if r4_readiness:
        return (
            "创新1 E4-R4：跨 SPN 一轮目标适配实验就绪检查，"
            f"目标 seed {r4_readiness.group('seed')}"
        )
    if run_id == ("i1_gift64_cross_spn_typed_transfer_r3_65536_joint_seed0_seed1"):
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
