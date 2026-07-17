from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from blockcipher_nd.evaluation.result_index import (
    DEFAULT_INDEX_LIMIT,
    DEFAULT_RESULT_ROOTS,
    DEFAULT_RETENTION_DAYS,
    build_result_index,
    write_result_index,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _set_mtime(path: Path, timestamp: float) -> None:
    os.utime(path, (timestamp, timestamp))


def test_default_result_roots_cover_local_and_remote_result_runs() -> None:
    assert DEFAULT_RESULT_ROOTS == (
        "local_smoke",
        "local_diagnostic",
        "local_audits",
        "smoke",
        "remote_results",
        "remote_results_incomplete",
    )


def test_result_index_includes_innovation2_local_output_property_audit(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = (
        "i2_present_r6_output_property_transition_"
        "width1_width2_seed0_20260717"
    )
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": (
                "innovation2_r6_two_nibble_"
                "output_prediction_benchmark_not_ready"
            ),
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert len(entries) == 1
    assert entries[0]["scope"] == "local_audits"
    assert entries[0]["display_name"] == (
        "创新2 E7：PRESENT 6轮积分输出性质活动宽度过渡审计"
    )
    assert entries[0]["decision_display"] == (
        "r6 两活动 nibble 的位置残差不足，转 5--7 活动 bit 审计"
    )


def test_result_index_labels_innovation2_active_bit_transition_audit(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r6_output_property_active_bits5_6_7_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": "innovation2_r6_active_bit_transition_benchmark_not_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E8：PRESENT 6轮积分输出性质细粒度活动 bit 审计"
    )
    assert entries[0]["decision_display"] == (
        "r6 的 5--7 活动 bit 均无可重复结构信号，停止当前单 mask 训练路线"
    )


def test_result_index_labels_innovation2_stable_subspace_audit(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_stable_balance_subspace_r5_r6_bits5_6_7_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": "innovation2_r6_stable_balance_subspace_not_found",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E9：PRESENT 输出平衡 mask 子空间稳定性审计"
    )
    assert entries[0]["decision_display"] == (
        "r4 校准通过，r6 当前结构族无稳定平衡子空间"
    )


def test_result_index_labels_innovation2_hwang_readiness(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_hwang_kernel_last16_bitorder_readiness_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_present_r7_hwang_bitorder_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E10：PRESENT 7轮论文输出 mask bit-order 校准"
    )
    assert entries[0]["decision_display"] == (
        "唯一 bit-order 复现论文输出 mask，可扩大新密钥复核"
    )


def test_result_index_labels_innovation2_hwang_convergence(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_hwang_kernel_convergence_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_present_r7_hwang_kernel_reproduced",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E11：PRESENT 7轮论文四维输出 kernel 收敛审判"
    )
    assert entries[0]["decision_display"] == (
        "128把新密钥复现论文四维输出 kernel，进入结构族扩展"
    )


def test_result_index_labels_innovation2_hwang_high16_control(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = (
        "i2_present_r7_hwang_kernel_convergence_high16_128keys_seed0_20260717"
    )
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": "innovation2_present_r7_hwang_kernel_underconstrained",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E11b：PRESENT 7轮高16位论文 kernel 同预算对照"
    )


def test_result_index_labels_innovation2_active_block_diversity(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_active_block_kernel_diversity_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": (
                "innovation2_present_r7_active_block_kernel_diversity_ready"
            ),
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E12：PRESENT 7轮活动块输出 kernel 多样性 readiness"
    )
    assert entries[0]["decision_display"] == (
        "不同活动块产生多个稳定输出 kernel，可构造结构条件标签表"
    )


def test_result_index_labels_innovation2_output_label_readiness(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_structure_mask_label_readiness_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": "innovation2_output_label_shortcut_dominated",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E13：PRESENT 7轮结构-mask输出标签边际捷径审计"
    )
    assert entries[0]["decision_display"] == (
        "活动块+mask简单边际已解释标签，禁止直接训练神经网络"
    )


def test_result_index_labels_innovation2_cyclic_geometry(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_cyclic_geometry_kernel_diversity_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_cyclic_geometry_kernel_diversity_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E14：PRESENT 7轮循环活动几何输出 kernel 扩展"
    )
    assert entries[0]["decision_display"] == (
        "循环滑动活动几何形成足够多稳定 kernel，可重建扩展标签表"
    )


def test_result_index_labels_innovation2_topology_geometry(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_topology_geometry_kernel_diversity_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_topology_geometry_kernel_diversity_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E15：PRESENT 7轮P-layer拓扑活动几何审计"
    )
    assert entries[0]["decision_display"] == (
        "P-layer拓扑活动几何形成足够多稳定 kernel，可重建标签表"
    )


def test_result_index_labels_innovation2_inactive_context(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_inactive_context_kernel_diversity_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": "innovation2_inactive_context_kernel_diversity_insufficient",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E16：PRESENT 7轮高16位固定上下文 kernel 审计"
    )
    assert entries[0]["decision_display"] == (
        "固定上下文的输出 kernel 多样性不足"
    )


def test_result_index_labels_innovation2_context_label(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_context_mask_label_readiness_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_context_label_interaction_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E17：PRESENT 7轮context-mask输出标签捷径审计"
    )
    assert entries[0]["decision_display"] == (
        "强基线未解释 context-mask 交互，可做 fresh-key 验证"
    )


def test_result_index_labels_innovation2_equal_prevalence_label(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_equal_prevalence_context_mask_readiness_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_equal_prevalence_context_label_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E17b：PRESENT 7轮等流行率翻转-mask标签审计"
    )
    assert entries[0]["decision_display"] == (
        "等流行率翻转-mask 标签未被强捷径解释，可做 fresh-key 验证"
    )


def test_result_index_labels_innovation2_group_disjoint(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_context_mask_group_disjoint_readiness_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_group_disjoint_shortcuts_controlled",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E17c：PRESENT 7轮context/mask双轴组外捷径审计"
    )
    assert entries[0]["decision_display"] == (
        "组外线性捷径受控，可进入 fresh-key 稳定性验证"
    )


def test_result_index_labels_innovation2_fresh_context_expansion(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_fresh_expanded_context_kernel_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_fresh_expanded_context_kernel_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E18：PRESENT 7轮64-context fresh-key kernel扩展"
    )
    assert entries[0]["decision_display"] == (
        "fresh-key稳定且64-context kernel 多样性充足，可重建标签"
    )


def test_result_index_labels_innovation2_balance_rate(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r7_context_mask_balance_rate_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_balance_rate_interaction_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E19：PRESENT 7轮跨密钥输出平衡概率审计"
    )
    assert entries[0]["decision_display"] == (
        "跨密钥平衡率 interaction 残差可重复，可设计连续预测"
    )


def test_result_index_labels_innovation2_skinny_hwang(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_skinny_r7_hwang_kernel_reproduced",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E20：SKINNY-64/64 7轮 Hwang exact-kernel 就绪审计"
    )
    assert entries[0]["decision_display"] == (
        "SKINNY-64/64 7轮 Hwang 18维 kernel 已精确复现"
    )


def test_result_index_labels_innovation2_skinny_hwang_r8(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_skinny_r8_hwang_kernel_reproduced",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E21：SKINNY-64/64 8轮 two-active-cell kernel 就绪审计"
    )
    assert entries[0]["decision_display"] == (
        "SKINNY-64/64 8轮 two-active-cell 一维 kernel 已精确复现"
    )


def test_result_index_labels_innovation2_skinny_geometry(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_skinny_r8_geometry_kernel_diversity_ready",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E22：SKINNY-64/64 8轮相邻活动pair kernel多样性审计"
    )
    assert entries[0]["decision_display"] == (
        "SKINNY 8轮相邻活动pair形成足够多稳定kernel，可构造标签"
    )


def test_result_index_labels_innovation2_skinny_bottom_row_closure(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717"
    run_root = outputs / "local_audits" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": "innovation2_skinny_r8_bottom_row_pair_family_not_closed",
        },
    )

    entries = build_result_index(outputs, limit=10)

    assert entries[0]["display_name"] == (
        "创新2 E23：SKINNY-64/64 8轮底行活动pair kernel闭合审判"
    )
    assert entries[0]["decision_display"] == (
        "SKINNY 8轮底行pair稳定kernel仅4/6，未达到闭合门"
    )


def test_result_index_defaults_keep_thirty_entries_and_seven_days() -> None:
    assert DEFAULT_INDEX_LIMIT == 30
    assert DEFAULT_RETENTION_DAYS == 7


def test_result_index_orders_by_decision_artifact_not_regenerated_plot(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    older = outputs / "local_smoke" / "older_run"
    newer = outputs / "local_smoke" / "newer_run"
    _write_json(older / "gate.json", {"status": "pass", "decision": "older"})
    _write_json(newer / "gate.json", {"status": "pass", "decision": "newer"})
    (older / "curves.svg").write_text("<svg/>", encoding="utf-8")
    (newer / "curves.svg").write_text("<svg/>", encoding="utf-8")
    _set_mtime(older / "gate.json", 100.0)
    _set_mtime(newer / "gate.json", 200.0)
    _set_mtime(older / "curves.svg", 999.0)
    _set_mtime(newer / "curves.svg", 201.0)

    entries = build_result_index(outputs, roots=("local_smoke",), limit=10)

    assert [entry["run_id"] for entry in entries] == ["newer_run", "older_run"]
    assert [entry["rank"] for entry in entries] == [1, 2]
    assert entries[0]["completion_source"] == "gate.json"
    assert entries[1]["completion_source"] == "gate.json"


def test_result_index_writes_numbered_chinese_markdown_with_artifact_links(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i1_gift64_cross_spn_typed_transfer_r2_joint_seed0_seed1"
    run_root = outputs / "local_smoke" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "two_seed_transfer_signal_confirmed",
            "claim_scope": "two-seed local diagnostic",
        },
    )
    (run_root / "curves.svg").write_text("<svg/>", encoding="utf-8")
    (run_root / "results.jsonl").write_text("{}\n", encoding="utf-8")
    (run_root / "history.csv").write_text("epoch\n", encoding="utf-8")
    markdown_output = outputs / "00_RECENT_RESULTS.md"
    json_output = outputs / "00_RECENT_RESULTS.json"

    report = write_result_index(
        outputs,
        markdown_output=markdown_output,
        json_output=json_output,
        roots=("local_smoke",),
        limit=20,
    )

    assert report["entries"] == 1
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "`001` 永远表示最新完成的结果" in markdown
    assert "至少显示最新 20 条" in markdown
    assert "7 天内的所有条目" in markdown
    assert "| 001 |" in markdown
    assert "PRESENT → GIFT-64 跨 SPN 双 seed 联合裁决" in markdown
    assert "双 seed 迁移信号已确认" in markdown
    assert f"[目录](local_smoke/{run_id})" in markdown
    assert f"[曲线](local_smoke/{run_id}/curves.svg)" in markdown
    assert f"[门控](local_smoke/{run_id}/gate.json)" in markdown
    assert f"[结果](local_smoke/{run_id}/results.jsonl)" in markdown
    assert f"[历史](local_smoke/{run_id}/history.csv)" in markdown
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["entries"][0]["rank_label"] == "001"
    assert payload["entries"][0]["status"] == "pass"
    assert payload["entries"][0]["decision"] == "two_seed_transfer_signal_confirmed"
    assert payload["retention"] == {
        "days_from_latest_completion": 7,
        "minimum_entries": 20,
    }


def test_result_index_prefers_local_readjudication_gate(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run = outputs / "remote_results" / "bridge"
    _write_json(
        run / "gate.json",
        {"status": "pass", "decision": "remote_old_gate"},
    )
    _write_json(
        run / "gate.local.json",
        {"status": "hold", "decision": "candidate_only_local_gate"},
    )

    entries = build_result_index(
        outputs,
        roots=("remote_results",),
        limit=10,
    )

    assert len(entries) == 1
    assert entries[0]["status"] == "hold"
    assert entries[0]["decision"] == "candidate_only_local_gate"
    assert entries[0]["completion_source"] == "gate.local.json"
    assert entries[0]["artifacts"]["gate"].endswith("gate.local.json")


def test_result_index_retains_every_result_from_latest_seven_days(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    latest_timestamp = 2_000_000.0
    for index in range(35):
        gate = outputs / "local_smoke" / f"recent_{index:02d}" / "gate.json"
        _write_json(gate, {"status": "pass", "decision": "recent"})
        _set_mtime(gate, latest_timestamp - index * 60 * 60)
    for index in range(10):
        gate = outputs / "local_smoke" / f"old_{index:02d}" / "gate.json"
        _write_json(gate, {"status": "pass", "decision": "old"})
        _set_mtime(gate, latest_timestamp - (8 * 24 * 60 * 60) - index)

    entries = build_result_index(
        outputs,
        roots=("local_smoke",),
        limit=30,
        retention_days=7,
    )

    assert len(entries) == 35
    assert entries[0]["run_id"] == "recent_00"
    assert entries[-1]["run_id"] == "recent_34"
    assert not any(str(entry["run_id"]).startswith("old_") for entry in entries)


def test_result_index_supports_remote_nested_result_artifacts(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "remote_run"
    run_root = outputs / "remote_results" / run_id
    result_path = run_root / "results" / f"{run_id}.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("{}\n", encoding="utf-8")
    (run_root / "plots").mkdir()
    (run_root / "plots" / f"{run_id}_curves.svg").write_text(
        "<svg/>",
        encoding="utf-8",
    )

    entries = build_result_index(outputs, roots=("remote_results",), limit=10)

    assert len(entries) == 1
    assert entries[0]["scope"] == "remote_results"
    assert entries[0]["status"] == "results_available"
    assert entries[0]["artifacts"]["results"] == (
        f"remote_results/{run_id}/results/{run_id}.jsonl"
    )
    assert entries[0]["artifacts"]["curves"] == (
        f"remote_results/{run_id}/plots/{run_id}_curves.svg"
    )


def test_result_index_ignores_empty_result_jsonl(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_root = outputs / "local_diagnostic" / "stopped_run"
    run_root.mkdir(parents=True)
    (run_root / "results.jsonl").write_text("", encoding="utf-8")
    (run_root / "progress.jsonl").write_text("{}\n", encoding="utf-8")

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert entries == []


def test_result_index_prefers_verified_remote_copy_for_same_run_id(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "same_remote_run"
    _write_json(
        outputs / "remote_results_incomplete" / run_id / "gate.json",
        {"status": "pass", "decision": "fallback"},
    )
    _write_json(
        outputs / "remote_results" / run_id / "gate.json",
        {"status": "pass", "decision": "verified"},
    )

    entries = build_result_index(outputs, limit=10)

    assert len(entries) == 1
    assert entries[0]["run_id"] == run_id
    assert entries[0]["scope"] == "remote_results"
    assert entries[0]["decision"] == "verified"


def test_result_index_supports_r3_local_diagnostic_chinese_names(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i1_gift64_cross_spn_typed_transfer_r3_65536_seed1"
    run_root = outputs / "local_diagnostic" / run_id
    _write_json(
        run_root / "gate.json",
        {"status": "pass", "decision": "e4_r3_seed_signal_preserved"},
    )

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert len(entries) == 1
    assert entries[0]["scope"] == "local_diagnostic"
    assert "跨 SPN 中等规模迁移，目标 seed 1" in entries[0]["display_name"]
    assert entries[0]["decision_display"] == "中等规模迁移信号保持"


def test_result_index_supports_r4_readiness_chinese_name(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i1_gift64_cross_spn_target_adaptation_r4_readiness_seed2"
    run_root = outputs / "local_smoke" / run_id
    _write_json(
        run_root / "gate.json",
        {"status": "pass", "decision": "implementation_ready"},
    )

    entries = build_result_index(outputs, roots=("local_smoke",), limit=10)

    assert len(entries) == 1
    assert "E4-R4" in entries[0]["display_name"]
    assert "目标 seed 2" in entries[0]["display_name"]


def test_result_index_supports_e4_final_synthesis_chinese_labels(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i1_cross_spn_e4_final_synthesis_20260715"
    run_root = outputs / "local_diagnostic" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": (
                "e4_typed_topology_attribution_robust_"
                "scratch_efficiency_conditional"
            ),
        },
    )

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert entries[0]["display_name"] == (
        "创新1 E4：跨 SPN 类型拓扑四个目标 cell 最终证据综合"
    )
    assert entries[0]["decision_display"] == (
        "类型拓扑归因稳健，短期 scratch 优势仅条件成立"
    )


def test_result_index_supports_e4_r5_remote_chinese_labels(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i1_gift64_cross_spn_source_seed_r5_65536_joint_seed4_seed5"
    run_root = outputs / "remote_results" / run_id
    _write_json(
        run_root / "gate.json",
        {"status": "pass", "decision": "e4_r5_source_seed_signal_unstable"},
    )

    entries = build_result_index(outputs, roots=("remote_results",), limit=10)

    assert entries[0]["display_name"] == (
        "创新1 E4-R5：独立 PRESENT source-seed 稳健性联合裁决"
    )
    assert entries[0]["decision_display"] == (
        "独立 source-seed 稳健性未确认，停止正式扩展"
    )


def test_result_index_supports_innovation2_integral_property_chinese_labels(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r5_structure_integral_parity_feasibility_seed0"
    run_root = outputs / "local_diagnostic" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": "innovation2_integral_property_redesign_before_scale",
        },
    )

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert entries[0]["display_name"] == (
        "创新2 E0：PRESENT 5轮结构条件积分平衡概率可行性诊断"
    )
    assert entries[0]["decision_display"] == (
        "结构排序有信号但概率误差未过门，校准后再扩展"
    )


def test_result_index_supports_innovation2_calibration_chinese_labels(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r5_integral_parity_calibration_seed0"
    run_root = outputs / "local_diagnostic" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "hold",
            "decision": "innovation2_integral_calibration_insufficient",
        },
    )

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert entries[0]["display_name"] == (
        "创新2 E1：PRESENT 5轮积分平衡概率校准与标签稳定性诊断"
    )
    assert entries[0]["decision_display"] == (
        "标签稳定但校准仍不足，下一步仅加入 P-layer 可达性特征"
    )


def test_result_index_supports_innovation2_ranking_chinese_labels(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r5_integral_parity_ranking_utility_seed0"
    run_root = outputs / "local_diagnostic" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": (
                "innovation2_integral_ranking_utility_advance_independent_confirmation"
            ),
        },
    )

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert entries[0]["display_name"] == (
        "创新2 E2：PRESENT 5轮积分输出平衡候选排序与 top-16 效用审判"
    )
    assert entries[0]["decision_display"] == (
        "排序与 top-16 筛选效用过门，进入独立 seed1 确认"
    )


def test_result_index_supports_innovation2_joint_ranking_labels(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = "i2_present_r5_integral_parity_ranking_utility_joint_seed0_seed1"
    run_root = outputs / "local_diagnostic" / run_id
    _write_json(
        run_root / "gate.json",
        {
            "status": "pass",
            "decision": "innovation2_integral_ranking_utility_two_seed_confirmed",
        },
    )

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert entries[0]["display_name"] == (
        "创新2 E3：PRESENT 5轮积分输出候选排序双 seed 联合裁决"
    )
    assert entries[0]["decision_display"] == (
        "双 seed 排序与 top-16 效用确认，进入几何组合留出"
    )


@pytest.mark.parametrize(
    ("decision", "expected_decision"),
    [
        (
            "innovation2_high_round_integral_two_seed_bridge_confirmed",
            "两颗 seed 均确认 PRESENT-80 8轮神经信号，准备论文参考规模近似实验",
        ),
        (
            "innovation2_high_round_integral_two_seed_bridge_not_confirmed",
            "双 seed 信号未共同过门，停止机械扩规模并审计 seed 敏感性",
        ),
        (
            "innovation2_high_round_integral_two_seed_bridge_invalid",
            "双 seed source、协议或控制证据无效，修复后重新裁决",
        ),
    ],
)
def test_result_index_supports_innovation2_high_round_joint_bridge_labels(
    tmp_path: Path,
    decision: str,
    expected_decision: str,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = (
        "i2_present_r8_high_round_integral_bridge_262144_joint_"
        "seed0_seed1_20260716"
    )
    run_root = outputs / "local_diagnostic" / run_id
    _write_json(run_root / "gate.json", {"status": "pass", "decision": decision})

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert entries[0]["display_name"] == (
        "创新2：PRESENT-80 8轮 262144-total 双 seed bridge 联合裁决"
    )
    assert entries[0]["decision_display"] == expected_decision


@pytest.mark.parametrize(
    ("decision", "expected_decision"),
    [
        (
            "innovation2_high_round_integral_paper_reference_candidate_advantage",
            "PRESENT-80 8轮论文参考规模近似通过，候选优于同预算锚点与强控制",
        ),
        (
            "innovation2_high_round_integral_paper_reference_round_reach_only",
            "PRESENT-80 8轮论文参考规模信号成立，但未确认候选架构优势",
        ),
        (
            "innovation2_high_round_integral_paper_reference_not_confirmed",
            "论文参考规模8轮信号未确认，停止机械放大并审计近似参数",
        ),
        (
            "innovation2_high_round_integral_paper_reference_invalid_control",
            "论文参考规模 source、缓存或控制无效，修复后重新裁决",
        ),
        (
            "innovation2_high_round_integral_paper_reference_plan_mismatch",
            "论文参考规模运行偏离冻结协议，不纳入结果比较",
        ),
    ],
)
def test_result_index_supports_innovation2_paper_reference_labels(
    tmp_path: Path,
    decision: str,
    expected_decision: str,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = (
        "i2_present_r8_high_round_integral_paper_reference_"
        "2pow21_seed0_gpu0_20260716"
    )
    run_root = outputs / "remote_results" / run_id
    _write_json(run_root / "gate.json", {"status": "pass", "decision": decision})

    entries = build_result_index(
        outputs,
        roots=("remote_results",),
        limit=10,
    )

    assert entries[0]["display_name"] == (
        "创新2：PRESENT-80 8轮 2^21-total / 50轮训练论文参考规模近似"
    )
    assert entries[0]["decision_display"] == expected_decision


@pytest.mark.parametrize(
    ("decision", "expected_decision"),
    [
        (
            "innovation2_high_round_integral_two_seed_paper_reference_"
            "candidate_advantage_confirmed",
            "双 seed 论文参考规模候选优势确认，可按限定范围写入论文",
        ),
        (
            "innovation2_high_round_integral_two_seed_paper_reference_"
            "round_reach_confirmed",
            "双 seed 达到 PRESENT-80 8轮，但未确认候选架构优势",
        ),
        (
            "innovation2_high_round_integral_two_seed_paper_reference_"
            "seed_variance_hold",
            "论文参考规模存在 seed 方差，停止机械追加并审计冻结假设",
        ),
        (
            "innovation2_high_round_integral_two_seed_paper_reference_invalid",
            "双 seed 论文参考规模证据链无效，修复后重新裁决",
        ),
    ],
)
def test_result_index_supports_innovation2_joint_paper_reference_labels(
    tmp_path: Path,
    decision: str,
    expected_decision: str,
) -> None:
    outputs = tmp_path / "outputs"
    run_id = (
        "i2_present_r8_high_round_integral_paper_reference_"
        "2pow21_joint_seed0_seed1"
    )
    run_root = outputs / "local_diagnostic" / run_id
    _write_json(run_root / "gate.json", {"status": "pass", "decision": decision})

    entries = build_result_index(
        outputs,
        roots=("local_diagnostic",),
        limit=10,
    )

    assert entries[0]["display_name"] == (
        "创新2：PRESENT-80 8轮论文参考规模双 seed 联合裁决"
    )
    assert entries[0]["decision_display"] == expected_decision
