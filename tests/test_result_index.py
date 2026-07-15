from __future__ import annotations

import json
import os
from pathlib import Path

from blockcipher_nd.evaluation.result_index import (
    DEFAULT_RESULT_ROOTS,
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
        "smoke",
        "remote_results",
        "remote_results_incomplete",
    )


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
