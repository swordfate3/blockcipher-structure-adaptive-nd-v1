from __future__ import annotations

import json
import os
from pathlib import Path

from blockcipher_nd.evaluation.result_index import (
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
