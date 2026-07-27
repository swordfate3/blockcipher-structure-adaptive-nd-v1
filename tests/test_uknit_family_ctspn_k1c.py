from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.audit_uknit_family_ctspn_k1c import main as run_k1c
from blockcipher_nd.cli.plot_uknit_family_ctspn_k1 import render_ctspn_k1c_svg
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1b import (
    CONTROL_CONDITIONS,
    EXPECTED_CIPHERS,
    EXPECTED_SEEDS,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1c import (
    EXPECTED_RESULT_ROWS,
    RUN_ID,
    adjudicate_k1c,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT / "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_native_endpoint_k1b_2048_seed0_seed1.csv"
)


def test_k1c_train_attribution_without_validation_routes_relative_paths() -> None:
    gate = adjudicate_k1c(
        rows=_rows(train_margin=0.020, validation_margin=-0.010),
        source_checks={"source_valid": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_uknit_family_ctspn_k1c_split_specific_topology_overfit_confirmed"
    )
    assert gate["attribution_summary"]["uknit_train_all_seeds"] is True
    assert gate["attribution_summary"]["uknit_validation_all_seeds"] is False
    assert "relative cross-transition path" in gate["next_action"]


def test_k1c_train_failure_closes_learned_endpoint_summary_route() -> None:
    rows = _rows(train_margin=0.020, validation_margin=-0.010)
    target = next(
        row
        for row in rows
        if row["cipher_key"] == "uknit64"
        and row["seed"] == 1
        and row["split"] == "train"
        and row["condition"] == "rotated"
    )
    target["auc"] = 0.706
    target["correct_minus_condition_auc"] = -0.006

    gate = adjudicate_k1c(rows=rows, source_checks={"source_valid": True})

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_uknit_family_ctspn_k1c_endpoint_summary_not_attributed_on_training"
    )
    assert gate["attribution_summary"]["uknit_train_all_seeds"] is False
    assert "exact cross-transition endpoint-composition" in gate["next_action"]


def test_k1c_fails_closed_when_validation_does_not_replay_k1b() -> None:
    rows = _rows(train_margin=0.020, validation_margin=-0.010)
    target = next(
        row
        for row in rows
        if row["split"] == "validation"
        and row["cipher_key"] == "dialga128"
        and row["seed"] == 0
        and row["condition"] == "correct_ordered"
    )
    target["source_validation_auc"] = float(target["auc"]) + 0.001

    gate = adjudicate_k1c(rows=rows, source_checks={"source_valid": True})

    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["validation_replays_k1b_exactly"] is False


def test_k1c_runner_does_not_create_output_for_invalid_source(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    for name, payload in (
        ("gate.json", {}),
        ("checkpoint_manifest.json", {}),
        ("preflight.json", {}),
    ):
        (source_root / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    for name in ("results.jsonl", "controls.jsonl"):
        (source_root / name).write_text("", encoding="utf-8")
    output_root = tmp_path / "audit"
    monkeypatch.setattr(
        "blockcipher_nd.cli.audit_uknit_family_ctspn_k1c.validate_k1c_source",
        lambda **_kwargs: {"source_valid": False},
    )

    exit_code = run_k1c(
        [
            "--plan",
            str(PLAN),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 4
    assert not output_root.exists()


def test_k1c_chinese_plot_explains_train_validation_attribution(
    tmp_path: Path,
) -> None:
    gate = adjudicate_k1c(
        rows=_rows(train_margin=0.020, validation_margin=-0.010),
        source_checks={"source_valid": True},
    )
    output = tmp_path / "curves.svg"

    render_ctspn_k1c_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "正确拓扑是在训练集过拟合，还是从未被模型学会" in svg
    assert "原训练缓存" in svg
    assert "未见验证数据" in svg
    assert "正确拓扑净优势" in svg
    assert "关闭逐层端点摘要路线" not in svg


def _rows(*, train_margin: float, validation_margin: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cipher in EXPECTED_CIPHERS:
        for seed in EXPECTED_SEEDS:
            checkpoint = f"checkpoint-{cipher}-{seed}"
            state = f"state-{cipher}-{seed}"
            for split in ("train", "validation"):
                correct_auc = 0.70 if cipher == "uknit64" else 0.96
                margin = train_margin if split == "train" else validation_margin
                dataset = f"dataset-{cipher}-{seed}-{split}"
                for condition in CONTROL_CONDITIONS:
                    condition_margin = 0.0 if condition == "correct_ordered" else margin
                    auc = (
                        correct_auc
                        if condition == "correct_ordered"
                        else correct_auc - margin
                    )
                    rows.append(
                        {
                            "run_id": RUN_ID,
                            "cipher_key": cipher,
                            "seed": seed,
                            "split": split,
                            "condition": condition,
                            "rows": 4096 if split == "train" else 2048,
                            "auc": auc,
                            "correct_minus_condition_auc": condition_margin,
                            "max_abs_probability_delta_from_correct": 0.0
                            if condition == "correct_ordered"
                            else 0.1,
                            "mean_abs_probability_delta_from_correct": 0.0
                            if condition == "correct_ordered"
                            else 0.01,
                            "dataset_sha256": dataset,
                            "source_validation_dataset_sha256": dataset
                            if split == "validation"
                            else None,
                            "source_validation_auc": auc
                            if split == "validation"
                            else None,
                            "checkpoint_sha256": checkpoint,
                            "expected_checkpoint_sha256": checkpoint,
                            "state_dict_sha256": state,
                            "strict_state_dict_load": True,
                            "training_performed": False,
                            "optimizer_steps": 0,
                        }
                    )
    assert len(rows) == EXPECTED_RESULT_ROWS
    return rows
