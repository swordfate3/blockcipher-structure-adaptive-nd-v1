from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli import audit_innovation2_hwang_readiness as cli
from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    BitIntegralStructure,
)
from blockcipher_nd.tasks.innovation2.integral_hwang_readiness import (
    HwangReadinessConfig,
    adjudicate_hwang_readiness,
    control_masks,
    paper_basis_masks,
    summarize_protocols,
)


def test_sixteen_active_bit_structure_is_supported() -> None:
    structure = BitIntegralStructure(
        structure_id="paper-last16",
        active_bits=tuple(range(48, 64)),
        output_nibble=0,
        output_mask=1,
        fixed_plaintext=0,
    )

    assert structure.set_size == 65536
    assert structure.plaintexts().shape == (65536,)


def test_paper_and_control_masks_preserve_weights_and_are_disjoint() -> None:
    direct = paper_basis_masks(output_mapping="direct")
    reflected = paper_basis_masks(output_mapping="reflected")
    controls = control_masks(seed=0, output_mapping="direct")

    assert direct == (
        1 << 0,
        (1 << 4) | (1 << 12),
        (1 << 16) | (1 << 48),
        (1 << 20) | (1 << 28) | (1 << 52) | (1 << 60),
    )
    assert reflected == tuple(
        sum(1 << (63 - bit) for bit in bits)
        for bits in ((0,), (4, 12), (16, 48), (20, 28, 52, 60))
    )
    assert [mask.bit_count() for mask in controls] == [1, 2, 2, 4]
    paper_span = {
        sum(
            (direct[index] for index in range(4) if selector & (1 << index)),
            start=0,
        )
        for selector in range(16)
    }
    assert set(controls).isdisjoint(paper_span)


def test_summarize_protocols_identifies_one_matching_candidate() -> None:
    config = HwangReadinessConfig(run_id="test", keys=4)
    direct_kernel_control_hit = (1 << 20) | (1 << 60)
    low_words = np.asarray(
        [direct_kernel_control_hit] * 4,
        dtype=np.uint64,
    )
    high_words = np.asarray([1, 2, 4, 8], dtype=np.uint64)

    rows, _ = summarize_protocols(
        config,
        xor_words_by_input={"low_0_15": low_words, "high_48_63": high_words},
    )

    assert len(rows) == 4
    assert any(row["candidate_pass"] for row in rows)


def test_adjudication_covers_ready_ambiguous_hold_and_invalid() -> None:
    config = HwangReadinessConfig(run_id="test", keys=4)
    readiness = {"ok": True}
    base = [
        {"candidate_id": "a", "candidate_pass": True},
        {"candidate_id": "b", "candidate_pass": False},
    ]

    ready = adjudicate_hwang_readiness(config, base, readiness)
    ambiguous = adjudicate_hwang_readiness(
        config,
        [{**row, "candidate_pass": True} for row in base],
        readiness,
    )
    hold = adjudicate_hwang_readiness(
        config,
        [{**row, "candidate_pass": False} for row in base],
        readiness,
    )
    invalid = adjudicate_hwang_readiness(config, base, {"ok": False})

    assert ready["decision"] == "innovation2_present_r7_hwang_bitorder_ready"
    assert ambiguous["decision"] == (
        "innovation2_present_r7_hwang_bitorder_ambiguous"
    )
    assert hold["decision"] == (
        "innovation2_present_r7_hwang_bitorder_not_reproduced"
    )
    assert invalid["decision"] == "innovation2_present_r7_hwang_protocol_invalid"


def test_cli_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    rows = [
        {
            "run_id": "test",
            "candidate_id": candidate,
            "input_orientation": candidate.split("__")[0],
            "output_mapping": candidate.split("__")[1],
            "keys": 8,
            "paper_mask_failures_discovery": 0,
            "paper_mask_failures_validation": 0,
            "paper_mask_failures_joint": 0,
            "control_mask_failures_joint": 5,
            "paper_masks_in_joint_kernel": True,
            "nonzero_output_parity_words": 8,
            "joint_kernel_dimension": 56,
            "candidate_pass": candidate == "low_0_15__direct",
        }
        for candidate in (
            "low_0_15__direct",
            "low_0_15__reflected",
            "high_48_63__direct",
            "high_48_63__reflected",
        )
    ]
    gate = {
        "run_id": "test",
        "status": "pass",
        "decision": "innovation2_present_r7_hwang_bitorder_ready",
        "passing_candidates": ["low_0_15__direct"],
        "selected_candidate": "low_0_15__direct",
        "next_action": {"training": False, "remote_scale": False},
    }
    monkeypatch.setattr(
        cli,
        "run_hwang_readiness_audit",
        lambda config, progress_callback: {
            "rows": rows,
            "mask_rows": [
                {
                    "run_id": "test",
                    "candidate_id": "low_0_15__direct",
                    "mask_role": "paper",
                }
            ],
            "gate": gate,
            "metadata": {"run_id": "test"},
        },
    )

    assert cli.main(["--run-id", "test", "--output-root", str(tmp_path)]) == 0

    assert {
        "results.jsonl",
        "mask_checks.csv",
        "progress.jsonl",
        "gate.json",
        "metadata.json",
        "curves.svg",
    }.issubset(path.name for path in tmp_path.iterdir())
    assert json.loads((tmp_path / "gate.json").read_text())["status"] == "pass"
    svg = (tmp_path / "curves.svg").read_text(encoding="utf-8")
    assert "论文积分输出 mask 的 bit-order 校准" in svg
    assert "不是积分/随机二分类" in svg
