from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli import audit_innovation2_context_diversity as cli
from blockcipher_nd.tasks.innovation2 import integral_context_diversity as context
from blockcipher_nd.tasks.innovation2.integral_context_diversity import (
    ContextDiversityConfig,
    adjudicate_context_diversity_audit,
    inactive_contexts,
)


def test_inactive_contexts_are_deterministic_unique_low48_values() -> None:
    contexts = inactive_contexts(seed=0)

    assert contexts == inactive_contexts(seed=0)
    assert len(contexts) == 16
    assert len(set(contexts)) == 16
    assert contexts[0] == 0
    assert all(value >> 48 == 0 for value in contexts)
    assert all(value != 0 for value in contexts[1:])


def test_context_gate_requires_four_signatures_and_eight_contexts() -> None:
    config = ContextDiversityConfig(run_id="test")
    readiness = {"ok": True}

    passed = adjudicate_context_diversity_audit(
        config,
        [],
        readiness,
        distinct_signatures=4,
        nontrivial_contexts=8,
    )
    held = adjudicate_context_diversity_audit(
        config,
        [],
        readiness,
        distinct_signatures=3,
        nontrivial_contexts=16,
    )
    invalid = adjudicate_context_diversity_audit(
        config,
        [],
        {"ok": False},
        distinct_signatures=8,
        nontrivial_contexts=16,
    )

    assert passed["decision"] == (
        "innovation2_inactive_context_kernel_diversity_ready"
    )
    assert held["decision"] == (
        "innovation2_inactive_context_kernel_diversity_insufficient"
    )
    assert invalid["decision"] == "innovation2_inactive_context_protocol_invalid"


def test_context_runner_returns_complete_result(monkeypatch) -> None:
    constrained = {0, 4, 12, 16, 48, 20, 28, 52, 60}
    orthogonal_rows = [1 << bit for bit in range(64) if bit not in constrained]
    orthogonal_rows.extend(
        [
            (1 << 4) | (1 << 12),
            (1 << 16) | (1 << 48),
            (1 << 20) | (1 << 28),
            (1 << 20) | (1 << 52),
            (1 << 20) | (1 << 60),
        ]
    )
    words = np.asarray(
        orthogonal_rows + orthogonal_rows + orthogonal_rows[:8],
        dtype=np.uint64,
    )
    monkeypatch.setattr(
        context,
        "_collect_xor_words",
        lambda structure, keys, **kwargs: words.copy(),
    )
    monkeypatch.setattr(
        context,
        "scalar_bit_integral_output_xor",
        lambda structure, rounds, key: int(words[0]),
    )

    result = context.run_context_diversity_audit(
        ContextDiversityConfig(run_id="context-runner")
    )

    assert len(result["rows"]) == 16
    assert result["gate"]["status"] == "hold"
    assert all(result["gate"]["readiness_checks"].values())
    assert result["metadata"]["task"] == (
        "innovation2_present_r7_inactive_context_kernel_diversity"
    )


def test_context_cli_writes_expected_artifacts(monkeypatch, tmp_path: Path) -> None:
    rows = [
        {
            "run_id": "test",
            "context_id": context_id,
            "discovery_kernel_dimension": 4,
            "validation_kernel_dimension": 4,
            "joint_kernel_dimension": 4,
            "joint_basis_signature": "paper",
        }
        for context_id in range(16)
    ]
    gate = {
        "run_id": "test",
        "status": "hold",
        "decision": "innovation2_inactive_context_kernel_diversity_insufficient",
        "distinct_joint_kernel_signatures": 1,
        "nontrivial_joint_kernel_contexts": 16,
        "next_action": {"training": False, "remote_scale": False},
    }
    monkeypatch.setattr(
        cli,
        "run_context_diversity_audit",
        lambda config, progress_callback: {
            "rows": rows,
            "basis_rows": [{"run_id": "test", "context_id": 0}],
            "gate": gate,
            "metadata": {"run_id": "test"},
        },
    )

    assert cli.main(["--run-id", "test", "--output-root", str(tmp_path)]) == 0

    assert {
        "results.jsonl",
        "kernel_basis.csv",
        "progress.jsonl",
        "gate.json",
        "metadata.json",
        "curves.svg",
    }.issubset(path.name for path in tmp_path.iterdir())
    assert json.loads((tmp_path / "gate.json").read_text())["status"] == "hold"
    svg = (tmp_path / "curves.svg").read_text(encoding="utf-8")
    assert "高16位积分的固定上下文审计" in svg
    assert "不是神经训练" in svg
