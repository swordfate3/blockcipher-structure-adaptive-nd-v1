from __future__ import annotations

import numpy as np

from blockcipher_nd.tasks.innovation2 import integral_context_fresh_expansion as fresh
from blockcipher_nd.tasks.innovation2.integral_context_fresh_expansion import (
    ContextFreshExpansionConfig,
    adjudicate_context_fresh_expansion,
    expanded_contexts,
)
from blockcipher_nd.tasks.innovation2.integral_hwang_readiness import (
    paper_basis_masks,
)


def test_expanded_contexts_preserve_anchors_and_add_48_unique_values() -> None:
    source = tuple(range(16))
    contexts = expanded_contexts(source, seed=0)

    assert contexts[:16] == source
    assert contexts == expanded_contexts(source, seed=0)
    assert len(contexts) == 64
    assert len(set(contexts)) == 64
    assert all(value >> 48 == 0 for value in contexts)


def test_fresh_expansion_gate_separates_instability_and_low_diversity() -> None:
    config = ContextFreshExpansionConfig(run_id="test")
    readiness = {"ok": True}

    unstable = adjudicate_context_fresh_expansion(
        config,
        readiness,
        reproduced_source_signatures=15,
        distinct_signatures=10,
        contexts_beyond_hwang=30,
    )
    ready = adjudicate_context_fresh_expansion(
        config,
        readiness,
        reproduced_source_signatures=16,
        distinct_signatures=8,
        contexts_beyond_hwang=24,
    )
    low_diversity = adjudicate_context_fresh_expansion(
        config,
        readiness,
        reproduced_source_signatures=16,
        distinct_signatures=7,
        contexts_beyond_hwang=30,
    )

    assert unstable["decision"] == "innovation2_context_kernel_fresh_key_unstable"
    assert ready["decision"] == "innovation2_fresh_expanded_context_kernel_ready"
    assert low_diversity["decision"] == (
        "innovation2_fresh_expanded_context_diversity_insufficient"
    )


def test_fresh_expansion_runner_returns_valid_low_diversity_result(
    monkeypatch,
) -> None:
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
        fresh,
        "_collect_xor_words",
        lambda structure, keys, **kwargs: words.copy(),
    )
    monkeypatch.setattr(
        fresh,
        "scalar_bit_integral_output_xor",
        lambda structure, rounds, key: int(words[0]),
    )
    basis_rows = [
        {
            "context_id": str(context_id),
            "fixed_plaintext": f"0x{context_id:016X}",
            "basis_index": str(basis_index),
            "vector_hex": f"0x{vector:016X}",
            "vector_weight": str(vector.bit_count()),
        }
        for context_id in range(16)
        for basis_index, vector in enumerate(
            paper_basis_masks(output_mapping="direct")
        )
    ]
    result = fresh.run_context_fresh_expansion_audit(
        ContextFreshExpansionConfig(run_id="runner"),
        source_gate={
            "run_id": "source",
            "status": "pass",
            "decision": "innovation2_inactive_context_kernel_diversity_ready",
        },
        source_metadata={
            "task": "innovation2_present_r7_inactive_context_kernel_diversity",
            "training_performed": False,
            "keys": 128,
            "key_generation_seed": 3301,
        },
        source_basis_rows=basis_rows,
    )

    assert len(result["rows"]) == 64
    assert result["gate"]["status"] == "hold"
    assert result["gate"]["reproduced_e16_context_signatures"] == 16
    assert result["gate"]["decision"] == (
        "innovation2_fresh_expanded_context_diversity_insufficient"
    )
    assert all(result["gate"]["readiness_checks"].values())
