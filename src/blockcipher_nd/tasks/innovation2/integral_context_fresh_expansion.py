from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    BitIntegralStructure,
    scalar_bit_integral_output_xor,
)
from blockcipher_nd.tasks.innovation2.integral_context_diversity import ACTIVE_BITS
from blockcipher_nd.tasks.innovation2.integral_context_label_readiness import (
    kernel_bases_and_contexts_from_rows,
)
from blockcipher_nd.tasks.innovation2.integral_hwang_readiness import (
    _collect_xor_words,
    paper_basis_masks,
)
from blockcipher_nd.tasks.innovation2.integral_property_prediction import make_keys
from blockcipher_nd.tasks.innovation2.integral_subspace_audit import (
    gf2_kernel_basis,
    kernel_basis_valid,
)


EXPECTED_SOURCE_DECISION = "innovation2_inactive_context_kernel_diversity_ready"
EXPECTED_SOURCE_TASK = "innovation2_present_r7_inactive_context_kernel_diversity"
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ContextFreshExpansionConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    contexts: int = 64
    keys: int = 128
    key_chunk_size: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen E18 audit requires PRESENT r7")
        if self.contexts != 64:
            raise ValueError("the frozen E18 audit requires exactly 64 contexts")
        if self.keys != 128:
            raise ValueError("the frozen E18 audit requires exactly 128 fresh keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")


def expanded_contexts(
    source_contexts: tuple[int, ...],
    *,
    seed: int,
) -> tuple[int, ...]:
    if (
        len(source_contexts) != 16
        or len(set(source_contexts)) != 16
        or source_contexts[0] != 0
        or any(not 0 <= value < (1 << 48) for value in source_contexts)
    ):
        raise ValueError("source contexts must be 16 unique low48 values starting at zero")
    rng = np.random.default_rng(seed + 7401)
    contexts = list(source_contexts)
    used = set(contexts)
    while len(contexts) < 64:
        value = int(rng.integers(0, 1 << 48, dtype=np.uint64))
        if value == 0 or value in used:
            continue
        contexts.append(value)
        used.add(value)
    return tuple(contexts)


def run_context_fresh_expansion_audit(
    config: ContextFreshExpansionConfig,
    *,
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_basis_rows: list[dict[str, str]],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source_checks = validate_source(source_gate, source_metadata, source_basis_rows)
    source_bases, source_context_map = kernel_bases_and_contexts_from_rows(
        source_basis_rows
    )
    source_contexts = tuple(source_context_map[index] for index in range(16))
    contexts = expanded_contexts(source_contexts, seed=config.seed)
    fresh_keys = make_keys(count=config.keys, seed=config.seed + 8801)
    source_keys = make_keys(
        count=int(source_metadata["keys"]),
        seed=int(source_metadata["key_generation_seed"]),
    )
    half = config.keys // 2
    structures = tuple(
        BitIntegralStructure(
            structure_id=f"present-r7-fresh-context-{context_id:02d}",
            active_bits=ACTIVE_BITS,
            output_nibble=0,
            output_mask=1,
            fixed_plaintext=context,
        )
        for context_id, context in enumerate(contexts)
    )
    xor_words_by_context: dict[int, np.ndarray] = {}
    for context_id, structure in enumerate(structures):
        xor_words_by_context[context_id] = _collect_xor_words(
            structure,
            fresh_keys,
            rounds=config.rounds,
            key_chunk_size=config.key_chunk_size,
            progress_callback=progress_callback,
        )
    scalar_matches = all(
        int(xor_words_by_context[context_id][0])
        == scalar_bit_integral_output_xor(
            structures[context_id],
            rounds=config.rounds,
            key=fresh_keys[0],
        )
        for context_id in (0, 16)
    )
    rows, basis_rows, readiness, gate = evaluate_context_fresh_expansion(
        config,
        contexts=contexts,
        xor_words_by_context=xor_words_by_context,
        source_bases=source_bases,
        source_checks=source_checks,
        scalar_matches=scalar_matches,
        fresh_key_halves_disjoint=set(fresh_keys[:half]).isdisjoint(
            fresh_keys[half:]
        ),
        fresh_keys_disjoint_from_source=set(fresh_keys).isdisjoint(source_keys),
    )
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_fresh_expanded_context_kernel_diversity",
            "source_run_id": source_gate.get("run_id"),
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "source_contexts": 16,
            "new_contexts": 48,
            "contexts": [f"0x{context:012X}" for context in contexts],
            "new_context_generation_seed": config.seed + 7401,
            "fresh_key_generation_seed": config.seed + 8801,
            "fresh_keys": config.keys,
            "fresh_key_half_size": half,
            "source_key_generation_seed": int(source_metadata["key_generation_seed"]),
            "source_keys": int(source_metadata["keys"]),
            "key_chunk_size": config.key_chunk_size,
            "active_bits": list(ACTIVE_BITS),
            "plaintexts_per_context_per_key": 1 << 16,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def validate_source(
    source_gate: dict[str, Any],
    source_metadata: dict[str, Any],
    source_basis_rows: list[dict[str, str]],
) -> dict[str, bool]:
    return {
        "source_gate_is_e16_context_diversity_pass": (
            source_gate.get("status") == "pass"
            and source_gate.get("decision") == EXPECTED_SOURCE_DECISION
        ),
        "source_metadata_is_e16_context_diversity": (
            source_metadata.get("task") == EXPECTED_SOURCE_TASK
            and source_metadata.get("training_performed") is False
            and int(source_metadata.get("keys", -1)) == 128
        ),
        "source_basis_rows_present": bool(source_basis_rows),
    }


def evaluate_context_fresh_expansion(
    config: ContextFreshExpansionConfig,
    *,
    contexts: tuple[int, ...],
    xor_words_by_context: dict[int, np.ndarray],
    source_bases: dict[int, tuple[int, ...]],
    source_checks: dict[str, bool],
    scalar_matches: bool,
    fresh_key_halves_disjoint: bool,
    fresh_keys_disjoint_from_source: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
    dict[str, Any],
]:
    if len(contexts) != 64 or len(set(contexts)) != 64 or contexts[0] != 0:
        raise ValueError("contexts must contain 64 unique values starting at zero")
    if set(xor_words_by_context) != set(range(64)):
        raise ValueError("xor_words_by_context must contain context ids 0 through 63")
    if set(source_bases) != set(range(16)):
        raise ValueError("source_bases must contain E16 context ids 0 through 15")

    half = config.keys // 2
    paper_masks = paper_basis_masks(output_mapping="direct")
    paper_span = _bounded_span(paper_masks, max_dimension=4)
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_joint_bases_validate = True
    reproduced_source_signatures = 0
    for context_id, context in enumerate(contexts):
        words = np.asarray(xor_words_by_context[context_id], dtype=np.uint64)
        if words.shape != (config.keys,):
            raise ValueError(
                f"xor words for context {context_id} must have shape ({config.keys},)"
            )
        discovery = words[:half]
        validation = words[half:]
        discovery_basis = gf2_kernel_basis(discovery)
        validation_basis = gf2_kernel_basis(validation)
        joint_basis = gf2_kernel_basis(words)
        validates = kernel_basis_valid(discovery, joint_basis) and kernel_basis_valid(
            validation, joint_basis
        )
        all_joint_bases_validate &= validates
        contains_hwang = kernel_basis_valid(words, paper_masks)
        source_signature_reproduced = (
            context_id < 16 and joint_basis == source_bases[context_id]
        )
        reproduced_source_signatures += int(source_signature_reproduced)
        signature = ":".join(f"{vector:016X}" for vector in joint_basis)
        rows.append(
            {
                "run_id": config.run_id,
                "context_id": context_id,
                "context_origin": "e16_anchor" if context_id < 16 else "new",
                "fixed_plaintext": f"0x{context:016X}",
                "discovery_kernel_dimension": len(discovery_basis),
                "validation_kernel_dimension": len(validation_basis),
                "joint_kernel_dimension": len(joint_basis),
                "joint_rank": 64 - len(joint_basis),
                "joint_basis_signature": signature,
                "joint_basis_valid_both_halves": validates,
                "joint_kernel_contains_hwang_span": contains_hwang,
                "joint_kernel_equals_hwang_span": (
                    len(joint_basis) == 4
                    and all(vector in paper_span for vector in joint_basis)
                ),
                "has_directions_beyond_hwang": (
                    contains_hwang and len(joint_basis) > 4
                ),
                "source_signature_reproduced": source_signature_reproduced,
                "nonzero_output_parity_words": int(np.count_nonzero(words)),
            }
        )
        for basis_index, vector in enumerate(joint_basis):
            basis_rows.append(
                {
                    "run_id": config.run_id,
                    "context_id": context_id,
                    "context_origin": "e16_anchor" if context_id < 16 else "new",
                    "fixed_plaintext": f"0x{context:016X}",
                    "basis_index": basis_index,
                    "vector_hex": f"0x{vector:016X}",
                    "vector_weight": vector.bit_count(),
                    "in_hwang_paper_span": vector in paper_span,
                }
            )

    signatures = {str(row["joint_basis_signature"]) for row in rows}
    beyond_hwang = sum(bool(row["has_directions_beyond_hwang"]) for row in rows)
    readiness = {
        **source_checks,
        "sixty_four_unique_contexts_present": len(rows) == 64,
        "first_sixteen_contexts_are_e16_anchors": all(
            row["context_origin"] == "e16_anchor" for row in rows[:16]
        ),
        "fresh_key_halves_nonempty_and_disjoint": fresh_key_halves_disjoint,
        "fresh_keys_disjoint_from_e16_keys": fresh_keys_disjoint_from_source,
        "zero_and_first_new_context_match_scalar": scalar_matches,
        "all_joint_bases_validate_both_fresh_halves": all_joint_bases_validate,
        "all_joint_kernels_contain_hwang_span": all(
            bool(row["joint_kernel_contains_hwang_span"]) for row in rows
        ),
        "zero_context_hwang_basis_stable": bool(
            rows[0]["joint_kernel_contains_hwang_span"]
        ),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "discovery_kernel_dimension",
                "validation_kernel_dimension",
                "joint_kernel_dimension",
                "joint_rank",
                "nonzero_output_parity_words",
            )
        ),
    }
    gate = adjudicate_context_fresh_expansion(
        config,
        readiness,
        reproduced_source_signatures=reproduced_source_signatures,
        distinct_signatures=len(signatures),
        contexts_beyond_hwang=beyond_hwang,
    )
    return rows, basis_rows, readiness, gate


def adjudicate_context_fresh_expansion(
    config: ContextFreshExpansionConfig,
    readiness_checks: dict[str, bool],
    *,
    reproduced_source_signatures: int,
    distinct_signatures: int,
    contexts_beyond_hwang: int,
) -> dict[str, Any]:
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    source_stable = reproduced_source_signatures == 16
    diversity_pass = distinct_signatures >= 8 and contexts_beyond_hwang >= 24
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_fresh_context_protocol_invalid"
        next_action = {
            "action": "repair source validation, fresh-key separation, or Hwang anchor",
            "training": False,
            "remote_scale": False,
        }
    elif not source_stable:
        status = "hold"
        decision = "innovation2_context_kernel_fresh_key_unstable"
        next_action = {
            "action": "stop PRESENT r7 inactive-context output-prediction branch",
            "reason": "E16 context kernel signatures do not reproduce on fresh keys",
            "training": False,
            "remote_scale": False,
        }
    elif diversity_pass:
        status = "pass"
        decision = "innovation2_fresh_expanded_context_kernel_ready"
        next_action = {
            "action": "rebuild 64-context span labels and group-disjoint controls",
            "next_adjudication": "E19 expanded-context label readiness",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_fresh_expanded_context_diversity_insufficient"
        next_action = {
            "action": "stop PRESENT r7 inactive-context output-prediction branch",
            "reason": "fresh contexts do not create enough stable kernel diversity",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "reproduced_e16_context_signatures": reproduced_source_signatures,
        "distinct_joint_kernel_signatures": distinct_signatures,
        "contexts_with_directions_beyond_hwang": contexts_beyond_hwang,
        "claim_scope": (
            "64-context local PRESENT r7 kernel audit under 128 fresh sampled keys; "
            "not a neural result or all-key proof"
        ),
        "next_action": next_action,
    }


def _bounded_span(basis: tuple[int, ...], *, max_dimension: int) -> set[int]:
    if len(basis) > max_dimension:
        raise ValueError("basis exceeds bounded span dimension")
    values = {0}
    for vector in basis:
        values |= {value ^ vector for value in tuple(values)}
    return values
