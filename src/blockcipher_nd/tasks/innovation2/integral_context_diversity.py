from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from blockcipher_nd.tasks.innovation2.integral_bit_transition_audit import (
    BitIntegralStructure,
    scalar_bit_integral_output_xor,
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


ACTIVE_BITS = tuple(range(48, 64))
CONTEXT_COUNT = 16
ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ContextDiversityConfig:
    run_id: str
    seed: int = 0
    rounds: int = 7
    contexts: int = CONTEXT_COUNT
    keys: int = 128
    key_chunk_size: int = 4

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.rounds != 7:
            raise ValueError("the frozen context audit requires PRESENT r7")
        if self.contexts != CONTEXT_COUNT:
            raise ValueError("the frozen context audit requires exactly 16 contexts")
        if self.keys != 128:
            raise ValueError("the frozen context audit requires exactly 128 keys")
        if self.key_chunk_size <= 0:
            raise ValueError("key_chunk_size must be positive")


def inactive_contexts(*, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed + 4401)
    contexts = [0]
    seen = {0}
    while len(contexts) < CONTEXT_COUNT:
        context = int(rng.integers(0, 1 << 48, dtype=np.uint64))
        if context == 0 or context in seen:
            continue
        contexts.append(context)
        seen.add(context)
    return tuple(contexts)


def run_context_diversity_audit(
    config: ContextDiversityConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    contexts = inactive_contexts(seed=config.seed)
    keys = make_keys(count=config.keys, seed=config.seed + 3301)
    half = config.keys // 2
    structures = tuple(
        BitIntegralStructure(
            structure_id=f"present-r7-high16-context-{context_id:02d}",
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
            keys,
            rounds=config.rounds,
            key_chunk_size=config.key_chunk_size,
            progress_callback=progress_callback,
        )
    scalar_matches = all(
        int(xor_words_by_context[context_id][0])
        == scalar_bit_integral_output_xor(
            structures[context_id],
            rounds=config.rounds,
            key=keys[0],
        )
        for context_id in (0, 1)
    )
    rows, basis_rows, readiness, gate = evaluate_context_diversity_audit(
        config,
        contexts=contexts,
        xor_words_by_context=xor_words_by_context,
        scalar_matches=scalar_matches,
        key_halves_disjoint=set(keys[:half]).isdisjoint(keys[half:]),
    )
    return {
        "rows": rows,
        "basis_rows": basis_rows,
        "gate": gate,
        "metadata": {
            "run_id": config.run_id,
            "task": "innovation2_present_r7_inactive_context_kernel_diversity",
            "cipher": "PRESENT-80",
            "rounds": config.rounds,
            "seed": config.seed,
            "context_generation_seed": config.seed + 4401,
            "key_generation_seed": config.seed + 3301,
            "keys": config.keys,
            "key_half_size": half,
            "key_chunk_size": config.key_chunk_size,
            "contexts": [f"0x{context:012X}" for context in contexts],
            "active_bits": list(ACTIVE_BITS),
            "plaintexts_per_context_per_key": 1 << 16,
            "training_performed": False,
            "claim_scope": gate["claim_scope"],
        },
    }


def evaluate_context_diversity_audit(
    config: ContextDiversityConfig,
    *,
    contexts: tuple[int, ...],
    xor_words_by_context: dict[int, np.ndarray],
    scalar_matches: bool,
    key_halves_disjoint: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bool],
    dict[str, Any],
]:
    if len(contexts) != config.contexts or len(set(contexts)) != config.contexts:
        raise ValueError("contexts must contain the 16 unique frozen values")
    if contexts[0] != 0 or any(not 0 <= context < (1 << 48) for context in contexts):
        raise ValueError("contexts must start at zero and fit in the inactive low 48 bits")
    if set(xor_words_by_context) != set(range(config.contexts)):
        raise ValueError("xor_words_by_context must contain context ids 0 through 15")

    half = config.keys // 2
    paper_span = _bounded_span(paper_basis_masks(output_mapping="direct"), max_dimension=4)
    rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    all_joint_bases_validate = True
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
        signature = ":".join(f"{vector:016X}" for vector in joint_basis)
        rows.append(
            {
                "run_id": config.run_id,
                "context_id": context_id,
                "fixed_plaintext": f"0x{context:016X}",
                "discovery_kernel_dimension": len(discovery_basis),
                "validation_kernel_dimension": len(validation_basis),
                "joint_kernel_dimension": len(joint_basis),
                "joint_rank": 64 - len(joint_basis),
                "joint_basis_signature": signature,
                "joint_basis_valid_both_halves": validates,
                "joint_kernel_equals_hwang_span": (
                    len(joint_basis) == 4
                    and all(vector in paper_span for vector in joint_basis)
                ),
                "nonzero_output_parity_words": int(np.count_nonzero(words)),
            }
        )
        for basis_index, vector in enumerate(joint_basis):
            basis_rows.append(
                {
                    "run_id": config.run_id,
                    "context_id": context_id,
                    "fixed_plaintext": f"0x{context:016X}",
                    "basis_index": basis_index,
                    "vector_hex": f"0x{vector:016X}",
                    "vector_weight": vector.bit_count(),
                    "in_hwang_paper_span": vector in paper_span,
                }
            )

    anchor = rows[0]
    signatures = {str(row["joint_basis_signature"]) for row in rows}
    nontrivial = sum(int(row["joint_kernel_dimension"]) > 0 for row in rows)
    readiness = {
        "sixteen_unique_inactive_contexts_present": len(rows) == CONTEXT_COUNT,
        "all_contexts_clear_high16_active_bits": all(
            context >> 48 == 0 for context in contexts
        ),
        "key_halves_nonempty_and_disjoint": key_halves_disjoint,
        "zero_and_first_nonzero_context_match_scalar": scalar_matches,
        "all_joint_bases_validate_both_halves": all_joint_bases_validate,
        "zero_context_hwang_anchor_exact": bool(
            anchor["joint_kernel_equals_hwang_span"]
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
    gate = adjudicate_context_diversity_audit(
        config,
        rows,
        readiness,
        distinct_signatures=len(signatures),
        nontrivial_contexts=nontrivial,
    )
    return rows, basis_rows, readiness, gate


def adjudicate_context_diversity_audit(
    config: ContextDiversityConfig,
    rows: list[dict[str, Any]],
    readiness_checks: dict[str, bool],
    *,
    distinct_signatures: int,
    nontrivial_contexts: int,
) -> dict[str, Any]:
    readiness_pass = bool(readiness_checks) and all(readiness_checks.values())
    diversity_pass = distinct_signatures >= 4 and nontrivial_contexts >= 8
    if not readiness_pass:
        status = "fail"
        decision = "innovation2_inactive_context_protocol_invalid"
        next_action = {
            "action": "repair context construction, scalar anchors, or Hwang calibration",
            "training": False,
            "remote_scale": False,
        }
    elif diversity_pass:
        status = "pass"
        decision = "innovation2_inactive_context_kernel_diversity_ready"
        next_action = {
            "action": "rebuild context-mask labels with marginal controls",
            "next_adjudication": "E17 context output-label shortcut audit",
            "training": False,
            "remote_scale": False,
        }
    else:
        status = "hold"
        decision = "innovation2_inactive_context_kernel_diversity_insufficient"
        next_action = {
            "action": "stop PRESENT r7 multi-structure output-prediction branch",
            "reason": "inactive affine contexts do not create enough stable kernels",
            "next_adjudication": "rank literature-backed alternative cipher families",
            "training": False,
            "remote_scale": False,
        }
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "readiness_checks": readiness_checks,
        "distinct_joint_kernel_signatures": distinct_signatures,
        "nontrivial_joint_kernel_contexts": nontrivial_contexts,
        "claim_scope": (
            "16-context local PRESENT r7 high16 kernel readiness under 128 sampled "
            "keys; not a neural result or all-key proof"
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
