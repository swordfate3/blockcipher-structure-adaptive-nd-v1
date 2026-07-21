from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from blockcipher_nd.ciphers.spn.present import (
    PRESENT_SBOX_ANF,
    Present80,
)
from blockcipher_nd.tasks.innovation2.output_parity_mask_geometry import (
    ALIGNED_MASKS,
    mask_positions,
)
from blockcipher_nd.tasks.innovation2.present_query_cone_sparse_anf_growth import (
    CappedPolynomialOps,
    Polynomial,
    QueryCapExceeded,
    SparseAnfGrowthConfig,
    required_state_cone,
)


RUN_ID = "i2_output_parity_prediction_op8_present_r1_r3_exact_anf_difficulty_20260721"
ROUNDS = (1, 2, 3)
MASK_INDICES = tuple(range(16))
TRAIN_ROWS = 4096
PRESENT_SBOX_PARITY_ANF = tuple(
    sorted(
        set(PRESENT_SBOX_ANF[0])
        ^ set(PRESENT_SBOX_ANF[1])
        ^ set(PRESENT_SBOX_ANF[2])
        ^ set(PRESENT_SBOX_ANF[3])
    )
)


@dataclass(frozen=True)
class OutputParityAnfDifficultyConfig:
    run_id: str = RUN_ID
    maximum_terms: int = 500_000
    maximum_seconds: float = 10.0
    maximum_memory_bytes: int = 1 << 30
    assignment_checks: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if min(self.maximum_terms, self.assignment_checks) <= 0:
            raise ValueError("term and assignment caps must be positive")
        if min(self.maximum_seconds, self.maximum_memory_bytes) <= 0:
            raise ValueError("time and memory caps must be positive")


def run_exact_anf_audit(
    config: OutputParityAnfDifficultyConfig,
    *,
    secret_key: int,
) -> list[dict[str, Any]]:
    rows = []
    for rounds in ROUNDS:
        for mask_index in MASK_INDICES:
            rows.append(
                compute_fixed_key_parity_anf(
                    config,
                    rounds=rounds,
                    mask_index=mask_index,
                    secret_key=secret_key,
                )
            )
    return rows


def compute_fixed_key_parity_anf(
    config: OutputParityAnfDifficultyConfig,
    *,
    rounds: int,
    mask_index: int,
    secret_key: int,
) -> dict[str, Any]:
    if rounds not in ROUNDS:
        raise ValueError("rounds must be one of 1, 2, 3")
    if mask_index not in MASK_INDICES:
        raise ValueError("mask_index must be in [0, 15]")
    mask = ALIGNED_MASKS[mask_index]
    output_bits = mask_positions(mask)
    sparse_config = SparseAnfGrowthConfig(
        run_id=f"{config.run_id}_r{rounds}_m{mask_index:02d}",
        mode="smoke",
        rounds=rounds,
        maximum_terms=config.maximum_terms,
        maximum_seconds=config.maximum_seconds,
        maximum_memory_bytes=config.maximum_memory_bytes,
        seed=config.seed,
    )
    ops = CappedPolynomialOps(sparse_config)
    cone = required_state_cone(rounds=rounds, output_bits=output_bits)
    state: dict[int, Polynomial] = {bit: frozenset({1 << bit}) for bit in cone[0]}
    round_keys = _round_keys(secret_key, rounds)
    inverse_player = _inverse_present_player()
    try:
        round_metrics = []
        for round_counter in range(1, rounds):
            keyed = {
                bit: (
                    ops.xor(state[bit], frozenset({0}))
                    if (round_keys[round_counter - 1] >> bit) & 1
                    else state[bit]
                )
                for bit in cone[round_counter - 1]
            }
            next_state: dict[int, Polynomial] = {}
            for target in cone[round_counter]:
                source = inverse_player[target]
                start = 4 * (source // 4)
                inputs = tuple(keyed[start + lane] for lane in range(4))
                next_state[target] = _evaluate_sbox_output(
                    inputs,
                    output_bit=source % 4,
                    ops=ops,
                )
            state = next_state
            round_metrics.append(
                {
                    "round": round_counter,
                    "input_cone_bits": len(cone[round_counter - 1]),
                    "output_cone_bits": len(cone[round_counter]),
                    "state_terms": sum(
                        len(polynomial) for polynomial in state.values()
                    ),
                    "maximum_state_terms": max(map(len, state.values()), default=0),
                }
            )
        final_keyed = {
            bit: (
                ops.xor(state[bit], frozenset({0}))
                if (round_keys[rounds - 1] >> bit) & 1
                else state[bit]
            )
            for bit in cone[rounds - 1]
        }
        final_start = 4 * mask_index
        final_inputs = tuple(final_keyed[final_start + lane] for lane in range(4))
        parity = _evaluate_anf_terms(
            final_inputs,
            terms=PRESENT_SBOX_PARITY_ANF,
            ops=ops,
        )
        if (round_keys[-1] & mask).bit_count() & 1:
            parity = ops.xor(parity, frozenset({0}))
        round_metrics.append(
            {
                "round": rounds,
                "input_cone_bits": len(cone[rounds - 1]),
                "output_cone_bits": len(cone[rounds]),
                "state_terms": len(parity),
                "maximum_state_terms": len(parity),
                "direct_parity_anf": True,
            }
        )
        resource_metrics = ops.finish()
    except QueryCapExceeded as exc:
        return {
            "run_id": config.run_id,
            "rounds": rounds,
            "mask_index": mask_index,
            "mask_hex": f"{mask:016x}",
            "output_bits": list(output_bits),
            "status": "cap_exceeded",
            "cap_reason": exc.reason,
            "maximum_observed_terms": exc.terms,
            "elapsed_seconds": exc.elapsed_seconds,
            "resident_bytes": exc.resident_bytes,
            "cone_widths": [len(cone[index]) for index in range(rounds + 1)],
        }
    support_mask = 0
    for monomial in parity:
        support_mask |= monomial
    support_bits = tuple(bit for bit in range(64) if (support_mask >> bit) & 1)
    exact_degree = max((monomial.bit_count() for monomial in parity), default=0)
    assignment_checks = _assignment_checks(
        parity,
        rounds=rounds,
        mask=mask,
        mask_index=mask_index,
        secret_key=secret_key,
        count=config.assignment_checks,
        seed=config.seed,
    )
    support_width = len(support_bits)
    monomial_count = len(parity)
    return {
        "run_id": config.run_id,
        "rounds": rounds,
        "mask_index": mask_index,
        "mask_hex": f"{mask:016x}",
        "output_bits": list(output_bits),
        "status": "completed",
        "cap_reason": None,
        "cone_widths": [len(cone[index]) for index in range(rounds + 1)],
        "structural_input_cone_bits": len(cone[0]),
        "functional_support_bits": list(support_bits),
        "functional_support_width": support_width,
        "exact_monomial_count": monomial_count,
        "exact_algebraic_degree": exact_degree,
        "constant_term_present": 0 in parity,
        "train_rows": TRAIN_ROWS,
        "log2_truth_table_size": support_width,
        "log2_train_coverage": math.log2(TRAIN_ROWS) - support_width,
        "train_rows_per_monomial": TRAIN_ROWS / max(1, monomial_count),
        "assignment_checks": assignment_checks,
        "all_assignment_checks_match": all(
            check["matches"] for check in assignment_checks
        ),
        "round_metrics": round_metrics,
        **resource_metrics,
    }


def adjudicate_exact_anf_difficulty(
    config: OutputParityAnfDifficultyConfig,
    rows: list[dict[str, Any]],
    source_gates: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    round_summaries = []
    for rounds in ROUNDS:
        selected = [row for row in completed if row["rounds"] == rounds]
        attempted = [row for row in rows if row["rounds"] == rounds]
        round_summaries.append(
            {
                "rounds": rounds,
                "completed_masks": len(selected),
                "cap_exceeded_masks": sum(
                    row["status"] == "cap_exceeded" for row in attempted
                ),
                "structural_input_cone_median": median(
                    [row["cone_widths"][0] for row in attempted]
                ),
                "functional_support_min": min(
                    (row["functional_support_width"] for row in selected), default=0
                ),
                "functional_support_median": median(
                    [row["functional_support_width"] for row in selected]
                )
                if selected
                else 0,
                "functional_support_max": max(
                    (row["functional_support_width"] for row in selected), default=0
                ),
                "exact_degree_min": min(
                    (row["exact_algebraic_degree"] for row in selected), default=0
                ),
                "exact_degree_median": median(
                    [row["exact_algebraic_degree"] for row in selected]
                )
                if selected
                else 0,
                "exact_degree_max": max(
                    (row["exact_algebraic_degree"] for row in selected), default=0
                ),
                "monomial_count_min": min(
                    (row["exact_monomial_count"] for row in selected), default=0
                ),
                "monomial_count_median": median(
                    [row["exact_monomial_count"] for row in selected]
                )
                if selected
                else 0,
                "monomial_count_max": max(
                    (row["exact_monomial_count"] for row in selected), default=0
                ),
                "maximum_observed_terms": max(
                    (int(row.get("maximum_observed_terms", 0)) for row in attempted),
                    default=0,
                ),
                "log2_train_coverage_median": median(
                    [row["log2_train_coverage"] for row in selected]
                )
                if selected
                else 0,
                "train_rows_per_monomial_median": median(
                    [row["train_rows_per_monomial"] for row in selected]
                )
                if selected
                else 0,
                "two_key_mean_aligned_auc": _source_auc(source_gates[rounds]),
            }
        )
    supports = [summary["functional_support_median"] for summary in round_summaries]
    degrees = [summary["exact_degree_median"] for summary in round_summaries]
    monomials = [summary["monomial_count_median"] for summary in round_summaries]
    aucs = [summary["two_key_mean_aligned_auc"] for summary in round_summaries]
    source_checks = {
        "r1_source_is_two_key_confirmed": source_gates[1]["decision"]
        == "innovation2_output_parity_mask_geometry_two_key_confirmed",
        "r2_source_is_two_key_supported": source_gates[2]["decision"]
        == "innovation2_output_parity_present_r2_two_key_supported",
        "r3_source_is_two_key_not_supported": source_gates[3]["decision"]
        == "innovation2_output_parity_present_r3_two_key_not_supported",
    }
    execution_checks = {
        "all_48_functions_completed": len(completed) == 48,
        "no_function_exceeded_hard_cap": len(completed) == len(rows),
        "all_assignment_checks_match_scalar_present": all(
            row.get("all_assignment_checks_match", False) for row in completed
        ),
        "each_round_has_sixteen_masks": all(
            summary["completed_masks"] == 16 for summary in round_summaries
        ),
        "functional_support_consistent_within_each_round": all(
            summary["functional_support_min"] == summary["functional_support_max"]
            for summary in round_summaries
        ),
        "median_support_strictly_increases": supports[0] < supports[1] < supports[2],
        "median_exact_degree_strictly_increases": degrees[0] < degrees[1] < degrees[2],
        "median_monomials_strictly_increase": monomials[0]
        < monomials[1]
        < monomials[2],
        "all_r3_functions_depend_on_64_plaintext_bits": round_summaries[2][
            "completed_masks"
        ]
        == 16
        and round_summaries[2]["functional_support_min"] == 64,
        "r3_median_monomials_exceed_train_rows": monomials[2] > TRAIN_ROWS,
        "two_key_mean_auc_strictly_decreases": aucs[0] > aucs[1] > aucs[2],
    }
    if (
        not all(source_checks.values())
        or not execution_checks["all_assignment_checks_match_scalar_present"]
    ):
        status = "fail"
        decision = "innovation2_output_parity_exact_anf_difficulty_protocol_invalid"
        action = (
            "repair only source-gate, sparse-ANF, bit-order, or scalar replay protocol"
        )
        next_adjudication = "repair_op8_protocol"
    elif not execution_checks["all_48_functions_completed"]:
        status = "hold"
        decision = "innovation2_output_parity_exact_anf_difficulty_hard_cap_exceeded"
        action = "retain the frozen hard-cap boundary and do not raise caps or scale training"
        next_adjudication = "output_parity_exact_anf_cap_boundary"
    elif all(execution_checks.values()):
        status = "pass"
        decision = "innovation2_output_parity_exact_anf_difficulty_transition_confirmed"
        action = (
            "preregister OP9 nested r3 training prefixes 4096, 8192, and 16384 with "
            "fixed validation/test sets and an aligned-label shuffle control"
        )
        next_adjudication = "op9_present_r3_nested_data_slope"
    else:
        status = "hold"
        decision = (
            "innovation2_output_parity_exact_anf_difficulty_transition_not_confirmed"
        )
        action = (
            "stop data and round scaling for the current fixed-key aligned-mask route"
        )
        next_adjudication = "output_parity_route_reassessment"
    return {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "execution_checks": execution_checks,
        "round_summaries": round_summaries,
        "metrics": {
            "completed_functions": len(completed),
            "cap_exceeded_functions": sum(
                row["status"] == "cap_exceeded" for row in rows
            ),
            "maximum_observed_terms": max(
                (int(row.get("maximum_observed_terms", 0)) for row in rows), default=0
            ),
            "maximum_elapsed_seconds": max(
                (float(row.get("elapsed_seconds", 0.0)) for row in rows), default=0.0
            ),
        },
        "claim_scope": (
            "exact fixed-key GF(2) ANF and functional-support audit for all 16 aligned "
            "ciphertext-output parity functions at PRESENT r1-r3; no neural training, "
            "attack-round, paper-reproduction, or SOTA claim"
        ),
        "next_action": {
            "action": action,
            "next_adjudication": next_adjudication,
            "training": False,
            "remote_scale": False,
            "sample_classification": False,
        },
    }


def serializable_config(config: OutputParityAnfDifficultyConfig) -> dict[str, Any]:
    return asdict(config)


def _round_keys(secret_key: int, rounds: int) -> list[int]:
    register = secret_key & ((1 << 80) - 1)
    keys = []
    for round_counter in range(1, rounds + 1):
        keys.append(register >> 16)
        register = Present80._update_key(register, round_counter)
    keys.append(register >> 16)
    return keys


def _inverse_present_player() -> tuple[int, ...]:
    inverse = [0] * 64
    for source in range(64):
        target = (16 * source) % 63 if source < 63 else 63
        inverse[target] = source
    return tuple(inverse)


def _evaluate_sbox_output(
    inputs: tuple[Polynomial, ...],
    *,
    output_bit: int,
    ops: CappedPolynomialOps,
) -> Polynomial:
    return _evaluate_anf_terms(inputs, terms=PRESENT_SBOX_ANF[output_bit], ops=ops)


def _evaluate_anf_terms(
    inputs: tuple[Polynomial, ...],
    *,
    terms: tuple[int, ...],
    ops: CappedPolynomialOps,
) -> Polynomial:
    output: Polynomial = frozenset()
    for term in terms:
        product: Polynomial = frozenset({0})
        for bit in range(4):
            if term & (1 << bit):
                product = ops.product(product, inputs[bit])
        output = ops.xor(output, product)
    return output


def _assignment_checks(
    polynomial: Polynomial,
    *,
    rounds: int,
    mask: int,
    mask_index: int,
    secret_key: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    checks = []
    cipher = Present80(rounds=rounds, key=secret_key)
    for check_index in range(count):
        payload = f"{rounds}:{mask_index}:{check_index}:{seed}".encode("ascii")
        plaintext = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        exact = _evaluate_polynomial(polynomial, plaintext)
        scalar = (cipher.encrypt(plaintext) & mask).bit_count() & 1
        checks.append(
            {
                "check_index": check_index,
                "plaintext_hex": f"{plaintext:016x}",
                "exact_parity": exact,
                "scalar_parity": scalar,
                "matches": exact == scalar,
            }
        )
    return checks


def _evaluate_polynomial(polynomial: Polynomial, assignment: int) -> int:
    value = 0
    for monomial in polynomial:
        value ^= int(assignment & monomial == monomial)
    return value


def _source_auc(gate: dict[str, Any]) -> float:
    return float(gate["metrics"]["mean_aligned_parity_macro_auc"])
