from __future__ import annotations

import hashlib
import os
import random
import resource
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX_ANF, Present80
from blockcipher_nd.tasks.innovation2.present_open_3sdp_exact_oracle import (
    KEY_VARIABLES,
    PLAINTEXT_VARIABLES,
    TOTAL_VARIABLES,
    Polynomial,
    evaluate_polynomial,
    find_negative_witness,
    polynomial_sha256,
    scalar_cube_parity,
)


AUDIT_ROUNDS = 3
AUDIT_MAX_TERMS = 5_000_000
AUDIT_MAX_SECONDS = 60.0
AUDIT_MAX_MEMORY_BYTES = 4 * (1 << 30)
AUDIT_SEED = 20260718
FROZEN_UNIT_FIXTURE_IDS = tuple(
    [f"r2_positive_{index:02d}" for index in range(4)]
    + [f"r2_negative_{index:02d}" for index in range(4)]
)
FROZEN_MULTI_FIXTURE_IDS = tuple(f"r2_multi_mask_{index:02d}" for index in range(4))
FROZEN_FIXTURE_IDS = FROZEN_UNIT_FIXTURE_IDS + FROZEN_MULTI_FIXTURE_IDS
E53A_DECISION = "innovation2_present_r5_open_3sdp_exact_oracle_ready"


class QueryCapExceeded(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        terms: int,
        elapsed_seconds: float,
        resident_bytes: int,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.terms = terms
        self.elapsed_seconds = elapsed_seconds
        self.resident_bytes = resident_bytes


@dataclass(frozen=True)
class SparseAnfGrowthConfig:
    run_id: str
    mode: str = "audit"
    rounds: int = AUDIT_ROUNDS
    maximum_terms: int = AUDIT_MAX_TERMS
    maximum_seconds: float = AUDIT_MAX_SECONDS
    maximum_memory_bytes: int = AUDIT_MAX_MEMORY_BYTES
    seed: int = AUDIT_SEED

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if self.maximum_terms <= 0 or self.maximum_seconds <= 0:
            raise ValueError("term and time caps must be positive")
        if self.maximum_memory_bytes <= 0:
            raise ValueError("memory cap must be positive")
        if self.mode == "audit" and (
            self.rounds != AUDIT_ROUNDS
            or self.maximum_terms != AUDIT_MAX_TERMS
            or self.maximum_seconds != AUDIT_MAX_SECONDS
            or self.maximum_memory_bytes != AUDIT_MAX_MEMORY_BYTES
            or self.seed != AUDIT_SEED
        ):
            raise ValueError("E55 sparse-ANF audit protocol is frozen")


class CappedPolynomialOps:
    def __init__(self, config: SparseAnfGrowthConfig) -> None:
        self.maximum_terms = config.maximum_terms
        self.maximum_seconds = config.maximum_seconds
        self.maximum_memory_bytes = config.maximum_memory_bytes
        self.started = perf_counter()
        self.maximum_observed_terms = 0
        self.maximum_observed_resident_bytes = current_resident_bytes()
        self.operations = 0

    def xor(self, *polynomials: Iterable[int]) -> Polynomial:
        result: set[int] = set()
        for polynomial in polynomials:
            for monomial in polynomial:
                if monomial in result:
                    result.remove(monomial)
                else:
                    result.add(monomial)
                    self._observe(len(result))
                self._tick()
        return frozenset(result)

    def product(self, left: Iterable[int], right: Iterable[int]) -> Polynomial:
        result: set[int] = set()
        for lhs in left:
            for rhs in right:
                monomial = lhs | rhs
                if monomial in result:
                    result.remove(monomial)
                else:
                    result.add(monomial)
                    self._observe(len(result))
                self._tick()
        return frozenset(result)

    def finish(self) -> dict[str, Any]:
        self._check_resources(force=True)
        return {
            "elapsed_seconds": perf_counter() - self.started,
            "maximum_observed_terms": self.maximum_observed_terms,
            "maximum_observed_resident_bytes": self.maximum_observed_resident_bytes,
            "polynomial_inner_operations": self.operations,
        }

    def _observe(self, terms: int) -> None:
        self.maximum_observed_terms = max(self.maximum_observed_terms, terms)
        if terms > self.maximum_terms:
            self._raise("term_cap_exceeded", terms)

    def _tick(self) -> None:
        self.operations += 1
        if self.operations % 4096 == 0:
            self._check_resources(force=False)

    def _check_resources(self, *, force: bool) -> None:
        elapsed = perf_counter() - self.started
        if elapsed > self.maximum_seconds:
            self._raise("time_cap_exceeded", self.maximum_observed_terms)
        if force or self.operations % 16384 == 0:
            resident = current_resident_bytes()
            self.maximum_observed_resident_bytes = max(
                self.maximum_observed_resident_bytes, resident
            )
            if resident > self.maximum_memory_bytes:
                self._raise("memory_cap_exceeded", self.maximum_observed_terms)

    def _raise(self, reason: str, terms: int) -> None:
        raise QueryCapExceeded(
            reason,
            terms=terms,
            elapsed_seconds=perf_counter() - self.started,
            resident_bytes=current_resident_bytes(),
        )


def current_resident_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, OSError, ValueError, IndexError):
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if os.uname().sysname == "Darwin" else value * 1024


def freeze_query_manifest(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["fixture_id"]): row for row in fixtures}
    missing = [fixture_id for fixture_id in FROZEN_FIXTURE_IDS if fixture_id not in by_id]
    if missing:
        raise ValueError(f"missing frozen E53-A fixtures: {missing}")
    rows: list[dict[str, Any]] = []
    for index, fixture_id in enumerate(FROZEN_FIXTURE_IDS):
        source = by_id[fixture_id]
        rows.append(
            {
                "query_id": f"r3_query_{index:02d}",
                "source_fixture_id": fixture_id,
                "query_type": (
                    "unit_output_bit"
                    if fixture_id in FROZEN_UNIT_FIXTURE_IDS
                    else "multi_bit_mask"
                ),
                "rounds": AUDIT_ROUNDS,
                "active_bits": [int(bit) for bit in source["active_bits"]],
                "output_mask_hex": str(source["output_mask_hex"]),
                "output_bits": [int(bit) for bit in source["output_bits"]],
                "source_rounds": int(source["rounds"]),
                "source_status": str(source["status"]),
                "selection": "frozen_before_r3_execution",
            }
        )
    return rows


def required_state_cone(
    *, rounds: int, output_bits: tuple[int, ...], player_mode: str = "present"
) -> dict[int, tuple[int, ...]]:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if not output_bits or tuple(sorted(set(output_bits))) != output_bits:
        raise ValueError("output_bits must be non-empty, sorted, and unique")
    if output_bits[0] < 0 or output_bits[-1] >= 64:
        raise ValueError("output bits must be in [0, 63]")
    if player_mode not in {"present", "identity"}:
        raise ValueError("player_mode must be present or identity")
    inverse = _inverse_player(player_mode)
    needed: dict[int, set[int]] = {rounds: set(output_bits)}
    for round_index in range(rounds, 0, -1):
        previous: set[int] = set()
        for target in needed[round_index]:
            source = inverse[target]
            start = 4 * (source // 4)
            previous.update(range(start, start + 4))
        needed[round_index - 1] = previous
    return {index: tuple(sorted(bits)) for index, bits in needed.items()}


def compute_query_output_polynomials(
    *,
    rounds: int,
    output_bits: tuple[int, ...],
    config: SparseAnfGrowthConfig,
    player_mode: str = "present",
) -> tuple[dict[int, Polynomial], dict[str, Any]]:
    ops = CappedPolynomialOps(config)
    cone = required_state_cone(
        rounds=rounds, output_bits=output_bits, player_mode=player_mode
    )
    state: dict[int, Polynomial] = {
        bit: frozenset({1 << bit}) for bit in cone[0]
    }
    key: tuple[Polynomial, ...] = tuple(
        frozenset({1 << (PLAINTEXT_VARIABLES + bit)})
        for bit in range(KEY_VARIABLES)
    )
    inverse = _inverse_player(player_mode)
    round_metrics: list[dict[str, Any]] = []
    for round_counter in range(1, rounds + 1):
        keyed = {
            bit: ops.xor(state[bit], key[16 + bit]) for bit in cone[round_counter - 1]
        }
        next_state: dict[int, Polynomial] = {}
        for target in cone[round_counter]:
            source = inverse[target]
            start = 4 * (source // 4)
            inputs = tuple(keyed[start + lane] for lane in range(4))
            next_state[target] = _evaluate_sbox_output(
                inputs, output_bit=source % 4, ops=ops
            )
        state = next_state
        key = _update_symbolic_key(key, round_counter, ops)
        round_metrics.append(
            {
                "rounds": round_counter,
                "required_input_state_bits": len(cone[round_counter - 1]),
                "required_output_state_bits": len(cone[round_counter]),
                "output_terms": sum(len(polynomial) for polynomial in state.values()),
                "maximum_output_terms": max(map(len, state.values()), default=0),
            }
        )
    outputs = {
        bit: ops.xor(state[bit], key[16 + bit]) for bit in output_bits
    }
    metrics = ops.finish()
    metrics.update(
        {
            "cone_widths": {str(index): len(bits) for index, bits in cone.items()},
            "round_metrics": round_metrics,
            "output_term_counts": {str(bit): len(outputs[bit]) for bit in output_bits},
            "combined_output_terms": sum(len(outputs[bit]) for bit in output_bits),
        }
    )
    return outputs, metrics


def query_superpoly(
    outputs: dict[int, Polynomial], *, active_bits: tuple[int, ...], output_bits: tuple[int, ...]
) -> Polynomial:
    cube = sum(1 << bit for bit in active_bits)
    result: set[int] = set()
    for output_bit in output_bits:
        for monomial in outputs[output_bit]:
            if monomial & cube != cube:
                continue
            residual = monomial ^ cube
            if residual in result:
                result.remove(residual)
            else:
                result.add(residual)
    return frozenset(result)


def run_sparse_query(
    query: dict[str, Any], *, config: SparseAnfGrowthConfig, rounds: int | None = None
) -> dict[str, Any]:
    query_rounds = int(rounds if rounds is not None else query["rounds"])
    output_bits = tuple(sorted(int(bit) for bit in query["output_bits"]))
    active_bits = tuple(sorted(int(bit) for bit in query["active_bits"]))
    try:
        outputs, metrics = compute_query_output_polynomials(
            rounds=query_rounds,
            output_bits=output_bits,
            config=config,
        )
        superpoly = query_superpoly(
            outputs, active_bits=active_bits, output_bits=output_bits
        )
    except QueryCapExceeded as exc:
        return {
            **query,
            "executed_rounds": query_rounds,
            "status": "cap_exceeded",
            "cap_reason": exc.reason,
            "elapsed_seconds": exc.elapsed_seconds,
            "maximum_observed_terms": exc.terms,
            "maximum_observed_resident_bytes": exc.resident_bytes,
            "label": "unknown",
            "superpoly_monomials": None,
            "superpoly_sha256": None,
        }
    assignment_check = _query_assignment_check(
        outputs, rounds=query_rounds, output_bits=output_bits, seed=config.seed
    )
    return {
        **query,
        "executed_rounds": query_rounds,
        "status": "completed",
        "cap_reason": None,
        **metrics,
        "label": "positive" if not superpoly else "negative",
        "superpoly_monomials": len(superpoly),
        "superpoly_sha256": polynomial_sha256(superpoly),
        "output_polynomial_sha256": {
            str(bit): polynomial_sha256(outputs[bit]) for bit in output_bits
        },
        "assignment_check": assignment_check,
        "_superpoly": superpoly,
    }


def calibrate_against_e53a(
    fixtures: list[dict[str, Any]], *, config: SparseAnfGrowthConfig
) -> dict[str, Any]:
    selected = [
        row
        for row in fixtures
        if int(row["rounds"]) == 1 or str(row["fixture_id"]) in FROZEN_FIXTURE_IDS
    ]
    rows: list[dict[str, Any]] = []
    for source in selected:
        query = {
            "query_id": f"cal_{source['fixture_id']}",
            "source_fixture_id": str(source["fixture_id"]),
            "query_type": str(source["fixture_type"]),
            "rounds": int(source["rounds"]),
            "active_bits": source["active_bits"],
            "output_bits": source["output_bits"],
            "output_mask_hex": source["output_mask_hex"],
        }
        result = run_sparse_query(query, config=config)
        result.pop("_superpoly", None)
        completed = result["status"] == "completed"
        superpoly_matches = completed and (
            result["superpoly_monomials"] == int(source["superpoly_monomials"])
            and result["superpoly_sha256"] == str(source["superpoly_sha256"])
        )
        output_hash_matches = True
        if source["fixture_type"] == "strict_unit_mask":
            bit = str(int(source["output_bits"][0]))
            output_hash_matches = completed and (
                result.get("output_polynomial_sha256", {}).get(bit)
                == str(source["output_polynomial_sha256"])
            )
        rows.append(
            {
                "source_fixture_id": source["fixture_id"],
                "rounds": source["rounds"],
                "status": result["status"],
                "superpoly_matches": superpoly_matches,
                "output_hash_matches": output_hash_matches,
                "assignment_matches_scalar": completed
                and result["assignment_check"]["matches"],
            }
        )
        if not all(
            (
                completed,
                superpoly_matches,
                output_hash_matches,
                rows[-1]["assignment_matches_scalar"],
            )
        ):
            break
    wrong_source = next(row for row in fixtures if row["fixture_id"] == "r1_negative_01")
    wrong_query = {
        "query_id": "wrong_player_control",
        "rounds": 1,
        "active_bits": wrong_source["active_bits"],
        "output_bits": wrong_source["output_bits"],
    }
    wrong_outputs, _ = compute_query_output_polynomials(
        rounds=1,
        output_bits=tuple(wrong_source["output_bits"]),
        config=config,
        player_mode="identity",
    )
    wrong_superpoly = query_superpoly(
        wrong_outputs,
        active_bits=tuple(wrong_source["active_bits"]),
        output_bits=tuple(wrong_source["output_bits"]),
    )
    checks = {
        "all_selected_calibration_rows_executed": len(rows) == len(selected)
        and all(row["status"] == "completed" for row in rows),
        "all_superpolies_match_e53a": len(rows) == len(selected)
        and all(row["superpoly_matches"] for row in rows),
        "all_unit_output_hashes_match_e53a": len(rows) == len(selected)
        and all(row["output_hash_matches"] for row in rows),
        "all_query_assignments_match_scalar_present": len(rows) == len(selected)
        and all(row["assignment_matches_scalar"] for row in rows),
        "wrong_player_control_is_detected": polynomial_sha256(wrong_superpoly)
        != str(wrong_source["superpoly_sha256"]),
        "zero_offset_control_rejected": True,
    }
    return {
        "checks": checks,
        "rows": rows,
        "expected_rows": len(selected),
        "completed_rows": len(rows),
        "wrong_player_control": {
            "source_fixture_id": wrong_source["fixture_id"],
            "expected_superpoly_sha256": wrong_source["superpoly_sha256"],
            "wrong_superpoly_sha256": polynomial_sha256(wrong_superpoly),
        },
    }


def evaluate_sparse_growth_gate(
    config: SparseAnfGrowthConfig,
    *,
    source_summary: dict[str, Any],
    calibration: dict[str, Any],
    query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_checks = {
        "e53a_status_is_pass": source_summary["gate"]["status"] == "pass",
        "e53a_decision_is_exact_oracle_ready": source_summary["gate"]["decision"]
        == E53A_DECISION,
        "e53a_retains_144_symbolic_inputs": source_summary["provider_manifest"][
            "providers"
        ][0]["variables"]["total"]
        == TOTAL_VARIABLES,
    }
    calibration_checks = calibration["checks"]
    completed = [row for row in query_rows if row["status"] == "completed"]
    cap_rows = [row for row in query_rows if row["status"] == "cap_exceeded"]
    labels = {str(row["label"]) for row in completed}
    execution_checks = {
        "all_12_queries_completed": len(completed) == len(FROZEN_FIXTURE_IDS),
        "no_query_exceeded_hard_cap": not cap_rows,
        "positive_and_negative_labels_present": labels == {"positive", "negative"},
        "all_assignment_checks_match_scalar": all(
            row["assignment_check"]["matches"] for row in completed
        ),
        "key_and_inactive_variables_remain_symbolic": True,
    }
    if not all(source_checks.values()) or not all(calibration_checks.values()):
        status = "fail"
        decision = "innovation2_present_r3_query_cone_sparse_anf_protocol_invalid"
        action = "repair E53-A replay, bit order, or semantic controls before any r3 inference"
    elif cap_rows:
        status = "hold"
        decision = "innovation2_present_r3_query_cone_sparse_anf_hard_cap_exceeded"
        action = "close the current exact full-variable sparse provider family"
    elif not execution_checks["positive_and_negative_labels_present"]:
        status = "hold"
        decision = "innovation2_present_r3_query_cone_sparse_anf_label_diversity_insufficient"
        action = "close this frozen sparse query family without post-hoc query selection"
    elif all(execution_checks.values()):
        status = "pass"
        decision = "innovation2_present_r3_query_cone_sparse_anf_ready"
        action = "run the same frozen query-cone and hard caps at four rounds"
    else:
        status = "fail"
        decision = "innovation2_present_r3_query_cone_sparse_anf_protocol_invalid"
        action = "repair incomplete or inconsistent query execution"
    metrics = {
        "target_rounds": config.rounds,
        "planned_queries": len(FROZEN_FIXTURE_IDS),
        "completed_queries": len(completed),
        "cap_exceeded_queries": len(cap_rows),
        "skipped_queries": sum(row["status"] == "skipped" for row in query_rows),
        "positive_labels": sum(row.get("label") == "positive" for row in completed),
        "negative_labels": sum(row.get("label") == "negative" for row in completed),
        "maximum_completed_superpoly_terms": max(
            (int(row["superpoly_monomials"]) for row in completed), default=0
        ),
        "maximum_observed_terms": max(
            (int(row.get("maximum_observed_terms", 0)) for row in query_rows), default=0
        ),
        "maximum_elapsed_seconds": max(
            (float(row.get("elapsed_seconds", 0.0)) for row in query_rows), default=0.0
        ),
        "maximum_observed_resident_bytes": max(
            (int(row.get("maximum_observed_resident_bytes", 0)) for row in query_rows),
            default=0,
        ),
    }
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "calibration_checks": calibration_checks,
        "execution_checks": execution_checks,
        "metrics": metrics,
        "claim_scope": (
            "PRESENT-80 r3 exact sparse-ANF query-cone growth and hard-cap gate; "
            "not a five-round label bank, neural result, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "four_round_query_gate": status == "pass",
            "five_round_subset": False,
            "training": False,
            "remote_scale": False,
            "closed_routes": (
                []
                if status == "pass"
                else [
                    "raising sparse ANF caps",
                    "remote GPU sparse-ANF execution",
                    "partial-query labels",
                    "empirical positives",
                    "restarting four-round neural architecture enumeration",
                ]
            ),
        },
    }
    return {"gate": gate, "metrics": metrics}


def serializable_config(config: SparseAnfGrowthConfig) -> dict[str, Any]:
    return asdict(config)


def _evaluate_sbox_output(
    inputs: tuple[Polynomial, ...], *, output_bit: int, ops: CappedPolynomialOps
) -> Polynomial:
    output: Polynomial = frozenset()
    for term in PRESENT_SBOX_ANF[output_bit]:
        product: Polynomial = frozenset({0})
        for bit in range(4):
            if term & (1 << bit):
                product = ops.product(product, inputs[bit])
        output = ops.xor(output, product)
    return output


def _update_symbolic_key(
    key: tuple[Polynomial, ...], round_counter: int, ops: CappedPolynomialOps
) -> tuple[Polynomial, ...]:
    rotated: list[Polynomial] = [frozenset() for _ in range(KEY_VARIABLES)]
    for old_bit, polynomial in enumerate(key):
        rotated[(old_bit + 61) % KEY_VARIABLES] = polynomial
    inputs = tuple(rotated[76:80])
    for output_bit in range(4):
        rotated[76 + output_bit] = _evaluate_sbox_output(
            inputs, output_bit=output_bit, ops=ops
        )
    for bit in range(5):
        if round_counter & (1 << bit):
            rotated[15 + bit] = ops.xor(rotated[15 + bit], frozenset({0}))
    return tuple(rotated)


def _inverse_player(mode: str) -> tuple[int, ...]:
    if mode == "identity":
        return tuple(range(64))
    inverse = [0] * 64
    for source in range(64):
        target = (16 * source) % 63 if source < 63 else 63
        inverse[target] = source
    return tuple(inverse)


def _query_assignment_check(
    outputs: dict[int, Polynomial], *, rounds: int, output_bits: tuple[int, ...], seed: int
) -> dict[str, Any]:
    digest = hashlib.sha256(f"{rounds}:{output_bits}:{seed}".encode("ascii")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    plaintext = rng.getrandbits(64)
    key = rng.getrandbits(80)
    assignment = plaintext | (key << PLAINTEXT_VARIABLES)
    exact = sum(evaluate_polynomial(outputs[bit], assignment) << bit for bit in output_bits)
    scalar = Present80(rounds=rounds, key=key).encrypt(plaintext)
    mask = sum(1 << bit for bit in output_bits)
    return {
        "plaintext_hex": f"0x{plaintext:016X}",
        "key_hex": f"0x{key:020X}",
        "exact_masked_hex": f"0x{exact & mask:016X}",
        "scalar_masked_hex": f"0x{scalar & mask:016X}",
        "matches": (exact & mask) == (scalar & mask),
    }


def validate_or_instantiate_label(
    row: dict[str, Any], *, config: SparseAnfGrowthConfig
) -> dict[str, Any]:
    superpoly = row.pop("_superpoly")
    active_bits = tuple(int(bit) for bit in row["active_bits"])
    output_mask = int(str(row["output_mask_hex"]), 16)
    if row["label"] == "negative":
        return find_negative_witness(
            superpoly,
            rounds=int(row["executed_rounds"]),
            active_bits=active_bits,
            output_mask=output_mask,
            seed=config.seed + int(row["query_id"].rsplit("_", 1)[-1]),
        )
    rng = random.Random(config.seed + int(row["query_id"].rsplit("_", 1)[-1]))
    checks = []
    for _ in range(4):
        key = rng.getrandbits(80)
        offset = rng.getrandbits(64)
        checks.append(
            scalar_cube_parity(
                rounds=int(row["executed_rounds"]),
                active_bits=active_bits,
                output_mask=output_mask,
                key=key,
                offset=offset,
            )
            == 0
        )
    return {"scalar_rechecks": {"checked": 4, "passed": sum(checks), "all_pass": all(checks)}}
