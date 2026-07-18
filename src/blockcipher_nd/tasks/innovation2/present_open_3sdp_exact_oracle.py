from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable

from blockcipher_nd.ciphers.spn.present import (
    PRESENT_SBOX,
    PRESENT_SBOX_ANF,
    Present80,
)


PLAINTEXT_VARIABLES = 64
KEY_VARIABLES = 80
TOTAL_VARIABLES = PLAINTEXT_VARIABLES + KEY_VARIABLES
AUDIT_ROUNDS = (1, 2)
AUDIT_FIXTURES_PER_CLASS = 8
AUDIT_VECTOR_CHECKS = {1: 8, 2: 4}
AUDIT_SEED = 20260718

Polynomial = frozenset[int]


@dataclass(frozen=True)
class ExactOracleConfig:
    run_id: str
    mode: str = "audit"
    rounds: tuple[int, ...] = AUDIT_ROUNDS
    fixtures_per_class: int = AUDIT_FIXTURES_PER_CLASS
    seed: int = AUDIT_SEED

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if not self.rounds or tuple(sorted(set(self.rounds))) != self.rounds:
            raise ValueError("rounds must be sorted and unique")
        if self.rounds[0] < 1 or self.rounds[-1] > 2:
            raise ValueError("the exact-oracle readiness scope is one or two rounds")
        if self.fixtures_per_class <= 0:
            raise ValueError("fixtures_per_class must be positive")
        if self.mode == "audit" and (
            self.rounds != AUDIT_ROUNDS
            or self.fixtures_per_class != AUDIT_FIXTURES_PER_CLASS
            or self.seed != AUDIT_SEED
        ):
            raise ValueError("E53-A audit protocol is frozen")


def polynomial_xor(*polynomials: Iterable[int]) -> Polynomial:
    result: set[int] = set()
    for polynomial in polynomials:
        result.symmetric_difference_update(polynomial)
    return frozenset(result)


def polynomial_product(left: Iterable[int], right: Iterable[int]) -> Polynomial:
    result: set[int] = set()
    for lhs in left:
        for rhs in right:
            monomial = lhs | rhs
            if monomial in result:
                result.remove(monomial)
            else:
                result.add(monomial)
    return frozenset(result)


def evaluate_sbox_anf(inputs: tuple[Polynomial, ...]) -> tuple[Polynomial, ...]:
    if len(inputs) != 4:
        raise ValueError("PRESENT S-box requires four input polynomials")
    outputs: list[Polynomial] = []
    for terms in PRESENT_SBOX_ANF:
        output: Polynomial = frozenset()
        for term in terms:
            product: Polynomial = frozenset({0})
            for bit in range(4):
                if term & (1 << bit):
                    product = polynomial_product(product, inputs[bit])
            output = polynomial_xor(output, product)
        outputs.append(output)
    return tuple(outputs)


def build_present_exact_anf_snapshots(
    rounds: tuple[int, ...],
) -> dict[int, tuple[Polynomial, ...]]:
    if not rounds or rounds[0] < 1 or rounds[-1] > 2:
        raise ValueError("exact ANF snapshots are limited to rounds one and two")
    state: tuple[Polynomial, ...] = tuple(
        frozenset({1 << bit}) for bit in range(PLAINTEXT_VARIABLES)
    )
    key: tuple[Polynomial, ...] = tuple(
        frozenset({1 << (PLAINTEXT_VARIABLES + bit)})
        for bit in range(KEY_VARIABLES)
    )
    snapshots: dict[int, tuple[Polynomial, ...]] = {}
    for round_counter in range(1, max(rounds) + 1):
        keyed_state = tuple(
            polynomial_xor(state[bit], key[16 + bit])
            for bit in range(PLAINTEXT_VARIABLES)
        )
        after_sbox: list[Polynomial] = []
        for nibble in range(16):
            after_sbox.extend(
                evaluate_sbox_anf(keyed_state[4 * nibble : 4 * nibble + 4])
            )
        permuted: list[Polynomial] = [frozenset() for _ in range(64)]
        for bit, polynomial in enumerate(after_sbox):
            target = (16 * bit) % 63 if bit < 63 else 63
            permuted[target] = polynomial
        state = tuple(permuted)
        key = _update_symbolic_key(key, round_counter)
        if round_counter in rounds:
            snapshots[round_counter] = tuple(
                polynomial_xor(state[bit], key[16 + bit])
                for bit in range(PLAINTEXT_VARIABLES)
            )
    return snapshots


def evaluate_polynomial(polynomial: Iterable[int], assignment: int) -> int:
    return sum((monomial & ~assignment) == 0 for monomial in polynomial) & 1


def evaluate_output_polynomials(
    outputs: tuple[Polynomial, ...], plaintext: int, key: int
) -> int:
    assignment = (plaintext & ((1 << 64) - 1)) | (
        (key & ((1 << 80) - 1)) << PLAINTEXT_VARIABLES
    )
    value = 0
    for bit, polynomial in enumerate(outputs):
        value |= evaluate_polynomial(polynomial, assignment) << bit
    return value


def cube_superpoly(
    outputs: tuple[Polynomial, ...],
    active_bits: tuple[int, ...],
    output_mask: int,
) -> Polynomial:
    if not active_bits:
        raise ValueError("active_bits must be non-empty")
    if tuple(sorted(set(active_bits))) != active_bits:
        raise ValueError("active_bits must be sorted and unique")
    if active_bits[0] < 0 or active_bits[-1] >= PLAINTEXT_VARIABLES:
        raise ValueError("active bits must be plaintext coordinates")
    if output_mask <= 0 or output_mask >= 1 << 64:
        raise ValueError("output_mask must be a nonzero 64-bit integer")
    cube = sum(1 << bit for bit in active_bits)
    result: set[int] = set()
    for output_bit in range(64):
        if not output_mask & (1 << output_bit):
            continue
        for monomial in outputs[output_bit]:
            if monomial & cube == cube:
                residual = monomial ^ cube
                if residual in result:
                    result.remove(residual)
                else:
                    result.add(residual)
    return frozenset(result)


def scalar_cube_parity(
    *,
    rounds: int,
    active_bits: tuple[int, ...],
    output_mask: int,
    key: int,
    offset: int,
) -> int:
    active_mask = sum(1 << bit for bit in active_bits)
    fixed_offset = offset & ~active_mask & ((1 << 64) - 1)
    parity = 0
    for assignment in range(1 << len(active_bits)):
        plaintext = fixed_offset
        for variable, bit in enumerate(active_bits):
            plaintext |= ((assignment >> variable) & 1) << bit
        ciphertext = Present80(rounds=rounds, key=key).encrypt(plaintext)
        parity ^= (ciphertext & output_mask).bit_count() & 1
    return parity


def build_strict_fixtures(
    outputs: tuple[Polynomial, ...],
    *,
    rounds: int,
    fixtures_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = [tuple([bit]) for bit in range(16)]
    candidates.extend(tuple(range(4 * nibble, 4 * nibble + 4)) for nibble in range(4))
    selected: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": []}
    output_hashes: dict[int, str] = {}
    for active_bits in candidates:
        for output_bit in range(64):
            output_mask = 1 << output_bit
            superpoly = cube_superpoly(outputs, active_bits, output_mask)
            status = "positive" if not superpoly else "negative"
            if len(selected[status]) >= fixtures_per_class:
                continue
            output_hashes.setdefault(
                output_bit, polynomial_sha256(outputs[output_bit])
            )
            row = {
                "fixture_id": f"r{rounds}_{status}_{len(selected[status]):02d}",
                "fixture_type": "strict_unit_mask",
                "rounds": rounds,
                "active_bits": list(active_bits),
                "output_mask_hex": f"0x{output_mask:016X}",
                "output_bits": [output_bit],
                "status": status,
                "superpoly_monomials": len(superpoly),
                "superpoly_sha256": polynomial_sha256(superpoly),
                "output_polynomial_sha256": output_hashes[output_bit],
                "certificate": (
                    "exact_full_anf_superpoly_is_zero"
                    if status == "positive"
                    else "exact_full_anf_superpoly_is_nonzero"
                ),
                "witness_key_hex": None,
                "witness_offset_hex": None,
                "witness_parity": None,
                "scalar_rechecks": None,
            }
            if status == "positive":
                scalar_checks = _positive_scalar_checks(
                    rounds=rounds,
                    active_bits=active_bits,
                    output_mask=output_mask,
                    seed=seed + rounds * 1000 + output_bit,
                )
                row["scalar_rechecks"] = scalar_checks
            else:
                witness = find_negative_witness(
                    superpoly,
                    rounds=rounds,
                    active_bits=active_bits,
                    output_mask=output_mask,
                    seed=seed + rounds * 1000 + output_bit,
                )
                row.update(witness)
            selected[status].append(row)
            if all(
                len(selected[label]) >= fixtures_per_class
                for label in ("positive", "negative")
            ):
                return selected["positive"] + selected["negative"]
    raise RuntimeError(
        f"could not build {fixtures_per_class} positive and negative r{rounds} fixtures"
    )


def build_multi_mask_fixtures(
    outputs: tuple[Polynomial, ...], *, rounds: int, count: int, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(combinations(range(8), 2)):
        active_bits = (index % 8,)
        output_mask = (1 << left) | (1 << (right + 16))
        left_superpoly = cube_superpoly(outputs, active_bits, 1 << left)
        right_superpoly = cube_superpoly(outputs, active_bits, 1 << (right + 16))
        combined = cube_superpoly(outputs, active_bits, output_mask)
        decomposed = polynomial_xor(left_superpoly, right_superpoly)
        status = "positive" if not combined else "negative"
        row: dict[str, Any] = {
            "fixture_id": f"r{rounds}_multi_mask_{len(rows):02d}",
            "fixture_type": "multi_bit_mask_xor",
            "rounds": rounds,
            "active_bits": list(active_bits),
            "output_mask_hex": f"0x{output_mask:016X}",
            "output_bits": [left, right + 16],
            "status": status,
            "superpoly_monomials": len(combined),
            "superpoly_sha256": polynomial_sha256(combined),
            "component_xor_matches": combined == decomposed,
            "certificate": "selected_output_superpolies_xor_over_gf2",
            "witness_key_hex": None,
            "witness_offset_hex": None,
            "witness_parity": None,
            "scalar_rechecks": None,
        }
        if status == "positive":
            row["scalar_rechecks"] = _positive_scalar_checks(
                rounds=rounds,
                active_bits=active_bits,
                output_mask=output_mask,
                seed=seed + index,
            )
        else:
            row.update(
                find_negative_witness(
                    combined,
                    rounds=rounds,
                    active_bits=active_bits,
                    output_mask=output_mask,
                    seed=seed + index,
                )
            )
        rows.append(row)
        if len(rows) >= count:
            return rows
    raise RuntimeError("not enough deterministic multi-mask fixtures")


def find_negative_witness(
    superpoly: Polynomial,
    *,
    rounds: int,
    active_bits: tuple[int, ...],
    output_mask: int,
    seed: int,
    attempts: int = 256,
) -> dict[str, Any]:
    if not superpoly:
        raise ValueError("negative witness requires a nonzero superpoly")
    active_mask = sum(1 << bit for bit in active_bits)
    rng = random.Random(seed)
    for _ in range(attempts):
        key = rng.getrandbits(80)
        offset = rng.getrandbits(64) & ~active_mask
        assignment = offset | (key << PLAINTEXT_VARIABLES)
        if evaluate_polynomial(superpoly, assignment) != 1:
            continue
        parity = scalar_cube_parity(
            rounds=rounds,
            active_bits=active_bits,
            output_mask=output_mask,
            key=key,
            offset=offset,
        )
        if parity != 1:
            raise AssertionError("exact superpoly witness disagrees with scalar PRESENT")
        return {
            "witness_key_hex": f"0x{key:020X}",
            "witness_offset_hex": f"0x{offset:016X}",
            "witness_parity": parity,
        }
    raise RuntimeError("could not instantiate a nonzero exact superpoly")


def validate_exact_outputs(
    outputs: tuple[Polynomial, ...], *, rounds: int, count: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed + rounds * 10000)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        plaintext = rng.getrandbits(64)
        key = rng.getrandbits(80)
        exact = evaluate_output_polynomials(outputs, plaintext, key)
        scalar = Present80(rounds=rounds, key=key).encrypt(plaintext)
        rows.append(
            {
                "index": index,
                "plaintext_hex": f"0x{plaintext:016X}",
                "key_hex": f"0x{key:020X}",
                "exact_hex": f"0x{exact:016X}",
                "scalar_hex": f"0x{scalar:016X}",
                "matches": exact == scalar,
            }
        )
    return {
        "checked": len(rows),
        "passed": sum(row["matches"] for row in rows),
        "all_pass": all(row["matches"] for row in rows),
        "rows": rows,
    }


def audit_sbox_transition_parity() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    order_invariant = True
    for output_exponent in range(16):
        selected = [
            bit for bit in range(4) if output_exponent & (1 << bit)
        ]
        counts = _raw_sbox_trail_counts(selected)
        reverse_counts = _raw_sbox_trail_counts(tuple(reversed(selected)))
        order_invariant &= counts == reverse_counts
        exact = _selected_sbox_output_polynomial(selected)
        for input_exponent in range(16):
            trail_count = counts.get(input_exponent, 0)
            parity = trail_count & 1
            exact_coefficient = int(input_exponent in exact)
            rows.append(
                {
                    "input_exponent": input_exponent,
                    "output_exponent": output_exponent,
                    "trail_count": trail_count,
                    "existence_only": trail_count > 0,
                    "trail_parity": parity,
                    "exact_coefficient": exact_coefficient,
                    "matches_exact": parity == exact_coefficient,
                    "existence_false_positive": trail_count > 0 and parity == 0,
                }
            )
    false_positives = [row for row in rows if row["existence_false_positive"]]
    return {
        "rows": rows,
        "metrics": {
            "candidate_transitions": len(rows),
            "existence_transitions": sum(row["existence_only"] for row in rows),
            "odd_parity_transitions": sum(row["trail_parity"] for row in rows),
            "existence_only_false_positives": len(false_positives),
            "maximum_cancelled_trail_count": max(
                (row["trail_count"] for row in false_positives), default=0
            ),
        },
        "checks": {
            "trail_parity_matches_exact_sbox_anf": all(
                row["matches_exact"] for row in rows
            ),
            "trail_order_invariant": order_invariant,
            "existence_only_control_is_rejected": bool(false_positives),
        },
        "cancellation_examples": false_positives[:16],
    }


def evaluate_exact_oracle_readiness(
    config: ExactOracleConfig,
    *,
    snapshots: dict[int, tuple[Polynomial, ...]],
    fixtures: list[dict[str, Any]],
    vector_checks: dict[int, dict[str, Any]],
    transition: dict[str, Any],
    glpk: dict[str, Any],
) -> dict[str, Any]:
    by_round: dict[int, dict[str, int]] = {}
    for rounds in config.rounds:
        strict = [
            row
            for row in fixtures
            if row["rounds"] == rounds
            and row["fixture_type"] == "strict_unit_mask"
        ]
        by_round[rounds] = {
            "positive": sum(row["status"] == "positive" for row in strict),
            "negative": sum(row["status"] == "negative" for row in strict),
            "multi_mask": sum(
                row["rounds"] == rounds
                and row["fixture_type"] == "multi_bit_mask_xor"
                for row in fixtures
            ),
        }
    positive_rows = [row for row in fixtures if row["status"] == "positive"]
    negative_rows = [row for row in fixtures if row["status"] == "negative"]
    multi_rows = [
        row for row in fixtures if row["fixture_type"] == "multi_bit_mask_xor"
    ]
    monomial_counts = {
        rounds: [len(polynomial) for polynomial in snapshots[rounds]]
        for rounds in config.rounds
    }
    protocol_checks = {
        "audit_protocol_frozen": config.mode != "audit"
        or (
            config.rounds == AUDIT_ROUNDS
            and config.fixtures_per_class == AUDIT_FIXTURES_PER_CLASS
            and config.seed == AUDIT_SEED
        ),
        "official_present_vector_matches": Present80(rounds=31, key=0).encrypt(0)
        == 0x5579C1387B228445,
        "sbox_anf_reconstructs": all(
            _evaluate_sbox_coordinates(value) == PRESENT_SBOX[value]
            for value in range(16)
        ),
        "player_is_permutation": sorted(
            (16 * bit) % 63 if bit < 63 else 63 for bit in range(64)
        )
        == list(range(64)),
        "exact_outputs_match_scalar_present": all(
            vector_checks[rounds]["all_pass"] for rounds in config.rounds
        ),
    }
    fixture_checks = {
        "each_round_meets_positive_fixture_count": all(
            by_round[rounds]["positive"] >= config.fixtures_per_class
            for rounds in config.rounds
        ),
        "each_round_meets_negative_fixture_count": all(
            by_round[rounds]["negative"] >= config.fixtures_per_class
            for rounds in config.rounds
        ),
        "all_positive_scalar_rechecks_pass": all(
            row["scalar_rechecks"]["all_pass"] for row in positive_rows
        ),
        "all_negative_witnesses_scalar_validate": all(
            row["witness_parity"] == 1 for row in negative_rows
        ),
        "all_multi_mask_component_xors_match": all(
            row["component_xor_matches"] for row in multi_rows
        ),
        "nonzero_superpoly_retains_key_variables": any(
            _superpoly_has_key_variable(row, snapshots) for row in negative_rows
        ),
        "nonzero_superpoly_retains_inactive_plaintext_variables": any(
            _superpoly_has_inactive_plaintext_variable(row, snapshots)
            for row in negative_rows
        ),
    }
    transition_checks = transition["checks"]
    glpk_checks = {
        "sage_available": bool(glpk.get("sage_executable")),
        "glpk_backend_verified": glpk.get("backend") == "GLPKBackend",
        "binary_milp_fixture_passed": bool(glpk.get("binary_fixture_passed")),
        "glpk_trail_enumerator_implemented": False,
        "five_round_subset_executed": False,
    }
    exact_oracle_ready = all(protocol_checks.values()) and all(
        fixture_checks.values()
    ) and all(transition_checks.values()) and all(
        glpk_checks[key]
        for key in (
            "sage_available",
            "glpk_backend_verified",
            "binary_milp_fixture_passed",
        )
    )
    if not all(protocol_checks.values()) or not all(fixture_checks.values()):
        status = "fail"
        decision = "innovation2_present_r5_open_3sdp_exact_oracle_invalid"
        action = "repair exact ANF, PRESENT bit order, fixture, mask XOR, or witness protocol"
    elif not all(transition_checks.values()):
        status = "fail"
        decision = "innovation2_present_r5_open_3sdp_cancellation_control_failed"
        action = "repair trail parity before implementing any solver provider"
    elif not exact_oracle_ready:
        status = "hold"
        decision = "innovation2_present_r5_open_3sdp_glpk_runtime_not_ready"
        action = "repair the local Sage/GLPK runtime without changing label semantics"
    else:
        status = "pass"
        decision = "innovation2_present_r5_open_3sdp_exact_oracle_ready"
        action = (
            "implement a GLPK trail enumerator and require exact agreement on all "
            "one/two-round fixtures before the frozen five-round 16x64 subset"
        )
    metrics = {
        "fixture_counts_by_round": by_round,
        "positive_fixtures": len(positive_rows),
        "negative_fixtures": len(negative_rows),
        "multi_mask_fixtures": len(multi_rows),
        "round_monomial_metrics": {
            str(rounds): {
                "minimum": min(monomial_counts[rounds]),
                "maximum": max(monomial_counts[rounds]),
                "total": sum(monomial_counts[rounds]),
            }
            for rounds in config.rounds
        },
        "vector_checks": {
            str(rounds): {
                key: value
                for key, value in vector_checks[rounds].items()
                if key != "rows"
            }
            for rounds in config.rounds
        },
        "transition": transition["metrics"],
        "glpk": glpk,
    }
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "protocol_checks": protocol_checks,
        "fixture_checks": fixture_checks,
        "transition_checks": transition_checks,
        "glpk_checks": glpk_checks,
        "metrics": metrics,
        "claim_scope": (
            "PRESENT-80 one/two-round exact-ANF and local monomial-transition "
            "readiness for an open 3SDP provider; not a five-round label result, "
            "neural training, a distinguisher, an attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "training": False,
            "remote_scale": False,
            "five_round_subset": False,
            "closed_routes": [
                "existence-only or 2SDP labels without GF(2) cancellation",
                "empirical finite-key positives",
                "five-round neural training before provider agreement",
                "remote GPU scale",
            ],
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_present_r5_open_3sdp_exact_oracle",
            "scope": f"present_r{rounds}_exact_anf",
            "rounds": rounds,
            "positive_fixtures": by_round[rounds]["positive"],
            "negative_fixtures": by_round[rounds]["negative"],
            "multi_mask_fixtures": by_round[rounds]["multi_mask"],
            "total_output_monomials": sum(monomial_counts[rounds]),
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for rounds in config.rounds
    ]
    result_rows.append(
        {
            "run_id": config.run_id,
            "task": "innovation2_present_r5_open_3sdp_exact_oracle",
            "scope": "present_sbox_transition_parity",
            "rounds": 0,
            **transition["metrics"],
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
    )
    return {"gate": gate, "metrics": metrics, "result_rows": result_rows}


def polynomial_sha256(polynomial: Iterable[int]) -> str:
    digest = hashlib.sha256()
    width = (TOTAL_VARIABLES + 7) // 8
    for monomial in sorted(polynomial):
        digest.update(monomial.to_bytes(width, "little"))
    return digest.hexdigest()


def serializable_config(config: ExactOracleConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["rounds"] = list(config.rounds)
    return payload


def _update_symbolic_key(
    key: tuple[Polynomial, ...], round_counter: int
) -> tuple[Polynomial, ...]:
    rotated: list[Polynomial] = [frozenset() for _ in range(KEY_VARIABLES)]
    for old_bit, polynomial in enumerate(key):
        rotated[(old_bit + 61) % KEY_VARIABLES] = polynomial
    rotated[76:80] = evaluate_sbox_anf(tuple(rotated[76:80]))
    for bit in range(5):
        if round_counter & (1 << bit):
            rotated[15 + bit] = polynomial_xor(rotated[15 + bit], frozenset({0}))
    return tuple(rotated)


def _positive_scalar_checks(
    *,
    rounds: int,
    active_bits: tuple[int, ...],
    output_mask: int,
    seed: int,
    count: int = 4,
) -> dict[str, Any]:
    rng = random.Random(seed)
    passed = 0
    for _ in range(count):
        key = rng.getrandbits(80)
        offset = rng.getrandbits(64)
        passed += int(
            scalar_cube_parity(
                rounds=rounds,
                active_bits=active_bits,
                output_mask=output_mask,
                key=key,
                offset=offset,
            )
            == 0
        )
    return {"checked": count, "passed": passed, "all_pass": passed == count}


def _raw_sbox_trail_counts(selected_output_bits: Iterable[int]) -> dict[int, int]:
    counts = {0: 1}
    for output_bit in selected_output_bits:
        next_counts: dict[int, int] = {}
        for partial, partial_count in counts.items():
            for term in PRESENT_SBOX_ANF[output_bit]:
                monomial = partial | term
                next_counts[monomial] = (
                    next_counts.get(monomial, 0) + partial_count
                )
        counts = next_counts
    return counts


def _selected_sbox_output_polynomial(
    selected_output_bits: Iterable[int],
) -> Polynomial:
    result: Polynomial = frozenset({0})
    coordinate_polynomials = tuple(
        frozenset(terms) for terms in PRESENT_SBOX_ANF
    )
    for output_bit in selected_output_bits:
        result = polynomial_product(result, coordinate_polynomials[output_bit])
    return result


def _evaluate_sbox_coordinates(value: int) -> int:
    assignment = sum(
        ((value >> bit) & 1) << bit for bit in range(4)
    )
    output = 0
    for output_bit, terms in enumerate(PRESENT_SBOX_ANF):
        output |= evaluate_polynomial(terms, assignment) << output_bit
    return output


def _superpoly_has_key_variable(
    row: dict[str, Any], snapshots: dict[int, tuple[Polynomial, ...]]
) -> bool:
    superpoly = cube_superpoly(
        snapshots[int(row["rounds"])],
        tuple(int(bit) for bit in row["active_bits"]),
        int(str(row["output_mask_hex"]), 16),
    )
    key_mask = ((1 << KEY_VARIABLES) - 1) << PLAINTEXT_VARIABLES
    return any(monomial & key_mask for monomial in superpoly)


def _superpoly_has_inactive_plaintext_variable(
    row: dict[str, Any], snapshots: dict[int, tuple[Polynomial, ...]]
) -> bool:
    active_mask = sum(1 << int(bit) for bit in row["active_bits"])
    inactive_mask = ((1 << PLAINTEXT_VARIABLES) - 1) ^ active_mask
    superpoly = cube_superpoly(
        snapshots[int(row["rounds"])],
        tuple(int(bit) for bit in row["active_bits"]),
        int(str(row["output_mask_hex"]), 16),
    )
    return any(monomial & inactive_mask for monomial in superpoly)
