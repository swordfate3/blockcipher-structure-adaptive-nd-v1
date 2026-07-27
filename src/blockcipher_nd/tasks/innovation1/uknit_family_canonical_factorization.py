from __future__ import annotations

import hashlib
from itertools import permutations
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from unittest.mock import patch

import networkx as nx

from blockcipher_nd.ciphers.spn import dialga as dialga_module
from blockcipher_nd.ciphers.spn.dialga import (
    DIALGA_BIT_PERMUTATIONS,
    DIALGA_BYTE_PERMUTATIONS,
    DIALGA_BYTE_SBOXES,
    DIALGA_SBOX,
)
from blockcipher_nd.ciphers.spn.uknit import (
    UKNIT_LINEAR_TARGET_SOURCES,
    UKNIT_SBOX_TABLES,
    UknitBc,
    uknit_linear_layer,
    uknit_round_keys,
)


RUN_ID = "i1_uknit_family_canonical_component_factorization_k0_20260727"
OFFICIAL_UKNIT_REPOSITORY = "https://github.com/syllab-ntu/UKNIT.git"
OFFICIAL_UKNIT_COMMIT = "f6493014fb7326cf3fffa2bb642b26cd59650e4f"
OFFICIAL_UKNIT_SOURCE = (
    "uknit-implementations/uKNIT-BC_alternative_representation.cpp"
)
MANTIS_SBOX = DIALGA_SBOX

# Source images in the MSB-indexed convention used by the uKNIT Appendix C oracle.
MIDORI_LINEAR_SOURCE_IMAGES_MSB = (
    0x0888000000000000,
    0x0444000000000000,
    0x0222000000000000,
    0x0111000000000000,
    0x8088000000000000,
    0x4044000000000000,
    0x2022000000000000,
    0x1011000000000000,
    0x8808000000000000,
    0x4404000000000000,
    0x2202000000000000,
    0x1101000000000000,
    0x8880000000000000,
    0x4440000000000000,
    0x2220000000000000,
    0x1110000000000000,
    0x0000088800000000,
    0x0000044400000000,
    0x0000022200000000,
    0x0000011100000000,
    0x0000808800000000,
    0x0000404400000000,
    0x0000202200000000,
    0x0000101100000000,
    0x0000880800000000,
    0x0000440400000000,
    0x0000220200000000,
    0x0000110100000000,
    0x0000888000000000,
    0x0000444000000000,
    0x0000222000000000,
    0x0000111000000000,
    0x0000000008880000,
    0x0000000004440000,
    0x0000000002220000,
    0x0000000001110000,
    0x0000000080880000,
    0x0000000040440000,
    0x0000000020220000,
    0x0000000010110000,
    0x0000000088080000,
    0x0000000044040000,
    0x0000000022020000,
    0x0000000011010000,
    0x0000000088800000,
    0x0000000044400000,
    0x0000000022200000,
    0x0000000011100000,
    0x0000000000000888,
    0x0000000000000444,
    0x0000000000000222,
    0x0000000000000111,
    0x0000000000008088,
    0x0000000000004044,
    0x0000000000002022,
    0x0000000000001011,
    0x0000000000008808,
    0x0000000000004404,
    0x0000000000002202,
    0x0000000000001101,
    0x0000000000008880,
    0x0000000000004440,
    0x0000000000002220,
    0x0000000000001110,
)

BitPermutation = tuple[int, ...]
SboxFactor = tuple[BitPermutation, BitPermutation]
LinearFactor = tuple[BitPermutation, BitPermutation]
ProgressCallback = Callable[[str, dict[str, Any]], None]


def load_and_validate_factorization_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("K0 config schema_version must be 1")
    if payload.get("run_id") != RUN_ID:
        raise ValueError(f"K0 run_id must be {RUN_ID}")
    if payload.get("audit") != {
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
    }:
        raise ValueError("K0 must preserve the zero-training local audit contract")
    source = payload.get("uknit_source", {})
    expected_source = {
        "repository": OFFICIAL_UKNIT_REPOSITORY,
        "commit": OFFICIAL_UKNIT_COMMIT,
        "path": OFFICIAL_UKNIT_SOURCE,
    }
    if source != expected_source:
        raise ValueError("K0 uKNIT source pin does not match the frozen oracle")
    if len(payload.get("uknit", {}).get("full_vectors", ())) != 4:
        raise ValueError("K0 requires four uKNIT full vectors")
    if len(payload.get("uknit", {}).get("prefix_zero_states", ())) != 11:
        raise ValueError("K0 requires eleven uKNIT prefix states")
    if len(payload.get("dialga", {}).get("full_vectors", ())) != 4:
        raise ValueError("K0 requires four Dialga full vectors")
    if len(payload.get("dialga", {}).get("trace", {}).get("states", ())) != 16:
        raise ValueError("K0 requires the complete Dialga 16-round trace")
    return payload


def recover_sbox_factor(table: Sequence[int]) -> SboxFactor:
    if len(table) != 16 or sorted(table) != list(range(16)):
        raise ValueError("four-bit S-box table must be a permutation of 0..15")
    bit_permutations = tuple(permutations(range(4)))
    for input_permutation in bit_permutations:
        substituted = tuple(
            MANTIS_SBOX[_permute_bits(value, input_permutation)]
            for value in range(16)
        )
        for output_permutation in bit_permutations:
            if all(
                _permute_bits(substituted[value], output_permutation)
                == table[value]
                for value in range(16)
            ):
                return input_permutation, output_permutation
    raise ValueError("S-box is not bit-permutation equivalent to MANTIS")


def apply_sbox_factor(value: int, factor: SboxFactor) -> int:
    input_permutation, output_permutation = factor
    return _permute_bits(
        MANTIS_SBOX[_permute_bits(value, input_permutation)],
        output_permutation,
    )


def recover_uknit_sbox_factors() -> tuple[tuple[SboxFactor, ...], ...]:
    return tuple(
        tuple(recover_sbox_factor(table) for table in round_tables)
        for round_tables in UKNIT_SBOX_TABLES
    )


def midori_linear_layer(state: int) -> int:
    output = 0
    for source_lsb in range(64):
        if (state >> source_lsb) & 1:
            output ^= MIDORI_LINEAR_SOURCE_IMAGES_MSB[63 - source_lsb]
    return output


def recover_uknit_linear_factors() -> tuple[LinearFactor, ...]:
    canonical = _linear_bipartite_graph(_midori_target_sources())
    factors: list[LinearFactor] = []
    for native_target_sources in UKNIT_LINEAR_TARGET_SOURCES:
        native = _linear_bipartite_graph(native_target_sources)
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            canonical,
            native,
            node_match=lambda left, right: left["kind"] == right["kind"],
        )
        mapping = next(matcher.isomorphisms_iter(), None)
        if mapping is None:
            raise ValueError("uKNIT linear layer is not permutation-equivalent to MIDORI")
        canonical_input_to_native = tuple(
            mapping[("input", bit)][1] for bit in range(64)
        )
        canonical_output_to_native = tuple(
            mapping[("output", bit)][1] for bit in range(64)
        )
        factors.append((canonical_input_to_native, canonical_output_to_native))
    return tuple(factors)


def apply_uknit_linear_factor(state: int, factor: LinearFactor) -> int:
    canonical_input_to_native, canonical_output_to_native = factor
    canonical_state = 0
    for canonical_bit, native_bit in enumerate(canonical_input_to_native):
        canonical_state |= ((state >> native_bit) & 1) << canonical_bit
    canonical_output = midori_linear_layer(canonical_state)
    native_output = 0
    for canonical_bit, native_bit in enumerate(canonical_output_to_native):
        native_output |= ((canonical_output >> canonical_bit) & 1) << native_bit
    return native_output


def canonical_uknit_encrypt(
    plaintext: int,
    key: int,
    rounds: int,
    sbox_factors: Sequence[Sequence[SboxFactor]],
    linear_factors: Sequence[LinearFactor],
) -> int:
    round_keys = uknit_round_keys(key)
    state = plaintext
    for round_index in range(min(rounds, 11)):
        state ^= round_keys[round_index]
        state = _canonical_uknit_substitution(
            state, sbox_factors[round_index]
        )
        state = apply_uknit_linear_factor(state, linear_factors[round_index])
    if rounds == 12:
        state ^= round_keys[11]
        state = _canonical_uknit_substitution(state, sbox_factors[11])
        state ^= round_keys[12]
    return state


def canonical_dialga_sub_cells(state: int) -> int:
    values = state.to_bytes(16, byteorder="big")
    transformed = bytes(
        _canonical_dialga_byte_sbox(value, DIALGA_BIT_PERMUTATIONS[index % 4])
        for index, value in enumerate(values)
    )
    return int.from_bytes(transformed, byteorder="big")


def canonical_dialga_inverse_sub_cells(state: int) -> int:
    values = state.to_bytes(16, byteorder="big")
    transformed = bytes(
        _canonical_dialga_byte_sbox(
            value,
            DIALGA_BIT_PERMUTATIONS[index % 4],
            inverse=True,
        )
        for index, value in enumerate(values)
    )
    return int.from_bytes(transformed, byteorder="big")


def canonical_dialga_mix_columns(state: int) -> int:
    values = state.to_bytes(16, byteorder="big")
    mixed = bytearray(16)
    for column in range(4):
        offset = 4 * column
        s0, s1, s2, s3 = values[offset : offset + 4]
        mixed[offset] = s1 ^ s2 ^ s3
        mixed[offset + 1] = s0 ^ s2 ^ s3
        mixed[offset + 2] = s0 ^ s1 ^ s3
        mixed[offset + 3] = s0 ^ s1 ^ s2
    return int.from_bytes(mixed, byteorder="big")


def canonical_dialga_linear_layer(state: int, round_type: int) -> int:
    state = _permute_dialga_bytes(state, DIALGA_BYTE_PERMUTATIONS[round_type])
    return canonical_dialga_mix_columns(state)


def canonical_dialga_inverse_linear_layer(state: int, round_type: int) -> int:
    state = canonical_dialga_mix_columns(state)
    return _permute_dialga_bytes(
        state, _inverse_target_source_permutation(DIALGA_BYTE_PERMUTATIONS[round_type])
    )


def canonical_dialga_round_function(state: int, round_type: int) -> int:
    return canonical_dialga_linear_layer(canonical_dialga_sub_cells(state), round_type)


def canonical_dialga_inverse_round_function(state: int, round_type: int) -> int:
    return canonical_dialga_inverse_sub_cells(
        canonical_dialga_inverse_linear_layer(state, round_type)
    )


def run_canonical_factorization_audit(
    config: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _emit(progress_callback, "uknit_factor_recovery_start")
    sbox_factors = recover_uknit_sbox_factors()
    linear_factors = recover_uknit_linear_factors()
    repeated_sbox_factors = recover_uknit_sbox_factors()
    repeated_linear_factors = recover_uknit_linear_factors()
    uknit_row = _audit_uknit(
        config["uknit"],
        sbox_factors=sbox_factors,
        linear_factors=linear_factors,
    )
    manifest = _factor_manifest_sha256(sbox_factors, linear_factors)
    repeated_manifest = _factor_manifest_sha256(
        repeated_sbox_factors, repeated_linear_factors
    )
    uknit_row["factor_manifest_sha256"] = manifest
    uknit_row["repeated_factor_manifest_sha256"] = repeated_manifest
    uknit_row["factor_recovery_deterministic"] = manifest == repeated_manifest
    _emit(progress_callback, "uknit_factor_recovery_done", row=uknit_row)

    _emit(progress_callback, "dialga_factor_audit_start")
    dialga_row = _audit_dialga(config["dialga"])
    _emit(progress_callback, "dialga_factor_audit_done", row=dialga_row)

    results = [uknit_row, dialga_row]
    checks = {
        "frozen_source_pin": config["uknit_source"]
        == {
            "repository": OFFICIAL_UKNIT_REPOSITORY,
            "commit": OFFICIAL_UKNIT_COMMIT,
            "path": OFFICIAL_UKNIT_SOURCE,
        },
        "zero_training_local_contract": config["audit"]
        == {"training_rows": 0, "optimizer_steps": 0, "remote": False},
        "uknit_all_192_sboxes_factorized": uknit_row["sbox_factors"] == 192,
        "uknit_all_3072_sbox_probes_match": uknit_row["sbox_probe_matches"]
        == uknit_row["sbox_probes"]
        == 3072,
        "uknit_all_11_linear_layers_factorized": uknit_row["linear_factors"] == 11,
        "uknit_all_704_linear_probes_match": uknit_row["linear_probe_matches"]
        == uknit_row["linear_probes"]
        == 704,
        "uknit_all_full_vectors_match": uknit_row["full_vector_matches"]
        == uknit_row["full_vectors"]
        == 4,
        "uknit_all_prefix_states_match": uknit_row["prefix_state_matches"]
        == uknit_row["prefix_states"]
        == 11,
        "uknit_factor_recovery_deterministic": uknit_row[
            "factor_recovery_deterministic"
        ],
        "uknit_wrong_controls_distinct": uknit_row["wrong_bit_factor_distinct"]
        and uknit_row["wrong_round_order_distinct"],
        "dialga_all_byte_sbox_probes_match": dialga_row["sbox_probe_matches"]
        == dialga_row["sbox_probes"]
        == 1024,
        "dialga_all_linear_probes_match": dialga_row["linear_probe_matches"]
        == dialga_row["linear_probes"]
        == 512,
        "dialga_all_full_vectors_match": dialga_row["full_vector_matches"]
        == dialga_row["full_vectors"]
        == 4,
        "dialga_full_trace_matches": dialga_row["trace_state_matches"]
        == dialga_row["trace_states"]
        == 16,
        "dialga_wrong_controls_distinct": dialga_row["wrong_bit_factor_distinct"]
        and dialga_row["wrong_byte_factor_distinct"]
        and dialga_row["wrong_round_order_distinct"],
    }
    errors = [name for name, passed in checks.items() if passed is not True]
    status = "pass" if not errors else "invalid"
    decision = (
        "innovation1_uknit_family_canonical_component_factorization_supported"
        if status == "pass"
        else "innovation1_uknit_family_canonical_component_factorization_invalid"
    )
    next_action = (
        "prepare K1 same-protocol local neural diagnostic: Runtime-E4 anchor versus "
        "CT-SPN canonical exact-state-view fusion, with frozen ordered/repeated/"
        "shuffled/corrupted/no-topology evaluations"
        if status == "pass"
        else "repair only failed K0 factorization, indexing or boundary semantics and rerun"
    )
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if not errors else "fail",
        "checks": checks,
        "errors": errors,
        "result_rows": len(results),
    }
    gate = {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "protocol_checks": checks,
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
        "next_action": next_action,
        "claim_scope": (
            "exact zero-training component-factorization evidence for uKNIT-BC and "
            "Dialga-128 only; not neural learnability, differential signal, transfer, "
            "attack, SOTA, formal same-family taxonomy or arbitrary-SPN evidence"
        ),
        "blocked_actions": [
            "remote uKNIT scale-up from K0 alone",
            "K2 nonlinear conditioning before K1 passes",
            "learned MoE or cipher-id routing",
            "including generalized-Feistel MSX in the CT-SPN family",
        ],
    }
    summary = {
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote": False,
        "result_rows": len(results),
        "metrics": {row["cipher"]: row for row in results},
        "next_action": next_action,
        "claim_scope": gate["claim_scope"],
    }
    _emit(progress_callback, "audit_adjudicated", status=status, decision=decision)
    return {
        "results": results,
        "validation": validation,
        "gate": gate,
        "summary": summary,
    }


def write_factorization_artifacts(
    payload: Mapping[str, Any], output_root: Path
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["results"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    for name in ("validation", "gate", "summary"):
        (output_root / f"{name}.json").write_text(
            json.dumps(payload[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _audit_uknit(
    config: Mapping[str, Any],
    *,
    sbox_factors: Sequence[Sequence[SboxFactor]],
    linear_factors: Sequence[LinearFactor],
) -> dict[str, Any]:
    sbox_matches = sum(
        apply_sbox_factor(value, factor) == table[value]
        for round_tables, round_factors in zip(
            UKNIT_SBOX_TABLES, sbox_factors, strict=True
        )
        for table, factor in zip(round_tables, round_factors, strict=True)
        for value in range(16)
    )
    linear_matches = sum(
        apply_uknit_linear_factor(1 << source, factor)
        == uknit_linear_layer(1 << source, round_index)
        for round_index, factor in enumerate(linear_factors)
        for source in range(64)
    )
    full_matches = 0
    for item in config["full_vectors"]:
        plaintext = _parse_hex(item["plaintext"])
        key = _parse_hex(item["key"])
        expected = _parse_hex(item["ciphertext"])
        canonical = canonical_uknit_encrypt(
            plaintext, key, 12, sbox_factors, linear_factors
        )
        native = UknitBc(rounds=12, key=key).encrypt(plaintext)
        full_matches += canonical == native == expected
    prefix_matches = sum(
        canonical_uknit_encrypt(0, 0, rounds, sbox_factors, linear_factors)
        == UknitBc(rounds=rounds, key=0).encrypt(0)
        == _parse_hex(expected)
        for rounds, expected in enumerate(config["prefix_zero_states"], start=1)
    )
    true_linear_fingerprint = _operator_panel_sha256(
        tuple(
            tuple(apply_uknit_linear_factor(1 << bit, factor) for bit in range(64))
            for factor in linear_factors
        )
    )
    wrong_factor = _swap_first_two(linear_factors[0][0]), linear_factors[0][1]
    wrong_linear_fingerprint = _operator_panel_sha256(
        (
            tuple(
                apply_uknit_linear_factor(1 << bit, wrong_factor)
                for bit in range(64)
            ),
            *tuple(
                tuple(
                    apply_uknit_linear_factor(1 << bit, factor)
                    for bit in range(64)
                )
                for factor in linear_factors[1:]
            ),
        )
    )
    shuffled_linear_fingerprint = _operator_panel_sha256(
        tuple(
            tuple(apply_uknit_linear_factor(1 << bit, factor) for bit in range(64))
            for factor in (*linear_factors[1:], linear_factors[0])
        )
    )
    return {
        "run_id": RUN_ID,
        "cipher": "uKNIT-BC",
        "status": "exact_audit",
        "sbox_factors": sum(len(round_factors) for round_factors in sbox_factors),
        "sbox_probes": 12 * 16 * 16,
        "sbox_probe_matches": sbox_matches,
        "linear_factors": len(linear_factors),
        "linear_probes": 11 * 64,
        "linear_probe_matches": linear_matches,
        "full_vectors": len(config["full_vectors"]),
        "full_vector_matches": full_matches,
        "prefix_states": len(config["prefix_zero_states"]),
        "prefix_state_matches": prefix_matches,
        "true_linear_fingerprint": true_linear_fingerprint,
        "wrong_linear_fingerprint": wrong_linear_fingerprint,
        "shuffled_linear_fingerprint": shuffled_linear_fingerprint,
        "wrong_bit_factor_distinct": wrong_linear_fingerprint
        != true_linear_fingerprint,
        "wrong_round_order_distinct": shuffled_linear_fingerprint
        != true_linear_fingerprint,
        "training_rows": 0,
        "optimizer_steps": 0,
    }


def _audit_dialga(config: Mapping[str, Any]) -> dict[str, Any]:
    sbox_matches = sum(
        _canonical_dialga_byte_sbox(value, permutation)
        == DIALGA_BYTE_SBOXES[index][value]
        for index, permutation in enumerate(DIALGA_BIT_PERMUTATIONS)
        for value in range(256)
    )
    linear_matches = sum(
        canonical_dialga_linear_layer(1 << source, round_type)
        == dialga_module.dialga_linear_layer(1 << source, round_type)
        for round_type in range(4)
        for source in range(128)
    )
    full_matches = 0
    with _canonical_dialga_operators():
        for item in config["full_vectors"]:
            plaintext = _parse_hex(item["plaintext"])
            key = _parse_hex(item["key"])
            tweak = _parse_hex(item["tweak"])
            total_rounds = int(item["total_rounds"])
            expected = _parse_hex(item["ciphertext"])
            canonical = dialga_module.dialga128_encrypt(
                plaintext, key, tweak, total_rounds=total_rounds
            )
            full_matches += canonical == expected
        trace_config = config["trace"]
        canonical_trace = dialga_module.dialga128_round_trace(
            _parse_hex(trace_config["plaintext"]),
            _parse_hex(trace_config["key"]),
            _parse_hex(trace_config["tweak"]),
            total_rounds=16,
        )
    native_trace = dialga_module.dialga128_round_trace(
        _parse_hex(config["trace"]["plaintext"]),
        _parse_hex(config["trace"]["key"]),
        _parse_hex(config["trace"]["tweak"]),
        total_rounds=16,
    )
    expected_trace = tuple(_parse_hex(value) for value in config["trace"]["states"])
    trace_matches = sum(
        canonical == native == expected
        for canonical, native, expected in zip(
            canonical_trace, native_trace, expected_trace, strict=True
        )
    )
    true_sbox_fingerprint = _operator_panel_sha256(
        tuple(tuple(table) for table in DIALGA_BYTE_SBOXES)
    )
    wrong_bit_permutation = _swap_first_two(DIALGA_BIT_PERMUTATIONS[0])
    wrong_sbox_fingerprint = _operator_panel_sha256(
        (
            tuple(
                _canonical_dialga_byte_sbox(value, wrong_bit_permutation)
                for value in range(256)
            ),
            *tuple(tuple(table) for table in DIALGA_BYTE_SBOXES[1:]),
        )
    )
    true_linear_panel = tuple(
        tuple(canonical_dialga_linear_layer(1 << bit, round_type) for bit in range(128))
        for round_type in range(4)
    )
    wrong_byte_permutation = _swap_first_two(DIALGA_BYTE_PERMUTATIONS[0])
    wrong_byte_panel = (
        tuple(
            canonical_dialga_mix_columns(
                _permute_dialga_bytes(1 << bit, wrong_byte_permutation)
            )
            for bit in range(128)
        ),
        *true_linear_panel[1:],
    )
    true_linear_fingerprint = _operator_panel_sha256(true_linear_panel)
    wrong_byte_fingerprint = _operator_panel_sha256(wrong_byte_panel)
    shuffled_linear_fingerprint = _operator_panel_sha256(
        (*true_linear_panel[1:], true_linear_panel[0])
    )
    return {
        "run_id": RUN_ID,
        "cipher": "Dialga-128",
        "status": "exact_audit",
        "sbox_factors": 4,
        "sbox_probes": 4 * 256,
        "sbox_probe_matches": sbox_matches,
        "linear_factors": 4,
        "linear_probes": 4 * 128,
        "linear_probe_matches": linear_matches,
        "full_vectors": len(config["full_vectors"]),
        "full_vector_matches": full_matches,
        "trace_states": len(expected_trace),
        "trace_state_matches": trace_matches,
        "true_sbox_fingerprint": true_sbox_fingerprint,
        "wrong_sbox_fingerprint": wrong_sbox_fingerprint,
        "true_linear_fingerprint": true_linear_fingerprint,
        "wrong_byte_fingerprint": wrong_byte_fingerprint,
        "shuffled_linear_fingerprint": shuffled_linear_fingerprint,
        "wrong_bit_factor_distinct": wrong_sbox_fingerprint
        != true_sbox_fingerprint,
        "wrong_byte_factor_distinct": wrong_byte_fingerprint
        != true_linear_fingerprint,
        "wrong_round_order_distinct": shuffled_linear_fingerprint
        != true_linear_fingerprint,
        "training_rows": 0,
        "optimizer_steps": 0,
    }


def _canonical_uknit_substitution(
    state: int, round_factors: Sequence[SboxFactor]
) -> int:
    cells = tuple((state >> (4 * (15 - index))) & 0xF for index in range(16))
    output = 0
    for index, value in enumerate(cells):
        output = (output << 4) | apply_sbox_factor(
            value, round_factors[15 - index]
        )
    return output


def _midori_target_sources() -> tuple[tuple[int, ...], ...]:
    rows: list[list[int]] = [[] for _ in range(64)]
    for source_lsb in range(64):
        image = MIDORI_LINEAR_SOURCE_IMAGES_MSB[63 - source_lsb]
        for target_lsb in range(64):
            if (image >> target_lsb) & 1:
                rows[target_lsb].append(source_lsb)
    return tuple(tuple(row) for row in rows)


def _linear_bipartite_graph(
    target_sources: Sequence[Sequence[int]],
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from((("input", bit) for bit in range(64)), kind="input")
    graph.add_nodes_from((("output", bit) for bit in range(64)), kind="output")
    for target, sources in enumerate(target_sources):
        graph.add_edges_from(
            (("input", source), ("output", target)) for source in sources
        )
    return graph


def _canonical_dialga_byte_sbox(
    value: int, permutation: Sequence[int], *, inverse: bool = False
) -> int:
    permuted = _permute_byte_target_sources(value, permutation)
    sbox = _inverse_value_permutation(MANTIS_SBOX) if inverse else MANTIS_SBOX
    substituted = (sbox[permuted >> 4] << 4) | sbox[permuted & 0xF]
    return _permute_byte_target_sources(
        substituted, _inverse_target_source_permutation(permutation)
    )


def _canonical_dialga_operators() -> Any:
    return _PatchStack(
        patch.object(dialga_module, "dialga_sub_cells", canonical_dialga_sub_cells),
        patch.object(
            dialga_module,
            "dialga_inverse_sub_cells",
            canonical_dialga_inverse_sub_cells,
        ),
        patch.object(
            dialga_module, "dialga_round_function", canonical_dialga_round_function
        ),
        patch.object(
            dialga_module,
            "dialga_inverse_round_function",
            canonical_dialga_inverse_round_function,
        ),
    )


class _PatchStack:
    def __init__(self, *patchers: Any) -> None:
        self.patchers = patchers

    def __enter__(self) -> None:
        for patcher in self.patchers:
            patcher.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()


def _permute_bits(value: int, source_to_target: Sequence[int]) -> int:
    output = 0
    for source, target in enumerate(source_to_target):
        output |= ((value >> source) & 1) << target
    return output


def _permute_byte_target_sources(value: int, target_sources: Sequence[int]) -> int:
    output = 0
    for target, source in enumerate(target_sources):
        output |= ((value >> (7 - source)) & 1) << (7 - target)
    return output


def _permute_dialga_bytes(state: int, target_sources: Sequence[int]) -> int:
    values = state.to_bytes(16, byteorder="big")
    return int.from_bytes(
        bytes(values[source] for source in target_sources), byteorder="big"
    )


def _inverse_target_source_permutation(
    target_sources: Sequence[int],
) -> tuple[int, ...]:
    inverse = [0] * len(target_sources)
    for target, source in enumerate(target_sources):
        inverse[source] = target
    return tuple(inverse)


def _inverse_value_permutation(values: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(values)
    for source, target in enumerate(values):
        inverse[target] = source
    return tuple(inverse)


def _swap_first_two(values: Sequence[int]) -> tuple[int, ...]:
    result = list(values)
    result[0], result[1] = result[1], result[0]
    return tuple(result)


def _factor_manifest_sha256(
    sbox_factors: Sequence[Sequence[SboxFactor]],
    linear_factors: Sequence[LinearFactor],
) -> str:
    return _sha256_json(
        {"sbox_factors": sbox_factors, "linear_factors": linear_factors}
    )


def _operator_panel_sha256(panel: Sequence[Sequence[int]]) -> str:
    return _sha256_json(panel)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_hex(value: str) -> int:
    return int(value, 16)


def _emit(
    callback: ProgressCallback | None, event: str, **payload: Any
) -> None:
    if callback is not None:
        callback(event, payload)


__all__ = [
    "RUN_ID",
    "apply_sbox_factor",
    "apply_uknit_linear_factor",
    "canonical_dialga_linear_layer",
    "canonical_dialga_sub_cells",
    "canonical_uknit_encrypt",
    "load_and_validate_factorization_config",
    "midori_linear_layer",
    "recover_sbox_factor",
    "recover_uknit_linear_factors",
    "recover_uknit_sbox_factors",
    "run_canonical_factorization_audit",
    "write_factorization_artifacts",
]
