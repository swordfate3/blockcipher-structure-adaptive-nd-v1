from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from blockcipher_nd.tasks.innovation2.atm_native_sat_witness_provider import (
    build_present_independent_key_model,
    count_models_parity,
    exponent_assumptions,
    key_mask_from_projected_literals,
    replay_key_monomial_parity,
)


def multicoordinate_support_pool() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "input_exponent": input_exponent,
            "input_exponent_hex": f"0x{input_exponent:016X}",
            "input_weight": input_exponent.bit_count(),
            "output_exponent": output_exponent,
            "output_exponent_hex": f"0x{output_exponent:016X}",
            "output_weight": output_exponent.bit_count(),
            "input_cell": 0,
            "output_cell": 0,
        }
        for input_exponent in range(16)
        for output_exponent in range(1, 16)
    )


def enumerate_key_polynomial_support(
    model: Sequence[Sequence[int]],
    input_vars: Sequence[int],
    output_vars: Sequence[int],
    key_vars: Sequence[int],
    *,
    input_exponent: int,
    output_exponent: int,
    projected_key_cap: int,
    trail_model_cap: int,
    solver_factory: Callable[..., Any],
    projected_model_enumerator: Callable[..., Iterable[Sequence[int]]],
    replay_all_odd_masks: bool = True,
) -> dict[str, Any]:
    assumptions = exponent_assumptions(
        input_vars, output_vars, input_exponent, output_exponent
    )
    projected_seen = 0
    odd_masks: list[int] = []
    total_trail_models = 0
    for projected in projected_model_enumerator(
        model, tuple(key_vars), assumptions=assumptions
    ):
        projected_seen += 1
        if projected_seen > projected_key_cap:
            return {
                "status": "unknown",
                "reason": "projected_key_cap_exceeded",
                "projected_keys_seen": projected_seen,
                "odd_key_exponents": [],
                "replay_verified": False,
            }
        key_literals = tuple(int(value) for value in projected)
        parity = count_models_parity(
            model,
            assumptions + key_literals,
            model_cap=trail_model_cap,
            solver_factory=solver_factory,
        )
        if parity["status"] != "exact":
            return {
                "status": "unknown",
                "reason": parity["reason"],
                "projected_keys_seen": projected_seen,
                "odd_key_exponents": [],
                "replay_verified": False,
            }
        total_trail_models += int(parity["models_seen"])
        if parity["parity"] == 1:
            odd_masks.append(key_mask_from_projected_literals(key_literals, key_vars))

    odd_masks = sorted(set(odd_masks))
    replay_rows: list[dict[str, Any]] = []
    if replay_all_odd_masks:
        for key_mask in odd_masks:
            replay = replay_key_monomial_parity(
                model,
                input_vars,
                output_vars,
                key_vars,
                input_exponent=input_exponent,
                output_exponent=output_exponent,
                key_exponent_mask=key_mask,
                trail_model_cap=trail_model_cap,
                solver_factory=solver_factory,
            )
            replay_rows.append(
                {
                    "key_exponent_hex": f"0x{key_mask:X}",
                    "status": replay["status"],
                    "parity": replay.get("parity"),
                    "models_seen": replay.get("models_seen"),
                }
            )
        replay_verified = all(
            row["status"] == "exact" and row["parity"] == 1
            for row in replay_rows
        )
    else:
        replay_verified = True
    nonzero_masks = [mask for mask in odd_masks if mask != 0]
    return {
        "status": "exact",
        "reason": None,
        "projected_keys_seen": projected_seen,
        "total_trail_models": total_trail_models,
        "constant_parity": int(0 in odd_masks),
        "odd_key_exponents": [f"0x{mask:X}" for mask in odd_masks],
        "nonzero_odd_key_exponents": [f"0x{mask:X}" for mask in nonzero_masks],
        "nonzero_support_size": len(nonzero_masks),
        "key_dependent": bool(nonzero_masks),
        "replay_verified": replay_verified,
        "replays": replay_rows,
    }


def find_low_weight_cancellation_relations(
    rows: Sequence[dict[str, Any]], *, limit: int = 32
) -> list[dict[str, Any]]:
    exact_rows = [
        row
        for row in rows
        if row.get("support", {}).get("status") == "exact"
        and row.get("support", {}).get("key_dependent") is True
    ]
    mask_universe = sorted(
        {
            int(mask, 16)
            for row in exact_rows
            for mask in row["support"]["nonzero_odd_key_exponents"]
        }
    )
    mask_index = {mask: index for index, mask in enumerate(mask_universe)}
    vectors = {
        int(row["query_index"]): sum(
            1 << mask_index[int(mask, 16)]
            for mask in row["support"]["nonzero_odd_key_exponents"]
        )
        for row in exact_rows
    }
    indices = sorted(vectors)
    vector_to_indices: dict[int, list[int]] = {}
    for index in indices:
        vector_to_indices.setdefault(vectors[index], []).append(index)

    relation_sets: set[tuple[int, ...]] = set()
    for group in vector_to_indices.values():
        if len(group) > 1:
            for left_index, left in enumerate(group):
                for right in group[left_index + 1 :]:
                    relation_sets.add((left, right))

    pair_groups: dict[int, list[tuple[int, int]]] = {}
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1 :]:
            pair_xor = vectors[left] ^ vectors[right]
            if pair_xor == 0:
                continue
            for third in vector_to_indices.get(pair_xor, []):
                if third not in {left, right}:
                    relation_sets.add(tuple(sorted((left, right, third))))
            pair_groups.setdefault(pair_xor, []).append((left, right))

    for pairs in pair_groups.values():
        for left_position, left_pair in enumerate(pairs):
            for right_pair in pairs[left_position + 1 :]:
                relation = tuple(sorted(set(left_pair + right_pair)))
                if len(relation) == 4:
                    relation_sets.add(relation)

    result: list[dict[str, Any]] = []
    for relation in sorted(relation_sets, key=lambda value: (len(value), value)):
        if len(result) >= limit:
            break
        xor_vector = 0
        for index in relation:
            xor_vector ^= vectors[index]
        if xor_vector != 0:
            continue
        result.append(
            {
                "coordinate_indices": list(relation),
                "relation_size": len(relation),
                "nonzero_support_xor_empty": True,
            }
        )
    return result


def pair_cancellation_relations_with_matched_negatives(
    rows: Sequence[dict[str, Any]],
    positive_relations: Sequence[dict[str, Any]],
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    row_by_index = {int(row["query_index"]): row for row in rows}
    exact_indices = sorted(
        index
        for index, row in row_by_index.items()
        if row.get("support", {}).get("status") == "exact"
        and row.get("support", {}).get("key_dependent") is True
    )
    supports = {
        index: {
            int(mask, 16)
            for mask in row_by_index[index]["support"][
                "nonzero_odd_key_exponents"
            ]
        }
        for index in exact_indices
    }
    pairs: list[dict[str, Any]] = []
    seen_negative: set[tuple[int, ...]] = set()
    for positive in positive_relations:
        positive_indices = tuple(int(value) for value in positive["coordinate_indices"])
        for removed in positive_indices:
            removed_query = row_by_index[removed]["query"]
            replacements = [
                candidate
                for candidate in exact_indices
                if candidate not in positive_indices
                and row_by_index[candidate]["query"]["input_weight"]
                == removed_query["input_weight"]
                and row_by_index[candidate]["query"]["output_weight"]
                == removed_query["output_weight"]
            ]
            matched = None
            for replacement in replacements:
                negative = tuple(
                    sorted(
                        replacement if index == removed else index
                        for index in positive_indices
                    )
                )
                if negative in seen_negative:
                    continue
                xor_support: set[int] = set()
                for index in negative:
                    xor_support.symmetric_difference_update(supports[index])
                if not xor_support:
                    continue
                matched = {
                    "positive_coordinate_indices": list(positive_indices),
                    "negative_coordinate_indices": list(negative),
                    "relation_size": len(positive_indices),
                    "removed_coordinate_index": removed,
                    "replacement_coordinate_index": replacement,
                    "matched_input_weight": removed_query["input_weight"],
                    "matched_output_weight": removed_query["output_weight"],
                    "negative_witness_key_exponent_hex": f"0x{min(xor_support):X}",
                }
                seen_negative.add(negative)
                break
            if matched is not None:
                pairs.append(matched)
                break
        if len(pairs) >= limit:
            break
    return pairs


def replay_relation(
    bundle: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    coordinate_indices: Sequence[int],
    *,
    key_exponent_mask: int,
    trail_model_cap: int,
) -> dict[str, Any]:
    row_by_index = {int(row["query_index"]): row for row in rows}
    coordinate_replays: list[dict[str, Any]] = []
    xor_parity = 0
    for coordinate_index in coordinate_indices:
        query = row_by_index[int(coordinate_index)]["query"]
        replay = replay_key_monomial_parity(
            bundle["model"],
            bundle["input_vars"],
            bundle["output_vars"],
            bundle["key_vars"],
            input_exponent=int(query["input_exponent"]),
            output_exponent=int(query["output_exponent"]),
            key_exponent_mask=key_exponent_mask,
            trail_model_cap=trail_model_cap,
            solver_factory=bundle["solver_factory"],
        )
        parity = replay.get("parity")
        if replay["status"] == "exact" and parity in {0, 1}:
            xor_parity ^= int(parity)
        coordinate_replays.append(
            {
                "query_index": int(coordinate_index),
                "status": replay["status"],
                "parity": parity,
                "models_seen": replay.get("models_seen"),
            }
        )
    exact = all(row["status"] == "exact" for row in coordinate_replays)
    return {
        "status": "exact" if exact else "unknown",
        "key_exponent_hex": f"0x{key_exponent_mask:X}",
        "xor_parity": xor_parity if exact else None,
        "coordinate_replays": coordinate_replays,
    }


def run_multicoordinate_support_phase_a(
    atm_root: Path,
    *,
    projected_key_cap: int,
    trail_model_cap: int,
    model_output: Path,
    supports_output: Path,
    relations_output: Path,
    progress_output: Path,
) -> dict[str, Any]:
    bundle = build_present_independent_key_model(atm_root, rounds=2)
    _write_json(model_output, bundle["metadata"])
    supports_output.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(multicoordinate_support_pool()):
        started = time.monotonic()
        support = enumerate_key_polynomial_support(
            bundle["model"],
            bundle["input_vars"],
            bundle["output_vars"],
            bundle["key_vars"],
            input_exponent=int(query["input_exponent"]),
            output_exponent=int(query["output_exponent"]),
            projected_key_cap=projected_key_cap,
            trail_model_cap=trail_model_cap,
            solver_factory=bundle["solver_factory"],
            projected_model_enumerator=bundle["projected_model_enumerator"],
        )
        row = {
            "query_index": query_index,
            "query": query,
            "support": support,
            "elapsed_seconds": time.monotonic() - started,
        }
        rows.append(row)
        _append_jsonl(supports_output, row)
        _append_jsonl(
            progress_output,
            {
                "event": "coordinate_done",
                "query_index": query_index,
                "status": support["status"],
                "key_dependent": support.get("key_dependent"),
            },
        )

    positives = find_low_weight_cancellation_relations(rows)
    pairs = pair_cancellation_relations_with_matched_negatives(rows, positives)
    replayed_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        positive_replay = replay_relation(
            bundle,
            rows,
            pair["positive_coordinate_indices"],
            key_exponent_mask=0,
            trail_model_cap=trail_model_cap,
        )
        witness_mask = int(pair["negative_witness_key_exponent_hex"], 16)
        negative_replay = replay_relation(
            bundle,
            rows,
            pair["negative_coordinate_indices"],
            key_exponent_mask=witness_mask,
            trail_model_cap=trail_model_cap,
        )
        replayed_pairs.append(
            {
                **pair,
                "positive_constant_replay": positive_replay,
                "negative_witness_replay": negative_replay,
            }
        )
    relation_payload = {
        "positive_relations": positives,
        "matched_relation_pairs": replayed_pairs,
    }
    _write_json(relations_output, relation_payload)
    return {"model": bundle["metadata"], "rows": rows, **relation_payload}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
