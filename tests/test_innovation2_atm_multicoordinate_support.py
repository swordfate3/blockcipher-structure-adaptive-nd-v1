from __future__ import annotations

from blockcipher_nd.tasks.innovation2.atm_multicoordinate_support import (
    enumerate_key_polynomial_support,
    find_low_weight_cancellation_relations,
    multicoordinate_support_pool,
    pair_cancellation_relations_with_matched_negatives,
)
from blockcipher_nd.cli.audit_innovation2_atm_multicoordinate_support_phase_a import (
    evaluate_phase_a,
)
from blockcipher_nd.cli.plot_innovation2_atm_multicoordinate_support_phase_a import (
    render_atm_multicoordinate_support_phase_a,
)
from blockcipher_nd.tasks.innovation2.atm_native_sat_witness_provider import (
    NativeSatProviderConfig,
)


class _OddSolver:
    def __init__(self, *, bootstrap_with: object) -> None:
        pass

    def __enter__(self) -> "_OddSolver":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def enum_models(self, *, assumptions: object) -> object:
        return iter([()])


def _projected_masks(
    model: object, key_vars: tuple[int, ...], *, assumptions: object
) -> object:
    del model, key_vars, assumptions
    return iter(((-3, -4), (3, -4)))


def test_multicoordinate_support_pool_is_frozen_cell_product() -> None:
    queries = multicoordinate_support_pool()

    assert len(queries) == 240
    assert queries[0]["input_exponent"] == 0
    assert queries[0]["output_exponent"] == 1
    assert queries[-1]["input_exponent"] == 15
    assert queries[-1]["output_exponent"] == 15
    assert {(row["input_cell"], row["output_cell"]) for row in queries} == {(0, 0)}


def test_support_enumerator_keeps_zero_and_nonzero_odd_masks() -> None:
    result = enumerate_key_polynomial_support(
        (),
        (1,),
        (2,),
        (3, 4),
        input_exponent=0,
        output_exponent=1,
        projected_key_cap=4,
        trail_model_cap=4,
        solver_factory=_OddSolver,
        projected_model_enumerator=_projected_masks,
    )

    assert result["status"] == "exact"
    assert result["constant_parity"] == 1
    assert result["odd_key_exponents"] == ["0x0", "0x1"]
    assert result["nonzero_odd_key_exponents"] == ["0x1"]
    assert result["key_dependent"] is True
    assert result["replay_verified"] is True


def test_support_enumerator_preserves_projected_cap_as_unknown() -> None:
    result = enumerate_key_polynomial_support(
        (),
        (1,),
        (2,),
        (3, 4),
        input_exponent=0,
        output_exponent=1,
        projected_key_cap=1,
        trail_model_cap=4,
        solver_factory=_OddSolver,
        projected_model_enumerator=_projected_masks,
    )

    assert result["status"] == "unknown"
    assert result["reason"] == "projected_key_cap_exceeded"
    assert result["replay_verified"] is False


def _support_row(index: int, masks: list[int]) -> dict[str, object]:
    return {
        "query_index": index,
        "query": {
            "input_weight": 1,
            "output_weight": 1,
            "input_exponent": index,
            "output_exponent": 1,
        },
        "support": {
            "status": "exact",
            "key_dependent": True,
            "nonzero_odd_key_exponents": [f"0x{mask:X}" for mask in masks],
        },
    }


def test_relation_search_finds_cancellation_and_matched_negative() -> None:
    rows = [
        _support_row(0, [1]),
        _support_row(1, [2]),
        _support_row(2, [1, 2]),
        _support_row(3, [4]),
    ]

    positives = find_low_weight_cancellation_relations(rows)
    pairs = pair_cancellation_relations_with_matched_negatives(rows, positives)

    assert positives[0] == {
        "coordinate_indices": [0, 1, 2],
        "relation_size": 3,
        "nonzero_support_xor_empty": True,
    }
    assert pairs
    assert pairs[0]["relation_size"] == 3
    assert len(pairs[0]["negative_coordinate_indices"]) == 3
    assert pairs[0]["matched_input_weight"] == 1
    assert pairs[0]["matched_output_weight"] == 1
    assert int(pairs[0]["negative_witness_key_exponent_hex"], 16) != 0


def test_phase_a_gate_opens_only_after_support_and_relation_width() -> None:
    rows = [
        {
            "query_index": index,
            "support": {
                "status": "exact",
                "key_dependent": True,
                "replay_verified": True,
            },
        }
        for index in range(240)
    ]
    pair = {
        "positive_constant_replay": {"status": "exact", "xor_parity": 0},
        "negative_witness_replay": {"status": "exact", "xor_parity": 1},
    }
    result = evaluate_phase_a(
        NativeSatProviderConfig(run_id="e61_gate_test"),
        e60_gate={
            "status": "hold",
            "decision": "innovation2_atm_r2_cone_matched_panel_width_not_ready",
            "metrics": {"scalar_validated_constant_rows": 16},
            "next_action": {"multi_coordinate_design": True},
        },
        model={
            "rounds": 2,
            "key_additions": 3,
            "key_variables": 192,
            "key_model": "independent_round_keys",
        },
        rows=rows,
        relation_payload={
            "positive_relations": [{} for _ in range(4)],
            "matched_relation_pairs": [pair for _ in range(4)],
        },
        planned_coordinates=240,
        worker_status="completed",
        wall_clock_cap_seconds=60,
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["decision"] == (
        "innovation2_atm_r2_multicoordinate_support_phase_b_ready"
    )
    assert result["gate"]["next_action"]["phase_b_label_atlas"] is True
    assert result["gate"]["next_action"]["training"] is False


def test_phase_a_timeout_keeps_route_on_hold() -> None:
    result = evaluate_phase_a(
        NativeSatProviderConfig(run_id="e61_timeout_test"),
        e60_gate={
            "status": "hold",
            "decision": "innovation2_atm_r2_cone_matched_panel_width_not_ready",
            "metrics": {"scalar_validated_constant_rows": 16},
            "next_action": {"multi_coordinate_design": True},
        },
        model={
            "rounds": 2,
            "key_additions": 3,
            "key_variables": 192,
            "key_model": "independent_round_keys",
        },
        rows=[],
        relation_payload=None,
        planned_coordinates=240,
        worker_status="timeout",
        wall_clock_cap_seconds=60,
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_atm_r2_multicoordinate_support_runtime_not_ready"
    )
    assert result["gate"]["next_action"]["remote_scale"] is False


def test_phase_a_plot_explains_runtime_stop(tmp_path) -> None:
    summary = {
        "gate": {
            "metrics": {
                "planned_coordinates": 240,
                "completed_coordinates": 8,
                "exact_coordinates": 7,
            },
            "source_checks": {"source": True},
            "readiness_checks": {
                "completed_coordinates_at_least_64": False,
                "exact_key_dependent_supports_at_least_16": False,
                "low_weight_positive_relations_at_least_4": False,
            },
            "next_action": {"training": False},
        }
    }
    rows = [
        {
            "query_index": index,
            "support": {"nonzero_support_size": size},
        }
        for index, size in enumerate((41, 41, 1763, 41, 1763, 1763, None, 41))
    ]
    output = tmp_path / "curves.svg"

    render_atm_multicoordinate_support_phase_a(summary, rows, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E61-A" in svg
    assert "60秒只落盘8/240" in svg
    assert "不提高cap、不转远程、不训练RCCA" in svg
