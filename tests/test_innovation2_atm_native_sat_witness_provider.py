from __future__ import annotations

from blockcipher_nd.tasks.innovation2.atm_native_sat_witness_provider import (
    NativeSatProviderConfig,
    algebraic_transition_coefficient,
    count_models_parity,
    evaluate_phase_a,
    evaluate_low_round_panel,
    evaluate_cone_matched_panel,
    evaluate_r9_probe,
    key_mask_from_projected_literals,
    select_singleton_relation_mutation,
    scalar_present_independent_key_coefficient,
    present_two_round_input_cone,
    two_round_cone_matched_queries,
)
from blockcipher_nd.cli.plot_innovation2_atm_native_sat_provider import (
    render_atm_native_sat_provider,
)
from blockcipher_nd.cli.plot_innovation2_atm_native_sat_r9_probe import (
    render_atm_native_sat_r9_probe,
)
from blockcipher_nd.cli.plot_innovation2_atm_r2_strict_relation_panel import (
    render_atm_r2_strict_relation_panel,
)
from blockcipher_nd.cli.plot_innovation2_atm_r2_cone_matched_panel import (
    render_atm_r2_cone_matched_panel,
)


class _FakeSolver:
    def __init__(self, *, bootstrap_with: object, models: int) -> None:
        self.models = models

    def __enter__(self) -> "_FakeSolver":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def enum_models(self, *, assumptions: object) -> object:
        return iter([()] * self.models)


def test_present_sbox_transition_coefficient_matches_known_anf_terms() -> None:
    present = (0xC, 5, 6, 0xB, 9, 0, 0xA, 0xD, 3, 0xE, 0xF, 8, 4, 7, 1, 2)

    assert algebraic_transition_coefficient(
        present, input_bits=4, output_exponent=1, input_exponent=1
    ) == 1
    assert algebraic_transition_coefficient(
        present, input_bits=4, output_exponent=1, input_exponent=15
    ) == 0
    assert algebraic_transition_coefficient(
        present, input_bits=4, output_exponent=0, input_exponent=0
    ) == 1


def test_key_mask_conversion_uses_projected_variable_order() -> None:
    assert key_mask_from_projected_literals((5, -9, 12), (5, 9, 12)) == 0b101


def test_model_cap_returns_unknown_instead_of_a_false_certificate() -> None:
    result = count_models_parity(
        ((1,),),
        (),
        model_cap=2,
        solver_factory=lambda **kwargs: _FakeSolver(**kwargs, models=3),
    )

    assert result["status"] == "unknown"
    assert result["parity"] is None


def test_phase_a_gate_opens_only_the_single_r9_probe() -> None:
    checks = {
        "all_256_sbox_coefficients_match": True,
        "toy_key_term_returns_witness": True,
        "toy_witness_is_nonzero_key_monomial": True,
        "toy_witness_replay_is_exactly_odd": True,
        "toy_constant_term_has_no_key_witness": True,
        "cap_exhaustion_is_unknown": True,
    }
    result = evaluate_phase_a(
        NativeSatProviderConfig(run_id="e58_test"),
        actual_commit=NativeSatProviderConfig(run_id="x").expected_commit,
        source={"checks": {"source": True}},
        calibration={"checks": checks, "metrics": {"sbox_matches": 256}},
        environment={
            "python_sat_available": True,
            "glucose4_available": True,
            "bitarrays_extension_available": True,
        },
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["next_action"]["r9_probe"] is True
    assert result["gate"]["next_action"]["training"] is False
    assert result["gate"]["next_action"]["remote_scale"] is False


def test_phase_a_plot_states_low_round_scope(tmp_path: object) -> None:
    from pathlib import Path

    output = Path(str(tmp_path)) / "curves.svg"
    summary = {
        "calibration": {
            "metrics": {
                "sbox_coefficients": 256,
                "sbox_matches": 256,
                "sbox_nonzero_coefficients": 90,
            },
            "checks": {
                "toy_key_term_returns_witness": True,
                "toy_witness_replay_is_exactly_odd": True,
                "toy_constant_term_has_no_key_witness": True,
                "cap_exhaustion_is_unknown": True,
            },
        },
        "gate": {
            "decision": "innovation2_atm_native_sat_mechanism_ready_for_r9_probe",
            "source_checks": {"atm_commit_matches_frozen_version": True},
            "environment_checks": {
                "glucose4_available": True,
                "bitarrays_extension_available": True,
            },
        },
    }

    render_atm_native_sat_provider(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E58-A" in svg
    assert "不是九轮负类" in svg
    assert "cap必须返回unknown" in svg


def test_singleton_mutation_preserves_marginals_and_avoids_positive_span() -> None:
    relations = (
        ((0b1110, 0b0001),),
        ((0b1101, 0b0010),),
        ((0b1011, 0b0100),),
    )

    candidate = select_singleton_relation_mutation(relations, state_bits=4)

    assert candidate["relation_size"] == 1
    assert candidate["source_output_weight"] == candidate["candidate_output_weight"]
    assert candidate["candidate_is_in_public_positive_span"] is False
    assert candidate["source_relation"] != candidate["candidate_relation"]


def test_r9_timeout_remains_unknown_and_does_not_open_training() -> None:
    candidate = {
        "source_is_public_positive": True,
        "candidate_is_in_public_positive_span": False,
        "relation_size": 1,
        "input_weight": 60,
        "source_output_weight": 1,
        "candidate_output_weight": 1,
    }

    result = evaluate_r9_probe(
        NativeSatProviderConfig(run_id="e58b_test"),
        candidate=candidate,
        worker_status="timeout",
        worker_result=None,
        wall_clock_cap_seconds=60,
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_atm_native_sat_r9_wall_clock_cap_exceeded"
    )
    assert result["gate"]["next_action"]["training"] is False


def test_r9_exact_odd_replay_opens_only_label_width_audit() -> None:
    candidate = {
        "source_is_public_positive": True,
        "candidate_is_in_public_positive_span": False,
        "relation_size": 1,
        "input_weight": 60,
        "source_output_weight": 1,
        "candidate_output_weight": 1,
    }
    worker = {
        "model": {"key_model": "independent_round_keys"},
        "probe": {
            "status": "witness",
            "witness": {"key_exponent_mask": 1},
        },
        "replay": {"status": "exact", "parity": 1},
    }

    result = evaluate_r9_probe(
        NativeSatProviderConfig(run_id="e58b_test"),
        candidate=candidate,
        worker_status="completed",
        worker_result=worker,
        wall_clock_cap_seconds=60,
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["next_action"]["label_width_audit"] is True
    assert result["gate"]["next_action"]["training"] is False


def test_r9_plot_calls_timeout_unknown_not_negative(tmp_path: object) -> None:
    from pathlib import Path

    output = Path(str(tmp_path)) / "curves.svg"
    summary = {
        "candidate": {
            "input_weight": 60,
            "source_output_weight": 1,
            "candidate_output_weight": 1,
        },
        "gate": {
            "decision": "innovation2_atm_native_sat_r9_wall_clock_cap_exceeded",
            "wall_clock_cap_seconds": 60,
            "projected_key_cap": 1 << 16,
            "trail_model_cap": 1 << 20,
            "candidate_checks": {"marginal": True},
            "witness_checks": {
                "worker_completed_within_wall_clock_cap": False,
                "nonzero_key_exponent_witness_found": False,
            },
            "next_action": {"training": False},
        },
    }

    render_atm_native_sat_r9_probe(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E58-B" in svg
    assert "超时不等于负类" in svg
    assert "保持unknown" in svg


def test_r2_panel_requires_both_strict_classes() -> None:
    phase_a = {
        "status": "pass",
        "decision": "innovation2_atm_native_sat_mechanism_ready_for_r9_probe",
    }
    model = {
        "rounds": 2,
        "key_additions": 3,
        "key_variables": 192,
        "key_model": "independent_round_keys",
    }
    rows = [
        {"label": 1, "probe": {"status": "no_witness"}, "replay": None}
        for _ in range(12)
    ]

    result = evaluate_low_round_panel(
        NativeSatProviderConfig(run_id="e59_test"),
        phase_a_gate=phase_a,
        model=model,
        rows=rows,
        planned_queries=16,
        rounds=2,
        worker_status="timeout",
        wall_clock_cap_seconds=60,
    )

    assert result["gate"]["status"] == "hold"
    assert not result["gate"]["width_checks"][
        "strict_key_dependent_rows_at_least_4"
    ]
    assert result["gate"]["next_action"]["training"] is False


def test_r2_panel_passes_with_replayed_balanced_width() -> None:
    phase_a = {
        "status": "pass",
        "decision": "innovation2_atm_native_sat_mechanism_ready_for_r9_probe",
    }
    model = {
        "rounds": 2,
        "key_additions": 3,
        "key_variables": 192,
        "key_model": "independent_round_keys",
    }
    constants = [
        {"label": 1, "probe": {"status": "no_witness"}, "replay": None}
        for _ in range(8)
    ]
    negatives = [
        {
            "label": 0,
            "probe": {
                "status": "witness",
                "witness": {"key_exponent_mask": index + 1},
            },
            "replay": {"status": "exact", "parity": 1},
        }
        for index in range(8)
    ]

    result = evaluate_low_round_panel(
        NativeSatProviderConfig(run_id="e59_test"),
        phase_a_gate=phase_a,
        model=model,
        rows=constants + negatives,
        planned_queries=16,
        rounds=2,
        worker_status="completed",
        wall_clock_cap_seconds=60,
    )

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["next_action"]["full_width_audit"] is True
    assert result["gate"]["next_action"]["training"] is False


def test_r2_plot_explains_single_class_hold(tmp_path: object) -> None:
    from pathlib import Path

    output = Path(str(tmp_path)) / "curves.svg"
    summary = {
        "model": {"cnf_clauses": 2080},
        "gate": {
            "decision": "innovation2_atm_r2_strict_relation_panel_not_ready",
            "metrics": {
                "completed_queries": 16,
                "explicit_unknown_rows": 0,
                "missing_timeout_rows": 0,
                "strict_constant_rows": 16,
                "strict_key_dependent_rows": 0,
            },
            "width_checks": {
                "completed_queries_at_least_12": True,
                "strict_constant_rows_at_least_4": True,
                "strict_key_dependent_rows_at_least_4": False,
            },
            "next_action": {"training": False},
        },
    }

    render_atm_r2_strict_relation_panel(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E59" in svg
    assert "单一类别阻止神经训练" in svg
    assert "不能训练RCCA" in svg


def test_two_round_cone_matched_queries_are_label_blind_weight_pairs() -> None:
    queries = two_round_cone_matched_queries()

    assert present_two_round_input_cone(0) == tuple(range(16))
    assert len(queries) == 16
    for weight in range(1, 9):
        pair = queries[2 * (weight - 1) : 2 * weight]
        assert [query["cone_group"] for query in pair] == ["inside", "outside"]
        assert all(query["weight"] == weight for query in pair)
        assert all(int(query["input_exponent"]).bit_count() == weight for query in pair)
        assert pair[0]["output_exponent"] == pair[1]["output_exponent"] == 1


def test_cone_panel_routes_shortcut_dominated_labels_to_multicoordinate() -> None:
    phase_a = {
        "status": "pass",
        "decision": "innovation2_atm_native_sat_mechanism_ready_for_r9_probe",
    }
    model = {
        "rounds": 2,
        "key_additions": 3,
        "key_variables": 192,
        "key_model": "independent_round_keys",
    }
    rows = []
    for query in two_round_cone_matched_queries():
        inside = bool(query["all_input_bits_inside_cone"])
        rows.append(
            {
                "input_exponent": query["input_exponent"],
                "output_exponent": query["output_exponent"],
                "label": 0 if inside else 1,
                "query_metadata": {
                    key: value
                    for key, value in query.items()
                    if key not in {"input_exponent", "output_exponent"}
                },
                "probe": (
                    {
                        "status": "witness",
                        "witness": {"key_exponent_mask": 1},
                    }
                    if inside
                    else {"status": "no_witness"}
                ),
                "replay": (
                    {"status": "exact", "parity": 1 if inside else 0}
                ),
            }
        )

    result = evaluate_cone_matched_panel(
        NativeSatProviderConfig(run_id="e60_test"),
        phase_a_gate=phase_a,
        model=model,
        rows=rows,
        planned_queries=16,
        worker_status="completed",
        wall_clock_cap_seconds=60,
    )

    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_atm_r2_singleton_relation_shortcut_dominated"
    )
    assert result["gate"]["metrics"][
        "cone_membership_strongest_direction_auc"
    ] == 1.0
    assert result["gate"]["next_action"]["multi_coordinate_design"] is True
    assert result["gate"]["next_action"]["training"] is False


def test_cone_panel_all_constant_routes_to_multicoordinate(tmp_path) -> None:
    phase_a = {
        "status": "pass",
        "decision": "innovation2_atm_native_sat_mechanism_ready_for_r9_probe",
    }
    model = {
        "rounds": 2,
        "key_additions": 3,
        "key_variables": 192,
        "key_model": "independent_round_keys",
        "cnf_clauses": 2080,
    }
    rows = []
    for query_index, query in enumerate(two_round_cone_matched_queries()):
        expected_constant = 1 if query_index == 0 else 0
        rows.append(
            {
                "query_index": query_index,
                "input_exponent": query["input_exponent"],
                "output_exponent": query["output_exponent"],
                "label": 1,
                "query_metadata": {
                    key: value
                    for key, value in query.items()
                    if key not in {"input_exponent", "output_exponent"}
                },
                "probe": {"status": "no_witness", "witness": None},
                "replay": {"status": "exact", "parity": expected_constant},
            }
        )

    result = evaluate_cone_matched_panel(
        NativeSatProviderConfig(run_id="e60_all_constant_test"),
        phase_a_gate=phase_a,
        model=model,
        rows=rows,
        planned_queries=16,
        worker_status="completed",
        wall_clock_cap_seconds=60,
    )

    gate = result["gate"]
    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_atm_r2_cone_matched_panel_width_not_ready"
    )
    assert gate["metrics"]["scalar_validated_constant_rows"] == 16
    assert gate["source_checks"][
        "all_constant_rows_match_three_scalar_key_sets"
    ] is True
    assert gate["next_action"]["multi_coordinate_design"] is True
    assert gate["next_action"]["training"] is False

    summary = {"gate": gate, "model": model}
    output = tmp_path / "curves.svg"
    render_atm_r2_cone_matched_panel(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E60" in svg
    assert "16/16条constant完成标量复核" in svg
    assert "转向多坐标GF(2)消去关系" in svg


def test_scalar_independent_key_coefficient_rejects_wrong_key_count() -> None:
    try:
        scalar_present_independent_key_coefficient(
            rounds=2,
            input_exponent=1,
            output_exponent=1,
            round_keys=(0, 0),
        )
    except ValueError as exc:
        assert "post-whitening" in str(exc)
    else:
        raise AssertionError("wrong key count was accepted")
