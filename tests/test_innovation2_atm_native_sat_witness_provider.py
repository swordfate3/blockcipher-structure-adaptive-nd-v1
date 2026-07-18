from __future__ import annotations

from blockcipher_nd.tasks.innovation2.atm_native_sat_witness_provider import (
    NativeSatProviderConfig,
    algebraic_transition_coefficient,
    count_models_parity,
    evaluate_phase_a,
    evaluate_r9_probe,
    key_mask_from_projected_literals,
    select_singleton_relation_mutation,
)
from blockcipher_nd.cli.plot_innovation2_atm_native_sat_provider import (
    render_atm_native_sat_provider,
)
from blockcipher_nd.cli.plot_innovation2_atm_native_sat_r9_probe import (
    render_atm_native_sat_r9_probe,
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
