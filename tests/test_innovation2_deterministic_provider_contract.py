from __future__ import annotations

import pickle
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_deterministic_provider import (
    render_provider_contract_svg,
)
from blockcipher_nd.tasks.innovation2 import deterministic_provider_contract as provider


def _write_pickle(path: Path, payload: object) -> None:
    path.write_bytes(pickle.dumps(payload, protocol=4))


def test_restricted_pickle_accepts_builtin_properties_and_rejects_globals(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.pkl"
    _write_pickle(valid, {frozenset({(3, 1)}), frozenset({(2, 1), (3, 2)})})
    assert len(provider.load_builtin_property_pickle(valid)) == 2

    invalid = tmp_path / "invalid.pkl"
    _write_pickle(invalid, Path("not-a-property"))
    try:
        provider.load_builtin_property_pickle(invalid)
    except pickle.UnpicklingError as exc:
        assert "global class loading is forbidden" in str(exc)
    else:
        raise AssertionError("restricted unpickler accepted a global class")


def test_atm_audit_recomputes_rank_and_preserves_unknown_label_semantics(
    tmp_path: Path,
) -> None:
    _write_pickle(
        tmp_path / "a.pkl",
        {
            frozenset({(3, 1)}),
            frozenset({(2, 1), (3, 2)}),
        },
    )
    _write_pickle(
        tmp_path / "b.pkl",
        {
            frozenset({(3, 1)}),
            frozenset({(2, 1), (3, 2)}),
        },
    )
    result = provider.audit_atm_results(
        tmp_path,
        expected_names=("a.pkl", "b.pkl"),
        published_dimension=2,
    )
    assert result["metrics"]["unique_serialized_basis_elements"] == 2
    assert result["metrics"]["union_gf2_rank"] == 2
    assert result["metrics"]["standard_basis_members"] == 1
    assert result["metrics"]["linear_output_standard_basis_members"] == 1
    assert result["semantic_checks"]["constant_value_zero_or_one_is_known"] is False
    assert result["semantic_checks"]["absence_from_found_subspace_is_complete_negative"] is False


def test_provider_gate_holds_on_semantic_mismatch() -> None:
    config = provider.ProviderAuditConfig(run_id="test")
    claasp = {
        "checks": {"source": True},
        "runtime": {"current_runtime_available": False},
        "semantic_checks": {
            "selected_output_bit_api_present": True,
            "current_runtime_can_verify_label_value": False,
            "current_runtime_can_generate_complete_negatives": False,
        },
    }
    atm = {
        "checks": {"source": True},
        "metrics": {"union_gf2_rank": 468},
        "semantic_checks": {
            "linear_output_candidates_exist": True,
            "constant_value_zero_or_one_is_known": False,
            "absence_from_found_subspace_is_complete_negative": False,
            "multi_term_relations_are_single_mask_labels": False,
            "published_dimension_matches_recomputed_union_rank": False,
        },
    }
    result = provider.evaluate_provider_contract(
        config,
        claasp=claasp,
        atm=atm,
        actual_claasp_commit=provider.CLAASP_COMMIT,
        actual_atm_commit=provider.ATM_COMMIT,
    )
    assert result["gate"]["status"] == "hold"
    assert result["gate"]["decision"] == (
        "innovation2_deterministic_provider_semantics_mismatch"
    )
    assert result["gate"]["next_action"]["training"] is False


def test_provider_plot_has_clear_scope_and_contract_labels(tmp_path: Path) -> None:
    summary = {
        "claasp": {
            "runtime": {"current_runtime_available": False},
            "semantic_checks": {"selected_output_bit_api_present": True},
        },
        "atm": {
            "metrics": {
                "result_files": 8,
                "unique_serialized_basis_elements": 470,
                "union_gf2_rank": 468,
                "support_coordinates": 673,
                "standard_basis_members": 305,
                "linear_output_standard_basis_members": 198,
            },
            "semantic_checks": {
                "linear_output_candidates_exist": True,
                "constant_value_zero_or_one_is_known": False,
                "absence_from_found_subspace_is_complete_negative": False,
                "published_dimension_matches_recomputed_union_rank": False,
            },
        },
        "gate": {"decision": "innovation2_deterministic_provider_semantics_mismatch"},
    }
    output = tmp_path / "curves.svg"
    render_provider_contract_svg(summary, output)
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E31" in svg
    assert "0/1常数明确" in svg
    assert "不是CLAASP-MP复现" in svg
