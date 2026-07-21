from __future__ import annotations

from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_r9_external_relation_source_readiness import (
    render_external_source_readiness,
)
from blockcipher_nd.tasks.innovation2.present_r9_external_relation_source_readiness import (
    ExternalRelationSourceConfig,
    _split_present_rows_present,
    _source_rows,
    adjudicate_external_relation_sources,
)


def test_e106_splitandcancel_table_check_tolerates_pdf_column_spacing() -> None:
    text = """
        PRESENT        9      263          28          1           -      reference
                       9      260           3          1         16.37    Subsection 5.1
                       9      260           4        ≥ 2−2       16.37    Subsection 5.1
    """

    assert _split_present_rows_present(text) is True
    assert _split_present_rows_present(text.replace("Subsection 5.1", "other")) is False


def _rows() -> list[dict[str, object]]:
    return _source_rows(
        ExternalRelationSourceConfig(),
        novelty={
            "heldout_exact_public_overlap": 318,
            "new_relation_space_dimensions": 0,
        },
        fold_overlap={"maximum_fold_training_overlap": 13_453},
    )


def test_e106_holds_e99_when_no_external_source_is_eligible() -> None:
    rows = _rows()
    gate = adjudicate_external_relation_sources(
        ExternalRelationSourceConfig(),
        protocol_checks={"sources_replay": True},
        source_rows=rows,
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_present_r9_external_relation_source_unavailable"
    )
    assert gate["metrics"]["candidate_sources"] == 5
    assert gate["metrics"]["eligible_external_sources"] == 0
    assert gate["metrics"]["machine_readable_candidate_sources"] == 1
    assert gate["metrics"]["maximum_known_new_dimensions"] == 0
    assert gate["next_action"]["e99_evaluation_open"] is False
    assert gate["next_action"]["target_representation_change_required"] is False
    assert gate["next_action"]["provider_research_open"] is False
    assert gate["next_action"]["thesis_consolidation"] is True
    assert "new sound provider" in gate["next_action"]["action"]


def test_e106_opens_only_for_a_fully_qualified_source() -> None:
    rows = _rows()
    synthetic = {
        **rows[1],
        "source_id": "synthetic_independent_source",
        "machine_readable_relations": True,
        "same_rounds": True,
        "same_key_model": True,
        "same_relation_semantics": True,
        "training_identity_disjoint": True,
        "new_relation_space_dimensions": 32,
        "minimum_novelty_met": True,
        "direct_e99_eligible": True,
    }
    gate = adjudicate_external_relation_sources(
        ExternalRelationSourceConfig(),
        protocol_checks={"sources_replay": True},
        source_rows=[*rows, synthetic],
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r9_external_relation_source_ready"
    assert gate["metrics"]["eligible_external_sources"] == 1
    assert gate["next_action"]["e99_evaluation_open"] is True


def test_e106_source_matrix_keeps_alternative_semantics_out_of_e99() -> None:
    rows = {row["source_id"]: row for row in _rows()}

    assert rows["atm_e104_split333"]["new_relation_space_dimensions"] == 0
    assert rows["atm_e104_split333"]["direct_e99_eligible"] is False
    assert rows["hwang_present_r9_masks"]["alternative_basis_dimension"] == 4
    assert rows["hwang_present_r9_masks"]["same_relation_semantics"] is False
    assert rows["splitandcancel_present"]["artifact_state"] == (
        "repository_stub_no_results"
    )
    assert rows["splitandcancel_present"]["direct_e99_eligible"] is False


def test_e106_plot_explains_source_gates_and_stop_boundary(
    tmp_path: Path,
) -> None:
    summary = {
        "source_matrix": _rows(),
        "e104_novelty": {"new_relation_space_dimensions": 0},
        "gate": {
            "decision": "innovation2_present_r9_external_relation_source_unavailable"
        },
    }
    output = tmp_path / "curves.svg"

    render_external_source_readiness(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "PRESENT九轮外部关系来源资格审计" in svg
    assert "当前没有来源满足全部条件" in svg
    assert "Hwang 2026 PRESENT R9四个输出mask" in svg
    assert "新增维度=0" in svg
    assert "遵守E97停止门并转论文收束" in svg
    assert "新的可靠provider" in svg
