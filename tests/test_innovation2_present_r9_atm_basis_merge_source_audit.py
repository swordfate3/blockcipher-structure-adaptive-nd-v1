from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_r9_atm_basis_merge_source import (
    _serializable_source_audit,
)
from blockcipher_nd.cli.plot_innovation2_present_r9_atm_basis_merge_source import (
    render_basis_merge_audit,
)
from blockcipher_nd.tasks.innovation2.present_r9_atm_basis_merge_source_audit import (
    EXPECTED_DECLARED_SPLITS,
    AtmBasisMergeAuditConfig,
    _gf2_rank_and_dependencies,
    _same_split_coverage,
    adjudicate_basis_merge_audit,
    inspect_merge_bases_source,
)


def test_gf2_dependency_fixture_recovers_three_member_xor() -> None:
    rank, dependencies = _gf2_rank_and_dependencies((0b001, 0b010, 0b011))

    assert rank == 2
    assert dependencies == (0b111,)
    assert 0b001 ^ 0b010 ^ 0b011 == 0


def test_merge_source_audit_detects_discarded_row_reduce_return(tmp_path: Path) -> None:
    discarded = tmp_path / "discarded.py"
    discarded.write_text(
        "def merge_bases(M):\n    M.row_reduce()\n    return M\n",
        encoding="utf-8",
    )
    captured = tmp_path / "captured.py"
    captured.write_text(
        "def merge_bases(M):\n    M = M.row_reduce()\n    return M\n",
        encoding="utf-8",
    )

    bad = inspect_merge_bases_source(discarded)
    good = inspect_merge_bases_source(captured)

    assert bad["discarded_row_reduce_calls"] == 1
    assert bad["row_reduce_return_value_applied"] is False
    assert good["captured_row_reduce_calls"] == 1
    assert good["row_reduce_return_value_applied"] is True


def test_source_summary_converts_tuple_dimension_keys_for_json() -> None:
    source = {"saved_dimensions": {(1, 7, 1): 455}, "checks": {"source": True}}

    converted = _serializable_source_audit(source)

    assert converted["saved_dimensions"] == {"1-7-1": 455}
    json.dumps(converted)


def test_paper_split_coverage_is_order_independent_but_duplicate_sensitive() -> None:
    expected = ((1, 7, 1), (2, 6, 1), (1, 6, 2))

    assert _same_split_coverage(tuple(reversed(expected)), expected)
    assert not _same_split_coverage(((1, 7, 1), (2, 6, 1), (2, 6, 1)), expected)


def test_e98a_passes_source_explanation_but_keeps_e99_closed() -> None:
    source = {
        "checks": {"frozen_source": True},
        "paper_dimension": 470,
        "analysis_merge_count": 470,
        "declared_splits": EXPECTED_DECLARED_SPLITS,
        "merge_contract": {
            "row_reduce_calls": 1,
            "discarded_row_reduce_calls": 1,
            "row_reduce_return_value_applied": False,
        },
    }
    relations = {
        "metrics": {
            "published_files": 8,
            "deduplicated_relations": 470,
            "recomputed_union_rank": 468,
            "union_nullity": 2,
            "recovered_dependencies": 2,
            "dependency_member_histogram": {3: 1, 4: 1},
            "all_files_individually_full_rank": True,
            "all_dependencies_xor_to_zero": True,
        }
    }
    split_rows = [
        {
            "split": "-".join(str(part) for part in split),
            "evidence_state": (
                "declared_without_public_result"
                if split == (3, 3, 3)
                else "published_result"
            ),
            "result_pickle_present": split != (3, 3, 3),
            "stats_file_present": split != (3, 3, 3),
            "notebook_saved_output": split != (3, 3, 3),
        }
        for split in EXPECTED_DECLARED_SPLITS
    ]
    e98_gate = {
        "status": "hold",
        "metrics": {"total_heldout_positives": 24, "eligible_heldout_groups": 1},
        "next_action": {"e99_open": False},
    }

    gate = adjudicate_basis_merge_audit(
        AtmBasisMergeAuditConfig(),
        source_audit=source,
        relation_audit=relations,
        split_rows=split_rows,
        e98_gate=e98_gate,
        e98_gate_hash=(
            "f4c560233616c720f8a9b7eea1bc93e29cda69978ce908b5cd9f8f01fc23bc5c"
        ),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_r9_atm_public_merge_count_not_rank"
    assert gate["next_action"]["e99_open"] is False
    assert gate["next_action"]["remote_scale"] is False
    assert gate["next_action"]["missing_split_generation"] is False


def test_e98a_plot_explains_count_rank_and_missing_split(tmp_path: Path) -> None:
    summary = {
        "file_ranks": [
            {
                "split": "-".join(str(part) for part in split),
                "serialized_basis_elements": 330 + index,
                "recomputed_rank": 330 + index,
            }
            for index, split in enumerate(EXPECTED_DECLARED_SPLITS[:-1])
        ],
        "dependencies": [
            {"dependency_id": "dependency_000", "members": 3},
            {"dependency_id": "dependency_001", "members": 4},
        ],
        "split_coverage": [
            {
                "split": "-".join(str(part) for part in split),
                "evidence_state": (
                    "declared_without_public_result"
                    if split == (3, 3, 3)
                    else "published_result"
                ),
            }
            for split in EXPECTED_DECLARED_SPLITS
        ],
        "gate": {
            "metrics": {
                "data_analysis_saved_merge_count": 470,
                "deduplicated_relations": 470,
                "recomputed_union_rank": 468,
                "union_nullity": 2,
            }
        },
    }
    output = tmp_path / "curves.svg"

    render_basis_merge_audit(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "创新2 E98-A" in svg
    assert "470是计数" in svg
    assert "468是重算秩" in svg
    assert "3-3-3" in svg
    assert "不训练E99" in svg
