from __future__ import annotations

import random
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_output_parity_exact_anf_difficulty import (
    render_exact_anf_difficulty,
)
from blockcipher_nd.tasks.innovation2.output_parity_exact_anf_difficulty import (
    OutputParityAnfDifficultyConfig,
    adjudicate_exact_anf_difficulty,
    compute_fixed_key_parity_anf,
)


def _secret_key() -> int:
    return random.Random(910_000).getrandbits(80)


def test_fixed_key_exact_anf_matches_scalar_present_across_rounds() -> None:
    config = OutputParityAnfDifficultyConfig(
        run_id="test-exact-anf",
        maximum_terms=100_000,
        maximum_seconds=5.0,
        maximum_memory_bytes=1 << 30,
        assignment_checks=3,
    )

    rows = [
        compute_fixed_key_parity_anf(
            config,
            rounds=rounds,
            mask_index=0,
            secret_key=_secret_key(),
        )
        for rounds in (1, 2, 3)
    ]

    assert all(row["status"] == "completed" for row in rows)
    assert all(row["all_assignment_checks_match"] for row in rows)
    assert [row["functional_support_width"] for row in rows] == [4, 16, 64]
    assert [row["exact_algebraic_degree"] for row in rows] == [3, 6, 12]
    assert rows[0]["exact_monomial_count"] < rows[1]["exact_monomial_count"]
    assert rows[1]["exact_monomial_count"] < rows[2]["exact_monomial_count"]


def test_exact_anf_gate_requires_complexity_growth_and_auc_decay() -> None:
    config = OutputParityAnfDifficultyConfig(run_id="test-exact-anf-gate")
    rows = []
    for rounds, support, degree, monomials in (
        (1, 4, 3, 5),
        (2, 16, 6, 200),
        (3, 64, 12, 20_000),
    ):
        for mask_index in range(16):
            rows.append(
                {
                    "rounds": rounds,
                    "mask_index": mask_index,
                    "status": "completed",
                    "cone_widths": [support],
                    "functional_support_width": support,
                    "exact_algebraic_degree": degree,
                    "exact_monomial_count": monomials,
                    "log2_train_coverage": 12 - support,
                    "train_rows_per_monomial": 4096 / monomials,
                    "all_assignment_checks_match": True,
                    "maximum_observed_terms": monomials,
                    "elapsed_seconds": 0.1,
                }
            )
    source_gates = {
        1: {
            "decision": "innovation2_output_parity_mask_geometry_two_key_confirmed",
            "metrics": {"mean_aligned_parity_macro_auc": 0.96},
        },
        2: {
            "decision": "innovation2_output_parity_present_r2_two_key_supported",
            "metrics": {"mean_aligned_parity_macro_auc": 0.63},
        },
        3: {
            "decision": "innovation2_output_parity_present_r3_two_key_not_supported",
            "metrics": {"mean_aligned_parity_macro_auc": 0.525},
        },
    }

    gate = adjudicate_exact_anf_difficulty(config, rows, source_gates)

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_output_parity_exact_anf_difficulty_transition_confirmed"
    )
    assert gate["next_action"]["sample_classification"] is False
    assert gate["next_action"]["next_adjudication"] == (
        "op9_present_r3_nested_data_slope"
    )

    capped_rows = rows[:-1] + [
        {
            "rounds": 3,
            "mask_index": 15,
            "status": "cap_exceeded",
            "cone_widths": [64, 16, 4, 4],
            "maximum_observed_terms": 500_001,
            "elapsed_seconds": 0.5,
        }
    ]
    capped_gate = adjudicate_exact_anf_difficulty(config, capped_rows, source_gates)
    assert capped_gate["status"] == "hold"
    assert capped_gate["decision"] == (
        "innovation2_output_parity_exact_anf_difficulty_hard_cap_exceeded"
    )


def test_exact_anf_plot_explains_deterministic_output_difficulty(
    tmp_path: Path,
) -> None:
    summary = {
        "gate": {
            "status": "pass",
            "round_summaries": [
                {
                    "completed_masks": 16,
                    "cap_exceeded_masks": 0,
                    "structural_input_cone_median": 4,
                    "functional_support_median": 4,
                    "exact_degree_median": 3,
                    "monomial_count_median": 5,
                    "maximum_observed_terms": 5,
                    "two_key_mean_aligned_auc": 0.96,
                    "log2_train_coverage_median": 8,
                },
                {
                    "completed_masks": 16,
                    "cap_exceeded_masks": 0,
                    "structural_input_cone_median": 16,
                    "functional_support_median": 16,
                    "exact_degree_median": 6,
                    "monomial_count_median": 200,
                    "maximum_observed_terms": 200,
                    "two_key_mean_aligned_auc": 0.63,
                    "log2_train_coverage_median": -4,
                },
                {
                    "completed_masks": 16,
                    "cap_exceeded_masks": 0,
                    "structural_input_cone_median": 64,
                    "functional_support_median": 64,
                    "exact_degree_median": 12,
                    "monomial_count_median": 20_000,
                    "maximum_observed_terms": 20_000,
                    "two_key_mean_aligned_auc": 0.525,
                    "log2_train_coverage_median": -52,
                },
            ],
        }
    }
    output = tmp_path / "curves.svg"

    render_exact_anf_difficulty(summary, output)

    svg = output.read_text(encoding="utf-8")
    assert "真实密文输出parity" in svg
    assert "精确GF(2)消去" in svg
    assert "无训练确定性审计" in svg
    assert "不证明增加样本一定成功" in svg
