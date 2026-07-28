from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1ae import render_k1ae_svg
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ae import adjudicate


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CONDITIONS = ("full", "histogram_off", "edge_off", "base_only")


def _rows(*, base_drops: tuple[float, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, base_drop in enumerate(base_drops):
        full = 0.999 - seed * 0.001
        aucs = {
            "full": full,
            "histogram_off": full - base_drop / 4,
            "edge_off": full - base_drop / 3,
            "base_only": full - base_drop,
        }
        for condition in CONDITIONS:
            edge = 0.0 if condition in {"edge_off", "base_only"} else 0.2
            histogram = 0.0 if condition in {"histogram_off", "base_only"} else 0.3
            rows.append(
                {
                    "run_id": "i1_uknit_family_ctspn_dialga_branch_ablation_k1ae_20260729",
                    "seed": seed,
                    "condition": condition,
                    "cipher_key": "dialga128",
                    "rounds": 4,
                    "auc": aucs[condition],
                    "source_full_auc": full,
                    "max_abs_probability_delta_from_full": 0.0 if condition == "full" else 0.2,
                    "mean_abs_probability_delta_from_full": 0.0 if condition == "full" else 0.02,
                    "learned_edge_gate": 0.2,
                    "learned_histogram_gate": 0.3,
                    "applied_edge_gate": edge,
                    "applied_histogram_gate": histogram,
                    "intervention_sha256": SHA_A,
                    "checkpoint_sha256": SHA_A if seed == 0 else SHA_B,
                    "checkpoint_selected": "best",
                    "checkpoint_reported_seed": seed,
                    "pre_intervention_state_dict_sha256": SHA_C,
                    "feature_sha256": SHA_A,
                    "label_sha256": SHA_B,
                    "metadata_sha256": SHA_C,
                    "cache_dir": f"cache-seed{seed}",
                    "source_k1ac_gate_sha256": SHA_A,
                    "source_k1ad_results_sha256": SHA_B,
                    "source_k1ad_gate_sha256": SHA_C,
                    "runtime_structure_window_sha256": SHA_A,
                    "samples_total": 2048,
                    "input_bits": 4096,
                    "pair_bits": 256,
                    "pairs_per_sample": 16,
                    "input_difference": 0x40,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "validation_seed": 10000 + seed,
                    "parameter_count": 214316,
                    "strict_state_dict_load": True,
                    "training_performed": False,
                    "optimizer_steps": 0,
                }
            )
    return rows


def test_k1ae_classifies_base_path_dominance_per_seed() -> None:
    gate = adjudicate(_rows(base_drops=(0.002, 0.004)))
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_family_ctspn_k1ae_gf2_base_path_dominates"
    assert all(gate["protocol_checks"].values())
    assert gate["research_checks"]["both_seeds_base_only_retains_within_0p010"] is True


def test_k1ae_classifies_joint_residual_necessity() -> None:
    gate = adjudicate(_rows(base_drops=(0.020, 0.030)))
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_family_ctspn_k1ae_residuals_jointly_necessary"
    assert gate["research_checks"]["both_seeds_joint_residuals_necessary"] is True


def test_k1ae_rejects_undeclared_gate_intervention() -> None:
    rows = deepcopy(_rows(base_drops=(0.002, 0.004)))
    rows[1]["applied_edge_gate"] = 0.0
    gate = adjudicate(rows)
    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["declared_gate_interventions_exact"] is False


def test_k1ae_rejects_cache_or_training_drift() -> None:
    rows = deepcopy(_rows(base_drops=(0.002, 0.004)))
    rows[-1]["optimizer_steps"] = 1
    rows[-1]["input_bits"] = 1024
    gate = adjudicate(rows)
    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["inference_only"] is False
    assert gate["protocol_checks"]["frozen_validation_geometry"] is False


def test_k1ae_plot_explains_the_gf2_base_path(tmp_path: Path) -> None:
    gate = adjudicate(_rows(base_drops=(0.002, 0.004)))
    output = tmp_path / "curves.svg"
    report = render_k1ae_svg(gate, output)
    text = output.read_text(encoding="utf-8")
    assert report["status"] == "rendered_pending_visual_qa"
    assert "近满分信号究竟来自哪条网络路径" in text
    assert "基础路径仍读取 GF(2) 线性扩散" in text
    assert "单对重放审计" in text
