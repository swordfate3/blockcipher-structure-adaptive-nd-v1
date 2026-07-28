from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_uknit_family_ctspn_k1af import render_k1af_svg
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1ac import read_tasks, task_map
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1af import (
    REPLAY_TOLERANCE,
    _build_one_pair_model,
    _direct_repeat_error,
    adjudicate,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_ctspn_dialga_retention_"
    "k1ac_16pair_2048_seed0_seed1.csv"
)


def _rows(*, exact_aucs: tuple[float, float], margins: tuple[float, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, (pooled_exact, margin) in enumerate(zip(exact_aucs, margins)):
        for scope, position in [("pooled", -1), *[("pair_position", index) for index in range(16)], ("mean_query_aggregate", -1)]:
            exact_auc = pooled_exact if scope == "pooled" else min(0.999, pooled_exact + 0.02)
            for condition in ("exact", "wrong_sbox"):
                auc = exact_auc if condition == "exact" else exact_auc - margin
                rows.append(
                    {
                        "run_id": "i1_uknit_family_ctspn_dialga_single_pair_replay_k1af_20260729",
                        "seed": seed,
                        "condition": condition,
                        "scope": scope,
                        "pair_position": position,
                        "auc": auc,
                        "exact_minus_condition_auc": 0.0 if condition == "exact" else margin,
                        "max_abs_probability_delta_from_exact": 0.0 if condition == "exact" else 0.2,
                        "mean_abs_probability_delta_from_exact": 0.0 if condition == "exact" else 0.02,
                        "observation_rows": 32768 if scope == "pooled" else 2048,
                        "checkpoint_sha256": SHA_A if seed == 0 else SHA_B,
                        "checkpoint_selected": "best",
                        "checkpoint_reported_seed": seed,
                        "state_dict_sha256": SHA_C,
                        "feature_sha256": SHA_A,
                        "label_sha256": SHA_B,
                        "metadata_sha256": SHA_C,
                        "cache_dir": f"cache-seed{seed}",
                        "source_k1ae_gate_sha256": SHA_A,
                        "one_pair_input_bits": 256,
                        "source_pairs_per_sample": 16,
                        "audit_pairs_per_observation": 1,
                        "parameter_count": 214316,
                        "direct_repeat_logit_max_error": 1e-7,
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "data_generation_performed": False,
                    }
                )
    return rows


def test_k1af_passes_only_for_usable_nonsaturated_semantic_surface() -> None:
    gate = adjudicate(_rows(exact_aucs=(0.75, 0.72), margins=(0.02, 0.015)))
    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation1_uknit_family_ctspn_k1af_one_pair_semantic_surface_supported"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_k1af_holds_a_nonsaturated_surface_without_semantic_margin() -> None:
    gate = adjudicate(_rows(exact_aucs=(0.75, 0.72), margins=(0.001, -0.002)))
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation1_uknit_family_ctspn_k1af_one_pair_semantic_attribution_failed"


def test_k1af_holds_saturated_or_weak_surfaces() -> None:
    saturated = adjudicate(_rows(exact_aucs=(0.97, 0.96), margins=(0.02, 0.02)))
    weak = adjudicate(_rows(exact_aucs=(0.53, 0.52), margins=(0.02, 0.02)))
    assert saturated["decision"] == "innovation1_uknit_family_ctspn_k1af_one_pair_still_saturated"
    assert weak["decision"] == "innovation1_uknit_family_ctspn_k1af_one_pair_signal_too_weak"


def test_k1af_rejects_replay_or_row_count_drift() -> None:
    rows = deepcopy(_rows(exact_aucs=(0.75, 0.72), margins=(0.02, 0.015)))
    rows[0]["direct_repeat_logit_max_error"] = 1e-3
    rows[-1]["observation_rows"] = 32768
    gate = adjudicate(rows)
    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["direct_repeat_equivalence"] is False
    assert gate["protocol_checks"]["scope_row_counts_exact"] is False


def test_k1af_proves_direct_repeat_equivalence_in_float64() -> None:
    task = task_map(read_tasks(PLAN))[(0, "virtual_slot_exact")]
    one_pair_model = _build_one_pair_model(task, "exact")
    fixture = np.random.default_rng(20260729).integers(
        0,
        2,
        size=(8, 256),
        dtype=np.uint8,
    )

    error = _direct_repeat_error(
        task=task,
        condition="exact",
        state=deepcopy(one_pair_model.state_dict()),
        one_pair_model=one_pair_model,
        fixture=fixture,
    )

    assert error <= REPLAY_TOLERANCE
    assert error <= 1e-12
    assert next(one_pair_model.parameters()).dtype == torch.float32


def test_k1af_plot_explains_single_pair_and_semantic_failure(tmp_path: Path) -> None:
    gate = adjudicate(_rows(exact_aucs=(0.80, 0.79), margins=(-0.004, -0.002)))
    output = tmp_path / "curves.svg"

    report = render_k1af_svg(gate, output)
    svg = output.read_text(encoding="utf-8")

    assert report["panels"] == 3
    assert "单 pair 解除了饱和" in svg
    assert "正确 - 错误 S盒" in svg
    assert "应用级多查询聚合" in svg
