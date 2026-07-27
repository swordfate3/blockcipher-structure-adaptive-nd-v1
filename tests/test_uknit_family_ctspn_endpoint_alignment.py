from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_endpoint_alignment import (
    K1_DECISION,
    RUN_ID,
    adjudicate_endpoint_alignment,
    frozen_k1_stages,
    native_endpoint_signature,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    RUN_ID as K1_RUN_ID,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1.csv"
)


def test_dialga_k1_pooling_erases_changed_native_endpoint_schedule() -> None:
    tasks = tasks_from_plan(
        PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )
    task = next(
        row
        for row in tasks
        if row["cipher_key"] == "dialga128"
        and row["seed"] == 0
        and row["model_key"] == "runtime_spn_ct_k1_canonical_true"
    )
    common = dict(task["model_options"])
    correct = build_model(
        "runtime_spn_ct_k1_canonical_true",
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options=common,
    )
    rotated = build_model(
        "runtime_spn_ct_k1_canonical_true",
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options={**common, "canonical_schedule_control": "rotated"},
    )
    rotated.load_state_dict(correct.state_dict(), strict=True)
    correct.eval()
    rotated.eval()
    endpoint_delta = (
        native_endpoint_signature(correct) != native_endpoint_signature(rotated)
    ).any(dim=-1)
    features = torch.randint(
        0,
        2,
        (4, 1024),
        generator=torch.Generator().manual_seed(20260728),
    ).to(torch.float32)
    with torch.inference_mode():
        correct_views, correct_summary, correct_logits = frozen_k1_stages(
            correct, features
        )
        rotated_views, rotated_summary, rotated_logits = frozen_k1_stages(
            rotated, features
        )

    assert float(endpoint_delta.float().mean()) >= 0.95
    assert float((correct_views - rotated_views).abs().max()) > 0.0
    assert float((correct_summary - rotated_summary).abs().max()) <= 1e-5
    assert float((correct_logits - rotated_logits).abs().max()) <= 1e-4


def test_k1b_native_endpoint_channels_retain_dialga_schedule_identity() -> None:
    common = {
        "runtime_rounds": 2,
        "processor_steps": 2,
        "pair_embedding_dim": 128,
        "temporal_hidden_dim": 76,
        "dropout": 0.0,
        "runtime_structure_window_control": "full",
        "canonical_schedule_control": "ordered",
    }
    uknit = build_model(
        "runtime_spn_ct_k1b_endpoint_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options={
            **common,
            "runtime_structure_path": "configs/runtime/spn/uknit64.json",
            "runtime_round_start": 3,
        },
    )
    dialga = build_model(
        "runtime_spn_ct_k1b_endpoint_true",
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options={
            **common,
            "runtime_structure_path": "configs/runtime/spn/dialga128.json",
            "runtime_round_start": 2,
        },
    )
    rotated = build_model(
        "runtime_spn_ct_k1b_endpoint_true",
        input_bits=1024,
        hidden_bits=64,
        pair_bits=256,
        structure="SPN",
        model_options={
            **common,
            "runtime_structure_path": "configs/runtime/spn/dialga128.json",
            "runtime_round_start": 2,
            "canonical_schedule_control": "rotated",
        },
    )
    dialga.load_state_dict(uknit.state_dict(), strict=True)
    rotated.load_state_dict(dialga.state_dict(), strict=True)
    features = torch.randint(
        0,
        2,
        (4, 1024),
        generator=torch.Generator().manual_seed(20260728),
    ).to(torch.float32)
    dialga.eval()
    rotated.eval()
    with torch.inference_mode():
        correct_views, correct_summary, correct_logits = frozen_k1_stages(
            dialga, features
        )
        rotated_views, rotated_summary, rotated_logits = frozen_k1_stages(
            rotated, features
        )

    assert dialga.backbone.edge_encoder[0].in_features == 22
    assert rotated.backbone.edge_encoder[0].in_features == 22
    assert dialga.canonical_endpoint_identity_mode == "native_cell_role"
    assert correct_views.shape[-1] == 22
    assert float((correct_views - rotated_views).abs().max()) > 0.0
    assert float((correct_summary - rotated_summary).abs().max()) > 1e-5
    assert float((correct_logits - rotated_logits).abs().max()) > 1e-6


def test_endpoint_alignment_gate_requires_both_seeds_and_controls() -> None:
    results = _supported_rows()
    gate = adjudicate_endpoint_alignment(
        k1_gate=_k1_gate(),
        results=results,
        task_keys=_cipher_seed_keys(),
        checkpoint_keys=_cipher_seed_keys(),
        strict_loads=[True] * 12,
        state_hashes_match=[True] * 12,
        edge_input_widths=[12] * 4,
    )

    assert gate["run_id"] == RUN_ID
    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = [dict(row) for row in results]
    target = next(
        row
        for row in failed
        if row["cipher_key"] == "dialga128"
        and row["seed"] == 1
        and row["condition"] == "rotated"
    )
    target["transition_summary_max_abs_delta"] = 0.01
    held = adjudicate_endpoint_alignment(
        k1_gate=_k1_gate(),
        results=failed,
        task_keys=_cipher_seed_keys(),
        checkpoint_keys=_cipher_seed_keys(),
        strict_loads=[True] * 12,
        state_hashes_match=[True] * 12,
        edge_input_widths=[12] * 4,
    )

    assert held["status"] == "hold"
    assert (
        held["research_checks"][
            "dialga128_seed1_rotated_pooled_summary_collapses"
        ]
        is False
    )


def test_endpoint_alignment_gate_fails_closed_on_position_aware_source() -> None:
    gate = adjudicate_endpoint_alignment(
        k1_gate=_k1_gate(),
        results=_supported_rows(),
        task_keys=_cipher_seed_keys(),
        checkpoint_keys=_cipher_seed_keys(),
        strict_loads=[True] * 12,
        state_hashes_match=[True] * 12,
        edge_input_widths=[22] * 4,
    )

    assert gate["status"] == "invalid"
    assert gate["protocol_checks"]["edge_tokens_have_no_endpoint_identity"] is False


def _cipher_seed_keys() -> set[tuple[str, int]]:
    return {(cipher, seed) for cipher in ("uknit64", "dialga128") for seed in (0, 1)}


def _k1_gate() -> dict[str, object]:
    protocol = {"complete": True}
    seed_results = {
        "uknit64": {
            str(seed): {
                "candidate_auc": 0.50,
                "candidate_minus_repeat_last": 0.002,
                "candidate_minus_rotated": 0.003,
            }
            for seed in (0, 1)
        },
        "dialga128": {
            str(seed): {
                "candidate_auc": 0.963,
                "candidate_minus_repeat_last": 0.0,
                "candidate_minus_rotated": 0.0,
            }
            for seed in (0, 1)
        },
    }
    return {
        "run_id": K1_RUN_ID,
        "status": "hold",
        "decision": K1_DECISION,
        "protocol_checks": protocol,
        "seed_results": seed_results,
    }


def _supported_rows() -> list[dict[str, object]]:
    rows = []
    for cipher in ("uknit64", "dialga128"):
        for seed in (0, 1):
            for condition in ("repeat_last", "rotated"):
                rows.append(
                    {
                        "cipher_key": cipher,
                        "seed": seed,
                        "condition": condition,
                        "training_performed": False,
                        "optimizer_steps": 0,
                        "native_endpoint_fraction_changed": 0.49
                        if condition == "repeat_last"
                        else 0.98,
                        "edge_value_max_abs_delta": 1.0,
                        "edge_value_mean_abs_delta": 0.2,
                        "transition_summary_max_abs_delta": 1e-6
                        if cipher == "dialga128"
                        else 0.2,
                        "transition_summary_mean_abs_delta": 1e-7
                        if cipher == "dialga128"
                        else 0.02,
                        "logit_max_abs_delta": 1e-6
                        if cipher == "dialga128"
                        else 0.002,
                        "logit_mean_abs_delta": 1e-7
                        if cipher == "dialga128"
                        else 0.0002,
                        "source_candidate_auc": 0.50
                        if cipher == "uknit64"
                        else 0.963,
                        "source_candidate_minus_control_auc": 0.0,
                    }
                )
    return rows
