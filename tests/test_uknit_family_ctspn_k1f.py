from __future__ import annotations

from pathlib import Path

import torch

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.cli.plot_uknit_family_ctspn_k1 import render_ctspn_k1f_svg
from blockcipher_nd.planning.matrix import tasks_from_plan
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1e import (
    RUN_ID as K1E_RUN_ID,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1f import (
    CANDIDATE_MODEL,
    CONTROL_CONDITIONS,
    EXPECTED_PARAMETER_COUNT,
    READINESS_RUN_ID,
    adjudicate_k1f,
    build_k1f_control,
    build_k1f_readiness,
    frozen_hypergraph_stages,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1d import (
    sorted_path_token_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_relative_path_k1d_2048_seed0_seed1.csv"
)
TRAINING_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_cell_path_hypergraph_k1f_2048_seed0_seed1.csv"
)


def _tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        SOURCE_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def _training_tasks() -> list[dict[str, object]]:
    return tasks_from_plan(
        TRAINING_PLAN,
        feature_encoding="ciphertext_pair_bits",
        pairs_per_sample=4,
        difference_profile=None,
        difference_member=0,
    )


def _k1e_gate() -> dict[str, object]:
    return {
        "run_id": K1E_RUN_ID,
        "status": "pass",
        "decision": (
            "innovation1_uknit_family_ctspn_k1e_"
            "split_specific_relative_path_overfit_confirmed"
        ),
        "protocol_checks": {"source_binding": True},
    }


def test_k1f_real_source_plan_passes_zero_training_readiness() -> None:
    manifests, gate = build_k1f_readiness(
        source_tasks=_tasks(),
        k1e_gate=_k1e_gate(),
    )

    assert len(manifests) == 4
    assert gate["status"] == "pass"
    assert gate["optimizer_step_authorized"] is True
    assert gate["training_rows"] == 0
    assert gate["optimizer_steps"] == 0
    assert all(gate["protocol_checks"].values())
    assert all(gate["evidence_checks"].values())
    assert {row["trainable_parameter_count"] for row in manifests} == {
        EXPECTED_PARAMETER_COUNT
    }


def test_k1f_incidence_shuffle_preserves_tokens_but_changes_messages() -> None:
    task = next(
        row for row in _tasks() if row["cipher_key"] == "uknit64" and row["seed"] == 0
    )
    correct = build_k1f_control(
        task=task,
        condition="correct_ordered",
        input_bits=512,
    )
    shuffled = build_k1f_control(
        task=task,
        condition="incidence_shuffled",
        input_bits=512,
    )
    shuffled.load_state_dict(correct.state_dict(), strict=True)
    features = torch.randint(
        0, 2, (3, 512), generator=torch.Generator().manual_seed(20260728)
    ).float()

    with torch.inference_mode():
        correct_views, correct_pooled, correct_logits, _ = frozen_hypergraph_stages(
            correct, features
        )
        shuffled_views, shuffled_pooled, shuffled_logits, _ = frozen_hypergraph_stages(
            shuffled, features
        )

    assert sorted_path_token_sha256(correct_views) == sorted_path_token_sha256(
        shuffled_views
    )
    assert correct.cell_path_routing_sha256 != shuffled.cell_path_routing_sha256
    assert float((correct_pooled - shuffled_pooled).abs().max()) > 1e-6
    assert float((correct_logits - shuffled_logits).abs().max()) > 1e-7


def test_k1f_geometry_is_shared_across_64_and_128_bit_ciphers() -> None:
    tasks = {(str(row["cipher_key"]), int(row["seed"])): row for row in _tasks()}
    uknit = build_k1f_control(
        task=tasks[("uknit64", 0)],
        condition="correct_ordered",
        input_bits=512,
    )
    dialga = build_k1f_control(
        task=tasks[("dialga128", 0)],
        condition="correct_ordered",
        input_bits=1024,
    )

    assert (
        model_metadata(uknit)["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
    )
    assert [
        (name, tuple(value.shape)) for name, value in uknit.state_dict().items()
    ] == [(name, tuple(value.shape)) for name, value in dialga.state_dict().items()]
    dialga.load_state_dict(uknit.state_dict(), strict=True)


def test_k1f_readiness_fails_closed_without_k1e_evidence() -> None:
    manifests, gate = build_k1f_readiness(source_tasks=_tasks(), k1e_gate={})

    assert manifests == []
    assert gate["status"] == "fail"
    assert gate["optimizer_step_authorized"] is False
    assert gate["protocol_checks"]["k1e_relative_path_overfit_confirmed"] is False


def test_k1f_training_gate_requires_incidence_control_per_seed() -> None:
    training_rows, control_rows = _passing_training_evidence()
    gate = adjudicate_k1f(
        tasks=_training_tasks(),
        training_rows=training_rows,
        control_rows=control_rows,
        readiness_gate=_readiness_gate(),
        k1d_gate=_k1d_gate(),
        k1e_gate=_k1e_gate(),
    )

    assert gate["status"] == "pass"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())

    failed = [dict(row) for row in control_rows]
    target = next(
        row
        for row in failed
        if row["cipher_key"] == "uknit64"
        and row["seed"] == 1
        and row["condition"] == "incidence_shuffled"
    )
    target["auc"] = 0.56
    held = adjudicate_k1f(
        tasks=_training_tasks(),
        training_rows=training_rows,
        control_rows=failed,
        readiness_gate=_readiness_gate(),
        k1d_gate=_k1d_gate(),
        k1e_gate=_k1e_gate(),
    )

    assert held["status"] == "hold"
    assert held["research_checks"]["uknit64_seed1_beats_incidence_shuffled"] is False


def test_k1f_plot_explains_shared_relation_control(tmp_path: Path) -> None:
    training_rows, control_rows = _passing_training_evidence()
    gate = adjudicate_k1f(
        tasks=_training_tasks(),
        training_rows=training_rows,
        control_rows=control_rows,
        readiness_gate=_readiness_gate(),
        k1d_gate=_k1d_gate(),
        k1e_gate=_k1e_gate(),
    )
    output = tmp_path / "curves.svg"

    render_ctspn_k1f_svg(gate, output)

    svg = output.read_text(encoding="utf-8")
    assert "保留共享 cell 关系能否改善" in svg
    assert "cell 编号只负责路由" in svg
    assert "共享关系打乱" in svg


def _readiness_gate() -> dict[str, object]:
    return {
        "run_id": READINESS_RUN_ID,
        "status": "pass",
        "optimizer_step_authorized": True,
        "protocol_checks": {"source": True},
        "evidence_checks": {"invariance": True},
    }


def _k1d_gate() -> dict[str, object]:
    seed_results = {
        cipher: {
            str(seed): {
                "candidate_auc": 0.52 if cipher == "uknit64" else 0.95,
                "anchor_auc": 0.51 if cipher == "uknit64" else 0.95,
            }
            for seed in (0, 1)
        }
        for cipher in ("uknit64", "dialga128")
    }
    return {
        "status": "hold",
        "decision": "innovation1_uknit_family_ctspn_k1d_relative_path_not_supported",
        "protocol_checks": {"source": True},
        "seed_results": seed_results,
    }


def _passing_training_evidence() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
    training_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for cipher in ("uknit64", "dialga128"):
        for seed in (0, 1):
            dataset = f"dataset-{cipher}-{seed}"
            state = f"state-{cipher}-{seed}"
            source_auc = 0.54 if cipher == "uknit64" else 0.96
            training_rows.append(
                {
                    "model": CANDIDATE_MODEL,
                    "cipher_key": cipher,
                    "seed": seed,
                    "samples_per_class": 2048,
                    "pairs_per_sample": 4,
                    "negative_mode": "encrypted_random_plaintexts",
                    "sample_structure": "independent_pairs",
                    "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
                    "metrics": {"auc": source_auc},
                    "training": {
                        "batch_size": 64,
                        "epochs": 10,
                        "checkpoint_metric": "val_auc",
                        "selected_checkpoint": "best",
                        "train_rows": 4096,
                        "validation_rows": 2048,
                    },
                }
            )
            for condition in CONTROL_CONDITIONS:
                auc = (
                    source_auc if condition == "correct_ordered" else source_auc - 0.01
                )
                control_rows.append(
                    {
                        "run_id": (
                            "i1_uknit_family_ctspn_cell_path_hypergraph_"
                            "k1f_2048_seed0_seed1_20260728"
                        ),
                        "cipher_key": cipher,
                        "seed": seed,
                        "condition": condition,
                        "auc": auc,
                        "source_auc": source_auc,
                        "correct_minus_source_auc": 0.0,
                        "correct_minus_condition_auc": (
                            0.0 if condition == "correct_ordered" else 0.01
                        ),
                        "max_abs_probability_delta_from_correct": 0.0,
                        "mean_abs_probability_delta_from_correct": 0.0,
                        "dataset_sha256": dataset,
                        "prior_k1d_dataset_sha256": dataset,
                        "checkpoint_sha256": f"checkpoint-{cipher}-{seed}",
                        "state_dict_sha256": state,
                        "checkpoint_selected": "best",
                        "checkpoint_metric": "val_auc",
                        "incidence_mode": (
                            "shuffled" if condition == "incidence_shuffled" else "true"
                        ),
                        "routing_sha256": (
                            f"routing-shuffled-{cipher}-{seed}"
                            if condition == "incidence_shuffled"
                            else f"routing-true-{cipher}-{seed}"
                        ),
                        "strict_state_dict_load": True,
                        "training_performed": False,
                        "optimizer_steps": 0,
                    }
                )
    return training_rows, control_rows
