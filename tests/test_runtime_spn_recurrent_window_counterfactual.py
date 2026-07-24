from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    uknit64_runtime_structure,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1 import (
    runtime_spn_recurrent_window_counterfactual as u4,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_recurrent_window_counterfactual import (
    FROZEN_MODEL_OPTIONS,
    U3_DECISION,
    U3_PLAN_SHA256,
    U3_RUN_ID,
    adjudicate_same_checkpoint_window_panel,
    adjudicate_window_counterfactual_source,
    evaluate_same_checkpoint_window_panel,
)


def _row(
    seed: int,
    condition: str,
    auc: float,
    probability_delta: float,
) -> dict[str, object]:
    window = {
        "full_correct": "1" * 64,
        "repeat_last": "2" * 64,
        "corrupted": "3" * 64,
        "no_topology": "1" * 64,
    }[condition]
    transitions = {
        "full_correct": ["a" * 64, "b" * 64],
        "repeat_last": ["b" * 64, "b" * 64],
        "corrupted": ["c" * 64, "d" * 64],
        "no_topology": ["a" * 64, "b" * 64],
    }[condition]
    intervention = {
        "full_correct": "4" * 64,
        "repeat_last": "5" * 64,
        "corrupted": "6" * 64,
        "no_topology": "7" * 64,
    }[condition]
    return {
        "seed": seed,
        "condition": condition,
        "relation_mode": "independent" if condition == "no_topology" else "true",
        "runtime_structure_transition_sha256s": transitions,
        "runtime_structure_window_sha256": window,
        "runtime_structure_unique_transition_count": len(set(transitions)),
        "runtime_structure_homogeneous": len(set(transitions)) == 1,
        "intervention_sha256": intervention,
        "auc": auc,
        "full_correct_minus_condition_auc": 0.0,
        "max_abs_probability_delta_from_full": probability_delta,
        "mean_probability": 0.5,
        "probability_sha256": ("8" if condition == "full_correct" else "9")
        * 64,
        "source_candidate_auc": 0.56,
        "full_correct_minus_source_auc": 0.0,
        "checkpoint_path": f"seed{seed}.pt",
        "checkpoint_sha256": ("a" if seed == 0 else "b") * 64,
        "checkpoint_selected": "best",
        "feature_path": f"seed{seed}_features.npy",
        "feature_sha256": ("c" if seed == 0 else "d") * 64,
        "label_path": f"seed{seed}_labels.npy",
        "label_sha256": ("e" if seed == 0 else "f") * 64,
        "metadata_path": f"seed{seed}_metadata.json",
        "metadata_sha256": ("0" if seed == 0 else "1") * 64,
        "descriptor_sha256": "2" * 64,
        "descriptor_round_start": 3,
        "descriptor_loaded_rounds": 2,
        "source_results_sha256": "3" * 64,
        "source_gate_sha256": "4" * 64,
        "source_validation_sha256": "5" * 64,
        "source_plan_validation_sha256": "6" * 64,
        "source_visual_qa_verified": True,
        "samples_total": 2048,
        "input_bits": 512,
        "pairs_per_sample": 4,
        "parameter_count": 442466,
        "model_options": deepcopy(FROZEN_MODEL_OPTIONS),
        "negative_mode": "encrypted_random_plaintexts",
        "strict_state_dict_load": True,
        "training_performed": False,
    }


def _passing_rows() -> list[dict[str, object]]:
    aucs = {
        "full_correct": 0.56,
        "repeat_last": 0.55,
        "corrupted": 0.54,
        "no_topology": 0.51,
    }
    return [
        _row(
            seed,
            condition,
            aucs[condition],
            0.0 if condition == "full_correct" else 0.01,
        )
        for seed in (0, 1)
        for condition in u4.EXPECTED_CONDITIONS
    ]


def test_u4_gate_passes_two_seed_same_checkpoint_panel() -> None:
    gate = adjudicate_same_checkpoint_window_panel(
        run_id="u4",
        rows=_passing_rows(),
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation1_runtime_spn_window_same_checkpoint_attribution_supported"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_u4_gate_holds_when_one_seed_does_not_use_earlier_window() -> None:
    rows = _passing_rows()
    repeat_last = next(
        row
        for row in rows
        if row["seed"] == 1 and row["condition"] == "repeat_last"
    )
    repeat_last["auc"] = 0.558

    gate = adjudicate_same_checkpoint_window_panel(run_id="u4-hold", rows=rows)

    assert gate["status"] == "hold"
    assert gate["protocol_checks"]["window_interventions_exact"] is True
    assert gate["research_checks"][
        "seed1_full_beats_repeat_last_by_0p005"
    ] is False


def test_u4_gate_fails_on_checkpoint_or_source_auc_replay_drift() -> None:
    rows = _passing_rows()
    rows[1]["checkpoint_sha256"] = "f" * 64
    rows[4]["full_correct_minus_source_auc"] = 1e-4

    gate = adjudicate_same_checkpoint_window_panel(run_id="u4-invalid", rows=rows)

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["same_checkpoint_within_seed"] is False
    assert gate["protocol_checks"]["full_auc_replays_u3_candidate"] is False


def test_u4_source_authorization_requires_exact_u3_and_visual_gate() -> None:
    persisted = {
        "run_id": U3_RUN_ID,
        "status": "pass",
        "decision": U3_DECISION,
        "protocol_checks": {"protocol": True},
        "research_checks": {"research": True},
    }
    validation = {
        "run_id": U3_RUN_ID,
        "status": "pass",
        "checks": persisted["protocol_checks"],
    }
    plan_validation = {
        "status": "pass",
        "expected_rows": 10,
        "result_rows": 10,
        "errors": [],
    }
    gate = adjudicate_window_counterfactual_source(
        run_id="u4-source",
        persisted_gate=persisted,
        replayed_gate=deepcopy(persisted),
        validation=validation,
        plan_validation=plan_validation,
        result_rows_count=10,
        candidate_rows_valid=True,
        candidate_checkpoints_exist=True,
        visual_qa_passed=True,
        plan_sha256=U3_PLAN_SHA256,
    )

    assert gate["status"] == "pass"
    assert gate["execution_authorized"] is True

    missing_visual = adjudicate_window_counterfactual_source(
        run_id="u4-source",
        persisted_gate=persisted,
        replayed_gate=deepcopy(persisted),
        validation=validation,
        plan_validation=plan_validation,
        result_rows_count=10,
        candidate_rows_valid=True,
        candidate_checkpoints_exist=True,
        visual_qa_passed=False,
        plan_sha256=U3_PLAN_SHA256,
    )
    assert missing_visual["status"] == "fail"
    assert missing_visual["execution_authorized"] is False


def test_u4_evaluation_uses_one_checkpoint_for_four_runtime_conditions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = build_model(
        "runtime_spn_e4_equivariant_true",
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=deepcopy(FROZEN_MODEL_OPTIONS),
    )
    checkpoint = tmp_path / "candidate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {"selected_checkpoint": "best"},
        },
        checkpoint,
    )
    labels = np.array([0] * 1024 + [1] * 1024, dtype=np.uint8)
    dataset = DifferentialDataset(
        features=np.zeros((2048, 512), dtype=np.uint8),
        labels=labels,
        metadata={},
    )

    def fake_probabilities(model, dataset, *, batch_size, device):
        del batch_size, device
        token = int(model.runtime_structure.window_sha256()[0], 16)
        if model.relation_mode == "independent":
            token += 17
        return np.asarray(dataset.labels, dtype=np.float64) * 0.6 + 0.1 + token * 1e-4

    monkeypatch.setattr(u4, "predict_binary_probabilities", fake_probabilities)
    source = {
        "seed": 0,
        "cipher_key": "uknit64",
        "rounds": 5,
        "model": "runtime_spn_e4_equivariant_true",
        "samples_per_class": 2048,
        "pairs_per_sample": 4,
        "negative_mode": "encrypted_random_plaintexts",
        "metrics": {"auc": 1.0},
        "validation": {"samples_per_class": 1024, "samples_total": 2048},
        "training": {
            "selected_checkpoint": "best",
            "restore_best_checkpoint": True,
            "validation_rows": 2048,
            "model_options": deepcopy(FROZEN_MODEL_OPTIONS),
        },
    }
    feature_path = tmp_path / "features.npy"
    label_path = tmp_path / "labels.npy"
    metadata_path = tmp_path / "metadata.json"

    rows = evaluate_same_checkpoint_window_panel(
        seed=0,
        source_row=source,
        checkpoint_path=checkpoint,
        dataset=dataset,
        correct_structure=uknit64_runtime_structure(2, round_start=3),
        checkpoint_sha256="a" * 64,
        feature_path=feature_path,
        feature_sha256="b" * 64,
        label_path=label_path,
        label_sha256="c" * 64,
        metadata_path=metadata_path,
        metadata_sha256="d" * 64,
        descriptor_sha256="e" * 64,
        source_hashes={
            "results": "1" * 64,
            "gate": "2" * 64,
            "validation": "3" * 64,
            "plan_validation": "4" * 64,
        },
    )

    assert [row["condition"] for row in rows] == list(u4.EXPECTED_CONDITIONS)
    assert {row["checkpoint_sha256"] for row in rows} == {"a" * 64}
    assert all(row["training_performed"] is False for row in rows)
    assert rows[0]["full_correct_minus_source_auc"] == 0.0
    assert rows[3]["relation_mode"] == "independent"


def test_u4_successor_is_local_only_and_waits_for_visual_qa() -> None:
    script = Path(
        "configs/remote/generated/monitor_i1_uknit_u4_after_u3_20260725.sh"
    ).read_text(encoding="utf-8")

    assert "audit-runtime-spn-recurrent-window-counterfactual" in script
    assert "u3_complete.marker" in script
    assert "visual_qa_passed.marker" in script
    assert "scripts/index-results" in script
    assert "ssh " not in script.lower()
    assert "scp " not in script.lower()
    assert "lxy-a6000" not in script.lower()
    assert "cmd.exe" not in script.lower()
