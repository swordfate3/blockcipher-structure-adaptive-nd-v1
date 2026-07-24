from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from blockcipher_nd.cli import run_runtime_spn_cross_cipher_head_adaptation as cli
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_head_adaptation import (
    EXPECTED_ROLES,
    FULL_TARGET_ANCHOR_AUCS,
    TRAINABLE_PARAMETER_COUNT,
    adjudicate_head_adaptation,
    deterministic_classifier_state,
    prepare_adaptation_model,
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_zero_step import (
    FROZEN_MODEL_OPTIONS,
)


def _row(seed: int, role: str, auc: float) -> dict[str, object]:
    source_role, target_mode = {
        "true_source_true_target": ("true", "true"),
        "corrupted_source_true_target": ("corrupted", "true"),
        "true_source_corrupted_target": ("true", "corrupted"),
        "random_source_true_target": ("random", "true"),
    }[role]
    true_source_hash = ("a" if seed == 0 else "b") * 64
    corrupted_source_hash = ("c" if seed == 0 else "d") * 64
    source_hash = {
        "true": true_source_hash,
        "corrupted": corrupted_source_hash,
        "random": None,
    }[source_role]
    true_structure_hash = "1" * 64
    structure_hash = true_structure_hash if target_mode == "true" else "2" * 64
    head_initial = "3" * 64
    history = [
        {
            "epoch": float(epoch),
            "train_loss": 0.25,
            "train_auc": 0.53 + 0.01 * epoch,
            "val_loss": 0.69,
            "val_auc": auc - 0.001 * (5 - epoch),
            "val_accuracy": 0.5,
            "learning_rate": 1e-4,
        }
        for epoch in range(1, 6)
    ]
    return {
        "seed": seed,
        "role": role,
        "source_role": source_role,
        "target_mode": target_mode,
        "source_checkpoint_sha256": source_hash,
        "source_selected_checkpoint": None if source_role == "random" else "best",
        "runtime_structure_sha256": structure_hash,
        "runtime_intervention_sha256": ("5" if target_mode == "true" else "6") * 64,
        "target_relation_mode": "true",
        "classifier_initial_sha256": head_initial,
        "classifier_final_sha256": ("7" + str(seed)) * 32,
        "backbone_initial_sha256": ("8" + str(seed)) * 32,
        "backbone_final_sha256": ("8" + str(seed)) * 32,
        "checkpoint_sha256": ("9" + str(seed)) * 32,
        "checkpoint_replay_verified": True,
        "strict_state_dict_load": True,
        "parameter_count": 442466,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "trainable_parameter_names": [
            "backbone.classifier.0.weight",
            "backbone.classifier.0.bias",
            "backbone.classifier.1.weight",
            "backbone.classifier.1.bias",
            "backbone.classifier.4.weight",
            "backbone.classifier.4.bias",
        ],
        "auc": auc,
        "accuracy": 0.55,
        "loss": 0.69,
        "history": history,
        "training": {
            "epochs": 5,
            "epochs_ran": 5,
            "batch_size": 256,
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "loss": "mse",
            "checkpoint_metric": "val_auc",
            "selected_checkpoint": "best",
        },
        "full_target_anchor_auc": FULL_TARGET_ANCHOR_AUCS[seed],
        "train_feature_sha256": ("a" + str(seed)) * 32,
        "train_label_sha256": ("b" + str(seed)) * 32,
        "train_metadata_sha256": ("c" + str(seed)) * 32,
        "validation_feature_sha256": ("d" + str(seed)) * 32,
        "validation_label_sha256": ("e" + str(seed)) * 32,
        "validation_metadata_sha256": ("f" + str(seed)) * 32,
        "source_cipher": "GIFT-64",
        "source_rounds": 6,
        "target_cipher": "SKINNY-64/64",
        "target_rounds": 7,
        "target_difference": 0x2000,
        "target_train_key": 0,
        "target_validation_key": 0x1111111111111111,
        "train_rows": 4096,
        "validation_rows": 2048,
        "pairs_per_sample": 4,
        "input_bits": 512,
        "negative_mode": "encrypted_random_plaintexts",
        "model_options": FROZEN_MODEL_OPTIONS,
        "backbone_frozen": True,
    }


def _passing_rows() -> list[dict[str, object]]:
    aucs = {
        "true_source_true_target": 0.60,
        "corrupted_source_true_target": 0.58,
        "true_source_corrupted_target": 0.57,
        "random_source_true_target": 0.54,
    }
    return [_row(seed, role, aucs[role]) for seed in (0, 1) for role in EXPECTED_ROLES]


def _source_state(model: str) -> dict[str, torch.Tensor]:
    source = build_model(
        model,
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=FROZEN_MODEL_OPTIONS,
    )
    return source.state_dict()


def test_prepare_adaptation_model_freezes_everything_except_classifier() -> None:
    model = prepare_adaptation_model(
        seed=0,
        source_role="true",
        target_mode="true",
        source_state_dicts={
            "true": _source_state("gift64_runtime_e4_equivariant_true"),
            "corrupted": _source_state("gift64_runtime_e4_equivariant_corrupted"),
        },
        classifier_state=deterministic_classifier_state(),
    )

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("backbone.classifier.") for name in trainable)
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        == TRAINABLE_PARAMETER_COUNT
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 442466

    features = torch.randint(0, 2, (2, 512), dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0])
    loss = torch.nn.functional.mse_loss(
        torch.sigmoid(model(features).squeeze(1)), labels
    )
    loss.backward()
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.classifier.")
    )
    assert all(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.classifier.")
    )


def test_classifier_initialization_is_deterministic_and_role_independent() -> None:
    state = deterministic_classifier_state()
    assert tensor_mapping_sha256(state) == tensor_mapping_sha256(
        deterministic_classifier_state()
    )
    source_states = {
        "true": _source_state("gift64_runtime_e4_equivariant_true"),
        "corrupted": _source_state("gift64_runtime_e4_equivariant_corrupted"),
    }
    hashes = set()
    for role, target in (
        ("true", "true"),
        ("corrupted", "true"),
        ("true", "corrupted"),
        ("random", "true"),
    ):
        model = prepare_adaptation_model(
            seed=0,
            source_role=role,
            target_mode=target,
            source_state_dicts=source_states,
            classifier_state=state,
        )
        hashes.add(tensor_mapping_sha256(model.backbone.classifier.state_dict()))
    assert hashes == {tensor_mapping_sha256(state)}


def test_x2_gate_passes_complete_two_seed_attribution_panel() -> None:
    gate = adjudicate_head_adaptation(run_id="x2", rows=_passing_rows())

    assert gate["status"] == "pass"
    assert gate["decision"] == "runtime_spn_frozen_backbone_target_head_supported"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_x2_gate_holds_when_candidate_loses_source_margin() -> None:
    rows = _passing_rows()
    rows[1]["auc"] = 0.598

    gate = adjudicate_head_adaptation(run_id="x2-hold", rows=rows)

    assert gate["status"] == "hold"
    assert gate["decision"] == "runtime_spn_target_head_signal_unstable"
    assert gate["research_checks"]["seed0_beats_source_by_0p005"] is False


def test_x2_gate_fails_if_frozen_backbone_changes() -> None:
    rows = deepcopy(_passing_rows())
    rows[0]["backbone_final_sha256"] = "0" * 64

    gate = adjudicate_head_adaptation(run_id="x2-fail", rows=rows)

    assert gate["status"] == "fail"
    assert gate["decision"] == "runtime_spn_target_head_protocol_invalid"
    assert gate["protocol_checks"]["frozen_backbone_unchanged"] is False


def test_x2_dependency_gate_requires_completed_rtg2b_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="remains closed"):
        cli._validate_dependency_gate(tmp_path / "missing.json")

    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "phase": "rtg2b",
                "status": "pass",
                "decision": "innovation1_rtg2b_skinny_scale_two_seed_supported",
                "protocol_checks": {"complete": True},
                "research_checks": {"supported": True},
            }
        ),
        encoding="utf-8",
    )
    assert cli._validate_dependency_gate(path)["status"] == "pass"


def test_x2_plot_uses_plain_language_labels(tmp_path: Path) -> None:
    rows = _passing_rows()
    gate = adjudicate_head_adaptation(run_id="x2-plot", rows=rows)
    output = tmp_path / "curves.svg"

    cli.render_head_adaptation_svg(gate, rows, output)

    svg = output.read_text(encoding="utf-8")
    assert "冻结 GIFT 结构主干，只训练 SKINNY 输出头" in svg
    assert "正确源=GIFT正确拓扑最佳主干" in svg
    assert "全量目标锚点" in svg
