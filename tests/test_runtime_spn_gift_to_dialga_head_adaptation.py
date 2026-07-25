from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from blockcipher_nd.cli.run_runtime_spn_gift_to_dialga_head_adaptation import (
    render_head_adaptation_svg,
)
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_zero_step import (
    FROZEN_MODEL_OPTIONS,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_gift_to_dialga_head_adaptation import (
    EXPECTED_ROLES,
    FULL_TARGET_ANCHOR_AUCS,
    TRAINABLE_PARAMETER_COUNT,
    adjudicate_head_adaptation,
    adjudicate_readiness,
    audit_role_readiness,
    audit_strict_load_matrix,
    deterministic_classifier_state,
    prepare_adaptation_model,
    target_model_options,
    tensor_mapping_sha256,
)


def _source_state(model_name: str) -> dict[str, torch.Tensor]:
    model = build_model(
        model_name,
        input_bits=512,
        hidden_bits=64,
        pair_bits=128,
        structure="SPN",
        model_options=FROZEN_MODEL_OPTIONS,
    )
    return model.state_dict()


def _source_states() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "true": _source_state("gift64_runtime_e4_equivariant_true"),
        "corrupted": _source_state("gift64_runtime_e4_equivariant_corrupted"),
    }


def test_x3_strict_load_crosses_block_size_without_adapter() -> None:
    states = _source_states()
    payloads = {
        seed: {role: {"state_dict": state} for role, state in states.items()}
        for seed in (0, 1)
    }

    audits = audit_strict_load_matrix(payloads)

    assert len(audits) == 8
    assert all(row["strict_load"] is True for row in audits)
    assert {row["parameter_count"] for row in audits} == {442466}
    assert {row["state_dict_key_count"] for row in audits} == {54}
    assert {(row["source_role"], row["target_mode"]) for row in audits} == {
        ("true", "true"),
        ("true", "corrupted"),
        ("corrupted", "true"),
        ("corrupted", "corrupted"),
    }


def test_x3_model_freezes_everything_except_existing_classifier() -> None:
    model = prepare_adaptation_model(
        seed=0,
        source_role="true",
        target_mode="true",
        source_state_dicts=_source_states(),
        classifier_state=deterministic_classifier_state(),
    )

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("backbone.classifier.") for name in trainable)
    assert sum(parameter.numel() for parameter in model.parameters()) == 442466
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        == TRAINABLE_PARAMETER_COUNT
    )
    assert model.runtime_structure.block_bits == 128


def test_x3_readiness_real_batch_preserves_frozen_hash() -> None:
    features = torch.randint(0, 2, (4, 1024), dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    audit = audit_role_readiness(
        seed=0,
        role="true_source_true_target",
        source_state_dicts=_source_states(),
        classifier_state=deterministic_classifier_state(),
        features=features,
        labels=labels,
    )

    assert audit["representation_shape"] == [4, 384]
    assert audit["representation_finite"] is True
    assert audit["representation_nonconstant"] is True
    assert audit["classifier_gradients_finite"] is True
    assert audit["classifier_gradient_l1"] > 0.0
    assert audit["frozen_gradients_absent"] is True
    assert audit["backbone_unchanged_after_step"] is True
    assert audit["classifier_changed_after_step"] is True


def test_x3_classifier_initialization_is_role_independent() -> None:
    classifier = deterministic_classifier_state()
    expected = tensor_mapping_sha256(classifier)
    states = _source_states()
    hashes = set()
    for source_role, target_mode in (
        ("true", "true"),
        ("corrupted", "true"),
        ("true", "corrupted"),
        ("random", "true"),
    ):
        model = prepare_adaptation_model(
            seed=0,
            source_role=source_role,
            target_mode=target_mode,
            source_state_dicts=states,
            classifier_state=classifier,
        )
        hashes.add(tensor_mapping_sha256(model.backbone.classifier.state_dict()))
    assert hashes == {expected}


def _result_row(seed: int, role: str, auc: float) -> dict[str, object]:
    source_role, target_mode = {
        "true_source_true_target": ("true", "true"),
        "corrupted_source_true_target": ("corrupted", "true"),
        "true_source_corrupted_target": ("true", "corrupted"),
        "random_source_true_target": ("random", "true"),
    }[role]
    source_hash = {
        "true": ("a" if seed == 0 else "b") * 64,
        "corrupted": ("c" if seed == 0 else "d") * 64,
        "random": None,
    }[source_role]
    structure_hash = "1" * 64 if target_mode == "true" else "2" * 64
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
        "runtime_intervention_sha256": ("3" if target_mode == "true" else "4") * 64,
        "target_relation_mode": "true",
        "classifier_initial_sha256": "5" * 64,
        "classifier_final_sha256": ("6" + str(seed)) * 32,
        "backbone_initial_sha256": ("7" + str(seed)) * 32,
        "backbone_final_sha256": ("7" + str(seed)) * 32,
        "checkpoint_sha256": ("8" + str(seed)) * 32,
        "checkpoint_replay_verified": True,
        "readiness_sha256": "9" * 64,
        "target_cache_tree_sha256_before": "a" * 64,
        "target_cache_tree_sha256_after": "a" * 64,
        "target_cache_unchanged": True,
        "strict_state_dict_load": None if source_role == "random" else True,
        "parameter_count": 442466,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "trainable_parameter_names": [
            "backbone.classifier.0.weight",
            "backbone.classifier.0.bias",
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
        "train_feature_sha256": ("b" + str(seed)) * 32,
        "train_label_sha256": ("c" + str(seed)) * 32,
        "train_metadata_sha256": ("d" + str(seed)) * 32,
        "validation_feature_sha256": ("e" + str(seed)) * 32,
        "validation_label_sha256": ("f" + str(seed)) * 32,
        "validation_metadata_sha256": ("0" + str(seed)) * 32,
        "source_cipher": "GIFT-64",
        "source_rounds": 6,
        "target_cipher": "Dialga-128",
        "target_rounds": 4,
        "target_difference": 0x40,
        "target_train_key": 0,
        "target_validation_key": int("11" * 32, 16),
        "train_rows": 4096,
        "validation_rows": 2048,
        "pairs_per_sample": 4,
        "input_bits": 1024,
        "pair_bits": 256,
        "negative_mode": "encrypted_random_plaintexts",
        "source_model_options": FROZEN_MODEL_OPTIONS,
        "target_model_options": target_model_options(target_mode),
        "backbone_frozen": True,
    }


def _passing_rows() -> list[dict[str, object]]:
    aucs = {
        "true_source_true_target": 0.60,
        "corrupted_source_true_target": 0.58,
        "true_source_corrupted_target": 0.57,
        "random_source_true_target": 0.54,
    }
    return [
        _result_row(seed, role, aucs[role])
        for seed in (0, 1)
        for role in EXPECTED_ROLES
    ]


def test_x3_gate_passes_complete_two_seed_panel() -> None:
    gate = adjudicate_head_adaptation(run_id="x3", rows=_passing_rows())

    assert gate["status"] == "pass"
    assert gate["decision"] == "runtime_spn_gift_to_dialga_x3_shared_backbone_supported"
    assert all(gate["protocol_checks"].values())
    assert all(gate["research_checks"].values())


def test_x3_gate_holds_when_one_seed_loses_target_margin() -> None:
    rows = _passing_rows()
    rows[2]["auc"] = 0.598

    gate = adjudicate_head_adaptation(run_id="x3-hold", rows=rows)

    assert gate["status"] == "hold"
    assert gate["research_checks"]["seed0_beats_target_by_0p005"] is False


def test_x3_gate_fails_if_target_cache_changes() -> None:
    rows = deepcopy(_passing_rows())
    rows[0]["target_cache_unchanged"] = False

    gate = adjudicate_head_adaptation(run_id="x3-fail", rows=rows)

    assert gate["status"] == "fail"
    assert gate["protocol_checks"]["target_cache_unchanged"] is False


def test_x3_readiness_gate_requires_complete_strict_matrix() -> None:
    role_audits = []
    for seed in (0, 1):
        for role in EXPECTED_ROLES:
            target_mode = (
                "corrupted" if role == "true_source_corrupted_target" else "true"
            )
            role_audits.append(
                {
                    "seed": seed,
                    "role": role,
                    "target_mode": target_mode,
                    "strict_state_dict_load": True,
                    "parameter_count": 442466,
                    "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
                    "trainable_parameter_names": ["backbone.classifier.0.weight"],
                    "classifier_initial_sha256": "1" * 64,
                    "classifier_final_sha256": "2" * 64,
                    "backbone_initial_sha256": "3" * 64,
                    "backbone_final_sha256": "3" * 64,
                    "runtime_structure_sha256": (
                        "4" * 64 if target_mode == "true" else "5" * 64
                    ),
                    "runtime_intervention_sha256": "6" * 64,
                    "target_relation_mode": "true",
                    "representation_shape": [4, 384],
                    "representation_finite": True,
                    "representation_nonconstant": True,
                    "logits_finite": True,
                    "logits_nonconstant": True,
                    "classifier_gradients_finite": True,
                    "classifier_gradient_l1": 1.0,
                    "frozen_gradients_absent": True,
                    "loss_finite": True,
                    "backbone_unchanged_after_step": True,
                    "classifier_changed_after_step": True,
                }
            )
    source_evidence = {
        f"seed{seed}": {
            "roles": {
                "true": {"checkpoint_sha256": ("a" + str(seed)) * 32},
                "corrupted": {"checkpoint_sha256": ("b" + str(seed)) * 32},
            }
        }
        for seed in (0, 1)
    }
    d1_evidence = {
        "gate_replay_exact": True,
        "validation_replay_exact": True,
        "decision": "innovation1_dialga_runtime_e4_d1_two_seed_supported",
    }
    cache = {"leaf_count": 4, "geometry_exact": True, "unchanged": True}
    strict = [
        {
            "seed": seed,
            "source_role": source,
            "target_mode": target,
            "strict_load": True,
            "parameter_count": 442466,
            "state_dict_key_count": 54,
            "source_state_dict_sha256": "c" * 64,
        }
        for seed in (0, 1)
        for source in ("true", "corrupted")
        for target in ("true", "corrupted")
    ]

    passed = adjudicate_readiness(
        run_id="x3",
        role_audits=role_audits,
        source_evidence=source_evidence,
        d1_evidence=d1_evidence,
        cache_evidence=cache,
        strict_load_audits=strict,
    )
    failed = adjudicate_readiness(
        run_id="x3",
        role_audits=role_audits,
        source_evidence=source_evidence,
        d1_evidence=d1_evidence,
        cache_evidence=cache,
        strict_load_audits=strict[:-1],
    )

    assert passed["status"] == "pass"
    assert all(passed["checks"].values())
    assert failed["status"] == "fail"
    assert failed["checks"]["strict_cross_cipher_state_load"] is False


def test_x3_plot_uses_plain_language_and_separate_margin_panel(
    tmp_path: Path,
) -> None:
    rows = _passing_rows()
    gate = adjudicate_head_adaptation(run_id="x3-plot", rows=rows)
    output = tmp_path / "curves.svg"

    render_head_adaptation_svg(gate, rows, output)

    svg = output.read_text(encoding="utf-8")
    assert "把 GIFT 结构主干迁移到 Dialga，只训练分类头" in svg
    assert "结构归因边际（通过门 +0.005，越高越好）" in svg
    assert "不参与X3通过门" in svg
