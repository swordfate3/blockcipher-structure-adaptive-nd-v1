from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import torch

from blockcipher_nd.cli.plot_uknit_family_multicipher_shared_weight_k1ao import (
    render_k1ao_svg,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1 import (
    tensor_mapping_sha256,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_STATE_ENTRIES,
    adjudicate_readiness,
    build_runtime_model,
    load_and_validate_config,
)


def test_k1ao_config_freezes_zero_training_three_cipher_protocol() -> None:
    config = load_and_validate_config()

    assert config["audit"] == {
        "training_rows": 0,
        "validation_rows": 0,
        "optimizer_steps": 0,
        "data_generation": False,
        "remote": False,
        "pairs_per_sample": 4,
    }
    assert tuple(row["cipher_key"] for row in config["ciphers"]) == EXPECTED_CIPHERS
    assert config["next_training"]["replicas"] == {
        "0": {"uknit64": 3, "midori64": 6, "dialga128": 0},
        "1": {"uknit64": 4, "midori64": 7, "dialga128": 1},
    }


def test_k1ao_same_seed_models_share_exact_geometry_and_state() -> None:
    config = load_and_validate_config()
    models = []
    for cipher in config["ciphers"]:
        torch.manual_seed(config["model"]["initialization_seed"])
        models.append(build_runtime_model(cipher, config["model"]))

    geometries = {
        tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())
        for model in models
    }
    hashes = {tensor_mapping_sha256(model.state_dict()) for model in models}

    assert len(geometries) == 1
    assert len(hashes) == 1
    assert {sum(parameter.numel() for parameter in model.parameters()) for model in models} == {
        EXPECTED_PARAMETER_COUNT
    }
    assert {len(model.state_dict()) for model in models} == {EXPECTED_STATE_ENTRIES}


def test_k1ao_one_backbone_switches_all_runtime_shapes_without_state_mutation() -> None:
    config = load_and_validate_config()
    correct = {}
    wrong = {}
    for cipher in config["ciphers"]:
        torch.manual_seed(29)
        correct[cipher["cipher_key"]] = build_runtime_model(cipher, config["model"])
        torch.manual_seed(29)
        wrong[cipher["cipher_key"]] = build_runtime_model(
            cipher,
            config["model"],
            wrong_sbox=True,
        )
    shared = correct["uknit64"]
    state_before = tensor_mapping_sha256(shared.state_dict())

    for cipher in config["ciphers"]:
        cipher_key = cipher["cipher_key"]
        features = torch.randint(
            0,
            2,
            (3, cipher["input_bits"]),
            dtype=torch.float32,
        )
        exact = shared.logits_with_runtime(
            features,
            correct[cipher_key].runtime_structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
        )
        wrong_logits = shared.logits_with_runtime(
            features,
            wrong[cipher_key].runtime_structure,
            apply_sboxes=True,
            transition_branch_enabled=True,
        )
        branch_off = shared.logits_with_runtime(
            features,
            correct[cipher_key].runtime_structure,
            apply_sboxes=True,
            transition_branch_enabled=False,
        )

        assert exact.shape == wrong_logits.shape == branch_off.shape == (3, 1)
        assert torch.isfinite(exact).all()
        assert not torch.equal(exact, wrong_logits)
        assert not torch.equal(exact, branch_off)

    assert tensor_mapping_sha256(shared.state_dict()) == state_before
    assert shared.runtime_structure is correct["uknit64"].runtime_structure
    assert set(inspect.signature(shared.logits_with_runtime).parameters) == {
        "features",
        "structure",
        "apply_sboxes",
        "transition_branch_enabled",
    }


def test_k1ao_gate_passes_only_when_every_check_passes() -> None:
    config = load_and_validate_config()
    gate = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        runtime_checks={"runtime": True},
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("shared_weight_runtime_ready")
    assert gate["training_rows"] == 0
    assert gate["optimizer_steps"] == 0
    assert gate["remote_scale"] == "no"
    assert "2048/class/cipher" in gate["next_action"]


def test_k1ao_gate_holds_on_any_runtime_failure() -> None:
    config = load_and_validate_config()
    gate = adjudicate_readiness(
        config=config,
        source_checks={"source": True},
        runtime_checks={"runtime": False},
    )

    assert gate["status"] == "hold"
    assert gate["failed_evidence_checks"] == ["runtime"]
    assert "do not train" in gate["next_action"]


def test_k1ao_wrong_sbox_control_does_not_change_linear_geometry() -> None:
    config = load_and_validate_config()
    for cipher in config["ciphers"]:
        exact = build_runtime_model(cipher, config["model"])
        wrong = build_runtime_model(cipher, config["model"], wrong_sbox=True)

        assert torch.equal(
            exact.runtime_structure.cell_membership,
            wrong.runtime_structure.cell_membership,
        )
        assert torch.equal(
            exact.runtime_structure.linear_matrices,
            wrong.runtime_structure.linear_matrices,
        )
        assert not torch.equal(
            exact.runtime_structure.sbox_truth_bits,
            wrong.runtime_structure.sbox_truth_bits,
        )


def test_k1ao_config_copy_cannot_silently_change_model_contract() -> None:
    config = deepcopy(load_and_validate_config())
    config["model"]["expected_trainable_parameters"] += 1

    assert config["model"]["expected_trainable_parameters"] != EXPECTED_PARAMETER_COUNT


def test_k1ao_plot_explains_readiness_without_claiming_auc(tmp_path: Path) -> None:
    config = load_and_validate_config()
    runtime_rows = [
        {
            "cipher_key": cipher["cipher_key"],
            "block_bits": cipher["block_bits"],
            "cells": cipher["cells"],
            "input_bits": cipher["input_bits"],
            "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
            "state_dict_entries": EXPECTED_STATE_ENTRIES,
            "state_sha256": "a" * 64,
        }
        for cipher in config["ciphers"]
    ]
    result_rows = [
        {
            "cipher_key": cipher["cipher_key"],
            "condition": condition,
            "max_abs_delta_from_correct": (
                0.0 if condition == "correct_runtime" else 0.001
            ),
        }
        for cipher in config["ciphers"]
        for condition in config["controls"]["conditions"]
    ]
    output = tmp_path / "curves.svg"

    report = render_k1ao_svg(
        {"status": "pass"},
        runtime_rows,
        result_rows,
        output,
    )
    svg = output.read_text(encoding="utf-8")

    assert report["auc_claim_present"] is False
    assert report["log_scale_used_for_close_nonzero_deltas"] is True
    assert "一套神经网络权重能否读取三种 SPN 结构" in svg
    assert "当前还没有训练 AUC" in svg
    assert "对数刻度，仅检查是否非零" in svg
