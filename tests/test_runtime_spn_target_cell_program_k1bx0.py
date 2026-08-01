from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from blockcipher_nd.cli.run_runtime_spn_target_cell_program_k1bx0 import (
    render_k1bx0_svg,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_structure_program_pretrain_k1bw import (
    structure_variants,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_target_cell_program_k1bx0 import (
    CONTROL_NAMES,
    adjudicate,
    build_encoder,
    load_and_validate_config,
    load_source_authority,
    load_structures,
    train_encoder,
)


def test_k1bx0_preserves_k1bw_protocol_and_hold_authority() -> None:
    config = load_and_validate_config()
    source_rows, checks = load_source_authority(config)

    assert config["training"] == {
        "train_ciphers": [
            "gift64",
            "present64",
            "rectangle64",
            "skinny64",
            "midori64",
            "uknit64",
        ],
        "holdout_ciphers": ["dialga128"],
        "seeds": [0, 1],
        "corruption_seeds": [11, 23, 37, 53],
        "epochs": 160,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "triplet_margin": 0.12,
        "device": "cpu",
        "execution": "local_structure_only_diagnostic",
    }
    assert source_rows
    assert all(checks.values())


def test_k1bx0_uses_one_fixed_geometry_across_all_seven_structures() -> None:
    config = load_and_validate_config()
    structures, manifest = load_structures(config)
    encoder = build_encoder(config)

    outputs = encoder(list(structures.values()))

    assert len(manifest) == 7
    assert outputs.shape == (7, 64)
    assert encoder.uses_cipher_identity is False
    assert encoder.uses_cipher_name is False
    assert encoder.aggregates_edges_at_actual_target_cell is True
    assert encoder.pools_only_after_cell_transition is True
    assert all("cipher" not in name for name, _value in encoder.named_parameters())


def test_k1bx0_cell_relabel_is_exact_and_wrong_binding_changes_output() -> None:
    config = load_and_validate_config()
    structure = load_structures(config)[0]["dialga128"]
    relabeled, positions, _variants = structure_variants(structure, seed=11)
    torch.manual_seed(0)
    encoder = build_encoder(config)

    correct = encoder.encode_structure(structure)
    transported = encoder.encode_structure(relabeled, cell_position_ids=positions)
    wrong_binding = encoder.encode_structure(structure, edge_binding_seed=11)

    assert float(torch.max(torch.abs(correct - transported)).detach()) <= 1e-6
    assert not torch.allclose(correct, wrong_binding, atol=1e-8, rtol=0.0)
    assert 1.0 - float(
        torch.nn.functional.cosine_similarity(correct, wrong_binding, dim=0).detach()
    ) > 0.0


def test_k1bx0_one_epoch_updates_only_shared_structure_encoder() -> None:
    config = deepcopy(load_and_validate_config())
    config["training"]["epochs"] = 1
    structures = load_structures(config)[0]
    torch.manual_seed(0)
    initial = build_encoder(config)
    initial_state = {
        name: value.detach().clone() for name, value in initial.state_dict().items()
    }

    encoder, history = train_encoder(config, structures, model_seed=0)

    assert len(history) == 1
    assert history[0]["loss"] >= 0.0
    assert any(
        not torch.equal(value, initial_state[name])
        for name, value in encoder.state_dict().items()
    )
    assert all("cipher" not in name for name, _value in encoder.named_parameters())


def test_k1bx0_gate_distinguishes_pass_hold_and_invalid() -> None:
    config = load_and_validate_config()
    rows = _synthetic_candidate_rows()
    source_rows = _synthetic_source_rows()

    passed = adjudicate(
        config,
        rows,
        source_rows,
        protocol_checks={"source": True},
    )
    assert passed["status"] == "pass"

    held_rows = deepcopy(rows)
    for row in held_rows:
        if row["phase"] == "trained" and row["scope"] == "holdout":
            row["wrong_cosine"] = 0.99
            row["wrong_distance"] = 0.01
            row["semantic_margin"] = 0.01
    held = adjudicate(
        config,
        held_rows,
        source_rows,
        protocol_checks={"source": True},
    )
    assert held["status"] == "hold"
    assert any("_margin" in name for name in held["failed_research_checks"])

    invalid = adjudicate(
        config,
        rows,
        source_rows,
        protocol_checks={"source": False},
    )
    assert invalid["status"] == "invalid"


def test_k1bx0_plot_explains_target_cell_scope_in_chinese(tmp_path: Path) -> None:
    config = load_and_validate_config()
    rows = _synthetic_candidate_rows()
    gate = adjudicate(
        config,
        rows,
        _synthetic_source_rows(),
        protocol_checks={"source": True},
    )
    history = [
        {
            "model_seed": seed,
            "epoch": epoch,
            "loss": 0.2 / epoch,
            "positive_distance": 0.0,
            "negative_distance": 0.1,
            "corruption_seed": 11,
        }
        for seed in (0, 1)
        for epoch in (1, 2)
    ]
    output = tmp_path / "curves.svg"

    report = render_k1bx0_svg(gate, rows, history, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert "GF(2)边先绑定目标cell" in text
    assert "错误目标cell绑定" in text
    assert "不是差分区分AUC" in text
    assert "Dialga" in text


def _synthetic_candidate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_seed in (0, 1):
        for cipher, scope in (("gift64", "train"), ("dialga128", "holdout")):
            for corruption_seed in (11, 23, 37, 53):
                for control in CONTROL_NAMES:
                    for phase, margin in (("initial", 0.01), ("trained", 0.06)):
                        rows.append(
                            {
                                "model_seed": model_seed,
                                "phase": phase,
                                "cipher_key": cipher,
                                "scope": scope,
                                "corruption_seed": corruption_seed,
                                "control": control,
                                "positive_cosine": 1.0,
                                "positive_distance": 0.0,
                                "wrong_cosine": 1.0 - margin,
                                "wrong_distance": margin,
                                "semantic_margin": margin,
                            }
                        )
    return rows


def _synthetic_source_rows() -> list[dict[str, object]]:
    return [
        {
            "model_seed": model_seed,
            "phase": "trained",
            "cipher_key": "dialga128",
            "scope": "holdout",
            "corruption_seed": corruption_seed,
            "control": control,
            "semantic_margin": 0.04,
        }
        for model_seed in (0, 1)
        for corruption_seed in (11, 23, 37, 53)
        for control in ("wrong_linear", "wrong_sbox", "wrong_order")
    ]
