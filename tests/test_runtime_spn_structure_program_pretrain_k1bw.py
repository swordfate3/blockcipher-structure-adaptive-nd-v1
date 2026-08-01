from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from blockcipher_nd.cli.run_runtime_spn_structure_program_pretrain_k1bw import (
    render_k1bw_svg,
)
from blockcipher_nd.models.structure.spn.structure_program_encoder import (
    EDGE_TOKEN_DIM,
    SBOX_TOKEN_DIM,
    RuntimeSpnProgramEncoder,
    StructureProgramEncoderSpec,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_structure_program_pretrain_k1bw import (
    adjudicate,
    load_and_validate_config,
    load_structures,
    structure_variants,
    train_encoder,
)


def test_k1bw_loads_seven_structures_with_fixed_model_geometry() -> None:
    config = load_and_validate_config()
    structures, manifest = load_structures(config)
    encoder = RuntimeSpnProgramEncoder(StructureProgramEncoderSpec())

    outputs = encoder(list(structures.values()))

    assert set(structures) == {
        "gift64",
        "present64",
        "rectangle64",
        "skinny64",
        "midori64",
        "uknit64",
        "dialga128",
    }
    assert len(manifest) == 7
    assert outputs.shape == (7, 64)
    assert EDGE_TOKEN_DIM == 17
    assert SBOX_TOKEN_DIM == 70
    assert encoder.uses_cipher_identity is False
    assert encoder.uses_cipher_name is False


def test_k1bw_transport_relabel_is_exact_but_semantic_controls_differ() -> None:
    config = load_and_validate_config()
    structure = load_structures(config)[0]["dialga128"]
    relabeled, positions, variants = structure_variants(structure, seed=11)
    torch.manual_seed(0)
    encoder = RuntimeSpnProgramEncoder(StructureProgramEncoderSpec())

    correct = encoder.encode_structure(structure)
    transported = encoder.encode_structure(relabeled, cell_position_ids=positions)
    wrong_distances = [
        1.0
        - float(
            torch.nn.functional.cosine_similarity(
                correct,
                encoder.encode_structure(wrong),
                dim=0,
            ).detach()
        )
        for wrong in variants.values()
    ]

    assert float(torch.max(torch.abs(correct - transported)).detach()) <= 1e-6
    assert variants.keys() == {"wrong_linear", "wrong_sbox", "wrong_order"}
    assert min(wrong_distances) > 0.0


def test_k1bw_one_epoch_updates_shared_encoder_without_cipher_parameters() -> None:
    config = deepcopy(load_and_validate_config())
    config["training"]["epochs"] = 1
    structures = load_structures(config)[0]

    encoder, history = train_encoder(config, structures, model_seed=0)

    assert len(history) == 1
    assert history[0]["loss"] >= 0.0
    assert encoder.uses_cipher_identity is False
    assert all("cipher" not in name for name, _value in encoder.named_parameters())


def test_k1bw_gate_requires_holdout_margin_and_gain() -> None:
    config = load_and_validate_config()
    rows = _synthetic_rows()
    protocol = {"source": True}

    passed = adjudicate(config, rows, protocol_checks=protocol)
    assert passed["status"] == "pass"

    held_rows = deepcopy(rows)
    for row in held_rows:
        if row["phase"] == "trained" and row["scope"] == "holdout":
            row["wrong_distance"] = 0.01
            row["semantic_margin"] = 0.01
    held = adjudicate(config, held_rows, protocol_checks=protocol)
    assert held["status"] == "hold"
    assert any("holdout_margin" in name for name in held["failed_research_checks"])

    invalid = adjudicate(config, rows, protocol_checks={"source": False})
    assert invalid["status"] == "invalid"


def test_k1bw_plot_explains_structure_only_scope_in_chinese(tmp_path: Path) -> None:
    config = load_and_validate_config()
    rows = _synthetic_rows()
    gate = adjudicate(config, rows, protocol_checks={"source": True})
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

    report = render_k1bw_svg(gate, rows, history, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert "学习密码结构程序" in text
    assert "不是差分区分AUC" in text
    assert "Dialga-128" in text


def _synthetic_rows() -> list[dict[str, object]]:
    rows = []
    for model_seed in (0, 1):
        for cipher, scope in (("gift64", "train"), ("dialga128", "holdout")):
            for corruption_seed in (11, 23, 37, 53):
                for control in ("wrong_linear", "wrong_sbox", "wrong_order"):
                    for phase, margin in (("initial", 0.01), ("trained", 0.04)):
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
