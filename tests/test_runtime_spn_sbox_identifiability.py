from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_identifiability import (
    CIPHERS,
    CONTROL_MODES,
    EXPECTED_SEEDS,
    adjudicate_sbox_identifiability,
    build_sbox_counterfactuals,
    load_and_validate_sbox_identifiability_config,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    _load_structures,
    load_and_validate_holdout_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_sbox_identifiability_s1_20260726.json"
)


def test_frozen_s1_config_and_counterfactual_contract() -> None:
    config = load_and_validate_sbox_identifiability_config(
        CONFIG,
        project_root=ROOT,
    )
    base = load_and_validate_holdout_config(
        ROOT / config["source"]["protocol_config_path"]
    )
    structures = _load_structures(base)
    panel = build_sbox_counterfactuals(structures)

    assert tuple(config["ciphers"]) == CIPHERS
    assert tuple(config["controls"]) == CONTROL_MODES
    assert set(panel) == set(CIPHERS)
    assert all(tuple(controls) == CONTROL_MODES for controls in panel.values())

    for cipher, controls in panel.items():
        exact = controls["exact"]
        for mode, item in controls.items():
            structure = item["structure"]
            assert structure.cell_membership.equal(exact["structure"].cell_membership)
            assert structure.bit_role.equal(exact["structure"].bit_role)
            assert structure.linear_matrices.equal(exact["structure"].linear_matrices)
            assert structure.inverse_linear_matrices.equal(
                exact["structure"].inverse_linear_matrices
            )
            if mode == "zero_descriptor":
                assert item["valid_sbox"] is False
            else:
                assert item["valid_sbox"] is True

        matching_mode = (
            f"broadcast_{cipher}"
            if cipher != "uknit64"
            else "broadcast_uknit64_reference"
        )
        if cipher == "uknit64":
            assert controls[matching_mode]["equivalent_to_exact"] is False
        else:
            assert controls[matching_mode]["equivalent_to_exact"] is True
            assert torch.equal(
                controls[matching_mode]["structure"].sbox_truth_bits,
                exact["structure"].sbox_truth_bits,
            )


def test_s1_gate_distinguishes_identifiable_aliased_ignored_and_invalid() -> None:
    config = load_and_validate_sbox_identifiability_config(
        CONFIG,
        project_root=ROOT,
    )
    rows = _synthetic_rows()

    passed = adjudicate_sbox_identifiability(
        config=config,
        rows=rows,
        validation={"status": "pass", "checks": {"valid": True}},
    )
    assert passed["status"] == "pass"
    assert passed["decision"].endswith("sbox_identifiable")

    aliased_rows = deepcopy(rows)
    row = next(
        item
        for item in aliased_rows
        if item["seed"] == 0
        and item["cipher"] == "dialga128"
        and item["control"] == "broadcast_gift64"
    )
    row["auc"] = 0.81
    aliased = adjudicate_sbox_identifiability(
        config=config,
        rows=aliased_rows,
        validation={"status": "pass", "checks": {"valid": True}},
    )
    assert aliased["status"] == "hold"
    assert aliased["decision"].endswith("sbox_responsive_but_not_identifiable")

    ignored_rows = deepcopy(rows)
    for item in ignored_rows:
        if item["seed"] == 1 and item["cipher"] == "skinny64":
            item["max_abs_probability_delta_from_exact"] = 0.0
    ignored = adjudicate_sbox_identifiability(
        config=config,
        rows=ignored_rows,
        validation={"status": "pass", "checks": {"valid": True}},
    )
    assert ignored["status"] == "hold"
    assert ignored["decision"].endswith("sbox_descriptor_functionally_ignored")

    invalid = adjudicate_sbox_identifiability(
        config=config,
        rows=rows,
        validation={"status": "fail", "checks": {"valid": False}},
    )
    assert invalid["status"] == "invalid"
    assert invalid["decision"].endswith("protocol_invalid")


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for cipher in CIPHERS:
            matching_mode = (
                f"broadcast_{cipher}"
                if cipher != "uknit64"
                else None
            )
            for control in CONTROL_MODES:
                equivalent = control == "exact" or control == matching_mode
                rows.append(
                    {
                        "seed": seed,
                        "cipher": cipher,
                        "control": control,
                        "auc": 0.8 if equivalent else 0.7,
                        "best_accuracy": 0.75 if equivalent else 0.65,
                        "valid_sbox": control != "zero_descriptor",
                        "equivalent_to_exact": equivalent,
                        "max_abs_probability_delta_from_exact": (
                            0.0 if equivalent else 0.1
                        ),
                    }
                )
    return rows
