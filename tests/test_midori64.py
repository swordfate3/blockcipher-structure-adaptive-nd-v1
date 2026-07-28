from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from blockcipher_nd.ciphers import Midori64
from blockcipher_nd.ciphers.spn.midori import (
    MIDORI64_ROUNDS,
    MIDORI64_SBOX,
    midori64_inverse_shuffle_cells,
    midori64_linear_layer,
    midori64_round_linear_layer,
    midori64_round_trace,
    midori64_shuffle_cells,
)
from blockcipher_nd.models.structure.spn.canonical_components import (
    midori_linear_layer,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    midori64_runtime_structure,
)
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.tasks.innovation1.uknit_family_midori64_qualification_k1ag import (
    CONFIG_PATH,
    DESCRIPTOR_PATH,
    EXPECTED_INTERMEDIATE_STATES,
    EXPECTED_PARAMETER_COUNT,
    RUN_ID,
    load_and_validate_config,
    run_qualification,
    write_qualification_artifacts,
)


def test_midori64_matches_two_public_vectors_and_all_round_prefixes() -> None:
    vectors = (
        (0, 0, 0x3C9CCEDA2BBD449A),
        (
            0x42C20FD3B586879E,
            0x687DED3B3C85B3F35B1009863E2A8CBF,
            0x66BCDC6270D901CD,
        ),
    )
    for plaintext, key, ciphertext in vectors:
        trace = midori64_round_trace(plaintext, key)
        assert len(trace) == MIDORI64_ROUNDS
        assert trace[-1] == ciphertext
        assert all(
            Midori64(rounds=rounds, key=key).encrypt(plaintext) == trace[rounds - 1]
            for rounds in range(1, MIDORI64_ROUNDS + 1)
        )
    assert isinstance(build_cipher("midori64", 16), Midori64)


def test_midori64_primitives_have_exact_inverse_and_canonical_properties() -> None:
    assert all(MIDORI64_SBOX[MIDORI64_SBOX[value]] == value for value in range(16))
    assert all(
        midori64_inverse_shuffle_cells(midori64_shuffle_cells(1 << bit)) == 1 << bit
        for bit in range(64)
    )
    assert all(
        midori64_linear_layer(midori64_linear_layer(1 << bit)) == 1 << bit
        for bit in range(64)
    )
    assert all(
        midori64_linear_layer(1 << bit) == midori_linear_layer(1 << bit)
        for bit in range(64)
    )


def test_midori64_runtime_descriptor_reconstructs_complete_round_linear_layer() -> None:
    loaded = load_runtime_spn_descriptor(DESCRIPTOR_PATH, rounds=2).structure
    native = midori64_runtime_structure(rounds=2)

    assert loaded.cells == 16
    assert loaded.rounds == 2
    assert loaded.unique_transition_count == 1
    assert loaded.is_homogeneous
    assert torch.equal(loaded.sbox_truth_bits, native.sbox_truth_bits)
    assert torch.equal(loaded.linear_matrices, native.linear_matrices)
    assert all(
        _apply_matrix_to_int(loaded.linear_matrices[round_index], 1 << bit)
        == midori64_round_linear_layer(1 << bit)
        for round_index in range(2)
        for bit in range(64)
    )


def test_k1ag_zero_training_qualification_passes_every_exact_gate() -> None:
    config = load_and_validate_config(CONFIG_PATH)
    payload = run_qualification(config)

    assert config["run_id"] == RUN_ID
    assert config["audit"]["training_rows"] == 0
    assert config["audit"]["optimizer_steps"] == 0
    assert payload["validation"]["status"] == "pass"
    assert payload["validation"]["errors"] == []
    assert all(payload["validation"]["checks"].values())
    assert payload["gate"]["status"] == "pass"
    assert payload["gate"]["remote_scale"] == "no"
    assert payload["gate"]["optimizer_steps"] == 0
    assert payload["summary"]["result_rows"] == 3
    assert payload["summary"]["public_vectors_matched"] == 2
    assert payload["summary"]["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert payload["rows"][2]["parameter_counts"] == {
        "4": EXPECTED_PARAMETER_COUNT,
        "16": EXPECTED_PARAMETER_COUNT,
    }
    assert payload["rows"][2]["geometry_equal"] is True


def test_k1ag_configuration_fails_closed_on_source_or_gate_drift(
    tmp_path: Path,
) -> None:
    config = load_and_validate_config(CONFIG_PATH)
    drifted = deepcopy(config)
    drifted["intermediate_trace"]["plaintext"] = "0000000000000000"
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="trace source drifted"):
        load_and_validate_config(path)

    drifted = deepcopy(config)
    drifted["gate"].pop("require_fixed_parameter_geometry")
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="gate requirements drifted"):
        load_and_validate_config(path)


def test_k1ag_freezes_intermediate_states_and_writes_indexable_artifacts(
    tmp_path: Path,
) -> None:
    config = load_and_validate_config(CONFIG_PATH)
    assert {
        name: int(config["intermediate_trace"][name], 16)
        for name in EXPECTED_INTERMEDIATE_STATES
    } == EXPECTED_INTERMEDIATE_STATES

    payload = run_qualification(config)
    write_qualification_artifacts(payload, tmp_path)

    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 3
    assert json.loads((tmp_path / "validation.json").read_text())["status"] == "pass"
    assert json.loads((tmp_path / "gate.json").read_text())["decision"].endswith(
        "midori64_k1ag_qualified"
    )
    progress = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in progress] == ["run_start", "run_done"]
    assert progress[-1]["status"] == "pass"
    assert not (tmp_path / "curves.svg").exists()


def _apply_matrix_to_int(matrix: torch.Tensor, state: int) -> int:
    output = 0
    for target in range(int(matrix.shape[0])):
        for source in torch.nonzero(matrix[target], as_tuple=False).flatten().tolist():
            output ^= ((state >> source) & 1) << target
    return output
