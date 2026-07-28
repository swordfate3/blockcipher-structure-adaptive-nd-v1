from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from blockcipher_nd.ciphers.spn.midori import (
    MIDORI64_ROUNDS,
    MIDORI64_SBOX,
    MIDORI64_SHUFFLE,
    Midori64,
    midori64_inverse_shuffle_cells,
    midori64_linear_layer,
    midori64_round_keys,
    midori64_round_linear_layer,
    midori64_round_trace,
    midori64_shuffle_cells,
    midori64_sub_cells,
)
from blockcipher_nd.engine.modeling import cipher_profile, model_metadata
from blockcipher_nd.models.structure.spn.canonical_components import (
    midori_linear_layer,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    midori64_runtime_structure,
)
from blockcipher_nd.registry.cipher_factory import build_cipher, default_difference
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[4]
RUN_ID = "i1_uknit_family_midori64_qualification_k1ag_20260729"
CONFIG_PATH = ROOT / (
    "configs/experiment/innovation1/"
    "innovation1_uknit_family_midori64_qualification_k1ag_20260729.json"
)
DESCRIPTOR_PATH = ROOT / "configs/runtime/spn/midori64.json"
MODEL_KEY = "runtime_spn_ct_k1aa_virtual_slot_histogram_true"
EXPECTED_PARAMETER_COUNT = 214_316
EXPECTED_PAIRS = (4, 16)
EXPECTED_VECTOR_COUNT = 2
EXPECTED_INTERMEDIATE_STATES = {
    "whitening_key": 0x336DE4BD02AF3F4C,
    "after_initial_whitening": 0x71AFEB6EB729B8D2,
    "round0_after_sub_cells": 0x7A1645F457D9582D,
    "round0_after_shuffle_cells": 0x7D5D249A765F4815,
    "round0_after_mix_columns": 0x5F7F71CFCDE4C09D,
    "round0_after_add_round_key": 0x37039DF5E170737F,
    "round1_after_sub_cells": 0x37C3926B4A7C7376,
    "round1_after_shuffle_cells": 0x372679C7A376B3C4,
    "round1_after_mix_columns": 0x37262C922BFEB3C4,
    "round1_after_add_round_key": 0x6D27351404D43F7B,
}
EXPECTED_GATE = {
    "require_all_public_vectors": True,
    "require_all_intermediate_states": True,
    "require_all_16_round_prefixes": True,
    "require_involutory_sbox": True,
    "require_shuffle_inverse": True,
    "require_mix_columns_involution": True,
    "require_64_basis_canonical_mix_equivalence": True,
    "require_64_basis_descriptor_linear_equivalence": True,
    "require_descriptor_sbox_equivalence": True,
    "require_fixed_parameter_geometry": True,
}


def load_and_validate_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("K1-AG schema_version must be 1")
    if (
        config.get("experiment")
        != "innovation1_uknit_family_midori64_qualification_k1ag"
    ):
        raise ValueError("K1-AG experiment name drifted")
    if config.get("run_id") != RUN_ID:
        raise ValueError("K1-AG run_id drifted")
    if config.get("cipher") != {
        "name": "Midori64",
        "block_bits": 64,
        "key_bits": 128,
        "full_rounds": 16,
        "runtime_descriptor": "configs/runtime/spn/midori64.json",
    }:
        raise ValueError("K1-AG cipher contract drifted")
    if len(config.get("public_vectors", ())) != EXPECTED_VECTOR_COUNT:
        raise ValueError("K1-AG must freeze exactly two public vectors")
    expected_vectors = (
        (
            "0000000000000000",
            "00000000000000000000000000000000",
            "3c9cceda2bbd449a",
        ),
        (
            "42c20fd3b586879e",
            "687ded3b3c85b3f35b1009863e2a8cbf",
            "66bcdc6270d901cd",
        ),
    )
    actual_vectors = tuple(
        (row.get("plaintext"), row.get("key"), row.get("ciphertext"))
        for row in config["public_vectors"]
    )
    if actual_vectors != expected_vectors:
        raise ValueError("K1-AG public vectors drifted")
    if config.get("audit") != {
        "training_rows": 0,
        "validation_rows": 0,
        "optimizer_steps": 0,
        "data_generation": False,
        "remote": False,
    }:
        raise ValueError("K1-AG zero-training audit contract drifted")
    model = config.get("model_geometry", {})
    if model != {
        "model": MODEL_KEY,
        "hidden_bits": 32,
        "virtual_projection_slots": 16,
        "pairs_per_sample": [4, 16],
        "expected_trainable_parameters": EXPECTED_PARAMETER_COUNT,
    }:
        raise ValueError("K1-AG model geometry contract drifted")
    if config.get("gate") != EXPECTED_GATE:
        raise ValueError("K1-AG exact gate requirements drifted")
    trace = config.get("intermediate_trace", {})
    if trace.get("plaintext") != "42c20fd3b586879e" or trace.get("key") != (
        "687ded3b3c85b3f35b1009863e2a8cbf"
    ):
        raise ValueError("K1-AG intermediate trace source drifted")
    if {
        name: int(trace.get(name, ""), 16) for name in EXPECTED_INTERMEDIATE_STATES
    } != EXPECTED_INTERMEDIATE_STATES:
        raise ValueError("K1-AG intermediate trace drifted")
    return config


def run_qualification(
    config: Mapping[str, Any],
    *,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    vector_rows = _public_vector_rows(config)
    intermediate_actual = _intermediate_trace(config)
    trace = midori64_round_trace(
        int(config["intermediate_trace"]["plaintext"], 16),
        int(config["intermediate_trace"]["key"], 16),
    )
    descriptor = load_runtime_spn_descriptor(
        project_root / config["cipher"]["runtime_descriptor"],
        rounds=2,
    )
    native_structure = midori64_runtime_structure(rounds=2)
    descriptor_checks = _descriptor_checks(descriptor.structure, native_structure)
    model_geometry = _model_geometry(
        project_root / config["cipher"]["runtime_descriptor"]
    )

    protocol_checks = {
        "configuration_frozen": config.get("run_id") == RUN_ID,
        "two_public_vectors_present": len(vector_rows) == EXPECTED_VECTOR_COUNT,
        "descriptor_path_exact": descriptor.path
        == (project_root / config["cipher"]["runtime_descriptor"]).resolve(),
        "zero_training_contract": config.get("audit")
        == {
            "training_rows": 0,
            "validation_rows": 0,
            "optimizer_steps": 0,
            "data_generation": False,
            "remote": False,
        },
    }
    research_checks = {
        "all_public_vectors_exact": all(row["matches"] for row in vector_rows),
        "all_intermediate_states_exact": intermediate_actual
        == EXPECTED_INTERMEDIATE_STATES,
        "all_16_round_prefixes_exact": len(trace) == MIDORI64_ROUNDS
        and all(
            Midori64(
                rounds=rounds, key=int(config["intermediate_trace"]["key"], 16)
            ).encrypt(int(config["intermediate_trace"]["plaintext"], 16))
            == trace[rounds - 1]
            for rounds in range(1, MIDORI64_ROUNDS + 1)
        ),
        "sbox_involutory_all_inputs": all(
            MIDORI64_SBOX[MIDORI64_SBOX[value]] == value for value in range(16)
        ),
        "shuffle_inverse_all_basis_bits": sorted(MIDORI64_SHUFFLE) == list(range(16))
        and all(
            midori64_inverse_shuffle_cells(midori64_shuffle_cells(1 << bit)) == 1 << bit
            for bit in range(64)
        ),
        "mix_columns_involutory_all_basis_bits": all(
            midori64_linear_layer(midori64_linear_layer(1 << bit)) == 1 << bit
            for bit in range(64)
        ),
        "canonical_mix_equivalent_all_64_basis_bits": all(
            midori64_linear_layer(1 << bit) == midori_linear_layer(1 << bit)
            for bit in range(64)
        ),
        **descriptor_checks,
        "cipher_factory_and_profile_registered": _registration_exact(config),
        "fixed_model_parameter_geometry": model_geometry["geometry_equal"]
        and model_geometry["parameter_counts"]
        == {str(pairs): EXPECTED_PARAMETER_COUNT for pairs in EXPECTED_PAIRS}
        and model_geometry["forward_shapes"]
        == {str(pairs): [2, 1] for pairs in EXPECTED_PAIRS},
    }
    validation_checks = {**protocol_checks, **research_checks}
    validation = {
        "run_id": RUN_ID,
        "status": "pass" if all(validation_checks.values()) else "fail",
        "checks": validation_checks,
        "errors": [name for name, passed in validation_checks.items() if not passed],
    }
    passed = validation["status"] == "pass"
    gate = {
        "run_id": RUN_ID,
        "status": "pass" if passed else "fail",
        "decision": (
            "innovation1_uknit_family_midori64_k1ag_qualified"
            if passed
            else "innovation1_uknit_family_midori64_k1ag_qualification_failed"
        ),
        "protocol_valid": all(protocol_checks.values()),
        "protocol_checks": protocol_checks,
        "research_checks": research_checks,
        "failed_protocol_checks": [
            name for name, value in protocol_checks.items() if not value
        ],
        "failed_research_checks": [
            name for name, value in research_checks.items() if not value
        ],
        "training_rows": 0,
        "validation_rows": 0,
        "optimizer_steps": 0,
        "remote_scale": "no",
        "next_action": (
            "preregister_midori64_difference_position_calibration"
            if passed
            else "repair_failed_midori64_adapter_or_descriptor_invariant_only"
        ),
    }
    rows = [
        {
            "run_id": RUN_ID,
            "row_kind": "cipher_adapter",
            "cipher": "midori64",
            "public_vectors": vector_rows,
            "matched_public_vectors": sum(row["matches"] for row in vector_rows),
            "round_prefix_states": len(trace),
            "intermediate_states_matched": sum(
                intermediate_actual[name] == expected
                for name, expected in EXPECTED_INTERMEDIATE_STATES.items()
            ),
            "training_rows": 0,
            "optimizer_steps": 0,
        },
        {
            "run_id": RUN_ID,
            "row_kind": "runtime_descriptor",
            "cipher": "midori64",
            "descriptor_path": str(descriptor.path),
            "descriptor_sha256": descriptor.sha256,
            "runtime_rounds": descriptor.structure.rounds,
            "runtime_cells": descriptor.structure.cells,
            "unique_transitions": descriptor.structure.unique_transition_count,
            "sbox_entries_matched": 16 * 16,
            "linear_basis_vectors_matched": 64,
            "training_rows": 0,
            "optimizer_steps": 0,
        },
        {
            "run_id": RUN_ID,
            "row_kind": "fixed_model_geometry",
            "cipher": "midori64",
            "model": MODEL_KEY,
            **model_geometry,
            "training_rows": 0,
            "optimizer_steps": 0,
        },
    ]
    summary = {
        "run_id": RUN_ID,
        "status": gate["status"],
        "decision": gate["decision"],
        "result_rows": len(rows),
        "public_vectors_matched": sum(row["matches"] for row in vector_rows),
        "descriptor_linear_basis_vectors_matched": 64
        if descriptor_checks["descriptor_linear_equivalent_all_64_basis_bits"]
        else 0,
        "trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
        "training_rows": 0,
        "optimizer_steps": 0,
        "remote_scale": "no",
        "next_action": gate["next_action"],
    }
    return {
        "rows": rows,
        "validation": validation,
        "gate": gate,
        "summary": summary,
    }


def write_qualification_artifacts(
    payload: Mapping[str, Any], output_root: Path
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat()
    (output_root / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload["rows"]),
        encoding="utf-8",
    )
    for name in ("validation", "gate", "summary"):
        (output_root / f"{name}.json").write_text(
            json.dumps(payload[name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    progress = (
        {"event": "run_start", "run_id": RUN_ID, "timestamp": timestamp},
        {
            "event": "run_done",
            "run_id": RUN_ID,
            "timestamp": timestamp,
            "status": payload["gate"]["status"],
            "decision": payload["gate"]["decision"],
        },
    )
    (output_root / "progress.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress),
        encoding="utf-8",
    )


def _public_vector_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vector in config["public_vectors"]:
        plaintext = int(vector["plaintext"], 16)
        key = int(vector["key"], 16)
        expected = int(vector["ciphertext"], 16)
        actual = Midori64(key=key).encrypt(plaintext)
        rows.append(
            {
                "plaintext": f"{plaintext:016x}",
                "key": f"{key:032x}",
                "expected_ciphertext": f"{expected:016x}",
                "actual_ciphertext": f"{actual:016x}",
                "matches": actual == expected,
            }
        )
    return rows


def _intermediate_trace(config: Mapping[str, Any]) -> dict[str, int]:
    source = config["intermediate_trace"]
    plaintext = int(source["plaintext"], 16)
    key = int(source["key"], 16)
    key0, key1 = key >> 64, key & ((1 << 64) - 1)
    whitening_key = key0 ^ key1
    round_keys = midori64_round_keys(key)
    state = plaintext ^ whitening_key
    actual = {
        "whitening_key": whitening_key,
        "after_initial_whitening": state,
    }
    for round_index in range(2):
        state = midori64_sub_cells(state)
        actual[f"round{round_index}_after_sub_cells"] = state
        state = midori64_shuffle_cells(state)
        actual[f"round{round_index}_after_shuffle_cells"] = state
        state = midori64_linear_layer(state)
        actual[f"round{round_index}_after_mix_columns"] = state
        state ^= round_keys[round_index]
        actual[f"round{round_index}_after_add_round_key"] = state
    return actual


def _descriptor_checks(
    descriptor: RuntimeSpnStructure,
    native: RuntimeSpnStructure,
) -> dict[str, bool]:
    expected_sbox = torch.tensor(MIDORI64_SBOX, dtype=torch.long)
    return {
        "descriptor_sbox_equivalent_all_cells_inputs": all(
            torch.equal(descriptor.sbox_tables(round_index)[cell], expected_sbox)
            for round_index in range(descriptor.rounds)
            for cell in range(descriptor.cells)
        ),
        "descriptor_linear_equivalent_all_64_basis_bits": all(
            _apply_matrix_to_int(descriptor.linear_matrices[round_index], 1 << bit)
            == midori64_round_linear_layer(1 << bit)
            for round_index in range(descriptor.rounds)
            for bit in range(64)
        ),
        "factory_and_descriptor_structures_equal": torch.equal(
            descriptor.sbox_truth_bits, native.sbox_truth_bits
        )
        and torch.equal(descriptor.linear_matrices, native.linear_matrices)
        and torch.equal(
            descriptor.inverse_linear_matrices, native.inverse_linear_matrices
        ),
        "descriptor_repeats_one_homogeneous_transition": descriptor.rounds == 2
        and descriptor.cells == 16
        and descriptor.unique_transition_count == 1
        and descriptor.is_homogeneous,
    }


def _model_geometry(descriptor_path: Path) -> dict[str, Any]:
    options = {
        "runtime_structure_path": str(descriptor_path),
        "runtime_round_start": 0,
        "runtime_rounds": 2,
        "virtual_projection_slots": 16,
        "input_difference_hex": "0x0000000000000040",
    }
    models = {
        pairs: build_model(
            MODEL_KEY,
            input_bits=pairs * 128,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        )
        for pairs in EXPECTED_PAIRS
    }
    geometries = {
        pairs: tuple(
            (name, tuple(value.shape)) for name, value in model.state_dict().items()
        )
        for pairs, model in models.items()
    }
    forward_shapes: dict[str, list[int]] = {}
    for pairs, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, pairs * 128))
        forward_shapes[str(pairs)] = list(output.shape)
    return {
        "pairs_per_sample": list(EXPECTED_PAIRS),
        "parameter_counts": {
            str(pairs): int(model_metadata(model)["trainable_parameter_count"])
            for pairs, model in models.items()
        },
        "geometry_equal": geometries[EXPECTED_PAIRS[0]]
        == geometries[EXPECTED_PAIRS[1]],
        "state_tensor_count": len(geometries[EXPECTED_PAIRS[0]]),
        "forward_shapes": forward_shapes,
    }


def _registration_exact(config: Mapping[str, Any]) -> bool:
    vector = config["public_vectors"][1]
    cipher = build_cipher("midori64", 16, int(vector["key"], 16))
    profile = cipher_profile("midori64")
    return bool(
        isinstance(cipher, Midori64)
        and cipher.encrypt(int(vector["plaintext"], 16))
        == int(vector["ciphertext"], 16)
        and cipher.name == profile.name == "Midori64"
        and profile.structure == "SPN"
        and profile.block_bits == 64
        and profile.key_bits == 128
        and default_difference("midori64") == 0x40
    )


def _apply_matrix_to_int(matrix: torch.Tensor, state: int) -> int:
    output = 0
    for target in range(int(matrix.shape[0])):
        bit = 0
        for source in torch.nonzero(matrix[target], as_tuple=False).flatten().tolist():
            bit ^= (state >> source) & 1
        output |= bit << target
    return output


__all__ = [
    "CONFIG_PATH",
    "DESCRIPTOR_PATH",
    "EXPECTED_INTERMEDIATE_STATES",
    "EXPECTED_PARAMETER_COUNT",
    "RUN_ID",
    "load_and_validate_config",
    "run_qualification",
    "write_qualification_artifacts",
]
