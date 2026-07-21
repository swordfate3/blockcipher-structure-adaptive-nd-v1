from __future__ import annotations

import hashlib
import inspect
import json
import random
from pathlib import Path

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX, Present80
from blockcipher_nd.cli.audit_innovation2_present_next_round_selected8_identifiability import (
    main as audit_main,
)
from blockcipher_nd.evaluation.result_index import DECISION_LABELS, display_name_for_run
from blockcipher_nd.tasks.innovation2.present_next_round_identifiability import (
    present_regular_round,
)
from blockcipher_nd.tasks.innovation2.present_next_round_selected8_identifiability import (
    PresentSelected8IdentifiabilityConfig,
    audit_present_selected8_identifiability,
    derive_selected8_geometry,
    predict_selected_bits_from_partial_key,
    selected_bits_from_state,
    update_key_nibble_candidates,
)


def test_selected8_geometry_is_derived_from_the_real_present_p_layer() -> None:
    geometry = derive_selected8_geometry()

    assert [
        (
            item.msb_position,
            item.inverse_p_source_bit,
            item.key_nibble,
            item.sbox_output_role,
        )
        for item in geometry
    ] == [
        (0, 63, 15, 3),
        (2, 55, 13, 3),
        (8, 31, 7, 3),
        (10, 23, 5, 3),
        (32, 61, 15, 1),
        (34, 53, 13, 1),
        (40, 29, 7, 1),
        (42, 21, 5, 1),
    ]
    assert all(
        Present80.permutation_layer(1 << item.inverse_p_source_bit)
        == 1 << item.destination_integer_bit
        for item in geometry
    )


def test_candidate_reducer_receives_no_full_next_state_or_actual_key() -> None:
    assert tuple(inspect.signature(update_key_nibble_candidates).parameters) == (
        "current_nibble",
        "observed_role_bits",
        "roles",
        "candidates",
    )


def test_two_role_projection_identifies_every_four_bit_key_with_all_inputs() -> None:
    roles = (3, 1)
    for actual_key in range(16):
        candidates = tuple(range(16))
        for current_nibble in range(16):
            substituted = PRESENT_SBOX[current_nibble ^ actual_key]
            observed = tuple((substituted >> role) & 1 for role in roles)
            candidates = update_key_nibble_candidates(
                current_nibble,
                observed,
                roles,
                candidates,
            )
        assert candidates == (actual_key,)


def test_partial_key_predictor_replays_only_the_selected_next_state_bits() -> None:
    rng = random.Random(20260722)
    geometry = derive_selected8_geometry()
    affected_nibbles = {item.key_nibble for item in geometry}
    for _ in range(32):
        current_state = rng.getrandbits(64)
        round_key = rng.getrandbits(64)
        next_state = present_regular_round(current_state, round_key)
        recovered = {
            nibble: (round_key >> (4 * nibble)) & 0xF
            for nibble in affected_nibbles
        }

        assert predict_selected_bits_from_partial_key(
            current_state, recovered, geometry
        ) == selected_bits_from_state(next_state, geometry)


def test_small_selected8_audit_is_complete_and_deterministic() -> None:
    config = PresentSelected8IdentifiabilityConfig(
        run_id="test-selected8-identifiability",
        master_keys=2,
        heldout_states_per_round=8,
    )

    first = audit_present_selected8_identifiability(config)
    second = audit_present_selected8_identifiability(config)
    gate = first["gate"]

    assert first["rows"] == second["rows"]
    assert len(first["rows"]) == 62
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_present_selected8_next_round_is_partial_subkey_recovery_not_diffusion_criticality"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["execution_checks"].values())
    assert gate["metrics"]["subkey_nibble_instances"] == 248
    assert gate["metrics"]["unique_key_nibbles"] == 248
    assert gate["metrics"]["heldout_selected8_total"] == 496
    assert gate["metrics"]["heldout_selected8_exact_rate"] == 1.0
    assert gate["next_action"]["train_selected8_complete_current_state_neural_model"] is False
    assert all(len(row["master_key_sha256"]) == 64 for row in first["rows"])


def test_formal_config_freezes_expected_evidence_scale() -> None:
    config = PresentSelected8IdentifiabilityConfig()

    assert config.master_keys * config.rounds == 496
    assert config.master_keys * config.rounds * 4 == 1_984
    assert (
        config.master_keys * config.rounds * config.heldout_states_per_round
        == 126_976
    )


def test_selected8_cli_writes_checksummed_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "audit"

    exit_code = audit_main(
        [
            "--output-root",
            str(output),
            "--run-id",
            "test-cli-selected8-identifiability",
            "--master-keys",
            "2",
            "--heldout-states-per-round",
            "4",
        ]
    )

    assert exit_code == 0
    assert {
        "results.jsonl",
        "summary.json",
        "gate.json",
        "metadata.json",
        "progress.jsonl",
        "artifact_manifest.json",
        "validation.json",
    }.issubset(path.name for path in output.iterdir())
    rows = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 62
    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for row in manifest:
        artifact = output / row["path"]
        assert row["bytes"] == artifact.stat().st_size
        assert row["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "pass"
    assert all(validation["checks"].values())


def test_result_index_names_and_explains_selected8_decision() -> None:
    run_id = "i2_present_next_round_selected8_partial_subkey_identifiability_audit_20260722"
    decision = (
        "innovation2_present_selected8_next_round_is_partial_subkey_recovery_not_diffusion_criticality"
    )

    assert "轮间八输出bit" in display_name_for_run(run_id)
    assert "部分子密钥" in display_name_for_run(run_id)
    assert "四个轮密钥nibble" in DECISION_LABELS[decision]
