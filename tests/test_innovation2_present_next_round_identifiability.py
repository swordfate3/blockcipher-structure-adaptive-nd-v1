from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from blockcipher_nd.ciphers.spn.present import Present80
from blockcipher_nd.cli.audit_innovation2_present_next_round_identifiability import (
    main as audit_main,
)
from blockcipher_nd.evaluation.result_index import DECISION_LABELS, display_name_for_run
from blockcipher_nd.tasks.innovation2.present_next_round_identifiability import (
    PresentNextRoundIdentifiabilityConfig,
    audit_present_next_round_identifiability,
    derive_present80_round_keys,
    encrypt_with_recovered_round_keys,
    present_regular_round,
    recover_present_round_key,
    trace_present80,
)


def test_one_complete_round_pair_recovers_the_exact_round_key() -> None:
    rng = random.Random(20260722)
    for _ in range(64):
        current_state = rng.getrandbits(64)
        round_key = rng.getrandbits(64)
        next_state = present_regular_round(current_state, round_key)

        assert recover_present_round_key(current_state, next_state) == round_key


def test_trace_and_explicit_round_keys_match_present80_encrypt() -> None:
    rng = random.Random(20260723)
    cases = [(0, 0)] + [
        (rng.getrandbits(64), rng.getrandbits(80)) for _ in range(16)
    ]
    for plaintext, master_key in cases:
        trace = trace_present80(plaintext, master_key)
        regular_keys, final_key = derive_present80_round_keys(master_key)

        assert trace.regular_round_keys == regular_keys
        assert trace.final_whitening_key == final_key
        assert trace.ciphertext == Present80(rounds=31, key=master_key).encrypt(
            plaintext
        )
        assert (
            encrypt_with_recovered_round_keys(plaintext, regular_keys, final_key)
            == trace.ciphertext
        )
    assert Present80(rounds=31, key=0).encrypt(0) == 0x5579C1387B228445


def test_single_calibration_trace_recovers_all_round_and_whitening_keys() -> None:
    plaintext = 0x0123456789ABCDEF
    master_key = 0x0123456789ABCDEF0123
    trace = trace_present80(plaintext, master_key)
    recovered = tuple(
        recover_present_round_key(current, following)
        for current, following in zip(
            trace.states_before_round,
            trace.states_after_round,
            strict=True,
        )
    )
    recovered_final = trace.states_after_round[-1] ^ trace.ciphertext

    assert recovered == trace.regular_round_keys
    assert recovered_final == trace.final_whitening_key


def test_small_identifiability_audit_is_complete_and_deterministic() -> None:
    config = PresentNextRoundIdentifiabilityConfig(
        run_id="test-identifiability",
        master_keys=2,
        heldout_states_per_round=8,
    )

    first = audit_present_next_round_identifiability(config)
    second = audit_present_next_round_identifiability(config)
    gate = first["gate"]

    assert first["rows"] == second["rows"]
    assert len(first["rows"]) == 62
    assert gate["status"] == "pass"
    assert gate["decision"] == (
        "innovation2_present_full_state_next_round_criticality_not_identifiable"
    )
    assert all(gate["protocol_checks"].values())
    assert all(gate["execution_checks"].values())
    assert gate["metrics"]["regular_round_keys_exact"] == 62
    assert gate["metrics"]["heldout_transition_total"] == 496
    assert gate["metrics"]["heldout_transition_exact_rate"] == 1.0
    assert gate["metrics"]["full_reconstruction_total"] == 16
    assert gate["metrics"]["full_reconstruction_exact_rate"] == 1.0
    assert gate["next_action"]["train_full_state_next_round_models"] is False
    assert all(len(row["master_key_sha256"]) == 64 for row in first["rows"])


def test_formal_config_freezes_expected_evidence_scale() -> None:
    config = PresentNextRoundIdentifiabilityConfig()

    assert config.master_keys * config.rounds == 496
    assert (
        config.master_keys * config.rounds * config.heldout_states_per_round
        == 126_976
    )
    assert config.master_keys * config.heldout_states_per_round == 4_096


def test_cli_writes_required_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "audit"

    exit_code = audit_main(
        [
            "--output-root",
            str(output),
            "--run-id",
            "test-cli-identifiability",
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
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    progress = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 62
    assert gate["status"] == "pass"
    assert progress[0]["event"] == "run_start"
    assert progress[-1]["event"] == "run_done"
    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest) == 5
    for row in manifest:
        artifact = output / row["path"]
        assert row["bytes"] == artifact.stat().st_size
        assert row["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "pass"
    assert all(validation["checks"].values())


def test_result_index_names_and_explains_identifiability_decision() -> None:
    run_id = "i2_present_next_round_full_state_identifiability_audit_20260722"
    decision = "innovation2_present_full_state_next_round_criticality_not_identifiable"

    assert "完整轮间状态" in display_name_for_run(run_id)
    assert "子密钥" in display_name_for_run(run_id)
    assert "不能测量随机猜测临界轮" in DECISION_LABELS[decision]
