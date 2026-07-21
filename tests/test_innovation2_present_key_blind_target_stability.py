from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_key_blind_target_stability import (
    main as audit_main,
)
from blockcipher_nd.evaluation.result_index import DECISION_LABELS, display_name_for_run
from blockcipher_nd.tasks.innovation2.present_key_blind_target_stability import (
    PresentKeyBlindStabilityConfig,
    _sample_keys,
    _sample_plaintexts,
    audit_present_key_blind_target_stability,
    encrypt_selected_bits_batch,
)
from blockcipher_nd.tasks.innovation2.selected_output_bit_head import (
    SELECTED_MSB_INDICES,
)
from blockcipher_nd.ciphers.spn.present import Present80


def _small_config(run_id: str = "test-key-blind") -> PresentKeyBlindStabilityConfig:
    return PresentKeyBlindStabilityConfig(
        run_id=run_id,
        plaintexts=32,
        reference_keys=16,
        evaluation_keys=16,
    )


def test_key_split_and_plaintexts_are_deterministic_unique_and_disjoint() -> None:
    config = _small_config()
    first_reference, first_evaluation = _sample_keys(config)
    second_reference, second_evaluation = _sample_keys(config)
    plaintexts = _sample_plaintexts(config)

    assert first_reference == second_reference
    assert first_evaluation == second_evaluation
    assert len(set(first_reference)) == config.reference_keys
    assert len(set(first_evaluation)) == config.evaluation_keys
    assert not (set(first_reference) & set(first_evaluation))
    assert len(set(plaintexts)) == config.plaintexts


def test_vectorized_present_outputs_match_scalar_implementation() -> None:
    config = _small_config()
    reference_keys, _ = _sample_keys(config)
    plaintexts = _sample_plaintexts(config)
    keys = reference_keys[:3]
    words = plaintexts[:5]

    actual = encrypt_selected_bits_batch(
        keys,
        words,
        rounds=3,
        selected_msb_indices=SELECTED_MSB_INDICES,
    )

    for key_index, key in enumerate(keys):
        cipher = Present80(rounds=3, key=key)
        for plaintext_index, plaintext in enumerate(words):
            ciphertext = cipher.encrypt(plaintext)
            expected = [
                (ciphertext >> (63 - position)) & 1
                for position in SELECTED_MSB_INDICES
            ]
            assert actual[key_index, plaintext_index].tolist() == expected


def test_small_key_blind_audit_is_deterministic_and_protocol_valid() -> None:
    config = _small_config()

    first = audit_present_key_blind_target_stability(config)
    second = audit_present_key_blind_target_stability(config)

    assert first["rows"] == second["rows"]
    assert len(first["rows"]) == 8
    assert all(first["gate"]["protocol_checks"].values())
    assert all(first["gate"]["execution_checks"].values())
    assert first["gate"]["status"] in {"pass", "hold"}
    assert first["gate"]["next_action"]["key_blind_zero_shot_model_authorized"] is False
    assert first["metadata"]["score_contract"] == (
        "per-plaintext per-bit frequency estimated only from reference keys"
    )
    assert all(row["sample_classification"] is False for row in first["rows"])


def test_formal_config_freezes_expected_evidence_scale() -> None:
    config = PresentKeyBlindStabilityConfig()

    assert config.reference_keys * config.plaintexts == 262_144
    assert config.evaluation_keys * config.plaintexts == 262_144
    assert config.reference_keys + config.evaluation_keys == 512
    assert config.rounds == 3


def test_cli_writes_checksummed_key_blind_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "audit"

    exit_code = audit_main(
        [
            "--output-root",
            str(output),
            "--run-id",
            "test-cli-key-blind",
            "--plaintexts",
            "32",
            "--reference-keys",
            "16",
            "--evaluation-keys",
            "16",
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
    assert len(rows) == 8
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


def test_result_index_names_and_explains_key_blind_decision() -> None:
    run_id = "i2_output_prediction_opk1_present_r3_key_blind_target_stability_audit_20260722"
    decision = "innovation2_present_r3_key_blind_zero_shot_target_not_stable"

    assert "OPK1" in display_name_for_run(run_id)
    assert "跨密钥" in display_name_for_run(run_id)
    assert "不能跨未见密钥稳定预测" in DECISION_LABELS[decision]
