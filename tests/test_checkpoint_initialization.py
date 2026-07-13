from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from blockcipher_nd.engine.checkpoint_initialization import (
    initialize_model_from_manifest,
)
from blockcipher_nd.engine.matrix_runner import main as train_matrix
from blockcipher_nd.models.structure.spn.cross_spn_typed_cell import (
    GiftCrossSpnTypedCellTrueDistinguisher,
    PresentCrossSpnTypedCellTrueDistinguisher,
)


SOURCE_MODEL = "present_cross_spn_typed_cell_true"
TARGET_MODEL = "gift_cross_spn_typed_cell_true_from_present_true"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    torch.manual_seed(20260713)
    source = PresentCrossSpnTypedCellTrueDistinguisher(
        input_bits=16 * 128,
        base_channels=32,
    )
    checkpoint = tmp_path / "source.pt"
    metadata = {
        "checkpoint_output": str(checkpoint),
        "seed": 0,
        "epochs": 10,
        "epochs_ran": 10,
        "selected_checkpoint": "best",
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "best_epoch": 10,
        "best_checkpoint_metric": 0.7438101470470428,
    }
    final_metrics = {
        "auc": 0.7438101470470428,
        "accuracy": 0.662353515625,
        "calibrated_accuracy": 0.67,
        "loss": 0.2,
    }
    torch.save(
        {
            "state_dict": source.state_dict(),
            "history": [{"epoch": epoch} for epoch in range(1, 11)],
            "final_metrics": final_metrics,
            "metadata": metadata,
        },
        checkpoint,
    )
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "cipher": "PRESENT-80",
                "rounds": 7,
                "seed": 0,
                "samples_per_class": 8192,
                "model": SOURCE_MODEL,
                "selected_model": SOURCE_MODEL,
                "metrics": final_metrics,
                "training": metadata,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "targets": {
                    TARGET_MODEL: {
                        "kind": "checkpoint",
                        "target_mapping": "true",
                        "source_checkpoint": str(checkpoint),
                        "source_checkpoint_sha256": _sha256(checkpoint),
                        "source_results": str(results),
                        "source_model": SOURCE_MODEL,
                        "source_cipher": "PRESENT-80",
                        "source_rounds": 7,
                        "source_seed": 0,
                        "source_samples_per_class": 8192,
                        "source_epochs": 10,
                        "source_mapping": "true",
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, results, manifest


def test_checkpoint_initializer_strictly_loads_verified_source(
    tmp_path: Path,
) -> None:
    checkpoint, results, manifest = _write_source_bundle(tmp_path)
    target = GiftCrossSpnTypedCellTrueDistinguisher(
        input_bits=4 * 128,
        base_channels=32,
    )

    report = initialize_model_from_manifest(
        target,
        target_model=TARGET_MODEL,
        target_mapping="true",
        manifest_path=manifest,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert target.state_dict().keys() == payload["state_dict"].keys()
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, payload["state_dict"][key], rtol=0, atol=0)
    assert report == {
        "kind": "checkpoint",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "source_results": str(results),
        "source_model": SOURCE_MODEL,
        "source_cipher": "PRESENT-80",
        "source_rounds": 7,
        "source_seed": 0,
        "source_samples_per_class": 8192,
        "source_epochs": 10,
        "source_mapping": "true",
        "target_model": TARGET_MODEL,
        "target_mapping": "true",
        "strict_state_dict_load": True,
        "state_dict_key_count": len(target.state_dict()),
        "initial_state_sha256": report["initial_state_sha256"],
    }


def test_checkpoint_initializer_preserves_exact_source_logits(tmp_path: Path) -> None:
    _, _, manifest = _write_source_bundle(tmp_path)
    left = GiftCrossSpnTypedCellTrueDistinguisher(input_bits=4 * 128)
    right = GiftCrossSpnTypedCellTrueDistinguisher(input_bits=4 * 128)
    initialize_model_from_manifest(
        left,
        target_model=TARGET_MODEL,
        target_mapping="true",
        manifest_path=manifest,
    )
    initialize_model_from_manifest(
        right,
        target_model=TARGET_MODEL,
        target_mapping="true",
        manifest_path=manifest,
    )
    generator = torch.Generator().manual_seed(99)
    features = torch.randint(0, 2, (3, 4 * 128), generator=generator).float()

    torch.testing.assert_close(left(features), right(features), rtol=0, atol=0)


def test_checkpoint_initializer_requires_manifest_target(tmp_path: Path) -> None:
    _, _, manifest = _write_source_bundle(tmp_path)
    target = GiftCrossSpnTypedCellTrueDistinguisher(input_bits=4 * 128)

    with pytest.raises(ValueError, match="manifest target missing"):
        initialize_model_from_manifest(
            target,
            target_model="missing_target",
            target_mapping="true",
            manifest_path=manifest,
        )


def test_checkpoint_initializer_rejects_digest_mismatch(tmp_path: Path) -> None:
    _, _, manifest = _write_source_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["targets"][TARGET_MODEL]["source_checkpoint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    target = GiftCrossSpnTypedCellTrueDistinguisher(input_bits=4 * 128)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        initialize_model_from_manifest(
            target,
            target_model=TARGET_MODEL,
            target_mapping="true",
            manifest_path=manifest,
        )


def test_checkpoint_initializer_rejects_source_identity_mismatch(
    tmp_path: Path,
) -> None:
    _, results, manifest = _write_source_bundle(tmp_path)
    row = json.loads(results.read_text(encoding="utf-8"))
    row["samples_per_class"] = 4096
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")
    target = GiftCrossSpnTypedCellTrueDistinguisher(input_bits=4 * 128)

    with pytest.raises(ValueError, match="source result samples_per_class"):
        initialize_model_from_manifest(
            target,
            target_model=TARGET_MODEL,
            target_mapping="true",
            manifest_path=manifest,
        )


def test_matrix_runner_applies_manifest_before_target_training(tmp_path: Path) -> None:
    _, _, manifest = _write_source_bundle(tmp_path)
    output = tmp_path / "run" / "results.jsonl"
    progress = tmp_path / "run" / "progress.jsonl"
    checkpoints = tmp_path / "run" / "checkpoints"

    train_matrix(
        [
            "--ciphers",
            "gift64",
            "--models",
            TARGET_MODEL,
            "--rounds",
            "6",
            "--seeds",
            "0",
            "--samples-per-class",
            "4",
            "--pairs-per-sample",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-bits",
            "32",
            "--device",
            "cpu",
            "--feature-encoding",
            "ciphertext_pair_bits",
            "--negative-mode",
            "encrypted_random_plaintexts",
            "--difference-profile",
            "gift64_shen2024_spn_screen",
            "--loss",
            "mse",
            "--checkpoint-metric",
            "val_auc",
            "--restore-best-checkpoint",
            "--dataset-cache-root",
            str(tmp_path / "cache"),
            "--dataset-cache-chunk-size",
            "4",
            "--checkpoint-output-dir",
            str(checkpoints),
            "--initialization-manifest",
            str(manifest),
            "--progress-output",
            str(progress),
            "--output",
            str(output),
        ]
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    progress_rows = [
        json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()
    ]
    assert row["initialization"]["kind"] == "checkpoint"
    assert row["initialization"]["source_model"] == SOURCE_MODEL
    assert row["initialization"]["strict_state_dict_load"] is True
    initialization_events = [
        event for event in progress_rows if event.get("event") == "initialization_ready"
    ]
    assert len(initialization_events) == 1
    assert initialization_events[0]["source_model"] == SOURCE_MODEL
    assert initialization_events[0]["strict_state_dict_load"] is True
