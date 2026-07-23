from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from blockcipher_nd.cli import run_innovation2_speck32_output_prediction as cli
from blockcipher_nd.tasks.innovation2.speck32_output_prediction_training import (
    A1_MODEL_NAMES,
    A2_NEW_MODEL_NAMES,
    FULL_MATRIX_MODEL_NAMES,
)


def test_formal_cli_runs_a1_then_a2_and_writes_standard_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "arx1"
    calls: list[str] = []
    seen: list[tuple[str, int, int]] = []
    source = {"metadata": {"key_seed": 21}}

    def prepare(config: Any, root: Path, *, progress: Any) -> dict[str, Any]:
        calls.append("prepare")
        seen.append((config.mode, config.train_rows, config.test_rows))
        return source

    def validate(config: Any, data: dict[str, Any]) -> dict[str, bool]:
        calls.append("validate")
        return {"source_valid": True}

    def train_a1(
        config: Any,
        data: dict[str, Any],
        root: Path,
        *,
        progress: Any,
    ) -> dict[str, Any]:
        calls.append("a1")
        seen.append((config.mode, config.epochs, config.batch_size))
        return {
            "history": _history(A1_MODEL_NAMES, config.epochs),
            "source_manifest": {"manifest_sha256": "a" * 64},
            "bundle_sha256": "b" * 64,
        }

    def train_a2(
        config: Any,
        data: dict[str, Any],
        root: Path,
        *,
        progress: Any,
    ) -> dict[str, Any]:
        calls.append("a2")
        return {
            "source_manifest": {"manifest_sha256": "a" * 64},
            "a1_bundle_sha256": "b" * 64,
            "bundle_sha256": "c" * 64,
            "full_matrix_rows": [
                {"model": name, "bap_avg": 0.5, "macro_auc": 0.5}
                for name in FULL_MATRIX_MODEL_NAMES
            ],
            "full_matrix_per_bit_rows": [
                {"model": name, "msb_index": bit, "auc": 0.5}
                for name in FULL_MATRIX_MODEL_NAMES
                for bit in range(32)
            ],
            "new_history": _history(A2_NEW_MODEL_NAMES, config.epochs),
            "checkpoints": [
                {"model": name, "path": f"models/{name}.pt", "sha256": "d" * 64}
                for name in FULL_MATRIX_MODEL_NAMES
            ],
            "fairness_checks": {"candidate_reused": True},
        }

    def adjudicate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("adjudicate")
        return {
            "status": "pass",
            "decision": "arx1_cli_test_pass",
            "claim_scope": "test scope",
            "next_action": {"next_adjudication": "arx1_b"},
        }

    monkeypatch.setattr(cli, "prepare_speck32_output_prediction_data", prepare)
    monkeypatch.setattr(cli, "validate_speck32_output_prediction_data", validate)
    monkeypatch.setattr(cli, "train_speck32_arx1_a1", train_a1)
    monkeypatch.setattr(cli, "train_speck32_arx1_a2", train_a2)
    monkeypatch.setattr(cli, "adjudicate_speck32_arx1_a", adjudicate)
    monkeypatch.setattr(cli, "jeong_anchor_protocols", lambda: {"fcnn": {}})
    monkeypatch.setattr(
        cli,
        "rotation_carry_protocols",
        lambda channels: {"candidate": {"channels": channels}},
    )

    exit_code = cli.main(
        [
            "--mode",
            "arx1_a",
            "--run-id",
            "arx1_cli_test",
            "--device",
            "cpu",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert calls == ["prepare", "validate", "a1", "a2", "adjudicate"]
    assert seen == [
        ("arx1_a", 1 << 20, 1 << 15),
        ("arx1_a", 100, 128),
    ]
    expected_artifacts = {
        "checkpoint_manifest.json",
        "gate.json",
        "history.csv",
        "metadata.json",
        "model_protocols.json",
        "progress.jsonl",
        "results.jsonl",
        "summary.json",
    }
    assert expected_artifacts <= {path.name for path in output_root.iterdir()}
    assert (
        len((output_root / "results.jsonl").read_text(encoding="utf-8").splitlines())
        == 5 + 5 * 32
    )
    assert (
        len((output_root / "history.csv").read_text(encoding="utf-8").splitlines())
        == 1 + 5 * 100
    )
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["sample_classification"] is False
    assert metadata["data_config"]["key_seed"] == 21
    assert metadata["training_config"]["weight_decay"] == 0.01
    assert metadata["source_manifest_sha256"] == "a" * 64


def test_invalid_source_stops_before_a1_or_a2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "prepare_speck32_output_prediction_data",
        lambda config, root, progress: {},
    )
    monkeypatch.setattr(
        cli,
        "validate_speck32_output_prediction_data",
        lambda config, source: {"source_valid": False},
    )
    monkeypatch.setattr(
        cli,
        "train_speck32_arx1_a1",
        lambda *args, **kwargs: pytest.fail("A1 must not start"),
    )
    monkeypatch.setattr(
        cli,
        "train_speck32_arx1_a2",
        lambda *args, **kwargs: pytest.fail("A2 must not start"),
    )

    with pytest.raises(ValueError, match="invalid ARX1 SPECK32 source data"):
        cli.main(["--output-root", str(tmp_path / "invalid")])


def _history(model_names: tuple[str, ...], epochs: int) -> list[dict[str, object]]:
    return [
        {"model": name, "epoch": epoch, "train_bce": 0.7}
        for name in model_names
        for epoch in range(1, epochs + 1)
    ]
