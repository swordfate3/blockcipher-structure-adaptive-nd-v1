from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blockcipher_nd.cli import (
    run_innovation2_gift64_output_prediction_discovery as cli,
)
from blockcipher_nd.tasks.innovation2.gift64_output_prediction_discovery import (
    MODEL_NAMES,
)


def test_formal_cli_freezes_candidates_before_fresh_and_writes_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "gx1"
    source = {
        "secret_key": 7,
        "metadata": {"key_seed": 11},
        "discovery_features": np.zeros((1, 64), dtype=np.float32),
        "discovery_targets": np.zeros((1, 64), dtype=np.float32),
    }
    fresh = {
        "features": np.zeros((1, 64), dtype=np.float32),
        "full_targets": np.zeros((1, 64), dtype=np.float32),
    }
    calls: list[str] = []
    seen_configs: list[tuple[str, str, int, int]] = []

    def prepare_source(config: Any, root: Path, *, progress: Any) -> dict[str, Any]:
        calls.append("prepare_source")
        seen_configs.append(
            (config.mode, "data", config.train_rows, config.discovery_rows)
        )
        return source

    def validate_source(config: Any, data: dict[str, Any]) -> dict[str, bool]:
        calls.append("validate_source")
        return {"source_valid": True}

    def train(
        config: Any,
        data: dict[str, Any],
        root: Path,
        *,
        progress: Any,
    ) -> dict[str, Any]:
        calls.append("train")
        seen_configs.append((config.mode, "training", config.epochs, config.batch_size))
        return {
            "rows": [{"model": model} for model in MODEL_NAMES],
            "history": [{"model": model, "epoch": 1} for model in MODEL_NAMES],
            "checkpoints": [
                {"model": model, "path": f"models/{model}.pt", "sha256": "b" * 64}
                for model in MODEL_NAMES
            ],
        }

    def evaluate(
        config: Any,
        root: Path,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        split: str,
        progress: Any,
    ) -> dict[str, Any]:
        calls.append(f"evaluate_{split}")
        return {
            "per_bit_rows": _metric_rows(split),
            "full_output_rows": [
                {"model": model, "split": split, "test_exact_match": 0.0}
                for model in MODEL_NAMES
            ],
        }

    def select(config: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append("select")
        return _candidates()

    def prepare_fresh(
        config: Any,
        data: dict[str, Any],
        root: Path,
        *,
        candidate_sha256: str,
        progress: Any,
    ) -> dict[str, Any]:
        calls.append("prepare_fresh")
        assert (root / "candidates.json").exists()
        assert candidate_sha256 in (root / "candidates.sha256").read_text(
            encoding="ascii"
        )
        events = [
            json.loads(line)["event"]
            for line in (root / "progress.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert "candidates_frozen_before_fresh_generation" in events
        return fresh

    def validate_fresh(
        config: Any,
        source_data: dict[str, Any],
        fresh_data: dict[str, Any],
        *,
        candidate_sha256: str,
    ) -> dict[str, bool]:
        calls.append("validate_fresh")
        return {"fresh_valid": True}

    def adjudicate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("adjudicate")
        return {
            "status": "pass",
            "decision": "gx1_test_pass",
            "metrics": {"fresh_confirmed_msb_indices": [0, 1, 2, 3]},
            "claim_scope": "test scope",
            "next_action": {"next_adjudication": "gx2"},
        }

    monkeypatch.setattr(cli, "prepare_gift64_source_data", prepare_source)
    monkeypatch.setattr(cli, "validate_gift64_source_data", validate_source)
    monkeypatch.setattr(cli, "train_gift64_discovery_matrix", train)
    monkeypatch.setattr(cli, "evaluate_gift64_output_split", evaluate)
    monkeypatch.setattr(cli, "select_gift64_discovery_candidates", select)
    monkeypatch.setattr(cli, "prepare_gift64_fresh_data", prepare_fresh)
    monkeypatch.setattr(cli, "validate_gift64_fresh_data", validate_fresh)
    monkeypatch.setattr(cli, "adjudicate_gift64_discovery", adjudicate)

    exit_code = cli.main(
        [
            "--mode",
            "formal",
            "--run-id",
            "gx1_cli_test",
            "--device",
            "cpu",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert seen_configs == [
        ("formal", "data", 1 << 17, 1 << 16),
        ("formal", "training", 100, 250),
    ]
    assert calls == [
        "prepare_source",
        "validate_source",
        "train",
        "evaluate_discovery",
        "select",
        "prepare_fresh",
        "validate_fresh",
        "evaluate_fresh_confirmation",
        "adjudicate",
    ]
    expected_artifacts = {
        "candidates.json",
        "candidates.sha256",
        "checkpoint_manifest.json",
        "gate.json",
        "history.csv",
        "metadata.json",
        "progress.jsonl",
        "ranking.csv",
        "results.jsonl",
        "summary.json",
    }
    assert expected_artifacts <= {path.name for path in output_root.iterdir()}
    results = (output_root / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 2 + 2 * 2 + 2 * 2 * 64
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["sample_classification"] is False
    assert metadata["data_config"]["key_seed"] == 11
    assert metadata["training_config"]["epochs"] == 100


def test_invalid_source_stops_before_training_or_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "prepare_gift64_source_data",
        lambda config, root, progress: {},
    )
    monkeypatch.setattr(
        cli,
        "validate_gift64_source_data",
        lambda config, source: {"source_valid": False},
    )
    monkeypatch.setattr(
        cli,
        "train_gift64_discovery_matrix",
        lambda *args, **kwargs: pytest.fail("training must not start"),
    )
    monkeypatch.setattr(
        cli,
        "prepare_gift64_fresh_data",
        lambda *args, **kwargs: pytest.fail("fresh data must not be prepared"),
    )

    with pytest.raises(ValueError, match="invalid GX1 source data"):
        cli.main(["--output-root", str(tmp_path / "invalid")])


def _metric_rows(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for bit in range(64):
            strong = model == MODEL_NAMES[0] and bit < 8
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "msb_index": bit,
                    "auc": 0.52 if strong else 0.5,
                    "accuracy_minus_majority": 0.015 if strong else 0.0,
                }
            )
    return rows


def _candidates() -> dict[str, Any]:
    ranking = [
        {
            "msb_index": bit,
            "eligible": bit < 8,
            "selection_score": 0.015 if bit < 8 else 0.0,
        }
        for bit in range(64)
    ]
    return {
        "candidate_msb_indices": list(range(8)),
        "candidates": ranking[:8],
        "all_64_discovery_ranking": ranking,
        "confirmation_split": "fresh_not_generated_or_read_when_frozen",
    }
