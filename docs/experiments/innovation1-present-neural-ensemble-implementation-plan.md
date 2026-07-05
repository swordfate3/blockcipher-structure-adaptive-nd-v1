# PRESENT Neural Ensemble Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen-score artifact path and fixed-rule ensemble evaluator for the Innovation 1 PRESENT neural ensemble aggregation route.

**Architecture:** Add a small evaluation module that writes/loads aligned per-model score artifacts, validates label/protocol identity, computes fixed ensemble metrics, and reports diversity/error-overlap. Add thin CLI wrappers under `scripts/` and `src/blockcipher_nd/cli/`; do not launch remote training in this implementation plan.

**Tech Stack:** Python, NumPy, PyTorch checkpoint loading, existing `blockcipher_nd` task/dataset/model factories, `uv run pytest`.

---

## Source Spec

Primary design document:

```text
docs/experiments/innovation1-present-neural-ensemble-aggregation-plan.md
```

This implementation must preserve the design constraints:

```text
strict encrypted_random_plaintexts negatives for PRESENT/SPN evidence
no benchmark/label/metric changes
fixed, predeclared score aggregation rules
no final-validation fitting for ensemble weights
claim scope = application-level aggregation only
```

## File Structure

Create:

```text
src/blockcipher_nd/evaluation/neural_ensemble.py
src/blockcipher_nd/cli/export_checkpoint_scores.py
src/blockcipher_nd/cli/evaluate_neural_ensemble.py
scripts/export-checkpoint-scores
scripts/evaluate-neural-ensemble
tests/test_neural_ensemble.py
tests/test_neural_ensemble_cli.py
```

Modify:

```text
src/blockcipher_nd/evaluation/__init__.py
```

Do not modify:

```text
training metrics
validation label generation
negative sample generation
plan/result alignment rules
remote launchers
```

## Task 1: Frozen Score Artifact Core

**Files:**
- Create: `src/blockcipher_nd/evaluation/neural_ensemble.py`
- Create: `tests/test_neural_ensemble.py`
- Modify: `src/blockcipher_nd/evaluation/__init__.py`

- [ ] **Step 1: Write failing tests for artifact round-trip and alignment**

Add to `tests/test_neural_ensemble.py`:

```python
from __future__ import annotations

import json

import numpy as np
import pytest

from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    evaluate_frozen_score_ensemble,
    load_score_artifact,
    write_score_artifact,
)


def artifact_metadata(model_key: str = "model_a") -> dict[str, object]:
    return {
        "cipher": "PRESENT-80",
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 8,
        "validation_samples_per_class": 4,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "train_key": "0x00000000000000000000",
        "validation_key": "0x11111111111111111111",
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "model_key": model_key,
        "model_options": {},
        "run_id": f"run_{model_key}",
        "checkpoint_path": f"/tmp/{model_key}.pt",
        "git_commit": "test",
    }


def test_score_artifact_round_trip(tmp_path):
    artifact_dir = tmp_path / "score_artifact"
    artifact = EnsembleScoreArtifact(
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
        probabilities=np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        logits=np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32),
        sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
        metadata=artifact_metadata(),
    )

    write_score_artifact(artifact_dir, artifact)
    loaded = load_score_artifact(artifact_dir)

    np.testing.assert_array_equal(loaded.labels, artifact.labels)
    np.testing.assert_allclose(loaded.probabilities, artifact.probabilities)
    np.testing.assert_allclose(loaded.logits, artifact.logits)
    np.testing.assert_array_equal(loaded.sample_ids, artifact.sample_ids)
    assert loaded.metadata["negative_mode"] == "encrypted_random_plaintexts"
    assert json.loads((artifact_dir / "models.json").read_text(encoding="utf-8"))["model_key"] == "model_a"


def test_evaluate_frozen_score_ensemble_reports_fixed_rules():
    left = EnsembleScoreArtifact(
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
        probabilities=np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32),
        logits=np.array([-2.2, -0.8, 0.8, 2.2], dtype=np.float32),
        sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
        metadata=artifact_metadata("left"),
    )
    right = EnsembleScoreArtifact(
        labels=np.array([0, 0, 1, 1], dtype=np.float32),
        probabilities=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
        logits=np.array([-1.4, -0.4, 0.4, 1.4], dtype=np.float32),
        sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
        metadata=artifact_metadata("right"),
    )

    summary = evaluate_frozen_score_ensemble([left, right])

    assert summary["status"] == "pass"
    assert [row["mode"] for row in summary["ensembles"]] == [
        "probability_mean",
        "logit_mean",
        "sum_logodds",
        "auc_positive_weighted_logit_mean",
        "rank_average",
    ]
    assert summary["best_single"]["model_key"] == "left"
    assert summary["claim_scope"].startswith("application-level")
    assert summary["diversity"]["pairwise"][0]["left"] == "left"


def test_evaluate_frozen_score_ensemble_rejects_misaligned_labels():
    left = EnsembleScoreArtifact(
        labels=np.array([0, 1], dtype=np.float32),
        probabilities=np.array([0.2, 0.8], dtype=np.float32),
        logits=np.array([-1.0, 1.0], dtype=np.float32),
        sample_ids=np.array(["0", "1"], dtype=str),
        metadata=artifact_metadata("left"),
    )
    right = EnsembleScoreArtifact(
        labels=np.array([1, 0], dtype=np.float32),
        probabilities=np.array([0.8, 0.2], dtype=np.float32),
        logits=np.array([1.0, -1.0], dtype=np.float32),
        sample_ids=np.array(["0", "1"], dtype=str),
        metadata=artifact_metadata("right"),
    )

    with pytest.raises(ValueError, match="labels differ"):
        evaluate_frozen_score_ensemble([left, right])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'blockcipher_nd.evaluation.neural_ensemble'
```

- [ ] **Step 3: Implement artifact and ensemble core**

Create `src/blockcipher_nd/evaluation/neural_ensemble.py` with these public functions:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blockcipher_nd.training.metrics import binary_auc, best_threshold_accuracy_and_threshold


@dataclass(frozen=True)
class EnsembleScoreArtifact:
    labels: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    sample_ids: np.ndarray
    metadata: dict[str, Any]


def write_score_artifact(output_dir: Path, artifact: EnsembleScoreArtifact) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_artifact_arrays(artifact)
    np.save(output_dir / "labels.npy", artifact.labels.astype(np.float32, copy=False))
    np.save(output_dir / "probabilities.npy", artifact.probabilities.astype(np.float32, copy=False))
    np.save(output_dir / "logits.npy", artifact.logits.astype(np.float32, copy=False))
    np.save(output_dir / "sample_ids.npy", artifact.sample_ids.astype(str, copy=False))
    (output_dir / "models.json").write_text(
        json.dumps(artifact.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_score_artifact(input_dir: Path) -> EnsembleScoreArtifact:
    metadata = json.loads((input_dir / "models.json").read_text(encoding="utf-8"))
    return EnsembleScoreArtifact(
        labels=np.load(input_dir / "labels.npy").astype(np.float32, copy=False),
        probabilities=np.load(input_dir / "probabilities.npy").astype(np.float32, copy=False),
        logits=np.load(input_dir / "logits.npy").astype(np.float32, copy=False),
        sample_ids=np.load(input_dir / "sample_ids.npy").astype(str, copy=False),
        metadata=metadata,
    )


def evaluate_frozen_score_ensemble(artifacts: list[EnsembleScoreArtifact]) -> dict[str, Any]:
    if len(artifacts) < 2:
        raise ValueError("neural ensemble requires at least two score artifacts")
    _validate_artifact_alignment(artifacts)
    labels = artifacts[0].labels.astype(np.float32, copy=False)
    probability_matrix = np.stack([item.probabilities for item in artifacts], axis=1)
    logit_matrix = np.stack([item.logits for item in artifacts], axis=1)
    row_reports = [_single_report(item) for item in artifacts]
    ensemble_reports = [
        _ensemble_report("probability_mean", probability_matrix.mean(axis=1)),
        _ensemble_report("logit_mean", _sigmoid(logit_matrix.mean(axis=1))),
        _ensemble_report("sum_logodds", _sigmoid(logit_matrix.sum(axis=1))),
        _ensemble_report(
            "auc_positive_weighted_logit_mean",
            _sigmoid(logit_matrix @ _auc_positive_weights(row_reports)),
        ),
        _ensemble_report("rank_average", _rank_average_probabilities(probability_matrix)),
    ]
    best_single = max(row_reports, key=lambda row: row["metrics"]["auc"])
    best_ensemble = max(ensemble_reports, key=lambda row: row["metrics"]["auc"])
    return {
        "status": "pass",
        "models": row_reports,
        "ensembles": ensemble_reports,
        "best_single": best_single,
        "best_ensemble": best_ensemble,
        "delta_best_ensemble_vs_single_auc": float(best_ensemble["metrics"]["auc"] - best_single["metrics"]["auc"]),
        "diversity": _diversity_report(labels, probability_matrix, logit_matrix, row_reports),
        "claim_scope": (
            "application-level frozen score aggregation diagnostic only; "
            "not raw single-sample SOTA, not architecture evidence by itself"
        ),
    }
```

Include helper functions in the same file:

```python
def _validate_artifact_arrays(artifact: EnsembleScoreArtifact) -> None:
    lengths = {
        len(artifact.labels),
        len(artifact.probabilities),
        len(artifact.logits),
        len(artifact.sample_ids),
    }
    if len(lengths) != 1:
        raise ValueError("labels, probabilities, logits, and sample_ids must have equal length")


def _validate_artifact_alignment(artifacts: list[EnsembleScoreArtifact]) -> None:
    first = artifacts[0]
    for artifact in artifacts[1:]:
        if not np.array_equal(first.labels, artifact.labels):
            raise ValueError("score artifact labels differ")
        if not np.array_equal(first.sample_ids, artifact.sample_ids):
            raise ValueError("score artifact sample_ids differ")
        for field in (
            "cipher",
            "rounds",
            "validation_samples_per_class",
            "pairs_per_sample",
            "feature_encoding",
            "negative_mode",
            "sample_structure",
            "difference_profile",
            "difference_member",
            "validation_key",
        ):
            if first.metadata.get(field) != artifact.metadata.get(field):
                raise ValueError(f"score artifact protocol field differs: {field}")
```

Implement the metric and diversity helpers in the same file:

```python
def _single_report(artifact: EnsembleScoreArtifact) -> dict[str, Any]:
    return {
        "model_key": str(artifact.metadata.get("model_key", "")),
        "run_id": str(artifact.metadata.get("run_id", "")),
        "checkpoint_path": str(artifact.metadata.get("checkpoint_path", "")),
        "metrics": _metrics_from_probabilities(artifact.labels, artifact.probabilities),
        "metadata": artifact.metadata,
    }


def _ensemble_report(mode: str, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return {
        "mode": mode,
        "metrics": _metrics_from_probabilities(labels, probabilities),
    }
```

Adjust the calls inside `evaluate_frozen_score_ensemble`:

```python
ensemble_reports = [
    _ensemble_report("probability_mean", labels, probability_matrix.mean(axis=1)),
    _ensemble_report("logit_mean", labels, _sigmoid(logit_matrix.mean(axis=1))),
    _ensemble_report("sum_logodds", labels, _sigmoid(logit_matrix.sum(axis=1))),
    _ensemble_report(
        "auc_positive_weighted_logit_mean",
        labels,
        _sigmoid(logit_matrix @ _auc_positive_weights(row_reports)),
    ),
    _ensemble_report("rank_average", labels, _rank_average_probabilities(probability_matrix)),
]
```

Use this exact metrics helper:

```python
def _metrics_from_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    label_array = labels.astype(np.float32, copy=False)
    probability_array = probabilities.astype(np.float32, copy=False)
    predictions = (probability_array >= 0.5).astype(np.float32)
    accuracy = float((predictions == label_array).mean()) if len(label_array) else 0.0
    calibrated_accuracy, threshold = best_threshold_accuracy_and_threshold(label_array, probability_array)
    return {
        "accuracy": accuracy,
        "advantage": 2.0 * accuracy - 1.0,
        "auc": binary_auc(label_array, probability_array),
        "best_accuracy": calibrated_accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_advantage": 2.0 * calibrated_accuracy - 1.0,
        "calibrated_threshold": threshold,
    }
```

Use these score-combination helpers:

```python
def _auc_positive_weights(row_reports: list[dict[str, Any]]) -> np.ndarray:
    weights = np.array(
        [max(0.0, float(row["metrics"]["auc"]) - 0.5) for row in row_reports],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0.0:
        return np.full((len(row_reports),), 1.0 / float(len(row_reports)), dtype=np.float64)
    return weights / total


def _rank_average_probabilities(probability_matrix: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(probability_matrix, axis=0), axis=0).astype(np.float64)
    denom = max(1.0, float(probability_matrix.shape[0] - 1))
    return ranks.mean(axis=1) / denom


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
```

Use these diversity helpers:

```python
def _diversity_report(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    logit_matrix: np.ndarray,
    row_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions = probability_matrix >= 0.5
    correct = predictions == labels.reshape(-1, 1)
    return {
        "oracle_accuracy_at_0_5": float(correct.any(axis=1).mean()) if len(labels) else 0.0,
        "all_models_wrong_rate_at_0_5": float((~correct.any(axis=1)).mean()) if len(labels) else 0.0,
        "pairwise": _pairwise_diversity(labels, probability_matrix, logit_matrix, row_reports),
    }


def _pairwise_diversity(
    labels: np.ndarray,
    probability_matrix: np.ndarray,
    logit_matrix: np.ndarray,
    row_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    predictions = probability_matrix >= 0.5
    correct = predictions == labels.reshape(-1, 1)
    for left in range(probability_matrix.shape[1]):
        for right in range(left + 1, probability_matrix.shape[1]):
            left_wrong = ~correct[:, left]
            right_wrong = ~correct[:, right]
            either_wrong = left_wrong | right_wrong
            both_wrong = left_wrong & right_wrong
            reports.append(
                {
                    "left": str(row_reports[left]["model_key"]),
                    "right": str(row_reports[right]["model_key"]),
                    "probability_correlation": _safe_correlation(
                        probability_matrix[:, left],
                        probability_matrix[:, right],
                    ),
                    "logit_correlation": _safe_correlation(logit_matrix[:, left], logit_matrix[:, right]),
                    "disagreement_rate_at_0_5": float((predictions[:, left] != predictions[:, right]).mean()),
                    "double_fault_rate_at_0_5": float(both_wrong.mean()),
                    "error_jaccard_at_0_5": (
                        float(both_wrong.sum() / either_wrong.sum()) if int(either_wrong.sum()) else 0.0
                    ),
                }
            )
    return reports


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 0.0 or right_std <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
```

- [ ] **Step 4: Export public names**

Modify `src/blockcipher_nd/evaluation/__init__.py`:

```python
from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    evaluate_frozen_score_ensemble,
    load_score_artifact,
    write_score_artifact,
)
```

Add those names to `__all__`.

- [ ] **Step 5: Run tests to verify Task 1 passes**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/blockcipher_nd/evaluation/neural_ensemble.py src/blockcipher_nd/evaluation/__init__.py tests/test_neural_ensemble.py
git commit -m "feat: add neural ensemble score artifacts"
```

## Task 2: Checkpoint Score Export CLI

**Files:**
- Create: `src/blockcipher_nd/cli/export_checkpoint_scores.py`
- Create: `scripts/export-checkpoint-scores`
- Modify: `tests/test_neural_ensemble_cli.py`

- [ ] **Step 1: Write failing CLI smoke test**

Add to `tests/test_neural_ensemble_cli.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blockcipher_nd.cli.export_checkpoint_scores import main as export_scores_main
from blockcipher_nd.cli.train import main as train_main


def write_tiny_speck_plan(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "cipher,structure,network,model_key,family,architecture_rank,score,rounds,seed,samples_per_class,pairs_per_sample,feature_encoding,negative_mode,train_key,validation_key,key_rotation_interval,sample_structure,integral_active_nibble,difference_profile,difference_member,loss,learning_rate,optimizer,weight_decay,lr_scheduler,max_learning_rate,checkpoint_metric,restore_best_checkpoint,early_stopping_patience,early_stopping_min_delta,model_options,evidence,literature",
                'SPECK32/64,ARX,Tiny-Speck-MLP,mlp,tiny,0,1,1,0,8,1,ciphertext_pair_bits,encrypted_random_plaintexts,0x1918111009080100,0x1918111009080101,0,independent_pairs,0,,,bce,0.001,adam,0,none,,val_auc,true,0,0.0,"{}","SMOKE only for neural ensemble checkpoint scoring","test"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_export_checkpoint_scores_writes_artifact(tmp_path):
    checkpoint = tmp_path / "model.pt"
    train_output = tmp_path / "train.jsonl"
    train_main(
        [
            "--ciphers", "speck32",
            "--models", "mlp",
            "--rounds", "1",
            "--seeds", "0",
            "--samples-per-class", "8",
            "--pairs-per-sample", "1",
            "--epochs", "1",
            "--batch-size", "4",
            "--hidden-bits", "8",
            "--device", "cpu",
            "--checkpoint-output", str(checkpoint),
            "--output", str(train_output),
        ]
    )
    plan = write_tiny_speck_plan(tmp_path / "eval_plan.csv")
    artifact_dir = tmp_path / "artifact"

    status = export_scores_main(
        [
            "--checkpoint", str(checkpoint),
            "--eval-plan", str(plan),
            "--eval-row-index", "0",
            "--model-key", "mlp",
            "--hidden-bits", "8",
            "--batch-size", "4",
            "--device", "cpu",
            "--output-dir", str(artifact_dir),
        ]
    )

    assert status == 0
    labels = np.load(artifact_dir / "labels.npy")
    probabilities = np.load(artifact_dir / "probabilities.npy")
    logits = np.load(artifact_dir / "logits.npy")
    metadata = json.loads((artifact_dir / "models.json").read_text(encoding="utf-8"))
    assert labels.shape == probabilities.shape == logits.shape
    assert labels.shape[0] == 16
    assert metadata["model_key"] == "mlp"
    assert metadata["negative_mode"] == "encrypted_random_plaintexts"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble_cli.py -q -k export_checkpoint_scores
```

Expected:

```text
ModuleNotFoundError: No module named 'blockcipher_nd.cli.export_checkpoint_scores'
```

- [ ] **Step 3: Implement `export_checkpoint_scores.py`**

Create `src/blockcipher_nd/cli/export_checkpoint_scores.py`. Use these boundaries:

```python
from blockcipher_nd.cli.evaluate_pairset_aggregation import checkpoint_metadata, parse_json_object
from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.modeling import configure_structure_aware_model, infer_pair_bits
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.training.data import make_loader, select_device
```

The CLI arguments must be:

```text
--checkpoint
--eval-plan
--eval-row-index
--samples-per-class
--model-key
--hidden-bits
--model-options
--batch-size
--device
--output-dir
```

Use the same plan row loading pattern as `evaluate_pairset_aggregation.eval_task_from_plan`.

Use validation seed `task["seed"] + 10_000` to match normal matrix validation split.

Use sample IDs:

```python
sample_ids = np.array([str(index) for index in range(len(eval_dataset.labels))], dtype=str)
```

Compute logits and probabilities in one helper:

```python
def predict_logits_and_probabilities(model, dataset, *, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    selected_device = select_device(device)
    model = model.to(selected_device)
    model.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    logits_out: list[float] = []
    probabilities_out: list[float] = []
    with torch.no_grad():
        for features, _labels in loader:
            features = features.to(selected_device)
            logits = model(features).squeeze(1)
            probabilities = torch.sigmoid(logits)
            logits_out.extend(float(item) for item in logits.detach().cpu().numpy())
            probabilities_out.extend(float(item) for item in probabilities.detach().cpu().numpy())
    return (
        np.array(logits_out, dtype=np.float32),
        np.array(probabilities_out, dtype=np.float32),
    )
```

Write `summary.json` beside the artifact arrays:

```python
summary = {
    "status": "pass",
    "output_dir": str(args.output_dir),
    "rows": int(len(eval_dataset.labels)),
    "metadata": metadata,
}
```

- [ ] **Step 4: Add script wrapper**

Create `scripts/export-checkpoint-scores`:

```python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockcipher_nd.cli.export_checkpoint_scores import main


if __name__ == "__main__":
    raise SystemExit(main())
```

Run:

```bash
chmod +x scripts/export-checkpoint-scores
```

- [ ] **Step 5: Run CLI smoke test**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble_cli.py -q -k export_checkpoint_scores
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/blockcipher_nd/cli/export_checkpoint_scores.py scripts/export-checkpoint-scores tests/test_neural_ensemble_cli.py
git commit -m "feat: export checkpoint score artifacts"
```

## Task 3: Frozen Artifact Ensemble CLI

**Files:**
- Create: `src/blockcipher_nd/cli/evaluate_neural_ensemble.py`
- Create: `scripts/evaluate-neural-ensemble`
- Modify: `tests/test_neural_ensemble_cli.py`

- [ ] **Step 1: Write failing test using two synthetic artifacts**

Add to `tests/test_neural_ensemble_cli.py`:

```python
from blockcipher_nd.cli.evaluate_neural_ensemble import main as evaluate_ensemble_main
from blockcipher_nd.evaluation.neural_ensemble import EnsembleScoreArtifact, write_score_artifact


def test_evaluate_neural_ensemble_cli_writes_summary(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    metadata = {
        "cipher": "PRESENT-80",
        "rounds": 7,
        "seed": 0,
        "samples_per_class": 8,
        "validation_samples_per_class": 4,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "validation_key": "0x11111111111111111111",
        "model_options": {},
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
        "run_id": "left",
        "checkpoint_path": "/tmp/left.pt",
        "git_commit": "test",
    }
    write_score_artifact(
        left_dir,
        EnsembleScoreArtifact(
            labels=np.array([0, 0, 1, 1], dtype=np.float32),
            probabilities=np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32),
            logits=np.array([-2.2, -0.8, 0.8, 2.2], dtype=np.float32),
            sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
            metadata={**metadata, "model_key": "left"},
        ),
    )
    write_score_artifact(
        right_dir,
        EnsembleScoreArtifact(
            labels=np.array([0, 0, 1, 1], dtype=np.float32),
            probabilities=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
            logits=np.array([-1.4, -0.4, 0.4, 1.4], dtype=np.float32),
            sample_ids=np.array(["0", "1", "2", "3"], dtype=str),
            metadata={**metadata, "model_key": "right", "run_id": "right"},
        ),
    )
    output = tmp_path / "ensemble_summary.json"

    status = evaluate_ensemble_main(
        [
            "--artifacts", str(left_dir), str(right_dir),
            "--output", str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert summary["best_single"]["model_key"] == "left"
    assert summary["ensembles"][0]["mode"] == "probability_mean"
    assert summary["claim_scope"].startswith("application-level")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble_cli.py -q -k evaluate_neural_ensemble
```

Expected:

```text
ModuleNotFoundError: No module named 'blockcipher_nd.cli.evaluate_neural_ensemble'
```

- [ ] **Step 3: Implement CLI**

Create `src/blockcipher_nd/cli/evaluate_neural_ensemble.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.evaluation.neural_ensemble import (
    evaluate_frozen_score_ensemble,
    load_score_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed-rule ensemble over frozen neural score artifacts."
    )
    parser.add_argument("--artifacts", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = [load_score_artifact(path) for path in args.artifacts]
    summary = evaluate_frozen_score_ensemble(artifacts)
    summary["artifact_dirs"] = [str(path) for path in args.artifacts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add script wrapper**

Create `scripts/evaluate-neural-ensemble`:

```python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockcipher_nd.cli.evaluate_neural_ensemble import main


if __name__ == "__main__":
    raise SystemExit(main())
```

Run:

```bash
chmod +x scripts/evaluate-neural-ensemble
```

- [ ] **Step 5: Run CLI test**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble_cli.py -q -k evaluate_neural_ensemble
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add src/blockcipher_nd/cli/evaluate_neural_ensemble.py scripts/evaluate-neural-ensemble tests/test_neural_ensemble_cli.py
git commit -m "feat: evaluate frozen neural ensembles"
```

## Task 4: End-To-End Local Smoke And Documentation Hook

**Files:**
- Modify: `docs/experiments/innovation1-present-neural-ensemble-aggregation-plan.md`
- Modify: `tests/test_neural_ensemble_cli.py`

- [ ] **Step 1: Add end-to-end smoke test**

Add a test that trains two tiny SPECK MLP checkpoints with different seeds, exports both artifacts against the same eval plan, and evaluates the ensemble:

```python
def test_neural_ensemble_end_to_end_tiny_smoke(tmp_path):
    plan = write_tiny_speck_plan(tmp_path / "eval_plan.csv")
    artifact_dirs = []
    for seed in [0, 1]:
        checkpoint = tmp_path / f"model_seed{seed}.pt"
        train_main(
            [
                "--ciphers", "speck32",
                "--models", "mlp",
                "--rounds", "1",
                "--seeds", str(seed),
                "--samples-per-class", "8",
                "--pairs-per-sample", "1",
                "--epochs", "1",
                "--batch-size", "4",
                "--hidden-bits", "8",
                "--device", "cpu",
                "--checkpoint-output", str(checkpoint),
                "--output", str(tmp_path / f"train_seed{seed}.jsonl"),
            ]
        )
        artifact_dir = tmp_path / f"artifact_seed{seed}"
        export_scores_main(
            [
                "--checkpoint", str(checkpoint),
                "--eval-plan", str(plan),
                "--eval-row-index", "0",
                "--model-key", "mlp",
                "--hidden-bits", "8",
                "--batch-size", "4",
                "--device", "cpu",
                "--output-dir", str(artifact_dir),
            ]
        )
        artifact_dirs.append(artifact_dir)
    output = tmp_path / "ensemble_summary.json"

    status = evaluate_ensemble_main(
        [
            "--artifacts", str(artifact_dirs[0]), str(artifact_dirs[1]),
            "--output", str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert summary["status"] == "pass"
    assert len(summary["models"]) == 2
    assert len(summary["ensembles"]) == 5
```

- [ ] **Step 2: Run end-to-end smoke**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble_cli.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 3: Update experiment plan with implemented commands**

Append a section to `docs/experiments/innovation1-present-neural-ensemble-aggregation-plan.md`:

```markdown
## Implemented Local Commands

Local artifact export shape:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/export-checkpoint-scores \
  --checkpoint <checkpoint.pt> \
  --eval-plan <same_protocol_eval_plan.csv> \
  --eval-row-index 0 \
  --model-key <model_key> \
  --hidden-bits <hidden_bits> \
  --batch-size 256 \
  --device cpu \
  --output-dir outputs/<run_id>/score_artifacts/<model_id>
```

Local ensemble evaluation shape:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-neural-ensemble \
  --artifacts outputs/<run_id>/score_artifacts/model_a outputs/<run_id>/score_artifacts/model_b \
  --output outputs/<run_id>/neural_ensemble_summary.json
```

These commands are local/smoke-ready only until a separate remote readiness
plan defines checkpoint paths and G:\lxy artifact roots for medium+ runs.
```

- [ ] **Step 4: Run focused tests and diff check**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble.py tests/test_neural_ensemble_cli.py -q
git diff --check
```

Expected:

```text
all tests pass
git diff --check emits no output
```

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add docs/experiments/innovation1-present-neural-ensemble-aggregation-plan.md tests/test_neural_ensemble_cli.py
git commit -m "test: smoke neural ensemble artifact flow"
```

## Task 5: Final Verification And Push

**Files:**
- No new files.

- [ ] **Step 1: Run relevant verification**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_neural_ensemble.py tests/test_neural_ensemble_cli.py tests/test_pairset_aggregation.py tests/test_pairset_aggregation_cli.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_project_structure.py -q -k "neural_ensemble or projection_ensemble or pairset_aggregation"
git diff --check
```

Expected:

```text
all selected tests pass
git diff --check emits no output
```

- [ ] **Step 2: Inspect workspace**

Run:

```bash
git status --short --branch
```

Expected before push:

```text
## main...origin/main [ahead N]
```

- [ ] **Step 3: Push commits**

Run:

```bash
git push origin main
```

If sandbox networking blocks the push, rerun the same command with escalation.

- [ ] **Step 4: Report status**

Report:

```text
implemented artifact writer
implemented checkpoint score export CLI
implemented frozen score ensemble CLI
local smoke/tests passed
commit hashes
push status
no remote training launched
next action = inventory existing checkpoints for retrospective PRESENT r7/r8 ensemble
```

## Self-Review

Spec coverage:

```text
artifact schema: Task 1 and Task 2
fixed ensemble rules: Task 1 and Task 3
label/protocol alignment: Task 1
diversity/error-overlap: Task 1
local smoke: Task 4
docs update: Task 4
no remote launch: stated in source spec and Task 5 report
```

Placeholder scan:

```text
No placeholder-red-flag steps are required for implementation.
All commands use exact paths and expected outcomes.
```

Type consistency:

```text
Core artifact type is EnsembleScoreArtifact.
Public functions are write_score_artifact, load_score_artifact, evaluate_frozen_score_ensemble.
CLI names are export-checkpoint-scores and evaluate-neural-ensemble.
```
