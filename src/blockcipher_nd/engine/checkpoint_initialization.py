from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialize_model_from_manifest(
    model: nn.Module,
    *,
    target_model: str,
    target_mapping: str,
    manifest_path: Path | None,
) -> dict[str, Any]:
    if manifest_path is None:
        return _scratch_report(model, target_model, target_mapping)
    manifest = _read_json_object(manifest_path, "initialization manifest")
    if manifest.get("version") != 1:
        raise ValueError(
            f"initialization manifest version expected=1 actual={manifest.get('version')!r}"
        )
    targets = manifest.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("initialization manifest targets must be an object")
    entry = targets.get(target_model)
    if not isinstance(entry, dict):
        raise ValueError(f"manifest target missing: {target_model}")
    if entry.get("target_mapping") != target_mapping:
        raise ValueError(
            "manifest target mapping mismatch "
            f"expected={target_mapping!r} actual={entry.get('target_mapping')!r}"
        )
    kind = entry.get("kind")
    if kind == "scratch":
        return _scratch_report(model, target_model, target_mapping)
    if kind != "checkpoint":
        raise ValueError(f"unsupported initialization kind: {kind!r}")

    checkpoint_path = _required_path(entry, "source_checkpoint")
    results_path = _required_path(entry, "source_results")
    if not checkpoint_path.is_file():
        raise ValueError(f"source checkpoint missing: {checkpoint_path}")
    actual_digest = file_sha256(checkpoint_path)
    expected_digest = _required_string(entry, "source_checkpoint_sha256")
    if actual_digest != expected_digest:
        raise ValueError(
            "source checkpoint SHA-256 mismatch "
            f"expected={expected_digest} actual={actual_digest}"
        )

    source_model = _required_string(entry, "source_model")
    source_row = _source_result_row(results_path, source_model)
    exact_fields = {
        "cipher": _required_string(entry, "source_cipher"),
        "rounds": _required_int(entry, "source_rounds"),
        "seed": _required_int(entry, "source_seed"),
        "samples_per_class": _required_int(entry, "source_samples_per_class"),
    }
    for field, expected in exact_fields.items():
        actual = source_row.get(field)
        if actual != expected:
            raise ValueError(
                f"source result {field} mismatch expected={expected!r} actual={actual!r}"
            )
    training = source_row.get("training")
    if not isinstance(training, dict):
        raise ValueError("source result training must be an object")
    source_epochs = _required_int(entry, "source_epochs")
    if training.get("epochs") != source_epochs:
        raise ValueError(
            "source result epochs mismatch "
            f"expected={source_epochs!r} actual={training.get('epochs')!r}"
        )
    recorded_checkpoint = training.get("checkpoint_output")
    if not isinstance(recorded_checkpoint, str) or (
        Path(recorded_checkpoint).resolve() != checkpoint_path.resolve()
    ):
        raise ValueError(
            "source result checkpoint path mismatch "
            f"expected={checkpoint_path} actual={recorded_checkpoint!r}"
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("source checkpoint payload must be an object")
    state_dict = payload.get("state_dict")
    metadata = payload.get("metadata")
    final_metrics = payload.get("final_metrics")
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("source checkpoint payload must contain a tensor state_dict")
    if not isinstance(metadata, dict):
        raise ValueError("source checkpoint metadata must be an object")
    if not isinstance(final_metrics, dict):
        raise ValueError("source checkpoint final_metrics must be an object")
    metadata_fields = {
        "checkpoint_output": str(checkpoint_path),
        "seed": exact_fields["seed"],
        "epochs": source_epochs,
        "selected_checkpoint": "best",
        "checkpoint_metric": "val_auc",
        "restore_best_checkpoint": True,
    }
    for field, expected in metadata_fields.items():
        actual = metadata.get(field)
        if field == "checkpoint_output" and isinstance(actual, str):
            matches = Path(actual).resolve() == checkpoint_path.resolve()
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"source checkpoint metadata {field} mismatch "
                f"expected={expected!r} actual={actual!r}"
            )
    source_metrics = source_row.get("metrics")
    if not isinstance(source_metrics, dict):
        raise ValueError("source result metrics must be an object")
    if final_metrics.get("auc") != source_metrics.get("auc"):
        raise ValueError(
            "source checkpoint/result AUC mismatch "
            f"checkpoint={final_metrics.get('auc')!r} result={source_metrics.get('auc')!r}"
        )

    model.load_state_dict(state_dict, strict=True)
    return {
        "kind": "checkpoint",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": actual_digest,
        "source_results": str(results_path),
        "source_model": source_model,
        "source_cipher": exact_fields["cipher"],
        "source_rounds": exact_fields["rounds"],
        "source_seed": exact_fields["seed"],
        "source_samples_per_class": exact_fields["samples_per_class"],
        "source_epochs": source_epochs,
        "source_mapping": _required_string(entry, "source_mapping"),
        "target_model": target_model,
        "target_mapping": target_mapping,
        "strict_state_dict_load": True,
        "state_dict_key_count": len(state_dict),
        "initial_state_sha256": state_dict_sha256(model.state_dict()),
    }


def _scratch_report(
    model: nn.Module,
    target_model: str,
    target_mapping: str,
) -> dict[str, Any]:
    return {
        "kind": "scratch",
        "target_model": target_model,
        "target_mapping": target_mapping,
        "strict_state_dict_load": False,
        "state_dict_key_count": len(model.state_dict()),
        "initial_state_sha256": state_dict_sha256(model.state_dict()),
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object: {path}")
    return value


def _source_result_row(path: Path, source_model: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"source results unreadable: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"source results invalid JSON line={line_number}: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"source results row must be an object line={line_number}")
        if row.get("selected_model") == source_model:
            rows.append(row)
    if len(rows) != 1:
        raise ValueError(
            f"source results model={source_model!r} rows={len(rows)} expected=1"
        )
    return rows[0]


def _required_string(entry: dict[str, Any], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest {field} must be a non-empty string")
    return value


def _required_path(entry: dict[str, Any], field: str) -> Path:
    return Path(_required_string(entry, field))


def _required_int(entry: dict[str, Any], field: str) -> int:
    value = entry.get(field)
    if type(value) is not int:
        raise ValueError(f"manifest {field} must be an integer")
    return value


__all__ = [
    "file_sha256",
    "initialize_model_from_manifest",
    "state_dict_sha256",
]
