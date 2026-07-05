from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.modeling import configure_structure_aware_model, infer_pair_bits
from blockcipher_nd.evaluation.neural_ensemble import (
    EnsembleScoreArtifact,
    write_score_artifact,
)
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.training.data import make_loader, select_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-sample logits and probabilities from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-plan", required=True, type=Path)
    parser.add_argument("--eval-row-index", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--hidden-bits", type=int, required=True)
    parser.add_argument(
        "--model-options",
        default=None,
        help="Optional JSON object overriding model_options from the eval plan row.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task = eval_task_from_plan(args)
    samples_per_class = validation_samples_per_class(task, args.samples_per_class)
    validation_key = task.get("validation_key")
    cipher = build_cipher(task["cipher_key"], task["rounds"], key=validation_key)
    eval_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=task["input_difference"],
            samples_per_class=samples_per_class,
            seed=int(task["seed"]) + 10_000,
            feature_encoding=task["feature_encoding"],
            pairs_per_sample=task["pairs_per_sample"],
            negative_mode=task["negative_mode"],
            key_rotation_interval=task["key_rotation_interval"],
            sample_structure=task["sample_structure"],
            integral_active_nibble=task["integral_active_nibble"],
            selected_bit_indices=task["selected_bit_indices"],
        )
    )
    pair_bits = infer_pair_bits(cipher.block_bits, task["feature_encoding"])
    model_options = model_options_from_args_or_task(args, task)
    model = build_model(
        args.model_key,
        input_bits=int(eval_dataset.features.shape[1]),
        hidden_bits=int(args.hidden_bits),
        pair_bits=pair_bits,
        structure=cipher.structure,
        model_options=model_options,
    )
    configure_structure_aware_model(model, task["cipher_key"], task["rounds"])
    load_checkpoint_state(model, args.checkpoint)
    logits, probabilities = predict_logits_and_probabilities(
        model,
        eval_dataset,
        batch_size=int(args.batch_size),
        device=args.device,
    )
    metadata = score_metadata(
        args=args,
        task=task,
        cipher_name=cipher.name,
        samples_per_class=samples_per_class,
        model_options=model_options,
    )
    artifact = EnsembleScoreArtifact(
        labels=eval_dataset.labels.astype(np.float32, copy=False),
        probabilities=probabilities,
        logits=logits,
        sample_ids=np.array([str(index) for index in range(len(eval_dataset.labels))], dtype=str),
        metadata=metadata,
    )
    write_score_artifact(args.output_dir, artifact)
    summary = {
        "status": "pass",
        "output_dir": str(args.output_dir),
        "rows": int(len(eval_dataset.labels)),
        "samples_per_class": int(task["samples_per_class"]),
        "validation_samples_per_class": int(samples_per_class),
        "model_key": args.model_key,
        "checkpoint": str(args.checkpoint),
        "claim_scope": "per-sample score artifact for frozen neural ensemble evaluation",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def eval_task_from_plan(args: argparse.Namespace) -> dict[str, Any]:
    tasks = build_tasks(
        SimpleNamespace(
            plan=str(args.eval_plan),
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=1,
            difference_profile=None,
            difference_member=0,
            key_rotation_interval=0,
            sample_structure="independent_pairs",
            integral_active_nibble=0,
        )
    )
    if args.eval_row_index < 0 or args.eval_row_index >= len(tasks):
        raise ValueError(f"eval-row-index {args.eval_row_index} outside 0..{len(tasks) - 1}")
    return dict(tasks[args.eval_row_index])


def validation_samples_per_class(task: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return int(override)
    return max(8, int(task["samples_per_class"]) // 2)


def model_options_from_args_or_task(args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    if args.model_options is not None:
        return parse_json_object(args.model_options, "--model-options")
    options = task.get("model_options") or {}
    if not isinstance(options, dict):
        raise ValueError("plan model_options must be a JSON object")
    return dict(options)


def parse_json_object(value: str, name: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def load_checkpoint_state(model: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint payload must contain a state_dict object")
    model.load_state_dict(state_dict)


def predict_logits_and_probabilities(
    model: torch.nn.Module,
    dataset: DifferentialDataset,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected_device = select_device(device)
    model = model.to(selected_device)
    model.eval()
    logits_out: list[float] = []
    probabilities_out: list[float] = []
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
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


def score_metadata(
    *,
    args: argparse.Namespace,
    task: dict[str, Any],
    cipher_name: str,
    samples_per_class: int,
    model_options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cipher": cipher_name,
        "cipher_key": task["cipher_key"],
        "rounds": int(task["rounds"]),
        "seed": int(task["seed"]),
        "samples_per_class": int(task["samples_per_class"]),
        "validation_samples_per_class": int(samples_per_class),
        "pairs_per_sample": int(task["pairs_per_sample"]),
        "feature_encoding": task["feature_encoding"],
        "negative_mode": task["negative_mode"],
        "sample_structure": task["sample_structure"],
        "difference_profile": task.get("difference_profile", ""),
        "difference_member": task.get("difference_member", ""),
        "train_key": task.get("train_key"),
        "validation_key": task.get("validation_key"),
        "checkpoint_metric": task.get("checkpoint_metric"),
        "restore_best_checkpoint": task.get("restore_best_checkpoint"),
        "model_key": args.model_key,
        "model_options": model_options,
        "run_id": "",
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata(args.checkpoint),
        "git_commit": current_git_commit(),
    }


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
