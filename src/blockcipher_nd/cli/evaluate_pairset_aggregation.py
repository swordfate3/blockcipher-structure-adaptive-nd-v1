from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from blockcipher_nd.data.differential import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.modeling import configure_structure_aware_model, infer_pair_bits
from blockcipher_nd.evaluation.pairset_aggregation import (
    PairSetAggregationConfig,
    pairset_aggregation_metrics,
)
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen single-pair scorer by aggregating scores over pair-set samples."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-plan", required=True, type=Path)
    parser.add_argument("--eval-row-index", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--pairs-per-sample", type=int, default=16)
    parser.add_argument("--scorer-model-key", required=True)
    parser.add_argument("--scorer-hidden-bits", type=int, required=True)
    parser.add_argument("--scorer-model-options", default="{}")
    parser.add_argument("--scorer-pairs-per-sample", type=int, default=1)
    parser.add_argument("--aggregation-mode", default="sum_logodds")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--lse-temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.scorer_pairs_per_sample) != 1:
        raise ValueError("frozen aggregation control currently requires a single-pair scorer")
    eval_task = eval_task_from_plan(args)
    samples_per_class = (
        int(args.samples_per_class)
        if args.samples_per_class is not None
        else int(eval_task["samples_per_class"])
    )
    validation_key = eval_task.get("validation_key")
    cipher = build_cipher(eval_task["cipher_key"], eval_task["rounds"], key=validation_key)
    pair_bits = infer_pair_bits(cipher.block_bits, eval_task["feature_encoding"])
    if pair_bits is None:
        raise ValueError(f"cannot infer pair bits for {eval_task['feature_encoding']}")

    scorer = build_frozen_scorer(
        args,
        input_bits=pair_bits * int(args.scorer_pairs_per_sample),
        pair_bits=pair_bits,
        structure=cipher.structure,
        cipher_key=eval_task["cipher_key"],
        rounds=eval_task["rounds"],
    )
    eval_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=eval_task["input_difference"],
            samples_per_class=samples_per_class,
            seed=int(eval_task["seed"]) + 20_000,
            feature_encoding=eval_task["feature_encoding"],
            pairs_per_sample=int(args.pairs_per_sample),
            negative_mode=eval_task["negative_mode"],
            key_rotation_interval=eval_task["key_rotation_interval"],
            sample_structure=eval_task["sample_structure"],
            integral_active_nibble=eval_task["integral_active_nibble"],
            selected_bit_indices=eval_task["selected_bit_indices"],
        )
    )
    aggregation = PairSetAggregationConfig(
        pair_bits=pair_bits,
        pairs_per_sample=int(args.pairs_per_sample),
        mode=args.aggregation_mode,
        top_k=int(args.top_k),
        lse_temperature=float(args.lse_temperature),
    )
    metrics = pairset_aggregation_metrics(
        scorer,
        eval_dataset,
        aggregation,
        batch_size=int(args.batch_size),
        device=args.device,
    )
    summary = {
        "status": "pass",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata(args.checkpoint),
        "eval_plan": str(args.eval_plan),
        "eval_row_index": int(args.eval_row_index),
        "samples_per_class": samples_per_class,
        "rows": int(len(eval_dataset.labels)),
        "cipher": cipher.name,
        "cipher_key": eval_task["cipher_key"],
        "rounds": eval_task["rounds"],
        "seed": eval_task["seed"],
        "validation_key": validation_key,
        "difference_profile": eval_task.get("difference_profile", ""),
        "sample_structure": eval_task["sample_structure"],
        "negative_mode": eval_task["negative_mode"],
        "feature_encoding": eval_task["feature_encoding"],
        "pair_bits": pair_bits,
        "pairs_per_sample": int(args.pairs_per_sample),
        "scorer_model_key": args.scorer_model_key,
        "scorer_hidden_bits": int(args.scorer_hidden_bits),
        "scorer_pairs_per_sample": int(args.scorer_pairs_per_sample),
        "aggregation": aggregation.__dict__,
        "metrics": metrics,
        "claim_scope": "frozen single-pair score aggregation control; not a learned pair-set model",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def eval_task_from_plan(args: argparse.Namespace) -> dict[str, Any]:
    tasks = build_tasks(
        SimpleNamespace(
            plan=str(args.eval_plan),
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=int(args.pairs_per_sample),
            difference_profile=None,
            difference_member=0,
            key_rotation_interval=0,
            sample_structure="independent_pairs",
            integral_active_nibble=0,
        )
    )
    if args.eval_row_index < 0 or args.eval_row_index >= len(tasks):
        raise ValueError(f"eval-row-index {args.eval_row_index} outside 0..{len(tasks) - 1}")
    task = dict(tasks[args.eval_row_index])
    task["pairs_per_sample"] = int(args.pairs_per_sample)
    return task


def build_frozen_scorer(
    args: argparse.Namespace,
    *,
    input_bits: int,
    pair_bits: int,
    structure: str,
    cipher_key: str,
    rounds: int,
):
    options = parse_json_object(args.scorer_model_options, "--scorer-model-options")
    model = build_model(
        args.scorer_model_key,
        input_bits=input_bits,
        hidden_bits=int(args.scorer_hidden_bits),
        pair_bits=pair_bits,
        structure=structure,
        model_options=options,
    )
    configure_structure_aware_model(model, cipher_key, rounds)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint payload must contain a state_dict object")
    model.load_state_dict(state_dict)
    return model


def parse_json_object(value: str, name: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
