from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from blockcipher_nd.engine.progress import (
    reset_progress,
    task_progress_payload,
    write_progress,
)
from blockcipher_nd.engine.task_runner import run_task
from blockcipher_nd.features.registry import FEATURE_ENCODINGS, is_supported_feature_encoding
from blockcipher_nd.planning.matrix import build_tasks, optional_int_tuple


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run innovation-one cipher/model experiment matrix and write JSONL."
    )
    parser.add_argument("--ciphers", nargs="+", default=["speck32"])
    parser.add_argument("--models", nargs="+", default=["mlp"])
    parser.add_argument("--rounds", type=int, nargs="+", default=[2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--samples-per-class", type=int, default=256)
    parser.add_argument("--pairs-per-sample", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-bits", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device passed to PyTorch, e.g. auto, cpu, cuda, cuda:0, cuda:1.",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "adamw", "lion"],
        help="Optimizer used for neural distinguisher training.",
    )
    parser.add_argument(
        "--amsgrad",
        action="store_true",
        help="Enable AMSGrad in Adam/AdamW.",
    )
    parser.add_argument(
        "--optimizer-state-transition",
        default="reset_each_stage",
        choices=["reset_each_stage", "carry_across_stages"],
        help="Whether curriculum stages share one optimizer state.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--loss",
        default="bce",
        choices=["bce", "mse"],
        help="Training loss. Use mse for Zhang/Wang-style probability regression.",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="none",
        choices=["none", "cyclic", "cosine_warmup", "official_cyclic"],
        help="Optional learning-rate scheduler.",
    )
    parser.add_argument(
        "--max-learning-rate",
        type=float,
        default=None,
        help="Maximum learning rate for cyclic scheduling.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        default="val_accuracy",
        choices=["val_accuracy", "val_auc", "val_loss"],
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--restore-best-checkpoint",
        action="store_true",
        help="Evaluate and report the best validation checkpoint instead of the final epoch.",
    )
    parser.add_argument(
        "--checkpoint-output",
        default=None,
        help=(
            "Optional .pt path for saving the selected PyTorch checkpoint. "
            "Use with single-row runs or provide distinct paths per launch wrapper."
        ),
    )
    parser.add_argument(
        "--checkpoint-output-dir",
        default=None,
        help=(
            "Optional directory for saving one selected PyTorch checkpoint per matrix row. "
            "Use for frozen-score ensemble workflows that need aligned per-model checkpoints."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving epochs; 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum checkpoint metric improvement required to reset patience.",
    )
    parser.add_argument(
        "--train-eval-interval",
        type=int,
        default=1,
        help=(
            "Evaluate full training-set metrics every N epochs. "
            "Use 0 to skip per-epoch train-set evaluation for faster medium/large runs."
        ),
    )
    parser.add_argument(
        "--pretrain-rounds",
        type=int,
        default=None,
        help="Optional curriculum pretraining round count before each target row.",
    )
    parser.add_argument(
        "--pretrain-round-sequence",
        type=lambda value: optional_int_tuple(
            value,
            field_name="pretrain_round_sequence",
        ),
        default=(),
        help="Optional JSON round list for sequential curriculum pretraining.",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help="Optional curriculum pretraining epochs before each target row.",
    )
    parser.add_argument(
        "--feature-encoding",
        default="ciphertext_pair_bits",
        help="Feature encoding for generated ciphertext pairs.",
    )
    parser.add_argument(
        "--negative-mode",
        default="random_ciphertext",
        choices=["random_ciphertext", "encrypted_random_plaintexts"],
        help="How negative-class ciphertext pairs are generated.",
    )
    parser.add_argument(
        "--key-rotation-interval",
        type=int,
        default=0,
        help=(
            "Number of sample groups that share one random key. "
            "Use 0 for the fixed cipher key from the plan/CLI."
        ),
    )
    parser.add_argument(
        "--sample-structure",
        default="independent_pairs",
        choices=[
            "independent_pairs",
            "plaintext_integral_nibble",
            "plaintext_integral_nibble_difference_matched_negative",
            "plaintext_integral_nibble_difference_matched_negative_pair_shuffled",
            "plaintext_integral_nibble_difference_matched_negative_partial8",
            "plaintext_integral_nibble_difference_matched_negative_random_active",
            "plaintext_integral_nibble_difference_matched_negative_random_active_metadata",
            "plaintext_integral_nibble_difference_matched_negative_random_active_relative",
            "plaintext_integral_nibble_matched_negative",
            "plaintext_integral_nibble_same_difference_random_negative",
            "plaintext_integral_nibble_strict_random_negative",
            "plaintext_integral_multi_nibble_difference_matched_negative",
            "plaintext_integral_nibble_scrambled_positive",
            "zhang_wang_case2_mcnd",
            "zhang_wang_case2_independent_mcnd",
            "zhang_wang_case2_official_mcnd",
        ],
        help="How multiple pairs inside one sample are organized.",
    )
    parser.add_argument(
        "--integral-active-nibble",
        type=int,
        default=0,
        help="Active plaintext nibble index for plaintext_integral_nibble samples.",
    )
    parser.add_argument(
        "--difference-profile",
        default=None,
        help="Optional literature-backed input-difference profile.",
    )
    parser.add_argument(
        "--difference-member",
        type=int,
        default=0,
        help="Member index for multi-fixed difference profiles.",
    )
    parser.add_argument(
        "--plan",
        default=None,
        help="Optional CSV experiment matrix from configs/experiment/.",
    )
    parser.add_argument(
        "--dataset-cache-root",
        default=None,
        help="Optional root directory for chunk-generated disk-backed datasets.",
    )
    parser.add_argument(
        "--dataset-cache-chunk-size",
        type=int,
        default=8192,
        help="Rows per class generated before flushing to dataset cache.",
    )
    parser.add_argument(
        "--dataset-cache-workers",
        type=int,
        default=1,
        help=(
            "Worker processes used for disk-backed dataset generation. "
            "The default 1 preserves the historical deterministic row stream."
        ),
    )
    parser.add_argument(
        "--progress-output",
        default=None,
        help="Optional JSONL path for run progress events.",
    )
    parser.add_argument("--output", default="outputs/innovation_one_matrix_results.jsonl")
    args = parser.parse_args(argv)
    if args.checkpoint_output and args.checkpoint_output_dir:
        parser.error("--checkpoint-output and --checkpoint-output-dir are mutually exclusive")
    if not is_supported_feature_encoding(args.feature_encoding):
        examples = ", ".join(sorted(FEATURE_ENCODINGS))
        parser.error(
            f"unsupported feature encoding: {args.feature_encoding}. "
            f"Known fixed encodings include: {examples}. "
            "Parameterized PRESENT SBox-DDT encodings such as "
            "present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits are also supported."
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    if args.checkpoint_output and len(tasks) != 1:
        raise ValueError("--checkpoint-output requires a single-row run; use --checkpoint-output-dir for matrices")
    reset_progress(args.progress_output)
    write_progress(
        args.progress_output,
        "run_start",
        {
            "total": len(tasks),
            "output": str(output),
            "dataset_cache_root": args.dataset_cache_root,
        },
    )

    try:
        with output.open("w", encoding="utf-8") as handle:
            for index, task in enumerate(tasks, start=1):
                write_progress(
                    args.progress_output,
                    "row_start",
                    {
                        "index": index,
                        "total": len(tasks),
                        **task_progress_payload(task),
                    },
                )
                row = run_task(
                    task,
                    args_for_task(args, task, row_index=index),
                    progress_path=args.progress_output,
                    index=index,
                    total=len(tasks),
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                write_progress(
                    args.progress_output,
                    "row_done",
                    {
                        "index": index,
                        "total": len(tasks),
                        "accuracy": row["metrics"]["accuracy"],
                        "selected_model": row["selected_model"],
                        **task_progress_payload(task),
                    },
                )
                print(
                    "[{index}/{total}] {cipher} r={rounds} model={model} "
                    "seed={seed} pairs={pairs}".format(
                        index=index,
                        total=len(tasks),
                        cipher=row["cipher"],
                        rounds=row["rounds"],
                        model=row["model"],
                        seed=row["seed"],
                        pairs=row["pairs_per_sample"],
                    ),
                    flush=True,
                )
    except Exception as exc:
        write_progress(
            args.progress_output,
            "run_failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    write_progress(
        args.progress_output,
        "run_done",
        {
            "total": len(tasks),
            "output": str(output),
        },
    )
    print(f"wrote {len(tasks)} rows to {output}")


def args_for_task(args: argparse.Namespace, task: dict[str, Any], *, row_index: int) -> argparse.Namespace:
    checkpoint_output_dir = getattr(args, "checkpoint_output_dir", None)
    if not checkpoint_output_dir:
        return args
    checkpoint_output = checkpoint_path_for_task(Path(checkpoint_output_dir), task, row_index=row_index)
    return SimpleNamespace(**{**vars(args), "checkpoint_output": str(checkpoint_output)})


def checkpoint_path_for_task(output_dir: Path, task: dict[str, Any], *, row_index: int) -> Path:
    model = safe_checkpoint_token(str(task.get("model_key") or task.get("architecture") or "model"))
    seed = safe_checkpoint_token(f"seed{task.get('seed', 'na')}")
    return output_dir / f"row{row_index:04d}_{model}_{seed}.pt"


def safe_checkpoint_token(value: str) -> str:
    token = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    token = "_".join(part for part in token.split("_") if part)
    return token or "value"


if __name__ == "__main__":
    main()
