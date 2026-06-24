from __future__ import annotations

import argparse
import json
from pathlib import Path

from blockcipher_nd.engine.progress import (
    reset_progress,
    task_progress_payload,
    write_progress,
)
from blockcipher_nd.engine.task_runner import run_task
from blockcipher_nd.features.registry import FEATURE_ENCODINGS, is_supported_feature_encoding
from blockcipher_nd.planning.matrix import build_tasks


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
        "--pretrain-rounds",
        type=int,
        default=None,
        help="Optional curriculum pretraining round count before each target row.",
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
        "--progress-output",
        default=None,
        help="Optional JSONL path for run progress events.",
    )
    parser.add_argument("--output", default="outputs/innovation_one_matrix_results.jsonl")
    args = parser.parse_args(argv)
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
                    args,
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


if __name__ == "__main__":
    main()
