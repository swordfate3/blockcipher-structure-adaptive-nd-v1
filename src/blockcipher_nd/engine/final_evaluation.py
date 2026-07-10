from __future__ import annotations

import argparse
from statistics import mean, pstdev
from typing import Any

from blockcipher_nd.engine.datasets import make_task_dataset
from blockcipher_nd.engine.progress import task_progress_payload, write_progress
from blockcipher_nd.engine.task_config import build_dataset_config
from blockcipher_nd.training import evaluate_binary_classifier


def run_final_evaluation(
    model,
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    cipher,
    progress_path: str | None,
    index: int | None,
    total: int | None,
) -> dict[str, Any] | None:
    repeats = int(task.get("final_test_repeats") or 0)
    samples_total = task.get("final_test_samples_total")
    if repeats == 0:
        return None
    if samples_total is None or int(samples_total) < 2:
        raise ValueError("final_test_repeats requires final_test_samples_total >= 2")

    repeat_metrics: list[dict[str, Any]] = []
    seeds: list[int] = []
    for repeat_index in range(repeats):
        seed = int(task["seed"]) + 50_000 + repeat_index
        seeds.append(seed)
        split = f"final_test_{repeat_index + 1}"
        dataset = make_task_dataset(
            build_dataset_config(
                task,
                cipher=cipher,
                samples_per_class=max(1, int(samples_total) // 2),
                samples_total=int(samples_total),
                seed=seed,
                split=split,
            ),
            args,
            task,
            split=split,
            progress_path=progress_path,
            index=index,
            total=total,
        )
        write_progress(
            progress_path,
            "final_test_start",
            {
                "index": index,
                "total": total,
                "repeat": repeat_index + 1,
                "repeats": repeats,
                "seed": seed,
                "samples_total": int(len(dataset.labels)),
                **task_progress_payload(task),
            },
        )
        metrics = evaluate_binary_classifier(
            model,
            dataset,
            batch_size=args.batch_size,
            device=args.device,
        )
        repeat_metrics.append(
            {
                "repeat": repeat_index + 1,
                "seed": seed,
                "samples_total": int(len(dataset.labels)),
                "positive_rows": int(dataset.metadata["positive_rows"]),
                "negative_rows": int(dataset.metadata["negative_rows"]),
                **metrics,
            }
        )
        write_progress(
            progress_path,
            "final_test_done",
            {
                "index": index,
                "total": total,
                **repeat_metrics[-1],
                **task_progress_payload(task),
            },
        )

    accuracies = [float(item["accuracy"]) for item in repeat_metrics]
    aucs = [float(item["auc"]) for item in repeat_metrics]
    return {
        "repeats": repeats,
        "samples_total_per_repeat": int(samples_total),
        "seeds": seeds,
        "metrics_by_repeat": repeat_metrics,
        "accuracy_mean": mean(accuracies),
        "accuracy_std": pstdev(accuracies),
        "auc_mean": mean(aucs),
        "auc_std": pstdev(aucs),
    }
