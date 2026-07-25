from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import pyplot as plt

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_zero_step import (
    file_sha256,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_gift_to_dialga_head_adaptation import (
    EXPECTED_ROLES,
    EXPECTED_SEEDS,
    adjudicate_head_adaptation,
    adjudicate_readiness,
    audit_role_readiness,
    audit_strict_load_matrix,
    cache_evidence,
    cache_tree_snapshot,
    deterministic_classifier_state,
    train_adaptation_seed,
    validate_d1_evidence,
    validate_source_evidence,
)


RUN_ID = "i1_rtg1_gift_to_dialga_frozen_backbone_target_head_x3_seed0_seed1_20260725"
DEFAULT_GIFT_ROOTS = (
    Path(
        "outputs/local_diagnostic/"
        "i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0"
    ),
    Path(
        "outputs/local_diagnostic/"
        "i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed1"
    ),
)
DEFAULT_DIALGA_ROOT = Path(
    "outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/local_diagnostic") / RUN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze GIFT-64 Runtime-E4 backbones and train only the existing "
            "Dialga-128 four-round classifier after an evidence-first readiness gate."
        )
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gift-roots",
        nargs=2,
        type=Path,
        default=DEFAULT_GIFT_ROOTS,
        metavar=("SEED0", "SEED1"),
    )
    parser.add_argument("--dialga-root", type=Path, default=DEFAULT_DIALGA_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--readiness-only",
        action="store_true",
        help="Persist readiness evidence and stop before performance training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    progress_path.unlink(missing_ok=True)
    _append_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source": [str(path) for path in args.gift_roots],
            "target": str(args.dialga_root),
        },
    )

    try:
        source_payloads, source_evidence = validate_source_evidence(
            tuple(args.gift_roots)
        )
        target_rows, d1_evidence = validate_d1_evidence(args.dialga_root)
        cache_root = args.dialga_root / "cache" / "dialga128" / "r4"
        before = cache_tree_snapshot(cache_root)
        datasets: dict[int, dict[str, DiskDifferentialDataset]] = {}
        paths_by_seed: dict[int, dict[str, dict[str, Path]]] = {}
        role_audits: list[dict[str, Any]] = []
        classifier_state = deterministic_classifier_state()
        strict_load_audits = audit_strict_load_matrix(source_payloads)
        for seed in EXPECTED_SEEDS:
            train_dataset, train_paths = _load_target_split(
                args.dialga_root, seed, "train"
            )
            validation_dataset, validation_paths = _load_target_split(
                args.dialga_root, seed, "validation"
            )
            datasets[seed] = {
                "train": train_dataset,
                "validation": validation_dataset,
            }
            paths_by_seed[seed] = {
                "train": train_paths,
                "validation": validation_paths,
            }
            features = torch.as_tensor(
                np.asarray(train_dataset.features[:16]).copy(),
                dtype=torch.float32,
            )
            labels = torch.as_tensor(
                np.asarray(train_dataset.labels[:16]).copy(),
                dtype=torch.float32,
            )
            source_state_dicts = {
                role: payload["state_dict"]
                for role, payload in source_payloads[seed].items()
            }
            for role in EXPECTED_ROLES:
                role_audits.append(
                    audit_role_readiness(
                        seed=seed,
                        role=role,
                        source_state_dicts=source_state_dicts,
                        classifier_state=classifier_state,
                        features=features,
                        labels=labels,
                    )
                )
        after = cache_tree_snapshot(cache_root)
        readiness = adjudicate_readiness(
            run_id=args.run_id,
            role_audits=role_audits,
            source_evidence=source_evidence,
            d1_evidence=d1_evidence,
            cache_evidence=cache_evidence(
                paths_by_seed=paths_by_seed,
                before=before,
                after=after,
            ),
            strict_load_audits=strict_load_audits,
        )
    except Exception as exc:
        readiness = {
            "run_id": args.run_id,
            "task": "innovation1_runtime_spn_gift_to_dialga_x3_readiness",
            "status": "fail",
            "decision": "runtime_spn_gift_to_dialga_x3_readiness_not_supported",
            "checks": {"readiness_execution": False},
            "error_type": type(exc).__name__,
            "error": str(exc),
            "next_action": (
                "stop before training; do not add compatibility layers or regenerate data"
            ),
        }
        _write_json(args.output_root / "readiness.json", readiness)
        _write_json(args.output_root / "gate.json", readiness)
        _append_progress(
            progress_path,
            "readiness_failed",
            {
                "run_id": args.run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 1

    readiness_path = args.output_root / "readiness.json"
    _write_json(readiness_path, readiness)
    _append_progress(
        progress_path,
        "readiness_done",
        {
            "run_id": args.run_id,
            "status": readiness["status"],
            "decision": readiness["decision"],
        },
    )
    if readiness["status"] != "pass":
        _write_json(args.output_root / "gate.json", readiness)
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 1
    if args.readiness_only:
        print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
        return 0

    readiness_sha256 = file_sha256(readiness_path)
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:

        def emit(event: str, payload: dict[str, Any]) -> None:
            _append_progress(progress_path, event, {"run_id": args.run_id, **payload})

        rows.extend(
            train_adaptation_seed(
                seed=seed,
                source_payloads=source_payloads[seed],
                source_evidence=source_evidence[f"seed{seed}"],
                target_rows=target_rows,
                train_dataset=datasets[seed]["train"],
                validation_dataset=datasets[seed]["validation"],
                train_paths=paths_by_seed[seed]["train"],
                validation_paths=paths_by_seed[seed]["validation"],
                readiness_sha256=readiness_sha256,
                checkpoint_dir=args.output_root / "checkpoints",
                device=args.device,
                progress_callback=emit,
            )
        )
        _append_progress(
            progress_path,
            "seed_done",
            {"run_id": args.run_id, "seed": seed, "rows": len(EXPECTED_ROLES)},
        )

    final_cache = cache_tree_snapshot(cache_root)
    initial_cache_sha256 = readiness["cache_evidence"]["tree_sha256_before"]
    final_cache_sha256 = _snapshot_sha256(final_cache)
    cache_unchanged = bool(
        readiness["cache_evidence"]["tree_sha256_before"] == final_cache_sha256
        and readiness["cache_evidence"]["file_count_before"] == len(final_cache)
    )
    for row in rows:
        row["target_cache_tree_sha256_before"] = initial_cache_sha256
        row["target_cache_tree_sha256_after"] = final_cache_sha256
        row["target_cache_unchanged"] = cache_unchanged

    gate = adjudicate_head_adaptation(run_id=args.run_id, rows=rows)
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "result_rows": len(rows),
        "expected_rows": len(EXPECTED_SEEDS) * len(EXPECTED_ROLES),
        "checks": gate["protocol_checks"],
        "errors": [
            name for name, passed in gate["protocol_checks"].items() if not passed
        ],
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_gift_to_dialga_frozen_backbone_x3",
        "source": "GIFT-64 r6 Runtime-E4 restored-best checkpoints",
        "target": "Dialga-128 prefix-r4 exact D1 cache",
        "train": "4096 total = 2048/class per seed",
        "validation": "2048 total = 1024/class per seed",
        "training": "classifier only, 198401 trainable of 442466 total, 5 epochs",
        "readiness": readiness,
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_history_csv(args.output_root / "history.csv", rows)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_head_adaptation_svg(gate, rows, args.output_root / "curves.svg")
    _append_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def _load_target_split(
    target_root: Path,
    seed: int,
    split: str,
) -> tuple[DiskDifferentialDataset, dict[str, Path]]:
    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported X3 target split: {split}")
    expected_seed = seed if split == "train" else 10_000 + seed
    split_root = target_root / "cache" / "dialga128" / "r4" / split
    matches = list(split_root.glob(f"seed-{expected_seed}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one Dialga {split} cache for seed {seed}, got {len(matches)}"
        )
    cache_dir = matches[0]
    paths = {
        "features": cache_dir / "features.npy",
        "labels": cache_dir / "labels.npy",
        "metadata": cache_dir / "metadata.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError(f"Dialga {split} cache for seed {seed} is incomplete")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    dataset = DiskDifferentialDataset(
        features=np.load(paths["features"], mmap_mode="r"),
        labels=np.load(paths["labels"], mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_dir,
    )
    return dataset, paths


def render_head_adaptation_svg(
    gate: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    role_labels = ("正确源+正确目标", "错误源", "错误目标", "随机主干")
    role_fields = (
        "candidate_auc",
        "corrupted_source_auc",
        "corrupted_target_auc",
        "random_frozen_auc",
    )
    margin_labels = ("减错误源", "减错误目标", "减随机主干")
    margin_fields = (
        "candidate_minus_source_auc",
        "candidate_minus_target_auc",
        "candidate_minus_random_auc",
    )
    colors = ("#047857", "#2563EB")
    role_values = [
        [float(gate["seed_results"][str(seed)][field]) for field in role_fields]
        for seed in EXPECTED_SEEDS
    ]
    margin_values = [
        [float(gate["seed_results"][str(seed)][field]) for field in margin_fields]
        for seed in EXPECTED_SEEDS
    ]
    anchors = [
        float(gate["seed_results"][str(seed)]["full_target_anchor_auc"])
        for seed in EXPECTED_SEEDS
    ]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.5, 8.8))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.72,
            bottom=0.22,
            wspace=0.27,
        )
        figure.suptitle(
            "创新1 X3：把 GIFT 结构主干迁移到 Dialga，只训练分类头",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.905,
            (
                "Dialga-128 前缀4轮；训练 2048/class、验证 1024/class；"
                "每条样本含4对密文；冻结主干，分类头训练5轮。"
            ),
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        conclusion = {
            "pass": "两颗 seed 均通过：GIFT 学到的共享主干可迁移到 128-bit 异构 Dialga。",
            "hold": "至少一颗 seed 未通过完整归因门：停止扩大 X3，保留已有 X2 边界。",
            "fail": "协议或冻结边界检查失败：结果不可解释，只允许修复证据。",
        }[gate["status"]]
        figure.text(
            0.075,
            0.85,
            conclusion,
            ha="left",
            va="top",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )

        x = np.arange(len(role_labels))
        width = 0.36
        flat_roles = [value for values in role_values for value in values]
        lower = min(0.48, min(flat_roles) - 0.025)
        upper = max(0.60, max(flat_roles) + 0.045)
        for seed, values in zip(EXPECTED_SEEDS, role_values, strict=True):
            bars = axes[0].bar(
                x + (seed - 0.5) * width,
                values,
                width,
                color=colors[seed],
                label=f"seed{seed}",
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in values],
                padding=3,
                fontsize=8.4,
                rotation=90,
            )
        axes[0].axhline(0.55, color="#DC2626", linestyle="--", label="候选门 0.55")
        axes[0].axhline(0.50, color="#334155", linestyle=":", label="随机基线 0.50")
        axes[0].set_ylim(lower, upper)
        axes[0].set_xticks(x, labels=role_labels)
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("同预算四角色结果", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)

        margin_x = np.arange(len(margin_labels))
        flat_margins = [value for values in margin_values for value in values]
        max_abs = max(0.015, max(abs(value) for value in flat_margins) + 0.01)
        for seed, values in zip(EXPECTED_SEEDS, margin_values, strict=True):
            bars = axes[1].bar(
                margin_x + (seed - 0.5) * width,
                values,
                width,
                color=colors[seed],
                label=f"seed{seed}",
            )
            axes[1].bar_label(
                bars,
                labels=[f"{value:+.4f}" for value in values],
                padding=3,
                fontsize=8.5,
                rotation=90,
            )
        axes[1].axhline(0.005, color="#DC2626", linestyle="--", label="通过门 +0.005")
        axes[1].axhline(0.0, color="#334155", linewidth=1.1)
        axes[1].set_ylim(-max_abs, max_abs)
        axes[1].set_xticks(margin_x, labels=margin_labels)
        axes[1].set_ylabel("候选 AUC 减控制 AUC")
        axes[1].set_title(
            "结构归因边际（通过门 +0.005，越高越好）",
            loc="left",
            fontweight="bold",
        )
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)

        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.125),
            frameon=False,
            ncols=4,
        )
        figure.text(
            0.075,
            0.065,
            (
                "正确源=GIFT正确拓扑最佳主干；错误源=GIFT错误拓扑主干；"
                "错误目标=Dialga错误扩散拓扑；随机主干=未训练主干。 "
                f"上下文锚点：Dialga端到端 Runtime-E4 seed0={anchors[0]:.4f}，"
                f"seed1={anchors[1]:.4f}（不参与X3通过门）。"
            ),
            ha="left",
            va="bottom",
            color="#475569",
            fontsize=9.2,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


def _snapshot_sha256(snapshot: dict[str, dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path, metadata in sorted(snapshot.items()):
        digest.update(path.encode("utf-8"))
        digest.update(str(metadata["size"]).encode("ascii"))
        digest.update(str(metadata["mtime_ns"]).encode("ascii"))
        digest.update(str(metadata["sha256"]).encode("ascii"))
    return digest.hexdigest()


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "role",
        "epoch",
        "train_loss",
        "train_auc",
        "val_loss",
        "val_auc",
        "val_accuracy",
        "learning_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for epoch in row["history"]:
                writer.writerow(
                    {
                        "seed": row["seed"],
                        "role": row["role"],
                        **{field: epoch[field] for field in fieldnames[2:]},
                    }
                )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["main", "render_head_adaptation_svg"]
