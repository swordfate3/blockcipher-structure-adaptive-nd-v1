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
from matplotlib import pyplot as plt

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_head_adaptation import (
    EXPECTED_ROLES,
    EXPECTED_SEEDS,
    adjudicate_head_adaptation,
    train_adaptation_seed,
)


RUN_ID = "i1_rtg1_gift_to_skinny_frozen_backbone_target_head_x2_seed0_seed1_20260724"
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
DEFAULT_SKINNY_ROOTS = (
    Path(
        "outputs/local_diagnostic/"
        "i1_rtg1_skinny64_general_gf2_attribution_t2c_2048_seed0_20260724"
    ),
    Path(
        "outputs/local_diagnostic/"
        "i1_rtg1_skinny64_general_gf2_attribution_t2c_2048_seed1_20260724"
    ),
)
DEFAULT_DEPENDENCY_GATE = Path(
    "outputs/remote_results_incomplete/"
    "i1_rtg2b_skinny64_general_gf2_scale_262144_joint_seed0_seed1_20260724/"
    "gate.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/local_diagnostic") / RUN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train only a fresh SKINNY target classifier on frozen GIFT Runtime-E4 "
            "backbones after the RTG2-B two-seed gate."
        )
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dependency-gate", type=Path, default=DEFAULT_DEPENDENCY_GATE)
    parser.add_argument(
        "--gift-roots",
        nargs=2,
        type=Path,
        default=DEFAULT_GIFT_ROOTS,
        metavar=("SEED0", "SEED1"),
    )
    parser.add_argument(
        "--skinny-roots",
        nargs=2,
        type=Path,
        default=DEFAULT_SKINNY_ROOTS,
        metavar=("SEED0", "SEED1"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dependency = _validate_dependency_gate(args.dependency_gate)
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    progress_path.unlink(missing_ok=True)
    _append_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "dependency_gate": str(args.dependency_gate),
            "dependency_decision": dependency["decision"],
        },
    )

    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        source_rows = _read_jsonl(args.gift_roots[seed] / "results.jsonl")
        target_rows = _read_jsonl(args.skinny_roots[seed] / "results.jsonl")
        train_dataset, train_paths = _load_target_split(
            args.skinny_roots[seed], seed, "train"
        )
        validation_dataset, validation_paths = _load_target_split(
            args.skinny_roots[seed], seed, "validation"
        )

        def emit(event: str, payload: dict[str, Any]) -> None:
            _append_progress(progress_path, event, {"run_id": args.run_id, **payload})

        seed_rows = train_adaptation_seed(
            seed=seed,
            source_rows=source_rows,
            target_rows=target_rows,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_paths=train_paths,
            validation_paths=validation_paths,
            checkpoint_dir=args.output_root / "checkpoints",
            device=args.device,
            progress_callback=emit,
        )
        rows.extend(seed_rows)
        _append_progress(
            progress_path,
            "seed_done",
            {"run_id": args.run_id, "seed": seed, "rows": len(seed_rows)},
        )

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
        "task": "innovation1_runtime_spn_frozen_backbone_target_head_x2",
        "dependency_gate": str(args.dependency_gate),
        "dependency": dependency,
        "source": "GIFT-64 r6 Runtime-E4 best checkpoints",
        "target": "SKINNY-64/64 r7",
        "train": "4096 total = 2048/class per seed",
        "validation": "2048 total = 1024/class per seed",
        "training": "classifier only, 198401 trainable of 442466 total, 5 epochs",
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


def _validate_dependency_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            "X2 remains closed until the local RTG2-B two-seed gate exists"
        )
    gate = json.loads(path.read_text(encoding="utf-8"))
    if (
        gate.get("phase") != "rtg2b"
        or gate.get("status") != "pass"
        or gate.get("decision") != "innovation1_rtg2b_skinny_scale_two_seed_supported"
        or not all(gate.get("protocol_checks", {}).values())
        or not all(gate.get("research_checks", {}).values())
    ):
        raise ValueError("X2 requires a complete passing RTG2-B two-seed gate")
    return gate


def _load_target_split(
    target_root: Path, seed: int, split: str
) -> tuple[DiskDifferentialDataset, dict[str, Path]]:
    expected_seed = seed if split == "train" else 10000 + seed
    split_root = target_root / "cache" / "skinny64" / "r7" / split
    matches = list(split_root.glob(f"seed-{expected_seed}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one SKINNY {split} cache for seed {seed}, got {len(matches)}"
        )
    cache_dir = matches[0]
    paths = {
        "features": cache_dir / "features.npy",
        "labels": cache_dir / "labels.npy",
        "metadata": cache_dir / "metadata.json",
    }
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    dataset = DiskDifferentialDataset(
        features=np.load(paths["features"], mmap_mode="r"),
        labels=np.load(paths["labels"], mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_dir,
    )
    return dataset, paths


def render_head_adaptation_svg(
    gate: dict[str, Any], rows: list[dict[str, Any]], output: Path
) -> None:
    labels = ("正确源", "错误源", "错误目标", "随机主干", "全量目标锚点")
    fields = (
        "candidate_auc",
        "corrupted_source_auc",
        "corrupted_target_auc",
        "random_frozen_auc",
        "full_target_anchor_auc",
    )
    colors = ("#059669", "#2563EB")
    values = [
        [float(gate["seed_results"][str(seed)][field]) for field in fields]
        for seed in EXPECTED_SEEDS
    ]
    history = {
        (int(row["seed"]), str(row["role"])): [
            float(epoch["val_auc"]) for epoch in row["history"]
        ]
        for row in rows
    }
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
            left=0.075, right=0.975, top=0.73, bottom=0.18, wspace=0.27
        )
        figure.suptitle(
            "创新1 X2：冻结 GIFT 结构主干，只训练 SKINNY 输出头",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.905,
            "SKINNY-64/64 7轮；训练 2048/class、验证 1024/class；每条样本含4对密文；5 epochs。",
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        conclusion = {
            "pass": "两颗 seed 均通过：共享主干可由目标输出头恢复，并依赖正确源/目标拓扑。",
            "hold": "至少一颗 seed 未通过完整门槛：输出头适配信号不稳定，不应扩大训练。",
            "fail": "协议或冻结边界检查失败：指标不可解释，只允许修复证据。",
        }[gate["status"]]
        figure.text(
            0.075,
            0.855,
            conclusion,
            ha="left",
            va="top",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )

        x = np.arange(len(labels))
        width = 0.36
        all_values = [value for seed_values in values for value in seed_values]
        for seed, seed_values in zip(EXPECTED_SEEDS, values, strict=True):
            bars = axes[0].bar(
                x + (seed - 0.5) * width,
                seed_values,
                width,
                color=colors[seed],
                label=f"seed{seed}",
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in seed_values],
                padding=4,
                fontsize=8.5,
                rotation=90,
            )
        axes[0].axhline(0.55, color="#DC2626", linestyle="--", label="候选门 0.55")
        axes[0].axhline(0.50, color="#334155", linestyle=":", label="随机基线 0.50")
        axes[0].set_ylim(
            min(0.47, min(all_values) - 0.025),
            max(0.66, max(all_values) + 0.045),
        )
        axes[0].set_xticks(x, labels=labels)
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("最终最佳检查点", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(loc="upper right", frameon=False, ncols=2)

        line_styles = {
            "true_source_true_target": ("正确源", "-", 2.2),
            "corrupted_source_true_target": ("错误源", "--", 1.6),
            "true_source_corrupted_target": ("错误目标", "-.", 1.6),
            "random_source_true_target": ("随机主干", ":", 1.8),
        }
        epochs = np.arange(1, 6)
        for seed in EXPECTED_SEEDS:
            for role, (label, style, linewidth) in line_styles.items():
                axes[1].plot(
                    epochs,
                    history[(seed, role)],
                    color=colors[seed],
                    linestyle=style,
                    linewidth=linewidth,
                    marker="o" if role == "true_source_true_target" else None,
                    label=f"seed{seed} {label}",
                )
        axes[1].axhline(0.50, color="#334155", linestyle=":", linewidth=1.2)
        axes[1].set_xticks(epochs)
        axes[1].set_xlabel("训练轮次（epoch）")
        axes[1].set_ylabel("验证 AUC")
        axes[1].set_title("只训练输出头的学习曲线", loc="left", fontweight="bold")
        axes[1].grid(True, color="#E5E7EB", linewidth=0.8)
        axes[1].legend(loc="best", frameon=False, ncols=2, fontsize=8.2)

        figure.text(
            0.075,
            0.065,
            "正确源=GIFT正确拓扑最佳主干；错误源=GIFT错误拓扑主干；错误目标=SKINNY错误GF(2)拓扑；全量目标锚点=同数据端到端训练。",
            ha="left",
            va="bottom",
            color="#475569",
            fontsize=9.3,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
