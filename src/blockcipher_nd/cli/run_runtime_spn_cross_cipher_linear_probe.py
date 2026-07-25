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

from blockcipher_nd.cli.run_runtime_spn_skinny_rectangle_transfer import (
    DEFAULT_SOURCE_ROOT,
    DEFAULT_TARGET_ROOT,
    _load_target_split,
    validate_authorities,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_cross_cipher_linear_probe import (
    EXPECTED_ROLES,
    RUN_ID,
    TARGET_SEEDS,
    adjudicate_linear_probe_panel,
    train_linear_probe_panel,
    verify_linear_probe_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe frozen formal SKINNY RuntimeE4 representations on RECTANGLE "
            "with a 385-parameter linear readout and strict topology controls."
        )
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root or Path("outputs/local_diagnostic") / args.run_id
    if args.verify_only:
        if not output_root.is_dir():
            raise ValueError(f"X4 output root does not exist: {output_root}")
        verification = verify_linear_probe_artifacts(
            output_root=output_root,
            rows=_read_jsonl(output_root / "results.jsonl"),
        )
        _write_json(output_root / "artifact-verification.json", verification)
        print(json.dumps(verification, ensure_ascii=False, sort_keys=True))
        return 0 if verification["status"] == "pass" else 1
    authority = validate_authorities(args.source_root, args.target_root)
    if output_root.exists():
        raise ValueError(f"X4 output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    progress_path = output_root / "progress.jsonl"
    _append_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_root": str(args.source_root),
            "target_root": str(args.target_root),
            "target_seeds": list(TARGET_SEEDS),
            "authority": authority,
        },
    )

    source_rows = _read_jsonl(args.source_root / "results.jsonl")
    target_rows = _read_jsonl(args.target_root / "results.jsonl")
    source_checkpoint_paths = {
        "true": args.source_root
        / "checkpoints/row0001_skinny64_runtime_e4_equivariant_true_seed0.pt",
        "corrupted": args.source_root
        / "checkpoints/row0002_skinny64_runtime_e4_equivariant_corrupted_seed0.pt",
    }
    target_datasets: dict[int, dict[str, Any]] = {}
    target_paths: dict[int, dict[str, Any]] = {}
    for seed in TARGET_SEEDS:
        train_dataset, train_paths = _load_target_split(
            args.target_root,
            split="train",
            expected_seed=seed,
        )
        validation_dataset, validation_paths = _load_target_split(
            args.target_root,
            split="validation",
            expected_seed=seed + 10_000,
        )
        target_datasets[seed] = {
            "train": train_dataset,
            "validation": validation_dataset,
        }
        target_paths[seed] = {
            "train": train_paths,
            "validation": validation_paths,
        }

    def emit(event: str, payload: dict[str, Any]) -> None:
        _append_progress(progress_path, event, {"run_id": args.run_id, **payload})

    rows = train_linear_probe_panel(
        source_rows=source_rows,
        source_checkpoint_paths=source_checkpoint_paths,
        target_rows=target_rows,
        target_datasets=target_datasets,
        target_paths=target_paths,
        representation_cache_root=output_root / "representation_cache",
        checkpoint_dir=output_root / "checkpoints",
        device=args.device,
        progress_callback=emit,
    )
    gate = adjudicate_linear_probe_panel(run_id=args.run_id, rows=rows)
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "result_rows": len(rows),
        "expected_rows": len(TARGET_SEEDS) * len(EXPECTED_ROLES),
        "checks": gate["protocol_checks"],
        "errors": [
            name for name, passed in gate["protocol_checks"].items() if not passed
        ],
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "authority": authority,
        "source": "SKINNY-64/64 r7 RTG3-A seed0 formal-scale best checkpoints",
        "target": (
            "RECTANGLE-80 r6 RCT1 seed0/seed1 train and seed10000/seed10001 "
            "validation disk caches"
        ),
        "train": "4096 total = 2048/class per seed",
        "validation": "2048 total = 1024/class per seed",
        "training": (
            "frozen RuntimeE4 representation; Linear(384,1), 385 trainable "
            "parameters; 100 epochs"
        ),
        "gate": gate,
    }
    _write_jsonl(output_root / "results.jsonl", rows)
    artifact_verification = verify_linear_probe_artifacts(
        output_root=output_root,
        rows=rows,
    )
    _write_json(
        output_root / "artifact-verification.json",
        artifact_verification,
    )
    if artifact_verification["status"] != "pass":
        raise ValueError("X4 independent artifact verification failed")
    validation["checks"]["independent_artifact_verification"] = True
    _write_history_csv(output_root / "history.csv", rows)
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "summary.json", summary)
    render_linear_probe_svg(gate, rows, output_root / "curves.svg")
    (output_root / "visual_qa_pending.marker").touch()
    _append_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
            "visual_qa": "pending",
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_linear_probe_svg(
    gate: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    role_labels = {
        "true_source_true_target": "正确源+正确目标",
        "corrupted_source_true_target": "错误源",
        "true_source_corrupted_target": "错误目标",
        "random_source_true_target": "随机源",
    }
    colors = {
        "true_source_true_target": "#047857",
        "corrupted_source_true_target": "#D97706",
        "true_source_corrupted_target": "#2563EB",
        "random_source_true_target": "#64748B",
    }
    grouped = {(int(row["seed"]), str(row["role"])): row for row in rows}
    final_values = [float(row["auc"]) for row in rows]
    y_min = 0.49
    y_max = min(1.0, max(final_values) + 0.05)

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
        figure, axes = plt.subplots(1, 2, figsize=(16.0, 9.0))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.72,
            bottom=0.19,
            wspace=0.25,
        )
        figure.suptitle(
            "创新1 X4：冻结 SPN 结构表示能否被线性层直接读出",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.905,
            (
                "来源为正式规模 SKINNY-64/64 7轮主干，目标为 RECTANGLE-80 6轮；"
                "每条样本含4对密文；双seed；只训练 Linear(384,1) 共385个参数。"
            ),
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        conclusion = {
            "pass": "通过：两颗seed的正确结构表示均可被线性探针直接读出，并同时超过三种控制。",
            "hold": "暂缓：冻结表示的信号需要非线性目标适配，停止线性探针路线的机械放大。",
            "fail": "无效：缓存、冻结边界或检查点协议不完整，只允许修复证据。",
        }[gate["status"]]
        figure.text(
            0.07,
            0.855,
            conclusion,
            ha="left",
            va="top",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )

        centers = np.arange(len(TARGET_SEEDS), dtype=float)
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(EXPECTED_ROLES))
        for offset, role in zip(offsets, EXPECTED_ROLES, strict=True):
            values = [float(grouped[(seed, role)]["auc"]) for seed in TARGET_SEEDS]
            bars = axes[0].bar(
                centers + offset,
                values,
                width=width,
                color=colors[role],
                label=role_labels[role],
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.3f}" for value in values],
                padding=3,
                fontsize=8.2,
                rotation=90,
            )
        axes[0].axhline(0.55, color="#DC2626", linestyle="--", label="候选门 0.55")
        axes[0].axhline(0.50, color="#334155", linestyle=":", label="随机基线 0.50")
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_xticks(centers, labels=["目标数据 seed0", "目标数据 seed1"])
        axes[0].set_ylabel("最佳验证 AUC")
        axes[0].set_title("最终结果与结构控制", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            frameon=False,
            ncols=3,
            fontsize=8.5,
        )

        for seed, color, style in ((0, "#047857", "-"), (1, "#7C3AED", "--")):
            history = grouped[(seed, "true_source_true_target")]["history"]
            epochs = [int(float(epoch["epoch"])) for epoch in history]
            values = [float(epoch["val_auc"]) for epoch in history]
            axes[1].plot(
                epochs,
                values,
                color=color,
                linestyle=style,
                linewidth=2.2,
                label=f"正确结构候选 seed{seed}",
            )
            best_index = int(np.argmax(values))
            axes[1].scatter(
                [epochs[best_index]],
                [values[best_index]],
                color=color,
                s=42,
                zorder=3,
            )
        candidate_histories = [
            float(epoch["val_auc"])
            for seed in TARGET_SEEDS
            for epoch in grouped[(seed, "true_source_true_target")]["history"]
        ]
        curve_min = max(0.45, min(candidate_histories) - 0.03)
        curve_max = min(1.0, max(candidate_histories) + 0.03)
        axes[1].set_ylim(curve_min, curve_max)
        axes[1].set_xlim(1, 100)
        axes[1].set_xlabel("线性探针训练轮次（epoch）")
        axes[1].set_ylabel("验证 AUC")
        axes[1].set_title("正确结构表示的线性可读性", loc="left", fontweight="bold")
        axes[1].grid(True, color="#E5E7EB", linewidth=0.8)
        axes[1].legend(loc="lower right", frameon=False)

        figure.text(
            0.07,
            0.055,
            (
                "错误源=使用错误拓扑训练的SKINNY检查点；错误目标=向主干提供错误RECTANGLE扩散拓扑；"
                "随机源=未训练主干。图中训练曲线只展示候选，三种控制见左图。"
            ),
            ha="left",
            va="bottom",
            color="#475569",
            fontsize=9.2,
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
                        **{field: epoch.get(field) for field in fieldnames[2:]},
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


__all__ = ["main", "render_linear_probe_svg"]
