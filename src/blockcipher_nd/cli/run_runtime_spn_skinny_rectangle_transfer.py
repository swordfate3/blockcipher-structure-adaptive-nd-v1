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
from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_rectangle_transfer import (
    EXPECTED_ROLES,
    RUN_ID,
    SOURCE_CHECKPOINT_SHA256S,
    adjudicate_transfer_panel,
    file_sha256,
    train_transfer_panel,
)


DEFAULT_SOURCE_ROOT = Path(
    "outputs/remote_results/i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725"
)
DEFAULT_TARGET_ROOT = Path(
    "outputs/local_diagnostic/"
    "i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/local_diagnostic") / RUN_ID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an independent RECTANGLE head on frozen formal-scale SKINNY "
            "RuntimeE4 representations with source and target topology controls."
        )
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    authority = validate_authorities(args.source_root, args.target_root)
    if args.output_root.exists():
        raise ValueError(f"X3-A output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    progress_path = args.output_root / "progress.jsonl"
    _append_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_root": str(args.source_root),
            "target_root": str(args.target_root),
            "authority": authority,
        },
    )

    source_rows = _read_jsonl(args.source_root / "results.jsonl")
    target_rows = _read_jsonl(args.target_root / "results.jsonl")
    train_dataset, train_paths = _load_target_split(
        args.target_root,
        split="train",
        expected_seed=0,
    )
    validation_dataset, validation_paths = _load_target_split(
        args.target_root,
        split="validation",
        expected_seed=10000,
    )
    source_checkpoint_paths = {
        "true": args.source_root
        / "checkpoints/row0001_skinny64_runtime_e4_equivariant_true_seed0.pt",
        "corrupted": args.source_root
        / "checkpoints/row0002_skinny64_runtime_e4_equivariant_corrupted_seed0.pt",
    }

    def emit(event: str, payload: dict[str, Any]) -> None:
        _append_progress(progress_path, event, {"run_id": args.run_id, **payload})

    rows = train_transfer_panel(
        source_rows=source_rows,
        source_checkpoint_paths=source_checkpoint_paths,
        target_rows=target_rows,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_paths=train_paths,
        validation_paths=validation_paths,
        checkpoint_dir=args.output_root / "checkpoints",
        device=args.device,
        progress_callback=emit,
    )
    gate = adjudicate_transfer_panel(run_id=args.run_id, rows=rows)
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "result_rows": len(rows),
        "expected_rows": len(EXPECTED_ROLES),
        "checks": gate["protocol_checks"],
        "errors": [
            name for name, passed in gate["protocol_checks"].items() if not passed
        ],
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_skinny_formal_to_rectangle_frozen_representation_x3a",
        "authority": authority,
        "source": "SKINNY-64/64 r7 RTG3-A seed0 formal-scale best checkpoints",
        "target": "RECTANGLE-80 r6 RCT1 seed0 exact disk caches",
        "train": "4096 total = 2048/class",
        "validation": "2048 total = 1024/class",
        "training": (
            "independent target head only; 198401 trainable of 640867 total; 5 epochs"
        ),
        "gate": gate,
    }
    _write_jsonl(args.output_root / "results.jsonl", rows)
    _write_history_csv(args.output_root / "history.csv", rows)
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_transfer_svg(gate, rows, args.output_root / "curves.svg")
    (args.output_root / "visual_qa_pending.marker").touch()
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


def validate_authorities(source_root: Path, target_root: Path) -> dict[str, Any]:
    source_gate = _read_json(source_root / "gate.local.json")
    source_checkpoints = _read_json(source_root / "checkpoint-verification.local.json")
    target_gate = _read_json(target_root / "gate.json")
    target_validation = _read_json(target_root / "validation.json")
    source_entries = {
        str(entry.get("model")): entry
        for entry in source_checkpoints.get("entries", [])
    }
    source_authorized = bool(
        source_gate.get("status") == "pass"
        and source_gate.get("decision")
        == "innovation1_rtg3a_skinny_formal_seed0_supported"
        and all(source_gate.get("protocol_checks", {}).values())
        and all(source_gate.get("research_checks", {}).values())
        and source_checkpoints.get("status") == "pass"
        and source_checkpoints.get("file_set_exact") is True
        and source_entries.get("skinny64_runtime_e4_equivariant_true", {}).get("sha256")
        == SOURCE_CHECKPOINT_SHA256S["true"]
        and source_entries.get("skinny64_runtime_e4_equivariant_corrupted", {}).get(
            "sha256"
        )
        == SOURCE_CHECKPOINT_SHA256S["corrupted"]
        and (source_root / "visual_qa_passed.marker").is_file()
    )
    target_authorized = bool(
        target_gate.get("status") == "pass"
        and target_gate.get("decision")
        == "innovation1_runtime_spn_rectangle_noncontiguous_attribution_supported"
        and all(target_gate.get("protocol_checks", {}).values())
        and all(
            all(checks.values())
            for checks in target_gate.get("research_checks", {}).values()
        )
        and target_validation.get("status") == "pass"
        and (target_root / "visual_qa_passed.marker").is_file()
    )
    if not source_authorized:
        raise ValueError("X3-A requires complete formal SKINNY seed0 authority")
    if not target_authorized:
        raise ValueError("X3-A requires complete RECTANGLE RCT1 authority")
    return {
        "source_authorized": source_authorized,
        "target_authorized": target_authorized,
        "source_gate_sha256": file_sha256(source_root / "gate.local.json"),
        "source_checkpoint_verification_sha256": file_sha256(
            source_root / "checkpoint-verification.local.json"
        ),
        "target_gate_sha256": file_sha256(target_root / "gate.json"),
        "target_validation_sha256": file_sha256(target_root / "validation.json"),
    }


def _load_target_split(
    target_root: Path,
    *,
    split: str,
    expected_seed: int,
) -> tuple[DiskDifferentialDataset, dict[str, Path]]:
    split_root = target_root / "cache/rectangle80/r6" / split
    matches = list(split_root.glob(f"seed-{expected_seed}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one RECTANGLE {split} cache for seed {expected_seed}, "
            f"got {len(matches)}"
        )
    cache_dir = matches[0]
    paths = {
        "features": cache_dir / "features.npy",
        "labels": cache_dir / "labels.npy",
        "metadata": cache_dir / "metadata.json",
    }
    metadata = _read_json(paths["metadata"])
    return (
        DiskDifferentialDataset(
            features=np.load(paths["features"], mmap_mode="r"),
            labels=np.load(paths["labels"], mmap_mode="r"),
            metadata=metadata,
            cache_dir=cache_dir,
        ),
        paths,
    )


def render_transfer_svg(
    gate: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    labels = ("正确源+正确目标", "错误源", "错误目标", "随机源", "端到端锚点")
    values = [
        float(gate["aucs"][field])
        for field in (
            "candidate",
            "corrupted_source",
            "corrupted_target",
            "random_source",
            "full_target_anchor",
        )
    ]
    role_labels = {
        "true_source_true_target": "正确源 + 正确目标",
        "corrupted_source_true_target": "错误源权重",
        "true_source_corrupted_target": "错误目标拓扑",
        "random_source_true_target": "随机源权重",
    }
    colors = ("#047857", "#2563EB", "#D97706", "#64748B", "#7C3AED")
    history = {
        str(row["role"]): [float(epoch["val_auc"]) for epoch in row["history"]]
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
        figure, axes = plt.subplots(1, 2, figsize=(15.8, 8.8))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.72,
            bottom=0.20,
            wspace=0.28,
        )
        figure.suptitle(
            "创新1 X3-A：正式 SKINNY 结构表示迁移到 RECTANGLE",
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
                "目标为 RECTANGLE-80 6轮；训练 2048/class、验证 1024/class；"
                "每条样本含4对密文；只训练独立目标头，5 epochs。"
            ),
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        conclusion = {
            "pass": "通过：正确源与正确目标组合同时超过三种控制，可等待 RCT2 后再审中等迁移。",
            "hold": "暂缓：正确组合未同时超过三种控制，停止这条冻结迁移路线的机械放大。",
            "fail": "无效：协议或冻结边界检查失败，只允许修复证据，不解释 AUC。",
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
        bars = axes[0].bar(x, values, color=colors, width=0.68)
        axes[0].bar_label(
            bars,
            labels=[f"{value:.4f}" for value in values],
            padding=4,
            fontsize=9.0,
        )
        axes[0].axhline(0.55, color="#DC2626", linestyle="--", label="候选门 0.55")
        axes[0].axhline(0.50, color="#334155", linestyle=":", label="随机基线 0.50")
        axes[0].set_ylim(min(0.47, min(values) - 0.04), max(0.84, max(values) + 0.04))
        axes[0].set_xticks(x, labels=labels, rotation=15, ha="right")
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("最终最佳检查点", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(loc="upper left", frameon=False, ncols=2)

        epochs = np.arange(1, 6)
        for role, color in zip(EXPECTED_ROLES, colors[:4], strict=True):
            axes[1].plot(
                epochs,
                history[role],
                color=color,
                linewidth=2.0,
                marker="o",
                label=role_labels[role],
            )
        axes[1].axhline(0.50, color="#334155", linestyle=":", linewidth=1.2)
        axes[1].set_xticks(epochs)
        axes[1].set_xlabel("目标头训练轮次（epoch）")
        axes[1].set_ylabel("验证 AUC")
        axes[1].set_title("冻结表示上的目标头学习曲线", loc="left", fontweight="bold")
        axes[1].grid(True, color="#E5E7EB", linewidth=0.8)
        axes[1].legend(
            loc="lower left",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="none",
            framealpha=0.96,
            fontsize=8.6,
        )

        figure.text(
            0.075,
            0.070,
            (
                "正确源=SKINNY正式规模正确拓扑检查点；错误源=SKINNY错误拓扑检查点；"
                "错误目标=RECTANGLE错误P层；端到端锚点=RCT1同数据全模型训练。"
            ),
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


__all__ = ["main", "render_transfer_svg", "validate_authorities"]
