from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.tasks.innovation2.skinny_hwang_readiness import (
    SkinnyHwangReadinessConfig,
    run_skinny_hwang_readiness_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the SKINNY-64/64 r7 Hwang raw-bit kernel fixture."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discovery-keys", type=int, default=512)
    parser.add_argument("--validation-keys", type=int, default=256)
    parser.add_argument("--key-chunk-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SkinnyHwangReadinessConfig(
        run_id=args.run_id,
        seed=args.seed,
        discovery_keys=args.discovery_keys,
        validation_keys=args.validation_keys,
        key_chunk_size=args.key_chunk_size,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "cipher": "SKINNY-64/64",
            "rounds": 7,
            "target_active_cell": 15,
            "control_active_cell": 0,
            "discovery_keys": args.discovery_keys,
            "validation_keys": args.validation_keys,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_skinny_hwang_readiness_audit(
        config,
        progress_callback=progress_callback,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "kernel_basis.csv", result["basis_rows"])
    np.save(args.output_root / "keys.npy", result["keys"], allow_pickle=False)
    np.save(
        args.output_root / "parity_rows.npy",
        result["parity_rows"],
        allow_pickle=False,
    )
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "gate.json", result["gate"])
    render_skinny_hwang_svg(
        result["rows"],
        result["gate"],
        args.output_root / "curves.svg",
    )
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["gate"]["status"],
                "decision": result["gate"]["decision"],
                "target": result["rows"][0],
                "control": result["rows"][1],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_skinny_hwang_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    by_role = {str(row["role"]): row for row in rows}
    target = by_role["target"]
    control = by_role["control"]
    splits = ("discovery", "validation", "joint")
    split_labels = ("发现512把", "验证256把", "联合768把")
    target_ranks = [int(target[f"{split}_rank"]) for split in splits]
    control_ranks = [int(control[f"{split}_rank"]) for split in splits]
    target_nullities = [int(target[f"{split}_nullity"]) for split in splits]
    control_nullities = [int(control[f"{split}_nullity"]) for split in splits]
    positions = np.arange(len(splits), dtype=np.float64)
    width = 0.34
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.0))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.72,
            bottom=0.24,
            wspace=0.28,
        )
        figure.suptitle(
            "创新2 E20：SKINNY-64/64 7轮论文 kernel 协议复现",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.89,
            "同一组768把密钥：目标为活动 cell15，控制为活动 cell0；raw 64-bit parity，MSB-first。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        rank_axis, nullity_axis = axes
        rank_target_bars = rank_axis.bar(
            positions - width / 2,
            target_ranks,
            width,
            color="#2563EB",
            label="目标 cell15",
        )
        rank_control_bars = rank_axis.bar(
            positions + width / 2,
            control_ranks,
            width,
            color="#64748B",
            label="控制 cell0",
        )
        rank_axis.axhline(
            46,
            color="#D97706",
            linestyle="--",
            linewidth=1.2,
            label="论文目标 rank=46",
        )
        rank_axis.bar_label(rank_target_bars, padding=3, fontsize=9)
        rank_axis.bar_label(rank_control_bars, padding=3, fontsize=9)
        rank_axis.set_title("经验 parity matrix 的 GF(2) rank", loc="left", fontweight="bold")
        rank_axis.set_ylabel("rank")
        rank_axis.set_xticks(positions, split_labels)
        rank_axis.set_ylim(0, 70)
        rank_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        rank_handles, rank_labels = rank_axis.get_legend_handles_labels()
        figure.legend(
            rank_handles,
            rank_labels,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.07, 0.81),
            ncol=3,
            fontsize=8.5,
        )

        nullity_target_bars = nullity_axis.bar(
            positions - width / 2,
            target_nullities,
            width,
            color="#059669",
            label="目标 cell15",
        )
        nullity_control_bars = nullity_axis.bar(
            positions + width / 2,
            control_nullities,
            width,
            color="#64748B",
            label="控制 cell0",
        )
        nullity_axis.axhline(
            18,
            color="#D97706",
            linestyle="--",
            linewidth=1.2,
            label="论文目标 nullity=18",
        )
        nullity_axis.bar_label(nullity_target_bars, padding=3, fontsize=9)
        nullity_axis.bar_label(nullity_control_bars, padding=3, fontsize=9)
        nullity_axis.set_title("经验输出 balance space 维数", loc="left", fontweight="bold")
        nullity_axis.set_ylabel("kernel 维数 / nullity")
        nullity_axis.set_xticks(positions, split_labels)
        nullity_axis.set_ylim(0, 22)
        nullity_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        nullity_handles, nullity_labels = nullity_axis.get_legend_handles_labels()
        figure.legend(
            nullity_handles,
            nullity_labels,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.975, 0.81),
            ncol=3,
            fontsize=8.5,
        )

        target_spans = gate["target_span_equalities"]
        span_text = " / ".join(
            f"{label}:{'相等' if target_spans[split] else '不等'}"
            for split, label in zip(splits, ("发现", "验证", "联合"), strict=True)
        )
        decision_labels = {
            "innovation2_skinny_r7_hwang_kernel_reproduced": (
                "cell15 的18维 span 精确复现，cell0 控制未复现"
            ),
            "innovation2_skinny_r7_hwang_kernel_not_reproduced": (
                "论文 kernel 签名未完整复现，先审计协议"
            ),
            "innovation2_skinny_r7_hwang_protocol_invalid": (
                "密码向量、缓存、拆分或 GF(2) 协议无效"
            ),
        }
        figure.text(
            0.07,
            0.09,
            f"与 Hwang Table 2 span 比较：{span_text}。",
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.07,
            0.045,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是本地 sampled-key readiness，不是论文规模、全密钥证明或神经训练。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires at least one row: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_progress(
    path: Path,
    event: str,
    payload: dict[str, Any],
    *,
    mode: str = "a",
) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
