from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


MODEL_ROWS = (
    ("structure_expert", "uKNIT 结构专家"),
    ("autond_dbitnet", "AutoND DBitNet"),
    ("zhang_wang_mcnd", "Zhang/Wang MCND 适配"),
    ("liu_case3_conv2d", "Liu Case-3 Conv2D 适配"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-BZ published-baseline comparison."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1bz_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bz_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"3", "4"}:
        raise ValueError("K1-BZ plot requires seed3 and seed4 results")
    auc_values = _auc_values(seed_results)
    delta_values, delta_labels = _delta_values(seed_results)

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16.0, 8.8))
        figure.subplots_adjust(
            left=0.19,
            right=0.96,
            top=0.67,
            bottom=0.13,
            wspace=0.39,
        )
        figure.suptitle(
            "创新1 K1-BZ：uKNIT 第5轮公开论文架构补充对比",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.89,
            "固定 cell11 差分、16对密文、2048/class、跨密钥验证、两颗 seed 和10个 epoch；仅替换网络架构。",
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.82,
            "裁决：MCND 与 raw Case-3 Conv2D 均未通过双种子晋级门，不进入远程扩样。",
            ha="left",
            fontsize=11.3,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.755,
            "这是统一 uKNIT 协议下的架构适配诊断，不是 Zhang/Wang、Liu 或 AutoND 原论文协议复现。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )

        _auc_heatmap(axes[0], auc_values)
        _delta_bars(axes[1], delta_values, delta_labels)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.0,
        "height_inches": 8.8,
        "language": "zh-CN",
        "panels": 2,
        "auc_values_annotated": True,
        "autond_delta_and_gate_visible": True,
        "paper_protocol_boundary_visible": True,
    }


def _auc_values(seed_results: Mapping[str, Any]) -> np.ndarray:
    columns = ("3", "4")
    values: list[list[float]] = []
    for model, _ in MODEL_ROWS:
        row = []
        for seed in columns:
            source = (
                seed_results[seed]["k1bs_anchors"]
                if model in {"structure_expert", "autond_dbitnet"}
                else seed_results[seed]["auc_by_architecture"]
            )
            row.append(float(source[model]))
        values.append(row)
    return np.asarray(values, dtype=float)


def _delta_values(
    seed_results: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[str, ...]]:
    architectures = (
        ("zhang_wang_mcnd", "MCND"),
        ("liu_case3_conv2d", "Liu Conv2D"),
    )
    values = []
    labels = []
    for architecture, label in architectures:
        for seed in ("3", "4"):
            values.append(
                float(seed_results[seed]["adapter_minus_autond"][architecture])
            )
            labels.append(f"{label} / seed{seed}")
    return np.asarray(values, dtype=float), tuple(labels)


def _auc_heatmap(axis: plt.Axes, values: np.ndarray) -> None:
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=0.48, vmax=0.95)
    axis.set_xticks((0, 1), ("seed3", "seed4"))
    axis.set_yticks(range(len(MODEL_ROWS)), [label for _, label in MODEL_ROWS])
    axis.set_title("跨密钥验证 AUC（统一协议）", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{float(values[row, column]):.6f}",
                ha="center",
                va="center",
                fontsize=10.8,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("AUC", rotation=270, labelpad=15)


def _delta_bars(axis: plt.Axes, values: np.ndarray, labels: tuple[str, ...]) -> None:
    positions = np.arange(len(values))
    colors = ["#047857" if value >= 0 else "#B91C1C" for value in values]
    bars = axis.barh(positions, values, color=colors, height=0.56)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.axvline(0.0, color="#475569", linewidth=1.1)
    axis.axvline(
        0.010,
        color="#B45309",
        linewidth=1.2,
        linestyle="--",
        label="单 seed 最低优势门 +0.010",
    )
    axis.set_xlim(-0.045, 0.025)
    axis.set_xlabel("相对 AutoND 的 AUC 差值")
    axis.set_title("新增架构相对 AutoND（两颗 seed 均须过门）", loc="left", fontweight="bold", pad=14)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=9)
    axis.legend(loc="upper right", frameon=False, fontsize=9.6)
    for bar, value in zip(bars, values, strict=True):
        offset = 0.0015 if value >= 0 else -0.0015
        axis.text(
            float(value) + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):+.4f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10.3,
            color="#111827",
        )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1bz_svg"]
