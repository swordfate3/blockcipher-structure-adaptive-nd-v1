from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-AC chart.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1ac_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ac_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"0", "1"}:
        raise ValueError("K1-AC plot requires seed0 and seed1 results")
    auc_rows = (
        ("exact_16pair_auc", "K1-AC：16对正确 S盒"),
        ("wrong_sbox_16pair_auc", "K1-AC：16对错误 S盒"),
        ("k1w_exact_4pair_auc", "K1-W：4对正确 S盒锚点"),
        ("k1w_wrong_sbox_4pair_auc", "K1-W：4对错误 S盒"),
    )
    auc_values = np.asarray(
        [[float(seeds[seed][field]) for seed in ("0", "1")] for field, _ in auc_rows]
    )
    margin_rows = (
        (
            "exact16_minus_k1w_exact4",
            "16对正确 - 4对锚点\n允许回退 0.020",
            -0.020,
        ),
        (
            "exact16_minus_wrong_sbox16",
            "16对正确 - 16对错误\n门槛 ≥ +0.010",
            0.010,
        ),
    )
    margins = np.asarray(
        [[float(seeds[seed][field]) for seed in ("0", "1")] for field, _, _ in margin_rows]
    )
    passed = gate.get("status") == "pass"

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.8,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16.2, 8.8))
        figure.subplots_adjust(
            left=0.18,
            right=0.95,
            top=0.64,
            bottom=0.13,
            wspace=0.44,
        )
        figure.suptitle(
            "创新1 K1-AC：Dialga 4轮强信号是否保留、正确 S盒是否必要",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=16.2,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.885,
            "Dialga-128 4轮、2048/class、10个 epoch；K1-AA 虚拟槽网络，每条样本含16对密文。",
            ha="left",
            fontsize=10.6,
            color="#4B5563",
        )
        if passed:
            decision = "裁决：两颗 seed 均保留强信号，且正确 S盒稳定优于错误 S盒。"
            next_text = "下一步：单独准备 uKNIT 65536/class 远程中等诊断；本结果仍不是正式训练。"
            decision_color = "#047857"
        elif gate.get("decision", "").endswith("semantic_attribution_failed"):
            decision = "裁决：强信号保留，但正确 S盒没有稳定领先；暂不能说模型理解了 Dialga S盒。"
            next_text = "下一步：用同一检查点、同一验证集做零训练归因审计，不增加数据或改网络。"
            decision_color = "#B45309"
        else:
            decision = "裁决：至少一颗 seed 未保留现有 Dialga 强锚点，跨密码设置暂不成立。"
            next_text = "下一步：16对只保留给 uKNIT，先审查 Dialga 的多对聚合。"
            decision_color = "#B45309"
        figure.text(
            0.05,
            0.81,
            decision,
            ha="left",
            fontsize=11.0,
            fontweight="bold",
            color=decision_color,
        )
        figure.text(
            0.05,
            0.745,
            next_text,
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        _auc_table(axes[0], auc_values, auc_rows)
        _margin_table(axes[1], margins, margin_rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.2,
        "height_inches": 8.8,
        "language": "zh-CN",
        "panels": 2,
        "all_values_annotated": True,
        "per_seed_gate_status_visible": True,
    }


def _auc_table(axis: plt.Axes, values: np.ndarray, rows: tuple) -> None:
    minimum = min(0.90, float(values.min()) - 0.01)
    maximum = min(1.0, float(values.max()) + 0.01)
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=minimum, vmax=maximum)
    axis.set_xticks((0, 1), ("Dialga 4轮\nseed0", "Dialga 4轮\nseed1"))
    axis.set_yticks(range(len(rows)), [label for _field, label in rows])
    axis.set_title("跨密钥验证 AUC", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{float(values[row, column]):.4f}",
                ha="center",
                va="center",
                fontsize=10.6,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)


def _margin_table(axis: plt.Axes, values: np.ndarray, rows: tuple) -> None:
    limit = max(0.05, float(np.abs(values).max()) + 0.015)
    axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=-limit, vmax=limit)
    axis.set_xticks((0, 1), ("Dialga 4轮\nseed0", "Dialga 4轮\nseed1"))
    axis.set_yticks(range(len(rows)), [label for _field, label, _gate in rows])
    axis.set_title("逐 seed 保留与归因门槛", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        threshold = float(rows[row][2])
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{value:+.4f}\n{'通过' if value >= threshold else '未通过'}",
                ha="center",
                va="center",
                fontsize=10.4,
                fontweight="bold",
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ac_svg"]
