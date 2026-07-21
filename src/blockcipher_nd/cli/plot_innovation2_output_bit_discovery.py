from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Innovation 2 OP10 easy ciphertext output-bit discovery."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_output_bit_discovery(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_output_bit_discovery(summary: dict[str, Any], output: Path) -> None:
    ranking = sorted(summary["ranking"], key=lambda row: int(row["msb_index"]))
    gate = summary["gate"]
    mode = summary.get("metadata", {}).get("mode", "unknown")
    candidates = set(summary["candidates"]["candidate_msb_indices"])
    confirmed = set(gate["metrics"]["fresh_confirmed_msb_indices"])
    x = np.arange(64)
    discovery_lstm = np.asarray(
        [row["discovery_lstm_auc"] for row in ranking], dtype=float
    )
    discovery_shuffle = np.asarray(
        [row["discovery_shuffle_auc"] for row in ranking], dtype=float
    )
    discovery_mlp = np.asarray(
        [row["discovery_mlp_auc"] for row in ranking], dtype=float
    )
    fresh_lstm = np.asarray([row["fresh_lstm_auc"] for row in ranking], dtype=float)
    fresh_shuffle = np.asarray(
        [row["fresh_shuffle_auc"] for row in ranking], dtype=float
    )
    fresh_mlp = np.asarray([row["fresh_mlp_auc"] for row in ranking], dtype=float)
    discovery_margin = np.asarray(
        [row["discovery_lstm_accuracy_margin"] for row in ranking], dtype=float
    )
    fresh_margin = np.asarray(
        [row["fresh_lstm_accuracy_margin"] for row in ranking], dtype=float
    )
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16.0, 10.5))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.79,
            bottom=0.13,
            hspace=0.42,
            wspace=0.24,
        )
        figure.text(
            0.07,
            0.965,
            "创新2 OP10：PRESENT三轮哪些真实密文输出bit容易预测",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.918,
            "横轴为MSB-first输出位置：0是密文最高位，63是最低位；预测目标是该位置的真实0/1输出值。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        figure.text(
            0.07,
            0.882,
            "候选只在OP9发现集选择；fresh曲线来自另一组无重合明文，不要求64-bit完整密文同时命中。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )
        figure.text(
            0.07,
            0.846,
            (
                "当前为64条发现 + 64条fresh的本地实现门，曲线波动不作性能结论。"
                if mode == "smoke"
                else "当前为2^16条发现 + 2^16条fresh的单固定密钥确认结果。"
            ),
            ha="left",
            va="top",
            fontsize=9.7,
            color="#475569",
        )

        _plot_auc_panel(
            axes[0, 0],
            x,
            discovery_lstm,
            discovery_mlp,
            discovery_shuffle,
            "发现集：64个输出bit的逐bit AUC",
            candidates,
        )
        _plot_auc_panel(
            axes[0, 1],
            x,
            fresh_lstm,
            fresh_mlp,
            fresh_shuffle,
            "全新明文确认集：冻结模型逐bit AUC",
            confirmed,
        )

        margin_axis = axes[1, 0]
        margin_axis.plot(
            x,
            discovery_margin,
            color="#0F766E",
            linewidth=1.5,
            label="发现集 LSTM",
        )
        margin_axis.plot(
            x,
            fresh_margin,
            color="#2563EB",
            linewidth=1.5,
            label="fresh确认 LSTM",
        )
        margin_axis.axhline(0.0, color="#475569", linestyle="--", linewidth=1.0)
        margin_axis.axhline(
            0.005,
            color="#B91C1C",
            linestyle=":",
            linewidth=1.2,
            label="确认门 +0.005",
        )
        margin_axis.set_xlim(0, 63)
        margin_axis.set_xticks(np.arange(0, 64, 4))
        margin_axis.set_xlabel("密文输出bit位置（MSB-first）")
        margin_axis.set_ylabel("Accuracy - majority")
        margin_axis.set_title(
            "逐bit准确率超过该bit多数类基线的幅度",
            loc="left",
            fontweight="bold",
        )
        margin_axis.grid(color="#E5E7EB", linewidth=0.7)
        margin_axis.legend(frameon=False, ncol=3, loc="upper left")

        candidate_axis = axes[1, 1]
        candidate_rows = gate["candidate_confirmation"]
        if candidate_rows:
            positions = np.arange(len(candidate_rows))
            fresh_auc_values = [row["fresh_auc"] for row in candidate_rows]
            shuffle_auc_values = [row["fresh_shuffle_auc"] for row in candidate_rows]
            width = 0.36
            candidate_axis.bar(
                positions - width / 2,
                fresh_auc_values,
                width=width,
                color="#0F766E",
                label="真实输出LSTM",
            )
            candidate_axis.bar(
                positions + width / 2,
                shuffle_auc_values,
                width=width,
                color="#B91C1C",
                label="标签打乱LSTM",
            )
            candidate_axis.axhline(
                0.51,
                color="#334155",
                linestyle="--",
                linewidth=1.1,
                label="AUC确认门0.510",
            )
            candidate_axis.set_xticks(
                positions,
                [f"bit {int(row['msb_index'])}" for row in candidate_rows],
            )
            candidate_axis.tick_params(axis="x", rotation=35)
            low = min(fresh_auc_values + shuffle_auc_values + [0.5]) - 0.01
            high = max(fresh_auc_values + shuffle_auc_values + [0.51]) + 0.015
            candidate_axis.set_ylim(max(0.0, low), min(1.0, high))
            candidate_axis.legend(frameon=False, loc="upper left")
        else:
            candidate_axis.text(
                0.5,
                0.5,
                "发现集没有通过预注册门的候选bit",
                ha="center",
                va="center",
                transform=candidate_axis.transAxes,
                fontsize=11.0,
                color="#475569",
            )
            candidate_axis.set_xticks([])
            candidate_axis.set_yticks([])
        candidate_axis.set_ylabel("Fresh AUC")
        candidate_axis.set_title(
            "冻结候选在全新明文上的确认",
            loc="left",
            fontweight="bold",
        )
        candidate_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)

        confirmed_text = (
            ", ".join(str(bit) for bit in sorted(confirmed)) if confirmed else "无"
        )
        decision_text = {
            "innovation2_output_bit_discovery_local_smoke_passed": (
                "逐bit发现与独立确认实现门通过"
            ),
            "innovation2_true_output_bits_fresh_confirmed": (
                "至少一个真实输出bit通过全新明文确认"
            ),
            "innovation2_no_true_output_bit_fresh_confirmed": (
                "没有候选bit通过全新明文确认"
            ),
            "innovation2_output_bit_discovery_protocol_invalid": "实验协议无效",
        }.get(gate["decision"], gate["decision"])
        figure.text(
            0.07,
            0.065,
            (
                f"裁决：{decision_text}；发现候选={len(candidates)}个；"
                f"fresh确认bit（MSB-first）={confirmed_text}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.07,
            0.027,
            "证据边界：单固定密钥PRESENT三轮逐bit真实输出预测；不是完整密文恢复、跨密钥结论或样本分类。",
            ha="left",
            va="bottom",
            fontsize=8.9,
            color="#64748B",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


def _plot_auc_panel(
    axis: Any,
    x: np.ndarray,
    lstm: np.ndarray,
    mlp: np.ndarray,
    shuffled: np.ndarray,
    title: str,
    highlighted_bits: set[int],
) -> None:
    axis.plot(x, lstm, color="#0F766E", linewidth=1.6, label="真实输出LSTM")
    axis.plot(x, mlp, color="#2563EB", linewidth=1.2, label="参数匹配MLP")
    axis.plot(
        x,
        shuffled,
        color="#B91C1C",
        linewidth=1.2,
        alpha=0.85,
        label="标签打乱LSTM",
    )
    axis.axhline(0.5, color="#475569", linestyle="--", linewidth=1.0)
    axis.axhline(
        0.51,
        color="#B91C1C",
        linestyle=":",
        linewidth=1.0,
        label="候选门0.510",
    )
    if highlighted_bits:
        bits = np.asarray(sorted(highlighted_bits), dtype=int)
        axis.scatter(
            bits,
            lstm[bits],
            s=38,
            facecolor="#F8FAFC",
            edgecolor="#0F172A",
            linewidth=1.1,
            zorder=5,
        )
    values = np.concatenate((lstm, mlp, shuffled, np.asarray([0.5, 0.51])))
    low = max(0.0, float(np.min(values)) - 0.012)
    high = min(1.0, float(np.max(values)) + 0.018)
    if high - low < 0.08:
        center = (high + low) / 2
        low = max(0.0, center - 0.04)
        high = min(1.0, center + 0.04)
    axis.set_ylim(low, high)
    axis.set_xlim(0, 63)
    axis.set_xticks(np.arange(0, 64, 4))
    axis.set_xlabel("密文输出bit位置（MSB-first）")
    axis.set_ylabel("AUC")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(color="#E5E7EB", linewidth=0.7)
    axis.legend(frameon=False, ncol=2, loc="upper left")


if __name__ == "__main__":
    raise SystemExit(main())
