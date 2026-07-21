from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Innovation 2 OP8 exact ANF output-parity difficulty audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_exact_anf_difficulty(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_exact_anf_difficulty(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    rounds = gate["round_summaries"]
    x = np.arange(3)
    labels = ("PRESENT一轮", "PRESENT二轮", "PRESENT三轮")
    support = [row["structural_input_cone_median"] for row in rounds]
    completed = [row["completed_masks"] for row in rounds]
    capped = [row["cap_exceeded_masks"] for row in rounds]
    monomials = [
        max(row["monomial_count_median"], row["maximum_observed_terms"])
        for row in rounds
    ]
    auc = [row["two_key_mean_aligned_auc"] for row in rounds]
    colors = ("#0F766E", "#2563EB", "#B91C1C")
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
        figure, axes = plt.subplots(2, 2, figsize=(15.2, 10.0))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.78, bottom=0.17, hspace=0.46, wspace=0.25
        )
        figure.text(
            0.075,
            0.965,
            "创新2 OP8：PRESENT真实密文输出parity的精确ANF难度跃迁",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.917,
            "全部16个结构对齐输出mask；固定seed0秘密密钥；多项式变量只有64个明文bit。",
            ha="left",
            va="top",
            fontsize=9.6,
            color="#475569",
        )
        figure.text(
            0.075,
            0.882,
            "这是无训练确定性审计：精确GF(2)消去后的支持、次数与单项式，不是神经网络指标估计。",
            ha="left",
            va="top",
            fontsize=9.6,
            color="#475569",
        )
        panels = (
            (
                axes[0, 0],
                support,
                "结构依赖锥覆盖的明文bit",
                "Structural cone",
                False,
            ),
            (
                axes[1, 0],
                monomials,
                "多项式执行峰值或冻结硬上限",
                "Peak terms / hard cap (log scale)",
                True,
            ),
            (axes[1, 1], auc, "冻结的双密钥输出预测AUC", "Macro AUC", False),
        )
        for axis, values, title, ylabel, log_scale in panels:
            axis.bar(x, values, width=0.62, color=colors)
            axis.set_xticks(x, labels)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
            if log_scale:
                axis.set_yscale("log")
            if title.endswith("AUC"):
                axis.axhline(0.5, color="#334155", linestyle="--", linewidth=1.1)
                axis.set_ylim(0.48, 1.02)
            for index, value in enumerate(values):
                if "硬上限" in title and capped[index] > 0:
                    label = f">={value:g}"
                else:
                    label = f"{value:.3f}" if isinstance(value, float) else f"{value:g}"
                axis.annotate(
                    label,
                    (index, value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontweight="bold",
                )
        count_axis = axes[0, 1]
        count_axis.bar(
            x,
            completed,
            width=0.62,
            color="#0F766E",
            label="精确完成",
        )
        count_axis.bar(
            x,
            capped,
            width=0.62,
            bottom=completed,
            color="#B91C1C",
            label="触及冻结硬上限",
        )
        count_axis.set_xticks(x, labels)
        count_axis.set_ylim(0, 18)
        count_axis.set_ylabel("Functions")
        count_axis.set_title(
            "16个输出函数的执行状态",
            loc="left",
            fontweight="bold",
        )
        count_axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        count_axis.legend(
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.01),
            ncol=2,
        )
        for index, (done, cap) in enumerate(zip(completed, capped, strict=True)):
            count_axis.text(
                index,
                16.4,
                f"{done}完成 / {cap}超限",
                ha="center",
                fontweight="bold",
            )
        r3 = rounds[2]
        decision_text = (
            "三轮难度与数据稀疏跃迁确认；下一步只开放固定验证/测试集的嵌套训练数据斜率。"
            if gate["status"] == "pass"
            else (
                "三轮15/16函数触及冻结硬上限；不提高上限，也不开放扩样本或扩轮。"
                if gate["decision"]
                == "innovation2_output_parity_exact_anf_difficulty_hard_cap_exceeded"
                else "难度跃迁未确认；停止当前mask路线的扩样本与扩轮。"
            )
        )
        figure.text(
            0.075,
            0.105,
            (
                f"三轮结构锥={r3['structural_input_cone_median']:.0f} bit；"
                f"{r3['completed_masks']}个精确完成，{r3['cap_exceeded_masks']}个触及500000项硬上限；"
                f"已完成函数次数={r3['exact_degree_median']:.0f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.075,
            0.063,
            f"裁决：{decision_text}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.075,
            0.025,
            "证据边界：固定密钥r1--r3精确函数审计；不证明增加样本一定成功，也不是攻击轮数或SOTA。",
            ha="left",
            va="bottom",
            fontsize=8.9,
            color="#64748B",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
