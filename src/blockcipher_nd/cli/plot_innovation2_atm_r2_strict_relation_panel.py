from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E59 ATM r2 label panel.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_atm_r2_strict_relation_panel(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_atm_r2_strict_relation_panel(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    model = summary["model"]
    decisions = {
        "innovation2_atm_r2_strict_relation_panel_not_ready": (
            "16条全部是exact constant，没有strict key-dependent；标签面板不能训练RCCA。"
        ),
        "innovation2_atm_r2_strict_relation_panel_ready": (
            "两类严格标签均有宽度，可进入1024-query完整审计。"
        ),
        "innovation2_atm_r2_strict_relation_panel_protocol_invalid": (
            "来源或模型契约无效，禁止解释标签。"
        ),
    }
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
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.4))
        figure.subplots_adjust(
            left=0.07, right=0.975, top=0.70, bottom=0.28, wspace=0.42
        )
        figure.text(
            0.07,
            0.955,
            "创新2 E59：PRESENT两轮能否形成可训练的严格relation标签面板",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "冻结查询：u=0xFFFFFFFFFFFFFFF0，输出unit bit为e0--e15；独立三组64-bit轮密钥。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "constant必须穷尽全部非零key monomial；key-dependent必须有具体odd witness并重放。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        completion_values = [
            int(metrics["completed_queries"]),
            int(metrics["explicit_unknown_rows"]),
            int(metrics["missing_timeout_rows"]),
        ]
        axes[0].bar(
            np.arange(3),
            completion_values,
            color=["#0F766E", "#94A3B8", "#94A3B8"],
            width=0.62,
        )
        axes[0].set_xticks(np.arange(3), ["完成", "显式unknown", "超时缺失"])
        axes[0].set_ylabel("query数量")
        axes[0].set_title("provider执行完整且无unknown", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(completion_values):
            axes[0].text(index, value + 0.35, str(value), ha="center", fontweight="bold")

        class_values = [
            int(metrics["strict_constant_rows"]),
            int(metrics["strict_key_dependent_rows"]),
        ]
        axes[1].bar(
            np.arange(2),
            class_values,
            color=["#2563EB", "#D97706"],
            width=0.62,
        )
        axes[1].set_xticks(np.arange(2), ["constant", "key-dependent"])
        axes[1].set_ylabel("严格标签数量")
        axes[1].set_title("单一类别阻止神经训练", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(class_values):
            axes[1].text(index, value + 0.35, str(value), ha="center", fontweight="bold")

        gate_labels = ["16条完成", "constant>=4", "negative>=4", "开放训练"]
        gate_values = [
            gate["width_checks"]["completed_queries_at_least_12"],
            gate["width_checks"]["strict_constant_rows_at_least_4"],
            gate["width_checks"]["strict_key_dependent_rows_at_least_4"],
            gate["next_action"]["training"],
        ]
        gate_colors = ["#0F766E" if value else "#94A3B8" for value in gate_values]
        axes[2].barh(np.arange(4), [1] * 4, color=gate_colors, height=0.58)
        axes[2].set_yticks(np.arange(4), gate_labels)
        axes[2].set_xlim(0, 1.12)
        axes[2].set_xticks([])
        axes[2].invert_yaxis()
        axes[2].set_title("readiness门只缺负类宽度", loc="left", fontweight="bold")
        for index, value in enumerate(gate_values):
            axes[2].text(
                0.5,
                index,
                "通过" if value else "未通过",
                ha="center",
                va="center",
                color="#FFFFFF",
                fontweight="bold",
            )

        figure.text(
            0.07,
            0.17,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.108,
            "下一步：预注册覆盖依赖锥内外的面板并加入reachability基线；不过门则转multi-coordinate cancellation。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            f"证据范围：两轮独立轮密钥严格标签readiness；模型{model['cnf_clauses']}条CNF，不是神经结果或PRESENT-80攻击。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
