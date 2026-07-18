from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E60 ATM r2 cone panel.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_atm_r2_cone_matched_panel(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_atm_r2_cone_matched_panel(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    model = summary["model"]
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
            left=0.07, right=0.975, top=0.69, bottom=0.28, wspace=0.43
        )
        figure.text(
            0.07,
            0.955,
            "创新2 E60：PRESENT两轮依赖锥匹配的严格标签审计",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "固定输出bit 0；输入重量1--8各配一条锥内查询与同重量锥外控制，共16条。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "严格constant由SAT穷尽非零密钥单项式，并用三组独立轮密钥完整标量计算复核。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        completion_values = [
            int(metrics["completed_queries"]),
            int(metrics["unknown_rows"]),
        ]
        axes[0].bar(
            np.arange(2), completion_values, color=["#0F766E", "#94A3B8"], width=0.62
        )
        axes[0].set_xticks(np.arange(2), ["完成", "unknown"])
        axes[0].set_ylabel("查询数量")
        axes[0].set_title("16条查询全部完成", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(completion_values):
            axes[0].text(index, value + 0.35, str(value), ha="center", fontweight="bold")

        label_values = [
            int(metrics["strict_constant_rows"]),
            int(metrics["strict_key_dependent_rows"]),
            int(metrics["scalar_validated_constant_rows"]),
        ]
        axes[1].bar(
            np.arange(3),
            label_values,
            color=["#2563EB", "#D97706", "#7C3AED"],
            width=0.62,
        )
        axes[1].set_xticks(
            np.arange(3), ["constant", "key-dependent", "标量复核"]
        )
        axes[1].set_ylabel("严格标签数量")
        axes[1].set_title("16/16条constant完成标量复核", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(label_values):
            axes[1].text(index, value + 0.35, str(value), ha="center", fontweight="bold")

        gate_labels = ["查询完整", "标量复核", "正负各>=4", "开放训练", "多坐标设计"]
        gate_values = [
            gate["width_checks"]["completed_queries_at_least_12"],
            gate["source_checks"]["all_constant_rows_match_three_scalar_key_sets"],
            gate["width_checks"]["strict_constant_rows_at_least_4"]
            and gate["width_checks"]["strict_key_dependent_rows_at_least_4"],
            gate["next_action"]["training"],
            gate["next_action"]["multi_coordinate_design"],
        ]
        gate_colors = ["#0F766E" if value else "#94A3B8" for value in gate_values]
        axes[2].barh(np.arange(5), [1] * 5, color=gate_colors, height=0.58)
        axes[2].set_yticks(np.arange(5), gate_labels)
        axes[2].set_xlim(0, 1.12)
        axes[2].set_xticks([])
        axes[2].invert_yaxis()
        axes[2].set_title("训练门关闭，下一路线明确", loc="left", fontweight="bold")
        for index, value in enumerate(gate_values):
            axes[2].text(
                0.5,
                index,
                "是" if value else "否",
                ha="center",
                va="center",
                color="#FFFFFF",
                fontweight="bold",
            )

        figure.text(
            0.07,
            0.17,
            "裁决：单坐标标签只有constant一类，不能训练RCCA；转向多坐标GF(2)消去关系。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.108,
            "下一步：构造坐标集合的多项式支撑异或，用严格零支撑与非零odd witness形成正负标签。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            f"证据范围：两轮独立轮密钥；{model['cnf_clauses']}条CNF。不是PRESENT-80主密钥攻击或神经训练结果。",
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
