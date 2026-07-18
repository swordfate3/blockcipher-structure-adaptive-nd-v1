from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E95 RECTANGLE nested-cube relation mechanism audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_relation_mechanism(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_relation_mechanism(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    reports = metrics["reports"]
    modes = (
        "independent_dimension",
        "true_nesting",
        "shuffled_nesting",
        "wrong_superset",
        "true_unconstrained",
    )
    labels = ("独立节点", "真实嵌套", "打乱嵌套", "错误超集", "真实但不投影")
    colors = ("#64748B", "#0F766E", "#D97706", "#DC2626", "#2563EB")
    validation_auc = [reports[mode]["validation_auc"] for mode in modes]
    margins = metrics["margins"]
    decisions = {
        "innovation2_rectangle80_nested_cube_relation_mechanism_ready": (
            "真实嵌套通过全部关系归因门；下一步允许两轮单调立方格神经网络readiness。"
        ),
        "innovation2_rectangle80_nested_cube_relation_not_attributed": (
            "真实嵌套未稳定超过必要控制；关闭当前嵌套cube神经路线。"
        ),
        "innovation2_rectangle80_nested_cube_relation_protocol_invalid": (
            "E94来源、关系映射、容量、拆分或单调投影无效；不解释结果。"
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
        figure, axes = plt.subplots(1, 3, figsize=(16.2, 8.8))
        figure.subplots_adjust(left=0.06, right=0.975, top=0.70, bottom=0.29, wspace=0.36)
        figure.text(
            0.06,
            0.955,
            "创新2 E95：正确的7/8/9-bit cube嵌套关系是否真的提供额外信息",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.06,
            0.895,
            "所有模式使用相同44列r3-prefix输入和train-only ridge；真实、打乱、错误关系使用相同单调投影。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.06,
            0.847,
            "本实验不训练神经网络，只决定是否值得开放Monotone Cube-Lattice Operator两轮readiness。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(modes))
        axes[0].bar(x, validation_auc, color=colors)
        for index, value in enumerate(validation_auc):
            axes[0].text(index, value + 0.012, f"{value:.3f}", ha="center")
        axes[0].axhline(0.70, color="#0F172A", linestyle="--", linewidth=1.2)
        axes[0].set_xticks(x, labels, rotation=18, ha="right")
        axes[0].set_ylim(0.45, max(0.82, max(validation_auc) + 0.06))
        axes[0].set_ylabel("chain-disjoint validation AUC")
        axes[0].set_title("同容量关系模式比较", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        margin_names = (
            "相对独立",
            "相对打乱",
            "相对错误超集",
            "相对不投影",
        )
        margin_keys = (
            "true_minus_independent",
            "true_minus_shuffled",
            "true_minus_wrong_superset",
            "true_minus_unconstrained",
        )
        margin_values = [margins[key] for key in margin_keys]
        margin_thresholds = (0.03, 0.03, 0.03, -0.01)
        margin_colors = [
            "#0F766E" if value >= threshold else "#DC2626"
            for value, threshold in zip(margin_values, margin_thresholds)
        ]
        axes[1].bar(np.arange(4), margin_values, color=margin_colors)
        for index, value in enumerate(margin_values):
            offset = 0.003 if value >= 0 else -0.008
            axes[1].text(index, value + offset, f"{value:+.3f}", ha="center")
        axes[1].scatter(
            np.arange(4),
            margin_thresholds,
            marker="_",
            s=420,
            linewidths=1.8,
            color="#0F172A",
            label="对应预注册门槛",
            zorder=4,
        )
        axes[1].axhline(0.0, color="#64748B", linewidth=0.9)
        axes[1].set_xticks(np.arange(4), margin_names, rotation=18, ha="right")
        axes[1].set_ylim(-0.012, 0.035)
        axes[1].set_ylabel("真实嵌套 AUC 差值")
        axes[1].set_title("预注册关系归因margin", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper right", fontsize=8.4)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        raw_violations = [reports[mode]["raw_monotonic_violations"] for mode in modes]
        final_violations = [reports[mode]["final_monotonic_violations"] for mode in modes]
        width = 0.34
        axes[2].bar(x - width / 2, raw_violations, width, color="#94A3B8", label="投影前")
        axes[2].bar(x + width / 2, final_violations, width, color="#0F766E", label="最终")
        for index, value in enumerate(raw_violations):
            axes[2].text(
                index - width / 2,
                value - max(raw_violations) * 0.025,
                str(value),
                ha="center",
                va="top",
                rotation=90,
                fontsize=8.0,
                color="#FFFFFF",
            )
        for index, value in enumerate(final_violations):
            axes[2].text(
                index + width / 2,
                value - max(raw_violations) * 0.025
                if value
                else max(raw_violations) * 0.012,
                str(value),
                ha="center",
                va="top" if value else "bottom",
                rotation=90 if value else 0,
                fontsize=8.0,
                color="#FFFFFF" if value else "#334155",
            )
        axes[2].set_xticks(x, labels, rotation=18, ha="right")
        axes[2].set_ylim(0, max(raw_violations + final_violations) * 1.12)
        axes[2].set_ylabel("单调顺序违反数量")
        axes[2].set_title("投影协议完整性", loc="left", fontweight="bold")
        axes[2].legend(frameon=False)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.06,
            0.185,
            (
                f"真实嵌套 train/validation AUC：{reports['true_nesting']['train_auc']:.3f} / "
                f"{reports['true_nesting']['validation_auc']:.3f}；"
                f"差值={margins['true_train_minus_validation']:+.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.06,
            0.112,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.06,
            0.048,
            "证据范围：RECTANGLE-80四轮确定性关系机制；不是神经收益、第三SPN正式确认、攻击或SOTA。",
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
