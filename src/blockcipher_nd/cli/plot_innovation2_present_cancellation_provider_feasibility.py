from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E97 provider feasibility evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_provider_feasibility(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_provider_feasibility(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    providers = metrics["provider_rows"]
    provider_names = [row["display_name"] for row in providers]
    columns = (
        "target_semantics_match",
        "sound_certificate",
        "currently_executable",
        "cancellation_aware",
        "within_frozen_cap",
        "real_present_r5",
    )
    column_labels = ("目标语义", "证书sound", "可执行", "识别相消", "低于cap", "真实PRESENT r5")
    matrix = np.asarray(
        [[bool(row[column]) for column in columns] for row in providers], dtype=np.int8
    )
    decisions = {
        "innovation2_present_cancellation_provider_not_feasible_under_frozen_caps": (
            "未找到可执行且语义匹配的非平凡相消提供器；停止当前provider研究，不启动高轮网络。"
        ),
        "innovation2_present_cancellation_provider_feasible": (
            "提供器通过；下一步只扩严格标签并做确定性机制门，仍不直接训练网络。"
        ),
        "innovation2_present_cancellation_provider_protocol_invalid": (
            "冻结来源或query面板漂移；只修协议，不解释科学结果。"
        ),
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16.6, 9.2))
        figure.subplots_adjust(left=0.055, right=0.975, top=0.70, bottom=0.30, wspace=0.38)
        figure.text(
            0.055,
            0.958,
            "创新2 E97：高轮输出预测为什么仍不能开训",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.055,
            0.897,
            "目标：为PRESENT-80五轮的input cube + 多bit输出mask生成全密钥、全inactive offset严格平衡证书。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )
        figure.text(
            0.055,
            0.850,
            "重放E52-E55、E61、E64、E69；12个冻结query全部来自当前unknown，不使用有限密钥投票。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )

        cmap = ListedColormap(("#FEE2E2", "#0F766E"))
        axes[0].imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axes[0].set_xticks(np.arange(len(columns)), column_labels, rotation=25, ha="right")
        axes[0].set_yticks(np.arange(len(providers)), provider_names)
        for row_index in range(len(providers)):
            for column_index in range(len(columns)):
                value = int(matrix[row_index, column_index])
                axes[0].text(
                    column_index,
                    row_index,
                    "是" if value else "否",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#FFFFFF" if value else "#991B1B",
                )
        axes[0].set_title("六类提供器逐门检查", loc="left", fontweight="bold")
        axes[0].tick_params(axis="both", length=0)

        status_labels = ("严格非平凡正类", "严格负类", "仍未决")
        status_values = (
            metrics["strict_nontrivial_present_positives"],
            0,
            metrics["panel_unresolved"],
        )
        status_colors = ("#0F766E", "#2563EB", "#D97706")
        y = np.arange(3)
        axes[1].barh(y, status_values, color=status_colors)
        for index, value in enumerate(status_values):
            axes[1].text(value + 0.18, index, str(value), va="center", fontsize=9.2)
        axes[1].set_yticks(y, status_labels)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, max(status_values) + 1.5)
        axes[1].set_xlabel("12个冻结查询中的数量")
        axes[1].set_title("PRESENT五轮query裁决", loc="left", fontweight="bold")
        axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)

        funnel_labels = ("已审计", "语义匹配", "识别相消", "最终合格")
        funnel_values = (
            metrics["providers_audited"],
            metrics["semantics_matching_providers"],
            metrics["cancellation_aware_providers"],
            metrics["eligible_providers"],
        )
        funnel_colors = ("#64748B", "#2563EB", "#7C3AED", "#0F766E")
        x = np.arange(4)
        axes[2].bar(x, funnel_values, color=funnel_colors)
        for index, value in enumerate(funnel_values):
            axes[2].text(index, value + 0.12, str(value), ha="center", fontsize=9.2)
        axes[2].set_xticks(x, funnel_labels, rotation=18, ha="right")
        axes[2].set_ylim(0, max(funnel_values) + 1.0)
        axes[2].set_ylabel("提供器数量")
        axes[2].set_title("资格漏斗", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.055,
            0.190,
            (
                f"面板：{metrics['panel_queries']}个query，已严格解决{metrics['panel_resolved']}个；"
                f"最终合格provider={metrics['eligible_providers']}，真实PRESENT非平凡正证书="
                f"{metrics['strict_nontrivial_present_positives']}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.055,
            0.112,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.6,
            color="#334155",
        )
        figure.text(
            0.055,
            0.048,
            "证据范围：标签提供器可行性审计；不是PRESENT 7-9轮标签、神经收益、区分器、攻击或SOTA。",
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
