from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation2.present_multibit_mask_profile_readiness import (
    MULTIBIT_FAMILIES,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E69 multi-bit mask audit.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_multibit_mask_profile(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_multibit_mask_profile(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    reports = gate["metrics"]["decomposition_reports"]
    feature_reports = gate["metrics"]["feature_reports"]["combined"]
    labels = ("nibble", "P-layer pair", "同nibble pair", "相邻nibble pair")
    trivial = [
        int(reports[family]["validation_positive"])
        - int(reports[family]["validation_nontrivial_positive"])
        for family in MULTIBIT_FAMILIES
    ]
    nontrivial = [
        int(reports[family]["validation_nontrivial_positive"])
        for family in MULTIBIT_FAMILIES
    ]
    baseline_labels = (
        "component units\nall positive",
        "静态set",
        "错误P可达",
        "正确P可达",
        "ANF前缀",
    )
    baseline_values = (
        float(reports["combined"]["validation_componentwise_auc"]),
        float(feature_reports["static_set"]["validation_auc"]),
        float(feature_reports["corrupted_topology"]["validation_auc"]),
        float(feature_reports["true_topology"]["validation_auc"]),
        float(feature_reports["anf_prefix"]["validation_auc"]),
    )
    decisions = {
        "innovation2_present_multibit_mask_query_ready": (
            "非平凡多bit标签与信号过门；允许轻量mask-query decoder readiness。"
        ),
        "innovation2_present_multibit_profile_componentwise_dominated": (
            "正类由unit平衡状态组合解释；停止mask-query decoder。"
        ),
        "innovation2_present_multibit_profile_marginal_dominated": (
            "边际捷径仍主导；禁止训练decoder。"
        ),
        "innovation2_present_multibit_profile_too_narrow": (
            "family宽度不足；停止多bit扩展。"
        ),
        "innovation2_present_multibit_profile_signal_not_ready": (
            "非平凡确定性信号不足；停止decoder。"
        ),
        "innovation2_present_multibit_profile_protocol_invalid": (
            "E43重放、matching、分解或特征协议无效。"
        ),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.27, wspace=0.31
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E69：多bit linear mask是否提供新的积分关系",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "审计E43剩余236个非unit mask；先分解componentwise正类与真正非平凡多bit正类。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "若unit标签的简单AND已经解释正类，就不为已确认的64-node operator增加新decoder。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(4)
        axes[0].bar(x - 0.18, trivial, 0.36, label="componentwise正类", color="#2563EB")
        axes[0].bar(x + 0.18, nontrivial, 0.36, label="非平凡正类", color="#D97706")
        for index, value in enumerate(trivial):
            axes[0].text(index - 0.18, value + 1, str(value), ha="center")
        for index, value in enumerate(nontrivial):
            axes[0].text(index + 0.18, value + 1, str(value), ha="center", fontweight="bold")
        axes[0].set_xticks(x, labels)
        axes[0].tick_params(axis="x", labelrotation=5)
        axes[0].set_ylabel("validation positive数量")
        axes[0].set_title("多bit正类来源分解", loc="left", fontweight="bold")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        bx = np.arange(5)
        axes[1].bar(
            bx,
            baseline_values,
            color=["#DC2626", "#94A3B8", "#D97706", "#2563EB", "#0F766E"],
        )
        for index, value in enumerate(baseline_values):
            axes[1].text(index, value + 0.015, f"{value:.3f}", ha="center", fontweight="bold")
        axes[1].axhline(0.80, color="#DC2626", linestyle="--", linewidth=1.2)
        axes[1].axhline(0.60, color="#D97706", linestyle=":", linewidth=1.2)
        axes[1].set_xticks(bx, baseline_labels)
        axes[1].set_ylim(0.4, 1.07)
        axes[1].set_ylabel("structure-disjoint验证 AUC")
        axes[1].set_title("语义强基线与确定性特征", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        combined = reports["combined"]
        figure.text(
            0.075,
            0.178,
            (
                f"combined nontrivial positive fraction="
                f"{combined['raw_nontrivial_positive_fraction']:.4f}；"
                f"componentwise AUC={combined['validation_componentwise_auc']:.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.075,
            0.111,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.053,
            "证据范围：PRESENT-80四轮严格多bit mask标签审计；不训练网络，不是高轮结论或新攻击。",
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
