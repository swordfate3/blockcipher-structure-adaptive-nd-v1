from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E45 PRESENT certificate-complexity attribution."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_certificate_complexity_attribution(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_certificate_complexity_attribution(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    reports = summary["reports"]
    families = (
        "static_set",
        "corrupted_topology",
        "true_topology",
        "anf_prefix",
        "final_oracle",
    )
    labels = (
        "静态set\n统计",
        "错误P-layer\n可达",
        "正确P-layer\n可达",
        "ANF 1–3轮\n前缀",
        "最终证书\noracle",
    )
    values = [reports[family]["validation_auc"] for family in families]
    colors = ["#94A3B8", "#D97706", "#2563EB", "#0F766E", "#64748B"]
    metrics = gate["metrics"]
    deltas = (
        metrics["true_minus_corrupted_topology"],
        metrics["prefix_minus_true_topology"],
        metrics["prefix_minus_static"],
    )
    delta_labels = (
        "正确拓扑 - 错误拓扑",
        "ANF前缀 - 正确拓扑",
        "ANF前缀 - 静态set",
    )
    decisions = {
        "innovation2_present_mspn_route_ready": (
            "ANF前缀路线过门；下一网络为Monomial Support Propagation Network。"
        ),
        "innovation2_present_query_nbfnet_route_ready": (
            "正确拓扑可达路线过门；下一网络为query-conditioned NBFNet。"
        ),
        "innovation2_present_static_set_route_dominant": (
            "静态集合统计主导；暂停拓扑网络并建立set-interaction锚点。"
        ),
        "innovation2_present_certificate_attribution_unresolved": (
            "非oracle特征均未过门；先补证书状态监督或审计unknown边界。"
        ),
        "innovation2_present_certificate_attribution_protocol_invalid": (
            "source、特征、标准化、oracle或metric协议无效。"
        ),
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.6,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 8.2))
        figure.subplots_adjust(
            left=0.075, right=0.975, top=0.70, bottom=0.27, wspace=0.30
        )
        figure.text(
            0.075,
            0.955,
            "创新2 E45：PRESENT四轮弱神经信号来自拓扑路径还是ANF证书复杂度",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "同一E43 checkerboard split、同一train-only ridge；不根据validation选择特征或正则强度。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.075,
            0.848,
            "最终证书oracle只验证标签语义，不是公平baseline；架构路由只比较前四类非oracle证据。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        x = np.arange(len(values))
        axes[0].bar(x, values, color=colors, width=0.64)
        for index, value in enumerate(values):
            axes[0].text(index, value + 0.018, f"{value:.3f}", ha="center")
        axes[0].axhline(0.60, color="#DC2626", linestyle="--", linewidth=1.3)
        axes[0].axhline(0.50, color="#64748B", linestyle=":", linewidth=1.1)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylim(0.4, 1.07)
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("确定性特征族归因", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        dx = np.arange(len(deltas))
        delta_colors = ["#2563EB", "#0F766E", "#0F766E"]
        axes[1].bar(dx, deltas, color=delta_colors, width=0.58)
        for index, value in enumerate(deltas):
            axes[1].text(index, value + 0.008, f"{value:+.3f}", ha="center")
        axes[1].axhline(0.03, color="#DC2626", linestyle="--", linewidth=1.3)
        axes[1].axhline(0.0, color="#64748B", linewidth=0.9)
        axes[1].set_xticks(dx, delta_labels)
        axes[1].tick_params(axis="x", labelrotation=8)
        axes[1].set_ylim(min(-0.04, min(deltas) - 0.03), max(0.22, max(deltas) + 0.04))
        axes[1].set_ylabel("验证 AUC 差值")
        axes[1].set_title("路线增量与0.03门槛", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.178,
            f"选择路线：{metrics['selected_route']}；E44 triangle AUC={metrics['e44_triangle_validation_auc']:.3f}。",
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
            "证据范围：PRESENT-80四轮严格标签的确定性归因与架构路由；不是神经性能、高轮结论或新攻击。",
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
