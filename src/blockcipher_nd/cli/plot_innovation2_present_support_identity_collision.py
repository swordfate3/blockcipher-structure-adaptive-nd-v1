from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E48 PRESENT support identity collision audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_support_identity_collision(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_support_identity_collision(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    reports = summary["reports"]
    collisions = summary["collisions"]
    auc_families = (
        "degree_only",
        "exact_identity",
        "sketch16",
        "sketch32",
        "sketch64",
        "permuted_sketch64",
        "corrupted_sketch64",
    )
    auc_labels = (
        "degree-only",
        "精确support\n身份",
        "sketch16",
        "sketch32",
        "sketch64",
        "变量身份\n打乱64",
        "错误P-layer\nsketch64",
    )
    auc_values = [reports[family]["validation_auc"] for family in auc_families]
    auc_colors = [
        "#2563EB",
        "#64748B",
        "#94A3B8",
        "#0EA5A4",
        "#0F766E",
        "#A855F7",
        "#D97706",
    ]
    collision_families = (
        "degree_only",
        "exact_identity",
        "sketch16",
        "sketch32",
        "sketch64",
    )
    collision_labels = (
        "degree-only",
        "精确身份",
        "binary16",
        "binary32",
        "binary64",
    )
    collision_values = [
        collisions[family]["conflicting_row_rate"] for family in collision_families
    ]
    decisions = {
        "innovation2_present_identity_sketch_route_ready": (
            "64维identity sketch全部过门；下一网络为Identity-Sketch Monomial Propagator。"
        ),
        "innovation2_present_exact_monomial_token_route_ready": (
            "只有精确support身份过门；下一网络为稀疏Monomial Token Set Transformer。"
        ),
        "innovation2_present_support_identity_not_supported": (
            "变量身份没有超过degree-only；关闭identity网络路线。"
        ),
        "innovation2_present_support_identity_protocol_invalid": (
            "source、support、投影、碰撞、ridge或metric协议无效。"
        ),
    }

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.4,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FFFFFF",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.6, 8.2))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.28, wspace=0.30
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E48：MSPN失败是否由单项式变量身份碰撞造成",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "同一E43 split、1–3轮support；degree-only、精确身份和固定16/32/64维sketch使用同一train-only ridge。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "变量身份打乱与错误P-layer保持宽度和拟合器不变；不使用第4轮full-cube oracle。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )

        x = np.arange(len(auc_values))
        axes[0].bar(x, auc_values, color=auc_colors, width=0.66)
        for index, value in enumerate(auc_values):
            axes[0].text(index, value + 0.016, f"{value:.3f}", ha="center")
        axes[0].axhline(
            0.62,
            color="#DC2626",
            linestyle="--",
            linewidth=1.3,
            label="候选门 0.62",
        )
        axes[0].axhline(
            0.5,
            color="#64748B",
            linestyle=":",
            linewidth=1.1,
            label="随机水平 0.50",
        )
        axes[0].set_xticks(x, auc_labels)
        axes[0].set_ylim(0.35, max(0.76, max(auc_values) + 0.06))
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("身份信息与transport归因", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(
            loc="upper right",
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            fontsize=8.2,
        )

        cx = np.arange(len(collision_values))
        axes[1].bar(
            cx,
            [value * 100 for value in collision_values],
            color=["#2563EB", "#64748B", "#94A3B8", "#0EA5A4", "#0F766E"],
            width=0.62,
        )
        for index, value in enumerate(collision_values):
            axes[1].text(index, value * 100 + 0.10, f"{value*100:.2f}%", ha="center")
        axes[1].set_xticks(cx, collision_labels)
        axes[1].set_ylim(0, max(3.5, max(collision_values) * 100 + 0.65))
        axes[1].set_ylabel("跨标签冲突行比例（%）")
        axes[1].set_title("签名碰撞并不是主要瓶颈", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        metrics = gate["metrics"]
        figure.text(
            0.065,
            0.184,
            f"sketch64-degree={metrics['sketch64_minus_degree']:+.3f}；sketch64-身份打乱={metrics['sketch64_minus_permuted']:+.3f}；sketch64-错误P={metrics['sketch64_minus_corrupted']:+.3f}。",
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.065,
            0.116,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.065,
            0.055,
            "证据范围：PRESENT-80四轮support身份碰撞与架构路由；不是神经结果、高轮结论或新攻击。",
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
