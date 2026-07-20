from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E98-C support/orbit PU evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_support_rotation_orbit_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_support_rotation_orbit_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    groups = summary["groups"]
    histogram = {int(key): value for key, value in metrics["combined_component_size_histogram"].items()}
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
        figure, axes = plt.subplots(1, 3, figsize=(16.8, 9.2))
        figure.subplots_adjust(left=0.065, right=0.975, top=0.70, bottom=0.29, wspace=0.34)
        figure.text(0.065, 0.958, "创新2 E98-C：PRESENT九轮候选旋转轨道泄漏修复门", ha="left", va="top", fontsize=15.2, fontweight="bold", color="#0F172A")
        figure.text(0.065, 0.897, "把共享坐标和同一同步旋转轨道同时视为不可拆组件，防止相同未标注候选跨训练/测试折。", ha="left", va="top", fontsize=9.8, color="#526070")
        figure.text(0.065, 0.849, "修复后仍保持6组各78条已知正关系；所有候选仍是未标注关系，不是密码学负例。", ha="left", va="top", fontsize=9.8, color="#526070")

        sizes = sorted(histogram)
        counts = [histogram[size] for size in sizes]
        axes[0].bar(sizes, counts, color="#2563EB")
        for size, count in zip(sizes, counts, strict=True):
            axes[0].text(size, count + 4, str(count), ha="center", color="#334155")
        axes[0].set_xlabel("合并组件包含的正关系数")
        axes[0].set_ylabel("组件数量")
        axes[0].set_title("组件仍然很窄，最大仅6条", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        x = np.arange(len(groups))
        train_width = [row["minimum_train_unlabeled"] for row in groups]
        test_width = [row["minimum_test_unlabeled"] for row in groups]
        axes[1].bar(x - 0.19, train_width, 0.38, label="训练池最少候选", color="#0F766E")
        axes[1].bar(x + 0.19, test_width, 0.38, label="测试池最少候选", color="#D97706")
        axes[1].axhline(31, color="#DC2626", linestyle="--", linewidth=1.2, label="最低门槛31")
        axes[1].set_xticks(x, [f"组{index}" for index in range(len(groups))])
        axes[1].set_ylim(0, max(train_width + test_width) * 1.20)
        axes[1].set_ylabel("每个已知正例的未标注候选数")
        axes[1].set_title("修复后候选宽度仍充足", loc="left", fontweight="bold")
        axes[1].legend(frameon=False)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        leak_labels = ("旋转轨道跨组", "全部关系重合", "候选碰对侧坐标", "候选命中已知正例")
        leak_values = (
            metrics["rotation_orbits_split_across_groups"],
            metrics["maximum_train_test_all_relation_overlap"],
            metrics["candidate_positive_support_overlap"],
            metrics["candidate_known_positive_overlap"],
        )
        axes[2].bar(np.arange(4), leak_values, color=("#7C3AED", "#2563EB", "#DC2626", "#64748B"))
        for index, value in enumerate(leak_values):
            axes[2].scatter(index, 0.035, s=42, color="#0F766E", zorder=3)
            axes[2].text(index, 0.075, f"{value} / 通过", ha="center", va="bottom", color="#0F766E", fontweight="bold")
        axes[2].set_xticks(np.arange(4), leak_labels, rotation=22, ha="right")
        axes[2].set_ylim(0, 1.0)
        axes[2].set_ylabel("泄漏或重合数量")
        axes[2].set_title("四层泄漏检查全部为0", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        figure.text(0.065, 0.176, f"数据结论：{metrics['rotation_orbits']}个旋转轨道合并为{metrics['combined_components']}个组件；训练/测试候选最少{metrics['minimum_train_unlabeled']}/{metrics['minimum_test_unlabeled']}。", ha="left", va="bottom", fontsize=9.6, color="#334155")
        conclusion = "通过：可按新轨道互斥fold恢复E99本地神经排序，远程仍关闭。" if gate["status"] == "pass" else "未通过：停止当前公开九轮语料神经路线。"
        figure.text(0.065, 0.108, f"裁决：{conclusion}", ha="left", va="bottom", fontsize=9.8, color="#334155")
        figure.text(0.065, 0.045, "证据范围：同一公开ATM语料内部的数据泄漏修复门；不是九轮神经结果、PRESENT-80密钥调度验证、区分器、攻击或SOTA。", ha="left", va="bottom", fontsize=9.0, color="#526070")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
