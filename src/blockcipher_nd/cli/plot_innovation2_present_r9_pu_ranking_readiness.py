from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E98 PU-ranking readiness evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_pu_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_pu_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    folds = summary["folds"]
    baselines = summary["ranking_baselines"]
    metrics = gate["metrics"]
    short_names = [row["heldout_file"].removeprefix("R9-complex-oracle-").removesuffix(".pkl") for row in folds]
    heldout = [row["heldout_relations"] for row in folds]
    baseline_names = {
        "deterministic_hash_random": "确定性随机",
        "file_id": "文件编号",
        "relation_size": "关系项数",
        "exponent_weight": "指数重量",
        "exact_training_frequency": "训练关系频率",
        "training_coordinate_frequency": "训练坐标频率",
        "training_support_overlap": "训练支撑重合",
        "absolute_bit_position": "绝对bit位置",
    }
    labels = [baseline_names[row["baseline"]] for row in baselines]
    recall5 = [row["recall_at_5"] for row in baselines]
    mrr = [row["mean_reciprocal_rank"] for row in baselines]

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
        figure, axes = plt.subplots(1, 2, figsize=(16.4, 9.0))
        figure.subplots_adjust(left=0.075, right=0.965, top=0.69, bottom=0.30, wspace=0.28)
        figure.text(
            0.075,
            0.955,
            "创新2 E98：PRESENT九轮关系能否形成可靠的候选排序任务",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.075,
            0.895,
            "已知ATM关系只作正例；循环平移生成的匹配候选全部是“未标注”，不是负例。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.text(
            0.075,
            0.847,
            "检验重点：按8个来源文件留一后，是否仍有足够独立正例，并排除重量、位置和坐标重合捷径。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )

        x = np.arange(len(folds))
        axes[0].bar(x, heldout, color="#0F766E")
        for index, value in enumerate(heldout):
            axes[0].text(index, value + 0.35, str(value), ha="center", va="bottom")
        axes[0].axhline(8, color="#DC2626", linestyle="--", linewidth=1.2, label="每组最低8条")
        axes[0].set_xticks(x, short_names, rotation=28, ha="right")
        axes[0].set_ylabel("严格未见的已知正关系数")
        axes[0].set_title("逐来源文件留一后的正例宽度", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="upper left")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        y = np.arange(len(labels))
        height = 0.36
        axes[1].barh(y - height / 2, recall5, height, label="Recall@5", color="#2563EB")
        axes[1].barh(y + height / 2, mrr, height, label="平均倒数排名", color="#D97706")
        axes[1].axvline(0.50, color="#DC2626", linestyle="--", linewidth=1.0)
        axes[1].set_yticks(y, labels)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1.0)
        axes[1].set_xlabel("已知正例恢复指标（越高越容易被捷径找回）")
        axes[1].set_title("同预算确定性基线", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="lower right")
        axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.075,
            0.185,
            (
                f"事实：470条序列化关系的并集GF(2)秩={metrics['union_gf2_rank']}（论文维数="
                f"{metrics['published_dimension']}）；仅{metrics['total_heldout_positives']}条关系只属于单一文件，"
                f"有效留出组={metrics['eligible_heldout_groups']}/8。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.075,
            0.108,
            "裁决：当前九轮正例-未标注排序基准未就绪；不训练E99，先补独立正关系来源并解释470与468的差异。",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.075,
            0.045,
            "证据范围：独立轮密钥ATM关系的本地readiness审计；不是PRESENT-80密钥调度验证、神经结果、区分器、攻击或SOTA。",
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
