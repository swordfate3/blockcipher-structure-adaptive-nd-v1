from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E55 PRESENT r3 query-cone sparse-ANF growth gate."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_sparse_growth_gate(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_sparse_growth_gate(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    config = summary["metadata"]["config"]
    rows = summary["query_rows"]
    calibration = summary["calibration"]
    labels = [str(row["query_id"]).replace("r3_query_", "Q") for row in rows]
    terms = [
        max(1, int(row.get("maximum_observed_terms") or 0)) for row in rows
    ]
    seconds = [max(0.0, float(row.get("elapsed_seconds") or 0.0)) for row in rows]
    colors = [
        "#0F766E"
        if row["status"] == "completed" and row["label"] == "positive"
        else "#2563EB"
        if row["status"] == "completed"
        else "#DC2626"
        if row["status"] == "cap_exceeded"
        else "#CBD5E1"
        for row in rows
    ]
    decisions = {
        "innovation2_present_r3_query_cone_sparse_anf_ready": (
            "12个三轮query全部过硬cap并具备正负标签；可进入同cap四轮门。"
        ),
        "innovation2_present_r3_query_cone_sparse_anf_hard_cap_exceeded": (
            "至少一个三轮query越过冻结硬cap；关闭当前full-variable sparse provider。"
        ),
        "innovation2_present_r3_query_cone_sparse_anf_label_diversity_insufficient": (
            "冻结query缺少正负标签多样性；不后验换query。"
        ),
        "innovation2_present_r3_query_cone_sparse_anf_protocol_invalid": (
            "E53-A重放、位序或语义控制失败；三轮结果不可解释。"
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
        figure, axes = plt.subplots(1, 3, figsize=(16.2, 8.4))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.27, wspace=0.38
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E55：PRESENT三轮指定输出的exact sparse-ANF能否在硬cap内完成",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "每个query只反向保留相关状态锥，但64个明文变量与80个master-key变量继续保持符号化。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "冻结12个query：4个二轮正类锚点、4个二轮负类锚点、4个multi-mask；不按三轮结果后验筛选。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )

        positions = np.arange(len(rows))
        axes[0].bar(positions, terms, color=colors, width=0.72)
        axes[0].axhline(
            int(config["maximum_terms"]),
            color="#DC2626",
            linestyle="--",
            linewidth=1.4,
            label=f"单项式门 {int(config['maximum_terms']):,}",
        )
        axes[0].set_yscale("log")
        axes[0].set_ylim(1, max(max(terms) * 3, int(config["maximum_terms"]) * 2))
        axes[0].set_xticks(positions, labels, rotation=45, ha="right")
        axes[0].set_ylabel("query内最大稀疏单项式数（log）")
        axes[0].set_title("单项式增长与停止点", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="upper right", fontsize=8.3)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")

        axes[1].bar(positions, seconds, color=colors, width=0.72)
        axes[1].axhline(
            float(config["maximum_seconds"]),
            color="#DC2626",
            linestyle="--",
            linewidth=1.4,
            label=f"时间门 {float(config['maximum_seconds']):g}秒",
        )
        axes[1].set_xticks(positions, labels, rotation=45, ha="right")
        axes[1].set_ylabel("墙钟时间（秒）")
        axes[1].set_title("每query本地CPU耗时", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper right", fontsize=8.3)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        checks = calibration["checks"]
        check_labels = [
            "32行校准全部执行",
            "superpoly完全一致",
            "unit输出hash一致",
            "标量PRESENT一致",
            "错误P层被识别",
            "零offset被拒绝",
        ]
        values = [int(value) for value in checks.values()]
        check_positions = np.arange(len(values))
        axes[2].barh(
            check_positions,
            values,
            color=["#0F766E" if value else "#DC2626" for value in values],
            height=0.65,
        )
        axes[2].set_xlim(0, 1.15)
        axes[2].set_xticks([0, 1], ["失败", "通过"])
        axes[2].set_yticks(check_positions, check_labels)
        axes[2].invert_yaxis()
        axes[2].set_title(
            f"E53-A校准：{calibration['completed_rows']}/{calibration['expected_rows']}行",
            loc="left",
            fontweight="bold",
        )
        axes[2].grid(axis="x", color="#E5E7EB", linewidth=0.8)

        figure.text(
            0.065,
            0.165,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=10.0,
            color="#334155",
        )
        next_actions = {
            "close the current exact full-variable sparse provider family": (
                "关闭当前全变量exact sparse-ANF提供器；不升cap、不转远程、不训练五轮网络。"
            ),
            "run the same frozen query-cone and hard caps at four rounds": (
                "以相同query和硬cap进入四轮计算门，仍不启动五轮网络。"
            ),
            "close this frozen sparse query family without post-hoc query selection": (
                "关闭当前冻结query族，不后验更换query补标签。"
            ),
            "repair E53-A replay, bit order, or semantic controls before any r3 inference": (
                "先修复E53-A重放、位序或语义控制，不解释三轮结果。"
            ),
            "repair incomplete or inconsistent query execution": (
                "修复不完整或不一致的query执行后重新校准。"
            ),
        }
        raw_next_action = str(gate["next_action"]["action"])
        next_action = next_actions.get(raw_next_action, raw_next_action)
        figure.text(
            0.065,
            0.105,
            f"下一步：{next_action}",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.065,
            0.048,
            "证据范围：PRESENT-80三轮exact sparse-ANF计算可行性；不是五轮标签、神经预测、攻击或SOTA结果。",
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
