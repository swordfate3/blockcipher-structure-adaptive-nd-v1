from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E101 high-round source/resume evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_source_generation_readiness(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_source_generation_readiness(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    splits = summary["split_coverage"]
    costs = summary["historical_costs"]
    resume = [row for row in summary["resume_contract"] if row["required_for_generation"]]
    declared = [sum(row["rounds"] == rounds for row in splits) for rounds in (9, 10)]
    public = [sum(row["rounds"] == rounds and row["pickle_present"] for row in splits) for rounds in (9, 10)]
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
        figure.text(0.065, 0.958, "创新2 E101：PRESENT九/十轮ATM新来源生成就绪审计", ha="left", va="top", fontsize=15.2, fontweight="bold", color="#0F172A")
        figure.text(0.065, 0.897, "公开notebook声明R9与R10高轮搜索；本图区分“声明可调用”“已有公开结果”和“长任务可恢复”。", ha="left", va="top", fontsize=9.8, color="#526070")
        figure.text(0.065, 0.849, "本次只读源码与历史stats，没有执行ATM搜索、神经训练或远程任务。", ha="left", va="top", fontsize=9.8, color="#526070")

        x = np.arange(2)
        axes[0].bar(x - 0.19, declared, 0.38, label="notebook声明", color="#2563EB")
        axes[0].bar(x + 0.19, public, 0.38, label="公开pickle", color="#0F766E")
        for index, (left, right) in enumerate(zip(declared, public, strict=True)):
            axes[0].text(index - 0.19, left + 0.2, str(left), ha="center")
            axes[0].text(index + 0.19, right + 0.2, str(right), ha="center")
        axes[0].set_xticks(x, ("PRESENT 9轮", "PRESENT 10轮"))
        axes[0].set_ylim(0, 10.5)
        axes[0].set_ylabel("split数量")
        axes[0].set_title("声明18个，公开结果只有R9的8个", loc="left", fontweight="bold")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        labels = [row["split"] for row in costs]
        hours = [row["hours"] for row in costs]
        calls = [row["oracle_calls"] for row in costs]
        cx = np.arange(len(costs))
        axes[1].bar(cx, hours, color="#D97706", label="历史耗时（小时）")
        axes[1].set_xticks(cx, labels, rotation=28, ha="right")
        axes[1].set_ylabel("历史耗时（小时）")
        axes[1].set_title("单个公开R9 split曾耗时0.75–6.61小时", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        call_axis = axes[1].twinx()
        call_axis.plot(cx, calls, color="#7C3AED", marker="o", linewidth=1.7, label="oracle calls")
        call_axis.set_yscale("log")
        call_axis.set_ylabel("oracle calls（对数轴）")
        handles1, labels1 = axes[1].get_legend_handles_labels()
        handles2, labels2 = call_axis.get_legend_handles_labels()
        axes[1].legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8.3)

        names = {
            "started_marker": "启动标记",
            "progress_jsonl": "持续进度",
            "incremental_candidate_or_layer_cache": "候选/层缓存",
            "parameter_matched_resume": "参数匹配恢复",
            "atomic_completion": "原子完成",
            "nonblocking_incremental_result_boundary": "增量返回边界",
            "resume_fixture_verified": "中断等价fixture",
        }
        values = [1 if row["passed"] else 0 for row in resume]
        colors = ["#0F766E" if value else "#DC2626" for value in values]
        ry = np.arange(len(resume))
        axes[2].barh(ry, [1] * len(resume), color=colors)
        axes[2].set_yticks(ry, [names[row["check"]] for row in resume])
        axes[2].set_xticks((0, 1), ("缺失", "具备"))
        axes[2].set_xlim(0, 1.15)
        axes[2].invert_yaxis()
        axes[2].set_title("原notebook不具备长任务恢复契约", loc="left", fontweight="bold")
        for index, value in enumerate(values):
            axes[2].text(0.5, index, "通过" if value else "缺失", ha="center", va="center", color="white", fontweight="bold")

        metrics = gate["metrics"]
        figure.text(0.065, 0.176, f"历史成本：8个R9 split耗时中位数{metrics['historical_median_seconds']/3600:.2f}小时；最大oracle calls={metrics['historical_max_oracle_calls']:,}。", ha="left", va="bottom", fontsize=9.6, color="#334155")
        decision_text = {
            "innovation2_present_high_round_source_generation_ready": "通过：只允许预注册可恢复的R9缺失split生成。",
            "innovation2_present_high_round_resumable_runner_required": "暂缓：先实现并验证可恢复runner，不启动长搜索。",
            "innovation2_present_high_round_source_generation_audit_protocol_invalid": "失败：冻结来源、notebook或stats重放无效。",
        }[gate["decision"]]
        figure.text(0.065, 0.108, f"裁决：{decision_text}", ha="left", va="bottom", fontsize=9.8, color="#334155")
        figure.text(0.065, 0.045, "证据范围：只读高轮来源/成本/恢复审计；缺失split不是零维或负证据，本次没有生成新关系、训练网络或运行远程任务。", ha="left", va="bottom", fontsize=9.0, color="#526070")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
