from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E58-B ATM r9 SAT probe.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_atm_native_sat_r9_probe(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_atm_native_sat_r9_probe(summary: dict[str, Any], output: Path) -> None:
    candidate = summary["candidate"]
    gate = summary["gate"]
    decisions = {
        "innovation2_atm_native_sat_r9_wall_clock_cap_exceeded": (
            "worker超过冻结60秒；candidate保持unknown，关闭当前九轮exact witness路线。"
        ),
        "innovation2_atm_native_sat_r9_strict_negative_found": (
            "九轮relation级odd key-monomial已复验，可进入严格负类宽度审计。"
        ),
        "innovation2_atm_native_sat_r9_negative_not_proven": (
            "worker完成但未证明负类；candidate保持unknown。"
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
        figure, axes = plt.subplots(1, 3, figsize=(16.0, 8.4))
        figure.subplots_adjust(
            left=0.07, right=0.975, top=0.70, bottom=0.28, wspace=0.42
        )
        figure.text(
            0.07,
            0.955,
            "创新2 E58-B：ATM原生SAT能否在九轮给出严格负类见证",
            ha="left",
            va="top",
            fontsize=14.8,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "冻结mutation：输入u不变，输出单bit从0x1旋转到0x2；relation size和重量边际完全保持。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.848,
            "严格规则：60秒内必须返回非零key指数，并在完整relation上独立重放为odd；否则只能是unknown。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        marginal_labels = ["relation size", "输入重量", "输出重量"]
        source_values = [1, int(candidate["input_weight"]), int(candidate["source_output_weight"])]
        candidate_values = [1, int(candidate["input_weight"]), int(candidate["candidate_output_weight"])]
        positions = np.arange(3)
        axes[0].bar(positions - 0.18, source_values, width=0.36, label="公开正类", color="#0F766E")
        axes[0].bar(positions + 0.18, candidate_values, width=0.36, label="冻结mutation", color="#D97706")
        axes[0].set_xticks(positions, marginal_labels)
        axes[0].set_ylabel("边际数值")
        axes[0].set_title("candidate只改变输出bit位置", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="upper left")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        caps = [
            int(gate["wall_clock_cap_seconds"]),
            int(gate["projected_key_cap"]),
            int(gate["trail_model_cap"]),
        ]
        cap_values = [math.log2(value) for value in caps]
        axes[1].bar(
            np.arange(3),
            cap_values,
            color=["#64748B", "#2563EB", "#2563EB"],
            width=0.62,
        )
        axes[1].set_xticks(np.arange(3), ["墙钟秒数", "key候选cap", "trail模型cap"])
        axes[1].set_ylabel("冻结上限的log2")
        axes[1].set_title("没有后验增加预算", loc="left", fontweight="bold")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(caps):
            label = "60秒" if index == 0 else f"2^{int(cap_values[index])}"
            axes[1].text(index, cap_values[index] + 0.5, label, ha="center", fontweight="bold")

        outcome_labels = ["边际控制", "worker完成", "strict witness", "开放训练"]
        outcome_values = [
            all(gate["candidate_checks"].values()),
            gate["witness_checks"]["worker_completed_within_wall_clock_cap"],
            gate["witness_checks"]["nonzero_key_exponent_witness_found"],
            gate["next_action"]["training"],
        ]
        outcome_colors = ["#0F766E", "#94A3B8", "#94A3B8", "#94A3B8"]
        axes[2].barh(np.arange(4), [1] * 4, color=outcome_colors, height=0.58)
        axes[2].set_yticks(np.arange(4), outcome_labels)
        axes[2].set_xlim(0, 1.12)
        axes[2].set_xticks([])
        axes[2].invert_yaxis()
        axes[2].set_title("结果：超时不等于负类", loc="left", fontweight="bold")
        labels = ["通过", "超时", "未证明", "关闭"]
        for index, label in enumerate(labels):
            axes[2].text(0.5, index, label, ha="center", va="center", color="#FFFFFF", fontweight="bold")

        figure.text(
            0.07,
            0.17,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.07,
            0.108,
            "下一步：不换mutation、不提高cap；关闭ATM九轮监督标签路线，重新排名可执行的低轮严格标签任务。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.07,
            0.048,
            "证据范围：独立轮密钥九轮单候选provider边界；不是PRESENT-80负类、神经结果、攻击或SOTA。",
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
