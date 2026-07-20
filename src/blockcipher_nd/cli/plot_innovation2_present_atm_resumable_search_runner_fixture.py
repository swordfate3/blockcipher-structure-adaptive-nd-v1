from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render E102 ATM resumable-runner fixture evidence.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_resumable_runner_fixture(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_resumable_runner_fixture(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    checks = gate["fixture_checks"]
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
        figure, axes = plt.subplots(1, 3, figsize=(16.8, 9.2))
        figure.subplots_adjust(left=0.065, right=0.975, top=0.70, bottom=0.29, wspace=0.38)
        figure.text(
            0.065,
            0.958,
            "创新2 E102：ATM逐候选断点恢复一致性门",
            ha="left",
            va="top",
            fontsize=15.2,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.897,
            "同一3-bit确定性Avec任务：完整运行，对比“完成1个候选后中断，再从原子缓存恢复”。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.065,
            0.849,
            "只验证runner持久化与恢复；没有构造或搜索PRESENT九/十轮关系，也没有训练网络或运行远程任务。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        run_labels = ("无中断锚点", "中断后恢复", "损坏缓存恢复")
        run_calls = (
            metrics["anchor_oracle_calls"],
            metrics["resumed_oracle_calls"],
            metrics["corrupt_recovery_oracle_calls"],
        )
        colors = ("#2563EB", "#0F766E", "#D97706")
        x = np.arange(len(run_labels))
        axes[0].bar(x, run_calls, color=colors, width=0.64)
        axes[0].set_xticks(x, run_labels, rotation=16, ha="right")
        axes[0].set_ylabel("Avec候选调用次数")
        axes[0].set_title("正常恢复没有增加oracle调用", loc="left", fontweight="bold")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(run_calls):
            axes[0].text(index, value + 0.25, str(value), ha="center", fontweight="bold")
        axes[0].text(
            0.03,
            0.96,
            f"中断时已落盘 {metrics['calls_durable_at_interrupt']} 个；恢复复用 {metrics['resumed_reused_candidates']} 个",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color="#475569",
        )

        selected = (
            ("结果逐字节相等", "canonical_result_bytes_equal"),
            ("完成候选不重算", "completed_candidate_not_recalled"),
            ("参数漂移拒绝", "parameter_mismatch_rejected"),
            ("损坏缓存重算", "corrupt_candidate_recomputed"),
            ("临时文件忽略", "incomplete_temporary_candidate_ignored"),
            ("完成hash复用", "completed_result_hash_reuse_zero_calls"),
            ("原子完成顺序", "result_precedes_complete_marker"),
        )
        values = [1 if checks[key] else 0 for _, key in selected]
        y = np.arange(len(selected))
        axes[1].barh(
            y,
            [1] * len(selected),
            color=["#0F766E" if value else "#DC2626" for value in values],
        )
        axes[1].set_yticks(y, [label for label, _ in selected])
        axes[1].set_xticks((0, 1), ("失败", "通过"))
        axes[1].set_xlim(0, 1.15)
        axes[1].invert_yaxis()
        axes[1].set_title("恢复与完整性契约逐项通过", loc="left", fontweight="bold")
        for index, value in enumerate(values):
            axes[1].text(
                0.5,
                index,
                "通过" if value else "失败",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        path_labels = ("直接basis", "WUV/nullspace", "key-dependent")
        path_values = (
            metrics["basis_candidates"],
            metrics["wuv_candidates"],
            metrics["key_dependent_candidates"],
        )
        px = np.arange(len(path_labels))
        axes[2].bar(px, path_values, color=("#2563EB", "#7C3AED", "#DC2626"), width=0.64)
        axes[2].set_xticks(px, path_labels, rotation=15, ha="right")
        axes[2].set_ylabel("fixture候选数")
        axes[2].set_title("官方搜索的三条结果路径均被覆盖", loc="left", fontweight="bold")
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        for index, value in enumerate(path_values):
            axes[2].text(index, value + 0.25, str(value), ha="center", fontweight="bold")

        decision = {
            "innovation2_present_atm_resumable_runner_fixture_passed": (
                "通过：开放E103真实ATM低成本兼容性门；R9/R10长搜索仍关闭。"
            ),
            "innovation2_present_atm_resumable_runner_fixture_insufficient": (
                "暂缓：fixture路径覆盖不足，只扩充fixture。"
            ),
            "innovation2_present_atm_resumable_runner_protocol_invalid": (
                "暂缓/失败：修复来源、恢复或完整性协议。"
            ),
        }[gate["decision"]]
        figure.text(
            0.065,
            0.176,
            f"检查：来源 {metrics['source_checks_passed']}/{metrics['source_checks']}；恢复协议 {metrics['fixture_checks_passed']}/{metrics['fixture_checks']}；产物 {metrics['artifact_checks_passed']}/{metrics['artifact_checks']}。",
            ha="left",
            va="bottom",
            fontsize=9.6,
            color="#334155",
        )
        figure.text(
            0.065,
            0.108,
            f"裁决：{decision}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            color="#334155",
        )
        figure.text(
            0.065,
            0.045,
            "证据范围：本地确定性fixture的runner持久化/恢复证据；不是九/十轮新关系、PRESENT-80区分器、攻击或SOTA结果。",
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
