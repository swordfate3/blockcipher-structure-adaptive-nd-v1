from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E53-B PRESENT S-box GLPK blocking gate."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_glpk_blocking_gate(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_glpk_blocking_gate(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    records = summary["metrics"]["records"]
    exponents = [int(record["output_exponent"]) for record in records]
    expected = [int(record["exact_expected_solutions"]) for record in records]
    completed = [
        int(record.get("solutions", 0))
        if record["status"] == "completed"
        else 0
        for record in records
    ]
    wall_seconds = [
        float(record.get("process_seconds", record["seconds"])) for record in records
    ]
    query_status = [record["status"] for record in records]
    timeout = float(summary["metrics"]["query_timeout_seconds"])
    decisions = {
        "innovation2_present_r5_open_3sdp_glpk_blocking_not_scalable": (
            "低复杂度计数正确，但逐解 blocking 在最重 S-box query 已超时；停止扩到五轮。"
        ),
        "innovation2_present_r5_open_3sdp_glpk_sbox_enumerator_ready": (
            "代表query均完成；下一步只扩一轮PRESENT电路fixture。"
        ),
        "innovation2_present_r5_open_3sdp_glpk_enumerator_invalid": (
            "GLPK约束、blocking、计数完整性或timeout分类无效。"
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
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 8.4))
        figure.subplots_adjust(
            left=0.065, right=0.975, top=0.70, bottom=0.25, wspace=0.38
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E53-B：Sage/GLPK逐解blocking能否承担消去感知的trail计数",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "每个解选择各输出坐标的一项S-box ANF term；完整枚举后按输入exponent计数并取GF(2)奇偶。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "冻结预算为10秒/query；timeout不提供部分parity，也不通过延长时间掩盖扩展性失败。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )

        positions = np.arange(len(records))
        width = 0.34
        axes[0].bar(
            positions - width / 2,
            expected,
            width,
            color="#CBD5E1",
            label="exact期望解数",
        )
        axes[0].bar(
            positions + width / 2,
            completed,
            width,
            color=["#0F766E" if value else "#DC2626" for value in completed],
            label="GLPK完整解数",
        )
        for index, (expected_value, completed_value) in enumerate(
            zip(expected, completed, strict=True)
        ):
            axes[0].text(
                index - width / 2,
                expected_value * 1.22,
                f"{expected_value:,}",
                ha="center",
                va="bottom",
                fontsize=8.6,
            )
            label = f"{completed_value:,}" if completed_value else "timeout"
            axes[0].text(
                index + width / 2,
                max(completed_value, 1) * 1.22,
                label,
                ha="center",
                va="bottom",
                fontsize=8.6,
                color="#B91C1C" if not completed_value else "#334155",
            )
        axes[0].set_yscale("log")
        axes[0].set_ylim(0.7, max(expected) * 3.5)
        axes[0].set_xticks(positions, [f"v={value}" for value in exponents])
        axes[0].set_ylabel("raw term-choice解数（log）")
        axes[0].set_title("完整计数要求与实际完成量", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")

        colors = ["#0F766E" if status == "completed" else "#DC2626" for status in query_status]
        axes[1].bar(positions, wall_seconds, color=colors, width=0.62)
        axes[1].axhline(
            timeout,
            color="#B91C1C",
            linestyle="--",
            linewidth=1.4,
            label=f"冻结timeout {timeout:g}秒",
        )
        for index, (seconds, status) in enumerate(zip(wall_seconds, query_status, strict=True)):
            label = f"{seconds:.2f}s" if status == "completed" else "timeout"
            axes[1].text(
                index,
                seconds + timeout * 0.035,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        axes[1].set_xticks(positions, [f"v={value}" for value in exponents])
        axes[1].set_ylim(0, timeout * 1.22)
        axes[1].set_ylabel("每个独立Sage进程墙钟时间（秒）")
        axes[1].set_title("blocking成本随term组合增长", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper left", fontsize=8.5)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)

        state_colors = [
            "#0F766E" if status == "completed" else "#DC2626"
            for status in query_status
        ]
        axes[2].barh(positions, np.ones(len(records)), color=state_colors, height=0.58)
        for index, record in enumerate(records):
            label = (
                "count与parity精确一致"
                if record["status"] == "completed"
                else "10秒未完成，parity未知"
            )
            axes[2].text(
                0.5,
                index,
                label,
                ha="center",
                va="center",
                color="#FFFFFF",
                fontweight="bold",
                fontsize=8.6,
            )
        axes[2].set_yticks(positions, [f"v={value}" for value in exponents])
        axes[2].set_xlim(0, 1)
        axes[2].set_xticks([])
        axes[2].set_title("逐query证据状态", loc="left", fontweight="bold")
        axes[2].invert_yaxis()
        axes[2].text(
            0.0,
            -0.13,
            "当前环境无PySAT、CryptoMiniSat、Z3、BDD或model-counter后端。",
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            color="#526070",
            fontsize=8.6,
        )

        figure.text(
            0.065,
            0.145,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=10.0,
            color="#334155",
        )
        figure.text(
            0.065,
            0.085,
            "推荐：转GF(2) transition tensor变量消元宽度审计；五轮子集、神经训练和远程GPU继续关闭。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.065,
            0.035,
            "证据范围：PRESENT S-box逐解GLPK正确性/扩展性门；不是全密码provider、五轮标签或攻击结果。",
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
