from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E54 PRESENT transition-tensor boundary audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_tensor_boundary_audit(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_tensor_boundary_audit(summary: dict[str, Any], output: Path) -> None:
    gate = summary["gate"]
    metrics = summary["metrics"]
    routes = metrics["routes"]
    sparse = metrics["sparse_evidence"]
    route_labels = ["全key\n全offset", "固定key\n全offset", "全key\n零offset", "固定key\n固定offset"]
    retained = [int(row["retained_variables"]) for row in routes]
    bitpacked_gib = [
        float(row["bitpacked_dense_bytes"]) / (1 << 30) for row in routes
    ]
    semantic = [bool(row["semantic_match"]) for row in routes]
    decisions = {
        "innovation2_present_r5_transition_tensor_boundary_infeasible": (
            "最终136维语义边界已超过dense tensor门；跳过内部因子图和min-fill。"
        ),
        "innovation2_present_r5_transition_tensor_internal_width_audit_ready": (
            "边界门通过；可构造真实PRESENT因子图。"
        ),
        "innovation2_present_r5_transition_tensor_boundary_protocol_invalid": (
            "E53-A来源或全key/全offset边界计算无效。"
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
            left=0.065, right=0.975, top=0.70, bottom=0.25, wspace=0.40
        )
        figure.text(
            0.065,
            0.955,
            "创新2 E54：五轮完整 superpoly 的GF(2) tensor边界是否可表示",
            ha="left",
            va="top",
            fontsize=15.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.895,
            "原始语义必须同时保留80个master-key变量和56个inactive plaintext变量；内部消元不能删除最终边界。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )
        figure.text(
            0.065,
            0.848,
            "Phase 0边界失败即停止：不以零offset、固定key或较小内部treewidth替代全key/全offset证书。",
            ha="left",
            va="top",
            fontsize=9.7,
            color="#526070",
        )

        positions = np.arange(len(routes))
        colors = ["#0F766E" if value else "#94A3B8" for value in semantic]
        axes[0].bar(positions, retained, color=colors, width=0.62)
        axes[0].axhline(
            metrics["maximum_retained_variables"],
            color="#DC2626",
            linestyle="--",
            linewidth=1.4,
            label="变量门 26",
        )
        for index, value in enumerate(retained):
            axes[0].text(index, value + 3, str(value), ha="center", va="bottom", fontweight="bold")
        axes[0].set_xticks(positions, route_labels)
        axes[0].set_ylim(0, max(retained) * 1.18)
        axes[0].set_ylabel("最终必须保留的二进制变量数")
        axes[0].set_title("语义边界下界", loc="left", fontweight="bold")
        axes[0].legend(frameon=False, loc="upper right")
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].text(
            0.02,
            0.94,
            "绿色是唯一符合原始语义的路线",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            color="#0F766E",
            fontsize=8.7,
        )

        axes[1].bar(positions, bitpacked_gib, color=colors, width=0.62)
        axes[1].axhline(
            metrics["maximum_dense_gib"],
            color="#DC2626",
            linestyle="--",
            linewidth=1.4,
            label="内存门 4 GiB",
        )
        for index, value in enumerate(bitpacked_gib):
            label = f"{value:.1e}" if value >= 1000 else f"{value:.2g}"
            axes[1].text(
                index,
                value * 1.35 if value > 0 else 1e-10,
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        axes[1].set_yscale("log")
        axes[1].set_ylim(1e-10, max(bitpacked_gib) * 20)
        axes[1].set_xticks(positions, route_labels)
        axes[1].set_ylabel("bit-packed dense tensor内存（GiB，log）")
        axes[1].set_title("最终tensor本身已不可存储", loc="left", fontweight="bold")
        axes[1].legend(frameon=False, loc="upper right")
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")

        round_positions = np.arange(2)
        output_terms = [
            sparse["round_output_anf_terms"]["1"],
            sparse["round_output_anf_terms"]["2"],
        ]
        superpoly_terms = [
            sparse["round_fixture_maximum_superpoly_terms"]["1"],
            sparse["round_fixture_maximum_superpoly_terms"]["2"],
        ]
        width = 0.34
        axes[2].bar(
            round_positions - width / 2,
            output_terms,
            width,
            color="#2563EB",
            label="全部输出ANF总项数",
        )
        axes[2].bar(
            round_positions + width / 2,
            superpoly_terms,
            width,
            color="#D97706",
            label="fixture最大superpoly",
        )
        for offset, values in ((-width / 2, output_terms), (width / 2, superpoly_terms)):
            for index, value in enumerate(values):
                axes[2].text(
                    index + offset,
                    value * 1.28,
                    f"{value:,}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
        axes[2].set_yscale("log")
        axes[2].set_ylim(5, max(output_terms) * 5)
        axes[2].set_xticks(round_positions, ["PRESENT 1轮", "PRESENT 2轮"])
        axes[2].set_ylabel("稀疏单项式数量（log）")
        axes[2].set_title(
            "稀疏ANF也快速增长\nr2/r1：总项2,282.6×，最大superpoly 4,107.1×",
            loc="left",
            fontweight="bold",
            fontsize=9.7,
            pad=8,
        )
        axes[2].legend(frameon=False, loc="upper left", fontsize=8.3)
        axes[2].grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")

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
            "推荐：只运行一次带硬cap的三轮query-cone sparse-ANF增长门；五轮子集、神经训练和远程GPU继续关闭。",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#334155",
        )
        figure.text(
            0.065,
            0.035,
            "证据范围：PRESENT-80五轮full-superpoly最终边界审计；不是内部treewidth、五轮标签或攻击结果。",
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
