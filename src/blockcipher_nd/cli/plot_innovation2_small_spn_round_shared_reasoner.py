from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_round_shared_reasoner import (
    E33R_STATIC_MEAN_AUC,
)
from blockcipher_nd.tasks.innovation2.small_spn_topology_training import BASELINE_AUC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E34 round-shared neural reasoner results."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_round_shared_reasoner_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_round_shared_reasoner_svg(summary: dict[str, Any], output: Path) -> None:
    rows = summary["rows"]
    gate = summary["gate"]
    groups = {
        "共享轮处理器\n真实P-layer": [
            row
            for row in rows
            if row["topology_mode"] == "true" and row["label_mode"] == "true"
        ],
        "共享轮处理器\n错误P-layer": [
            row
            for row in rows
            if row["topology_mode"] == "shuffled" and row["label_mode"] == "true"
        ],
        "共享轮处理器\n标签打乱": [row for row in rows if row["label_mode"] == "shuffled"],
    }
    means = {
        name: {
            split: float(np.mean([row[f"{split}_auc"] for row in group]))
            for split in ("unseen_sbox", "unseen_player", "dual_unseen")
        }
        for name, group in groups.items()
    }
    decisions = {
        "innovation2_small_spn_round_shared_readiness_passed": "共享轮处理器就绪；运行冻结两seed矩阵。",
        "innovation2_small_spn_round_shared_reasoner_confirmed": "共享轮处理器通过；只开放同处理器SCGT增益审计。",
        "innovation2_small_spn_round_shared_topology_not_attributed": "候选收益仍未归因于真实P-layer。",
        "innovation2_small_spn_round_shared_reasoner_not_ready": "共享轮处理器未过门；停止合成GraphGPS/looped家族。",
        "innovation2_small_spn_round_shared_protocol_invalid": "可变步数、等变性、来源或metric协议无效。",
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
        figure, axes = plt.subplots(1, 2, figsize=(14.5, 7.6))
        figure.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.22, wspace=0.28)
        figure.text(
            0.07,
            0.955,
            "创新2 E34：按实际轮数循环的共享图处理器能否恢复拓扑贡献",
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "唯一变量：3个独立静态block改为1个共享block，并按样本轮数执行2至5次；其他协议与E33-R一致。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.850,
            "关键门：真实P-layer必须超过ID边际和错误P-layer各0.03；否则停止合成GraphGPS/looped家族。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        dual_groups = {
            "ID边际": [BASELINE_AUC["dual_unseen"]],
            "E33-R静态\n真实P": [E33R_STATIC_MEAN_AUC["true_player"]["dual_unseen"]],
            "E33-R静态\n错误P": [E33R_STATIC_MEAN_AUC["wrong_player"]["dual_unseen"]],
            "E34共享\n真实P": [
                float(row["dual_unseen_auc"])
                for row in groups["共享轮处理器\n真实P-layer"]
            ],
            "E34共享\n错误P": [
                float(row["dual_unseen_auc"])
                for row in groups["共享轮处理器\n错误P-layer"]
            ],
            "共享\n标签打乱": [
                float(row["dual_unseen_auc"])
                for row in groups["共享轮处理器\n标签打乱"]
            ],
        }
        colors = ["#64748B", "#2563EB", "#D97706", "#0F766E", "#B45309", "#DC2626"]
        for index, ((name, points), color) in enumerate(zip(dual_groups.items(), colors, strict=True)):
            jitter = np.linspace(-0.07, 0.07, len(points)) if len(points) > 1 else np.asarray([0.0])
            axes[0].scatter(
                np.full(len(points), index) + jitter,
                points,
                s=62,
                color=color,
                zorder=3,
            )
            axes[0].plot(
                [index - 0.13, index + 0.13],
                [float(np.mean(points)), float(np.mean(points))],
                color="#0F172A",
                linewidth=1.5,
            )
        axes[0].axhline(
            BASELINE_AUC["dual_unseen"],
            color="#64748B",
            linestyle=":",
            linewidth=1.2,
            label="ID边际 0.7265",
        )
        axes[0].axhline(
            BASELINE_AUC["dual_unseen"] + 0.03,
            color="#DC2626",
            linestyle="--",
            linewidth=1.1,
            label="推进线 0.7565",
        )
        axes[0].set_xticks(np.arange(len(dual_groups)), list(dual_groups), rotation=8, ha="right")
        axes[0].set_ylim(0.4, 1.0)
        axes[0].set_ylabel("dual-unseen AUC")
        axes[0].set_title("静态锚点、共享处理器与控制", loc="left", fontweight="bold", pad=12)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper right", fontsize=8.5)

        comparison_names = ["ID边际", *groups]
        split_keys = ["unseen_sbox", "unseen_player", "dual_unseen"]
        split_labels = ["未见S-box", "未见P-layer", "双轴未见"]
        split_colors = ["#2563EB", "#D97706", "#0F766E"]
        x = np.arange(len(comparison_names))
        width = 0.24
        for split_index, (split, label, color) in enumerate(
            zip(split_keys, split_labels, split_colors, strict=True)
        ):
            values = [BASELINE_AUC[split]] + [means[name][split] for name in groups]
            axes[1].bar(
                x + (split_index - 1) * width,
                values,
                width,
                label=label,
                color=color,
            )
        axes[1].axhline(
            BASELINE_AUC["dual_unseen"] + 0.03,
            color="#DC2626",
            linestyle="--",
            linewidth=1.1,
            label="dual推进线",
        )
        axes[1].set_xticks(x, comparison_names)
        axes[1].set_ylim(0.4, 1.0)
        axes[1].set_ylabel("heldout AUC（seed均值）")
        axes[1].set_title("三个组外拆分", loc="left", fontweight="bold", pad=12)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(frameon=False, ncol=2, fontsize=8.5)

        relabel_error = gate.get("cell_relabeling_max_abs_logit_error", "unknown")
        figure.text(
            0.07,
            0.135,
            f"确定性检查：共享处理器按轮数执行；cell同时重标号最大logit误差={relabel_error}；黑线为seed均值。",
            ha="left",
            va="bottom",
            fontsize=9.3,
            color="#334155",
        )
        figure.text(
            0.07,
            0.090,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.07,
            0.045,
            "证据范围：冻结16-bit合成SPN的共享轮处理器审计；不是PRESENT/GIFT/SKINNY真实密码结果或高轮SOTA。",
            ha="left",
            va="bottom",
            fontsize=8.9,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())

