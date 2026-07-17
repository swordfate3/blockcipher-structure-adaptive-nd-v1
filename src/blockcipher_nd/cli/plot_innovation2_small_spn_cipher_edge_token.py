from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_cipher_edge_token import (
    E33R_BEST_NEURAL_ANCHOR,
)
from blockcipher_nd.tasks.innovation2.small_spn_topology_training import BASELINE_AUC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E35 Cipher Edge-Token Transformer results."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_cipher_edge_token_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_cipher_edge_token_svg(summary: dict[str, Any], output: Path) -> None:
    rows = summary["rows"]
    gate = summary["gate"]
    fair_control = "fair_control" in str(summary.get("run_id", ""))
    wrong_rows = [
        row
        for row in rows
        if row["topology_mode"] != "true" and row["label_mode"] == "true"
    ]
    wrong_label = (
        "CETT\nrolled-P（协议无效）"
        if any(row["topology_mode"] == "shuffled" for row in wrong_rows)
        else "CETT\n公平corrupted-P"
    )
    groups = {
        "CETT\n真实P-layer": [
            row
            for row in rows
            if row["topology_mode"] == "true" and row["label_mode"] == "true"
        ],
        wrong_label: wrong_rows,
        "CETT\n标签打乱": [row for row in rows if row["label_mode"] == "shuffled"],
    }
    means = {
        name: {
            split: float(np.mean([row[f"{split}_auc"] for row in group]))
            for split in ("unseen_sbox", "unseen_player", "dual_unseen")
        }
        for name, group in groups.items()
    }
    decisions = {
        "innovation2_small_spn_cipher_edge_token_readiness_passed": "37-token表示与cell不变性就绪；运行冻结两seed矩阵。",
        "innovation2_small_spn_cipher_edge_token_confirmed": "显式edge-token通过；准备真实密码迁移readiness。",
        "innovation2_small_spn_cipher_edge_token_not_attributed": "edge-token收益未归因于真实P-layer。",
        "innovation2_small_spn_cipher_edge_token_not_ready": "edge-token未过门；关闭合成架构搜索并返回标签任务设计。",
        "innovation2_small_spn_cipher_edge_token_protocol_invalid": "tokenization、cell不变性、来源或metric协议无效。",
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
            (
                "创新2 E35b：显式密码边token的公平P-layer控制重裁决"
                if fair_control
                else "创新2 E35：显式密码边token能否建立真实P-layer贡献"
            ),
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.07,
            0.895,
            "CETT输入为16个node、16个有向P-layer edge、4个S-box relation和1个output-mask query，共37个token。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.07,
            0.850,
            "关键门：真实P-layer必须超过ID边际和错误P-layer各0.03；失败则关闭当前合成神经架构搜索。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )

        dual_groups = {
            "ID边际": [BASELINE_AUC["dual_unseen"]],
            "E33-R最佳\n神经锚点": [E33R_BEST_NEURAL_ANCHOR["dual_unseen"]],
            "CETT\n真实P": [
                float(row["dual_unseen_auc"])
                for row in groups["CETT\n真实P-layer"]
            ],
            "CETT\n错误P": [
                float(row["dual_unseen_auc"])
                for row in groups[wrong_label]
            ],
            "CETT\n标签打乱": [
                float(row["dual_unseen_auc"])
                for row in groups["CETT\n标签打乱"]
            ],
        }
        colors = ["#64748B", "#2563EB", "#0F766E", "#D97706", "#DC2626"]
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
        axes[0].set_xticks(np.arange(len(dual_groups)), list(dual_groups))
        axes[0].set_ylim(0.4, 1.0)
        axes[0].set_ylabel("dual-unseen AUC")
        axes[0].set_title(
            "边token候选与公平归因控制" if fair_control else "边token候选与归因控制",
            loc="left",
            fontweight="bold",
            pad=12,
        )
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)

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

        contract = gate.get("token_contract", {})
        figure.text(
            0.07,
            0.135,
            "确定性检查：token数={}；cell同时重标号最大logit误差={}；黑线表示seed均值。".format(
                contract.get("token_count", "unknown"),
                contract.get("cell_relabeling_max_abs_logit_error", "unknown"),
            ),
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
            "证据范围：冻结16-bit合成SPN的edge-token归因审计；不是PRESENT/GIFT/SKINNY真实密码结果或高轮SOTA。",
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
