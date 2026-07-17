from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation2.small_spn_topology_training import BASELINE_AUC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E33 small-SPN GraphGPS/SCGT attribution results."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_topology_training_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_topology_training_svg(summary: dict[str, Any], output: Path) -> None:
    rows = summary["rows"]
    gate = summary["gate"]
    groups = {
        "GraphGPS\n真实拓扑": [
            row
            for row in rows
            if row["model_name"] == "graphgps"
            and row["topology_mode"] == "true"
            and row["label_mode"] == "true"
        ],
        "SCGT\n真实拓扑": [
            row
            for row in rows
            if row["model_name"] == "scgt" and row["label_mode"] == "true"
        ],
        "GraphGPS\n错误P-layer": [
            row
            for row in rows
            if row["topology_mode"] == "shuffled" and row["label_mode"] == "true"
        ],
        "GraphGPS\n标签打乱": [row for row in rows if row["label_mode"] == "shuffled"],
    }
    group_names = ["ID边际\nbaseline", *groups]
    split_keys = ["unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["未见S-box", "未见P-layer", "双轴未见"]
    colors = ["#2563EB", "#D97706", "#0F766E"]
    values: dict[str, list[float]] = {
        split: [BASELINE_AUC[split]]
        + [float(np.mean([row[f"{split}_auc"] for row in group])) for group in groups.values()]
        for split in split_keys
    }
    decisions = {
        "innovation2_small_spn_topology_training_readiness_passed": "训练实现就绪；运行冻结两seed归因矩阵。",
        "innovation2_small_spn_topology_predictor_ready": "真实topology predictor过门；进入真实密码迁移readiness。",
        "innovation2_small_spn_topology_signal_not_attributed": "模型收益未归因于真实拓扑或label-shuffle异常。",
        "innovation2_small_spn_topology_predictor_not_ready": "GraphGPS未超过冻结边际；停止当前网络路线。",
        "innovation2_small_spn_topology_training_protocol_invalid": "来源、split、forward、checkpoint或metric协议无效。",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.3,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16.0, 8.4), gridspec_kw={"width_ratios": [1.45, 1]})
        figure.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.25, wspace=0.24)
        figure.suptitle(
            "创新2 E33：小状态SPN GraphGPS / SCGT 拓扑归因",
            x=0.08,
            y=0.96,
            ha="left",
            fontsize=15.2,
            fontweight="bold",
        )
        figure.text(
            0.08,
            0.886,
            "训练仅使用9个train topology的5301行matched标签；三个heldout split不参与cell选择、checkpoint或超参数选择。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.838,
            "比较真实S-box/P-layer、错误P-layer和label-shuffle；所有神经行使用同一预算，结果不是实际密码攻击。",
            ha="left",
            va="top",
            fontsize=9.5,
            color="#526070",
        )
        x = np.arange(len(group_names))
        width = 0.24
        for split_index, (split, label, color) in enumerate(
            zip(split_keys, split_labels, colors, strict=True)
        ):
            offset = (split_index - 1) * width
            axes[0].bar(x + offset, values[split], width, label=label, color=color)
        axes[0].axhline(
            BASELINE_AUC["dual_unseen"] + 0.03,
            color="#DC2626",
            linestyle="--",
            linewidth=1.1,
            label="dual推进线 0.7565",
        )
        axes[0].set_xticks(x, group_names)
        axes[0].set_ylim(0.4, 1.0)
        axes[0].set_ylabel("heldout AUC（seed均值）")
        axes[0].set_title("同预算模型与必要控制", loc="left", fontweight="bold", pad=12)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(frameon=False, ncol=2, fontsize=8.5)

        dual_baseline = BASELINE_AUC["dual_unseen"]
        axes[1].axhline(dual_baseline, color="#64748B", linestyle=":", linewidth=1.2, label="dual ID边际")
        axes[1].axhline(dual_baseline + 0.03, color="#DC2626", linestyle="--", linewidth=1.1, label="dual推进线")
        for group_index, (name, group) in enumerate(groups.items()):
            points = [row["dual_unseen_auc"] for row in group]
            jitter = np.linspace(-0.07, 0.07, len(points)) if len(points) > 1 else np.asarray([0.0])
            axes[1].scatter(
                np.full(len(points), group_index) + jitter,
                points,
                s=58,
                color=["#2563EB", "#7C3AED", "#D97706", "#DC2626"][group_index],
                zorder=3,
            )
            axes[1].plot(
                [group_index - 0.12, group_index + 0.12],
                [float(np.mean(points)), float(np.mean(points))],
                color="#0F172A",
                linewidth=1.5,
            )
        axes[1].set_xticks(np.arange(len(groups)), [name.replace("\n", " ") for name in groups], rotation=18, ha="right")
        axes[1].set_ylim(0.4, 1.0)
        axes[1].set_ylabel("dual-unseen AUC")
        axes[1].set_title("逐seed dual-unseen归因", loc="left", fontweight="bold", pad=12)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)
        basis_state = gate.get("metrics", {}).get("scgt_basis_branch_keep", "smoke阶段未裁决")
        figure.text(
            0.08,
            0.108,
            f"SCGT basis branch保留={basis_state}。黑色短线表示组内seed均值；单点控制只运行冻结seed0。",
            ha="left",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
        figure.text(
            0.08,
            0.064,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.08,
            0.020,
            "证据范围：冻结16-bit合成SPN matched标签的本地神经归因；不是PRESENT/GIFT/SKINNY高轮结果、SOTA比较或真实攻击。",
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
