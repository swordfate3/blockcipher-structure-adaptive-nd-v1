from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CIPHER_LABELS = {
    "uknit64": "uKNIT",
    "midori64": "Midori",
    "dialga128": "Dialga",
}
SPLIT_LABELS = {
    "same_key_fresh": "同密钥",
    "cross_key_validation": "跨密钥",
}
CONDITION_LABELS = {
    "same_summary_corrupted_operator": "同摘要错误拓扑",
    "cross_cipher_operator": "跨密码错误拓扑",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BC training and attribution chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    with args.history.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    controls = [
        json.loads(line)
        for line in args.controls.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = render_k1bc_svg(gate, history, controls, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bc_svg(
    gate: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    panels = _ordered_panels(controls)
    topology_threshold = 0.001
    no_harm_threshold = -0.005
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(18.0, 12.0))
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.745,
            bottom=0.18,
            hspace=0.72,
            wspace=0.28,
        )
        figure.suptitle(
            "创新1 K1-BC：位置保持算子能表示，但训练后没有使用正确拓扑",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=17.2,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.925,
            (
                "本地机制诊断：uKNIT-BC 5轮、Midori64 4轮、Dialga-128 4轮；"
                "每密码2048/class、每样本4对密文、2个replica、10轮训练。"
            ),
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        macros = gate["macro_results"]
        figure.text(
            0.05,
            0.875,
            (
                "相对冻结 K1-AZ：replica0 "
                f"{float(macros['replica0']['improvement']):+.6f}，replica1 "
                f"{float(macros['replica1']['improvement']):+.6f}；两个副本都未保持基线。"
            ),
            ha="left",
            fontsize=10.9,
            fontweight="bold",
            color="#B91C1C",
        )
        topology = gate["topology_results"]
        figure.text(
            0.05,
            0.830,
            (
                "正确拓扑归因失败：同摘要错误拓扑 "
                f"{int(topology['same_summary_corrupted_operator']['passing_panels'])}/12，"
                "跨密码错误拓扑 "
                f"{int(topology['cross_cipher_operator']['passing_panels'])}/12；要求各至少10/12。"
            ),
            ha="left",
            fontsize=10.9,
            fontweight="bold",
            color="#B91C1C",
        )
        figure.text(
            0.05,
            0.787,
            "裁决：保持 K1-BB 表示实现，暂停 K1-BC 训练路线；当前不能扩大 pair、数据、轮数、宽度或上远程 GPU。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_training_curves(axes[0, 0], history)
        _render_macro_retention(axes[0, 1], macros)
        _render_topology_attribution(
            axes[1, 0], panels, topology_threshold=topology_threshold
        )
        _render_no_harm(axes[1, 1], panels, threshold=no_harm_threshold)

        figure.text(
            0.05,
            0.060,
            (
                "推荐下一步：冻结两个 K1-BC 检查点，审计新编码器的梯度、通道调制幅度及三密码梯度方向，"
                "区分“优化没学到”与“共享训练互相抵消”。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.023,
            "这是2048/class/cipher的本地4-pair机制诊断，不是正式训练、攻击、任意SPN泛化或SOTA证据。",
            ha="left",
            fontsize=9.8,
            color="#4B5563",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 18.0,
        "height_inches": 12.0,
        "language": "zh-CN",
        "panels": 4,
        "evaluation_panels": len(panels),
        "control_rows": len(controls),
        "status_from_gate": gate.get("status"),
        "formal_scale_claim_present": False,
    }


def _ordered_panels(
    controls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in controls:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        by_key.setdefault(key, {})[str(row["condition"])] = row
    ordered: list[dict[str, Any]] = []
    for replica in (0, 1):
        for cipher in CIPHER_LABELS:
            for split in ("same_key_fresh", "cross_key_validation"):
                conditions = by_key[(replica, cipher, split)]
                correct = float(conditions["correct_operator"]["auc"])
                ordered.append(
                    {
                        "label": (
                            f"{CIPHER_LABELS[cipher]} R{replica}\n{SPLIT_LABELS[split]}"
                        ),
                        "correct_minus_k1az": correct
                        - float(conditions["disabled_k1az"]["auc"]),
                        "same_summary_corrupted_operator": correct
                        - float(conditions["same_summary_corrupted_operator"]["auc"]),
                        "cross_cipher_operator": correct
                        - float(conditions["cross_cipher_operator"]["auc"]),
                    }
                )
    return ordered


def _render_training_curves(
    axis: plt.Axes, history: Sequence[Mapping[str, Any]]
) -> None:
    colors = {0: "#2563EB", 1: "#DC2626"}
    for replica in (0, 1):
        rows = [row for row in history if int(row["replica"]) == replica]
        epochs = [int(row["epoch"]) for row in rows]
        macro = [float(row["cross_key_macro_auc"]) for row in rows]
        minimum = [float(row["cross_key_minimum_auc"]) for row in rows]
        axis.plot(
            epochs,
            macro,
            marker="o",
            markersize=3.8,
            linewidth=1.8,
            color=colors[replica],
            label=f"R{replica} 三密码平均",
        )
        axis.plot(
            epochs,
            minimum,
            linestyle="--",
            linewidth=1.5,
            color=colors[replica],
            alpha=0.72,
            label=f"R{replica} 最弱密码",
        )
    axis.set_xticks(range(1, 11))
    axis.set_xlabel("训练轮次")
    axis.set_ylabel("跨密钥 AUC")
    axis.set_title("训练稳定，但新增路径没有带来基线增益", loc="left", fontweight="bold")
    axis.grid(color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="center right", frameon=False, fontsize=8.5, ncol=2)


def _render_macro_retention(
    axis: plt.Axes, macros: Mapping[str, Mapping[str, Any]]
) -> None:
    labels = ["replica0", "replica1"]
    deltas = [float(macros[label]["improvement"]) * 1000.0 for label in labels]
    positions = np.arange(len(labels))
    bars = axis.bar(positions, deltas, width=0.52, color="#DC2626")
    axis.axhline(0.0, color="#047857", linestyle="--", linewidth=1.5, label="通过线：不低于 K1-AZ")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("候选 - K1-AZ（AUC × 1000）")
    axis.set_title("两个副本都没有保持同预算基线", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, deltas, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.12,
            f"{value:+.3f}",
            ha="center",
            va="top",
            fontsize=9.2,
            fontweight="bold",
        )
    axis.legend(loc="lower right", frameon=False, fontsize=8.7)


def _render_topology_attribution(
    axis: plt.Axes,
    panels: Sequence[Mapping[str, Any]],
    *,
    topology_threshold: float,
) -> None:
    positions = np.arange(len(panels))
    for condition, marker, color in (
        ("same_summary_corrupted_operator", "o", "#B45309"),
        ("cross_cipher_operator", "s", "#2563EB"),
    ):
        ratios = [
            float(panel[condition]) / topology_threshold for panel in panels
        ]
        axis.scatter(
            positions,
            ratios,
            marker=marker,
            s=34,
            color=color,
            label=CONDITION_LABELS[condition],
            zorder=3,
        )
    axis.axhline(1.0, color="#047857", linestyle="--", linewidth=1.5, label="通过线：正确拓扑领先0.001")
    axis.axhline(0.0, color="#9CA3AF", linewidth=1.0)
    axis.set_ylim(-0.12, 1.12)
    axis.set_xticks(positions, [str(panel["label"]) for panel in panels], rotation=25, ha="right")
    axis.set_ylabel("实际领先量 / 要求领先量")
    axis.set_title("12个面板都远未形成正确拓扑归因", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False, fontsize=8.2, ncol=2)


def _render_no_harm(
    axis: plt.Axes,
    panels: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> None:
    positions = np.arange(len(panels))
    values = [float(panel["correct_minus_k1az"]) * 100.0 for panel in panels]
    colors = ["#DC2626" if value < threshold * 100.0 else "#0F766E" for value in values]
    bars = axis.bar(positions, values, width=0.58, color=colors)
    axis.axhline(
        threshold * 100.0,
        color="#B45309",
        linestyle="--",
        linewidth=1.5,
        label="允许下限：-0.5个百分点",
    )
    axis.axhline(0.0, color="#9CA3AF", linewidth=1.0)
    axis.set_xticks(positions, [str(panel["label"]) for panel in panels], rotation=25, ha="right")
    axis.set_ylabel("候选 - K1-AZ（AUC 百分点）")
    axis.set_title("uKNIT replica0 同密钥面板超过伤害下限", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        if value < threshold * 100.0:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.025,
                f"{value:+.3f}",
                ha="center",
                va="top",
                fontsize=8.0,
                fontweight="bold",
                color="#991B1B",
            )
    axis.legend(loc="lower right", frameon=False, fontsize=8.7)


if __name__ == "__main__":
    raise SystemExit(main())
