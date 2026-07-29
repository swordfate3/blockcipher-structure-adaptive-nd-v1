from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CIPHER_LABELS = {
    "uknit64": "uKNIT-BC",
    "midori64": "Midori64",
    "dialga128": "Dialga-128",
}
SHORT_LABELS = {
    "uknit64": "uKNIT",
    "midori64": "Midori",
    "dialga128": "Dialga",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BB position-preserving readiness chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    report = render_k1bb_svg(gate, rows, controls["operator_rows"], args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bb_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    operator_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    panels = _ordered_panel_minima(rows)
    operators = _ordered_operator_rows(operator_rows)
    minimum = gate["minimum_response"]
    maximum = gate["maximum_compatibility_delta"]
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
            left=0.08,
            right=0.975,
            top=0.745,
            bottom=0.17,
            hspace=0.66,
            wspace=0.30,
        )
        figure.suptitle(
            "创新1 K1-BB：真实 GF(2) 连线已能进入样本边调制",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=17.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.925,
            (
                "零训练就绪审判：冻结 K1-AZ 两个检查点与原数据；"
                "uKNIT-BC 5轮、Midori64 4轮、Dialga-128 4轮，每样本4对密文。"
            ),
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.875,
            (
                "同摘要错误拓扑现在可分：最小算子嵌入差 "
                f"{minimum['operator_embedding_delta']:.6f}，最小样本边调制差 "
                f"{minimum['edge_modulation_delta']:.6f}，最小 logit 差 "
                f"{minimum['logit_delta']:.6f}。"
            ),
            ha="left",
            fontsize=10.9,
            fontweight="bold",
            color="#047857",
        )
        figure.text(
            0.05,
            0.830,
            (
                "兼容性保持：关闭新路径时 K1-AZ 最大复现误差 "
                f"{maximum['disabled_replay_delta']:.1e}；联合重标号最大 logit 误差 "
                f"{maximum['joint_relabel_logit_delta']:.2e}。"
            ),
            ha="left",
            fontsize=10.7,
            fontweight="bold",
            color="#1D4ED8",
        )
        figure.text(
            0.05,
            0.787,
            "裁决：表示层就绪通过；这只证明拓扑信息不再被数学抹掉，还没有证明训练后 AUC 会提高。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_operator_separation(axes[0, 0], operators)
        _render_panel_metric(
            axes[0, 1],
            panels,
            field="modulation",
            threshold=1e-6,
            title="样本边调制对同摘要错误拓扑有响应",
            ylabel="正确 vs 错误算子的最大调制差",
            color="#0F766E",
        )
        _render_panel_metric(
            axes[1, 0],
            panels,
            field="logit",
            threshold=1e-6,
            title="错误拓扑的影响已传到冻结分类器输出",
            ylabel="正确 vs 错误算子的最大 logit 差",
            color="#B45309",
        )
        _render_compatibility(axes[1, 1], maximum)

        figure.text(
            0.05,
            0.058,
            (
                "下一步 K1-BC：保持2048/class、4 pair、两个replica和10 epoch不变；"
                "只训练这一位置保持算子路径，对比 K1-AZ、同摘要错误算子和跨密码错配。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.022,
            "当前是本地零训练 readiness，不是准确率结果、正式规模、攻击、任意 SPN 泛化或 SOTA 证据。",
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
        "result_panels": len(rows),
        "operator_controls": len(operator_rows),
        "status_from_gate": gate.get("status"),
        "formal_scale_claim_present": False,
    }


def _ordered_panel_minima(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["replica"]), str(row["cipher_key"])), []).append(row)
    ordered = []
    for replica in (0, 1):
        for cipher in CIPHER_LABELS:
            current = grouped[(replica, cipher)]
            ordered.append(
                {
                    "label": f"{SHORT_LABELS[cipher]} R{replica}",
                    "modulation": min(
                        float(row["correct_vs_corrupted_edge_modulation_delta"])
                        for row in current
                    ),
                    "logit": min(
                        float(row["correct_vs_corrupted_logit_delta"])
                        for row in current
                    ),
                }
            )
    return ordered


def _ordered_operator_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_key = {
        (int(row["replica"]), str(row["cipher_key"])): row for row in rows
    }
    return [
        by_key[(replica, cipher)]
        for replica in (0, 1)
        for cipher in CIPHER_LABELS
    ]


def _render_operator_separation(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = [
        f"{SHORT_LABELS[str(row['cipher_key'])]} R{int(row['replica'])}"
        for row in rows
    ]
    collision = [
        float(row["correct_vs_corrupted_embedding_max_abs_delta"]) for row in rows
    ]
    cross = [
        float(row["correct_vs_cross_cipher_embedding_max_abs_delta"]) for row in rows
    ]
    positions = np.arange(len(rows))
    width = 0.36
    axis.bar(
        positions - width / 2,
        collision,
        width,
        label="同摘要错误拓扑",
        color="#DC2626",
    )
    axis.bar(
        positions + width / 2,
        cross,
        width,
        label="跨密码拓扑",
        color="#2563EB",
    )
    axis.axhline(1e-4, color="#7C3AED", linestyle="--", linewidth=1.3, label="可分门槛 1e-4")
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel("算子嵌入最大绝对差")
    axis.set_title("18维统计会碰撞；真实边 token 不再碰撞", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False, fontsize=8.7, ncol=2)


def _render_panel_metric(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    threshold: float,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    labels = [str(row["label"]) for row in rows]
    values = [float(row[field]) for row in rows]
    positions = np.arange(len(rows))
    bars = axis.bar(positions, values, width=0.58, color=color)
    axis.axhline(
        threshold,
        color="#DC2626",
        linestyle="--",
        linewidth=1.3,
        label=f"响应门槛 {threshold:.0e}",
    )
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.13,
            f"{value:.2e}",
            ha="center",
            fontsize=8.0,
            fontweight="bold",
        )
    axis.legend(loc="lower right", frameon=False, fontsize=8.7)


def _render_compatibility(
    axis: plt.Axes,
    maximum: Mapping[str, Any],
) -> None:
    labels = ["关闭路径\n复现K1-AZ", "重标号\n算子嵌入", "重标号\n样本调制", "重标号\n最终logit"]
    raw_values = [
        float(maximum["disabled_replay_delta"]),
        float(maximum["joint_relabel_embedding_delta"]),
        float(maximum["joint_relabel_modulation_delta"]),
        float(maximum["joint_relabel_logit_delta"]),
    ]
    values = [max(value, 1e-12) for value in raw_values]
    limits = [1e-12, 1e-6, 1e-5, 1e-5]
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, width=0.58, color=["#1D4ED8", "#0F766E", "#0F766E", "#0F766E"])
    axis.scatter(positions, limits, marker="_", s=620, linewidth=2.0, color="#DC2626", label="各项容差")
    axis.set_yscale("log")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("最大绝对误差（对数刻度）")
    axis.set_title("旧模型逐位复现，位置运输保持重标号等变", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")
    axis.set_axisbelow(True)
    for bar, raw, plotted in zip(bars, raw_values, values, strict=True):
        label = "0.0" if raw == 0.0 else f"{raw:.2e}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            plotted * 1.35,
            label,
            ha="center",
            fontsize=8.6,
            fontweight="bold",
        )
    axis.legend(loc="upper left", frameon=False, fontsize=8.7)


if __name__ == "__main__":
    raise SystemExit(main())
