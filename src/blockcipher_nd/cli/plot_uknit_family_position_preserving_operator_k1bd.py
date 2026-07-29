from __future__ import annotations

import argparse
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
CONDITION_LABELS = {
    "same_summary_corrupted_operator": "同摘要错误拓扑",
    "cross_cipher_operator": "跨密码错误拓扑",
}
GROUP_LABELS = {
    "bit_encoder": "样本比特\n编码",
    "token_encoder": "拓扑边\ntoken",
    "edge_message": "边消息\n交互",
    "bit_update": "比特状态\n更新",
    "bit_update_norm": "状态\n归一化",
    "pair_projection": "样本对\n投影",
    "structure_projection": "就绪探针\n投影",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BD gradient/coupling audit chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
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
    report = render_k1bd_svg(gate, rows, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bd_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    topology = [
        row
        for row in rows
        if row.get("metric_type") == "topology_gradient_similarity"
        and row.get("encoder_state") == "selected_encoder"
    ]
    cross = [
        row
        for row in rows
        if row.get("metric_type") == "cross_cipher_gradient_similarity"
        and row.get("encoder_state") == "selected_encoder"
    ]
    groups = [
        row
        for row in rows
        if row.get("metric_type") == "gradient_group"
        and row.get("encoder_state") == "selected_encoder"
        and row.get("condition") == "correct_operator"
        and row.get("parameter_group") in GROUP_LABELS
    ]
    interventions = [
        row for row in rows if row.get("metric_type") == "fresh_intervention"
    ]
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
            bottom=0.175,
            hspace=0.73,
            wspace=0.29,
        )
        figure.suptitle(
            "创新1 K1-BD：网络学强了通用调制，却没有学强拓扑特定调制",
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
                "零训练归因审计：复用K1-BC的2048/class/cipher、每样本4对密文；"
                "两种编码器状态、两个replica、三种拓扑条件、64个不重叠批次。"
            ),
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.875,
            (
                "排除共享优化主因：稳定冲突密码对为0，梯度最大/最小比为 "
                f"{float(gate['norm_ratios']['0']['max_to_min_median_norm_ratio']):.2f}x / "
                f"{float(gate['norm_ratios']['1']['max_to_min_median_norm_ratio']):.2f}x，"
                "均未达到4x门槛。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#1D4ED8",
        )
        figure.text(
            0.05,
            0.830,
            (
                "结构问题：就绪探针投影12672个参数始终零梯度，占声明可训练参数 "
                f"{100.0 * float(gate['disconnected_parameter_fraction']):.1f}%；"
                "训练后正确/错误拓扑差异进一步缩小。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#B91C1C",
        )
        figure.text(
            0.05,
            0.787,
            "裁决：暂停当前调制耦合；下一候选必须让真实边 token 以不可绕过的方式控制消息，而不是放大整体调制系数。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_topology_gradient(axes[0, 0], topology)
        _render_cross_cipher_gradient(axes[0, 1], cross)
        _render_group_norms(axes[1, 0], groups)
        ratios = _render_intervention_ratio(axes[1, 1], interventions)

        figure.text(
            0.05,
            0.058,
            (
                "推荐下一步 K1-BE：保持数据、4 pair、replica与冻结K1-AZ不变；"
                "只把边 token 从可被样本分支绕过的拼接输入，改为乘性控制边消息的必经门。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.022,
            "这是本地零更新机制审计，不是新训练结果、正式规模、攻击、任意SPN泛化或SOTA证据。",
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
        "topology_summary_rows": len(topology),
        "cross_cipher_summary_rows": len(cross),
        "gradient_group_rows": len(groups),
        "intervention_rows": len(interventions),
        "intervention_ratios_percent": ratios,
        "status_from_gate": gate.get("status"),
        "formal_scale_claim_present": False,
    }


def _render_topology_gradient(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    by_key = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["wrong_condition"])): row
        for row in rows
    }
    panels = [
        (replica, cipher)
        for replica in (0, 1)
        for cipher in CIPHER_LABELS
    ]
    positions = np.arange(len(panels))
    for condition, marker, color in (
        ("same_summary_corrupted_operator", "o", "#B45309"),
        ("cross_cipher_operator", "s", "#2563EB"),
    ):
        values = [
            float(by_key[(replica, cipher, condition)]["median_cosine"])
            for replica, cipher in panels
        ]
        axis.scatter(
            positions,
            values,
            marker=marker,
            s=48,
            color=color,
            label=CONDITION_LABELS[condition],
            zorder=3,
        )
    axis.axhline(
        0.99,
        color="#047857",
        linestyle="--",
        linewidth=1.4,
        label="几乎同向门槛 0.99",
    )
    axis.set_ylim(0.95, 1.002)
    axis.set_xticks(
        positions,
        [f"{CIPHER_LABELS[cipher]} R{replica}" for replica, cipher in panels],
        rotation=22,
        ha="right",
    )
    axis.set_ylabel("正确 vs 错误拓扑的梯度余弦")
    axis.set_title("replica0仍可分；replica1已接近拓扑无关", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8.2, ncol=2)


def _render_cross_cipher_gradient(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    pair_labels = {
        "uknit64__midori64": "uKNIT/Midori",
        "uknit64__dialga128": "uKNIT/Dialga",
        "midori64__dialga128": "Midori/Dialga",
    }
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["replica"]),
            list(pair_labels).index(str(row["cipher_pair"])),
        ),
    )
    positions = np.arange(len(ordered))
    values = [float(row["median_cosine"]) for row in ordered]
    bars = axis.bar(
        positions,
        values,
        width=0.58,
        color=["#DC2626" if value <= -0.05 else "#0F766E" for value in values],
    )
    axis.axhline(
        -0.05,
        color="#B45309",
        linestyle="--",
        linewidth=1.4,
        label="方向冲突门槛 -0.05",
    )
    axis.axhline(0.0, color="#9CA3AF", linewidth=1.0)
    axis.set_ylim(-0.36, 0.08)
    axis.set_xticks(
        positions,
        [
            f"{pair_labels[str(row['cipher_pair'])]}\nR{int(row['replica'])}"
            for row in ordered
        ],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("正确拓扑的跨密码梯度余弦")
    axis.set_title("局部冲突不跨两个replica稳定复现", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, row, value in zip(bars, ordered, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.018 if value < 0.0 else value + 0.012,
            f"负向{100 * float(row['negative_cosine_frequency']):.0f}%",
            ha="center",
            va="top" if value < 0.0 else "bottom",
            fontsize=7.8,
        )
    axis.legend(loc="lower left", frameon=False, fontsize=8.5)


def _render_group_norms(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    medians = {
        group: float(
            np.median(
                [
                    float(row["median_gradient_norm"])
                    for row in rows
                    if row.get("parameter_group") == group
                ]
            )
        )
        for group in GROUP_LABELS
    }
    groups = list(GROUP_LABELS)
    raw_values = [medians[group] for group in groups]
    values = [max(value, 1e-8) for value in raw_values]
    positions = np.arange(len(groups))
    colors = [
        "#DC2626" if group == "structure_projection" else "#2563EB"
        if group in {"token_encoder", "edge_message"}
        else "#0F766E"
        for group in groups
    ]
    bars = axis.bar(positions, values, width=0.58, color=colors)
    axis.set_yscale("log")
    axis.set_ylim(5e-9, 3e-2)
    axis.set_xticks(positions, [GROUP_LABELS[group] for group in groups])
    axis.set_ylabel("最终编码器的梯度范数中位数（对数）")
    axis.set_title("拓扑token梯度远弱于后端样本对投影", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, which="both")
    axis.set_axisbelow(True)
    for bar, raw, plotted in zip(bars, raw_values, values, strict=True):
        label = "0（未连接）" if raw == 0.0 else f"{raw:.1e}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            plotted * 1.45,
            label,
            ha="center",
            fontsize=7.8,
            fontweight="bold",
        )


def _render_intervention_ratio(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    ratios: dict[str, dict[str, float]] = {}
    for state in ("initial_encoder", "selected_encoder"):
        selected = [row for row in rows if row.get("encoder_state") == state]
        off = float(
            np.median(
                [float(row["correct_vs_disabled_probability_rms"]) for row in selected]
            )
        )
        ratios[state] = {
            condition: 100.0
            * float(
                np.median(
                    [
                        float(row[f"{condition}_probability_rms"])
                        for row in selected
                    ]
                )
            )
            / off
            for condition in CONDITION_LABELS
        }
    states = ["initial_encoder", "selected_encoder"]
    positions = np.arange(len(states))
    width = 0.34
    for offset, (condition, color) in zip(
        (-width / 2, width / 2),
        (
            ("same_summary_corrupted_operator", "#B45309"),
            ("cross_cipher_operator", "#2563EB"),
        ),
        strict=True,
    ):
        values = [ratios[state][condition] for state in states]
        bars = axis.bar(
            positions + offset,
            values,
            width,
            color=color,
            label=CONDITION_LABELS[condition],
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    axis.set_xticks(positions, ["训练前随机编码器", "K1-BC最终编码器"])
    axis.set_ylabel("错误拓扑影响 / 整条新路径影响（%）")
    axis.set_title("训练增强整体路径，却进一步压低拓扑占比", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False, fontsize=8.5)
    return ratios


if __name__ == "__main__":
    raise SystemExit(main())
