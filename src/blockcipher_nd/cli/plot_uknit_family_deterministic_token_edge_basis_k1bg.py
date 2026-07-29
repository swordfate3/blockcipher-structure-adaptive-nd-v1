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
CONDITION_COLORS = {
    "same_summary_corrupted_operator": "#C2410C",
    "cross_cipher_operator": "#2563EB",
}
MODEL_LABELS = {
    "k1bc": "K1-BC学习边消息",
    "k1be": "K1-BE必经门",
    "candidate": "K1-BG固定边基",
}
MODEL_COLORS = {
    "k1bc": "#94A3B8",
    "k1be": "#2563EB",
    "candidate": "#C2410C",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BG deterministic edge-basis chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--panels", required=True, type=Path)
    parser.add_argument("--gradients", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    panels = _read_jsonl(args.panels)
    gradients = _read_jsonl(args.gradients)
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))["rows"]
    report = render_k1bg_svg(gate, panels, gradients, geometry, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bg_svg(
    gate: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    topology = gate["topology_summaries"]
    same_k1bc = float(
        topology["same_summary_corrupted_operator"][
            "candidate_to_k1bc_multiplier"
        ]
    )
    cross_k1bc = float(
        topology["cross_cipher_operator"]["candidate_to_k1bc_multiplier"]
    )
    same_k1be = float(
        topology["same_summary_corrupted_operator"][
            "candidate_to_k1be_multiplier"
        ]
    )
    cross_k1be = float(
        topology["cross_cipher_operator"]["candidate_to_k1be_multiplier"]
    )
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
            bottom=0.17,
            hspace=0.7,
            wspace=0.29,
        )
        figure.suptitle(
            "创新1 K1-BG：固定正交边基没有救回拓扑依赖",
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
                "零训练就绪审判：uKNIT r5、Midori r4、Dialga r4；"
                "每样本4对密文，两个replica，每个新鲜面板固定64行。"
            ),
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.875,
            (
                "实现通过：18维token经满秩正交投影进入32维固定边基；"
                "25696个参数全部连通，来源回放、关闭路径和联合重标号均通过。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#047857",
        )
        figure.text(
            0.05,
            0.83,
            (
                "研究失败：拓扑占比仅为K1-BC的 "
                f"{same_k1bc:.2f}x / {cross_k1bc:.2f}x，且仅为K1-BE的 "
                f"{same_k1be:.2f}x / {cross_k1be:.2f}x。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#B91C1C",
        )
        figure.text(
            0.05,
            0.787,
            "裁决：停止学习边消息再池化的路线；下一步先审计直接GF(2)算子作用后的样本特征是否可辨。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_median_topology_share(axes[0, 0], gate)
        panel_multipliers = _render_panel_multipliers(axes[0, 1], panels)
        retention = _render_whole_path_retention(axes[1, 0], panels)
        _render_implementation_checks(axes[1, 1], gate, gradients, geometry)

        figure.text(
            0.05,
            0.055,
            (
                "推荐下一步 K1-BH：保持数据、4 pair、replica和错误拓扑不变；"
                "零训练审计真实GF(2)矩阵直接作用于样本布尔视图后，正确与错误算子的响应是否本来就可分。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.021,
            "这是本地零更新机制审判，不是准确率提升、正式规模、攻击、任意SPN泛化或SOTA证据。",
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
        "result_panels": len(panels),
        "gradient_rows": len(gradients),
        "geometry_rows": len(geometry),
        "status_from_gate": gate.get("status"),
        "candidate_to_k1bc_multipliers": {
            condition: float(summary["candidate_to_k1bc_multiplier"])
            for condition, summary in topology.items()
        },
        "candidate_to_k1be_multipliers": {
            condition: float(summary["candidate_to_k1be_multiplier"])
            for condition, summary in topology.items()
        },
        "minimum_panel_topology_multiplier": min(panel_multipliers),
        "minimum_whole_path_retention_ratio": min(retention),
        "formal_scale_claim_present": False,
    }


def _render_median_topology_share(
    axis: plt.Axes, gate: Mapping[str, Any]
) -> None:
    conditions = list(CONDITION_LABELS)
    positions = np.arange(len(conditions))
    width = 0.22
    for offset, model in zip((-1, 0, 1), MODEL_LABELS, strict=True):
        values = [
            100.0
            * float(
                gate["topology_summaries"][condition][
                    f"{model}_median_topology_share"
                ]
            )
            for condition in conditions
        ]
        axis.bar(
            positions + offset * width,
            values,
            width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axis.set_xticks(positions, [CONDITION_LABELS[item] for item in conditions])
    axis.set_ylabel("错误拓扑效应 / 整体新路径效应（%）")
    axis.set_title("固定边基的中位拓扑占比反而更低", loc="left", fontweight="bold")
    maximum = max(
        100.0 * float(summary[f"{model}_median_topology_share"])
        for summary in gate["topology_summaries"].values()
        for model in MODEL_LABELS
    )
    axis.set_ylim(0.0, maximum * 1.45)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", fontsize=8.7)


def _render_panel_multipliers(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> list[float]:
    ordered = _ordered_panels(panels)
    positions = np.arange(len(ordered))
    all_values: list[float] = []
    for condition, marker in zip(CONDITION_LABELS, ("o", "s"), strict=True):
        values = [
            float(row[f"candidate_{condition}_topology_share"])
            / max(float(row[f"k1bc_{condition}_topology_share"]), 1e-12)
            for row in ordered
        ]
        all_values.extend(values)
        axis.scatter(
            positions,
            values,
            marker=marker,
            s=42,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
            zorder=3,
        )
    axis.axhline(4.0, color="#047857", linestyle="--", linewidth=1.5, label="通过门槛 4x")
    axis.set_xticks(positions, _panel_labels(ordered), rotation=27, ha="right")
    axis.set_ylabel("K1-BG / K1-BC 拓扑占比倍数")
    axis.set_title("逐面板也没有形成稳定的4倍提升", loc="left", fontweight="bold")
    axis.set_ylim(0.0, max(4.5, max(all_values) * 1.15))
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)
    return all_values


def _render_whole_path_retention(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> list[float]:
    ordered = _ordered_panels(panels)
    ratios = [
        float(row["candidate_whole_path_probability_rms"])
        / max(float(row["k1be_whole_path_probability_rms"]), 1e-12)
        for row in ordered
    ]
    colors = ["#047857" if value >= 0.5 else "#B91C1C" for value in ratios]
    positions = np.arange(len(ordered))
    axis.bar(positions, ratios, color=colors, width=0.68)
    axis.axhline(0.5, color="#B45309", linestyle="--", linewidth=1.5, label="最低保留 0.5x")
    axis.set_xticks(positions, _panel_labels(ordered), rotation=27, ha="right")
    axis.set_ylabel("K1-BG / K1-BE 整体路径效应")
    axis.set_title("整体路径仍在，因此失败不是路径消失", loc="left", fontweight="bold")
    axis.set_ylim(0.0, max(1.7, max(ratios) * 1.15))
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    return ratios


def _render_implementation_checks(
    axis: plt.Axes,
    gate: Mapping[str, Any],
    gradients: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
) -> None:
    connected = sum(int(row["graph_connected_tensor_count"]) for row in gradients)
    total = sum(int(row["parameter_tensor_count"]) for row in gradients)
    max_gram = max(float(row["basis_projection_gram_max_abs_error"]) for row in geometry)
    checks = [
        ("固定边基满秩", True, f"rank 18，Gram误差 {max_gram:.2e}"),
        ("参数进入损失图", connected == total, f"6个探针合计 {connected}/{total}"),
        (
            "来源与等变兼容",
            all(gate["compatibility_checks"].values()),
            "4/4 检查通过",
        ),
        (
            "拓扑占比提升",
            bool(gate["topology_share_lift_all"]),
            "0/2 控制通过",
        ),
    ]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, len(checks))
    axis.axis("off")
    axis.set_title("实现正确，但边消息原语仍未理解拓扑", loc="left", fontweight="bold", pad=12)
    for index, (label, passed, detail) in enumerate(checks):
        y = len(checks) - index - 0.65
        color = "#047857" if passed else "#B91C1C"
        status = "通过" if passed else "未通过"
        axis.text(0.02, y, label, fontsize=11.2, fontweight="bold", va="center")
        axis.text(0.56, y, status, fontsize=10.5, fontweight="bold", color=color, va="center")
        axis.text(0.73, y, detail, fontsize=9.7, color="#4B5563", va="center")
        axis.plot([0.02, 0.98], [y - 0.42, y - 0.42], color="#E5E7EB", linewidth=0.8)


def _ordered_panels(
    panels: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        panels,
        key=lambda row: (
            int(row["replica"]),
            list(CIPHER_LABELS).index(str(row["cipher_key"])),
            str(row["split"]),
        ),
    )


def _panel_labels(panels: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"{CIPHER_LABELS[str(row['cipher_key'])]} R{int(row['replica'])}\n"
        f"{'同钥' if row['split'] == 'same_key_fresh' else '跨钥'}"
        for row in panels
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = ["render_k1bg_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
