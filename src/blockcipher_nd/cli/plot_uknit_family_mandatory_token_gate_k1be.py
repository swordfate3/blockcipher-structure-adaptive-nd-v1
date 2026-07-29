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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-BE mandatory token-gate chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--panels", required=True, type=Path)
    parser.add_argument("--gradients", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    panels = _read_jsonl(args.panels)
    gradients = _read_jsonl(args.gradients)
    report = render_k1be_svg(gate, panels, gradients, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1be_svg(
    gate: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
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
            "创新1 K1-BE：必经乘法门保住了路径，但没有增加拓扑依赖",
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
                "通过：26368个参数、22/22张量进入任务损失；关闭新路径精确回放K1-AZ；"
                "12/12面板保持重标号等变与至少50%的整体路径强度。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#047857",
        )
        topology = gate["topology_summaries"]
        same_multiplier = float(
            topology["same_summary_corrupted_operator"][
                "candidate_to_anchor_multiplier"
            ]
        )
        cross_multiplier = float(
            topology["cross_cipher_operator"]["candidate_to_anchor_multiplier"]
        )
        figure.text(
            0.05,
            0.83,
            (
                "失败：正确拓扑占比相对K1-BC初始锚点仅为 "
                f"{same_multiplier:.2f}x / {cross_multiplier:.2f}x，"
                "远低于两种控制各自4x的预注册门槛。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#B91C1C",
        )
        figure.text(
            0.05,
            0.787,
            "裁决：停止K1-BF训练；乘法门没有解决拓扑被样本路径稀释的问题，下一步改测确定性token条件边基。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_median_topology_share(axes[0, 0], gate)
        panel_multipliers = _render_panel_multipliers(axes[0, 1], panels)
        retention = _render_whole_path_retention(axes[1, 0], panels)
        _render_compatibility(axes[1, 1], gate, gradients)

        figure.text(
            0.05,
            0.055,
            (
                "推荐下一步 K1-BG：保持K1-AZ、数据、4 pair、replica与零训练协议不变；"
                "只把学习型token门改为由真实边token确定生成的边基，并继续比较两种错误拓扑。"
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
        "status_from_gate": gate.get("status"),
        "median_topology_multipliers": {
            condition: float(summary["candidate_to_anchor_multiplier"])
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
    anchor = [
        100.0
        * float(gate["topology_summaries"][condition]["anchor_median_topology_share"])
        for condition in conditions
    ]
    candidate = [
        100.0
        * float(
            gate["topology_summaries"][condition]["candidate_median_topology_share"]
        )
        for condition in conditions
    ]
    positions = np.arange(len(conditions))
    width = 0.31
    axis.bar(
        positions - width / 2,
        anchor,
        width,
        color="#94A3B8",
        label="K1-BC初始锚点",
    )
    axis.bar(
        positions + width / 2,
        candidate,
        width,
        color="#047857",
        label="K1-BE必经门",
    )
    for index, (left, right) in enumerate(zip(anchor, candidate, strict=True)):
        axis.text(index - width / 2, left + 0.025, f"{left:.3f}%", ha="center", fontsize=9)
        axis.text(index + width / 2, right + 0.025, f"{right:.3f}%", ha="center", fontsize=9)
    axis.set_xticks(positions, [CONDITION_LABELS[item] for item in conditions])
    axis.set_ylabel("错误拓扑效应 / 整体新路径效应（%）")
    axis.set_title("中位拓扑占比几乎没有变化", loc="left", fontweight="bold")
    axis.set_ylim(0.0, max(anchor + candidate) * 1.45)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")


def _render_panel_multipliers(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> list[float]:
    ordered = sorted(
        panels,
        key=lambda row: (
            int(row["replica"]),
            list(CIPHER_LABELS).index(str(row["cipher_key"])),
            str(row["split"]),
        ),
    )
    positions = np.arange(len(ordered))
    all_values: list[float] = []
    for condition, marker in zip(CONDITION_LABELS, ("o", "s"), strict=True):
        values = [
            float(row[f"candidate_{condition}_topology_share"])
            / max(float(row[f"anchor_{condition}_topology_share"]), 1e-12)
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
    labels = [
        f"{CIPHER_LABELS[str(row['cipher_key'])]} R{int(row['replica'])}\n"
        f"{'同钥' if row['split'] == 'same_key_fresh' else '跨钥'}"
        for row in ordered
    ]
    axis.set_xticks(positions, labels, rotation=27, ha="right")
    axis.set_ylabel("K1-BE / K1-BC 拓扑占比倍数")
    axis.set_title("12个面板没有形成稳定的4倍提升", loc="left", fontweight="bold")
    axis.set_ylim(0.0, max(4.5, max(all_values) * 1.15))
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)
    return all_values


def _render_whole_path_retention(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> list[float]:
    ordered = sorted(
        panels,
        key=lambda row: (
            int(row["replica"]),
            list(CIPHER_LABELS).index(str(row["cipher_key"])),
            str(row["split"]),
        ),
    )
    ratios = [
        float(row["candidate_whole_path_probability_rms"])
        / max(float(row["anchor_whole_path_probability_rms"]), 1e-12)
        for row in ordered
    ]
    colors = ["#047857" if value >= 0.5 else "#B91C1C" for value in ratios]
    positions = np.arange(len(ordered))
    axis.bar(positions, ratios, color=colors, width=0.68)
    axis.axhline(0.5, color="#B45309", linestyle="--", linewidth=1.5, label="最低保留 0.5x")
    labels = [
        f"{CIPHER_LABELS[str(row['cipher_key'])]} R{int(row['replica'])}\n"
        f"{'同钥' if row['split'] == 'same_key_fresh' else '跨钥'}"
        for row in ordered
    ]
    axis.set_xticks(positions, labels, rotation=27, ha="right")
    axis.set_ylabel("K1-BE / K1-BC 整体路径效应")
    axis.set_title("整体新路径强度在12/12面板通过", loc="left", fontweight="bold")
    axis.set_ylim(0.0, max(1.7, max(ratios) * 1.15))
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper left")
    return ratios


def _render_compatibility(
    axis: plt.Axes,
    gate: Mapping[str, Any],
    gradients: Sequence[Mapping[str, Any]],
) -> None:
    connected = sum(
        int(row["graph_connected_tensor_count"]) for row in gradients
    )
    total = sum(int(row["parameter_tensor_count"]) for row in gradients)
    checks = [
        (
            "参数进入损失图",
            connected == total,
            f"6个探针合计 {connected}/{total}",
        ),
        (
            "关闭路径回放",
            bool(gate["compatibility_checks"]["disabled_path_exactly_replays_k1az"]),
            "最大差值 = 0",
        ),
        (
            "联合重标号等变",
            bool(gate["compatibility_checks"]["joint_relabel_is_equivariant"]),
            "12/12 面板",
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
    axis.set_title("实现正确，但研究假设未通过", loc="left", fontweight="bold", pad=12)
    for index, (label, passed, detail) in enumerate(checks):
        y = len(checks) - index - 0.65
        color = "#047857" if passed else "#B91C1C"
        status = "通过" if passed else "未通过"
        axis.text(0.02, y, label, fontsize=11.2, fontweight="bold", va="center")
        axis.text(0.56, y, status, fontsize=10.5, fontweight="bold", color=color, va="center")
        axis.text(0.73, y, detail, fontsize=10.0, color="#4B5563", va="center")
        axis.plot([0.02, 0.98], [y - 0.42, y - 0.42], color="#E5E7EB", linewidth=0.8)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = ["render_k1be_svg"]


if __name__ == "__main__":
    raise SystemExit(main())
