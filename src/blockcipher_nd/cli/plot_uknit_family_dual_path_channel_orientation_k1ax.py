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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AX channel-orientation audit chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1ax_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ax_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    panels = _unique_panels(gate["panel_results"])
    routing = gate["routing_results"]
    harm = gate["path_harm_results"]
    cancellation = gate["cancellation_results"]
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
        figure, axes = plt.subplots(2, 2, figsize=(18.0, 11.8))
        figure.subplots_adjust(
            left=0.085,
            right=0.97,
            top=0.745,
            bottom=0.17,
            hspace=0.72,
            wspace=0.34,
        )
        figure.suptitle(
            "创新1 通道方向审计：共享结构编码器把两类语义混入了错误门控",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=17.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.923,
            (
                "冻结 K1-AW 两个最佳检查点；三种密码、两种新样本分割、每个分割1024/class；"
                "只重放路径，不训练、不更新参数。"
            ),
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.872,
            (
                f"S盒摘要只在 {routing['sbox_aligned_panels']}/12 面板更强地控制S盒路径；"
                f"线性摘要只在 {routing['linear_aligned_panels']}/12 面板更强地控制GF(2)路径；"
                "两者都未达到10/12门槛。"
            ),
            ha="left",
            fontsize=11.0,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.828,
            (
                f"排除项：GF(2)路径有害 {harm['edge_harmful_panels']}/12，"
                f"S盒路径有害 {harm['transition_harmful_panels']}/12，"
                f"严重抵消 {cancellation['cancellation_heavy_panels']}/12；"
                "主要故障是结构分量路由串线。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#0F766E",
        )
        figure.text(
            0.05,
            0.787,
            "裁决：下一版拆分结构输入，线性18维只控制边路径，S盒16维只控制转移路径。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )

        _render_routing_counts(axes[0, 0], routing)
        _render_component_deltas(
            axes[0, 1],
            panels,
            component="sbox",
            title="S盒摘要路由：副本0的uKNIT、Dialga接反",
        )
        _render_component_deltas(
            axes[1, 0],
            panels,
            component="linear",
            title="线性摘要路由：副本1的uKNIT、Midori接反",
        )
        _render_alternative_mechanisms(axes[1, 1], harm, cancellation)

        figure.text(
            0.05,
            0.054,
            (
                "下一步 K1-AY：保留K1-AW主干和检查点，先做0训练迁移/readiness；"
                "验证两套分量专属编码器、错配隔离和关闭态逐位回放，再决定是否训练。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.022,
            "当前是冻结检查点机制审计，不授权16 pairs、扩样本、远程GPU、专家或MoE。",
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
        "height_inches": 11.8,
        "language": "zh-CN",
        "panels": 4,
        "audit_panels": len(gate["panel_results"]),
        "status_from_gate": gate.get("status"),
        "formal_scale_claim_present": False,
    }


def _unique_panels(rows: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected = [row for row in rows.values() if row["split"] == "cross_key_validation"]
    cipher_order = {name: index for index, name in enumerate(CIPHER_LABELS)}
    return sorted(
        selected,
        key=lambda row: (cipher_order[str(row["cipher_key"])], int(row["replica"])),
    )


def _render_routing_counts(axis: plt.Axes, routing: Mapping[str, Any]) -> None:
    labels = ["S盒摘要→S盒门", "线性摘要→GF(2)门"]
    values = [routing["sbox_aligned_panels"], routing["linear_aligned_panels"]]
    positions = np.arange(2)
    axis.bar(positions, values, width=0.56, color=["#B45309", "#2563EB"])
    axis.axhline(10, color="#7C3AED", linestyle="--", linewidth=1.4, label="通过线 10/12")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 12.8)
    axis.set_ylabel("方向正确的面板数（共12）")
    axis.set_title("分量路由门：两类都只有8/12", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)
    for index, value in enumerate(values):
        axis.text(index, value + 0.25, f"{value}/12", ha="center", fontweight="bold")


def _render_component_deltas(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    component: str,
    title: str,
) -> None:
    labels = [f"{CIPHER_LABELS[str(row['cipher_key'])]} R{row['replica']}" for row in rows]
    y = np.arange(len(rows), dtype=float)
    height = 0.34
    edge = np.asarray([float(row[f"{component}_edge_gate_delta"]) for row in rows])
    transition = np.asarray(
        [float(row[f"{component}_transition_gate_delta"]) for row in rows]
    )
    axis.barh(y - height / 2, edge, height, label="GF(2)边门变化", color="#2563EB")
    axis.barh(y + height / 2, transition, height, label="S盒转移门变化", color="#B45309")
    axis.set_xscale("log")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("错误分量引起的门控绝对变化（对数轴）")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=8.8,
        ncol=2,
    )


def _render_alternative_mechanisms(
    axis: plt.Axes,
    harm: Mapping[str, Any],
    cancellation: Mapping[str, Any],
) -> None:
    labels = ["GF(2)路径有害", "S盒路径有害", "严重抵消"]
    values = [
        harm["edge_harmful_panels"],
        harm["transition_harmful_panels"],
        cancellation["cancellation_heavy_panels"],
    ]
    positions = np.arange(3)
    axis.bar(positions, values, width=0.58, color=["#2563EB", "#B45309", "#64748B"])
    axis.axhline(3, color="#DC2626", linestyle="--", linewidth=1.4, label="机制线 3/12")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 4.2)
    axis.set_ylabel("满足机制条件的面板数")
    axis.set_title("其他解释未成立：路径本身均有正贡献", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)
    for index, value in enumerate(values):
        axis.text(index, value + 0.10, f"{value}/12", ha="center", fontweight="bold")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ax_svg"]
