from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ROWS = (
    ("correct", "正确 Dialga 拓扑"),
    ("corrupted", "扰动拓扑"),
    ("autond", "通用 AutoND"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese Dialga DMC result.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_dmc1_svg(gate, args.output)
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_dmc1_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"0", "1"}:
        raise ValueError("DMC1 plot requires seed0 and seed1")
    values = np.asarray(
        [
            [
                float(seed_results[seed]["auc_by_architecture"][name])
                for seed in ("0", "1")
            ]
            for name, _ in ROWS
        ]
    )
    topology_margins = [
        float(seed_results[seed]["correct_minus_corrupted"])
        for seed in ("0", "1")
    ]
    autond_margins = [
        float(seed_results[seed]["correct_minus_autond"])
        for seed in ("0", "1")
    ]
    title, protocol = _figure_text(gate)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.5, 8.5))
        figure.subplots_adjust(
            left=0.16, right=0.96, top=0.65, bottom=0.14, wspace=0.35
        )
        figure.suptitle(
            title,
            x=0.05,
            y=0.95,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.88,
            protocol,
            ha="left",
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.81,
            _decision(gate),
            ha="left",
            fontsize=11.5,
            fontweight="bold",
            color={"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(
                str(gate.get("status")), "#374151"
            ),
        )
        figure.text(
            0.05,
            0.745,
            "门槛：正确拓扑 AUC≥0.900，且分别领先扰动拓扑 +0.005、AutoND +0.010。",
            ha="left",
            color="#4B5563",
        )

        lower = min(0.45, float(values.min()) - 0.03)
        upper = max(0.95, float(values.max()) + 0.03)
        image = axes[0].imshow(values, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=upper)
        axes[0].set_xticks((0, 1), ("seed0", "seed1"))
        axes[0].set_yticks(range(3), [label for _, label in ROWS])
        axes[0].set_title("跨密钥验证 AUC（越高越好）", loc="left", fontweight="bold", pad=14)
        for row in range(3):
            for column in range(2):
                axes[0].text(column, row, f"{values[row, column]:.6f}", ha="center", va="center", fontsize=10.5)
        axes[0].tick_params(length=0, pad=9)
        figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

        x = np.arange(2)
        width = 0.32
        bars_topology = axes[1].bar(x - width / 2, topology_margins, width, label="正确 - 扰动拓扑", color="#0F766E")
        bars_autond = axes[1].bar(x + width / 2, autond_margins, width, label="正确 - AutoND", color="#2563EB")
        axes[1].axhline(0.005, color="#B45309", linestyle="--", linewidth=1.2, label="拓扑门槛 +0.005")
        axes[1].axhline(0.010, color="#B91C1C", linestyle=":", linewidth=1.4, label="AutoND 门槛 +0.010")
        axes[1].set_xticks(x, ("seed0", "seed1"))
        axes[1].set_ylabel("正确拓扑 AUC 优势")
        axes[1].set_title("正确拓扑相对两类控制的优势", loc="left", fontweight="bold", pad=14)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper right", fontsize=9)
        all_margins = topology_margins + autond_margins
        span = max(0.04, max(abs(value) for value in all_margins) * 1.35)
        axes[1].set_ylim(min(-0.02, min(all_margins) - span * 0.15), max(0.04, max(all_margins) + span * 0.4))
        for bars, margins in ((bars_topology, topology_margins), (bars_autond, autond_margins)):
            for bar, margin in zip(bars, margins, strict=True):
                axes[1].text(bar.get_x() + bar.get_width() / 2, margin + span * 0.035, f"{margin:+.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "language": "zh-CN",
        "panels": 2,
        "auc_values_annotated": True,
        "both_control_margins_visible": True,
    }


def _figure_text(gate: Mapping[str, Any]) -> tuple[str, str]:
    if "dmc2" in str(gate.get("run_id", "")).lower():
        return (
            "创新1 DMC2：Dialga 第4轮异构拓扑扩样确认",
            "每个模型训练 262144/class，跨密钥验证 65536/class；每条样本含4对密文。",
        )
    return (
        "创新1 DMC1：Dialga 第4轮异构拓扑中等规模验证",
        "每个模型训练 65536/class，跨密钥验证 16384/class；每条样本含4对密文。",
    )


def _decision(gate: Mapping[str, Any]) -> str:
    return {
        "innovation1_dialga_dmc1_medium_topology_supported": "裁决：两颗 seed 全部门槛通过，允许进入 262144/class。",
        "innovation1_dialga_dmc1_medium_topology_not_supported": "裁决：至少一项门槛未通过，停止机械放大并检查训练动态。",
        "innovation1_dialga_dmc1_medium_protocol_invalid": "裁决：计划、缓存、检查点或结果绑定无效，本次指标不可解释。",
        "innovation1_dialga_dmc2_scale_topology_supported": "裁决：两颗 seed 全部门槛通过，允许预注册正式规模 DFC1。",
        "innovation1_dialga_dmc2_scale_topology_not_supported": "裁决：至少一项门槛未通过，停止机械放大并检查训练动态。",
        "innovation1_dialga_dmc2_scale_protocol_invalid": "裁决：计划、缓存、检查点或结果绑定无效，本次指标不可解释。",
    }.get(str(gate.get("decision")), f"裁决：{gate.get('decision', '')}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_dmc1_svg"]
