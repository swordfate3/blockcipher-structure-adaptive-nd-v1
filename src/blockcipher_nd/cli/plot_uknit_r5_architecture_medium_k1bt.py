from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ROWS = (("uknit_structure_expert", "uKNIT 结构专家"), ("autond_dbitnet", "AutoND DBitNet"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Chinese K1-BT medium architecture result.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1bt_svg(gate, args.output)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def render_k1bt_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"3", "4"}:
        raise ValueError("K1-BT plot requires seed3 and seed4")
    values = np.asarray([
        [float(seed_results[seed]["auc_by_architecture"][name]) for seed in ("3", "4")]
        for name, _ in ROWS
    ])
    margins = [float(seed_results[seed]["expert_minus_autond"]) for seed in ("3", "4")]
    with plt.rc_context({
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "font.size": 10.8,
        "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#CBD5E1",
        "text.color": "#111827", "svg.fonttype": "none",
    }):
        figure, axes = plt.subplots(1, 2, figsize=(15.5, 8.5))
        figure.subplots_adjust(left=0.15, right=0.96, top=0.65, bottom=0.14, wspace=0.34)
        figure.suptitle("创新1 K1-BT：uKNIT 第5轮中等规模网络对比", x=0.05, y=0.95, ha="left", fontsize=17, fontweight="bold")
        figure.text(0.05, 0.88, "每个模型训练 65536/class，跨密钥验证 16384/class；每条样本含16对密文，只替换神经网络。", ha="left", color="#4B5563")
        figure.text(0.05, 0.81, _decision(gate), ha="left", fontsize=11.5, fontweight="bold", color={"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(str(gate.get("status")), "#374151"))
        figure.text(0.05, 0.745, f"seed3 专家优势 {margins[0]:+.4f}；seed4 专家优势 {margins[1]:+.4f}；门槛均为专家 AUC≥0.550 且优势≥+0.010。", ha="left", color="#4B5563")

        lower = min(0.45, float(values.min()) - 0.03)
        upper = max(0.60, float(values.max()) + 0.03)
        image = axes[0].imshow(values, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=upper)
        axes[0].set_xticks((0, 1), ("seed3", "seed4"))
        axes[0].set_yticks((0, 1), [label for _, label in ROWS])
        axes[0].set_title("跨密钥验证 AUC（越高越好）", loc="left", fontweight="bold", pad=14)
        for row in range(2):
            for column in range(2):
                axes[0].text(column, row, f"{values[row, column]:.6f}", ha="center", va="center", fontsize=11)
        axes[0].tick_params(length=0, pad=9)
        figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

        colors = ["#0F766E" if margin >= 0.01 else "#B45309" for margin in margins]
        bars = axes[1].bar((0, 1), margins, color=colors, width=0.55)
        axes[1].axhline(0.01, color="#B91C1C", linestyle="--", linewidth=1.4, label="预注册优势门槛 +0.010")
        axes[1].set_xticks((0, 1), ("seed3", "seed4"))
        axes[1].set_ylabel("专家 AUC - AutoND AUC")
        axes[1].set_title("结构专家相对通用基线的优势", loc="left", fontweight="bold", pad=14)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper right")
        span = max(0.04, max(abs(value) for value in margins) * 1.35)
        axes[1].set_ylim(min(-0.02, min(margins) - span * 0.15), max(0.04, max(margins) + span * 0.2))
        for bar, margin in zip(bars, margins, strict=True):
            axes[1].text(bar.get_x() + bar.get_width() / 2, margin + span * 0.04, f"{margin:+.6f}", ha="center", va="bottom", fontweight="bold")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {"status": "rendered_pending_visual_qa", "figure": str(output), "language": "zh-CN", "panels": 2, "auc_values_annotated": True, "margins_visible": True}


def _decision(gate: Mapping[str, Any]) -> str:
    return {
        "innovation1_uknit_k1bt_medium_structure_expert_supported": "裁决：两颗 seed 均通过，允许进入 262144/class 远程确认。",
        "innovation1_uknit_k1bt_medium_structure_expert_not_supported": "裁决：至少一颗 seed 未通过，停止机械放大并核对训练动态。",
        "innovation1_uknit_k1bt_medium_protocol_invalid": "裁决：计划、缓存、检查点或结果绑定无效，本次指标不可解释。",
    }.get(str(gate.get("decision")), f"裁决：{gate.get('decision', '')}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_k1bt_svg"]
