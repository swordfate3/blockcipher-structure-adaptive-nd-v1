from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the Chinese uKNIT r6 K1-BV result.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = render(args.gate, args.output)
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def render(gate_path: Path, output: Path) -> dict[str, Any]:
    gate: Mapping[str, Any] = json.loads(gate_path.read_text(encoding="utf-8"))
    seeds = gate.get("seed_results", {})
    if set(seeds) != {"3", "4"}:
        raise ValueError("K1-BV plot requires seed3 and seed4")
    conditions = (("exact4", "正确结构 / 4对"), ("exact16", "正确结构 / 16对"), ("wrong16", "错误S盒 / 16对"))
    values = np.asarray([[float(seeds[seed]["aucs"][name]) for seed in ("3", "4")] for name, _ in conditions])
    pair_gains = [float(seeds[seed]["pair_gain"]) for seed in ("3", "4")]
    semantic_gaps = [float(seeds[seed]["semantic_gap"]) for seed in ("3", "4")]
    with plt.rc_context({"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "font.size": 10.5, "svg.fonttype": "none", "axes.facecolor": "#FFFFFF", "text.color": "#111827"}):
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.5))
        fig.subplots_adjust(left=0.15, right=0.96, top=0.64, bottom=0.14, wspace=0.33)
        fig.suptitle("创新1 K1-BV：uKNIT 第6轮增加密文对是否带来信号", x=0.05, y=0.95, ha="left", fontsize=17, fontweight="bold")
        fig.text(0.05, 0.88, "每个条件训练 2048/class，跨密钥验证 1024/class；只比较4对与16对，并用错误S盒验证结构语义。", ha="left", color="#4B5563")
        fig.text(0.05, 0.81, _decision(str(gate.get("decision", ""))), ha="left", fontsize=11.5, fontweight="bold", color={"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(str(gate.get("status")), "#374151"))
        fig.text(0.05, 0.745, f"seed3：pair增益 {pair_gains[0]:+.4f}，结构差距 {semantic_gaps[0]:+.4f}；seed4：pair增益 {pair_gains[1]:+.4f}，结构差距 {semantic_gaps[1]:+.4f}。", ha="left", color="#4B5563")
        lo = min(0.47, float(values.min()) - 0.02)
        hi = max(0.56, float(values.max()) + 0.02)
        image = axes[0].imshow(values, cmap="RdYlGn", aspect="auto", vmin=lo, vmax=hi)
        axes[0].set_xticks((0, 1), ("seed3", "seed4"))
        axes[0].set_yticks(range(3), [label for _, label in conditions])
        axes[0].set_title("跨密钥验证 AUC（0.5约等于随机猜）", loc="left", fontweight="bold", pad=14)
        for row in range(3):
            for col in range(2):
                axes[0].text(col, row, f"{values[row, col]:.6f}", ha="center", va="center", fontsize=10.8)
        axes[0].tick_params(length=0, pad=9)
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        x = np.arange(2)
        width = 0.34
        bars1 = axes[1].bar(x - width / 2, pair_gains, width, label="16对 - 4对", color="#0F766E")
        bars2 = axes[1].bar(x + width / 2, semantic_gaps, width, label="正确S盒 - 错误S盒", color="#2563EB")
        axes[1].axhline(0.01, color="#B91C1C", linestyle="--", linewidth=1.3, label="预注册门槛 +0.010")
        axes[1].set_xticks(x, ("seed3", "seed4"))
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_title("pair放大增益与正确结构贡献", loc="left", fontweight="bold", pad=14)
        axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper right")
        all_deltas = pair_gains + semantic_gaps
        span = max(0.04, max(abs(v) for v in all_deltas) * 1.5)
        axes[1].set_ylim(min(-0.02, min(all_deltas) - span * 0.2), max(0.04, max(all_deltas) + span * 0.25))
        for bars, vals in ((bars1, pair_gains), (bars2, semantic_gaps)):
            for bar, value in zip(bars, vals, strict=True):
                axes[1].text(bar.get_x() + bar.get_width() / 2, value + span * 0.035, f"{value:+.4f}", ha="center", va="bottom", fontsize=9.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="svg")
        plt.close(fig)
    return {"status": "rendered_pending_visual_qa", "figure": str(output), "language": "zh-CN", "panels": 2, "auc_values_annotated": True, "deltas_visible": True}


def _decision(decision: str) -> str:
    return {
        "innovation1_uknit_k1bv_pair_amplification_strong": "裁决：两颗 seed 都显示强信号，允许进入 65536/class 中等规模确认。",
        "innovation1_uknit_k1bv_pair_amplification_weak": "裁决：两颗 seed 仅有弱信号，先用新 seed 复核，不扩大数据。",
        "innovation1_uknit_k1bv_pair_amplification_not_supported": "裁决：至少一颗 seed 未通过，关闭当前6轮 pair 放大路线。",
        "innovation1_uknit_k1bv_protocol_invalid": "裁决：协议或归档绑定无效，本次指标不能用于结论。",
    }.get(decision, f"裁决：{decision}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render"]
