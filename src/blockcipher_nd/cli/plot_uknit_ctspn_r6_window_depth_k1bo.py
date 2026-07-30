from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from blockcipher_nd.tasks.innovation1.uknit_ctspn_r6_window_depth_k1bo import (
    EXPECTED_SEEDS,
    FRESH_SPLITS,
)


ROUTE_TITLES = {
    "position": "保留原生 cell 位置",
    "invariant": "抹除 cell 位置",
}
SPLIT_TITLES = {
    "same_key_fresh": "同密钥新样本",
    "cross_key_validation": "跨密钥验证",
}
SERIES = (
    ("exact2_auc", "2轮正确窗口", "#64748B"),
    ("exact3_auc", "3轮正确窗口", "#0F766E"),
    ("wrong_sbox_auc", "3轮错误S盒", "#C2417B"),
    ("label_shuffled_auc", "标签打乱", "#7C3AED"),
    ("raw_auc", "原始密文", "#D97706"),
)


def render_k1bo_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    route_results = gate.get("route_results", {})
    values = [
        float(summary[name])
        for route in route_results.values()
        for seed in route.values()
        for summary in seed.values()
        for name, _, _ in SERIES
    ]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("K1-BO plot requires finite route metrics")
    lower = max(0.0, min(0.48, min(values) - 0.025))
    upper = min(1.0, max(0.60, max(values) + 0.055))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "WenQuanYi Zen Hei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), sharey=True)
    fig.patch.set_facecolor("#F8FAFC")
    for route_index, route in enumerate(("position", "invariant")):
        for split_index, split in enumerate(FRESH_SPLITS):
            axis = axes[route_index, split_index]
            axis.set_facecolor("#FFFFFF")
            width = 0.15
            x = np.arange(len(EXPECTED_SEEDS), dtype=float)
            for series_index, (name, label, color) in enumerate(SERIES):
                series_values = [
                    float(route_results[route][str(seed)][split][name])
                    for seed in EXPECTED_SEEDS
                ]
                positions = x + (series_index - 2) * width
                bars = axis.bar(
                    positions,
                    series_values,
                    width=width * 0.88,
                    label=label,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                )
                for bar, value in zip(bars, series_values, strict=True):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.004,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        rotation=90,
                        color="#1F2937",
                    )
            axis.axhline(0.5, color="#334155", linewidth=1.0, linestyle="--")
            axis.axhline(0.55, color="#DC2626", linewidth=1.0, linestyle=":")
            axis.set_ylim(lower, upper)
            axis.set_xticks(x, [f"seed {seed}" for seed in EXPECTED_SEEDS])
            axis.set_title(
                f"{ROUTE_TITLES[route]} · {SPLIT_TITLES[split]}",
                fontsize=12,
                pad=10,
                color="#0F172A",
            )
            axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.65)
            axis.set_axisbelow(True)
            if split_index == 0:
                axis.set_ylabel("AUC", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
        fontsize=10,
    )
    status = str(gate.get("status", "unknown"))
    passed = ", ".join(gate.get("passed_routes", [])) or "无"
    fig.suptitle(
        "uKNIT 6轮：最后2轮与最后3轮公开结构窗口对比",
        fontsize=18,
        fontweight="bold",
        y=0.985,
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.942,
        "同一批6轮密文、4 pair、seed3/4；仅改变公开逆算子窗口深度",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.text(
        0.5,
        0.025,
        f"裁决状态：{status}　通过路线：{passed}　虚线=随机0.50，点线=推进门槛0.55",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.075, right=0.985, hspace=0.34, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "status": "pass",
        "output": str(output),
        "figure_inches": [13.6, 8.8],
        "panels": 4,
        "series": [label for _, label, _ in SERIES],
        "y_limits": [lower, upper],
        "title_zh": "uKNIT 6轮：最后2轮与最后3轮公开结构窗口对比",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plot the uKNIT r6 K1-BO window-depth gate.")
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    print(json.dumps(render_k1bo_svg(gate, args.output), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
