from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURE_ROWS = (
    ("uknit_structure_expert", "uKNIT 结构专家"),
    ("autond_dbitnet", "AutoND DBitNet"),
    ("generic_spn_cell_pairset", "通用 SPN Cell-PairSet"),
    ("generic_spn_token_mixer", "通用 SPN Token Mixer"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese uKNIT K1-BS architecture comparison."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    report = render_k1bs_svg(gate, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1bs_svg(gate: Mapping[str, Any], output: Path) -> dict[str, Any]:
    seed_results = gate.get("seed_results", {})
    if set(seed_results) != {"3", "4"}:
        raise ValueError("K1-BS plot requires seed3 and seed4 results")
    columns = ("3", "4")
    auc_values = np.asarray(
        [
            [
                float(seed_results[seed]["auc_by_architecture"][architecture])
                for seed in columns
            ]
            for architecture, _ in ARCHITECTURE_ROWS
        ],
        dtype=float,
    )
    parameter_counts = gate.get("parameter_counts", {})
    parameters_million = np.asarray(
        [float(parameter_counts[architecture]) / 1_000_000 for architecture, _ in ARCHITECTURE_ROWS]
    )

    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.6,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16.0, 8.8))
        figure.subplots_adjust(
            left=0.19,
            right=0.96,
            top=0.67,
            bottom=0.13,
            wspace=0.38,
        )
        figure.suptitle(
            "创新1 K1-BS：uKNIT 第5轮神经网络横向对比",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.89,
            "固定 cell11 差分、16对密文、2048/class、跨密钥验证、两颗 seed 和10个 epoch；只替换神经网络。",
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.82,
            _decision_text(gate),
            ha="left",
            fontsize=11.3,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.05,
            0.755,
            _margin_text(seed_results),
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )

        _auc_heatmap(axes[0], auc_values)
        _parameter_bars(axes[1], parameters_million)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.0,
        "height_inches": 8.8,
        "language": "zh-CN",
        "panels": 2,
        "title_explains_cipher_round_and_comparison": True,
        "auc_values_annotated": True,
        "capacity_difference_visible": True,
    }


def _auc_heatmap(axis: plt.Axes, values: np.ndarray) -> None:
    lower = min(0.45, float(values.min()) - 0.03)
    upper = max(0.80, float(values.max()) + 0.03)
    image = axis.imshow(values, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=upper)
    axis.set_xticks((0, 1), ("seed3", "seed4"))
    axis.set_yticks(
        range(len(ARCHITECTURE_ROWS)),
        [label for _, label in ARCHITECTURE_ROWS],
    )
    axis.set_title("跨密钥验证 AUC（越高越好）", loc="left", fontweight="bold", pad=14)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{float(values[row, column]):.6f}",
                ha="center",
                va="center",
                fontsize=10.8,
                color="#111827",
            )
    axis.tick_params(length=0, axis="both", pad=9)
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _parameter_bars(axis: plt.Axes, values: np.ndarray) -> None:
    positions = np.arange(len(ARCHITECTURE_ROWS))
    colors = ("#0F766E", "#64748B", "#2563EB", "#B45309")
    bars = axis.barh(positions, values, color=colors, height=0.58)
    axis.set_yticks(positions, [label for _, label in ARCHITECTURE_ROWS])
    axis.invert_yaxis()
    axis.set_xlabel("可训练参数（百万）")
    axis.set_title("模型容量（本次并非参数量匹配实验）", loc="left", fontweight="bold", pad=14)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=9)
    right = max(float(values.max()) * 1.20, 0.2)
    axis.set_xlim(0.0, right)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            float(value) + right * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.3f}M",
            va="center",
            fontsize=10.5,
            color="#111827",
        )


def _margin_text(seed_results: Mapping[str, Any]) -> str:
    parts = []
    for seed in ("3", "4"):
        values = seed_results[seed]
        parts.append(
            f"seed{seed}：专家 - 最强通用网络 = "
            f"{float(values['expert_minus_best_generic']):+.4f}"
        )
    return "；".join(parts) + "；预注册优势门槛为每颗 seed 均不低于 +0.010。"


def _decision_text(gate: Mapping[str, Any]) -> str:
    labels = {
        "innovation1_uknit_k1bs_structure_expert_retained": (
            "裁决：当前 uKNIT 结构专家在两颗 seed 上都明显优于最强通用网络。"
        ),
        "innovation1_uknit_k1bs_structure_expert_not_necessary": (
            "裁决：专家信号存在，但没有在两颗 seed 上都拉开预注册优势。"
        ),
        "innovation1_uknit_k1bs_expert_signal_not_reproduced": (
            "裁决：当前专家没有复现预期信号，需先核对训练与 K1-V 是否一致。"
        ),
        "innovation1_uknit_k1bs_architecture_protocol_invalid": (
            "裁决：计划、缓存、输入形状或训练产物不完整，本次指标不可解释。"
        ),
    }
    decision = str(gate.get("decision", ""))
    return labels.get(decision, f"裁决：{decision}")


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "hold": "#B45309", "invalid": "#B91C1C"}.get(
        status, "#374151"
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1bs_svg"]
