#!/usr/bin/env python3
"""Render the uKNIT K1-CA/K1-CB five-model paper comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "chinese-core-innovation1"
K1CA_RESULTS = (
    REPO_ROOT
    / "outputs"
    / "remote_results_incomplete"
    / "i1_uknit_r5_k1ca_invariant_autond_262144_s3s4_20260803"
    / "results.jsonl"
)
K1CB_RESULTS = (
    REPO_ROOT
    / "outputs"
    / "remote_results_incomplete"
    / "i1_uknit_r5_k1cb_published_comparison_262144_s3s4_20260803"
    / "results.jsonl"
)

MODEL_ORDER = [
    "runtime_spn_ct_k1t_position_histogram_invariant",
    "autond_dbitnet2023",
    "spn_zhang_wang_mcnd_adapter",
    "spn_liu_case3_conv2d_adapter",
    "spn_gohr_style_resnet_pairset_adapter",
]
MODEL_LABELS = [
    "本文位置不变结构专家",
    "AutoND/DBitNet 适配",
    "Zhang/Wang MCND 适配",
    "Liu Case-3 Conv2D 适配",
    "Gohr-style ResNet 适配",
]
BASELINE_LABELS = [
    "AutoND/DBitNet",
    "Zhang/Wang MCND",
    "Liu Case-3 Conv2D",
    "Gohr-style ResNet",
]
SEEDS = [3, 4]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def collect_auc() -> np.ndarray:
    rows = load_rows(K1CA_RESULTS) + load_rows(K1CB_RESULTS)
    by_key = {(row["model"], int(row["seed"])): row for row in rows}
    expected = {(model, seed) for model in MODEL_ORDER for seed in SEEDS}
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        unexpected = sorted(set(by_key) - expected)
        raise ValueError(f"result matrix mismatch: missing={missing}, unexpected={unexpected}")

    for row in rows:
        if row["samples_per_class"] != 262144:
            raise ValueError("all paper rows must use 262144 samples/class")
        if row["pairs_per_sample"] != 4 or row["target_epochs"] != 10:
            raise ValueError("all paper rows must use 4 pairs and 10 epochs")
        if row["negative_mode"] != "encrypted_random_plaintexts":
            raise ValueError("strict negative protocol mismatch")

    return np.asarray(
        [[by_key[(model, seed)]["metrics"]["auc"] for seed in SEEDS] for model in MODEL_ORDER],
        dtype=float,
    )


def render() -> None:
    auc = collect_auc()
    margins = auc[0:1, :] - auc[1:, :]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.edgecolor": "#94a3b8",
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )

    figure = plt.figure(figsize=(12.8, 6.8), facecolor="white")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.15],
        left=0.205,
        right=0.975,
        top=0.78,
        bottom=0.17,
        wspace=0.34,
    )
    ax_auc = figure.add_subplot(grid[0, 0])
    ax_margin = figure.add_subplot(grid[0, 1])

    heatmap = ax_auc.imshow(auc, cmap="RdYlGn", vmin=0.48, vmax=1.0, aspect="auto")
    ax_auc.set_xticks(range(2), ["seed 3", "seed 4"], fontsize=10.5)
    ax_auc.set_yticks(range(5), MODEL_LABELS, fontsize=9.8)
    ax_auc.set_title("跨密钥验证 AUC", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax_auc.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax_auc.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax_auc.grid(which="minor", color="white", linewidth=1.5)
    ax_auc.tick_params(which="minor", bottom=False, left=False)
    ax_auc.tick_params(axis="both", length=0)

    for row_index in range(auc.shape[0]):
        for seed_index in range(auc.shape[1]):
            value = auc[row_index, seed_index]
            color = "white" if value >= 0.86 else "#111827"
            ax_auc.text(
                seed_index,
                row_index,
                f"{value:.6f}",
                ha="center",
                va="center",
                fontsize=10.4,
                fontweight="bold" if row_index == 0 else "normal",
                color=color,
            )

    colorbar = figure.colorbar(heatmap, ax=ax_auc, fraction=0.045, pad=0.04)
    colorbar.set_ticks([0.5, 0.75, 1.0])
    colorbar.ax.tick_params(labelsize=9, length=3)
    colorbar.outline.set_edgecolor("#cbd5e1")

    positions = np.arange(len(BASELINE_LABELS))
    bar_height = 0.32
    seed_colors = ["#0f766e", "#c2410c"]
    ax_margin.barh(
        positions - bar_height / 2,
        margins[:, 0],
        height=bar_height,
        color=seed_colors[0],
        label="seed 3",
    )
    ax_margin.barh(
        positions + bar_height / 2,
        margins[:, 1],
        height=bar_height,
        color=seed_colors[1],
        label="seed 4",
    )
    ax_margin.set_yticks(positions, BASELINE_LABELS, fontsize=9.8)
    ax_margin.invert_yaxis()
    ax_margin.set_xlim(0.0, 0.5)
    ax_margin.set_xticks(np.arange(0.0, 0.51, 0.1))
    ax_margin.set_xlabel("本文方法相对基线的 AUC 差值", fontsize=10.5, labelpad=8)
    ax_margin.set_title("逐 seed 优势", loc="left", fontsize=12, fontweight="bold", pad=12)
    ax_margin.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax_margin.set_axisbelow(True)
    ax_margin.spines[["top", "right"]].set_visible(False)
    ax_margin.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.105),
        frameon=False,
        fontsize=9.8,
        ncol=2,
        handlelength=1.4,
        columnspacing=1.2,
    )

    for seed_index, offset in enumerate([-bar_height / 2, bar_height / 2]):
        for row_index, value in enumerate(margins[:, seed_index]):
            ax_margin.text(
                value - 0.008,
                row_index + offset,
                f"{value:.6f}",
                ha="right",
                va="center",
                fontsize=9.2,
                color="white",
                fontweight="bold",
            )

    figure.suptitle(
        "uKNIT-BC r5：冻结项目协议下的五模型主规模比较",
        x=0.205,
        y=0.94,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.205,
        0.865,
        "262144/class 训练 · 65536/class 跨密钥验证 · 4 pairs/sample · 10 epochs",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#475569",
    )
    figure.text(
        0.205,
        0.065,
        "注：比较为统一 uKNIT 项目协议下的公开架构适配，不是原论文精确复现或充分超参数搜索。",
        ha="left",
        va="center",
        fontsize=9.6,
        color="#475569",
    )

    svg_path = PAPER_ROOT / "figures" / "fig_uknit_k1cb_main_comparison.svg"
    pdf_path = PAPER_ROOT / "figures_pdf" / "fig_uknit_k1cb_main_comparison.pdf"
    figure.savefig(svg_path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    figure.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    print(svg_path)
    print(pdf_path)


if __name__ == "__main__":
    render()
