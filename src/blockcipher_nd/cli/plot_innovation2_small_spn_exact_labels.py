from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render E32 small-SPN exact-label width and shortcut audit."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_small_spn_exact_label_svg(summary, args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))
    return 0


def render_small_spn_exact_label_svg(
    summary: dict[str, Any], output: Path
) -> None:
    gate = summary["gate"]
    metrics = gate["metrics"]
    split_metrics = summary["split_metrics"]
    baselines = summary["marginal_baselines"]
    split_names = ["train", "unseen_sbox", "unseen_player", "dual_unseen"]
    split_labels = ["训练拓扑块", "未见S-box", "未见P-layer", "双轴未见"]
    positive = [split_metrics[name]["positive"] for name in split_names]
    negative = [split_metrics[name]["negative"] for name in split_names]
    heldout_names = ["unseen_sbox", "unseen_player", "dual_unseen"]
    heldout_labels = ["未见S-box", "未见P-layer", "双轴未见"]
    predictor_names = [
        "global",
        "mask_only",
        "round_mask",
        "structure_mask",
        "round_structure_mask",
    ]
    predictor_labels = ["全局", "mask", "轮数+mask", "结构+mask", "轮数+结构+mask"]
    decisions = {
        "innovation2_small_spn_exact_label_readiness_passed": "实现就绪；运行冻结16-cipher全密钥审计。",
        "innovation2_small_spn_exact_label_family_ready": "精确标签宽度与反捷径门通过；准备E33网络比较。",
        "innovation2_small_spn_exact_label_shortcut_dominated": "标签可被组外ID边际解释；禁止训练图网络。",
        "innovation2_small_spn_exact_label_too_narrow": "精确标签族宽度或cipher交互不足；停止当前toy family。",
        "innovation2_small_spn_exact_label_protocol_invalid": "双射、全key、缓存、parity或标签协议无效。",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.3,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(16.0, 8.4), gridspec_kw={"width_ratios": [1, 1.25]})
        figure.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.25, wspace=0.25)
        figure.suptitle(
            "创新2 E32：16-bit小状态SPN全密钥精确标签审计",
            x=0.08,
            y=0.96,
            ha="left",
            fontsize=15.2,
            fontweight="bold",
        )
        figure.text(
            0.08,
            0.888,
            "16个S-box/P-layer组合 × 4个轮数 × 14个完整输入集合 × 64个线性mask；每个标签穷举冻结toy family的全部256把master key。",
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.08,
            0.84,
            "该结果只判断精确标签是否足够宽且不能被位置/ID边际解释，不是PRESENT/GIFT/SKINNY攻击或神经训练结果。",
            ha="left",
            va="top",
            fontsize=9.5,
            color="#526070",
        )

        x = np.arange(len(split_names))
        width = 0.36
        axes[0].bar(x - width / 2, positive, width, label="正类：全key XOR恒为0", color="#16A34A")
        axes[0].bar(x + width / 2, negative, width, label="负类", color="#DC2626")
        axes[0].set_xticks(x, split_labels, rotation=15, ha="right")
        axes[0].set_ylabel("标签数量")
        axes[0].set_title("各组外拆分的标签宽度", loc="left", fontweight="bold", pad=12)
        axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper right")
        for index, value in enumerate(positive):
            axes[0].text(index - width / 2, value, str(value), ha="center", va="bottom", fontsize=8.4)
        for index, value in enumerate(negative):
            axes[0].text(index + width / 2, value, str(value), ha="center", va="bottom", fontsize=8.4)

        colors = ["#64748B", "#2563EB", "#D97706", "#7C3AED", "#0F766E"]
        y = np.arange(len(heldout_names))
        offsets = np.linspace(-0.24, 0.24, len(predictor_names))
        for predictor, label, color, offset in zip(
            predictor_names, predictor_labels, colors, offsets, strict=True
        ):
            values = [baselines[split][predictor] for split in heldout_names]
            axes[1].scatter(values, y + offset, label=label, color=color, s=46, zorder=3)
        axes[1].axvline(0.75, color="#DC2626", linestyle="--", linewidth=1.1, label="双轴停止线 0.75")
        axes[1].axvline(0.80, color="#D97706", linestyle=":", linewidth=1.2, label="单轴停止线 0.80")
        axes[1].set_yticks(y, heldout_labels)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0.45, 1.01)
        axes[1].set_xlabel("heldout AUC")
        axes[1].set_title("仅用训练块标签得到的ID边际基线", loc="left", fontweight="bold", pad=12)
        axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=4,
            fontsize=8.2,
        )
        figure.text(
            0.08,
            0.112,
            (
                f"总标签={metrics['total_labels']}，正类={metrics['positive_labels']}，负类={metrics['negative_labels']}，"
                f"不同64-mask签名={metrics['distinct_label_signatures']}，跨cipher变化cell比例={metrics['cipher_variable_cell_fraction']:.3f}。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.08,
            0.067,
            f"裁决：{decisions.get(str(gate['decision']), str(gate['decision']))}",
            ha="left",
            va="bottom",
            fontsize=9.7,
            color="#334155",
        )
        figure.text(
            0.08,
            0.022,
            "证据范围：冻结16-bit合成SPN与8-bit toy全key family；不是实际密码高轮结果、真实key schedule结论或神经性能。",
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#526070",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
