from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


CONDITIONS = (
    "candidate",
    "anchor",
    "repeat_last",
    "rotated",
    "corrupted",
    "no_topology",
)
CONDITION_LABELS = {
    "candidate": "CT-SPN 正确顺序",
    "anchor": "Runtime-E4 锚点",
    "repeat_last": "重复末层",
    "rotated": "旋转顺序",
    "corrupted": "错误拓扑",
    "no_topology": "无拓扑",
}
CONDITION_COLORS = {
    "candidate": "#0F766E",
    "anchor": "#2563EB",
    "repeat_last": "#D97706",
    "rotated": "#7C3AED",
    "corrupted": "#DC2626",
    "no_topology": "#64748B",
}
CIPHER_LABELS = {
    "uknit64": "uKNIT-BC 五轮",
    "dialga128": "Dialga-128 四轮",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese CT-SPN K1 AUC and attribution-margin chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--variant",
        choices=("k1", "k1b", "k1c", "k1d", "k1e", "k1f", "k1g", "k1h"),
        default="k1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = _read_json(args.gate)
    if args.variant == "k1h":
        render_ctspn_k1h_svg(gate, args.output)
    elif args.variant == "k1g":
        render_ctspn_k1g_svg(gate, args.output)
    elif args.variant == "k1f":
        render_ctspn_k1f_svg(gate, args.output)
    elif args.variant == "k1e":
        render_ctspn_k1e_svg(gate, args.output)
    elif args.variant == "k1d":
        render_ctspn_k1d_svg(gate, args.output)
    elif args.variant == "k1c":
        render_ctspn_k1c_svg(gate, args.output)
    elif args.variant == "k1b":
        render_ctspn_k1b_svg(gate, args.output)
    else:
        render_ctspn_k1_svg(gate, args.output)
    report = {
        "status": "rendered_pending_visual_qa",
        "run_id": gate.get("run_id"),
        "gate_status": gate.get("status"),
        "decision": gate.get("decision"),
        "output": str(args.output),
        "variant": args.variant,
        "visual_qa_required": True,
    }
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def render_ctspn_k1_svg(gate: Mapping[str, Any], output: Path) -> None:
    _render_ctspn_svg(
        gate,
        output,
        figure_title=("创新1 K1：规范化线性层顺序是否真正帮助 uKNIT 类 SPN 区分"),
        subtitle=(
            "相同数据与训练预算；候选和锚点分别训练，结构控制复用对应最佳权重且不重新训练。"
        ),
        candidate_label="CT-SPN 正确顺序",
        anchor_label="Runtime-E4 锚点",
        decision_text=_decision_text(gate),
    )


def render_ctspn_k1b_svg(gate: Mapping[str, Any], output: Path) -> None:
    adapted = {**gate, "seed_results": _adapt_k1b_seed_results(gate)}
    _render_ctspn_svg(
        adapted,
        output,
        figure_title=("创新1 K1-B：保留原生端点身份能否恢复 uKNIT 类 SPN 结构归因"),
        subtitle=(
            "只增加来源/目标 cell 位置与 bit-role；数据、训练预算和主干保持不变，所有结构控制复用最佳权重。"
        ),
        candidate_label="K1-B 原生端点",
        anchor_label="最强旧锚点",
        decision_text=_k1b_decision_text(gate),
    )


def render_ctspn_k1d_svg(gate: Mapping[str, Any], output: Path) -> None:
    adapted = {**gate, "seed_results": _adapt_k1d_seed_results(gate)}
    _render_ctspn_svg(
        adapted,
        output,
        figure_title=("创新1 K1-D：相对跨层扩散路径能否改善 uKNIT 类 SPN 区分"),
        subtitle=(
            "不输入绝对 cell 编号；先组合两段线性层之间的源、中间和目标路径，再汇总四个密文对。"
        ),
        candidate_label="K1-D 相对跨层路径",
        anchor_label="同预算旧模型",
        decision_text=_k1d_decision_text(gate),
    )


def render_ctspn_k1c_svg(gate: Mapping[str, Any], output: Path) -> None:
    _render_ctspn_attribution_svg(
        gate,
        output,
        figure_title=("创新1 K1-C：正确拓扑是在训练集过拟合，还是从未被模型学会"),
        subtitle=(
            "复用四个 K1-B 最佳权重；只切换原训练缓存与未见验证数据，五种拓扑均不重新训练。"
        ),
        decision_text=_k1c_decision_text(gate),
    )


def render_ctspn_k1e_svg(gate: Mapping[str, Any], output: Path) -> None:
    _render_ctspn_attribution_svg(
        gate,
        output,
        figure_title=("创新1 K1-E：相对路径是否只记住训练样本，还是丢失共享 cell 关系"),
        subtitle=(
            "复用四个 K1-D 最佳权重与八个原始缓存；只切换数据划分和拓扑控制，不训练新模型。"
        ),
        decision_text=_k1e_decision_text(gate),
    )


def render_ctspn_k1f_svg(gate: Mapping[str, Any], output: Path) -> None:
    seed_results = _validated_k1f_seed_results(gate)
    labels = {
        "candidate": "K1-F 正确超图",
        "anchor": "同预算旧模型",
        "repeat_last": "重复末层",
        "rotated": "旋转顺序",
        "corrupted": "错误拓扑",
        "incidence_shuffled": "共享关系打乱",
        "no_topology": "无拓扑",
    }
    colors = {
        **CONDITION_COLORS,
        "incidence_shuffled": "#0891B2",
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 10.2))
        figure.subplots_adjust(
            left=0.18,
            right=0.97,
            top=0.78,
            bottom=0.08,
            hspace=0.48,
            wspace=0.32,
        )
        figure.suptitle(
            "创新1 K1-F：保留共享 cell 关系能否改善 uKNIT 类 SPN 区分",
            x=0.07,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.905,
            "路径通过共同的源、中间和目标 cell 交换消息；cell 编号只负责路由，不作为网络输入。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.07,
            0.855,
            f"裁决：{_k1f_decision_text(gate)}",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        for column, cipher in enumerate(("uknit64", "dialga128")):
            _plot_k1f_auc_panel(
                axes[0, column],
                cipher,
                seed_results[cipher],
                labels=labels,
                colors=colors,
            )
            _plot_k1f_margin_panel(
                axes[1, column],
                cipher,
                seed_results[cipher],
                labels=labels,
                colors=colors,
            )
        handles = [
            plt.Line2D(
                [0],
                [0],
                color="#111827",
                marker=marker,
                linestyle="none",
                markersize=7,
                label=label,
            )
            for marker, label in (("o", "seed0"), ("s", "seed1"))
        ]
        figure.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.965),
            frameon=False,
            ncol=2,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


def render_ctspn_k1g_svg(gate: Mapping[str, Any], output: Path) -> None:
    seed_results = _validated_k1g_seed_results(gate)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 10.2))
        figure.subplots_adjust(
            left=0.09,
            right=0.97,
            top=0.77,
            bottom=0.11,
            hspace=0.52,
            wspace=0.25,
        )
        figure.suptitle(
            "创新1 K1-G：uKNIT 超图失败来自记住样本，还是只适用于训练密钥",
            x=0.06,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.06,
            0.905,
            "复用四个 K1-F 最佳权重；比较原训练样本、同一密钥的新明文和不同密钥的原验证集，全程不训练。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.06,
            0.85,
            f"裁决：{_k1g_decision_text(gate)}",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_k1g_decision_color(gate),
        )
        for column, cipher in enumerate(("uknit64", "dialga128")):
            _plot_k1g_auc_panel(axes[0, column], cipher, seed_results[cipher])
            _plot_k1g_margin_panel(axes[1, column], cipher, seed_results[cipher])
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.4,
                markersize=7,
                label=label,
            )
            for color, marker, linestyle, label in (
                ("#0F766E", "o", "-", "正确共享关系 seed0"),
                ("#0F766E", "s", "-", "正确共享关系 seed1"),
                ("#DC2626", "x", "--", "最强错误关系"),
            )
        ]
        figure.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.965),
            frameon=False,
            ncol=1,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


def render_ctspn_k1h_svg(gate: Mapping[str, Any], output: Path) -> None:
    seed_results = _validated_k1h_seed_results(gate)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 10.2))
        figure.subplots_adjust(
            left=0.09,
            right=0.97,
            top=0.76,
            bottom=0.11,
            hspace=0.52,
            wspace=0.25,
        )
        figure.suptitle(
            "创新1 K1-H：固定真实扩散算子能否让 uKNIT 类模型泛化",
            x=0.06,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.06,
            0.905,
            "矩阵只规定潜变量的传递路线；网络不接收 bit/cell 编号、密码编号、S 盒、DDT 或路径 token。",
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.06,
            0.85,
            "裁决：低容量精确算子模型未通过；同密钥新明文仍接近随机，Dialga 的强信号也被明显削弱。",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        for column, cipher in enumerate(("uknit64", "dialga128")):
            _plot_k1h_auc_panel(axes[0, column], cipher, seed_results[cipher])
            _plot_k1h_margin_panel(axes[1, column], cipher, seed_results[cipher])
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=6,
                label=label,
            )
            for color, marker, linestyle, label in (
                ("#0F766E", "o", "-", "精确算子 seed0"),
                ("#0F766E", "s", "-", "精确算子 seed1"),
                ("#2563EB", "o", "--", "旧 Runtime-E4 锚点 seed0"),
                ("#2563EB", "s", "--", "旧 Runtime-E4 锚点 seed1"),
            )
        ]
        figure.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.965),
            frameon=False,
            ncol=2,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


def _plot_k1h_auc_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    splits = ("train_seen", "same_key_fresh", "cross_key_validation")
    x = list(range(3))
    all_values: list[float] = []
    for seed, marker, shift in (("0", "o", -0.04), ("1", "s", 0.04)):
        candidate = [float(seeds[seed][split]["candidate_auc"]) for split in splits]
        anchor = [float(seeds[seed][split]["anchor_auc"]) for split in splits]
        positions = [value + shift for value in x]
        axis.plot(
            positions,
            candidate,
            color="#0F766E",
            marker=marker,
            linewidth=1.5,
            markersize=6,
        )
        axis.plot(
            positions,
            anchor,
            color="#2563EB",
            marker=marker,
            linestyle="--",
            linewidth=1.3,
            markersize=5,
        )
        all_values.extend(candidate)
        all_values.extend(anchor)
        x_offset, y_offset, alignment = (
            (-8, 10, "right") if seed == "0" else (8, -16, "left")
        )
        for position, value in zip(positions, candidate, strict=True):
            axis.annotate(
                f"{value:.4f}",
                (position, value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=alignment,
                fontsize=8.0,
                color="#0F766E",
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                },
            )
    span = max(0.04, max(all_values) - min(all_values))
    axis.set_ylim(
        max(0.0, min(all_values) - 0.12 * span),
        min(1.0, max(all_values) + 0.16 * span),
    )
    axis.set_xticks(
        x,
        ("原训练样本\n训练 key", "新明文\n同一训练 key", "原验证集\n不同固定 key"),
    )
    axis.set_title(
        f"{CIPHER_LABELS[cipher]}：精确算子模型与旧锚点 AUC",
        loc="left",
        fontweight="bold",
    )
    axis.set_ylabel("AUC（纵轴按本面板数据缩放）")
    axis.axhline(0.5, color="#9CA3AF", linestyle=(0, (3, 3)), linewidth=1)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _plot_k1h_margin_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    splits = ("train_seen", "same_key_fresh", "cross_key_validation")
    labels = (
        "训练 s0",
        "训练 s1",
        "同 key 新样本 s0",
        "同 key 新样本 s1",
        "跨 key s0",
        "跨 key s1",
    )
    values = [
        (seed, split, seeds[seed][split]) for split in splits for seed in ("0", "1")
    ]
    controls = (
        ("candidate_minus_operator_reversed", "算子反序", "#7C3AED", "o"),
        ("candidate_minus_operator_corrupted", "算子破坏", "#DC2626", "s"),
        ("candidate_minus_no_topology", "无拓扑", "#64748B", "^"),
    )
    flat: list[float] = []
    for key, label, color, marker in controls:
        margins = [float(row[key]) for _, _, row in values]
        flat.extend(margins)
        axis.plot(
            range(len(values)),
            margins,
            color=color,
            marker=marker,
            linewidth=1.2,
            markersize=5,
            label=label,
        )
    span = max(0.02, max(flat) - min(flat))
    axis.set_ylim(
        min(-0.01, min(flat) - 0.15 * span),
        max(0.015, max(flat) + 0.2 * span),
    )
    axis.set_xticks(range(len(values)), labels, rotation=18, ha="right")
    axis.set_title(
        f"{CIPHER_LABELS[cipher]}：正确算子相对控制的净优势",
        loc="left",
        fontweight="bold",
    )
    axis.set_ylabel("精确算子 AUC - 控制 AUC（局部放大）")
    axis.axhline(0.0, color="#9CA3AF", linewidth=1)
    axis.axhline(0.005, color="#047857", linestyle=(0, (4, 3)), linewidth=1.3)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left")


def _validated_k1h_seed_results(
    gate: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]:
    results = gate.get("seed_results")
    if not isinstance(results, Mapping) or set(results) != {"uknit64", "dialga128"}:
        raise ValueError("K1-H gate must contain both cipher seed panels")
    required_splits = {"train_seen", "same_key_fresh", "cross_key_validation"}
    for cipher in ("uknit64", "dialga128"):
        seeds = results[cipher]
        if not isinstance(seeds, Mapping) or set(seeds) != {"0", "1"}:
            raise ValueError("K1-H gate must contain both seeds")
        for seed in ("0", "1"):
            split_rows = seeds[seed]
            if (
                not isinstance(split_rows, Mapping)
                or set(split_rows) != required_splits
            ):
                raise ValueError("K1-H gate must contain all three data splits")
    return results  # type: ignore[return-value]


def _plot_k1g_auc_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    splits = ("train_seen", "same_key_fresh", "cross_key_validation")
    x = list(range(len(splits)))
    for seed, marker, offset in (("0", "o", -0.06), ("1", "s", 0.06)):
        correct = [float(seeds[seed][split]["correct_ordered_auc"]) for split in splits]
        strongest_wrong = [
            max(
                float(value)
                for key, value in seeds[seed][split].items()
                if key.endswith("_auc") and key != "correct_ordered_auc"
            )
            for split in splits
        ]
        positions = [value + offset for value in x]
        axis.plot(
            positions,
            correct,
            color="#0F766E",
            marker=marker,
            linewidth=1.5,
            markersize=6,
        )
        axis.plot(
            positions,
            strongest_wrong,
            color="#DC2626",
            marker="x",
            linestyle="--",
            linewidth=1.1,
            markersize=6,
            alpha=0.8,
        )
        label_offset = 8 if seed == "0" else -14
        for position, value in zip(positions, correct, strict=True):
            axis.annotate(
                f"{value:.4f}",
                (position, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8.0,
                color="#0F766E",
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )
    values = []
    for seed in ("0", "1"):
        for split in splits:
            split_values = seeds[seed][split]
            values.append(float(split_values["correct_ordered_auc"]))
            values.append(
                max(
                    float(value)
                    for key, value in split_values.items()
                    if key.endswith("_auc") and key != "correct_ordered_auc"
                )
            )
    zoomed = cipher == "dialga128" and min(values) > 0.8
    if zoomed:
        lower = max(0.0, min(values) - 0.025)
        upper = min(1.0, max(values) + 0.015)
    else:
        lower = max(0.0, min(0.48, min(values) - 0.025))
        upper = min(1.0, max(values) + max(0.04, 0.15 * (max(values) - lower)))
    axis.set_ylim(lower, upper)
    axis.set_xticks(
        x,
        ("原训练样本\n训练 key", "新明文\n仍用训练 key", "原验证集\n更换固定 key"),
    )
    axis.set_title(
        (
            f"{CIPHER_LABELS[cipher]}：正确关系与最强错误关系 AUC"
            + ("（纵轴局部放大）" if zoomed else "")
        ),
        loc="left",
        fontweight="bold",
    )
    axis.set_ylabel("AUC")
    if lower <= 0.5 <= upper:
        axis.axhline(0.5, color="#9CA3AF", linestyle=(0, (3, 3)), linewidth=1)
    if cipher == "uknit64":
        axis.axhline(0.52, color="#047857", linestyle=(0, (4, 3)), linewidth=1.2)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _plot_k1g_margin_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    splits = ("train_seen", "same_key_fresh", "cross_key_validation")
    x = list(range(len(splits)))
    for seed, marker, offset in (("0", "o", -0.06), ("1", "s", 0.06)):
        minimum = [
            min(
                float(value)
                for key, value in seeds[seed][split].items()
                if key.startswith("correct_minus_")
            )
            for split in splits
        ]
        incidence = [
            float(seeds[seed][split]["correct_minus_incidence_shuffled"])
            for split in splits
        ]
        positions = [value + offset for value in x]
        axis.plot(
            positions,
            minimum,
            color="#7C3AED",
            marker=marker,
            linewidth=1.5,
            markersize=6,
        )
        axis.plot(
            positions,
            incidence,
            color="#0891B2",
            marker=marker,
            linestyle="--",
            linewidth=1.2,
            markersize=5,
        )
        label_offset = 8 if seed == "0" else -14
        for position, value in zip(positions, minimum, strict=True):
            axis.annotate(
                f"{value:+.4f}",
                (position, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8.0,
                color="#6D28D9",
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )
    all_margins = [
        float(value)
        for seed in ("0", "1")
        for split in splits
        for key, value in seeds[seed][split].items()
        if key.startswith("correct_minus_")
    ]
    span = max(0.02, max(all_margins) - min(all_margins))
    axis.set_ylim(
        min(-0.02, min(all_margins) - 0.2 * span),
        max(0.025, max(all_margins) + 0.25 * span),
    )
    axis.set_xticks(
        x,
        ("原训练样本\n训练 key", "新明文\n仍用训练 key", "原验证集\n更换固定 key"),
    )
    axis.set_title(
        f"{CIPHER_LABELS[cipher]}：最弱控制优势与共享关系净优势",
        loc="left",
        fontweight="bold",
    )
    axis.set_ylabel("正确关系 - 控制（AUC）")
    axis.axhline(0.0, color="#9CA3AF", linewidth=1)
    axis.axhline(0.005, color="#047857", linestyle=(0, (4, 3)), linewidth=1.2)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.text(
        0.01,
        0.98,
        "紫色：五种控制中的最小优势；蓝色虚线：仅共享-cell关系打乱",
        transform=axis.transAxes,
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )


def _plot_k1f_auc_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, float]],
    *,
    labels: Mapping[str, str],
    colors: Mapping[str, str],
) -> None:
    keys = (
        "candidate",
        "anchor",
        "repeat_last",
        "rotated",
        "corrupted",
        "incidence_shuffled",
        "no_topology",
    )
    values = {
        seed: {
            "candidate": float(row["candidate_auc"]),
            "anchor": float(row["anchor_auc"]),
            **{key: float(row[f"{key}_auc"]) for key in keys[2:]},
        }
        for seed, row in seeds.items()
    }
    y_positions = list(reversed(range(len(keys))))
    for index, key in enumerate(keys):
        y = y_positions[index]
        seed0 = values["0"][key]
        seed1 = values["1"][key]
        axis.plot((seed0, seed1), (y - 0.12, y + 0.12), color=colors[key], alpha=0.55)
        for value, offset, marker in ((seed0, -0.12, "o"), (seed1, 0.12, "s")):
            axis.scatter(
                value, y + offset, color=colors[key], marker=marker, s=44, zorder=3
            )
            axis.annotate(
                f"{value:.4f}",
                (value, y + offset),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.1,
                color="#374151",
            )
    flat = [values[seed][key] for seed in ("0", "1") for key in keys]
    lower = max(0.0, min(0.5, min(flat)) - 0.012)
    upper = min(1.0, max(flat) + max(0.035, (max(flat) - lower) * 0.2))
    axis.set_xlim(lower, upper)
    axis.set_yticks(y_positions, [labels[key] for key in keys])
    axis.set_title(
        f"{CIPHER_LABELS[cipher]}：冻结验证 AUC", loc="left", fontweight="bold"
    )
    axis.set_xlabel("AUC")
    axis.axvline(0.5, color="#9CA3AF", linestyle=(0, (3, 3)), linewidth=1)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)


def _plot_k1f_margin_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, float]],
    *,
    labels: Mapping[str, str],
    colors: Mapping[str, str],
) -> None:
    keys = (
        "anchor",
        "repeat_last",
        "rotated",
        "corrupted",
        "incidence_shuffled",
        "no_topology",
    )
    margins = {
        seed: {key: float(row[f"candidate_minus_{key}"]) for key in keys}
        for seed, row in seeds.items()
    }
    focus_keys = keys[:-1]
    focus = [margins[seed][key] for seed in ("0", "1") for key in focus_keys]
    no_topology = [margins[seed]["no_topology"] for seed in ("0", "1")]
    separate = max(abs(value) for value in no_topology) > max(
        0.05,
        8.0 * max(abs(value) for value in focus),
    )
    visible = focus_keys if separate else keys
    y_positions = list(reversed(range(len(visible))))
    for index, key in enumerate(visible):
        _plot_seed_pair(
            axis,
            margins["0"][key],
            margins["1"][key],
            y=y_positions[index],
            color=colors[key],
        )
    flat = [margins[seed][key] for seed in ("0", "1") for key in visible]
    span = max(0.02, max(flat) - min(flat))
    axis.set_xlim(
        min(-0.01, min(flat) - 0.15 * span), max(0.02, max(flat) + 0.3 * span)
    )
    axis.set_yticks(y_positions, [f"正确超图 - {labels[key]}" for key in visible])
    axis.set_title(
        f"{CIPHER_LABELS[cipher]}：正确关系的净优势", loc="left", fontweight="bold"
    )
    axis.set_xlabel("AUC 差值")
    axis.axvline(0.0, color="#9CA3AF", linewidth=1)
    axis.axvline(0.005, color="#047857", linestyle=(0, (4, 3)), linewidth=1.3)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    if separate:
        _add_no_topology_inset(axis, no_topology)


def _render_ctspn_attribution_svg(
    gate: Mapping[str, Any],
    output: Path,
    *,
    figure_title: str,
    subtitle: str,
    decision_text: str,
) -> None:
    seed_results = _validated_k1c_seed_results(gate)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 10.2))
        figure.subplots_adjust(
            left=0.17,
            right=0.97,
            top=0.76,
            bottom=0.09,
            hspace=0.58,
            wspace=0.29,
        )
        figure.suptitle(
            figure_title,
            x=0.07,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.905,
            subtitle,
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.07,
            0.855,
            f"裁决：{decision_text}",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.07,
            0.81,
            "横轴是正确拓扑 AUC 减去错误控制 AUC；所有点都必须越过 +0.005 绿线才算学会正确拓扑。",
            ha="left",
            fontsize=9.5,
            color="#6B7280",
        )
        for column, cipher in enumerate(("uknit64", "dialga128")):
            _plot_k1c_split_margin(
                axes[0, column], cipher, seed_results[cipher], split="train"
            )
            _plot_k1c_split_margin(
                axes[1, column], cipher, seed_results[cipher], split="validation"
            )
        handles = [
            plt.Line2D(
                [0],
                [0],
                color="#111827",
                marker=marker,
                linestyle="none",
                markersize=7,
                label=label,
            )
            for marker, label in (("o", "seed0"), ("s", "seed1"))
        ]
        figure.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.965),
            frameon=False,
            ncol=2,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


def _plot_k1c_split_margin(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    split: str,
) -> None:
    margin_keys = ("repeat_last", "rotated", "corrupted", "no_topology")
    margins = {
        seed: {key: float(values[split][f"correct_minus_{key}"]) for key in margin_keys}
        for seed, values in seeds.items()
    }
    focus_values = [
        margins[seed][key] for seed in ("0", "1") for key in margin_keys[:-1]
    ]
    no_topology_values = [margins[seed]["no_topology"] for seed in ("0", "1")]
    separate_no_topology = max(abs(value) for value in no_topology_values) > max(
        0.05,
        8.0 * max(abs(value) for value in focus_values),
    )
    visible_keys = margin_keys[:-1] if separate_no_topology else margin_keys
    y_positions = list(reversed(range(len(visible_keys))))
    for index, key in enumerate(visible_keys):
        _plot_seed_pair(
            axis,
            margins["0"][key],
            margins["1"][key],
            y=y_positions[index],
            color=CONDITION_COLORS[key],
        )
    flat = [margins[seed][key] for seed in ("0", "1") for key in visible_keys]
    span = max(0.02, max(flat) - min(flat))
    axis.set_xlim(
        min(-0.01, min(flat) - 0.15 * span),
        max(0.02, max(flat) + 0.30 * span),
    )
    axis.set_yticks(
        y_positions,
        [f"正确拓扑 - {CONDITION_LABELS[key]}" for key in visible_keys],
    )
    split_label = "原训练缓存" if split == "train" else "未见验证数据"
    correct_aucs = [
        float(seeds[seed][split]["correct_ordered_auc"]) for seed in ("0", "1")
    ]
    axis.set_title(
        f"{CIPHER_LABELS[cipher]}：{split_label}",
        loc="left",
        fontweight="bold",
        pad=24,
    )
    axis.text(
        0.0,
        1.02,
        f"正确拓扑 AUC：seed0 {correct_aucs[0]:.4f}，seed1 {correct_aucs[1]:.4f}",
        transform=axis.transAxes,
        fontsize=8.8,
        color="#4B5563",
    )
    axis.set_xlabel("正确拓扑净优势（AUC 差值）")
    axis.axvline(0.0, color="#9CA3AF", linewidth=1)
    axis.axvline(0.005, color="#047857", linestyle=(0, (4, 3)), linewidth=1.3)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    if separate_no_topology:
        _add_no_topology_inset(axis, no_topology_values)


def _render_ctspn_svg(
    gate: Mapping[str, Any],
    output: Path,
    *,
    figure_title: str,
    subtitle: str,
    candidate_label: str,
    anchor_label: str,
    decision_text: str,
) -> None:
    seed_results = _validated_seed_results(gate)
    condition_labels = {
        **CONDITION_LABELS,
        "candidate": candidate_label,
        "anchor": anchor_label,
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#111827",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#374151",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(16, 10.2))
        figure.subplots_adjust(
            left=0.15,
            right=0.97,
            top=0.76,
            bottom=0.09,
            hspace=0.5,
            wspace=0.27,
        )
        figure.suptitle(
            figure_title,
            x=0.07,
            y=0.96,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.905,
            subtitle,
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.07,
            0.855,
            f"裁决：{decision_text}",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=_decision_color(str(gate.get("status", ""))),
        )
        figure.text(
            0.07,
            0.81,
            "AUC 图越靠右越好；边际图必须同时超过 0.005 虚线，任一密码或 seed 失败均不通过。",
            ha="left",
            fontsize=9.5,
            color="#6B7280",
        )

        for column, cipher in enumerate(("uknit64", "dialga128")):
            _plot_auc_panel(
                axes[0, column],
                cipher,
                seed_results[cipher],
                condition_labels=condition_labels,
            )
            _plot_margin_panel(
                axes[1, column],
                cipher,
                seed_results[cipher],
                condition_labels=condition_labels,
            )

        handles = [
            plt.Line2D(
                [0],
                [0],
                color="#111827",
                marker=marker,
                linestyle="none",
                markersize=7,
                label=label,
            )
            for marker, label in (("o", "seed0"), ("s", "seed1"))
        ]
        figure.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.965, 0.965),
            frameon=False,
            ncol=2,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)


def _plot_auc_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, float]],
    *,
    condition_labels: Mapping[str, str],
) -> None:
    values = _auc_values(seeds)
    y_positions = list(reversed(range(len(CONDITIONS))))
    for index, condition in enumerate(CONDITIONS):
        y = y_positions[index]
        color = CONDITION_COLORS[condition]
        seed0 = values["0"][condition]
        seed1 = values["1"][condition]
        axis.plot(
            (seed0, seed1),
            (y - 0.12, y + 0.12),
            color=color,
            linewidth=1.6,
            alpha=0.55,
        )
        for value, offset, marker in ((seed0, -0.12, "o"), (seed1, 0.12, "s")):
            axis.scatter(
                value,
                y + offset,
                color=color,
                marker=marker,
                s=48,
                zorder=3,
            )
            axis.annotate(
                f"{value:.4f}",
                (value, y + offset),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.3,
                color="#374151",
            )
    flat = [value for seed in values.values() for value in seed.values()]
    lower = max(0.0, min(0.5, min(flat)) - 0.012)
    upper = min(1.0, max(flat) + max(0.035, (max(flat) - lower) * 0.2))
    axis.set_xlim(lower, upper)
    axis.set_yticks(y_positions, [condition_labels[key] for key in CONDITIONS])
    axis.set_title(
        f"{CIPHER_LABELS[cipher]}：冻结验证 AUC", loc="left", fontweight="bold"
    )
    axis.set_xlabel("AUC")
    axis.axvline(0.5, color="#9CA3AF", linestyle=(0, (3, 3)), linewidth=1)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)


def _plot_margin_panel(
    axis: plt.Axes,
    cipher: str,
    seeds: Mapping[str, Mapping[str, float]],
    *,
    condition_labels: Mapping[str, str],
) -> None:
    margin_keys = (
        "anchor",
        "repeat_last",
        "rotated",
        "corrupted",
        "no_topology",
    )
    margins = {
        seed: {key: float(values[f"candidate_minus_{key}"]) for key in margin_keys}
        for seed, values in seeds.items()
    }
    focus_values = [
        margins[seed][key] for seed in ("0", "1") for key in margin_keys[:-1]
    ]
    no_topology_values = [margins[seed]["no_topology"] for seed in ("0", "1")]
    separate_no_topology = max(abs(value) for value in no_topology_values) > max(
        0.05,
        8.0 * max(abs(value) for value in focus_values),
    )
    visible_keys = margin_keys[:-1] if separate_no_topology else margin_keys
    y_positions = list(reversed(range(len(visible_keys))))
    for index, key in enumerate(visible_keys):
        y = y_positions[index]
        color = CONDITION_COLORS[key]
        seed0 = margins["0"][key]
        seed1 = margins["1"][key]
        _plot_seed_pair(axis, seed0, seed1, y=y, color=color)
    flat = [margins[seed][key] for seed in ("0", "1") for key in visible_keys]
    span = max(0.02, max(flat) - min(flat))
    axis.set_xlim(
        min(-0.01, min(flat) - 0.15 * span), max(0.02, max(flat) + 0.3 * span)
    )
    axis.set_yticks(
        y_positions,
        [f"候选 - {condition_labels[key]}" for key in visible_keys],
    )
    axis.set_title(
        f"{CIPHER_LABELS[cipher]}：正确顺序的净优势", loc="left", fontweight="bold"
    )
    axis.set_xlabel("AUC 差值")
    axis.axvline(0.0, color="#9CA3AF", linewidth=1)
    axis.axvline(0.005, color="#047857", linestyle=(0, (4, 3)), linewidth=1.3)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    if separate_no_topology:
        _add_no_topology_inset(axis, no_topology_values)


def _plot_seed_pair(
    axis: plt.Axes,
    seed0: float,
    seed1: float,
    *,
    y: float,
    color: str,
) -> None:
    axis.plot(
        (seed0, seed1),
        (y - 0.12, y + 0.12),
        color=color,
        linewidth=1.6,
        alpha=0.55,
    )
    for value, offset, marker in ((seed0, -0.12, "o"), (seed1, 0.12, "s")):
        axis.scatter(
            value,
            y + offset,
            color=color,
            marker=marker,
            s=48,
            zorder=3,
        )
        axis.annotate(
            f"{value:+.4f}",
            (value, y + offset),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.3,
            color="#374151",
        )


def _add_no_topology_inset(axis: plt.Axes, values: list[float]) -> None:
    position = axis.get_position()
    axis.set_position(
        [position.x0, position.y0, position.width * 0.68, position.height]
    )
    inset = axis.figure.add_axes(
        [
            position.x0 + position.width * 0.76,
            position.y0 + position.height * 0.24,
            position.width * 0.24,
            position.height * 0.42,
        ]
    )
    _plot_seed_pair(
        inset,
        values[0],
        values[1],
        y=0.0,
        color=CONDITION_COLORS["no_topology"],
    )
    span = max(0.01, abs(values[1] - values[0]))
    inset.set_xlim(min(values) - 0.5 * span, max(values) + 1.1 * span)
    inset.set_ylim(-0.45, 0.45)
    inset.set_yticks([])
    inset.set_title("无拓扑优势（单独尺度）", fontsize=9, fontweight="bold")
    inset.set_xlabel("AUC 差值", fontsize=8.5)
    inset.tick_params(axis="x", labelsize=8)
    inset.grid(axis="x", color="#E5E7EB", linewidth=0.8)


def _auc_values(
    seeds: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        seed: {
            "candidate": float(values["candidate_auc"]),
            "anchor": float(values["anchor_auc"]),
            "repeat_last": float(values["candidate_repeat_last_auc"]),
            "rotated": float(values["candidate_rotated_auc"]),
            "corrupted": float(values["candidate_corrupted_auc"]),
            "no_topology": float(values["candidate_no_topology_auc"]),
        }
        for seed, values in seeds.items()
    }


def _validated_seed_results(
    gate: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    raw = gate.get("seed_results")
    if not isinstance(raw, dict) or set(raw) != {"uknit64", "dialga128"}:
        raise ValueError("K1 gate must contain uKNIT and Dialga seed results")
    required = {
        "candidate_auc",
        "anchor_auc",
        "candidate_repeat_last_auc",
        "candidate_rotated_auc",
        "candidate_corrupted_auc",
        "candidate_no_topology_auc",
        "candidate_minus_anchor",
        "candidate_minus_repeat_last",
        "candidate_minus_rotated",
        "candidate_minus_corrupted",
        "candidate_minus_no_topology",
    }
    result: dict[str, dict[str, dict[str, float]]] = {}
    for cipher, seeds in raw.items():
        if not isinstance(seeds, dict) or set(seeds) != {"0", "1"}:
            raise ValueError(f"K1 {cipher} must contain seed0 and seed1")
        result[cipher] = {}
        for seed, values in seeds.items():
            if not isinstance(values, dict) or not required.issubset(values):
                raise ValueError(f"K1 {cipher} seed{seed} metrics are incomplete")
            if any(not isinstance(values[key], (int, float)) for key in required):
                raise ValueError(f"K1 {cipher} seed{seed} metrics must be numeric")
            result[cipher][seed] = {key: float(values[key]) for key in required}
    return result


def _adapt_k1b_seed_results(
    gate: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    raw = gate.get("seed_results")
    if not isinstance(raw, dict) or set(raw) != {"uknit64", "dialga128"}:
        raise ValueError("K1-B gate must contain uKNIT and Dialga seed results")
    required = {
        "candidate_auc",
        "prior_anchor_auc",
        "prior_edge_invariant_auc",
        "candidate_minus_strongest_prior",
        "candidate_repeat_last_auc",
        "candidate_rotated_auc",
        "candidate_corrupted_auc",
        "candidate_no_topology_auc",
        "candidate_minus_repeat_last",
        "candidate_minus_rotated",
        "candidate_minus_corrupted",
        "candidate_minus_no_topology",
    }
    adapted: dict[str, dict[str, dict[str, float]]] = {}
    for cipher, seeds in raw.items():
        if not isinstance(seeds, dict) or set(seeds) != {"0", "1"}:
            raise ValueError(f"K1-B {cipher} must contain seed0 and seed1")
        adapted[cipher] = {}
        for seed, values in seeds.items():
            if not isinstance(values, dict) or not required.issubset(values):
                raise ValueError(f"K1-B {cipher} seed{seed} metrics are incomplete")
            if any(not isinstance(values[key], (int, float)) for key in required):
                raise ValueError(f"K1-B {cipher} seed{seed} metrics must be numeric")
            adapted[cipher][seed] = {
                "candidate_auc": float(values["candidate_auc"]),
                "anchor_auc": max(
                    float(values["prior_anchor_auc"]),
                    float(values["prior_edge_invariant_auc"]),
                ),
                "candidate_repeat_last_auc": float(values["candidate_repeat_last_auc"]),
                "candidate_rotated_auc": float(values["candidate_rotated_auc"]),
                "candidate_corrupted_auc": float(values["candidate_corrupted_auc"]),
                "candidate_no_topology_auc": float(values["candidate_no_topology_auc"]),
                "candidate_minus_anchor": float(
                    values["candidate_minus_strongest_prior"]
                ),
                "candidate_minus_repeat_last": float(
                    values["candidate_minus_repeat_last"]
                ),
                "candidate_minus_rotated": float(values["candidate_minus_rotated"]),
                "candidate_minus_corrupted": float(values["candidate_minus_corrupted"]),
                "candidate_minus_no_topology": float(
                    values["candidate_minus_no_topology"]
                ),
            }
    return adapted


def _adapt_k1d_seed_results(
    gate: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    raw = gate.get("seed_results")
    if not isinstance(raw, dict) or set(raw) != {"uknit64", "dialga128"}:
        raise ValueError("K1-D gate must contain uKNIT and Dialga seed results")
    required = {
        "candidate_auc",
        "anchor_auc",
        "candidate_minus_anchor",
        "repeat_last_auc",
        "rotated_auc",
        "corrupted_auc",
        "no_topology_auc",
        "candidate_minus_repeat_last",
        "candidate_minus_rotated",
        "candidate_minus_corrupted",
        "candidate_minus_no_topology",
    }
    adapted: dict[str, dict[str, dict[str, float]]] = {}
    for cipher, seeds in raw.items():
        if not isinstance(seeds, dict) or set(seeds) != {"0", "1"}:
            raise ValueError(f"K1-D {cipher} must contain seed0 and seed1")
        adapted[cipher] = {}
        for seed, values in seeds.items():
            if not isinstance(values, dict) or not required.issubset(values):
                raise ValueError(f"K1-D {cipher} seed{seed} metrics are incomplete")
            if any(not isinstance(values[key], (int, float)) for key in required):
                raise ValueError(f"K1-D {cipher} seed{seed} metrics must be numeric")
            adapted[cipher][seed] = {
                "candidate_auc": float(values["candidate_auc"]),
                "anchor_auc": float(values["anchor_auc"]),
                "candidate_repeat_last_auc": float(values["repeat_last_auc"]),
                "candidate_rotated_auc": float(values["rotated_auc"]),
                "candidate_corrupted_auc": float(values["corrupted_auc"]),
                "candidate_no_topology_auc": float(values["no_topology_auc"]),
                "candidate_minus_anchor": float(values["candidate_minus_anchor"]),
                "candidate_minus_repeat_last": float(
                    values["candidate_minus_repeat_last"]
                ),
                "candidate_minus_rotated": float(values["candidate_minus_rotated"]),
                "candidate_minus_corrupted": float(values["candidate_minus_corrupted"]),
                "candidate_minus_no_topology": float(
                    values["candidate_minus_no_topology"]
                ),
            }
    return adapted


def _validated_k1c_seed_results(
    gate: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    raw = gate.get("seed_results")
    if not isinstance(raw, dict) or set(raw) != {"uknit64", "dialga128"}:
        raise ValueError("K1-C gate must contain uKNIT and Dialga seed results")
    required = {
        "correct_ordered_auc",
        "repeat_last_auc",
        "rotated_auc",
        "corrupted_auc",
        "no_topology_auc",
        "correct_minus_repeat_last",
        "correct_minus_rotated",
        "correct_minus_corrupted",
        "correct_minus_no_topology",
    }
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for cipher, seeds in raw.items():
        if not isinstance(seeds, dict) or set(seeds) != {"0", "1"}:
            raise ValueError(f"K1-C {cipher} must contain seed0 and seed1")
        result[cipher] = {}
        for seed, splits in seeds.items():
            if not isinstance(splits, dict) or set(splits) != {
                "train",
                "validation",
            }:
                raise ValueError(f"K1-C {cipher} seed{seed} must contain both splits")
            result[cipher][seed] = {}
            for split, values in splits.items():
                if not isinstance(values, dict) or not required.issubset(values):
                    raise ValueError(
                        f"K1-C {cipher} seed{seed} {split} metrics are incomplete"
                    )
                if any(not isinstance(values[key], (int, float)) for key in required):
                    raise ValueError(
                        f"K1-C {cipher} seed{seed} {split} metrics must be numeric"
                    )
                result[cipher][seed][split] = {
                    key: float(values[key]) for key in required
                }
    return result


def _validated_k1f_seed_results(
    gate: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    raw = gate.get("seed_results")
    if not isinstance(raw, dict) or set(raw) != {"uknit64", "dialga128"}:
        raise ValueError("K1-F gate must contain uKNIT and Dialga seed results")
    controls = (
        "repeat_last",
        "rotated",
        "corrupted",
        "incidence_shuffled",
        "no_topology",
    )
    required = {
        "candidate_auc",
        "anchor_auc",
        "candidate_minus_anchor",
        *(f"{key}_auc" for key in controls),
        *(f"candidate_minus_{key}" for key in controls),
    }
    result: dict[str, dict[str, dict[str, float]]] = {}
    for cipher, seeds in raw.items():
        if not isinstance(seeds, dict) or set(seeds) != {"0", "1"}:
            raise ValueError(f"K1-F {cipher} must contain seed0 and seed1")
        result[cipher] = {}
        for seed, values in seeds.items():
            if not isinstance(values, dict) or not required.issubset(values):
                raise ValueError(f"K1-F {cipher} seed{seed} metrics are incomplete")
            if any(not isinstance(values[key], (int, float)) for key in required):
                raise ValueError(f"K1-F {cipher} seed{seed} metrics must be numeric")
            result[cipher][seed] = {key: float(values[key]) for key in required}
    return result


def _validated_k1g_seed_results(
    gate: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    raw = gate.get("seed_results")
    if not isinstance(raw, dict) or set(raw) != {"uknit64", "dialga128"}:
        raise ValueError("K1-G gate must contain uKNIT and Dialga seed results")
    splits = ("train_seen", "same_key_fresh", "cross_key_validation")
    required = {
        "correct_ordered_auc",
        "repeat_last_auc",
        "rotated_auc",
        "corrupted_auc",
        "no_topology_auc",
        "incidence_shuffled_auc",
        "correct_minus_repeat_last",
        "correct_minus_rotated",
        "correct_minus_corrupted",
        "correct_minus_no_topology",
        "correct_minus_incidence_shuffled",
    }
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for cipher, seeds in raw.items():
        if not isinstance(seeds, dict) or set(seeds) != {"0", "1"}:
            raise ValueError(f"K1-G {cipher} must contain seed0 and seed1")
        result[cipher] = {}
        for seed, seed_splits in seeds.items():
            if not isinstance(seed_splits, dict) or set(seed_splits) != set(splits):
                raise ValueError(f"K1-G {cipher} seed{seed} splits are incomplete")
            result[cipher][seed] = {}
            for split, values in seed_splits.items():
                if not isinstance(values, dict) or not required.issubset(values):
                    raise ValueError(
                        f"K1-G {cipher} seed{seed} {split} metrics are incomplete"
                    )
                if any(not isinstance(values[key], (int, float)) for key in required):
                    raise ValueError(
                        f"K1-G {cipher} seed{seed} {split} metrics must be numeric"
                    )
                result[cipher][seed][split] = {
                    key: float(values[key]) for key in required
                }
    return result


def _decision_text(gate: Mapping[str, Any]) -> str:
    status = str(gate.get("status", "unknown"))
    if status == "pass":
        return "通过，正确规范顺序在两种密码和两颗 seed 上均满足绝对 AUC 与归因边际"
    if status == "hold":
        return "暂缓，至少一个密码、seed 或结构控制未达到预注册门槛"
    return "协议无效，必须修复证据绑定后按原计划重跑"


def _k1b_decision_text(gate: Mapping[str, Any]) -> str:
    status = str(gate.get("status", "unknown"))
    if status == "pass":
        return (
            "通过，原生端点候选在两种密码和两颗 seed 上均满足绝对、旧锚点和结构控制门槛"
        )
    if status == "hold":
        return "暂缓，原生端点身份单独不足以在所有密码、seed 和结构控制上建立稳定优势"
    return "协议无效，必须修复数据、检查点或控制绑定后按原计划重跑"


def _k1c_decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("split_specific_topology_overfit_confirmed"):
        return "训练集学会、验证集失效，确认绝对端点摘要发生结构过拟合"
    if decision.endswith("endpoint_summary_not_attributed_on_training"):
        return "训练集也未稳定偏好正确拓扑，关闭逐层端点摘要路线"
    if decision.endswith("source_replay_inconsistency"):
        return "冻结重放与 K1-B 裁决不一致，必须先审计权重和证据绑定"
    return "协议无效，必须修复源数据、权重或重放绑定后原样重跑"


def _k1d_decision_text(gate: Mapping[str, Any]) -> str:
    status = str(gate.get("status", "unknown"))
    if status == "pass":
        return "通过，相对路径在两种密码、两颗 seed 和全部错误拓扑控制上均达到门槛"
    if status == "hold":
        return "暂缓，相对路径未在所有 uKNIT seed、旧模型锚点和错误拓扑控制上建立优势"
    return "协议无效，必须修复数据、检查点或控制绑定后按原计划重跑"


def _k1e_decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("split_specific_relative_path_overfit_confirmed"):
        return "训练集偏好正确路径、验证集失效，确认相对路径仍发生样本划分过拟合"
    if decision.endswith("anonymous_path_relation_collapse_confirmed"):
        return "训练集也未稳定归因正确路径，确认无身份路径集合丢失共享 cell 关系"
    if decision.endswith("source_replay_inconsistency"):
        return "冻结重放与 K1-D 裁决不一致，必须先审计权重和证据绑定"
    return "协议无效，必须修复源数据、权重或重放绑定后原样重跑"


def _k1f_decision_text(gate: Mapping[str, Any]) -> str:
    status = str(gate.get("status", "unknown"))
    if status == "pass":
        return "通过，正确 cell/path 超图在两种密码、两颗 seed 和全部关系控制上达到门槛"
    if status == "hold":
        return "暂缓，正确超图未在所有 uKNIT seed、旧模型和错误关系控制上建立稳定优势"
    return "协议无效，必须修复数据、检查点或关系控制绑定后按原计划重跑"


def _k1g_decision_text(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("key_specific_hypergraph_signal_confirmed"):
        return "同 key 新样本可泛化，但更换固定 key 后失效：主要是密钥特定信号"
    if decision.endswith("sample_specific_hypergraph_attribution_overfit_confirmed"):
        return "同 key 新样本不再偏好正确关系：确认共享关系归因只适用于训练样本"
    if decision.endswith("same_key_relation_generalizes_but_signal_weak"):
        return "同 key 新样本保留关系归因，但绝对 AUC 仍低于门槛：任务信号太弱"
    if decision.endswith("shared_cell_relation_underuse_confirmed"):
        return "训练集上正确共享关系也不占优：超图共享关系没有被可靠使用"
    if decision.endswith("hypergraph_generalization_supported_but_k1f_gate_held"):
        return "三种数据均保留结构归因；需定位 K1-F 的剩余锚点门槛失败"
    if decision.endswith("mixed_failure_confirmed"):
        return "两颗 seed 的失败原因不一致，禁止平均掩盖并继续扩样"
    return "协议无效：先修复数据、密钥、重合或检查点绑定"


def _k1g_decision_color(gate: Mapping[str, Any]) -> str:
    decision = str(gate.get("decision", ""))
    if decision.endswith("hypergraph_generalization_supported_but_k1f_gate_held"):
        return "#047857"
    if str(gate.get("status", "")) == "invalid":
        return "#B91C1C"
    return "#B45309"


def _decision_color(status: str) -> str:
    return {"pass": "#047857", "hold": "#B45309", "fail": "#B91C1C"}.get(
        status, "#374151"
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
