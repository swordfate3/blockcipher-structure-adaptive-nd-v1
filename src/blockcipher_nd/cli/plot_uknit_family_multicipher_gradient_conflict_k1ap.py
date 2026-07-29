from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CIPHER_LABELS = {
    "uknit64": "uKNIT-BC",
    "midori64": "Midori64",
    "dialga128": "Dialga-128",
}
PAIR_LABELS = {
    "uknit64__midori64": "uKNIT / Midori",
    "uknit64__dialga128": "uKNIT / Dialga",
    "midori64__dialga128": "Midori / Dialga",
}
CONDITION_LABELS = {
    "correct_runtime": "正确结构",
    "wrong_sbox_same_checkpoint": "错误 S盒",
    "transition_branch_off_same_checkpoint": "关闭分支",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AP gradient audit chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.results)
    report = render_k1ap_svg(gate, rows, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ap_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    pair_rows, norm_rows = _validate_rows(rows)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.4,
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
        figure, axes = plt.subplots(2, 2, figsize=(17.8, 10.8))
        figure.subplots_adjust(
            left=0.09,
            right=0.97,
            top=0.78,
            bottom=0.14,
            hspace=0.47,
            wspace=0.30,
        )
        figure.suptitle(
            "创新1 K1-AP：共享模型失败是梯度方向冲突，还是梯度幅度失衡",
            x=0.05,
            y=0.973,
            ha="left",
            fontsize=17.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.918,
            (
                "零更新本地审计：复用 K1-AO 两个检查点；每种密码2048/class、4 pairs，"
                "每副本64组32正+32负平衡 batch。"
            ),
            ha="left",
            fontsize=11.1,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.865,
            (
                "结论：方向冲突只在副本0出现，不能跨副本确认；Dialga 梯度幅度却在两个副本都最大，"
                "分别是 Midori 的4.28倍和6.02倍。"
            ),
            ha="left",
            fontsize=11.3,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.818,
            "裁决：开放一种最小梯度归一化对照；不使用 PCGrad、16 pairs、MoE 或远程放大。",
            ha="left",
            fontsize=10.8,
            color="#991B1B",
        )

        _render_correct_norms(axes[0, 0], norm_rows)
        _render_condition_ratios(axes[0, 1], norm_rows)
        _render_pair_cosines(axes[1, 0], pair_rows)
        _render_negative_frequency(axes[1, 1], pair_rows)

        figure.text(
            0.05,
            0.052,
            (
                "下一步 K1-AQ：保持模型、数据、4 pairs、10 epochs、种子和控制完全不变；"
                "只把每个密码的 batch 梯度归一到相同范数后再合并。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.022,
            "判定仍以三密码最小跨密钥 AUC、独立锚点保留和同检查点结构控制为准；本图不是新的训练 AUC。",
            ha="left",
            fontsize=10.1,
            color="#4B5563",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 17.8,
        "height_inches": 10.8,
        "language": "zh-CN",
        "panels": 4,
        "summary_rows": len(rows),
        "pair_summary_rows": len(pair_rows),
        "norm_summary_rows": len(norm_rows),
        "optimizer_steps": 0,
        "training_auc_claim_present": False,
        "formal_scale_claim_present": False,
        "status_from_gate": gate.get("status"),
    }


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    pair_rows = [row for row in rows if row.get("metric_type") == "pairwise_cosine"]
    norm_rows = [row for row in rows if row.get("metric_type") == "gradient_norm"]
    if len(rows) != 72 or len(pair_rows) != 36 or len(norm_rows) != 36:
        raise ValueError("K1-AP plot requires the complete 72-row summary")
    return pair_rows, norm_rows


def _render_correct_norms(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    selected = {
        (int(row["replica"]), str(row["cipher_key"])): float(
            row["median_gradient_norm"]
        )
        for row in rows
        if row["condition"] == "correct_runtime"
        and row["parameter_group"] == "all_trainable"
    }
    labels = [f"{CIPHER_LABELS[cipher]} · 副本{replica}" for replica in (0, 1) for cipher in CIPHER_LABELS]
    values = np.asarray(
        [selected[(replica, cipher)] for replica in (0, 1) for cipher in CIPHER_LABELS]
    )
    colors = ["#0F766E", "#2563EB", "#DC2626"] * 2
    y = np.arange(len(values), dtype=float)
    axis.barh(y, values, height=0.62, color=colors)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, max(values) * 1.18)
    axis.set_xlabel("正确结构的中位全参数梯度范数")
    axis.set_title("梯度幅度：Dialga 在两个副本都最大", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for index, value in enumerate(values):
        axis.text(value + max(values) * 0.018, y[index], f"{value:.2f}", va="center", fontsize=9.0)


def _render_condition_ratios(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    medians = {
        (
            int(row["replica"]),
            str(row["condition"]),
            str(row["cipher_key"]),
        ): float(row["median_gradient_norm"])
        for row in rows
        if row["parameter_group"] == "all_trainable"
    }
    labels = []
    values = []
    colors = []
    palette = {0: "#0F766E", 1: "#7C3AED"}
    for replica in (0, 1):
        for condition in CONDITION_LABELS:
            condition_values = [
                medians[(replica, condition, cipher)] for cipher in CIPHER_LABELS
            ]
            labels.append(f"副本{replica} · {CONDITION_LABELS[condition]}")
            values.append(max(condition_values) / min(condition_values))
            colors.append(palette[replica])
    y = np.arange(len(values), dtype=float)
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(4.0, color="#B45309", linestyle="--", linewidth=1.4)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, max(values) * 1.16)
    axis.set_xlabel("三密码最大/最小中位梯度范数")
    axis.set_title("幅度比例：正确结构跨副本稳定超过门槛", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.text(
        4.08,
        -0.62,
        "预注册门 4.0倍",
        color="#92400E",
        fontsize=9.0,
        va="center",
    )
    for index, value in enumerate(values):
        axis.text(value + max(values) * 0.018, y[index], f"{value:.2f}×", va="center", fontsize=9.0)


def _correct_pair_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row["condition"] == "correct_runtime"
        and row["parameter_group"] == "all_trainable"
    ]
    pair_order = {pair: index for index, pair in enumerate(PAIR_LABELS)}
    return sorted(
        selected,
        key=lambda row: (int(row["replica"]), pair_order[str(row["cipher_pair"])]),
    )


def _render_pair_cosines(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    selected = _correct_pair_rows(rows)
    labels = [
        f"R{int(row['replica'])} · {PAIR_LABELS[str(row['cipher_pair'])]}"
        for row in selected
    ]
    values = np.asarray([float(row["median_cosine"]) for row in selected])
    colors = np.where(values <= -0.05, "#DC2626", "#0F766E")
    y = np.arange(len(values), dtype=float)
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(-0.05, color="#B45309", linestyle="--", linewidth=1.4)
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.85, 0.85)
    axis.set_xlabel("正确结构的中位梯度余弦（负值=方向相反）")
    axis.set_title("梯度方向：副本0冲突，副本1同向", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.text(
        -0.04,
        -0.62,
        "冲突门 -0.05",
        color="#92400E",
        fontsize=9.0,
        va="center",
    )


def _render_negative_frequency(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    selected = _correct_pair_rows(rows)
    labels = [
        f"R{int(row['replica'])} · {PAIR_LABELS[str(row['cipher_pair'])]}"
        for row in selected
    ]
    values = np.asarray(
        [float(row["negative_cosine_frequency"]) for row in selected]
    )
    colors = np.where(values >= 0.50, "#DC2626", "#2563EB")
    y = np.arange(len(values), dtype=float)
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(0.50, color="#B45309", linestyle="--", linewidth=1.4, label="冲突门 50%")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("64个 batch 中梯度余弦为负的比例")
    axis.set_title("负方向频率：没有密码对在两个副本都通过", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False)
    for index, value in enumerate(values):
        axis.text(value + 0.018, y[index], f"{value:.1%}", va="center", fontsize=9.0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ap_svg"]
