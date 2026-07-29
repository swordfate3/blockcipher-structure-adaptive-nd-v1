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
SHORT_CIPHER_LABELS = {
    "uknit64": "uKNIT",
    "midori64": "Midori",
    "dialga128": "Dialga",
}
CONDITION_LABELS = {
    "full_mismatch": "完整",
    "sbox_only_mismatch": "S盒",
    "linear_only_mismatch": "线性层",
}
CIPHER_ORDER = ("uknit64", "midori64", "dialga128")
CONDITION_ORDER = (
    "full_mismatch",
    "sbox_only_mismatch",
    "linear_only_mismatch",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AU layerwise identifiability chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    results = _read_jsonl(args.results)
    controls = _read_jsonl(args.controls)
    report = render_k1au_svg(gate, results, controls, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1au_svg(
    gate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    result_rows = _ordered_results(results)
    control_rows = _unique_controls(controls)
    jacobian_cosines = gate["cross_replica_jacobian_cosine_by_cipher"]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.8,
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
        figure, axes = plt.subplots(2, 2, figsize=(18.0, 12.0))
        figure.subplots_adjust(
            left=0.11,
            right=0.97,
            top=0.76,
            bottom=0.13,
            hspace=0.60,
            wspace=0.34,
        )
        figure.suptitle(
            "创新1 K1-AU：结构信息保留到隐藏层，但单标量门控的密码排序不稳定",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=17.2,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.925,
            (
                "零训练分层审计：冻结K1-AT两个epoch-9检查点；三种密码、两种新样本分割，"
                "每个分割只检查固定前32行；训练和优化器更新均为0。"
            ),
            ha="left",
            fontsize=10.6,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.877,
            (
                f"信息保留：原始摘要最小距离 {gate['minimum_raw_summary_l2_distance']:.6f}，"
                f"隐藏层最小距离 {gate['minimum_hidden_l2_distance']:.6f}；"
                "两副本的三密码隐藏表示秩均为2。"
            ),
            ha="left",
            fontsize=10.9,
            fontweight="bold",
            color="#0F766E",
        )
        figure.text(
            0.05,
            0.835,
            (
                f"标量不稳定：18/18个错配方向与最终权重有数值对齐，但两副本的密码门控排序"
                f"相关系数为 {gate['gate_rank_correlation']:.1f}（要求1.0）。"
            ),
            ha="left",
            fontsize=10.9,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.795,
            "裁决：瓶颈位于共享隐藏表示之后的单标量映射；这不是新AUC、正式规模或攻击结果。",
            ha="left",
            fontsize=10.4,
            color="#4B5563",
        )

        _render_layer_distances(axes[0, 0], control_rows)
        _render_gate_ordering(axes[0, 1], result_rows, gate)
        _render_projection_alignment(axes[1, 0], control_rows)
        _render_component_jacobians(axes[1, 1], result_rows, jacobian_cosines)

        figure.text(
            0.05,
            0.052,
            (
                "下一步 K1-AV：只做readiness，把同一运行时摘要映射到现有GF(2)边残差与"
                "S盒转移残差两个有界通道；先通过关闭调制精确重放和错配可观察性，再允许训练。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.021,
            "不增加pairs、样本、epoch、宽度或seed；不使用损失重加权、专家、MoE或远程GPU。",
            ha="left",
            fontsize=9.9,
            color="#4B5563",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)

    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 18.0,
        "height_inches": 12.0,
        "language": "zh-CN",
        "panels": 4,
        "result_rows": len(results),
        "control_rows": len(controls),
        "unique_control_panels": len(control_rows),
        "status_from_gate": gate.get("status"),
        "auc_claim_present": False,
        "formal_scale_claim_present": False,
    }


def _ordered_results(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if len(rows) != 6:
        raise ValueError("K1-AU plot requires six result rows")
    by_key = {(int(row["replica"]), str(row["cipher_key"])): row for row in rows}
    expected = {(replica, cipher) for replica in (0, 1) for cipher in CIPHER_ORDER}
    if set(by_key) != expected:
        raise ValueError("K1-AU result panel binding is incomplete")
    return [by_key[(replica, cipher)] for replica in (0, 1) for cipher in CIPHER_ORDER]


def _unique_controls(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if len(rows) != 36:
        raise ValueError("K1-AU plot requires 36 control rows")
    by_key: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["condition"]))
        current = by_key.setdefault(key, row)
        for field in (
            "raw_summary_l2_distance",
            "hidden_l2_distance",
            "projection_alignment_abs_cosine",
            "effective_gate_delta",
        ):
            if float(current[field]) != float(row[field]):
                raise ValueError("K1-AU split-independent control metric drifted")
    expected = {
        (replica, cipher, condition)
        for replica in (0, 1)
        for cipher in CIPHER_ORDER
        for condition in CONDITION_ORDER
    }
    if set(by_key) != expected:
        raise ValueError("K1-AU unique control panel binding is incomplete")
    return [
        by_key[(replica, cipher, condition)]
        for replica in (0, 1)
        for cipher in CIPHER_ORDER
        for condition in CONDITION_ORDER
    ]


def _control_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"R{row['replica']} {SHORT_CIPHER_LABELS[str(row['cipher_key'])]} · "
        f"{CONDITION_LABELS[str(row['condition'])]}"
        for row in rows
    ]


def _render_layer_distances(axis: plt.Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = _control_labels(rows)
    y = np.arange(len(rows), dtype=float)
    raw_ratio = np.asarray(
        [float(row["raw_summary_l2_distance"]) / 1e-3 for row in rows]
    )
    hidden_ratio = np.asarray([float(row["hidden_l2_distance"]) / 1e-4 for row in rows])
    axis.scatter(
        raw_ratio, y - 0.14, color="#2563EB", marker="s", s=28, label="34维原始摘要"
    )
    axis.scatter(
        hidden_ratio, y + 0.14, color="#0F766E", marker="o", s=28, label="12维隐藏表示"
    )
    axis.axvline(
        1.0, color="#DC2626", linestyle="--", linewidth=1.3, label="各层通过线"
    )
    axis.set_xscale("log")
    axis.set_yticks(y, labels, fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlim(0.7, 25000)
    axis.set_xlabel("错配距离 / 该层门槛（对数坐标，大于1通过）")
    axis.set_title(
        "信息是否保留：18/18错配在两层都可区分", loc="left", fontweight="bold"
    )
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        fontsize=8.4,
        ncol=3,
    )


def _render_gate_ordering(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> None:
    by_key = {(int(row["replica"]), str(row["cipher_key"])): row for row in rows}
    x = np.arange(len(CIPHER_ORDER), dtype=float)
    for replica, color, marker in ((0, "#0F766E", "o"), (1, "#B45309", "s")):
        values = [
            float(by_key[(replica, cipher)]["effective_gate"])
            for cipher in CIPHER_ORDER
        ]
        axis.plot(
            x,
            values,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            label=f"副本{replica}",
        )
        for index, value in enumerate(values):
            axis.text(index, value + 0.0025, f"{value:.3f}", ha="center", fontsize=8.4)
    axis.set_xticks(x, [CIPHER_LABELS[cipher] for cipher in CIPHER_ORDER])
    axis.set_ylim(0.145, 0.222)
    axis.set_ylabel("正确描述符的有效转移门控")
    axis.set_title(
        "排序稳定性：两个副本学出不同密码顺序", loc="left", fontweight="bold"
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", frameon=False, ncol=2)
    axis.text(
        0.03,
        0.07,
        f"Spearman ρ = {gate['gate_rank_correlation']:.1f}（要求 1.0）",
        transform=axis.transAxes,
        fontsize=10.0,
        fontweight="bold",
        color="#991B1B",
    )


def _render_projection_alignment(
    axis: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    labels = _control_labels(rows)
    values = np.asarray([float(row["projection_alignment_abs_cosine"]) for row in rows])
    y = np.arange(len(rows), dtype=float)
    colors = np.where(values >= 0.1, "#0F766E", "#DC2626")
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(
        0.1, color="#7C3AED", linestyle="--", linewidth=1.3, label="通过线 0.1"
    )
    axis.set_yticks(y, labels, fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 0.62)
    axis.set_xlabel("|cos(隐藏错配方向, 最终投影权重)|")
    axis.set_title(
        "最终投影有响应：18/18方向通过数值对齐门", loc="left", fontweight="bold"
    )
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, fontsize=8.6)


def _render_component_jacobians(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    jacobian_cosines: Mapping[str, Any],
) -> None:
    labels = [
        f"R{row['replica']} {SHORT_CIPHER_LABELS[str(row['cipher_key'])]}"
        for row in rows
    ]
    x = np.arange(len(rows), dtype=float)
    width = 0.34
    sbox = np.asarray([float(row["sbox_jacobian_l2"]) for row in rows])
    linear = np.asarray([float(row["linear_jacobian_l2"]) for row in rows])
    axis.bar(x - width / 2, sbox, width, color="#B45309", label="S盒16维雅可比")
    axis.bar(x + width / 2, linear, width, color="#2563EB", label="GF(2) 18维雅可比")
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 0.052)
    axis.set_ylabel("有效门控对摘要分量的梯度 L2")
    axis.set_title(
        "两个结构分量都可影响门控，跨副本方向仅中等一致", loc="left", fontweight="bold"
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False, fontsize=8.7)
    cosine_text = "  ".join(
        f"{SHORT_CIPHER_LABELS[cipher]} {float(jacobian_cosines[cipher]):.3f}"
        for cipher in CIPHER_ORDER
    )
    axis.text(
        0.98,
        0.94,
        f"跨副本雅可比余弦：{cosine_text}",
        transform=axis.transAxes,
        fontsize=8.8,
        color="#4B5563",
        ha="right",
        va="top",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1au_svg"]
