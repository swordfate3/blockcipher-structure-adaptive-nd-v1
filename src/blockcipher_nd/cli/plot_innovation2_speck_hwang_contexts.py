from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CONTEXT_LABELS = (
    "00\nPhase C anchor",
    "01\n精确枚举",
    "10\n精确枚举",
    "11\n推导 + 直验",
)
SPLITS = (
    ("discovery", "发现组32把", "#2563EB"),
    ("validation", "验证组32把", "#059669"),
    ("joint", "联合64把", "#DC2626"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Innovation 2 SPECK fixed-context kernel audit."
    )
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    render_context_audit_svg(rows, gate, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "rows": len(rows)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def render_context_audit_svg(
    rows: list[dict[str, Any]], gate: dict[str, Any], output_path: Path
) -> None:
    by_context_round = {
        (str(row["context"]), int(row["rounds"])): row for row in rows
    }
    required = [(context, rounds) for context in ("00", "01", "10", "11") for rounds in (6, 7)]
    if len(by_context_round) != 8 or any(key not in by_context_round for key in required):
        raise ValueError("E26 plot requires four contexts at rounds 6 and 7")
    decision_labels = {
        "innovation2_speck_hwang_context_invariant": (
            "四种固定值共享相同论文 kernel；固定值不是标签变量，下一步扩展固定位置族。"
        ),
        "innovation2_speck_hwang_context_dependent_stable": (
            "固定值产生多个跨密钥稳定 kernel；下一步先做 context/mask 捷径审计。"
        ),
        "innovation2_speck_hwang_context_family_not_stable": (
            "固定值 kernel 未形成稳定不变或多样结构族；停止机械扩展并审计。"
        ),
        "innovation2_speck_hwang_context_protocol_invalid": (
            "baseline、推导、CUDA缓存、计时或GF(2)协议无效；本结果不可裁决。"
        ),
    }
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.4))
        figure.subplots_adjust(
            left=0.07,
            right=0.975,
            top=0.70,
            bottom=0.25,
            wspace=0.25,
        )
        figure.suptitle(
            "创新2 E26：SPECK32/64 四种固定 context 的输出平衡 kernel",
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.895,
            (
                "固定位置均为 bit {5,6}；00复用Phase C，01/10完整枚举2³⁰，"
                "11由置换分区恒等式推导并用一把密钥直接验证。"
            ),
            ha="left",
            va="top",
            fontsize=10.2,
            color="#526070",
        )
        x = np.arange(4, dtype=np.float64)
        width = 0.22
        legend_handles = None
        legend_labels = None
        for axis, rounds, expected_nullity in zip(axes, (6, 7), (9, 1)):
            ordered = [by_context_round[(context, rounds)] for context in ("00", "01", "10", "11")]
            max_nullity = expected_nullity
            for split_index, (split, split_label, color) in enumerate(SPLITS):
                values = [int(row[f"{split}_nullity"]) for row in ordered]
                max_nullity = max(max_nullity, *values)
                bars = axis.bar(
                    x + (split_index - 1) * width,
                    values,
                    width=width,
                    color=color,
                    label=split_label,
                    zorder=3,
                )
                axis.bar_label(
                    bars,
                    labels=[str(value) for value in values],
                    padding=3,
                    fontsize=8.8,
                    color="#334155",
                )
            axis.axhline(
                expected_nullity,
                color="#D97706",
                linestyle="--",
                linewidth=1.4,
                label="论文期望 nullity（各轮见虚线）",
                zorder=1,
            )
            axis.set_title(
                f"{rounds}轮：四种 fixed context",
                loc="left",
                fontweight="bold",
                pad=10,
            )
            axis.set_ylabel("输出平衡 kernel 维数 / nullity")
            axis.set_xticks(x, CONTEXT_LABELS)
            axis.set_ylim(0, max_nullity + max(0.8, max_nullity * 0.18))
            if max_nullity <= 12:
                y_ticks = list(range(max_nullity + 1))
            else:
                y_ticks = sorted(
                    set(int(value) for value in np.linspace(0, max_nullity, 9))
                )
            axis.set_yticks(y_ticks)
            axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
            if legend_handles is None:
                legend_handles, legend_labels = axis.get_legend_handles_labels()
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.07, 0.815),
            ncol=4,
            frameon=False,
            fontsize=9.0,
        )
        metrics = gate.get("metrics", {})
        signatures = metrics.get("distinct_joint_signatures_by_round", {})
        figure.text(
            0.07,
            0.092,
            (
                "joint kernel签名数："
                f"6轮={signatures.get('6', 'NA')}，7轮={signatures.get('7', 'NA')}。 "
                f"裁决：{decision_labels.get(str(gate.get('decision')), str(gate.get('decision')))}"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.07,
            0.04,
            (
                "证据范围：Phase C相同32+32把密钥的paired context审计；"
                "不是神经训练、论文密钥规模、全密钥证明或新积分性质。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#526070",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
