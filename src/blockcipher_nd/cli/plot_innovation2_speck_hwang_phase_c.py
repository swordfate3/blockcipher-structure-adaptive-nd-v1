from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


CHECK_ROWS = (
    ("6轮 joint rank/nullity = 23/9", "anchor_r6_joint_rank_nullity_23_9"),
    ("6轮 joint kernel = 论文九维空间", "anchor_r6_joint_kernel_equals_hwang_span"),
    (
        "6轮论文方向在两组新密钥均成立",
        "anchor_r6_hwang_directions_valid_both_key_halves",
    ),
    ("7轮 joint rank/nullity = 31/1", "anchor_r7_joint_rank_nullity_31_1"),
    ("7轮 joint kernel = 论文一维空间", "anchor_r7_joint_kernel_equals_hwang_span"),
    (
        "7轮论文方向在两组新密钥均成立",
        "anchor_r7_hwang_direction_valid_both_key_halves",
    ),
    ("位置控制不包含7轮论文 mask", "control_r7_does_not_contain_hwang_mask"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Innovation 2 SPECK32/64 Phase C kernel result chart."
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
    render_phase_c_svg(rows, gate, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "rows": len(rows)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def render_phase_c_svg(
    rows: list[dict[str, Any]], gate: dict[str, Any], output_path: Path
) -> None:
    by_role_round = {(str(row["role"]), int(row["rounds"])): row for row in rows}
    expected_keys = (("anchor", 6), ("anchor", 7), ("control", 7))
    if len(by_role_round) != 3 or any(key not in by_role_round for key in expected_keys):
        raise ValueError("Phase C plot requires anchor r6/r7 and control r7 rows")
    ordered = [by_role_round[key] for key in expected_keys]
    labels = (
        "6轮论文结构\n固定 bit {5,6}",
        "7轮论文结构\n固定 bit {5,6}",
        "7轮位置控制\n固定 bit {0,1}",
    )
    splits = (
        ("discovery", "发现组32把", "#2563EB"),
        ("validation", "验证组32把", "#059669"),
        ("joint", "联合64把", "#DC2626"),
    )
    x = np.arange(3, dtype=np.float64)
    width = 0.22
    signal_checks = gate.get("signal_checks", {})
    check_values = [bool(signal_checks.get(key, False)) for _, key in CHECK_ROWS]
    decision_labels = {
        "innovation2_speck_hwang_phase_c_kernel_reproduced": (
            "论文 kernel 在32+32把新密钥上精确复现，位置控制通过；进入固定上下文多样性审计。"
        ),
        "innovation2_speck_hwang_phase_c_position_control_not_specific": (
            "论文 mask 在位置控制中也成立，结构特异性不足；禁止构造神经标签。"
        ),
        "innovation2_speck_hwang_phase_c_kernel_not_reproduced": (
            "论文 kernel 未在联合64把密钥上精确复现；停止机械增加密钥并审计协议。"
        ),
        "innovation2_speck_hwang_phase_c_protocol_invalid": (
            "密钥、结构、CUDA缓存、计时或GF(2)协议无效；本结果不可用于研究裁决。"
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
        figure, (nullity_axis, check_axis) = plt.subplots(
            1,
            2,
            figsize=(15.5, 7.4),
            gridspec_kw={"width_ratios": [1.55, 1.25]},
        )
        figure.subplots_adjust(
            left=0.075,
            right=0.975,
            top=0.70,
            bottom=0.24,
            wspace=0.34,
        )
        figure.suptitle(
            "创新2 E25 Phase C：SPECK32/64 输出平衡 kernel 复现",
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            (
                "每行完整枚举2³⁰个明文；32把发现密钥 + 32把全新验证密钥；"
                "控制只把固定位置从{5,6}移到{0,1}。"
            ),
            ha="left",
            va="top",
            fontsize=10.2,
            color="#526070",
        )

        max_nullity = 9
        for split_index, (split, split_label, color) in enumerate(splits):
            values = [int(row[f"{split}_nullity"]) for row in ordered]
            max_nullity = max(max_nullity, *values)
            positions = x + (split_index - 1) * width
            bars = nullity_axis.bar(
                positions,
                values,
                width=width,
                color=color,
                label=split_label,
                zorder=3,
            )
            nullity_axis.bar_label(
                bars,
                labels=[str(value) for value in values],
                padding=3,
                fontsize=8.8,
                color="#334155",
            )
        nullity_axis.scatter(
            x[:2],
            [9, 1],
            marker="D",
            s=58,
            color="#D97706",
            edgecolor="#FFFFFF",
            linewidth=0.7,
            label="论文期望 joint nullity",
            zorder=5,
        )
        nullity_axis.set_title(
            "经验 kernel 维数（nullity）",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        nullity_axis.set_ylabel("输出平衡 kernel 维数")
        nullity_axis.set_xticks(x, labels)
        nullity_axis.set_ylim(0, max_nullity + max(2.0, max_nullity * 0.18))
        if max_nullity <= 12:
            y_ticks = list(range(0, max_nullity + 1))
        else:
            y_ticks = sorted(
                set(int(value) for value in np.linspace(0, max_nullity, 9))
            )
        nullity_axis.set_yticks(y_ticks)
        nullity_axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
        handles, legend_labels = nullity_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.075, 0.82),
            ncol=4,
            frameon=False,
            fontsize=9.0,
        )

        matrix = np.asarray(check_values, dtype=np.int8).reshape(-1, 1)
        check_axis.imshow(
            matrix,
            cmap=ListedColormap(["#FEE2E2", "#DCFCE7"]),
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        check_axis.set_title(
            "预注册裁决门",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        check_axis.set_xticks([])
        check_axis.set_yticks(
            np.arange(len(CHECK_ROWS)), [label for label, _ in CHECK_ROWS]
        )
        check_axis.tick_params(axis="y", length=0, pad=8, labelsize=9.0)
        for row_index, passed in enumerate(check_values):
            check_axis.text(
                0,
                row_index,
                "通过" if passed else "未通过",
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold",
                color="#166534" if passed else "#991B1B",
            )
        for spine in check_axis.spines.values():
            spine.set_visible(False)

        decision = str(gate.get("decision", "unknown"))
        figure.text(
            0.075,
            0.085,
            f"裁决：{decision_labels.get(decision, decision)}",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        figure.text(
            0.075,
            0.038,
            (
                "证据范围：完整2³⁰结构的64把 sampled-key 复现与一个同预算位置控制；"
                "不是论文规模、全密钥证明、神经训练或新积分性质。"
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
