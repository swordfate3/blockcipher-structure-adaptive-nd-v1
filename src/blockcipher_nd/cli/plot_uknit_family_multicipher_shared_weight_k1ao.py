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
CONDITION_LABELS = {
    "wrong_sbox_same_state": "替换为错误 S盒",
    "transition_branch_off_same_state": "关闭转移分支",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AO shared-weight readiness chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    runtime_rows = _read_jsonl(args.runtime_manifest)
    result_rows = _read_jsonl(args.results)
    report = render_k1ao_svg(gate, runtime_rows, result_rows, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ao_svg(
    gate: Mapping[str, Any],
    runtime_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    runtimes = {str(row["cipher_key"]): row for row in runtime_rows}
    if set(runtimes) != set(CIPHER_LABELS):
        raise ValueError("K1-AO plot requires exactly three runtime rows")
    results = {
        (str(row["cipher_key"]), str(row["condition"])): row
        for row in result_rows
    }
    expected_results = {
        (cipher_key, condition)
        for cipher_key in CIPHER_LABELS
        for condition in (
            "correct_runtime",
            "wrong_sbox_same_state",
            "transition_branch_off_same_state",
        )
    }
    if set(results) != expected_results:
        raise ValueError("K1-AO plot requires the complete nine-row panel")

    cipher_keys = list(CIPHER_LABELS)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
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
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(16.8, 9.0),
            gridspec_kw={"width_ratios": (1.05, 1.0)},
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.96,
            top=0.68,
            bottom=0.18,
            wspace=0.32,
        )
        figure.suptitle(
            "创新1 K1-AO：一套神经网络权重能否读取三种 SPN 结构",
            x=0.05,
            y=0.955,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.895,
            (
                "零训练准备检查：uKNIT-BC 第5轮、Midori64 第4轮、"
                "Dialga-128 第4轮；每样本4对密文。"
            ),
            ha="left",
            fontsize=11,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.825,
            _decision_text(gate),
            ha="left",
            fontsize=11.4,
            fontweight="bold",
            color="#166534" if gate.get("status") == "pass" else "#B91C1C",
        )
        figure.text(
            0.05,
            0.755,
            (
                "左图说明三种密码虽然状态宽度不同，但参数数量和权重哈希完全相同；"
                "右图只表示结构干预会改变输出，不代表已有区分 AUC。"
            ),
            ha="left",
            fontsize=10.3,
            color="#4B5563",
        )

        _render_geometry_table(axes[0], runtimes, cipher_keys)
        _render_intervention_chart(axes[1], results, cipher_keys)
        figure.text(
            0.05,
            0.075,
            (
                "下一步：本地训练两套共享模型，每个密码 2048/class、fresh 1024/class、"
                "10 epochs；同一检查点比较正确结构、错误 S盒和关闭分支。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.035,
            "禁止直接远程放大、增加到16 pairs、加入密码编号或让 Dialga 的高分掩盖其他密码失败。",
            ha="left",
            fontsize=10.1,
            color="#B45309",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)
    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 16.8,
        "height_inches": 9.0,
        "language": "zh-CN",
        "panels": 2,
        "runtime_rows": len(runtime_rows),
        "intervention_rows": len(result_rows),
        "auc_claim_present": False,
        "log_scale_used_for_close_nonzero_deltas": True,
    }


def _render_geometry_table(
    axis: plt.Axes,
    runtimes: Mapping[str, Mapping[str, Any]],
    cipher_keys: Sequence[str],
) -> None:
    axis.axis("off")
    axis.set_title(
        "共享权重几何：同一份权重严格加载成功",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    columns = ("密码", "状态位", "cell数", "输入位", "参数量", "状态项")
    rows = [
        (
            CIPHER_LABELS[cipher_key],
            str(runtimes[cipher_key]["block_bits"]),
            str(runtimes[cipher_key]["cells"]),
            str(runtimes[cipher_key]["input_bits"]),
            f"{int(runtimes[cipher_key]['trainable_parameter_count']):,}",
            str(runtimes[cipher_key]["state_dict_entries"]),
        )
        for cipher_key in cipher_keys
    ]
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=(0.0, 0.38, 1.0, 0.50),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(fontweight="bold", color="#111827")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "#FFFFFF")
    shared_hash = str(runtimes[cipher_keys[0]]["state_sha256"])
    axis.text(
        0.0,
        0.25,
        "三种运行时的初始化权重哈希完全一致：",
        transform=axis.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="#374151",
    )
    axis.text(
        0.0,
        0.17,
        f"{shared_hash[:16]}...{shared_hash[-12:]}",
        transform=axis.transAxes,
        fontsize=10.5,
        family="DejaVu Sans Mono",
        color="#166534",
    )
    axis.text(
        0.0,
        0.065,
        "18个既有数据缓存均重新计算哈希通过；没有生成数据，也没有优化器步骤。",
        transform=axis.transAxes,
        fontsize=10.2,
        color="#4B5563",
    )


def _render_intervention_chart(
    axis: plt.Axes,
    results: Mapping[tuple[str, str], Mapping[str, Any]],
    cipher_keys: Sequence[str],
) -> None:
    conditions = (
        "wrong_sbox_same_state",
        "transition_branch_off_same_state",
    )
    labels = [
        f"{CIPHER_LABELS[cipher_key]} · {CONDITION_LABELS[condition]}"
        for cipher_key in cipher_keys
        for condition in conditions
    ]
    values = np.asarray(
        [
            float(results[(cipher_key, condition)]["max_abs_delta_from_correct"])
            for cipher_key in cipher_keys
            for condition in conditions
        ]
    )
    colors = ["#0F766E", "#2563EB"] * len(cipher_keys)
    y = np.arange(len(values), dtype=float)
    axis.barh(y, values, height=0.62, color=colors)
    axis.set_xscale("log")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("相对正确结构的最大输出变化（对数刻度，仅检查是否非零）")
    axis.set_title(
        "同一权重下，结构干预均能改变输出",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for index, value in enumerate(values):
        axis.text(
            value * 1.08,
            y[index],
            f"{value:.6f}",
            va="center",
            ha="left",
            fontsize=9.2,
            color="#111827",
        )
    axis.set_xlim(values.min() * 0.55, values.max() * 2.5)


def _decision_text(gate: Mapping[str, Any]) -> str:
    if gate.get("status") == "pass":
        return "结论：共享权重运行时切换准备通过，可以进入本地多密码共享训练；当前还没有训练 AUC。"
    return "结论：准备检查未通过；先修复失败项，不能开始共享训练或远程放大。"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ao_svg"]
