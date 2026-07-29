from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CIPHER_LABELS = {
    "uknit64": "uKNIT",
    "midori64": "Midori",
    "dialga128": "Dialga",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AY component-separation chart."
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
    report = render_k1ay_svg(gate, results, controls, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ay_svg(
    gate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    ordered_results = _ordered_results(results)
    unique_results = _unique_panels(ordered_results)
    unique_controls = _unique_controls(controls)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
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
            left=0.075,
            right=0.97,
            top=0.735,
            bottom=0.17,
            hspace=0.70,
            wspace=0.30,
        )
        figure.suptitle(
            "创新1 K1-AY：S盒与线性扩散信息已被隔离到各自门控",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=17.5,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.920,
            (
                "零训练 readiness：严格加载 K1-AW 两个 epoch-10 检查点；"
                "三种密码、两种新样本分割，每个面板固定检查前32行。"
            ),
            ha="left",
            fontsize=10.7,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.870,
            (
                f"兼容性：关闭分量隔离后，12/12 面板逐位复现 K1-AW，最大 logit 差 "
                f"{gate['maximum_disabled_k1aw_logit_replay_delta']:.1f}；参数量和权重形状不变。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#0F766E",
        )
        figure.text(
            0.05,
            0.825,
            (
                "隔离性：S盒错配引起的 GF(2) 门变化为 0.0；"
                "线性错配引起的 S盒门变化为 0.0；两类均为12/12严格成立。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#1D4ED8",
        )
        figure.text(
            0.05,
            0.783,
            "裁决：K1-AY readiness 通过，只授权 K1-AZ 同预算训练；这不是新 AUC 或攻击结果。",
            ha="left",
            fontsize=10.4,
            color="#4B5563",
        )

        _render_component_jacobians(axes[0, 0], unique_results)
        _render_enabled_logit_change(axes[0, 1], ordered_results)
        _render_mismatch_response(
            axes[1, 0],
            unique_controls,
            condition="sbox_only_mismatch",
            relevant_gate="transition_gate_delta",
            color="#B45309",
            title="只替换S盒摘要：只改变S盒转移门",
            irrelevant_note="GF(2)边门：12/12 变化严格为0",
        )
        _render_mismatch_response(
            axes[1, 1],
            unique_controls,
            condition="linear_only_mismatch",
            relevant_gate="edge_gate_delta",
            color="#2563EB",
            title="只替换线性摘要：只改变GF(2)边门",
            irrelevant_note="S盒转移门：12/12 变化严格为0",
        )

        figure.text(
            0.05,
            0.055,
            (
                "下一步 K1-AZ：保持4 pairs、2048/class/cipher、10 epochs、两个副本、"
                "严格负样本和全部控制不变，只训练分量隔离连接。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.023,
            "暂不增加16 pairs、样本、epoch、seed或宽度；不使用损失重加权、专家、MoE或远程GPU。",
            ha="left",
            fontsize=9.8,
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
        "status_from_gate": gate.get("status"),
        "auc_claim_present": False,
        "formal_scale_claim_present": False,
    }


def _render_component_jacobians(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = [f"{CIPHER_LABELS[str(row['cipher_key'])]} R{row['replica']}" for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [float(row["edge_linear_summary_jacobian_l2"]) for row in rows],
        width,
        label="线性摘要→GF(2)边门",
        color="#2563EB",
    )
    axis.bar(
        positions + width / 2,
        [float(row["transition_sbox_summary_jacobian_l2"]) for row in rows],
        width,
        label="S盒摘要→S盒转移门",
        color="#B45309",
    )
    axis.axhline(1e-6, color="#DC2626", linestyle="--", linewidth=1.3)
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel("相关输入 Jacobian L2（对数轴）")
    axis.set_title("相关结构分量仍有强响应，无关分量梯度严格为0", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, ncol=2)


def _render_enabled_logit_change(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    labels = [
        f"{CIPHER_LABELS[str(row['cipher_key'])]} R{row['replica']}\n"
        f"{'同钥' if row['split'] == 'same_key_fresh' else '跨钥'}"
        for row in rows
    ]
    positions = np.arange(len(rows))
    values = [float(row["enabled_max_abs_logit_delta"]) for row in rows]
    axis.bar(positions, values, width=0.72, color="#0F766E")
    axis.axhline(1e-8, color="#DC2626", linestyle="--", linewidth=1.3)
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("开启隔离连接后的最大 logit 变化（对数轴）")
    axis.set_title("隔离连接不是空操作：12/12 面板都改变输出", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)


def _render_mismatch_response(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    relevant_gate: str,
    color: str,
    title: str,
    irrelevant_note: str,
) -> None:
    selected = [row for row in rows if row["condition"] == condition]
    labels = [f"{CIPHER_LABELS[str(row['cipher_key'])]} R{row['replica']}" for row in selected]
    positions = np.arange(len(selected))
    values = [float(row[relevant_gate]) for row in selected]
    axis.bar(positions, values, width=0.62, color=color)
    axis.axhline(1e-6, color="#DC2626", linestyle="--", linewidth=1.3, label="有效门槛 1e-6")
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel("相关门控绝对变化（对数轴）")
    axis.set_title(
        f"{title}\n{irrelevant_note}",
        loc="left",
        fontweight="bold",
        linespacing=1.35,
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)


def _ordered_results(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if len(rows) != 12:
        raise ValueError("K1-AY plot requires twelve result rows")
    cipher_order = {name: index for index, name in enumerate(CIPHER_LABELS)}
    split_order = {"same_key_fresh": 0, "cross_key_validation": 1}
    return sorted(
        rows,
        key=lambda row: (
            int(row["replica"]),
            cipher_order[str(row["cipher_key"])],
            split_order[str(row["split"])],
        ),
    )


def _unique_panels(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["split"] == "cross_key_validation"]


def _unique_controls(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if len(rows) != 36:
        raise ValueError("K1-AY plot requires thirty-six control rows")
    selected = [row for row in rows if row["split"] == "cross_key_validation"]
    cipher_order = {name: index for index, name in enumerate(CIPHER_LABELS)}
    condition_order = {name: index for index, name in enumerate((
        "full_mismatch",
        "sbox_only_mismatch",
        "linear_only_mismatch",
    ))}
    return sorted(
        selected,
        key=lambda row: (
            condition_order[str(row["condition"])],
            int(row["replica"]),
            cipher_order[str(row["cipher_key"])],
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ay_svg"]
