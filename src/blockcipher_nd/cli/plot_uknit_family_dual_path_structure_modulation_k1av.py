from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


CIPHER_ORDER = ("uknit64", "midori64", "dialga128")
CIPHER_LABELS = {
    "uknit64": "uKNIT",
    "midori64": "Midori",
    "dialga128": "Dialga",
}
SPLIT_LABELS = {
    "same_key_fresh": "同密钥",
    "cross_key_validation": "跨密钥",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AV dual-path readiness chart."
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
    report = render_k1av_svg(gate, results, controls, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1av_svg(
    gate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    ordered_results = _ordered_results(results)
    unique_results = _unique_result_rows(ordered_results)
    unique_controls = _unique_control_rows(controls)
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.7,
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
            left=0.07,
            right=0.97,
            top=0.75,
            bottom=0.14,
            hspace=0.62,
            wspace=0.28,
        )
        figure.suptitle(
            "创新1 K1-AV：同一结构摘要已能分别控制 GF(2) 边路径和 S盒转移路径",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=17.0,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.925,
            (
                "零训练 readiness：冻结 K1-AT 两个 epoch-9 检查点；三种密码、两种新样本分割，"
                "每个分割固定检查前32行；训练和优化器更新均为0。"
            ),
            ha="left",
            fontsize=10.5,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.875,
            (
                "兼容性通过：关闭新增 GF(2) 调制后，12/12 面板逐位复现 K1-AT，"
                f"最大 logit 差为 {gate['maximum_disabled_k1at_logit_replay_delta']:.1f}。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#0F766E",
        )
        figure.text(
            0.05,
            0.833,
            (
                f"通道分工通过：GF(2) 最小梯度 {gate['minimum_edge_linear_summary_jacobian_l2']:.6f}，"
                f"S盒最小梯度 {gate['minimum_transition_sbox_summary_jacobian_l2']:.6f}，"
                "跨通道梯度严格为0。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#1D4ED8",
        )
        figure.text(
            0.05,
            0.791,
            "裁决：K1-AV readiness 通过，只授权 K1-AW 同预算训练；这不是新 AUC 或攻击结果。",
            ha="left",
            fontsize=10.3,
            color="#4B5563",
        )

        _render_enabled_response(axes[0, 0], ordered_results)
        _render_component_sensitivity(axes[0, 1], unique_results)
        _render_descriptor_deltas(axes[1, 0], unique_controls)
        _render_output_wiring(axes[1, 1], unique_results)

        figure.text(
            0.05,
            0.055,
            (
                "下一步 K1-AW：保持数据、4 pairs、10 epochs、两个副本和严格负样本不变，"
                "只比较双通道候选与 K1-AT 单标量锚点。"
            ),
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.023,
            "暂不增加16 pairs、样本、宽度或seed；不使用损失重加权、专家、MoE或远程GPU。",
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


def _render_enabled_response(
    axis: Any, rows: Sequence[Mapping[str, Any]]
) -> None:
    labels = [
        f"R{row['replica']} {CIPHER_LABELS[str(row['cipher_key'])]}\n"
        f"{SPLIT_LABELS[str(row['split'])]}"
        for row in rows
    ]
    values = [float(row["enabled_max_abs_logit_delta"]) for row in rows]
    positions = np.arange(len(rows))
    axis.bar(positions, values, color="#0F766E", width=0.72)
    axis.axhline(1e-8, color="#DC2626", linestyle="--", linewidth=1.3)
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, rotation=36, ha="right")
    axis.set_ylabel("开启新 GF(2) 通道后的最大 logit 变化（对数）")
    axis.set_title("新通道不是空接线：12/12 面板都改变输出", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.02,
        0.95,
        "红虚线：可观察门槛 1e-8\n兼容模式回放差：12/12 均为0",
        transform=axis.transAxes,
        va="top",
        fontsize=9.1,
        color="#4B5563",
    )


def _render_component_sensitivity(
    axis: Any, rows: Sequence[Mapping[str, Any]]
) -> None:
    labels = [f"R{row['replica']} {CIPHER_LABELS[str(row['cipher_key'])]}" for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [float(row["edge_linear_summary_jacobian_l2"]) for row in rows],
        width,
        label="GF(2)边通道 对线性摘要",
        color="#2563EB",
    )
    axis.bar(
        positions + width / 2,
        [float(row["transition_sbox_summary_jacobian_l2"]) for row in rows],
        width,
        label="S盒转移通道 对S盒摘要",
        color="#B45309",
    )
    axis.axhline(1e-6, color="#DC2626", linestyle="--", linewidth=1.3)
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel("摘要 Jacobian L2")
    axis.set_title("两个通道都能读取对应结构分量", fontweight="bold")
    axis.legend(loc="upper left", frameon=False)
    axis.grid(axis="y", alpha=0.25)


def _render_descriptor_deltas(
    axis: Any, rows: Sequence[Mapping[str, Any]]
) -> None:
    labels = [f"R{row['replica']} {CIPHER_LABELS[str(row['cipher_key'])]}" for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [float(row["linear_edge_delta"]) for row in rows],
        width,
        label="换线性层摘要 → GF(2)边门控",
        color="#2563EB",
    )
    axis.bar(
        positions + width / 2,
        [float(row["sbox_transition_delta"]) for row in rows],
        width,
        label="换S盒摘要 → S盒转移门控",
        color="#B45309",
    )
    axis.axhline(1e-6, color="#DC2626", linestyle="--", linewidth=1.3)
    axis.set_yscale("log")
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel("相关门控变化（对数）")
    axis.set_title("错误结构控制可观察：6/6 面板通过", fontweight="bold")
    axis.set_ylim(6e-7, 8e-2)
    axis.legend(loc="upper center", frameon=False)
    axis.grid(axis="y", alpha=0.25)


def _render_output_wiring(axis: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [f"R{row['replica']} {CIPHER_LABELS[str(row['cipher_key'])]}" for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [float(row["edge_own_row_parameter_jacobian_l2"]) for row in rows],
        width,
        label="GF(2)门控 → 输出行0",
        color="#2563EB",
    )
    axis.bar(
        positions + width / 2,
        [float(row["transition_own_row_parameter_jacobian_l2"]) for row in rows],
        width,
        label="S盒门控 → 输出行1",
        color="#B45309",
    )
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel("自身输出行参数 Jacobian L2")
    axis.set_title("路径接线独立：自身非零，跨通道严格为0", fontweight="bold")
    axis.set_ylim(0.0, 1.9)
    axis.legend(loc="upper right", frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.02,
        0.96,
        "所有 GF(2)→行1 与 S盒→行0 梯度均为 0.0",
        transform=axis.transAxes,
        va="top",
        fontsize=9.2,
        color="#0F766E",
        fontweight="bold",
    )


def _ordered_results(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if len(rows) != 12:
        raise ValueError("K1-AV plot requires twelve result rows")
    by_key = {
        (int(row["replica"]), str(row["cipher_key"]), str(row["split"])): row
        for row in rows
    }
    return [
        by_key[(replica, cipher, split)]
        for replica in (0, 1)
        for cipher in CIPHER_ORDER
        for split in ("same_key_fresh", "cross_key_validation")
    ]


def _unique_result_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    unique = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]))
        current = unique.setdefault(key, row)
        for field in (
            "edge_linear_summary_jacobian_l2",
            "transition_sbox_summary_jacobian_l2",
            "edge_own_row_parameter_jacobian_l2",
            "transition_own_row_parameter_jacobian_l2",
        ):
            if float(current[field]) != float(row[field]):
                raise ValueError("K1-AV split-independent result metric drifted")
    return [unique[(replica, cipher)] for replica in (0, 1) for cipher in CIPHER_ORDER]


def _unique_control_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != 36:
        raise ValueError("K1-AV plot requires thirty-six control rows")
    grouped: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]))
        condition = str(row["condition"])
        current = grouped.setdefault(key, {}).setdefault(condition, row)
        for field in ("edge_gate_delta", "transition_gate_delta"):
            if float(current[field]) != float(row[field]):
                raise ValueError("K1-AV split-independent control metric drifted")
    result = []
    for replica in (0, 1):
        for cipher in CIPHER_ORDER:
            controls = grouped[(replica, cipher)]
            result.append(
                {
                    "replica": replica,
                    "cipher_key": cipher,
                    "linear_edge_delta": float(
                        controls["linear_only_mismatch"]["edge_gate_delta"]
                    ),
                    "sbox_transition_delta": float(
                        controls["sbox_only_mismatch"]["transition_gate_delta"]
                    ),
                }
            )
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = ["render_k1av_svg"]
