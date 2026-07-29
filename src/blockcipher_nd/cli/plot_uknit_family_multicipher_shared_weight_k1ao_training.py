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
SPLIT_LABELS = {
    "same_key_fresh": "同密钥新样本",
    "cross_key_validation": "跨密钥新样本",
}
SHORT_CIPHER_LABELS = {
    "uknit64": "uKNIT",
    "midori64": "Midori",
    "dialga128": "Dialga",
}
SHORT_SPLIT_LABELS = {
    "same_key_fresh": "同密钥",
    "cross_key_validation": "跨密钥",
}
CONDITIONS = {
    "correct_runtime",
    "wrong_sbox_same_checkpoint",
    "transition_branch_off_same_checkpoint",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Chinese K1-AO shared-training result chart."
    )
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.controls)
    report = render_k1ao_training_svg(gate, rows, args.output)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def render_k1ao_training_svg(
    gate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> dict[str, Any]:
    panels = _collect_panels(rows)
    cross_key = [
        panel for panel in panels if panel["split"] == "cross_key_validation"
    ]
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.2,
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
        figure, axes = plt.subplots(2, 2, figsize=(18.0, 11.2))
        figure.subplots_adjust(
            left=0.085,
            right=0.97,
            top=0.79,
            bottom=0.14,
            hspace=0.48,
            wspace=0.31,
        )
        figure.suptitle(
            "创新1 K1-AO：三种 SPN 共用一套权重后的真实训练结果",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=18,
            fontweight="bold",
        )
        figure.text(
            0.05,
            0.925,
            (
                "本地诊断：每种密码 2048/class、每样本4对密文、10 epochs、"
                "2个独立初始化；不是正式规模或主流攻击对比。"
            ),
            ha="left",
            fontsize=11.2,
            color="#4B5563",
        )
        figure.text(
            0.05,
            0.875,
            (
                "结论：正确 S盒在12/12面板优于错误 S盒，但 uKNIT/Midori 未保留独立模型强度；"
                "Midori 还有1个面板关闭结构分支更好。"
            ),
            ha="left",
            fontsize=11.3,
            fontweight="bold",
            color="#B45309",
        )
        figure.text(
            0.05,
            0.832,
            "裁决：当前共享训练暂缓，不增加 pairs/样本/epoch；先检查三种密码的梯度是否互相冲突。",
            ha="left",
            fontsize=10.8,
            color="#991B1B",
        )

        _render_cross_key_anchor(axes[0, 0], cross_key)
        _render_cross_key_controls(axes[0, 1], cross_key)
        _render_retention_margins(axes[1, 0], panels)
        _render_semantic_margins(axes[1, 1], panels)

        figure.text(
            0.05,
            0.055,
            (
                "下一步 K1-AP：冻结这两个检查点，在相同缓存与64组平衡 batch 上测量三密码梯度夹角、"
                "冲突频率和梯度范数；0次参数更新。"
            ),
            ha="left",
            fontsize=10.8,
            fontweight="bold",
            color="#1F2937",
        )
        figure.text(
            0.05,
            0.025,
            "只有确认梯度竞争后，才比较一种最小冲突缓解方法；若不存在冲突，则返回结构表示设计。",
            ha="left",
            fontsize=10.2,
            color="#4B5563",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg")
        plt.close(figure)

    return {
        "status": "rendered_pending_visual_qa",
        "figure": str(output),
        "width_inches": 18.0,
        "height_inches": 11.2,
        "language": "zh-CN",
        "panels": 4,
        "evaluation_rows": len(rows),
        "comparison_panels": len(panels),
        "status_from_gate": gate.get("status"),
        "auc_claim_present": True,
        "formal_scale_claim_present": False,
    }


def _collect_panels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["replica"]), str(row["cipher_key"]), str(row["split"]))
        grouped.setdefault(key, {})[str(row["condition"])] = row
    if len(rows) != 36 or len(grouped) != 12:
        raise ValueError("K1-AO training plot requires 36 rows in 12 panels")
    if any(set(conditions) != CONDITIONS for conditions in grouped.values()):
        raise ValueError("K1-AO training plot has an incomplete condition panel")

    panels = []
    cipher_order = {key: index for index, key in enumerate(CIPHER_LABELS)}
    split_order = {key: index for index, key in enumerate(SPLIT_LABELS)}
    for (replica, cipher_key, split), conditions in grouped.items():
        correct = conditions["correct_runtime"]
        wrong = conditions["wrong_sbox_same_checkpoint"]
        branch = conditions["transition_branch_off_same_checkpoint"]
        correct_auc = float(correct["auc"])
        anchor_auc = float(correct["anchor_auc"])
        wrong_auc = float(wrong["auc"])
        branch_auc = float(branch["auc"])
        panels.append(
            {
                "replica": replica,
                "cipher_key": cipher_key,
                "split": split,
                "label": f"{CIPHER_LABELS[cipher_key]} · 副本{replica}",
                "full_label": (
                    f"{SHORT_CIPHER_LABELS[cipher_key]} R{replica} · "
                    f"{SHORT_SPLIT_LABELS[split]}"
                ),
                "correct_auc": correct_auc,
                "anchor_auc": anchor_auc,
                "wrong_auc": wrong_auc,
                "branch_auc": branch_auc,
                "retention_margin": correct_auc - anchor_auc,
                "wrong_margin": correct_auc - wrong_auc,
                "branch_margin": correct_auc - branch_auc,
            }
        )
    return sorted(
        panels,
        key=lambda panel: (
            cipher_order[str(panel["cipher_key"])],
            int(panel["replica"]),
            split_order[str(panel["split"])],
        ),
    )


def _render_cross_key_anchor(axis: plt.Axes, panels: Sequence[Mapping[str, Any]]) -> None:
    labels = [str(panel["label"]) for panel in panels]
    y = np.arange(len(panels), dtype=float)
    height = 0.34
    correct = np.asarray([float(panel["correct_auc"]) for panel in panels])
    anchors = np.asarray([float(panel["anchor_auc"]) for panel in panels])
    axis.barh(y - height / 2, correct, height, label="共享模型", color="#0F766E")
    axis.barh(y + height / 2, anchors, height, label="独立模型锚点", color="#94A3B8")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.45, 1.035)
    axis.set_xlabel("跨密钥 AUC（从0.45开始，仅用于展开差异）")
    axis.set_title("跨密钥强度：共享模型保住了多少信号", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)
    for index, value in enumerate(correct):
        axis.text(value + 0.008, y[index] - height / 2, f"{value:.3f}", va="center", fontsize=8.8)


def _render_cross_key_controls(
    axis: plt.Axes, panels: Sequence[Mapping[str, Any]]
) -> None:
    labels = [str(panel["label"]) for panel in panels]
    y = np.arange(len(panels), dtype=float)
    for index, panel in enumerate(panels):
        values = [
            float(panel["wrong_auc"]),
            float(panel["branch_auc"]),
            float(panel["correct_auc"]),
        ]
        axis.plot(
            [min(values), max(values)],
            [y[index], y[index]],
            color="#CBD5E1",
            linewidth=2,
        )
    axis.scatter(
        [float(panel["correct_auc"]) for panel in panels],
        y,
        color="#0F766E",
        s=45,
        label="正确结构",
        zorder=3,
    )
    axis.scatter(
        [float(panel["wrong_auc"]) for panel in panels],
        y,
        marker="x",
        color="#DC2626",
        s=50,
        label="错误 S盒",
        zorder=3,
    )
    axis.scatter(
        [float(panel["branch_auc"]) for panel in panels],
        y,
        marker="s",
        facecolors="none",
        edgecolors="#2563EB",
        s=46,
        label="关闭结构分支",
        zorder=3,
    )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.45, 1.035)
    axis.set_xlabel("跨密钥 AUC（同一检查点，0次更新）")
    axis.set_title("结构归因：正确结构是否优于两个控制", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)


def _render_retention_margins(axis: plt.Axes, panels: Sequence[Mapping[str, Any]]) -> None:
    labels = [str(panel["full_label"]) for panel in panels]
    values = np.asarray([float(panel["retention_margin"]) for panel in panels])
    y = np.arange(len(panels), dtype=float)
    colors = np.where(values >= -0.010, "#0F766E", "#DC2626")
    axis.barh(y, values, height=0.62, color=colors)
    axis.axvline(
        -0.010,
        color="#B45309",
        linestyle="--",
        linewidth=1.4,
        label="通过线 -0.010",
    )
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlim(-0.10, 0.03)
    axis.set_xlabel("共享模型 AUC - 独立模型锚点 AUC")
    axis.set_title("信号保留门：红色表示强度损失过大", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", frameon=False)


def _render_semantic_margins(axis: plt.Axes, panels: Sequence[Mapping[str, Any]]) -> None:
    labels = [str(panel["full_label"]) for panel in panels]
    y = np.arange(len(panels), dtype=float)
    wrong = np.asarray([float(panel["wrong_margin"]) for panel in panels])
    branch = np.asarray([float(panel["branch_margin"]) for panel in panels])
    axis.scatter(wrong, y - 0.13, color="#DC2626", s=40, label="正确 - 错误 S盒")
    axis.scatter(
        branch,
        y + 0.13,
        marker="s",
        color="#2563EB",
        s=35,
        label="正确 - 关闭分支",
    )
    axis.axvline(
        0.005,
        color="#B45309",
        linestyle="--",
        linewidth=1.4,
        label="通过线 +0.005",
    )
    axis.axvline(0.0, color="#64748B", linewidth=0.8)
    axis.set_yticks(y, labels, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlim(-0.025, 0.205)
    axis.set_xlabel("正确结构相对控制的 AUC 优势")
    axis.set_title("语义门：S盒全部通过，分支关闭有1项失败", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False, fontsize=8.8)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parse_args", "render_k1ao_training_svg"]
