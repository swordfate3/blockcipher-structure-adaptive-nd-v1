from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


SPLIT_TITLES = {
    "same_key_fresh": "同密钥新样本",
    "cross_key_validation": "跨密钥验证",
}


def render_k1bp_svg(
    *,
    discovery_rows: Sequence[Mapping[str, Any]],
    full_oracle_rows: Sequence[Mapping[str, Any]],
    sparse_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    if len(discovery_rows) != 16 or len(full_oracle_rows) != 4 or len(sparse_rows) != 4:
        raise ValueError("K1-BP plot requires 16 discovery and four rows per confirmation panel")
    numeric = [
        float(value)
        for row in (*discovery_rows, *full_oracle_rows, *sparse_rows)
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise ValueError("K1-BP plot requires finite metrics")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "WenQuanYi Zen Hei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(13.8, 9.4), facecolor="#F8FAFC")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0), hspace=0.38, wspace=0.23)
    discovery_axis = fig.add_subplot(grid[0, :])
    oracle_axis = fig.add_subplot(grid[1, 0])
    rank_axis = fig.add_subplot(grid[1, 1])
    for axis in (discovery_axis, oracle_axis, rank_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.65)
        axis.set_axisbelow(True)

    ordered = sorted(discovery_rows, key=lambda row: int(row["target_cell"]))
    cells = np.arange(16)
    fresh = np.asarray([float(row["minimum_fresh_auc"]) for row in ordered])
    selected = int(gate["selected_cell"])
    colors = ["#0F766E" if cell == selected else "#94A3B8" for cell in cells]
    bars = discovery_axis.bar(cells, fresh, color=colors, width=0.72)
    discovery_axis.axhline(0.5, color="#334155", linestyle="--", linewidth=1)
    discovery_axis.axhline(
        0.51, color="#D97706", linestyle="-.", linewidth=1.2, label="微弱信号线 0.51"
    )
    discovery_axis.axhline(
        0.55, color="#DC2626", linestyle=":", linewidth=1.2, label="强候选线 0.55"
    )
    discovery_axis.set_ylim(max(0.45, float(fresh.min()) - 0.03), min(1.0, float(fresh.max()) + 0.08))
    discovery_axis.set_xticks(cells, [str(cell) for cell in cells])
    discovery_axis.set_xlabel("候选内部 cell（仅 seed2 选择）")
    discovery_axis.set_ylabel("两种新样本中较低的 AUC")
    discovery_axis.set_title("第一步：寻找单-cell、4-bit 有效子密钥的稀疏 r5 读出", fontsize=12.5, pad=10)
    discovery_axis.legend(frameon=False, fontsize=8.5, loc="upper right")
    discovery_axis.text(
        bars[selected].get_x() + bars[selected].get_width() / 2,
        fresh[selected] + 0.012,
        f"选中 cell {selected}\n{fresh[selected]:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#0F5132",
    )

    labels = [
        f"s{row['seed']}\n{SPLIT_TITLES[str(row['split'])]}" for row in full_oracle_rows
    ]
    x = np.arange(4)
    width = 0.36
    correct = [float(row["correct_key_auc"]) for row in full_oracle_rows]
    wrong = [float(row["best_wrong_key_auc"]) for row in full_oracle_rows]
    oracle_axis.bar(x - width / 2, correct, width, label="完整正确 K5（oracle）", color="#0F766E")
    oracle_axis.bar(x + width / 2, wrong, width, label="最强错误完整 K5", color="#C2417B")
    oracle_axis.axhline(0.5, color="#334155", linestyle="--", linewidth=1)
    oracle_axis.axhline(0.9, color="#DC2626", linestyle=":", linewidth=1.2)
    oracle_axis.set_ylim(0.45, 1.02)
    oracle_axis.set_xticks(x, labels, fontsize=8.5)
    oracle_axis.set_ylabel("冻结 r5 模型 AUC")
    oracle_axis.set_title("第二步：完整 64-bit K5 只作为上界", fontsize=12, pad=28)
    oracle_axis.legend(
        frameon=False,
        fontsize=8.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )

    exact_ranks = [int(row["true_rank"]) for row in sparse_rows]
    wrong_ranks = [int(row["wrong_sbox_true_rank"]) for row in sparse_rows]
    shuffled_ranks = [int(row["label_shuffle_true_rank"]) for row in sparse_rows]
    rank_labels = [
        f"s{row['seed']}\n{SPLIT_TITLES[str(row['split'])]}" for row in sparse_rows
    ]
    rank_axis.plot(x, exact_ranks, "o-", color="#0F766E", linewidth=2, label="正确结构")
    rank_axis.plot(x, wrong_ranks, "s--", color="#C2417B", linewidth=1.6, label="错误 S 盒")
    rank_axis.plot(x, shuffled_ranks, "^--", color="#7C3AED", linewidth=1.6, label="标签打乱")
    rank_axis.axhline(1, color="#DC2626", linestyle=":", linewidth=1.2)
    rank_axis.set_yscale("log")
    rank_axis.set_ylim(0.8, 24)
    rank_axis.invert_yaxis()
    rank_axis.set_xticks(x, rank_labels, fontsize=8.5)
    rank_axis.set_ylabel("真 4-bit 有效假设名次（越靠上越好）")
    rank_axis.set_title("第三步：16 个有效子密钥候选完整枚举", fontsize=12, pad=28)
    rank_axis.legend(
        frameon=False,
        fontsize=8.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )

    fig.suptitle(
        "uKNIT 6轮：末轮密钥假设能否复用强 r5 信号",
        fontsize=18,
        fontweight="bold",
        y=0.985,
        color="#0F172A",
    )
    fig.text(
        0.5,
        0.945,
        "完整模型需猜 64 bit（不可行上界）；单个稀疏 cell 只含 4 个有效子密钥 bit（16 个候选）",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.text(
        0.5,
        0.018,
        "裁决：保持路线但不扩展　仅 seed2 发现微弱信号，seed3/4 未确认；"
        "完整 oracle 不能称为六轮攻击",
        ha="center",
        fontsize=10,
        color="#334155",
    )
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.075, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "status": "pass",
        "output": str(output),
        "figure_inches": [13.8, 9.4],
        "panels": 3,
        "selected_cell": selected,
        "title_zh": "uKNIT 6轮：末轮密钥假设能否复用强 r5 信号",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plot the uKNIT r6 K1-BP gate.")
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    gate = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    discovery = _read_jsonl(root / "discovery_results.jsonl")
    oracle = _read_jsonl(root / "full_oracle_results.jsonl")
    sparse = _read_jsonl(root / "sparse_rank_results.jsonl")
    print(
        json.dumps(
            render_k1bp_svg(
                discovery_rows=discovery,
                full_oracle_rows=oracle,
                sparse_rows=sparse,
                gate=gate,
                output=root / "curves.svg",
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())
