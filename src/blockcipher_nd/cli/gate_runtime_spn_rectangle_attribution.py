from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.tasks.innovation1.runtime_spn_rectangle_attribution import (
    EXPECTED_SEEDS,
    adjudicate_runtime_spn_rectangle_attribution,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate RECTANGLE RuntimeE4 non-contiguous topology attribution."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = args.run_root / "results.jsonl"
    rows = _read_jsonl(results_path)
    gate = adjudicate_runtime_spn_rectangle_attribution(
        run_id=args.run_id,
        rows=rows,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(results_path),
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_runtime_spn_rectangle_noncontiguous_attribution_rct1",
        "training_performed": True,
        "train_samples_per_class": 2048,
        "validation_samples_per_class": 1024,
        "epochs": 5,
        "seeds": list(EXPECTED_SEEDS),
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    write_history_csv(results_path, args.run_root / "history.csv")
    render_rectangle_attribution_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.run_root / "progress.jsonl",
        "gate_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_rectangle_attribution_svg(gate: dict[str, Any], output: Path) -> None:
    roles = ("true", "corrupted", "independent")
    labels = ("正确 ShiftRow 拓扑", "打乱 ShiftRow 拓扑", "无线性拓扑")
    colors = ("#059669", "#DC2626", "#64748B")
    seeds = tuple(int(seed) for seed in gate["seeds"])
    passed = gate["status"] == "pass"
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.6))
        figure.subplots_adjust(
            left=0.075,
            right=0.97,
            top=0.72,
            bottom=0.18,
            wspace=0.30,
        )
        figure.suptitle(
            "创新1 RCT1：RECTANGLE 非连续 cell 的运行时拓扑归因",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.895,
            (
                "六轮；文献轨迹差分 0x0000002100010020；4 对/样本；"
                "训练 2048/class，验证 1024/class；双 seed 同预算。"
            ),
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        conclusion = (
            "两颗 seed 均通过正确拓扑信号门和两种控制门。"
            if passed
            else "冻结双 seed 拓扑归因门未通过；不得远程扩样。"
        )
        figure.text(
            0.075,
            0.845,
            conclusion,
            ha="left",
            va="top",
            color="#047857" if passed else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )

        x = np.arange(len(seeds), dtype=float)
        width = 0.23
        all_aucs: list[float] = []
        for role_index, (role, label, color) in enumerate(
            zip(roles, labels, colors, strict=True)
        ):
            values = [float(gate["aucs"][str(seed)][role]) for seed in seeds]
            all_aucs.extend(values)
            bars = axes[0].bar(
                x + (role_index - 1) * width,
                values,
                width=width,
                color=color,
                label=label,
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in values],
                padding=4,
                fontsize=8.5,
            )
        axes[0].axhline(
            0.55,
            color="#1D4ED8",
            linestyle="--",
            linewidth=1.4,
            label="正确拓扑信号门 0.55",
        )
        axes[0].axhline(0.5, color="#334155", linestyle=":", linewidth=1.2)
        axes[0].set_ylim(
            min(0.47, min(all_aucs) - 0.02),
            max(0.62, max(all_aucs) + 0.065),
        )
        axes[0].set_xticks(x, labels=[f"seed{seed}" for seed in seeds])
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("正确、打乱与无拓扑对照", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            legend_labels,
            loc="center",
            bbox_to_anchor=(0.29, 0.785),
            frameon=False,
            fontsize=8.8,
            ncol=2,
        )

        margin_labels = ("正确 - 打乱", "正确 - 无拓扑")
        margin_keys = ("true_minus_corrupted", "true_minus_independent")
        margin_width = 0.34
        all_margins: list[float] = []
        for seed_index, seed in enumerate(seeds):
            values = [
                float(gate["margins"][str(seed)][key]) for key in margin_keys
            ]
            all_margins.extend(values)
            bars = axes[1].bar(
                np.arange(2) + (seed_index - 0.5) * margin_width,
                values,
                width=margin_width,
                color="#059669" if seed == 0 else "#2563EB",
                label=f"seed{seed}",
            )
            axes[1].bar_label(
                bars,
                labels=[f"{value:+.4f}" for value in values],
                padding=4,
                fontsize=9.0,
            )
        axes[1].axhline(
            0.005,
            color="#1D4ED8",
            linestyle="--",
            linewidth=1.4,
            label="归因门 +0.005",
        )
        axes[1].axhline(0.0, color="#334155", linewidth=1.0)
        axes[1].set_ylim(
            min(-0.015, min(all_margins) - 0.015),
            max(0.12, max(all_margins) + 0.03),
        )
        axes[1].set_xticks(np.arange(2), labels=margin_labels)
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_title("双 seed 拓扑归因边际", loc="left", fontweight="bold")
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(loc="upper right", frameon=False, fontsize=8.8)
        figure.text(
            0.075,
            0.055,
            (
                "证据范围：RECTANGLE r6 本地 2048/class 非连续 cell 拓扑归因；"
                "不是正式规模、论文复现、攻击、SOTA 或通用 SPN 结论。"
            ),
            ha="left",
            va="bottom",
            color="#334155",
            fontsize=10.0,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="svg", bbox_inches="tight")
        plt.close(figure)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_progress(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
