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

from matplotlib import pyplot as plt

from blockcipher_nd.evaluation.plots import write_history_csv
from blockcipher_nd.tasks.innovation1.runtime_spn_rectangle_attribution import (
    adjudicate_runtime_spn_rectangle_medium,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate RECTANGLE RuntimeE4 RCT2 medium confirmation."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root or args.run_root
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.run_root / "results.jsonl"
    gate = adjudicate_runtime_spn_rectangle_medium(
        run_id=args.run_id,
        rows=_read_jsonl(results_path),
        expected_seed=args.seed,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "results": str(results_path),
    }
    summary = {
        "run_id": args.run_id,
        "task": "innovation1_rectangle_runtime_spn_rct2_medium_confirmation",
        "training_performed": True,
        "train_samples_per_class": gate["samples_per_class"],
        "train_samples_total": gate["train_rows"],
        "validation_samples_per_class": gate["validation_rows"] // 2,
        "validation_samples_total": gate["validation_rows"],
        "pairs_per_sample": 4,
        "epochs": 5,
        "seed": args.seed,
        "gate": gate,
    }
    _write_json(output_root / "validation.json", validation)
    _write_json(output_root / "gate.json", gate)
    _write_json(output_root / "summary.json", summary)
    write_history_csv(results_path, output_root / "history.csv")
    if not args.no_plot:
        render_rectangle_medium_svg(gate, output_root / "curves.svg")
    _append_progress(
        args.progress or output_root / "progress.jsonl",
        "gate_done",
        {
            "run_id": args.run_id,
            "status": gate["status"],
            "decision": gate["decision"],
            "plot_deferred": bool(args.no_plot),
        },
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_rectangle_medium_svg(gate: dict[str, Any], output: Path) -> None:
    roles = ("true", "corrupted", "independent")
    labels = ("正确 ShiftRow 拓扑", "打乱 ShiftRow 拓扑", "无线性拓扑")
    colors = ("#059669", "#DC2626", "#64748B")
    aucs = [float(gate["aucs"][role]) for role in roles]
    margins = [
        float(gate["margins"]["true_minus_corrupted"]),
        float(gate["margins"]["true_minus_independent"]),
    ]
    passed = gate["status"] == "pass"
    seed = int(gate["seed"])
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.3))
        figure.subplots_adjust(
            left=0.075,
            right=0.97,
            top=0.72,
            bottom=0.18,
            wspace=0.30,
        )
        figure.suptitle(
            "创新1 RCT2：RECTANGLE 中等规模运行时拓扑确认",
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
                f"六轮；文献轨迹差分 0x0000002100010020；4 对/样本；seed{seed}；"
                "训练 65536/class，验证 32768/class。"
            ),
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        figure.text(
            0.075,
            0.845,
            (
                "正确拓扑通过信号门与两项归因门。"
                if passed
                else "本次结果未开放后续扩样；按裁决修复证据或停止规模阶梯。"
            ),
            ha="left",
            va="top",
            color="#047857" if passed else "#B42318",
            fontweight="bold",
            fontsize=10.3,
        )

        bars = axes[0].bar(range(3), aucs, color=colors, width=0.62)
        axes[0].bar_label(
            bars,
            labels=[f"{value:.4f}" for value in aucs],
            padding=4,
            fontsize=9.0,
        )
        axes[0].axhline(
            0.55,
            color="#1D4ED8",
            linestyle="--",
            linewidth=1.4,
            label="正确拓扑信号门 0.55",
        )
        axes[0].axhline(0.5, color="#334155", linestyle=":", linewidth=1.2)
        axes[0].set_ylim(min(0.47, min(aucs) - 0.02), max(0.62, max(aucs) + 0.06))
        axes[0].set_xticks(range(3), labels=labels)
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("正确、打乱与无拓扑对照", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(loc="upper right", frameon=False, fontsize=8.8)

        margin_bars = axes[1].bar(
            range(2),
            margins,
            color=("#059669", "#2563EB"),
            width=0.58,
        )
        axes[1].bar_label(
            margin_bars,
            labels=[f"{value:+.4f}" for value in margins],
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
            min(-0.015, min(margins) - 0.015),
            max(0.04, max(margins) + 0.025),
        )
        axes[1].set_xticks(range(2), labels=("正确 - 打乱", "正确 - 无拓扑"))
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_title(f"seed{seed} 拓扑归因边际", loc="left", fontweight="bold")
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(loc="upper right", frameon=False, fontsize=8.8)
        figure.text(
            0.075,
            0.055,
            (
                "证据范围：RECTANGLE r6 单 seed 65536/class 中等规模拓扑确认；"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
