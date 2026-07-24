from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from blockcipher_nd.tasks.innovation1.runtime_spn_skinny_medium import (
    adjudicate_runtime_spn_skinny_medium,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate RTG2-A SKINNY medium general-GF(2) topology replication."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Progress JSONL to append; defaults to RUN_ROOT/progress.jsonl.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Defer Matplotlib rendering until verified local retrieval.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = args.run_root / "results.jsonl"
    rows = _read_jsonl(results_path)
    gate = adjudicate_runtime_spn_skinny_medium(
        run_id=args.run_id,
        rows=rows,
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
        "task": "innovation1_rtg2a_skinny_general_gf2_medium_replication",
        "training_performed": True,
        "train_samples_per_class": 65_536,
        "train_samples_total": 131_072,
        "validation_samples_per_class": 32_768,
        "validation_samples_total": 65_536,
        "pairs_per_sample": 4,
        "epochs": 5,
        "seed": args.seed,
        "gate": gate,
    }
    _write_json(args.run_root / "validation.json", validation)
    _write_json(args.run_root / "gate.json", gate)
    _write_json(args.run_root / "summary.json", summary)
    _write_history_csv(rows, args.run_root / "history.csv")
    if not args.no_plot:
        render_skinny_medium_svg(gate, args.run_root / "curves.svg")
    _append_progress(
        args.progress or args.run_root / "progress.jsonl",
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


def render_skinny_medium_svg(gate: dict[str, Any], output: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt

    roles = ("true", "corrupted", "independent")
    labels = ("正确 GF(2) 拓扑", "确定性打乱拓扑", "无线性拓扑")
    values = [float(gate["aucs"][role]) for role in roles]
    margins = [
        float(gate["margins"]["true_minus_corrupted"]),
        float(gate["margins"]["true_minus_independent"]),
    ]
    seed = int(gate["seed"])
    status = str(gate["status"])
    conclusion = {
        "pass": "正确拓扑同时通过信号门和两项归因门；可按计划推进下一颗种子。",
        "hold": "中等规模优势未满足冻结门槛；停止扩样并审计方差或训练动态。",
        "fail": "协议证据不完整或不一致；仅修复失败检查，不解释 AUC。",
    }[status]
    status_color = {"pass": "#047857", "hold": "#B45309", "fail": "#B42318"}[status]
    finite_values = [value for value in values if math.isfinite(value)] or [0.5]
    finite_margins = [value for value in margins if math.isfinite(value)] or [0.0]

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
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.6))
        figure.subplots_adjust(
            left=0.075, right=0.97, top=0.72, bottom=0.18, wspace=0.32
        )
        figure.suptitle(
            "创新1 RTG2-A：SKINNY 中等规模 GF(2) 拓扑复验",
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
                f"seed{seed}；7轮；差分 0x2000；每条样本 4 个密文对；"
                "训练 65536/class，验证 32768/class，5 epochs。"
            ),
            ha="left",
            va="top",
            color="#475569",
            fontsize=10.5,
        )
        figure.text(
            0.075,
            0.845,
            conclusion,
            ha="left",
            va="top",
            color=status_color,
            fontweight="bold",
            fontsize=10.3,
        )

        x = np.arange(3)
        bars = axes[0].bar(
            x,
            values,
            color=("#059669", "#DC2626", "#64748B"),
            width=0.62,
        )
        axes[0].bar_label(
            bars,
            labels=[f"{value:.6f}" for value in values],
            padding=5,
        )
        axes[0].axhline(
            0.55,
            color="#1D4ED8",
            linestyle="--",
            linewidth=1.4,
            label="正确拓扑信号门 0.55",
        )
        axes[0].axhline(
            0.5,
            color="#334155",
            linestyle=":",
            linewidth=1.2,
            label="随机基线 0.50",
        )
        axes[0].set_ylim(
            min(0.48, min(finite_values) - 0.015),
            max(0.56, max(finite_values) + 0.025),
        )
        axes[0].set_xticks(x, labels=labels)
        axes[0].set_ylabel("验证 AUC")
        axes[0].set_title("同预算的三种运行时拓扑", loc="left", fontweight="bold")
        axes[0].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[0].legend(loc="upper right", frameon=False)

        margin_x = np.arange(2)
        margin_bars = axes[1].bar(
            margin_x,
            margins,
            color=["#059669" if value >= 0.005 else "#DC2626" for value in margins],
            width=0.58,
        )
        axes[1].bar_label(
            margin_bars,
            labels=[f"{value:+.6f}" for value in margins],
            padding=5,
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
            min(-0.01, min(finite_margins) - 0.01),
            max(0.02, max(finite_margins) + 0.015),
        )
        axes[1].set_xticks(
            margin_x, labels=("正确 - 打乱拓扑", "正确 - 无拓扑")
        )
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_title("正确拓扑相对控制组的优势", loc="left", fontweight="bold")
        axes[1].grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        axes[1].legend(loc="upper right", frameon=False)
        figure.text(
            0.075,
            0.055,
            (
                "证据范围：SKINNY-64/64 7轮、65536/class 中等规模架构/协议复验；"
                "不是正式规模、论文复现、攻击、SOTA 或突破。"
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


def _write_history_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = (
        "model",
        "seed",
        "epoch",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "learning_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for history_row in row.get("history", []):
                writer.writerow(
                    {
                        "model": row.get("model"),
                        "seed": row.get("seed"),
                        **history_row,
                    }
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
