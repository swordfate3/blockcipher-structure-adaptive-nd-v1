from __future__ import annotations

import argparse
import csv
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

from blockcipher_nd.tasks.innovation2.integral_context_group_split_readiness import (
    ContextGroupSplitConfig,
    run_context_group_split_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Innovation 2 context/mask group-disjoint shortcuts."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ContextGroupSplitConfig(
        run_id=args.run_id,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
        folds=args.folds,
    )
    source_gate = _read_json(args.source_root / "gate.json")
    source_metadata = _read_json(args.source_root / "metadata.json")
    source_label_rows = _read_csv(args.source_root / "labels.csv")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "source_run_id": source_gate.get("run_id"),
            "folds": args.folds,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_context_group_split_audit(
        config,
        source_gate=source_gate,
        source_metadata=source_metadata,
        source_label_rows=source_label_rows,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_json(args.output_root / "gate.json", result["gate"])
    _write_json(args.output_root / "metadata.json", result["metadata"])
    render_group_split_svg(
        result["rows"],
        result["gate"],
        args.output_root / "curves.svg",
    )
    _write_progress(
        progress_path,
        "run_done",
        {
            "run_id": args.run_id,
            "status": result["gate"]["status"],
            "decision": result["gate"]["decision"],
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["gate"]["status"],
                "decision": result["gate"]["decision"],
                "metrics": result["gate"]["metrics"],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_group_split_svg(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    labels = [str(row["baseline_label"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    aucs = [float(row["auc"]) for row in rows]
    directional_aucs = [float(row["directional_auc"]) for row in rows]
    briers = [float(row["brier"]) for row in rows]
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
        figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.2))
        figure.subplots_adjust(
            left=0.09,
            right=0.975,
            top=0.70,
            bottom=0.27,
            wspace=0.38,
        )
        figure.suptitle(
            "创新2 E17c：context/mask 双轴组外捷径审计",
            x=0.09,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.09,
            0.89,
            "固定 E17b 的512行标签；只改变交叉验证拆分；每行获得一次真正 out-of-group 预测。",
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        y = np.arange(len(rows), dtype=np.float64)
        auc_axis, calibration_axis = axes
        raw_bars = auc_axis.barh(
            y - 0.17,
            aucs,
            height=0.30,
            color="#2563EB",
            label="原始 AUC",
        )
        directional_bars = auc_axis.barh(
            y + 0.17,
            directional_aucs,
            height=0.30,
            color="#DC2626",
            label="方向无关 AUC",
        )
        auc_axis.axvline(
            0.75,
            color="#D97706",
            linestyle="--",
            linewidth=1.2,
            label="捷径停止线 0.75",
            zorder=0,
        )
        _label_horizontal_bars(auc_axis, raw_bars, aucs)
        _label_horizontal_bars(auc_axis, directional_bars, directional_aucs)
        auc_axis.set_title(
            "组外排序能力",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        auc_axis.set_xlabel("AUC")
        auc_axis.set_yticks(y, labels=labels)
        auc_axis.set_xlim(0.0, 1.05)
        auc_axis.invert_yaxis()
        auc_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        auc_axis.grid(False, axis="y")
        auc_axis.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.20),
            ncol=3,
            frameon=False,
            fontsize=8.3,
        )

        accuracy_bars = calibration_axis.barh(
            y - 0.17,
            accuracies,
            height=0.30,
            color="#059669",
            label="准确率",
        )
        brier_bars = calibration_axis.barh(
            y + 0.17,
            briers,
            height=0.30,
            color="#7C3AED",
            label="Brier（低为好）",
        )
        _label_horizontal_bars(calibration_axis, accuracy_bars, accuracies)
        _label_horizontal_bars(calibration_axis, brier_bars, briers)
        calibration_axis.set_title(
            "固定阈值与概率误差",
            loc="left",
            fontweight="bold",
            pad=10,
        )
        calibration_axis.set_xlabel("指标值")
        calibration_axis.set_yticks(y, labels=labels)
        calibration_axis.set_xlim(0.0, 1.05)
        calibration_axis.invert_yaxis()
        calibration_axis.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
        calibration_axis.grid(False, axis="y")
        calibration_axis.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.20),
            ncol=2,
            frameon=False,
            fontsize=8.3,
        )

        decision_labels = {
            "innovation2_group_disjoint_shortcuts_controlled": (
                "组外线性捷径受控，可进入 fresh-key 验证"
            ),
            "innovation2_group_disjoint_shortcut_generalizes": (
                "至少一种组外拆分仍保留位模式捷径，禁止训练"
            ),
            "innovation2_group_disjoint_protocol_invalid": (
                "组外拆分、覆盖或源标签无效"
            ),
        }
        figure.text(
            0.09,
            0.055,
            (
                f"裁决：{decision_labels.get(gate['decision'], gate['decision'])}。"
                "这是线性捷径 readiness，不是神经训练，也不是 fresh-key 结果。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _label_horizontal_bars(
    axis: Any,
    bars: Any,
    values: list[float],
) -> None:
    for bar, value in zip(bars, values, strict=True):
        inside = value >= 0.90
        axis.text(
            value - 0.008 if inside else value + 0.010,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="right" if inside else "left",
            va="center",
            fontsize=7.8,
            color="#FFFFFF" if inside else "#334155",
            zorder=4,
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_progress(
    path: Path,
    event: str,
    payload: dict[str, Any],
    *,
    mode: str = "a",
) -> None:
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
