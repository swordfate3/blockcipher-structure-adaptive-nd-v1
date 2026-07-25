from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "blockcipher_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d5 import (
    EXPECTED_BIT_INDICES,
    PANEL_SPECS,
    REFERENCE_BIT_INDEX,
    adjudicate_difference_screen,
    evaluate_difference_candidate,
    prepare_difference_screen_panel,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen all Dialga-128 single-bit differences at prefix r5."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results_path = args.output_root / "results.jsonl"
    progress_path = args.output_root / "progress.jsonl"
    existing = _read_jsonl(results_path) if results_path.exists() else []
    existing_keys = {
        (int(row["bit_index_lsb"]), str(row["key_role"])) for row in existing
    }
    if existing:
        _write_progress(progress_path, "run_resume", args.run_id, rows=len(existing))
    else:
        _write_progress(progress_path, "run_start", args.run_id)

    descriptor = load_runtime_spn_descriptor(
        "configs/runtime/spn/dialga128.json", rounds=2, round_start=2
    )
    panels = {
        key_role: prepare_difference_screen_panel(
            key_role=key_role,
            key=key,
            seed=seed,
        )
        for key_role, (key, seed) in PANEL_SPECS.items()
    }
    rows = list(existing)
    for bit_index in EXPECTED_BIT_INDICES:
        for key_role in PANEL_SPECS:
            if (bit_index, key_role) in existing_keys:
                continue
            row = evaluate_difference_candidate(
                bit_index=bit_index,
                panel=panels[key_role],
                runtime_structure=descriptor.structure,
            )
            rows.append(row)
            _write_jsonl(results_path, _sorted_rows(rows))
            _write_progress(
                progress_path,
                "candidate_panel_done",
                args.run_id,
                bit_index_lsb=bit_index,
                key_role=key_role,
                screen_auc=row["screen_auc"],
            )

    rows = _sorted_rows(rows)
    gate = adjudicate_difference_screen(run_id=args.run_id, rows=rows)
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "training_performed": False,
        "data_generation_performed": True,
        "candidate_count": len(EXPECTED_BIT_INDICES),
        "gate": gate,
    }
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_dialga_d5_svg(gate, args.output_root / "curves.svg")
    _write_progress(
        progress_path,
        "run_done",
        args.run_id,
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_dialga_d5_svg(gate: dict[str, Any], output_path: Path) -> None:
    candidates = list(gate["top_candidates"])
    labels = [f"bit {candidate['bit_index_lsb']}" for candidate in candidates]
    train_values = [candidate["train_key_auc"] for candidate in candidates]
    validation_values = [candidate["validation_key_auc"] for candidate in candidates]
    worst_values = [candidate["worst_key_auc"] for candidate in candidates]
    anchor = gate["reference"]
    anchor_floor = (
        anchor["worst_key_auc"] + gate["thresholds"]["worst_key_auc_margin_over_0x40"]
        if anchor
        else 0.52
    )
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#0F172A",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(15.8, 6.8))
        figure.subplots_adjust(
            left=0.06, right=0.98, top=0.70, bottom=0.22, wspace=0.28
        )
        figure.suptitle(
            "创新1 D5：Dialga 五轮单比特输入差分筛选",
            x=0.06,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.06,
            0.89,
            "比较全部 128 个单比特差分；图中展示按最差密钥 AUC 排名的前 12 项，不训练神经网络。",
            ha="left",
            color="#475569",
        )
        figure.text(
            0.06,
            0.81,
            f"裁决：{_decision_text(gate)}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )

        x = np.arange(len(candidates), dtype=np.float64)
        width = 0.36
        train_bars = axes[0].bar(
            x - width / 2, train_values, width, label="训练密钥", color="#2563EB"
        )
        validation_bars = axes[0].bar(
            x + width / 2,
            validation_values,
            width,
            label="独立验证密钥",
            color="#D97706",
        )
        axes[0].bar_label(
            train_bars,
            labels=[f"{value:.3f}" for value in train_values],
            padding=3,
            fontsize=8,
            rotation=90,
        )
        axes[0].bar_label(
            validation_bars,
            labels=[f"{value:.3f}" for value in validation_values],
            padding=3,
            fontsize=8,
            rotation=90,
        )
        axes[0].axhline(
            0.52,
            color="#64748B",
            linestyle="--",
            linewidth=1.1,
            label="双密钥门槛 0.52",
        )
        axes[0].set_title(
            "前 12 个候选在两把固定密钥上的 AUC", loc="left", fontweight="bold"
        )
        axes[0].set_ylabel("朴素贝叶斯验证 AUC")
        axes[0].set_xticks(x, labels, rotation=35, ha="right")
        all_values = train_values + validation_values + [0.5]
        axes[0].set_ylim(
            max(0.45, min(all_values) - 0.03), min(1.0, max(all_values) + 0.08)
        )
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper right")

        worst_bars = axes[1].bar(
            x,
            worst_values,
            width=0.62,
            color=[
                "#DC2626"
                if candidate["bit_index_lsb"] == REFERENCE_BIT_INDEX
                else "#0F9D76"
                for candidate in candidates
            ],
        )
        axes[1].bar_label(
            worst_bars,
            labels=[f"{value:.4f}" for value in worst_values],
            padding=4,
            fontsize=8,
            rotation=90,
        )
        axes[1].axhline(
            anchor_floor,
            color="#2563EB",
            linestyle="--",
            linewidth=1.2,
            label=f"0x40 最差密钥 + 0.01 = {anchor_floor:.4f}",
        )
        axes[1].axhline(
            0.52, color="#64748B", linestyle=":", linewidth=1.1, label="绝对门槛 0.52"
        )
        axes[1].set_title("跨密钥保守排名（取较低 AUC）", loc="left", fontweight="bold")
        axes[1].set_ylabel("两把密钥中较低的 AUC")
        axes[1].set_xticks(x, labels, rotation=35, ha="right")
        axes[1].set_ylim(
            max(0.45, min(worst_values + [0.5]) - 0.03),
            min(1.0, max(worst_values) + 0.08),
        )
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper right")
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _decision_text(gate: dict[str, Any]) -> str:
    if gate["status"] == "fail":
        return "筛选协议无效，必须修复后才能解释。"
    if gate["shortlist"]:
        top = gate["shortlist"][0]
        return f"bit {top['bit_index_lsb']} 同时超过两把密钥和 0x40 锚点，可进入同预算神经门。"
    return "没有候选同时通过双密钥 0.52 和 0x40 增益门，停止机械差分搜索。"


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_order = {key_role: index for index, key_role in enumerate(PANEL_SPECS)}
    return sorted(
        rows,
        key=lambda row: (
            int(row["bit_index_lsb"]),
            key_order[str(row["key_role"])],
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_progress(path: Path, event: str, run_id: str, **payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    "run_id": run_id,
                    **payload,
                },
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
