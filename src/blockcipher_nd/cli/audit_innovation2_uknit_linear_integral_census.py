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
from matplotlib.colors import PowerNorm

from blockcipher_nd.tasks.innovation2.uknit_linear_integral_census import (
    ACTIVE_CELLS,
    TARGET_ROUNDS,
    UknitLinearIntegralCensusConfig,
    run_uknit_linear_integral_census,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit uKNIT rounds with Hwang-style GF(2) integral kernels."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discovery-trials", type=int, default=128)
    parser.add_argument("--validation-trials", type=int, default=128)
    parser.add_argument("--trial-chunk-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = UknitLinearIntegralCensusConfig(
        run_id=args.run_id,
        seed=args.seed,
        discovery_trials=args.discovery_trials,
        validation_trials=args.validation_trials,
        trial_chunk_size=args.trial_chunk_size,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.jsonl"

    def progress_callback(event: str, payload: dict[str, Any]) -> None:
        _write_progress(progress_path, event, payload)

    _write_progress(
        progress_path,
        "run_start",
        {
            "run_id": args.run_id,
            "cipher": "uKNIT-BC",
            "calibration_round": 1,
            "target_rounds": list(TARGET_ROUNDS),
            "active_cells": list(ACTIVE_CELLS),
            "discovery_trials": args.discovery_trials,
            "validation_trials": args.validation_trials,
            "plaintexts_per_multiset": 16,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_uknit_linear_integral_census(
        config,
        progress_callback=progress_callback,
    )
    (args.output_root / "results.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in result["rows"]
        ),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "round_summary.csv", result["round_summaries"])
    _write_csv(args.output_root / "kernel_basis.csv", result["basis_rows"])
    np.save(
        args.output_root / "keys.npy",
        _keys_to_u64_pairs(result["keys"]),
        allow_pickle=False,
    )
    np.save(
        args.output_root / "base_plaintexts.npy",
        result["base_plaintexts"],
        allow_pickle=False,
    )
    np.save(
        args.output_root / "parity_rows.npy",
        result["parity_rows"],
        allow_pickle=False,
    )
    np.save(
        args.output_root / "random_control_rows.npy",
        result["random_control_rows"],
        allow_pickle=False,
    )
    _write_json(args.output_root / "metadata.json", result["metadata"])
    _write_json(args.output_root / "gate.json", result["gate"])
    render_uknit_census_svg(
        result["rows"],
        result["round_summaries"],
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
            "highest_supported_round": result["gate"]["highest_supported_round"],
            "training_performed": False,
        },
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": result["gate"]["status"],
                "decision": result["gate"]["decision"],
                "highest_supported_round": result["gate"][
                    "highest_supported_round"
                ],
                "highest_supported_cells": result["gate"][
                    "highest_supported_cells"
                ],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_uknit_census_svg(
    rows: list[dict[str, Any]],
    round_summaries: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    target_rows = [row for row in rows if int(row["rounds"]) in TARGET_ROUNDS]
    by_setting = {
        (int(row["rounds"]), int(row["active_cell"])): row for row in target_rows
    }
    nullities = np.asarray(
        [
            [int(by_setting[(rounds, cell)]["joint_nullity"]) for rounds in TARGET_ROUNDS]
            for cell in ACTIVE_CELLS
        ],
        dtype=np.int64,
    )
    target_summaries = [
        row for row in round_summaries if int(row["rounds"]) in TARGET_ROUNDS
    ]
    stable_counts = [int(row["stable_cells"]) for row in target_summaries]
    random_counts = [
        int(row["random_control_nontrivial_cells"]) for row in target_summaries
    ]
    positions = np.arange(len(TARGET_ROUNDS))
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.5,
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
        figure, (heat_axis, count_axis) = plt.subplots(
            1,
            2,
            figsize=(15.6, 8.2),
            gridspec_kw={"width_ratios": (1.55, 1.0)},
        )
        figure.subplots_adjust(
            left=0.065,
            right=0.975,
            top=0.76,
            bottom=0.19,
            wspace=0.25,
        )
        figure.suptitle(
            "创新2：uKNIT-BC 单活动 cell 的积分核轮数普查",
            x=0.065,
            y=0.965,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.065,
            0.895,
            (
                "每格表示16个明文积分集合在256次完全独立试验后的 "
                "64-bit 输出核维数；128次发现 + 128次独立验证。"
            ),
            ha="left",
            va="top",
            fontsize=10.0,
            color="#526070",
        )
        figure.text(
            0.065,
            0.85,
            "r1 校准要求16个位置全部 rank/nullity=0/64；图中只展示目标 r3-r11，避免校准值压缩目标差异。",
            ha="left",
            va="top",
            fontsize=9.4,
            color="#526070",
        )

        vmax = max(1, int(nullities.max(initial=0)))
        image = heat_axis.imshow(
            nullities,
            cmap="YlGnBu",
            norm=PowerNorm(gamma=0.35, vmin=0, vmax=vmax),
            aspect="auto",
            interpolation="nearest",
        )
        heat_axis.set_title("联合矩阵的 kernel 维数（nullity）", loc="left", fontweight="bold")
        heat_axis.set_xlabel("uKNIT prefix 轮数")
        heat_axis.set_ylabel("活动 cell（0 为最高有效 nibble）")
        heat_axis.set_xticks(positions, [f"r{rounds}" for rounds in TARGET_ROUNDS])
        heat_axis.set_yticks(np.arange(len(ACTIVE_CELLS)), ACTIVE_CELLS)
        colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.035, pad=0.025)
        colorbar.set_label("nullity")
        for cell_index in range(len(ACTIVE_CELLS)):
            for round_index in range(len(TARGET_ROUNDS)):
                value = int(nullities[cell_index, round_index])
                if value > 0:
                    heat_axis.text(
                        round_index,
                        cell_index,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=8.2,
                        color="white" if value > vmax / 2 else "#0F172A",
                        fontweight="bold",
                    )

        width = 0.36
        target_bars = count_axis.bar(
            positions - width / 2,
            stable_counts,
            width,
            color="#047857",
            label="uKNIT 稳定非零核位置",
        )
        control_bars = count_axis.bar(
            positions + width / 2,
            random_counts,
            width,
            color="#64748B",
            label="随机输出秩亏位置",
        )
        count_axis.bar_label(target_bars, padding=3, fontsize=8.5)
        count_axis.bar_label(control_bars, padding=3, fontsize=8.5)
        count_axis.set_title("每轮通过独立验证的位置数", loc="left", fontweight="bold")
        count_axis.set_ylabel("位置数（共16个）")
        count_axis.set_xlabel("uKNIT prefix 轮数")
        count_axis.set_xticks(positions, [f"r{rounds}" for rounds in TARGET_ROUNDS])
        count_axis.set_ylim(0, 17.5)
        count_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        count_axis.legend(frameon=False, loc="upper right", fontsize=8.7)

        highest = gate.get("highest_supported_round")
        if highest is None:
            result_text = "目标 r3-r11 未发现独立验证后的单-cell非零线性核。"
        else:
            result_text = (
                f"当前本地抽样最高支持 r{highest}，活动位置："
                f"{gate.get('highest_supported_cells') or '无'}。"
            )
        figure.text(
            0.065,
            0.075,
            f"裁决：{result_text}",
            ha="left",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color="#0F172A",
        )
        figure.text(
            0.065,
            0.035,
            (
                "范围：本地128+128轮数筛查；不是全密钥证明、论文默认1000+1000确认、"
                "神经网络准确率或完整密钥恢复结论。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#526070",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _keys_to_u64_pairs(keys: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        [[key >> 64, key & ((1 << 64) - 1)] for key in keys],
        dtype=np.uint64,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"CSV output requires at least one row: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
