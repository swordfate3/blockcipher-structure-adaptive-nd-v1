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

from blockcipher_nd.cli.audit_innovation2_uknit_linear_integral_census import (
    _keys_to_u64_pairs,
    _write_csv,
    _write_json,
    _write_progress,
)
from blockcipher_nd.tasks.innovation2.uknit_topology_pair_integral_census import (
    CONTROL_PAIRS,
    PAIR_STRUCTURES,
    TARGET_ROUNDS,
    TOPOLOGY_PAIRS,
    UknitTopologyPairCensusConfig,
    run_uknit_topology_pair_integral_census,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit uKNIT topology pairs with Hwang-style GF(2) kernels."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discovery-trials", type=int, default=128)
    parser.add_argument("--validation-trials", type=int, default=128)
    parser.add_argument("--trial-chunk-size", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = UknitTopologyPairCensusConfig(
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
            "target_rounds": list(TARGET_ROUNDS),
            "topology_pairs": [list(pair) for pair in TOPOLOGY_PAIRS],
            "cross_group_controls": [list(pair) for pair in CONTROL_PAIRS],
            "discovery_trials": args.discovery_trials,
            "validation_trials": args.validation_trials,
            "plaintexts_per_multiset": 256,
            "training_performed": False,
        },
        mode="w",
    )
    result = run_uknit_topology_pair_integral_census(
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
    render_uknit_pair_census_svg(
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
                "highest_supported_pairs": result["gate"][
                    "highest_supported_pairs"
                ],
                "next_action": result["gate"]["next_action"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["status"] != "fail" else 1


def render_uknit_pair_census_svg(
    rows: list[dict[str, Any]],
    round_summaries: list[dict[str, Any]],
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    by_setting = {
        (int(row["rounds"]), str(row["active_pair"])): row
        for row in rows
        if int(row["rounds"]) in TARGET_ROUNDS
    }
    pair_labels = [f"{a}+{b}" for a, b in PAIR_STRUCTURES]
    display_labels = [
        label if pair in TOPOLOGY_PAIRS else f"控制 {label}"
        for pair, label in zip(PAIR_STRUCTURES, pair_labels, strict=True)
    ]
    nullities = np.asarray(
        [
            [
                int(by_setting[(rounds, label)]["joint_nullity"])
                for rounds in TARGET_ROUNDS
            ]
            for label in pair_labels
        ],
        dtype=np.int64,
    )
    target_summaries = [
        row for row in round_summaries if int(row["rounds"]) in TARGET_ROUNDS
    ]
    topology_fraction = [
        100.0 * float(row["topology_stable_fraction"])
        for row in target_summaries
    ]
    control_fraction = [
        100.0 * float(row["control_stable_fraction"])
        for row in target_summaries
    ]
    positions = np.arange(len(TARGET_ROUNDS))
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 9.3,
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
            figsize=(15.8, 10.4),
            gridspec_kw={"width_ratios": (1.5, 1.0)},
        )
        figure.subplots_adjust(
            left=0.09,
            right=0.975,
            top=0.79,
            bottom=0.16,
            wspace=0.27,
        )
        figure.suptitle(
            "创新2：uKNIT-BC 拓扑双 cell 的积分核轮数普查",
            x=0.09,
            y=0.97,
            ha="left",
            fontsize=15.5,
            fontweight="bold",
        )
        figure.text(
            0.09,
            0.915,
            (
                "24个首轮扩散组内 pair + 4个跨组控制；每个积分集合256个明文，"
                "128次发现 + 128次完全独立验证。"
            ),
            ha="left",
            va="top",
            fontsize=9.8,
            color="#526070",
        )
        figure.text(
            0.09,
            0.875,
            "热图显示联合64-bit parity matrix的核维数；只有非零格标数值，空白格就是满秩/nullity=0。",
            ha="left",
            va="top",
            fontsize=9.3,
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
        heat_axis.set_title("双 cell 输出平衡空间维数", loc="left", fontweight="bold")
        heat_axis.set_xlabel("uKNIT prefix 轮数")
        heat_axis.set_ylabel("活动 pair（底部4行为跨组控制）")
        heat_axis.set_xticks(positions, [f"r{rounds}" for rounds in TARGET_ROUNDS])
        heat_axis.set_yticks(np.arange(len(display_labels)), display_labels)
        heat_axis.axhline(len(TOPOLOGY_PAIRS) - 0.5, color="#DC2626", linewidth=1.2)
        colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.035, pad=0.025)
        colorbar.set_label("nullity")
        for pair_index in range(len(PAIR_STRUCTURES)):
            for round_index in range(len(TARGET_ROUNDS)):
                value = int(nullities[pair_index, round_index])
                if value > 0:
                    heat_axis.text(
                        round_index,
                        pair_index,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=7.8,
                        color="white" if value > vmax / 2 else "#0F172A",
                        fontweight="bold",
                    )

        width = 0.36
        target_bars = count_axis.bar(
            positions - width / 2,
            topology_fraction,
            width,
            color="#047857",
            label="组内拓扑 pair 通过率",
        )
        control_bars = count_axis.bar(
            positions + width / 2,
            control_fraction,
            width,
            color="#64748B",
            label="跨组控制 pair 通过率",
        )
        count_axis.bar_label(target_bars, fmt="%.0f%%", padding=3, fontsize=8.2)
        count_axis.bar_label(control_bars, fmt="%.0f%%", padding=3, fontsize=8.2)
        count_axis.set_title("独立验证后的稳定非零核比例", loc="left", fontweight="bold")
        count_axis.set_ylabel("pair 通过率")
        count_axis.set_xlabel("uKNIT prefix 轮数")
        count_axis.set_xticks(positions, [f"r{rounds}" for rounds in TARGET_ROUNDS])
        count_axis.set_ylim(0, 108)
        count_axis.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        count_axis.legend(frameon=False, loc="upper right", fontsize=8.6)

        highest = gate.get("highest_supported_round")
        if highest is None:
            result_text = "r4-r8 未发现通过独立验证的双-cell线性核。"
        else:
            highest_summary = next(
                row
                for row in target_summaries
                if int(row["rounds"]) == int(highest)
            )
            maximum_nullity = max(
                (gate.get("highest_pair_nullities") or {}).values(),
                default=0,
            )
            strongest_pairs = [
                pair
                for pair, nullity in (gate.get("highest_pair_nullities") or {}).items()
                if int(nullity) == int(maximum_nullity)
            ]
            result_text = (
                f"当前本地抽样最高支持 r{highest}：组内 "
                f"{highest_summary['stable_topology_pairs']}/24，跨组 "
                f"{highest_summary['stable_control_pairs']}/4；最大维数 "
                f"{maximum_nullity}（{','.join(strongest_pairs)}）。"
            )
        figure.text(
            0.09,
            0.065,
            f"裁决：{result_text}",
            ha="left",
            va="bottom",
            fontsize=9.8,
            fontweight="bold",
        )
        figure.text(
            0.09,
            0.028,
            (
                "范围：本地128+128轮数筛查；不是全密钥证明、论文默认确认、"
                "神经网络结果或完整密钥恢复结论。"
            ),
            ha="left",
            va="bottom",
            fontsize=9.1,
            color="#526070",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
