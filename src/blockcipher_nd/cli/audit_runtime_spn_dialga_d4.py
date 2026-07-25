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

from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d1 import (
    adjudicate_runtime_spn_dialga_d1,
    adjudicate_runtime_spn_dialga_d3,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_dialga_d4 import (
    D1_DECISION,
    D3_DECISION,
    EXPECTED_CONDITIONS,
    adjudicate_factorial_dialga,
    evaluate_factorial_dialga,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_sbox_counterfactual import (
    file_sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross Dialga D1/D3 data with runtime windows using frozen D1 checkpoints."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--d3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    d1_rows, d1_gate, d1_paths = _verify_source(
        root=args.d1_root,
        adjudicator=adjudicate_runtime_spn_dialga_d1,
        expected_status="pass",
        expected_decision=D1_DECISION,
    )
    d3_rows, d3_gate, d3_paths = _verify_source(
        root=args.d3_root,
        adjudicator=adjudicate_runtime_spn_dialga_d3,
        expected_status="hold",
        expected_decision=D3_DECISION,
    )
    descriptors = {
        start: load_runtime_spn_descriptor(
            "configs/runtime/spn/dialga128.json", rounds=2, round_start=start
        )
        for start in (2, 3)
    }
    if descriptors[2].sha256 != descriptors[3].sha256:
        raise ValueError("D4 descriptor SHA drifted between runtime windows")

    source_hashes = {
        "d1_results_sha256": file_sha256(d1_paths["results"]),
        "d1_gate_sha256": file_sha256(d1_paths["gate"]),
        "d3_results_sha256": file_sha256(d3_paths["results"]),
        "d3_gate_sha256": file_sha256(d3_paths["gate"]),
    }
    result_rows: list[dict[str, Any]] = []
    progress_path = args.output_root / "progress.jsonl"
    _write_progress(progress_path, "run_start", args.run_id)
    for seed in (0, 1):
        d1_source = _source_candidate(d1_rows, seed)
        d3_source = _source_candidate(d3_rows, seed)
        d1_dataset, d1_hashes = _load_validation_dataset(d1_source, seed, rounds=4)
        d3_dataset, d3_hashes = _load_validation_dataset(d3_source, seed, rounds=5)
        checkpoint_path = Path(d1_source["training"]["checkpoint_output"])
        result_rows.extend(
            evaluate_factorial_dialga(
                seed=seed,
                model_options=dict(d1_source["training"]["model_options"]),
                checkpoint_path=checkpoint_path,
                datasets={"d1_r4": d1_dataset, "d3_r5": d3_dataset},
                dataset_hashes={"d1_r4": d1_hashes, "d3_r5": d3_hashes},
                structures={
                    start: descriptor.structure
                    for start, descriptor in descriptors.items()
                },
                anchor_auc=float(d1_source["metrics"]["auc"]),
                checkpoint_sha256=file_sha256(checkpoint_path),
                source_hashes=source_hashes,
                descriptor_name=descriptors[2].name,
                descriptor_path=str(descriptors[2].path),
                descriptor_sha256=descriptors[2].sha256,
                batch_size=args.batch_size,
                device=args.device,
            )
        )
        _write_progress(progress_path, "seed_done", args.run_id, seed=seed)

    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    gate = adjudicate_factorial_dialga(run_id=args.run_id, rows=result_rows)
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {
            "d1_results": str(d1_paths["results"]),
            "d1_gate": str(d1_paths["gate"]),
            "d3_results": str(d3_paths["results"]),
            "d3_gate": str(d3_paths["gate"]),
        },
        "source_gates_recomputed": True,
        "d1_source_status": d1_gate["status"],
        "d3_source_status": d3_gate["status"],
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": "Dialga-128",
        "training_performed": False,
        "data_generation_performed": False,
        "source_runs": {"d1": str(args.d1_root), "d3": str(args.d3_root)},
        "gate": gate,
    }
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_dialga_d4_svg(gate, args.output_root / "curves.svg")
    _write_progress(
        progress_path,
        "run_done",
        args.run_id,
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_dialga_d4_svg(gate: dict[str, Any], output_path: Path) -> None:
    labels = (
        "四轮数据\n旧窗口",
        "四轮数据\n新窗口",
        "五轮数据\n旧窗口",
        "五轮数据\n新窗口",
    )
    colors = ("#2563EB", "#7C3AED", "#D97706", "#0F9D76")
    seed_results = gate["seed_results"]
    aucs = {
        seed: [
            seed_results[str(seed)][f"{condition}_auc"]
            for condition in EXPECTED_CONDITIONS
        ]
        for seed in (0, 1)
    }
    effects = {
        seed: [
            seed_results[str(seed)]["data_at_w2_effect"],
            seed_results[str(seed)]["data_at_w3_effect"],
            seed_results[str(seed)]["window_at_r4_effect"],
            seed_results[str(seed)]["window_at_r5_effect"],
            seed_results[str(seed)]["interaction"],
        ]
        for seed in (0, 1)
    }
    effect_labels = ("数据@旧窗", "数据@新窗", "窗口@四轮", "窗口@五轮", "交互项")

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
        figure, axes = plt.subplots(1, 2, figsize=(15.8, 6.7))
        figure.subplots_adjust(
            left=0.06, right=0.98, top=0.70, bottom=0.17, wspace=0.24
        )
        figure.suptitle(
            "创新1 D4：Dialga 四轮/五轮数据与运行时窗口交叉审计",
            x=0.06,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.06,
            0.89,
            "固定 D1 最佳权重，不训练、不生成数据；分别改变加密深度和外部结构窗口，定位五轮失效原因。",
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

        x = np.arange(4, dtype=np.float64)
        width = 0.34
        for seed, offset in ((0, -width / 2), (1, width / 2)):
            bars = axes[0].bar(
                x + offset,
                aucs[seed],
                width,
                label=f"seed{seed}",
                color=[_shade(color, seed) for color in colors],
                edgecolor="#FFFFFF",
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in aucs[seed]],
                padding=4,
                fontsize=9,
            )
        axes[0].axhline(
            0.5, color="#64748B", linestyle="--", linewidth=1.1, label="随机线 0.5"
        )
        axes[0].set_title(
            "同一权重下四个因子格的验证 AUC", loc="left", fontweight="bold"
        )
        axes[0].set_ylabel("AUC")
        axes[0].set_xticks(x, labels)
        all_aucs = [value for seed_values in aucs.values() for value in seed_values]
        axes[0].set_ylim(max(0.0, min(all_aucs) - 0.05), min(1.0, max(all_aucs) + 0.08))
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper right", ncol=3)

        effect_x = np.arange(5, dtype=np.float64)
        for seed, offset, color in (
            (0, -width / 2, "#2563EB"),
            (1, width / 2, "#D97706"),
        ):
            bars = axes[1].bar(
                effect_x + offset,
                effects[seed],
                width,
                label=f"seed{seed}",
                color=color,
            )
            axes[1].bar_label(
                bars,
                labels=[f"{value:+.3f}" for value in effects[seed]],
                padding=4,
                fontsize=9,
            )
        axes[1].axhline(0.0, color="#64748B", linewidth=1.0)
        axes[1].set_title(
            "数据、窗口及二者交互的 AUC 效应", loc="left", fontweight="bold"
        )
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_xticks(effect_x, effect_labels, rotation=16, ha="right")
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[1].legend(frameon=False, loc="best")
        axes[1].margins(y=0.23)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _decision_text(gate: dict[str, Any]) -> str:
    labels = {
        "fifth_round_data_signal_loss": "主要是五轮密文数据中的原差分信号消失，先筛输入差分。",
        "runtime_window_incompatibility": "主要是旧模型不适配新结构窗口，优先改残差/门控拓扑处理器。",
        "both_data_and_window_degrade": "数据深度和窗口都独立降级，必须分开做单变量实验。",
        "joint_data_window_interaction": "单因素可保留，但五轮数据与新窗口组合失效，优先改拓扑处理器。",
        "frozen_transfer_supported": "冻结权重能跨数据与窗口，D3 更像训练不稳定，需要审计优化。",
        "mixed_seed_factor_response": "两颗 seed 的因子响应不一致，暂不选择路线。",
        "protocol_invalid": "来源或协议验证失败，结果不能解释。",
    }
    return labels.get(str(gate.get("diagnosis")), str(gate.get("diagnosis")))


def _verify_source(
    *, root: Path, adjudicator: Any, expected_status: str, expected_decision: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Path]]:
    paths = {"results": root / "results.jsonl", "gate": root / "gate.json"}
    rows = _read_jsonl(paths["results"])
    persisted = _read_json(paths["gate"])
    recomputed = adjudicator(run_id=str(persisted.get("run_id", "")), rows=rows)
    if persisted != recomputed:
        raise ValueError(f"persisted source gate does not match {root}")
    if (
        persisted.get("status") != expected_status
        or persisted.get("decision") != expected_decision
    ):
        raise ValueError(f"source gate at {root} does not match frozen D4 route")
    return rows, persisted, paths


def _source_candidate(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("seed") == seed
        and row.get("model") == "runtime_spn_e4_equivariant_true"
        and row.get("runtime_structure_mode") == "true"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one correct Dialga source row for seed {seed}")
    return matches[0]


def _load_validation_dataset(
    source: dict[str, Any], seed: int, *, rounds: int
) -> tuple[DiskDifferentialDataset, dict[str, str]]:
    cache_root = Path(source["training"]["dataset_cache_root"])
    validation_root = cache_root / "dialga128" / f"r{rounds}" / "validation"
    matches = list(validation_root.glob(f"seed-{10000 + seed}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one Dialga r{rounds} validation cache for seed {seed}"
        )
    cache_dir = matches[0]
    paths = {
        "feature_sha256": cache_dir / "features.npy",
        "label_sha256": cache_dir / "labels.npy",
        "metadata_sha256": cache_dir / "metadata.json",
    }
    dataset = DiskDifferentialDataset(
        features=np.load(paths["feature_sha256"], mmap_mode="r"),
        labels=np.load(paths["label_sha256"], mmap_mode="r"),
        metadata=_read_json(paths["metadata_sha256"]),
        cache_dir=cache_dir,
    )
    return dataset, {field: file_sha256(path) for field, path in paths.items()}


def _shade(color: str, seed: int) -> str:
    if seed == 0:
        return color
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round(channel * 0.72 + 255 * 0.28) for channel in rgb)
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


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
