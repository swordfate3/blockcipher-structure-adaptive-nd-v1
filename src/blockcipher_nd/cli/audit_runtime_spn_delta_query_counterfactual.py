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
from blockcipher_nd.tasks.innovation1.runtime_spn_delta_query_counterfactual import (
    adjudicate_same_checkpoint_delta_query,
    evaluate_same_checkpoint_delta_query,
    file_sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit U2-F delta-query use with frozen candidate checkpoints."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--u2f-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_rows = _read_jsonl(args.u2f_root / "results.jsonl")
    result_rows: list[dict[str, Any]] = []
    _write_progress(args.output_root / "progress.jsonl", "run_start", args.run_id)

    descriptor_path = Path("configs/runtime/spn/uknit64.json")
    descriptor = load_runtime_spn_descriptor(
        descriptor_path,
        rounds=2,
        round_start=2,
    )
    correct_structure = descriptor.structure
    shuffled_structure = correct_structure.shuffled_sbox_assignments(20260724)

    for seed in (0, 1):
        source = _source_candidate(source_rows, seed)
        dataset, feature_path, label_path = _load_validation_dataset(source)
        checkpoint_path = Path(source["training"]["checkpoint_output"])
        result_rows.extend(
            evaluate_same_checkpoint_delta_query(
                seed=seed,
                model_options=dict(source["training"]["model_options"]),
                checkpoint_path=checkpoint_path,
                dataset=dataset,
                correct_structure=correct_structure,
                shuffled_structure=shuffled_structure,
                checkpoint_sha256=file_sha256(checkpoint_path),
                feature_sha256=file_sha256(feature_path),
                label_sha256=file_sha256(label_path),
                descriptor_sha256=descriptor.sha256,
                batch_size=args.batch_size,
                device=args.device,
            )
        )
        _write_progress(
            args.output_root / "progress.jsonl",
            "seed_done",
            args.run_id,
            seed=seed,
        )

    _write_jsonl(args.output_root / "results.jsonl", result_rows)
    gate = adjudicate_same_checkpoint_delta_query(
        run_id=args.run_id,
        rows=result_rows,
    )
    validation = {
        "run_id": args.run_id,
        "status": "pass" if all(gate["protocol_checks"].values()) else "fail",
        "checks": gate["protocol_checks"],
        "source_paths": {"u2f_results": str(args.u2f_root / "results.jsonl")},
    }
    summary = {
        "run_id": args.run_id,
        "task": gate["task"],
        "cipher": "uKNIT-BC",
        "training_performed": False,
        "source_run": str(args.u2f_root),
        "gate": gate,
    }
    _write_json(args.output_root / "validation.json", validation)
    _write_json(args.output_root / "gate.json", gate)
    _write_json(args.output_root / "summary.json", summary)
    render_delta_query_counterfactual_svg(gate, args.output_root / "curves.svg")
    _write_progress(
        args.output_root / "progress.jsonl",
        "run_done",
        args.run_id,
        status=gate["status"],
        decision=gate["decision"],
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 1 if gate["status"] == "fail" else 0


def render_delta_query_counterfactual_svg(
    gate: dict[str, Any],
    output_path: Path,
) -> None:
    labels = ("正确 ΔU", "打乱归属 ΔU", "ΔV 身份查询")
    colors = ("#2563EB", "#D97706", "#0F9D76")
    seed_results = gate["seed_results"]
    aucs = {
        seed: (
            seed_results[str(seed)]["correct_delta_u_auc"],
            seed_results[str(seed)]["shuffled_delta_u_auc"],
            seed_results[str(seed)]["delta_v_identity_auc"],
        )
        for seed in (0, 1)
    }
    margins = (
        seed_results["0"]["correct_minus_shuffled_auc"],
        seed_results["0"]["correct_minus_identity_auc"],
        seed_results["1"]["correct_minus_shuffled_auc"],
        seed_results["1"]["correct_minus_identity_auc"],
    )
    with plt.rc_context(
        {
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10.5,
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
        figure, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
        figure.subplots_adjust(left=0.075, right=0.97, top=0.73, bottom=0.16, wspace=0.3)
        figure.suptitle(
            "创新1 uKNIT U2-G：同一权重是否真正使用运行时 ΔU 查询",
            x=0.075,
            y=0.96,
            ha="left",
            fontsize=16,
            fontweight="bold",
        )
        figure.text(
            0.075,
            0.89,
            "冻结 U2-F 两颗 seed 的最佳候选权重与验证集；只替换第三个查询 token，不重新训练。",
            ha="left",
            color="#475569",
        )
        figure.text(
            0.075,
            0.83,
            f"裁决：{_decision_text(gate)}",
            ha="left",
            color="#047857" if gate["status"] == "pass" else "#B42318",
            fontweight="bold",
        )

        x = np.arange(2, dtype=np.float64)
        width = 0.24
        for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
            values = [aucs[seed][index] for seed in (0, 1)]
            bars = axes[0].bar(
                x + (index - 1) * width,
                values,
                width,
                label=label,
                color=color,
            )
            axes[0].bar_label(
                bars,
                labels=[f"{value:.4f}" for value in values],
                padding=4 + index * 7,
                fontsize=9,
            )
        axes[0].set_title("冻结 checkpoint 的验证 AUC", loc="left", fontweight="bold")
        axes[0].set_ylabel("AUC")
        axes[0].set_xticks(x, ("seed0", "seed1"))
        axes[0].axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1)
        flat_aucs = [value for values in aucs.values() for value in values]
        lower = max(0.0, min(0.5, *flat_aucs) - 0.006)
        upper = min(1.0, max(*flat_aucs) + 0.012)
        axes[0].set_ylim(lower, upper)
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[0].legend(frameon=False, loc="upper left")

        margin_labels = (
            "seed0\n对打乱归属",
            "seed0\n对身份查询",
            "seed1\n对打乱归属",
            "seed1\n对身份查询",
        )
        bars = axes[1].bar(
            np.arange(4),
            margins,
            width=0.62,
            color=["#059669" if value >= 0.005 else "#DC2626" for value in margins],
        )
        axes[1].bar_label(
            bars,
            labels=[f"{value:+.6f}" for value in margins],
            padding=5,
            fontsize=9,
        )
        axes[1].axhline(0.0, color="#64748B", linewidth=1)
        axes[1].axhline(
            0.005,
            color="#2563EB",
            linestyle="--",
            linewidth=1.2,
            label="通过门槛 +0.005",
        )
        axes[1].set_title("正确 ΔU 相对控制的 AUC 差值", loc="left", fontweight="bold")
        axes[1].set_ylabel("AUC 差值")
        axes[1].set_xticks(np.arange(4), margin_labels)
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axes[1].legend(frameon=False, loc="upper left")
        axes[1].margins(y=0.25)
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)


def _decision_text(gate: dict[str, Any]) -> str:
    if gate["status"] == "pass":
        return "两颗 seed 的同一权重都依赖正确 ΔU 查询，可进入跨窗口复验。"
    if gate["status"] == "fail":
        return "协议检查失败，反事实 AUC 不得用于研究判断。"
    return "同一权重下查询优势未稳定保留，停止扩样并关闭该查询路线。"


def _source_candidate(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("seed") == seed
        and row.get("runtime_structure_mode") == "true"
        and row.get("training", {}).get("model_options", {}).get("cell_input_mode")
        == "state_triplet_delta_u_query"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one U2-F candidate for seed={seed}, got {len(matches)}"
        )
    return matches[0]


def _load_validation_dataset(
    source: dict[str, Any],
) -> tuple[DiskDifferentialDataset, Path, Path]:
    seed = int(source["seed"])
    cache_root = Path(source["training"]["dataset_cache_root"])
    validation_root = cache_root / "uknit64" / "r4" / "validation"
    matches = list(validation_root.glob(f"seed-{10000 + seed}_*"))
    if len(matches) != 1:
        raise ValueError(f"expected one validation cache for seed {seed}, got {len(matches)}")
    cache_dir = matches[0]
    feature_path = cache_dir / "features.npy"
    label_path = cache_dir / "labels.npy"
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset = DiskDifferentialDataset(
        features=np.load(feature_path, mmap_mode="r"),
        labels=np.load(label_path, mmap_mode="r"),
        metadata=metadata,
        cache_dir=cache_dir,
    )
    return dataset, feature_path, label_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
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


def _write_progress(
    path: Path,
    event: str,
    run_id: str,
    **details: Any,
) -> None:
    payload = {
        "event": event,
        "run_id": run_id,
        "time": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
