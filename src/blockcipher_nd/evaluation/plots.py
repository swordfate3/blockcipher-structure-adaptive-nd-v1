from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = ("accuracy", "auc", "loss")
PLOT_COLORS = {
    "train": "#2563eb",
    "val": "#dc2626",
}


def plot_jsonl_training_curves(
    results_path: Path,
    output_path: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    title: str | None = None,
) -> dict[str, Any]:
    rows = load_jsonl_rows(results_path)
    series = training_curve_series(rows, metrics=metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_training_curves_svg(series, title=title or results_path.name), encoding="utf-8")
    return {
        "rows": len(rows),
        "series": len(series),
        "output": str(output_path),
        "metrics": list(metrics),
    }


def write_history_csv(results_path: Path, output_path: Path) -> dict[str, Any]:
    rows = load_jsonl_rows(results_path)
    records = history_records(rows)
    fieldnames = [
        "run_index",
        "run_label",
        "cipher",
        "model",
        "selected_model",
        "rounds",
        "seed",
        "samples_per_class",
        "pairs_per_sample",
        "epoch",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "learning_rate",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return {
        "rows": len(records),
        "output": str(output_path),
    }


def training_curve_series(
    rows: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for run_index, row in enumerate(rows, start=1):
        label = run_label(row, run_index)
        for metric in metrics:
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"
            for split, key in (("train", train_key), ("val", val_key)):
                points = [
                    (float(item["epoch"]), float(item[key]))
                    for item in row.get("history", [])
                    if key in item and item.get(key) is not None
                ]
                if points:
                    series.append(
                        {
                            "metric": metric,
                            "split": split,
                            "label": label,
                            "points": points,
                        }
                    )
    return series


def history_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_index, row in enumerate(rows, start=1):
        label = run_label(row, run_index)
        for item in row.get("history", []):
            records.append(
                {
                    "run_index": run_index,
                    "run_label": label,
                    "cipher": row.get("cipher", ""),
                    "model": row.get("model", ""),
                    "selected_model": row.get("selected_model", ""),
                    "rounds": row.get("rounds", ""),
                    "seed": row.get("seed", ""),
                    "samples_per_class": row.get("samples_per_class", ""),
                    "pairs_per_sample": row.get("pairs_per_sample", ""),
                    **item,
                }
            )
    return records


def render_training_curves_svg(
    series: list[dict[str, Any]],
    *,
    title: str,
    width: int = 1120,
    panel_height: int = 260,
) -> str:
    metrics = list(dict.fromkeys(item["metric"] for item in series))
    if not metrics:
        metrics = list(DEFAULT_METRICS)
    margin_left = 70
    margin_right = 30
    margin_top = 56
    margin_bottom = 48
    gap = 42
    height = margin_top + len(metrics) * panel_height + max(0, len(metrics) - 1) * gap + margin_bottom
    body: list[str] = [
        svg_text(width / 2, 30, title, size=20, anchor="middle", weight="700"),
        svg_text(width - margin_right, 30, "blue=train, red=validation", size=13, anchor="end", fill="#475569"),
    ]

    for index, metric in enumerate(metrics):
        y_top = margin_top + index * (panel_height + gap)
        metric_series = [item for item in series if item["metric"] == metric]
        body.extend(
            render_metric_panel(
                metric,
                metric_series,
                x=margin_left,
                y=y_top,
                width=width - margin_left - margin_right,
                height=panel_height,
            )
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def render_metric_panel(
    metric: str,
    series: list[dict[str, Any]],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[str]:
    plot_height = height - 42
    all_points = [point for item in series for point in item["points"]]
    if not all_points:
        return [
            svg_text(x, y + 18, metric, size=16, weight="700"),
            svg_text(x, y + 60, "no history points", size=13, fill="#64748b"),
        ]
    min_epoch = min(point[0] for point in all_points)
    max_epoch = max(point[0] for point in all_points)
    min_value = min(point[1] for point in all_points)
    max_value = max(point[1] for point in all_points)
    if metric in {"accuracy", "auc"}:
        min_value = min(0.0, min_value)
        max_value = max(1.0, max_value)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5

    elements = [
        svg_text(x, y + 18, metric, size=16, weight="700"),
        f'<line x1="{x}" y1="{y + plot_height}" x2="{x + width}" y2="{y + plot_height}" stroke="#cbd5e1"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + plot_height}" stroke="#cbd5e1"/>',
        svg_text(x - 10, y + 4, format_value(max_value), size=12, anchor="end", fill="#64748b"),
        svg_text(x - 10, y + plot_height, format_value(min_value), size=12, anchor="end", fill="#64748b"),
        svg_text(x, y + plot_height + 30, f"epoch {format_value(min_epoch)}", size=12, fill="#64748b"),
        svg_text(x + width, y + plot_height + 30, f"epoch {format_value(max_epoch)}", size=12, anchor="end", fill="#64748b"),
    ]
    for item in series:
        color = PLOT_COLORS.get(item["split"], "#334155")
        path_data = " ".join(
            (
                "M" if idx == 0 else "L"
            )
            + f" {scale_x(epoch, min_epoch, max_epoch, x, width):.2f} {scale_y(value, min_value, max_value, y, plot_height):.2f}"
            for idx, (epoch, value) in enumerate(item["points"])
        )
        elements.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
        )
        last_epoch, last_value = item["points"][-1]
        elements.append(
            svg_text(
                scale_x(last_epoch, min_epoch, max_epoch, x, width) + 5,
                scale_y(last_value, min_value, max_value, y, plot_height),
                f"{item['split']} {format_value(last_value)}",
                size=11,
                fill=color,
            )
        )
    return elements


def scale_x(value: float, min_value: float, max_value: float, x: int, width: int) -> float:
    if min_value == max_value:
        return x + width / 2
    return x + (value - min_value) / (max_value - min_value) * width


def scale_y(value: float, min_value: float, max_value: float, y: int, height: int) -> float:
    return y + height - (value - min_value) / (max_value - min_value) * height


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int,
    anchor: str = "start",
    fill: str = "#0f172a",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(str(text))}</text>"
    )


def format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.4g}"


def run_label(row: dict[str, Any], run_index: int) -> str:
    return (
        f"run{run_index}: {row.get('cipher', '')} "
        f"r{row.get('rounds', '')} {row.get('model', row.get('selected_model', ''))} "
        f"seed{row.get('seed', '')}"
    )


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
