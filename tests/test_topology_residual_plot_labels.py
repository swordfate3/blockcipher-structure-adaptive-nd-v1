from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree

import pytest
from matplotlib import pyplot as plt

from blockcipher_nd.evaluation.plots import (
    plot_jsonl_training_curves,
    render_metric_panel,
)


MODEL_KEYS = (
    "present_nibble_invp_only_spn_only",
    "present_nibble_invp_topology_residual_spn_only",
    "present_nibble_shuffled_p_topology_residual_spn_only",
    "present_nibble_delta_topology_residual_spn_only",
)
VISIBLE_LABELS = (
    "InvP token mixer",
    "true InvP residual",
    "shuffled P residual",
    "Delta-only residual",
)


def _visible_svg_text(root: ElementTree.Element) -> str:
    return " ".join(
        element.text or "" for element in root.iter() if element.tag.endswith("text")
    )


@pytest.mark.parametrize(
    ("metric", "values"),
    [
        ("accuracy", (0.500, 0.501, 0.502, 0.503)),
        ("auc", (0.501, 0.502, 0.503, 0.504)),
        ("loss", (0.690, 0.691, 0.692, 0.693)),
    ],
)
def test_metric_panel_endpoint_label_bboxes_do_not_overlap(
    metric: str,
    values: tuple[float, ...],
) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 2.4))
    series = [
        {
            "metric": metric,
            "split": "val",
            "label": model_key,
            "run_index": index,
            "model": model_key,
            "points": [(1.0, value)],
        }
        for index, (model_key, value) in enumerate(
            zip(MODEL_KEYS, values, strict=True),
            start=1,
        )
    ]

    render_metric_panel(axis, metric, series)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    labels = [text for text in axis.texts if text.get_text() in VISIBLE_LABELS]
    assert len(labels) == 4
    boxes = [label.get_window_extent(renderer) for label in labels]
    for left, right in combinations(boxes, 2):
        assert not left.overlaps(right)
    plt.close(fig)


def test_topology_residual_plot_uses_distinct_visible_role_labels(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "curves.svg"
    rows = [
        {
            "cipher": "PRESENT-80",
            "model": model_key,
            "selected_model": model_key,
            "rounds": 7,
            "seed": 0,
            "history": [
                {
                    "epoch": 1,
                    "train_auc": 0.60 + index * 0.01,
                    "val_auc": 0.61 + index * 0.01,
                }
            ],
        }
        for index, model_key in enumerate(MODEL_KEYS)
    ]
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = plot_jsonl_training_curves(results, output, metrics=("auc",))

    assert report["rows"] == 4
    assert report["series"] == 8
    root = ElementTree.parse(output).getroot()
    visible_text = _visible_svg_text(root)
    for label in VISIBLE_LABELS:
        assert label in visible_text
    assert "topology residual spn only" not in visible_text


def test_r0_like_endpoint_labels_keep_svg_aspect_ratio_bounded(tmp_path: Path) -> None:
    results = tmp_path / "r0-results.jsonl"
    output = tmp_path / "r0-curves.svg"
    rows = [
        {
            "cipher": "PRESENT-80",
            "model": model_key,
            "selected_model": model_key,
            "rounds": 7,
            "seed": 0,
            "history": [
                {
                    "epoch": 1,
                    "train_accuracy": 0.500 + index * 0.001,
                    "train_auc": 0.501 + index * 0.001,
                    "train_eval_loss": 0.690 + index * 0.001,
                    "val_accuracy": 0.502 + index * 0.001,
                    "val_auc": 0.503 + index * 0.001,
                    "val_loss": 0.691 + index * 0.001,
                }
            ],
        }
        for index, model_key in enumerate(MODEL_KEYS)
    ]
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = plot_jsonl_training_curves(results, output)

    assert report["rows"] == 4
    assert report["series"] == 24
    root = ElementTree.parse(output).getroot()
    width = float(root.attrib["width"].removesuffix("pt"))
    height = float(root.attrib["height"].removesuffix("pt"))
    _, _, view_width, view_height = map(float, root.attrib["viewBox"].split())
    assert width == view_width
    assert height == view_height
    assert height / width < 1.2
    visible_text = _visible_svg_text(root)
    for label in VISIBLE_LABELS:
        assert label in visible_text
