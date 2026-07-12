from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves


def test_topology_residual_plot_uses_distinct_visible_role_labels(
    tmp_path: Path,
) -> None:
    model_keys = (
        "present_nibble_invp_only_spn_only",
        "present_nibble_invp_topology_residual_spn_only",
        "present_nibble_shuffled_p_topology_residual_spn_only",
        "present_nibble_delta_topology_residual_spn_only",
    )
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
        for index, model_key in enumerate(model_keys)
    ]
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = plot_jsonl_training_curves(results, output, metrics=("auc",))

    assert report["rows"] == 4
    assert report["series"] == 8
    root = ElementTree.parse(output).getroot()
    visible_text = " ".join(
        element.text or "" for element in root.iter() if element.tag.endswith("text")
    )
    for label in (
        "InvP token mixer",
        "true InvP residual",
        "shuffled P residual",
        "Delta-only residual",
    ):
        assert label in visible_text
    assert "topology residual spn only" not in visible_text
