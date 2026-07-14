from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest
from matplotlib import pyplot as plt

from blockcipher_nd.evaluation.plots import (
    _compact_label,
    _display_title,
    _plot_rc_params,
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
CASE3_MODEL_KEYS = (
    "present_nibble_invp_only_spn_only",
    "present_nibble_case3_invp_topology_residual_spn_only",
    "present_nibble_case3_shuffled_p_topology_residual_spn_only",
    "present_nibble_case3_raw_topology_residual_spn_only",
)
CASE3_VISIBLE_LABELS = (
    "InvP token mixer",
    "Case3 true InvP",
    "Case3 shuffled P",
    "Case3 raw triple",
)


def test_e6_plot_titles_and_roles_are_explained_in_chinese() -> None:
    assert "源端功能性拓扑边际目标" in _display_title(
        "i1_cross_spn_e6_functional_margin_readiness"
    )
    assert "PRESENT → GIFT-64 严格迁移" in _display_title(
        "i1_cross_spn_e6_target_readiness"
    )
    assert "本地诊断，源 seed 0" in _display_title(
        "i1_cross_spn_e6_functional_margin_8192_seed0"
    )
    assert "源 seed 0，目标 seed 3" in _display_title(
        "i1_cross_spn_e6_target_8192_source_seed0_target_seed3"
    )
    assert _compact_label(
        {"model": "present_cross_spn_typed_cell_e6_functional_margin"}
    ) == "候选：真拓扑功能边际"
    assert _compact_label(
        {
            "model": (
                "gift_cross_spn_typed_cell_e6_from_present_shuffled_placebo"
            )
        }
    ) == "安慰剂迁移：源打乱功能边际"


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
def test_metric_panel_uses_tight_range_and_distinct_validation_markers(
    metric: str,
    values: tuple[float, ...],
) -> None:
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

    with plt.rc_context(_plot_rc_params()):
        fig, axis = plt.subplots(figsize=(8.4, 2.4))
        render_metric_panel(axis, metric, series)
        fig.canvas.draw()

        lower, upper = axis.get_ylim()
        assert upper - lower < (0.15 if metric in {"accuracy", "auc"} else 0.02)
        validation_lines = [
            line for line in axis.lines if line.get_label().endswith("validation")
        ]
        assert len(validation_lines) == 4
        assert len({line.get_marker() for line in validation_lines}) == 4
        assert not [text for text in axis.texts if text.get_text() in VISIBLE_LABELS]
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


def test_case3_plot_uses_distinct_visible_role_labels(tmp_path: Path) -> None:
    results = tmp_path / "case3-results.jsonl"
    output = tmp_path / "case3-curves.svg"
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
        for index, model_key in enumerate(CASE3_MODEL_KEYS)
    ]
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = plot_jsonl_training_curves(results, output, metrics=("auc",))

    assert report["rows"] == 4
    root = ElementTree.parse(output).getroot()
    visible_text = _visible_svg_text(root)
    for label in CASE3_VISIBLE_LABELS:
        assert label in visible_text


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


def test_cross_spn_transfer_plot_explains_title_protocol_and_all_roles_in_chinese(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "curves.svg"
    models = (
        "gift_cross_spn_aligned_token_mixer_raw_anchor",
        "gift_cross_spn_typed_cell_true",
        "gift_cross_spn_typed_cell_true_from_present_true",
        "gift_cross_spn_typed_cell_true_from_present_shuffled",
        "gift_cross_spn_typed_cell_shuffled_from_present_true",
    )
    visible_labels = (
        "GIFT 原始输入基线",
        "GIFT 结构网络（从零训练）",
        "PRESENT 真结构 → GIFT 真结构",
        "PRESENT 打乱结构 → GIFT 真结构",
        "PRESENT 真结构 → GIFT 打乱结构",
    )
    rows = []
    for index, model in enumerate(models):
        rows.append(
            {
                "cipher": "GIFT-64",
                "model": model,
                "selected_model": model,
                "rounds": 6,
                "seed": 1,
                "samples_per_class": 8192,
                "pairs_per_sample": 4,
                "validation": {"samples_per_class": 4096},
                "history": [
                    {
                        "epoch": epoch,
                        "train_accuracy": 0.52 + index * 0.005 + epoch * 0.001,
                        "train_auc": 0.53 + index * 0.006 + epoch * 0.001,
                        "train_eval_loss": 0.69 - index * 0.001,
                        "val_accuracy": 0.51 + index * 0.004 + epoch * 0.001,
                        "val_auc": 0.52 + index * 0.005 + epoch * 0.001,
                        "val_loss": 0.692 - index * 0.001,
                    }
                    for epoch in (1, 2)
                ],
            }
        )
    results.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )

    plot_jsonl_training_curves(
        results,
        output,
        title="i1_gift64_cross_spn_typed_transfer_r2_seed1",
    )

    root = ElementTree.parse(output).getroot()
    visible_text = _visible_svg_text(root)
    assert "创新1：PRESENT → GIFT-64 跨 SPN 结构迁移" in visible_text
    assert "E4-R2，目标 seed 1" in visible_text
    assert "训练 8,192/类" in visible_text
    assert "验证 4,096/类" in visible_text
    assert "每样本 4 对" in visible_text
    assert "验证集结果汇总" in visible_text
    for label in visible_labels:
        assert label in visible_text
    assert "i1 gift64 cross spn typed transfer r2 seed1" not in visible_text
    for model in models:
        assert model not in visible_text

    plot_jsonl_training_curves(
        results,
        output,
        title="i1_gift64_cross_spn_typed_transfer_r3_readiness_seed1",
    )
    readiness_text = _visible_svg_text(ElementTree.parse(output).getroot())
    assert "E4-R3 就绪检查，目标 seed 1" in readiness_text
    assert "i1 gift64 cross spn typed transfer" not in readiness_text

    plot_jsonl_training_curves(
        results,
        output,
        title="i1_gift64_cross_spn_typed_transfer_r3_65536_seed1",
    )
    medium_text = _visible_svg_text(ElementTree.parse(output).getroot())
    assert "E4-R3 中等规模，目标 seed 1" in medium_text
    assert "i1 gift64 cross spn typed transfer" not in medium_text

    plot_jsonl_training_curves(
        results,
        output,
        title="i1_gift64_cross_spn_target_adaptation_r4_readiness_seed2",
    )
    r4_readiness_text = _visible_svg_text(ElementTree.parse(output).getroot())
    assert "PRESENT → GIFT-64 目标适配效率" in r4_readiness_text
    assert "E4-R4 就绪检查，目标 seed 2" in r4_readiness_text
    assert "i1 gift64 cross spn target adaptation" not in r4_readiness_text

    plot_jsonl_training_curves(
        results,
        output,
        title="i1_gift64_cross_spn_target_adaptation_r4_65536_seed3",
    )
    r4_medium_text = _visible_svg_text(ElementTree.parse(output).getroot())
    assert "E4-R4 中等规模，目标 seed 3" in r4_medium_text
    assert "i1 gift64 cross spn target adaptation" not in r4_medium_text


def test_e5_plot_explains_source_objective_roles_in_chinese(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "curves.svg"
    models = (
        "gift_cross_spn_typed_cell_e5_scratch",
        "gift_cross_spn_typed_cell_e5_from_present_off",
        "gift_cross_spn_typed_cell_e5_from_present_true_shuffled",
        "gift_cross_spn_typed_cell_e5_from_present_shuffled_placebo",
    )
    labels = (
        "GIFT 从零训练",
        "迁移基线：源辅助损失关闭",
        "候选迁移：源真拓扑 vs 打乱拓扑",
        "安慰剂迁移：源打乱 vs 打乱",
    )
    rows = [
        {
            "cipher": "GIFT-64",
            "model": model,
            "selected_model": model,
            "rounds": 6,
            "seed": 3,
            "samples_per_class": 8192,
            "pairs_per_sample": 4,
            "validation": {"samples_per_class": 4096},
            "history": [
                {
                    "epoch": 1,
                    "train_accuracy": 0.52 + index * 0.01,
                    "train_auc": 0.53 + index * 0.01,
                    "train_eval_loss": 0.69 - index * 0.001,
                    "val_accuracy": 0.51 + index * 0.01,
                    "val_auc": 0.52 + index * 0.01,
                    "val_loss": 0.692 - index * 0.001,
                }
            ],
        }
        for index, model in enumerate(models)
    ]
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    plot_jsonl_training_curves(
        results,
        output,
        title="i1_cross_spn_e5_target_8192_source_seed0_target_seed3",
    )

    visible_text = _visible_svg_text(ElementTree.parse(output).getroot())
    assert "创新1 E5-R0：PRESENT → GIFT-64 一轮迁移门控" in visible_text
    assert "源 seed 0，目标 seed 3" in visible_text
    for label in labels:
        assert label in visible_text
    for model in models:
        assert model not in visible_text


def test_plot_without_history_uses_generic_subtitle(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "curves.svg"
    results.write_text(
        json.dumps(
            {
                "cipher": "PRESENT-80",
                "model": "present_nibble_invp_only_spn_only",
                "rounds": 8,
                "seed": 0,
                "metrics": {"auc": 0.51},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = plot_jsonl_training_curves(results, output, title="history_free_result")

    assert report["rows"] == 1
    assert report["series"] == 0
    root = ElementTree.parse(output).getroot()
    visible_text = _visible_svg_text(root)
    assert "0 个模型/对照" in visible_text
    assert "验证集重点显示" in visible_text
