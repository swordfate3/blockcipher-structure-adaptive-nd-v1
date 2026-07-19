from __future__ import annotations

import csv
import json
from pathlib import Path

from blockcipher_nd.cli.plot_innovation2_present_cancellation_provider_feasibility import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_cancellation_provider_feasibility import (
    EXPECTED_PANEL_KEYS,
    PANEL_MASK_FAMILIES,
    SOURCE_SPECS,
    CancellationProviderConfig,
    adjudicate_provider_feasibility,
    select_frozen_panel,
)


def _write_labels(path: Path) -> None:
    fieldnames = (
        "structure_index",
        "structure_id",
        "structure_role",
        "active_mask_hex",
        "mask_index",
        "mask_id",
        "mask_family",
        "mask_hex",
        "mask_weight",
        "status",
    )
    family_to_weight = {
        "nibble": 4,
        "player_pair": 2,
        "same_nibble_pair": 2,
        "adjacent_nibble_pair": 2,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (structure, mask), family in zip(
            EXPECTED_PANEL_KEYS,
            PANEL_MASK_FAMILIES * 3,
            strict=True,
        ):
            writer.writerow(
                {
                    "structure_index": structure,
                    "structure_id": f"cube_{structure:03d}",
                    "structure_role": "coordinate_nibble_pair",
                    "active_mask_hex": f"0x{structure + 1:016X}",
                    "mask_index": mask,
                    "mask_id": f"mask_{mask:03d}",
                    "mask_family": family,
                    "mask_hex": f"0x{mask:016X}",
                    "mask_weight": family_to_weight[family],
                    "status": "unknown",
                }
            )


def _providers() -> list[dict[str, object]]:
    return [
        {
            "provider_id": "support",
            "display_name": "support",
            "target_semantics_match": True,
            "sound_certificate": True,
            "currently_executable": True,
            "cancellation_aware": False,
            "within_frozen_cap": True,
            "real_present_r5": True,
            "finite_key_voting": False,
            "nontrivial_positive_certificates": 0,
            "panel_resolved": 0,
            "limitation": "not cancellation aware",
        },
        {
            "provider_id": "exact",
            "display_name": "exact",
            "target_semantics_match": True,
            "sound_certificate": True,
            "currently_executable": False,
            "cancellation_aware": True,
            "within_frozen_cap": False,
            "real_present_r5": True,
            "finite_key_voting": False,
            "nontrivial_positive_certificates": 0,
            "panel_resolved": 0,
            "limitation": "over cap",
        },
    ]


def test_e97_source_contract_and_query_panel_are_frozen(tmp_path: Path) -> None:
    assert len(SOURCE_SPECS) == 8
    labels = tmp_path / "labels.csv"
    _write_labels(labels)

    panel = select_frozen_panel(labels)

    assert len(panel) == 12
    assert tuple((row["structure_index"], row["mask_index"]) for row in panel) == (
        EXPECTED_PANEL_KEYS
    )
    assert {row["mask_family"] for row in panel} == set(PANEL_MASK_FAMILIES)
    assert all(row["e97_status"] == "unresolved" for row in panel)


def test_e97_gate_holds_without_eligible_nontrivial_provider(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _write_labels(labels)
    panel = select_frozen_panel(labels)

    gate = adjudicate_provider_feasibility(
        CancellationProviderConfig(), {"sources": True}, panel, _providers()
    )

    assert gate["status"] == "hold"
    assert gate["decision"] == (
        "innovation2_present_cancellation_provider_not_feasible_under_frozen_caps"
    )
    assert gate["metrics"]["eligible_providers"] == 0
    assert gate["metrics"]["panel_unresolved"] == 12
    assert gate["next_action"]["neural_training"] is False
    assert gate["next_action"]["remote_scale"] is False


def test_e97_gate_advances_only_with_panel_certificate(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _write_labels(labels)
    panel = select_frozen_panel(labels)
    panel[0]["e97_status"] = "strict_nontrivial_positive"
    panel[0]["strict_positive_certificate"] = "fixture"
    providers = _providers()
    providers[1].update(
        {
            "currently_executable": True,
            "within_frozen_cap": True,
            "nontrivial_positive_certificates": 1,
        }
    )

    gate = adjudicate_provider_feasibility(
        CancellationProviderConfig(), {"sources": True}, panel, providers
    )

    assert gate["status"] == "pass"
    assert gate["decision"] == "innovation2_present_cancellation_provider_feasible"
    assert gate["next_action"]["label_expansion"] is True
    assert gate["next_action"]["neural_training"] is False


def test_e97_plot_has_clear_chinese_scope(tmp_path: Path) -> None:
    providers = _providers()
    gate = {
        "decision": "innovation2_present_cancellation_provider_not_feasible_under_frozen_caps",
        "metrics": {
            "provider_rows": providers,
            "providers_audited": 2,
            "semantics_matching_providers": 2,
            "cancellation_aware_providers": 1,
            "eligible_providers": 0,
            "strict_nontrivial_present_positives": 0,
            "panel_queries": 12,
            "panel_resolved": 0,
            "panel_unresolved": 12,
        },
    }
    summary = tmp_path / "summary.json"
    output = tmp_path / "curves.svg"
    summary.write_text(json.dumps({"gate": gate}), encoding="utf-8")

    assert plot_main(["--summary", str(summary), "--output", str(output)]) == 0
    svg = output.read_text(encoding="utf-8")
    assert "创新2 E97" in svg
    assert "高轮输出预测为什么仍不能开训" in svg
    assert "不启动高轮网络" in svg
