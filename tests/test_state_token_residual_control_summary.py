from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.cli.summarize_state_token_residual_controls import (
    main as summarize_state_token_main,
)


def test_state_token_residual_control_summary_holds_when_drop_coordinate_matches_candidate(
    tmp_path: Path,
):
    candidate = [
        _write_report(tmp_path / f"candidate_seed{seed}.json", seed=seed, base_auc=0.90, validation_auc=0.92)
        for seed in [0, 1]
    ]
    dropcoord = [
        _write_report(tmp_path / f"dropcoord_seed{seed}.json", seed=seed, base_auc=0.90, validation_auc=0.921)
        for seed in [0, 1]
    ]
    coordshuffle = [
        _write_report(tmp_path / f"coordshuffle_seed{seed}.json", seed=seed, base_auc=0.90, validation_auc=0.91)
        for seed in [0, 1]
    ]
    labelshuffle = [
        _write_report(tmp_path / f"labelshuffle_seed{seed}.json", seed=seed, base_auc=0.90, validation_auc=0.50)
        for seed in [0, 1]
    ]
    output = tmp_path / "summary.json"

    status = summarize_state_token_main(
        [
            "--candidate-reports",
            *[str(path) for path in candidate],
            "--drop-coordinate-reports",
            *[str(path) for path in dropcoord],
            "--coordinate-shuffle-reports",
            *[str(path) for path in coordshuffle],
            "--label-shuffle-reports",
            *[str(path) for path in labelshuffle],
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "hold"
    assert report["decision"] == "hold_state_token_coordinate_controls"
    assert report["seed_count"] == 2
    assert report["failing_seed_count"] == 2
    assert report["failing_control_event_count"] == 2
    assert report["next_action"]["branch"] == "do_not_promote_state_token_coordinate_route"
    assert all(seed_report["candidate_delta_vs_drop_coordinate_auc"] < 0.0 for seed_report in report["seed_reports"])


def test_state_token_residual_control_summary_passes_when_candidate_beats_controls(
    tmp_path: Path,
):
    candidate = [
        _write_report(tmp_path / f"candidate_seed{seed}.json", seed=seed, base_auc=0.90, validation_auc=0.94)
        for seed in [0, 1]
    ]
    dropcoord = [
        _write_report(tmp_path / f"dropcoord_seed{seed}.json", seed=seed, base_auc=0.90, validation_auc=0.91)
        for seed in [0, 1]
    ]
    coordshuffle = [
        _write_report(tmp_path / f"coordshuffle_seed{seed}.json", seed=seed, base_auc=0.90, validation_auc=0.905)
        for seed in [0, 1]
    ]
    labelshuffle = [
        _write_report(tmp_path / f"labelshuffle_seed{seed}.json", seed=seed, base_auc=0.90, validation_auc=0.50)
        for seed in [0, 1]
    ]
    output = tmp_path / "summary.json"

    status = summarize_state_token_main(
        [
            "--candidate-reports",
            *[str(path) for path in candidate],
            "--drop-coordinate-reports",
            *[str(path) for path in dropcoord],
            "--coordinate-shuffle-reports",
            *[str(path) for path in coordshuffle],
            "--label-shuffle-reports",
            *[str(path) for path in labelshuffle],
            "--min-base-delta",
            "0.01",
            "--min-control-delta",
            "0.01",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "pass"
    assert report["decision"] == "state_token_residual_controls_pass"
    assert report["failing_seed_count"] == 0
    assert report["failing_control_event_count"] == 0
    assert report["next_action"]["branch"] == "eligible_for_local_residual_pool_screen"


def _write_report(path: Path, *, seed: int, base_auc: float, validation_auc: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "seed": seed,
                "model_key": "present_state_token_residual_correction",
                "coordinate_mode": "additive",
                "validation_base_logit_mean_metrics": {"auc": base_auc, "accuracy": 0.5},
                "validation_metrics": {"auc": validation_auc, "accuracy": 0.5},
                "delta_validation_corrected_vs_base_logit_mean_auc": validation_auc - base_auc,
                "claim_scope": "test diagnostic only",
            }
        ),
        encoding="utf-8",
    )
    return path
