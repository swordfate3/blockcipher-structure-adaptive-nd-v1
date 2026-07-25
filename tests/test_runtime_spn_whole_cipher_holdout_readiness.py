from __future__ import annotations

from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_whole_cipher_holdout import (
    build_holdout_readiness,
    load_and_validate_holdout_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/innovation1/innovation1_runtime_spn_rectangle_whole_cipher_holdout_h1_2048_seed0_seed1.json"
)


def test_h1_readiness_passes_with_existing_five_cipher_cache() -> None:
    config = load_and_validate_holdout_config(CONFIG)

    manifest, gate = build_holdout_readiness(config, project_root=ROOT)

    assert gate["status"] == "pass"
    assert all(gate["checks"].values())
    assert manifest["target_training_loaded"] is False
    assert manifest["checkpoint_selection_tasks"] == list(config["source_ciphers"])


def test_h1_entrypoint_scripts_exist() -> None:
    assert (ROOT / "scripts/check-runtime-spn-whole-cipher-holdout-readiness").is_file()
    assert (ROOT / "scripts/run-runtime-spn-whole-cipher-holdout").is_file()
