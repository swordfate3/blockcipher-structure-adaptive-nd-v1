from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from blockcipher_nd.engine.matrix_runner import parse_args
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment
from blockcipher_nd.registry.cipher_profiles import CipherProfile
from blockcipher_nd.tasks.innovation1.spn_candidate_evidence import make_candidate_dataset


def test_matrix_runner_lives_in_engine_package():
    args = parse_args(["--ciphers", "speck32", "--models", "mlp", "--rounds", "1"])

    assert args.ciphers == ["speck32"]
    assert args.models == ["mlp"]
    assert args.rounds == [1]
    assert args.learning_rate == 1e-3


def test_removed_legacy_experiment_and_generated_script_roots():
    assert not Path("experiments").exists()
    assert not Path("scripts/generated").exists()
    assert not Path("src/blockcipher_nd/innovation_one.py").exists()
    assert Path("configs/experiment/innovation1").is_dir()
    assert Path("configs/remote").is_dir()


def test_cipher_profile_lives_in_registry():
    profile = CipherProfile.present80()

    assert profile.name == "PRESENT-80"
    assert profile.structure == "SPN"


def test_scripts_are_thin_package_entrypoints():
    wrappers = [
        Path("scripts/train"),
        Path("scripts/smoke"),
        Path("scripts/spn-candidate-evidence"),
        Path("scripts/spn-active-pattern"),
        Path("scripts/audit-spn-features"),
        Path("scripts/validate-results"),
        Path("scripts/evaluate-zhang-wang-checkpoint"),
    ]

    for wrapper in wrappers:
        text = wrapper.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) <= 12, wrapper
        assert "blockcipher_nd.cli" in text


def test_differential_data_layer_has_small_modules():
    generator = Path("src/blockcipher_nd/data/differential/generator.py")
    rows = Path("src/blockcipher_nd/data/differential/rows.py")
    metadata = Path("src/blockcipher_nd/data/differential/metadata.py")
    validation = Path("src/blockcipher_nd/data/differential/validation.py")

    assert generator.exists()
    assert rows.exists()
    assert metadata.exists()
    assert validation.exists()
    assert len(generator.read_text(encoding="utf-8").splitlines()) <= 80


def test_present_inception_mcnd_is_split_by_architecture():
    facade = Path("src/blockcipher_nd/models/structure/spn/present_inception_mcnd.py")
    expected_modules = [
        Path("src/blockcipher_nd/models/structure/spn/present_inception_blocks.py"),
        Path("src/blockcipher_nd/models/structure/spn/present_inception_pair.py"),
        Path("src/blockcipher_nd/models/structure/spn/present_inception_matrix.py"),
        Path("src/blockcipher_nd/models/structure/spn/present_inception_global_matrix.py"),
        Path("src/blockcipher_nd/models/structure/spn/present_inception_pair_stack.py"),
    ]

    assert facade.exists()
    assert len(facade.read_text(encoding="utf-8").splitlines()) <= 60
    for module in expected_modules:
        assert module.exists()


def test_engine_task_runner_is_pipeline_orchestration():
    runner = Path("src/blockcipher_nd/engine/task_runner.py")
    expected_modules = [
        Path("src/blockcipher_nd/engine/task_config.py"),
        Path("src/blockcipher_nd/engine/pretraining.py"),
        Path("src/blockcipher_nd/engine/results.py"),
    ]

    assert runner.exists()
    assert len(runner.read_text(encoding="utf-8").splitlines()) <= 160
    for module in expected_modules:
        assert module.exists()


def test_candidate_evidence_cache_writes_and_reuses(tmp_path):
    progress_path = tmp_path / "progress.jsonl"
    features, labels = make_candidate_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=3,
        samples_per_class=4,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_mcnd",
        key_rotation_interval=2,
        beam_width=2,
        depth=2,
        feature_cache_root=tmp_path / "candidate_cache",
        feature_cache_chunk_size=2,
        progress_output=progress_path,
        split="train",
    )

    assert features.shape == (8, 126)
    assert labels.shape == (8,)
    assert set(np.unique(labels).tolist()) == {0, 1}

    reused_features, reused_labels = make_candidate_dataset(
        rounds=7,
        key=0,
        input_difference=0x9,
        seed=3,
        samples_per_class=4,
        pairs_per_sample=2,
        negative_mode="encrypted_random_plaintexts",
        sample_structure="zhang_wang_case2_mcnd",
        key_rotation_interval=2,
        beam_width=2,
        depth=2,
        feature_cache_root=tmp_path / "candidate_cache",
        feature_cache_chunk_size=2,
        progress_output=progress_path,
        split="train",
    )

    assert np.array_equal(np.asarray(features), np.asarray(reused_features))
    assert np.array_equal(np.asarray(labels), np.asarray(reused_labels))
    progress_text = progress_path.read_text(encoding="utf-8")
    assert "candidate_cache_done" in progress_text
    assert "candidate_cache_reuse" in progress_text


def test_result_plan_alignment_is_planning_api(tmp_path):
    plan_path = tmp_path / "plan.csv"
    result_path = tmp_path / "results.jsonl"
    plan_row = {
        "cipher": "speck32",
        "model": "mlp",
        "rounds": "1",
        "seed": "0",
        "samples_per_class": "8",
        "pairs_per_sample": "1",
    }
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan_row))
        writer.writeheader()
        writer.writerow(plan_row)
    result_path.write_text(
        json.dumps(
            {
                "cipher": "speck32",
                "model": "mlp",
                "rounds": 1,
                "seed": 0,
                "samples_per_class": 8,
                "pairs_per_sample": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_result_plan_alignment(plan_path, result_path)

    assert report["status"] == "pass"
    assert report["plan_rows"] == 1
    assert report["result_rows"] == 1
