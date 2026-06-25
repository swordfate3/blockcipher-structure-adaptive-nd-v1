from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.engine.matrix_runner import parse_args
from blockcipher_nd.data.differential.config import DifferentialDatasetConfig
from blockcipher_nd.data.differential.generator import make_differential_dataset
from blockcipher_nd.engine.modeling import infer_pair_bits
from blockcipher_nd.features.registry import (
    encode_ciphertext_pair,
    is_supported_feature_encoding,
    pair_bits_for_encoding,
)
from blockcipher_nd.planning.result_alignment import validate_result_plan_alignment
from blockcipher_nd.planning.matrix import build_tasks
from blockcipher_nd.registry.cipher_profiles import CipherProfile
from blockcipher_nd.registry.cipher_factory import build_cipher
from blockcipher_nd.registry.model_factory import build_model
from blockcipher_nd.tasks.innovation1.spn_candidate_evidence import make_candidate_dataset


def test_matrix_runner_lives_in_engine_package():
    args = parse_args(["--ciphers", "speck32", "--models", "mlp", "--rounds", "1"])

    assert args.ciphers == ["speck32"]
    assert args.models == ["mlp"]
    assert args.rounds == [1]
    assert args.learning_rate == 1e-3


def test_official_epoch_cyclic_lr_matches_zhang_wang_schedule():
    from blockcipher_nd.training.optim import make_scheduler
    from blockcipher_nd.training.types import TrainingConfig

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    config = TrainingConfig(
        epochs=12,
        batch_size=4,
        learning_rate=0.0001,
        lr_scheduler="official_cyclic",
        max_learning_rate=0.002,
    )
    scheduler = make_scheduler(optimizer, config, train_size=16)

    observed = []
    for epoch in range(1, 12):
        scheduler.step_epoch(epoch)
        observed.append(round(optimizer.param_groups[0]["lr"], 10))

    assert observed[0] == 0.002
    assert observed[9] == 0.0001
    assert observed[10] == 0.002
    assert observed[:10] == sorted(observed[:10], reverse=True)


def test_zhang_wang_262k_official_cyclic_plan_is_medium_diagnostic():
    plan = "configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_cyclic_r7_262k.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]

    assert task["rounds"] == 7
    assert task["samples_per_class"] == 262144
    assert task["pairs_per_sample"] == 16
    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["model_key"] == "present_zhang_wang_keras_mcnd"
    assert task["lr_scheduler"] == "official_cyclic"
    assert task["max_learning_rate"] == 0.002
    assert task["checkpoint_metric"] == "val_auc"
    assert task["restore_best_checkpoint"] is True


def test_zhang_wang_1m_official_cyclic_plan_is_single_seed_paper_scale():
    plan = "configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_cyclic_r7_1m.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]

    assert task["rounds"] == 7
    assert task["seed"] == 0
    assert task["samples_per_class"] == 1_000_000
    assert task["pairs_per_sample"] == 16
    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["model_key"] == "present_zhang_wang_keras_mcnd"
    assert task["lr_scheduler"] == "official_cyclic"
    assert task["max_learning_rate"] == 0.002
    assert task["checkpoint_metric"] == "val_auc"
    assert task["restore_best_checkpoint"] is True
    assert "SINGLE_SEED" in task["matching_evidence"]


def test_present_nibble_paligned_mcnd_smoke_plan_preserves_official_protocol():
    plan = "configs/experiment/innovation1/innovation1_spn_present_nibble_paligned_mcnd_smoke.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]

    assert task["rounds"] == 7
    assert task["samples_per_class"] == 8
    assert task["pairs_per_sample"] == 16
    assert task["feature_encoding"] == "ciphertext_pair_bits"
    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert task["negative_mode"] == "encrypted_random_plaintexts"
    assert task["model_key"] == "present_nibble_paligned_mcnd"
    assert task["model_options"]["blocks"] == 1
    assert task["model_options"]["spn_mixer_depth"] == 1
    assert "SMOKE only" in task["matching_evidence"]


def test_present_nibble_paligned_mcnd_64k_screen_plan_is_same_budget_diagnostic():
    plan = "configs/experiment/innovation1/innovation1_spn_present_nibble_paligned_mcnd_r7_64k_screen.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_mcnd",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["samples_per_class"] == 65536
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 65536/class diagnostic" in task["matching_evidence"]


def test_present_nibble_paligned_mcnd_262k_scalecheck_plan_is_same_budget_diagnostic():
    plan = "configs/experiment/innovation1/innovation1_spn_present_nibble_paligned_mcnd_r7_262k_scalecheck.csv"
    args = parse_args(["--plan", plan])
    tasks = build_tasks(args)

    assert [task["model_key"] for task in tasks] == [
        "present_zhang_wang_keras_mcnd",
        "present_nibble_paligned_mcnd",
    ]
    for task in tasks:
        assert task["rounds"] == 7
        assert task["samples_per_class"] == 262144
        assert task["pairs_per_sample"] == 16
        assert task["feature_encoding"] == "ciphertext_pair_bits"
        assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
        assert task["negative_mode"] == "encrypted_random_plaintexts"
        assert task["lr_scheduler"] == "official_cyclic"
        assert task["max_learning_rate"] == 0.002
        assert task["checkpoint_metric"] == "val_auc"
        assert task["restore_best_checkpoint"] is True
        assert "MEDIUM 262144/class diagnostic" in task["matching_evidence"]


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
        Path("scripts/plot-results"),
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


def test_zhang_wang_official_anchor_plan_generates_dataset():
    plan = "configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_anchor_smoke.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]
    cipher = build_cipher(task["cipher_key"], rounds=task["rounds"], key=task["train_key"])

    dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=task["input_difference"],
            samples_per_class=2,
            seed=task["seed"],
            feature_encoding=task["feature_encoding"],
            pairs_per_sample=task["pairs_per_sample"],
            negative_mode=task["negative_mode"],
            sample_structure=task["sample_structure"],
        )
    )

    assert task["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert dataset.features.shape == (4, 2048)
    assert dataset.metadata["sample_structure"] == "zhang_wang_case2_official_mcnd"
    assert dataset.metadata["key_schedule"] == "per_pair_random"
    assert set(np.unique(dataset.labels).tolist()) == {0, 1}


def test_present_nibble_paligned_view_encodes_delta_and_inverse_p_layer():
    cipher = build_cipher("present80", rounds=7, key=0)
    left = 0x0123456789ABCDEF
    right = 0x1111111111111111

    encoded = encode_ciphertext_pair(
        left,
        right,
        width=64,
        feature_encoding="present_nibble_paligned_view",
        cipher=cipher,
    )
    words = _decode_present_cell_matrix_words(encoded, word_count=2, width=64)

    assert is_supported_feature_encoding("present_nibble_paligned_view")
    assert pair_bits_for_encoding(64, "present_nibble_paligned_view") == 128
    assert words[0] == left ^ right
    assert words[1] == cipher.inverse_permutation_layer(left ^ right)


def test_present_nibble_paligned_mcnd_derives_same_spn_view_as_feature_encoder():
    from blockcipher_nd.features.encoders.bitwise import pair_to_bits
    from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
        PresentNibblePAlignedMCNDDistinguisher,
    )

    cipher = build_cipher("present80", rounds=7, key=0)
    left = 0x0123456789ABCDEF
    right = 0x1111111111111111
    model = PresentNibblePAlignedMCNDDistinguisher(
        input_bits=128,
        pair_bits=128,
        base_channels=4,
        blocks=1,
        spn_mixer_depth=1,
    )
    raw = torch.tensor([pair_to_bits(left, right, 64)], dtype=torch.float32)
    expected = encode_ciphertext_pair(
        left,
        right,
        width=64,
        feature_encoding="present_nibble_paligned_view",
        cipher=cipher,
    )

    observed = model._present_nibble_paligned_view(raw).reshape(-1).to(torch.uint8).tolist()

    assert observed == expected


def test_zhang_wang_official_anchor_uses_independent_key_per_basic_pair(monkeypatch):
    from blockcipher_nd.ciphers.spn.present import Present80

    observed_keys: list[int] = []
    original_encrypt = Present80.encrypt

    def record_key(self, plaintext):
        observed_keys.append(self.key)
        return original_encrypt(self, plaintext)

    monkeypatch.setattr(Present80, "encrypt", record_key)
    cipher = build_cipher("present80", rounds=7, key=0)

    make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x9,
            samples_per_class=1,
            seed=0,
            shuffle=False,
            feature_encoding="ciphertext_pair_bits",
            pairs_per_sample=16,
            negative_mode="encrypted_random_plaintexts",
            sample_structure="zhang_wang_case2_official_mcnd",
        )
    )

    positive_keys = observed_keys[:32]
    negative_keys = observed_keys[32:]
    assert len(positive_keys) == 32
    assert len(negative_keys) == 32
    assert len(set(positive_keys[::2])) == 16
    assert len(set(negative_keys[::2])) == 16
    for pair_index in range(16):
        offset = pair_index * 2
        assert positive_keys[offset] == positive_keys[offset + 1]
        assert negative_keys[offset] == negative_keys[offset + 1]


def test_zhang_wang_official_anchor_model_alias_builds():
    plan = "configs/experiment/innovation1/innovation1_spn_present_zhang_wang2022_keras_official_anchor_smoke.csv"
    args = parse_args(["--plan", plan])
    task = build_tasks(args)[0]
    pair_bits = infer_pair_bits(64, task["feature_encoding"])

    model = build_model(
        task["model_key"],
        input_bits=pair_bits * task["pairs_per_sample"],
        hidden_bits=8,
        pair_bits=pair_bits,
        structure="SPN",
        model_options=task["model_options"],
    )

    assert model.__class__.__name__ == "PresentZhangWangKerasMCNDDistinguisher"
    with torch.no_grad():
        logits = model(torch.zeros(2, pair_bits * task["pairs_per_sample"]))
    assert logits.shape == (2, 1)


def test_present_nibble_paligned_mcnd_model_alias_builds():
    pair_bits = infer_pair_bits(64, "ciphertext_pair_bits")
    model = build_model(
        "present_nibble_paligned_mcnd",
        input_bits=pair_bits * 16,
        hidden_bits=4,
        pair_bits=pair_bits,
        structure="SPN",
        model_options={"blocks": 1, "spn_mixer_depth": 1},
    )

    assert model.__class__.__name__ == "PresentNibblePAlignedMCNDDistinguisher"
    with torch.no_grad():
        logits = model(torch.zeros(2, pair_bits * 16))
    assert logits.shape == (2, 1)


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


def test_training_history_plot_outputs_svg_and_csv(tmp_path):
    from blockcipher_nd.evaluation.plots import plot_jsonl_training_curves, write_history_csv

    results_path = tmp_path / "results.jsonl"
    svg_path = tmp_path / "curves.svg"
    csv_path = tmp_path / "history.csv"
    results_path.write_text(
        json.dumps(
            {
                "cipher": "SPECK32/64",
                "model": "mlp",
                "selected_model": "mlp",
                "rounds": 1,
                "seed": 0,
                "samples_per_class": 8,
                "pairs_per_sample": 1,
                "history": [
                    {
                        "epoch": 1.0,
                        "train_loss": 0.7,
                        "train_eval_loss": 0.69,
                        "train_accuracy": 0.55,
                        "train_auc": 0.58,
                        "val_loss": 0.71,
                        "val_accuracy": 0.5,
                        "val_auc": 0.52,
                        "learning_rate": 0.001,
                    },
                    {
                        "epoch": 2.0,
                        "train_loss": 0.65,
                        "train_eval_loss": 0.64,
                        "train_accuracy": 0.65,
                        "train_auc": 0.7,
                        "val_loss": 0.68,
                        "val_accuracy": 0.6,
                        "val_auc": 0.62,
                        "learning_rate": 0.001,
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_report = plot_jsonl_training_curves(results_path, svg_path)
    csv_report = write_history_csv(results_path, csv_path)

    assert plot_report["series"] == 6
    assert csv_report["rows"] == 2
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "Matplotlib" in svg_text
    assert "Epoch" in svg_text
    assert "Accuracy (%)" in svg_text
    assert "AUC (%)" in svg_text
    assert "Validation summary" in svg_text
    assert "62.0% @ e2" in svg_text
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "train_accuracy" in csv_text
    assert "val_auc" in csv_text


def test_training_history_records_train_and_validation_metrics():
    from blockcipher_nd.ciphers.arx.speck import Speck32_64
    from blockcipher_nd.data.differential import DifferentialDatasetConfig
    from blockcipher_nd.data.differential.generator import make_differential_dataset
    from blockcipher_nd.models.baseline.mlp import MlpDistinguisher
    from blockcipher_nd.training import TrainingConfig, train_binary_classifier

    cipher = Speck32_64(rounds=1, key=0)
    train_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x40,
            samples_per_class=8,
            seed=0,
            shuffle=True,
        )
    )
    validation_dataset = make_differential_dataset(
        DifferentialDatasetConfig(
            cipher=cipher,
            input_difference=0x40,
            samples_per_class=8,
            seed=1,
            shuffle=True,
        )
    )
    model = MlpDistinguisher(input_bits=train_dataset.features.shape[1], hidden_bits=8)

    result = train_binary_classifier(
        model,
        train_dataset,
        validation_dataset,
        TrainingConfig(epochs=1, batch_size=4, device="cpu"),
    )

    epoch = result.history[0]
    for key in [
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_auc",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "learning_rate",
    ]:
        assert key in epoch


def _decode_present_cell_matrix_words(encoded: list[int], *, word_count: int, width: int) -> list[int]:
    cells = [[0, 0, 0, 0] for _ in range((word_count * width) // 4)]
    offset = 0
    for bit_index in range(4):
        for cell in cells:
            cell[bit_index] = encoded[offset]
            offset += 1
    bits = [bit for cell in cells for bit in cell]
    words = []
    for start in range(0, len(bits), width):
        value = 0
        for bit in bits[start : start + width]:
            value = (value << 1) | bit
        words.append(value)
    return words
