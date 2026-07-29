from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.plot_uknit_family_multicipher_gradient_conflict_k1ap import (
    render_k1ap_svg,
)
from blockcipher_nd.data.differential import DiskDifferentialDataset
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_gradient_conflict_k1ap import (
    BATCH_TRIPLETS,
    CONDITIONS,
    EXPECTED_NORM_ROWS,
    EXPECTED_PAIR_ROWS,
    EXPECTED_SUMMARY_ROWS,
    PARAMETER_GROUPS,
    adjudicate,
    load_and_validate_config,
    load_authority,
    make_stratified_batches,
    measure_gradient_vectors,
)
from blockcipher_nd.tasks.innovation1.uknit_family_multicipher_shared_weight_k1ao import (
    EXPECTED_CIPHERS,
    build_runtime_model,
)


def test_k1ap_config_freezes_zero_update_gradient_audit() -> None:
    config = load_and_validate_config()

    assert config["audit"]["batch_triplets_per_replica"] == 64
    assert config["audit"]["optimizer_steps"] == 0
    assert tuple(config["audit"]["conditions"]) == CONDITIONS
    assert tuple(config["audit"]["parameter_groups"]) == PARAMETER_GROUPS
    assert config["gates"]["remote_scale"] == "no"


def test_k1ap_stratified_batches_cover_every_row_once(tmp_path: Path) -> None:
    generator = np.random.default_rng(20260729)
    dataset = DiskDifferentialDataset(
        features=generator.integers(0, 2, size=(4096, 16), dtype=np.uint8),
        labels=np.concatenate(
            (
                np.ones(2048, dtype=np.uint8),
                np.zeros(2048, dtype=np.uint8),
            )
        ),
        metadata={},
        cache_dir=tmp_path,
    )

    batches = make_stratified_batches(dataset, seed=17)

    assert len(batches) == BATCH_TRIPLETS
    assert sorted(np.concatenate(batches).tolist()) == list(range(4096))
    assert all(int(dataset.labels[indices].sum()) == 32 for indices in batches)


def test_k1ap_authority_and_one_gradient_measurement_are_immutable() -> None:
    config = load_and_validate_config()
    readiness, datasets, checkpoints, checks, _dataset_rows = load_authority(config)
    cipher = readiness["ciphers"][0]
    model = build_runtime_model(cipher, readiness["model"])
    model.load_state_dict(checkpoints[0]["state_dict"], strict=True)
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    dataset = datasets[("uknit64", 3, "train_seen")]
    indices = make_stratified_batches(dataset, seed=19)[0]
    features = torch.as_tensor(
        np.array(dataset.features[indices], copy=True), dtype=torch.float32
    )
    labels = torch.as_tensor(
        np.array(dataset.labels[indices], copy=True), dtype=torch.float32
    ).reshape(-1, 1)

    vectors, loss = measure_gradient_vectors(
        model=model,
        features=features,
        labels=labels,
        structure=model.runtime_structure,
        transition_branch_enabled=True,
    )

    assert all(checks.values()), checks
    assert set(checkpoints) == {0, 1}
    assert loss > 0.0
    assert vectors["all_trainable"].numel() == 219_320
    assert vectors["transition_semantic"].numel() > 0
    assert float(torch.linalg.vector_norm(vectors["transition_semantic"])) > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(value, state_before[name])
        for name, value in model.state_dict().items()
    )


def test_k1ap_gate_opens_pcgrad_only_for_stable_two_replica_conflict() -> None:
    pair_rows, norm_rows, summaries = synthetic_audit_rows(conflict=True)

    gate = adjudicate(
        source_checks={"source": True},
        state_checks=synthetic_state_checks(),
        pair_rows=pair_rows,
        norm_rows=norm_rows,
        summaries=summaries,
    )

    assert gate["status"] == "pass"
    assert gate["decision"].endswith("systematic_gradient_conflict_supported")
    assert gate["stable_conflict_pairs"] == ["uknit64__midori64"]
    assert "PCGrad" in gate["next_action"]


def test_k1ap_gate_returns_to_representation_without_trigger() -> None:
    pair_rows, norm_rows, summaries = synthetic_audit_rows(conflict=False)

    gate = adjudicate(
        source_checks={"source": True},
        state_checks=synthetic_state_checks(),
        pair_rows=pair_rows,
        norm_rows=norm_rows,
        summaries=summaries,
    )

    assert gate["status"] == "hold"
    assert gate["decision"].endswith("optimizer_conflict_not_supported")
    assert gate["stable_conflict_pairs"] == []
    assert gate["stable_gradient_norm_imbalance"] is False


def test_k1ap_plot_requires_complete_summary(tmp_path: Path) -> None:
    _pair_rows, _norm_rows, summaries = synthetic_audit_rows(conflict=False)
    output = tmp_path / "curves.svg"

    report = render_k1ap_svg({"status": "pass"}, summaries, output)

    assert output.is_file()
    assert report["summary_rows"] == EXPECTED_SUMMARY_ROWS
    assert report["optimizer_steps"] == 0
    assert report["training_auc_claim_present"] is False


def synthetic_audit_rows(
    *,
    conflict: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cipher_pairs = [
        f"{left}__{right}"
        for left_index, left in enumerate(EXPECTED_CIPHERS)
        for right in EXPECTED_CIPHERS[left_index + 1 :]
    ]
    pair_rows = []
    norm_rows = []
    summaries = []
    for replica in (0, 1):
        for condition in CONDITIONS:
            for group in PARAMETER_GROUPS:
                for cipher_pair in cipher_pairs:
                    median = (
                        -0.10
                        if conflict
                        and condition == "correct_runtime"
                        and group == "all_trainable"
                        and cipher_pair == "uknit64__midori64"
                        else 0.10
                    )
                    frequency = 0.75 if median < 0.0 else 0.10
                    summaries.append(
                        {
                            "metric_type": "pairwise_cosine",
                            "replica": replica,
                            "condition": condition,
                            "parameter_group": group,
                            "cipher_pair": cipher_pair,
                            "median_cosine": median,
                            "negative_cosine_frequency": frequency,
                            "optimizer_steps": 0,
                        }
                    )
                    pair_rows.extend(
                        {
                            "replica": replica,
                            "condition": condition,
                            "parameter_group": group,
                            "cipher_pair": cipher_pair,
                            "cosine": median,
                            "optimizer_steps": 0,
                        }
                        for _batch in range(BATCH_TRIPLETS)
                    )
                for cipher_index, cipher_key in enumerate(EXPECTED_CIPHERS):
                    norm = 1.0 + cipher_index * 0.5
                    summaries.append(
                        {
                            "metric_type": "gradient_norm",
                            "replica": replica,
                            "condition": condition,
                            "parameter_group": group,
                            "cipher_key": cipher_key,
                            "median_gradient_norm": norm,
                            "optimizer_steps": 0,
                        }
                    )
                    norm_rows.extend(
                        {
                            "replica": replica,
                            "condition": condition,
                            "parameter_group": group,
                            "cipher_key": cipher_key,
                            "gradient_norm": norm,
                            "optimizer_steps": 0,
                        }
                        for _batch in range(BATCH_TRIPLETS)
                    )
    assert len(pair_rows) == EXPECTED_PAIR_ROWS
    assert len(norm_rows) == EXPECTED_NORM_ROWS
    assert len(summaries) == EXPECTED_SUMMARY_ROWS
    return pair_rows, norm_rows, summaries


def synthetic_state_checks() -> dict[str, bool]:
    return {
        "replica0_state_immutable": True,
        "replica0_all_parameter_grads_none": True,
        "replica1_state_immutable": True,
        "replica1_all_parameter_grads_none": True,
    }
