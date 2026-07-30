from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from blockcipher_nd.cli.run_uknit_family_ctspn_k1r import read_tasks
from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.models.structure.spn.exact_operator_composition import (
    COMPOSITION_STAGE_NAMES,
    composition_stage_names,
    exact_operator_composition_views,
)
from blockcipher_nd.models.structure.spn.position_histogram_residual import (
    deterministic_position_histogram,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    load_runtime_spn_descriptor,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1n import build_k1n_control
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1o import (
    _position_histograms,
)
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1r import task_map
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1s import TAPS
from blockcipher_nd.tasks.innovation1.uknit_family_ctspn_k1k import project_features
from blockcipher_nd.registry.model_factory import build_model


ROOT = Path(__file__).resolve().parents[1]
K1R_PLAN = (
    ROOT
    / "configs/experiment/innovation1/innovation1_uknit_family_ctspn_cell11_neural_attribution_k1r_2048_seed3_seed4.csv"
)
UKNIT_DESCRIPTOR = ROOT / "configs/runtime/spn/uknit64.json"


def test_exact_operator_composition_extends_two_round_prefix_to_three_rounds() -> None:
    two = load_runtime_spn_descriptor(
        UKNIT_DESCRIPTOR,
        rounds=2,
        round_start=4,
    ).structure
    three = load_runtime_spn_descriptor(
        UKNIT_DESCRIPTOR,
        rounds=3,
        round_start=3,
    ).structure
    runtime = torch.randint(
        0,
        2,
        (3, 4, 2, 64),
        generator=torch.Generator().manual_seed(20260730),
    ).float()

    two_views = exact_operator_composition_views(runtime, two)
    three_views = exact_operator_composition_views(runtime, three)
    two_histogram = deterministic_position_histogram(runtime, two)
    three_histogram = deterministic_position_histogram(runtime, three)

    assert COMPOSITION_STAGE_NAMES == composition_stage_names(2)
    assert composition_stage_names(3) == (
        "ciphertext",
        "inverse_linear_2",
        "inverse_sbox_2",
        "inverse_linear_1",
        "inverse_sbox_1",
        "inverse_linear_0",
        "inverse_sbox_0",
    )
    assert two_views.shape == (3, 4, 64, 15)
    assert three_views.shape == (3, 4, 64, 21)
    assert torch.equal(three_views[..., :15], two_views)
    assert two_histogram.shape == (3, 5, 16, 16)
    assert three_histogram.shape == (3, 7, 16, 16)
    assert torch.equal(three_histogram[:, :5], two_histogram)


def test_position_histogram_exactly_matches_k1s_t0_geometry_and_values() -> None:
    task = task_map(read_tasks(K1R_PLAN))[(3, "exact_composition")]
    model = build_k1n_control(
        task=task,
        condition="exact_composition",
        input_bits=512,
    )
    features = torch.randint(
        0,
        2,
        (7, 512),
        generator=torch.Generator().manual_seed(20260728),
    ).float()
    runtime = project_features(features, model.runtime_structure)
    observed = deterministic_position_histogram(runtime, model.runtime_structure)
    expected = _position_histograms(
        exact_operator_composition_views(runtime, model.runtime_structure),
        model.runtime_structure,
    )

    assert tuple(observed.shape[1:]) == (5, 16, 16)
    assert observed.reshape(7, -1).shape[1] == 1280
    assert np.array_equal(observed.numpy(), expected)
    assert TAPS[0].startswith("T0_exact_position_histogram")


def test_position_histogram_controls_keep_geometry_and_change_semantics() -> None:
    task = task_map(read_tasks(K1R_PLAN))[(3, "exact_composition")]
    options = dict(task["model_options"])
    names = {
        "exact": "runtime_spn_ct_k1t_position_histogram_true",
        "wrong": "runtime_spn_ct_k1t_position_histogram_wrong_sbox",
        "invariant": "runtime_spn_ct_k1t_position_histogram_invariant",
    }
    models = {
        condition: build_model(
            name,
            input_bits=512,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options=options,
        )
        for condition, name in names.items()
    }
    geometries = {
        condition: tuple(
            (name, tuple(value.shape)) for name, value in model.state_dict().items()
        )
        for condition, model in models.items()
    }
    features = torch.randint(
        0,
        2,
        (8, 512),
        generator=torch.Generator().manual_seed(11),
    ).float()
    exact = models["exact"]
    runtime = project_features(features, exact.runtime_structure)
    exact_histogram = deterministic_position_histogram(
        runtime,
        exact.runtime_structure,
    )
    invariant_histogram = deterministic_position_histogram(
        runtime,
        exact.runtime_structure,
        invariant_cells=True,
    )
    wrong = models["wrong"]
    wrong_histogram = deterministic_position_histogram(
        project_features(features, wrong.runtime_structure),
        wrong.runtime_structure,
    )

    assert len(set(geometries.values())) == 1
    assert model_metadata(exact)["trainable_parameter_count"] == 214316
    assert model_metadata(exact)["trainable_parameter_count"] <= 225000
    assert not torch.equal(exact_histogram, wrong_histogram)
    assert not torch.equal(exact_histogram, invariant_histogram)
    expected_invariant = exact_histogram.mean(dim=2, keepdim=True).expand_as(
        exact_histogram
    )
    assert torch.equal(invariant_histogram, expected_invariant)


def test_position_histogram_shared_state_controls_are_observable_and_trainable() -> None:
    task = task_map(read_tasks(K1R_PLAN))[(3, "exact_composition")]
    options = dict(task["model_options"])
    exact = build_model(
        "runtime_spn_ct_k1t_position_histogram_true",
        input_bits=512,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    )
    wrong = build_model(
        "runtime_spn_ct_k1t_position_histogram_wrong_sbox",
        input_bits=512,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    )
    invariant = build_model(
        "runtime_spn_ct_k1t_position_histogram_invariant",
        input_bits=512,
        hidden_bits=32,
        pair_bits=128,
        structure="SPN",
        model_options=options,
    )
    wrong.load_state_dict(exact.state_dict(), strict=True)
    invariant.load_state_dict(exact.state_dict(), strict=True)
    features = torch.randint(
        0,
        2,
        (8, 512),
        generator=torch.Generator().manual_seed(17),
    ).float()
    exact_logits = exact(features)
    wrong_logits = wrong(features)
    invariant_logits = invariant(features)

    assert torch.isfinite(exact_logits).all()
    assert not torch.equal(exact_logits, wrong_logits)
    assert not torch.equal(exact_logits, invariant_logits)
    loss = torch.nn.functional.mse_loss(
        torch.sigmoid(exact_logits).flatten(),
        torch.arange(8, dtype=torch.float32).remainder(2),
    )
    loss.backward()
    histogram_gradients = [
        parameter.grad
        for name, parameter in exact.named_parameters()
        if "histogram_" in name and parameter.grad is not None
    ]
    assert histogram_gradients
    assert all(torch.isfinite(gradient).all() for gradient in histogram_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in histogram_gradients) > 0.0
