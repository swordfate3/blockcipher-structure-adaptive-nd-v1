from __future__ import annotations

import pytest
import numpy as np
import torch

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    RuntimeE4EquivariantSpnDistinguisher,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    runtime_spn_structure,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    dialga128_runtime_structure,
    gift64_runtime_structure,
    rectangle80_runtime_structure,
    skinny64_runtime_structure,
    uknit64_runtime_structure,
)
from blockcipher_nd.data.differential import DifferentialDataset
from blockcipher_nd.training.runtime_spn_joint import (
    RuntimeSpnJointTask,
    train_runtime_spn_joint,
)
from blockcipher_nd.training.types import TrainingConfig


def _spec(
    mode: str,
    effect: str = "additive",
) -> RuntimeParameterizedSpnSpec:
    return RuntimeParameterizedSpnSpec(
        hidden_dim=16,
        pair_embedding_dim=24,
        processor_steps=2,
        dropout=0.0,
        sbox_context_mode="edge_gate",
        cell_input_mode="state_triplet",
        round_window_mode="recurrent_window",
        primitive_adapter_mode=mode,
        primitive_adapter_rank=4,
        primitive_adapter_scale=0.1,
        primitive_adapter_effect=effect,
    )


def _structures() -> dict[str, RuntimeSpnStructure]:
    return {
        "gift64": gift64_runtime_structure(2),
        "skinny64": skinny64_runtime_structure(2),
        "rectangle80": rectangle80_runtime_structure(2),
        "uknit64": uknit64_runtime_structure(2, round_start=3),
        "dialga128": dialga128_runtime_structure(2, round_start=2),
    }


def _mixed_fan_in_structure() -> RuntimeSpnStructure:
    linear = torch.eye(8, dtype=torch.uint8)
    linear[4:, 4:] = torch.tensor(
        (
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 1, 0),
            (1, 1, 1, 1),
        ),
        dtype=torch.uint8,
    )
    return runtime_spn_structure(
        cell_membership=(0, 0, 0, 0, 1, 1, 1, 1),
        bit_role=(3, 2, 1, 0, 3, 2, 1, 0),
        sbox_tables=PRESENT_SBOX,
        linear_matrices=linear.unsqueeze(0).repeat(2, 1, 1),
    )


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _dataset(block_bits: int, seed: int) -> DifferentialDataset:
    generator = np.random.default_rng(seed)
    return DifferentialDataset(
        features=generator.integers(
            0,
            2,
            size=(4, 2 * 2 * block_bits),
            dtype=np.uint8,
        ),
        labels=np.asarray((0, 1, 0, 1), dtype=np.uint8),
        metadata={"negative_mode": "encrypted_random_plaintexts"},
    )


def test_dense_and_routed_primitive_adapters_are_parameter_matched() -> None:
    dense = RuntimeE4EquivariantSpnDistinguisher(_spec("dense"))
    candidate = RuntimeE4EquivariantSpnDistinguisher(_spec("correct"))

    assert _parameter_count(dense) == _parameter_count(candidate)


def test_multiplicative_gate_is_parameter_matched_and_changes_only_effect() -> None:
    additive = RuntimeE4EquivariantSpnDistinguisher(_spec("correct", "additive")).eval()
    gated = RuntimeE4EquivariantSpnDistinguisher(
        _spec("correct", "multiplicative_gate")
    ).eval()
    gated.load_state_dict(additive.state_dict(), strict=True)
    structure = _mixed_fan_in_structure()
    pairs = torch.randint(0, 2, (3, 2, 2, 8), dtype=torch.float32)

    assert _parameter_count(additive) == _parameter_count(gated)
    assert additive.state_dict().keys() == gated.state_dict().keys()
    with torch.no_grad():
        additive_logits = additive(pairs, structure)
        gated_logits = gated(pairs, structure)
    assert not torch.equal(additive_logits, gated_logits)


def test_multiplicative_gate_preserves_routed_role_geometry() -> None:
    models = {
        role: RuntimeE4EquivariantSpnDistinguisher(_spec(role, "multiplicative_gate"))
        for role in ("correct", "uniform", "shuffled")
    }
    state = models["correct"].state_dict()

    models["uniform"].load_state_dict(state, strict=True)
    models["shuffled"].load_state_dict(state, strict=True)
    assert models["correct"].primitive_adapter_summary()["effect"] == (
        "multiplicative_gate"
    )


def test_routed_roles_have_identical_state_geometry_and_active_compute() -> None:
    candidate = RuntimeE4EquivariantSpnDistinguisher(_spec("correct"))
    uniform = RuntimeE4EquivariantSpnDistinguisher(_spec("uniform"))
    shuffled = RuntimeE4EquivariantSpnDistinguisher(_spec("shuffled"))

    uniform.load_state_dict(candidate.state_dict(), strict=True)
    shuffled.load_state_dict(candidate.state_dict(), strict=True)
    assert candidate.state_dict().keys() == uniform.state_dict().keys()
    assert candidate.state_dict().keys() == shuffled.state_dict().keys()
    assert candidate.primitive_adapter_summary()["active_adapter_evaluations"] == 2
    assert uniform.primitive_adapter_summary()["active_adapter_evaluations"] == 2
    assert shuffled.primitive_adapter_summary()["active_adapter_evaluations"] == 2


def test_one_shared_routed_state_handles_all_five_runtime_widths() -> None:
    model = RuntimeE4EquivariantSpnDistinguisher(_spec("correct")).eval()

    with torch.no_grad():
        for structure in _structures().values():
            pairs = torch.randint(
                0,
                2,
                (2, 2, 2, structure.block_bits),
                dtype=torch.float32,
            )
            assert model(pairs, structure).shape == (2, 1)


def test_correct_and_shuffled_assignments_change_only_routed_composition() -> None:
    structure = _mixed_fan_in_structure()
    pairs = torch.randint(0, 2, (3, 2, 2, 8), dtype=torch.float32)
    candidate = RuntimeE4EquivariantSpnDistinguisher(_spec("correct")).eval()
    shuffled = RuntimeE4EquivariantSpnDistinguisher(_spec("shuffled")).eval()
    uniform = RuntimeE4EquivariantSpnDistinguisher(_spec("uniform")).eval()
    shuffled.load_state_dict(candidate.state_dict(), strict=True)
    uniform.load_state_dict(candidate.state_dict(), strict=True)

    correct_weights = candidate.primitive_routing_weights(
        structure,
        round_index=0,
        mode="correct",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    shuffled_weights = candidate.primitive_routing_weights(
        structure,
        round_index=0,
        mode="shuffled",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    uniform_weights = candidate.primitive_routing_weights(
        structure,
        round_index=0,
        mode="uniform",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(shuffled_weights, correct_weights.flip(-1))
    torch.testing.assert_close(uniform_weights, torch.full_like(uniform_weights, 0.5))
    with torch.no_grad():
        correct_logits = candidate(pairs, structure)
        shuffled_logits = shuffled(pairs, structure)
        uniform_logits = uniform(pairs, structure)
    assert not torch.equal(correct_logits, shuffled_logits)
    assert not torch.equal(correct_logits, uniform_logits)


def test_mixed_fan_in_routes_traffic_and_gradients_to_both_adapters() -> None:
    structure = _mixed_fan_in_structure()
    model = RuntimeE4EquivariantSpnDistinguisher(_spec("correct"))
    pairs = torch.randint(0, 2, (4, 2, 2, 8), dtype=torch.float32)
    labels = torch.tensor((0.0, 1.0, 0.0, 1.0))

    logits = model(pairs, structure).squeeze(1)
    torch.nn.functional.mse_loss(torch.sigmoid(logits), labels).backward()

    assert model.last_primitive_adapter_traffic["fan_in_1"] > 0
    assert model.last_primitive_adapter_traffic["multi_source"] > 0
    assert model.primitive_adapters is not None
    for adapter in model.primitive_adapters.values():
        gradients = [
            parameter.grad
            for parameter in adapter.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_joint_optimizer_uses_one_equal_weight_shared_step_for_five_tasks() -> None:
    tasks = [
        RuntimeSpnJointTask(
            name=name,
            group="core" if name in {"gift64", "skinny64", "rectangle80"} else "stress",
            structure=structure,
            train_dataset=_dataset(structure.block_bits, index),
            validation_dataset=_dataset(structure.block_bits, 100 + index),
        )
        for index, (name, structure) in enumerate(_structures().items())
    ]
    model = RuntimeE4EquivariantSpnDistinguisher(_spec("correct"))

    result = train_runtime_spn_joint(
        model,
        tasks,
        TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            seed=0,
            device="cpu",
            optimizer="adam",
            weight_decay=1e-5,
            lr_scheduler="none",
            checkpoint_metric="val_macro_auc",
            restore_best_checkpoint=True,
            loss="mse",
        ),
    )

    assert result.metadata["shared_state_dict_count"] == 1
    assert result.metadata["task_specific_trainable_state"] is False
    assert result.metadata["optimizer_steps"] == 2
    assert set(result.metadata["task_weights"].values()) == {0.2}
    assert set(result.metadata["task_batch_counts"].values()) == {2}
    assert set(result.validation_metrics) == set(_structures())
    assert result.gradient_diagnostics["all_gradients_finite"] is True
    assert (
        result.gradient_diagnostics["adapter_gradient_mean_abs_sum"][
            "primitive_adapters.fan_in_1"
        ]
        > 0.0
    )
    assert (
        result.gradient_diagnostics["adapter_gradient_mean_abs_sum"][
            "primitive_adapters.multi_source"
        ]
        > 0.0
    )
    assert sum(result.router_traffic["gift64"].values()) > 0.0
    assert sum(result.router_traffic["skinny64"].values()) > 0.0


@pytest.mark.parametrize("cipher_name", tuple(_structures()))
def test_correct_primitive_routing_preserves_cell_relabeling(cipher_name: str) -> None:
    structure = _structures()[cipher_name]
    relabeled, bit_permutation = structure.relabel_cells(
        tuple(reversed(range(structure.cells)))
    )
    pairs = torch.randint(
        0,
        2,
        (1, 2, 2, structure.block_bits),
        dtype=torch.float32,
    )
    relabeled_pairs = torch.empty_like(pairs)
    relabeled_pairs[..., bit_permutation] = pairs
    model = RuntimeE4EquivariantSpnDistinguisher(_spec("correct")).eval()

    with torch.no_grad():
        original = model(pairs, structure)
        permuted = model(relabeled_pairs, relabeled)

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-6)
