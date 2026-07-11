# PRESENT InvP State-Matrix Conv2D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Completed steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build, validate, and adjudicate a strict-protocol PRESENT-80 r7 state-matrix Conv2D candidate against the existing InvP token-mixer anchor and matched shuffled-P/DeltaC-only controls.

**Architecture:** Keep the existing raw `16 x 128` ciphertext-pair dataset and construct a `[batch, pair, 4 bit planes, 16 cells]` view inside the model. Share one residual Conv2D implementation across true-InvP, shuffled-P, and DeltaC-only subclasses, retain mean/max pair aggregation, and use a dedicated result gate to enforce protocol identity, parameter-count equality, and the approved architecture/topology/representation margins.

**Tech Stack:** Python 3.10.16, PyTorch, NumPy-backed disk dataset cache, project CSV matrix runner, JSONL result validation, pytest, Ruff, Git.

**Design reference:** `docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md`

---

## File Map

Create or modify only these implementation-owned files:

```text
src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py
  expose one shared deterministic PRESENT mapping-index helper and make the
  existing InvP anchor consume it without changing its tensor values

src/blockcipher_nd/models/structure/spn/present_invp_state_matrix_conv2d.py
  implement the shared state-matrix view, residual Conv2D pair encoder,
  mean/max pair aggregation, and three mapping-specific model classes

src/blockcipher_nd/models/structure/spn/__init__.py
src/blockcipher_nd/models/structure/__init__.py
  export the three new classes

src/blockcipher_nd/registry/model_families/spn.py
  construct the three new model keys through the production model factory

src/blockcipher_nd/engine/modeling.py
  add exact total/trainable parameter counts to result metadata

src/blockcipher_nd/planning/invp_state_matrix_conv2d_gate.py
  validate result protocol and compute per-seed/joint decision branches

src/blockcipher_nd/cli/gate_invp_state_matrix_conv2d.py
scripts/gate-invp-state-matrix-conv2d
  expose the gate as a thin CLI and project script

tests/test_invp_state_matrix_conv2d.py
  cover mapping semantics, layout, forward/backward, parameter counts,
  exports, factory construction, and unchanged anchor behavior

tests/test_invp_state_matrix_conv2d_gate.py
  cover strict protocol validation, margins, decision branches, CLI output,
  and experiment matrix contracts

configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_smoke_seed0.csv
  four-row 64/class implementation-readiness matrix

configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_8192_seed0.csv
  four-row 8192/class seed0 diagnostic matrix

docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md
  record R0 and R1 run ids, artifacts, gates, metrics, decision, and next action
```

Do not modify the differential dataset generator, labels, negative definition,
validation split, metric code, checkpoint selection, DDT features, E1 graph
models, or the active AutoND paper-scale package.

## Frozen Names And Commands

```text
anchor model:
  present_nibble_invp_only_spn_only

new models:
  present_nibble_invp_state_matrix_conv2d_spn_only
  present_nibble_shuffled_p_state_matrix_conv2d_spn_only
  present_nibble_delta_state_matrix_conv2d_spn_only

R0 run id:
  i1_present_invp_state_matrix_conv2d_smoke_seed0

R1 run id:
  i1_present_invp_state_matrix_conv2d_8192_seed0
```

All implementation and test commands must use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ...
```

### Task 1: Share The Verified InvP Mapping

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py:832-918`
- Test: `tests/test_invp_state_matrix_conv2d.py`

- [x] **Step 1: Write failing mapping-helper tests**

Create `tests/test_invp_state_matrix_conv2d.py` with the first tests:

```python
from __future__ import annotations

import torch

from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    PresentNibbleInvPOnlySpnOnlyDistinguisher,
    present_inverse_p_indices,
)


def test_present_inverse_p_indices_are_deterministic_permutations() -> None:
    true_first = present_inverse_p_indices("true")
    true_second = present_inverse_p_indices("true")
    shuffled_first = present_inverse_p_indices("shuffled")
    shuffled_second = present_inverse_p_indices("shuffled")

    assert true_first.dtype == torch.long
    assert true_first.tolist() == true_second.tolist()
    assert shuffled_first.tolist() == shuffled_second.tolist()
    assert sorted(true_first.tolist()) == list(range(64))
    assert sorted(shuffled_first.tolist()) == list(range(64))
    assert true_first.tolist() != shuffled_first.tolist()


def test_present_inverse_p_indices_reject_unknown_alignment() -> None:
    try:
        present_inverse_p_indices("unknown")
    except ValueError as exc:
        assert str(exc) == "unsupported p_alignment: unknown"
    else:
        raise AssertionError("unknown alignment must fail")


def test_existing_invp_anchor_uses_shared_true_mapping() -> None:
    model = PresentNibbleInvPOnlySpnOnlyDistinguisher(
        input_bits=16 * 128,
        pair_bits=128,
        base_channels=32,
    )
    assert model.spn_encoder.inverse_p_indices.tolist() == present_inverse_p_indices("true").tolist()
```

- [x] **Step 2: Run the mapping tests and confirm the missing symbol failure**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d.py -q
```

Expected: collection fails because `present_inverse_p_indices` is not exported.

- [x] **Step 3: Add the shared helper and route the anchor through it**

Add immediately above `_present_inverse_p_index`:

```python
def present_inverse_p_indices(p_alignment: str) -> torch.Tensor:
    if p_alignment == "true":
        indices = [_present_inverse_p_index(index) for index in range(64)]
    elif p_alignment == "shuffled":
        generator = torch.Generator().manual_seed(20260627)
        indices = torch.randperm(64, generator=generator).tolist()
    else:
        raise ValueError(f"unsupported p_alignment: {p_alignment}")
    return torch.tensor(indices, dtype=torch.long)
```

Replace the mapping block inside `_PresentNibblePAlignedSpnEncoder.__init__`:

```python
self.register_buffer(
    "inverse_p_indices",
    present_inverse_p_indices(p_alignment),
    persistent=False,
)
```

Add `"present_inverse_p_indices"` to the module `__all__`. Do not change the
other historical encoders in this task; the approved candidate and current
InvP anchor are the only consumers requiring shared identity.

- [x] **Step 4: Run the focused tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d.py -q
```

Expected: `3 passed`.

- [x] **Step 5: Commit the mapping helper**

```bash
git add src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py tests/test_invp_state_matrix_conv2d.py
git commit -m "refactor: share PRESENT inverse mapping indices"
```

### Task 2: Implement The State-Matrix Conv2D Models

**Files:**
- Create: `src/blockcipher_nd/models/structure/spn/present_invp_state_matrix_conv2d.py`
- Modify: `tests/test_invp_state_matrix_conv2d.py`

- [x] **Step 1: Add failing layout and model tests**

Append:

```python
from blockcipher_nd.models.structure.spn.present_invp_state_matrix_conv2d import (
    PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher,
)


def _raw_pair_features(batch: int = 2, pairs: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260711)
    return torch.randint(0, 2, (batch, pairs * 128), generator=generator).float()


def _candidate(model_type):
    return model_type(
        input_bits=16 * 128,
        pair_bits=128,
        base_channels=32,
        conv_depth=3,
        kernel_size=3,
        activation="relu",
        norm="batchnorm2d",
        dropout=0.0,
    )


def test_true_state_matrix_matches_existing_invp_anchor_view() -> None:
    features = _raw_pair_features()
    anchor = PresentNibbleInvPOnlySpnOnlyDistinguisher(
        input_bits=16 * 128,
        pair_bits=128,
        base_channels=32,
    )
    candidate = _candidate(PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher)

    expected = anchor.spn_encoder.present_nibble_paligned_view(features).reshape(2, 16, 4, 16)
    observed = candidate.state_matrix_view(features)

    torch.testing.assert_close(observed, expected)


def test_delta_state_matrix_is_unpermuted_ciphertext_difference() -> None:
    features = _raw_pair_features(batch=1)
    candidate = _candidate(PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher)
    raw = features.reshape(1, 16, 2, 64)
    difference = (raw[:, :, 0] - raw[:, :, 1]).abs()
    expected = difference.reshape(1, 16, 16, 4).permute(0, 1, 3, 2)

    torch.testing.assert_close(candidate.state_matrix_view(features), expected)


def test_shuffled_state_matrix_is_deterministic_and_not_true_mapping() -> None:
    features = _raw_pair_features()
    true_model = _candidate(PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher)
    shuffled_a = _candidate(PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher)
    shuffled_b = _candidate(PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher)

    torch.testing.assert_close(shuffled_a.state_matrix_view(features), shuffled_b.state_matrix_view(features))
    assert not torch.equal(shuffled_a.state_matrix_view(features), true_model.state_matrix_view(features))


def test_state_matrix_models_have_equal_capacity_and_finite_gradients() -> None:
    models = [
        _candidate(PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher),
        _candidate(PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher),
        _candidate(PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher),
    ]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert len(set(counts)) == 1

    features = _raw_pair_features()
    for model in models:
        logits = model(features)
        assert logits.shape == (2, 1)
        logits.square().mean().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        assert gradients
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_state_matrix_model_validates_shape_and_options() -> None:
    for model_type, kwargs, message in [
        (PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher, {"pair_bits": 64}, "expects raw 128-bit ciphertext pairs"),
        (PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher, {"conv_depth": 0}, "conv_depth must be >= 1"),
        (PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher, {"kernel_size": 2}, "kernel_size must be a positive odd integer"),
        (PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher, {"mapping_mode": "bad"}, "unsupported mapping_mode: bad"),
    ]:
        options = {
            "input_bits": 16 * 128,
            "pair_bits": 128,
            "base_channels": 32,
            **kwargs,
        }
        try:
            model_type(**options)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected failure for {kwargs}")
```

- [x] **Step 2: Run tests and confirm the module is missing**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d.py -q
```

Expected: collection fails with `ModuleNotFoundError` for
`present_invp_state_matrix_conv2d`.

- [x] **Step 3: Implement the complete candidate module**

Create the module with this implementation:

```python
from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.common.components import build_activation, build_norm
from blockcipher_nd.models.structure.spn.present_inception_blocks import conv2d_norm
from blockcipher_nd.models.structure.spn.present_nibble_paligned_mcnd import (
    present_inverse_p_indices,
)


class PresentStateMatrixResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        activation: str,
        norm: str,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = conv2d_norm(norm, channels)
        self.activation1 = build_activation(activation)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = conv2d_norm(norm, channels)
        self.activation2 = build_activation(activation)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.activation1(self.norm1(self.conv1(features)))
        hidden = self.norm2(self.conv2(self.dropout(hidden)))
        return self.activation2(features + hidden)


class PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        conv_depth: int = 3,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str = "batchnorm2d",
        dropout: float = 0.0,
        mapping_mode: str = "true",
    ) -> None:
        super().__init__()
        if pair_bits != 128:
            raise ValueError("PRESENT state-matrix Conv2D expects raw 128-bit ciphertext pairs")
        if input_bits % pair_bits != 0:
            raise ValueError("input_bits must be a multiple of pair_bits")
        if conv_depth < 1:
            raise ValueError("conv_depth must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if mapping_mode not in {"true", "shuffled", "delta"}:
            raise ValueError(f"unsupported mapping_mode: {mapping_mode}")
        self.input_bits = input_bits
        self.pair_bits = pair_bits
        self.pairs_per_sample = input_bits // pair_bits
        self.base_channels = base_channels
        self.conv_depth = conv_depth
        self.kernel_size = kernel_size
        self.mapping_mode = mapping_mode
        indices = (
            torch.arange(64, dtype=torch.long)
            if mapping_mode == "delta"
            else present_inverse_p_indices(mapping_mode)
        )
        self.register_buffer("mapping_indices", indices, persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=1),
            conv2d_norm(norm, base_channels),
            build_activation(activation),
        )
        self.blocks = nn.ModuleList(
            [
                PresentStateMatrixResidualBlock(
                    base_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for _ in range(conv_depth)
            ]
        )
        self.embedding_bits = base_channels * 4
        self.pair_projection = nn.Sequential(
            nn.Linear(base_channels * 2, self.embedding_bits),
            build_activation(activation),
        )
        self.classifier = nn.Sequential(
            build_norm("layernorm", self.embedding_bits * 2),
            nn.Linear(self.embedding_bits * 2, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )

    def set_cipher_structure(self, structure: str) -> None:
        return None

    def set_structure_features(self, features: torch.Tensor) -> None:
        return None

    def state_matrix_view(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_bits:
            raise ValueError(f"expected {self.input_bits} input bits, got {tuple(features.shape)}")
        raw_pairs = features.reshape(features.shape[0], self.pairs_per_sample, 2, 64)
        difference = (raw_pairs[:, :, 0] - raw_pairs[:, :, 1]).abs()
        mapped = difference.index_select(dim=2, index=self.mapping_indices)
        return mapped.reshape(features.shape[0], self.pairs_per_sample, 16, 4).permute(0, 1, 3, 2)

    def encode_pairs(self, state_matrices: torch.Tensor) -> torch.Tensor:
        batch, pairs, bit_planes, cells = state_matrices.shape
        hidden = self.stem(state_matrices.reshape(batch * pairs, 1, bit_planes, cells))
        for block in self.blocks:
            hidden = block(hidden)
        mean_embedding = hidden.mean(dim=(2, 3))
        max_embedding = hidden.amax(dim=(2, 3))
        projected = self.pair_projection(torch.cat([mean_embedding, max_embedding], dim=1))
        return projected.reshape(batch, pairs, self.embedding_bits)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pair_embeddings = self.encode_pairs(self.state_matrix_view(features.float()))
        pair_mean = pair_embeddings.mean(dim=1)
        pair_max = pair_embeddings.max(dim=1).values
        return self.classifier(torch.cat([pair_mean, pair_max], dim=1))


class PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher(
    PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["mapping_mode"] = "true"
        super().__init__(*args, **kwargs)


class PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher(
    PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["mapping_mode"] = "shuffled"
        super().__init__(*args, **kwargs)


class PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher(
    PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher
):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["mapping_mode"] = "delta"
        super().__init__(*args, **kwargs)


__all__ = [
    "PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher",
    "PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher",
    "PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher",
    "PresentNibbleStateMatrixConv2DSpnOnlyDistinguisher",
    "PresentStateMatrixResidualBlock",
]
```

- [x] **Step 4: Run model tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit the model**

```bash
git add src/blockcipher_nd/models/structure/spn/present_invp_state_matrix_conv2d.py tests/test_invp_state_matrix_conv2d.py
git commit -m "feat: add PRESENT InvP state-matrix Conv2D"
```

### Task 3: Export, Register, And Record Capacity

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/__init__.py`
- Modify: `src/blockcipher_nd/models/structure/__init__.py`
- Modify: `src/blockcipher_nd/registry/model_families/spn.py`
- Modify: `src/blockcipher_nd/engine/modeling.py:19-43`
- Modify: `tests/test_invp_state_matrix_conv2d.py`

- [x] **Step 1: Add failing factory and metadata tests**

Append:

```python
from torch import nn

from blockcipher_nd.engine.modeling import model_metadata
from blockcipher_nd.registry.model_factory import build_model


MODEL_KEYS = (
    "present_nibble_invp_state_matrix_conv2d_spn_only",
    "present_nibble_shuffled_p_state_matrix_conv2d_spn_only",
    "present_nibble_delta_state_matrix_conv2d_spn_only",
)


def test_production_factory_builds_all_state_matrix_controls() -> None:
    models = [
        build_model(
            key,
            input_bits=16 * 128,
            hidden_bits=32,
            pair_bits=128,
            structure="SPN",
            model_options={
                "conv_depth": 3,
                "kernel_size": 3,
                "activation": "relu",
                "norm": "batchnorm2d",
                "dropout": 0.0,
            },
        )
        for key in MODEL_KEYS
    ]
    assert [model.mapping_mode for model in models] == ["true", "shuffled", "delta"]
    assert len({sum(parameter.numel() for parameter in model.parameters()) for model in models}) == 1


def test_model_metadata_records_exact_parameter_counts() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 1))
    model[1].weight.requires_grad_(False)
    metadata = model_metadata(model)
    assert metadata["parameter_count"] == 19
    assert metadata["trainable_parameter_count"] == 16
```

- [x] **Step 2: Run tests and confirm unsupported model failures**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d.py -q
```

Expected: factory construction fails with `unsupported model`, and metadata
assertions fail because count fields do not exist.

- [x] **Step 3: Export and register the model classes**

In both `structure/spn/__init__.py` and `structure/__init__.py`, import and add
to `__all__`:

```python
PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher
PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher
PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher
```

In `registry/model_families/spn.py`, import the same classes and insert these
branches immediately after the existing InvP-only branch:

```python
state_matrix_options = {
    "input_bits": input_bits,
    "pair_bits": pair_bits or 128,
    "base_channels": hidden_bits,
    "conv_depth": int_option(options, "conv_depth", 3) or 3,
    "kernel_size": int_option(options, "kernel_size", 3) or 3,
    "activation": str(options.get("activation", "relu")),
    "norm": str(options.get("norm", "batchnorm2d")),
    "dropout": float(options.get("dropout", 0.0)),
}
if name == "present_nibble_invp_state_matrix_conv2d_spn_only":
    return PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher(**state_matrix_options)
if name == "present_nibble_shuffled_p_state_matrix_conv2d_spn_only":
    return PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher(**state_matrix_options)
if name == "present_nibble_delta_state_matrix_conv2d_spn_only":
    return PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher(**state_matrix_options)
```

Do not construct `state_matrix_options` for every model name. To avoid unused
work and preserve the current sequential factory, wrap the block in:

```python
if name in {
    "present_nibble_invp_state_matrix_conv2d_spn_only",
    "present_nibble_shuffled_p_state_matrix_conv2d_spn_only",
    "present_nibble_delta_state_matrix_conv2d_spn_only",
}:
    ...
```

- [x] **Step 4: Add exact counts to result metadata**

Initialize `metadata` in `model_metadata` as:

```python
metadata: dict[str, Any] = {
    "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    "trainable_parameter_count": int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    ),
}
```

This is additive metadata only; do not alter metrics or training behavior.

- [x] **Step 5: Run focused and existing model-factory tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d.py tests/test_autond_dbitnet2023.py -q
```

Expected: pass.

- [x] **Step 6: Commit integration and metadata**

```bash
git add src/blockcipher_nd/models/structure/spn/__init__.py src/blockcipher_nd/models/structure/__init__.py src/blockcipher_nd/registry/model_families/spn.py src/blockcipher_nd/engine/modeling.py tests/test_invp_state_matrix_conv2d.py
git commit -m "feat: register InvP state-matrix Conv2D models"
```

### Task 4: Implement The Strict Attribution Gate

**Files:**
- Create: `src/blockcipher_nd/planning/invp_state_matrix_conv2d_gate.py`
- Create: `tests/test_invp_state_matrix_conv2d_gate.py`

- [x] **Step 1: Write failing gate tests with exact result fixtures**

Create a fixture builder that emits the real result shape:

```python
from __future__ import annotations

import json
from pathlib import Path

from blockcipher_nd.planning.invp_state_matrix_conv2d_gate import (
    gate_invp_state_matrix_conv2d,
)


ANCHOR = "present_nibble_invp_only_spn_only"
TRUE = "present_nibble_invp_state_matrix_conv2d_spn_only"
SHUFFLED = "present_nibble_shuffled_p_state_matrix_conv2d_spn_only"
DELTA = "present_nibble_delta_state_matrix_conv2d_spn_only"


def _row(model: str, auc: float, *, seed: int = 0, parameter_count: int = 1000) -> dict[str, object]:
    options = (
        {"spn_mixer_depth": 2, "activation": "relu", "norm": "layernorm"}
        if model == ANCHOR
        else {
            "conv_depth": 3,
            "kernel_size": 3,
            "activation": "relu",
            "norm": "batchnorm2d",
            "dropout": 0.0,
        }
    )
    return {
        "selected_model": model,
        "rounds": 7,
        "seed": seed,
        "samples_per_class": 8192,
        "pairs_per_sample": 16,
        "feature_encoding": "ciphertext_pair_bits",
        "negative_mode": "encrypted_random_plaintexts",
        "train_key": 0,
        "validation_key": int("11" * 10, 16),
        "key_rotation_interval": 0,
        "sample_structure": "zhang_wang_case2_official_mcnd",
        "difference_profile": "present_zhang_wang2022_mcnd",
        "difference_member": 0,
        "parameter_count": 900 if model == ANCHOR else parameter_count,
        "trainable_parameter_count": 900 if model == ANCHOR else parameter_count,
        "metrics": {
            "auc": auc,
            "accuracy": 0.6,
            "calibrated_accuracy": 0.61,
            "loss": 0.68,
        },
        "training": {
            "key_schedule": "per_pair_random",
            "input_bits": 2048,
            "pair_bits": 128,
            "train_rows": 16384,
            "validation_rows": 8192,
            "epochs": 10,
            "checkpoint_metric": "val_auc",
            "selected_checkpoint": "best",
            "restore_best_checkpoint": True,
            "loss": "mse",
            "optimizer": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "lr_scheduler": "official_cyclic",
            "max_learning_rate": 0.002,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.0001,
            "dataset_cache_root": "outputs/local_cache/shared",
            "model_options": options,
        },
        "validation": {
            "key_schedule": "per_pair_random",
            "samples_per_class": 4096,
            "negative_mode": "encrypted_random_plaintexts",
            "sample_structure": "zhang_wang_case2_official_mcnd",
        },
    }


def _write(path: Path, aucs: dict[str, float], *, seed: int = 0) -> None:
    rows = [_row(model, auc, seed=seed) for model, auc in aucs.items()]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_gate_promotes_clear_seed0_result(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.60, TRUE: 0.61, SHUFFLED: 0.605, DELTA: 0.604})
    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))
    assert report["status"] == "pass"
    assert report["decision"] == "promote_seed1"
    assert report["seeds"]["0"]["architecture_margin"] == 0.010000000000000009
    assert report["seeds"]["0"]["topology_margin"] == 0.0050000000000000044
    assert report["seeds"]["0"]["representation_margin"] == 0.006000000000000005


def test_gate_stops_when_candidate_loses_to_anchor(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.61, TRUE: 0.60, SHUFFLED: 0.59, DELTA: 0.58})
    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))
    assert report["decision"] == "stop_conv2d_route"
    assert report["next_action"] == "keep_token_mixer_anchor_and_do_not_scale_conv2d"


def test_gate_rejects_generic_locality_without_true_topology(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.59, TRUE: 0.61, SHUFFLED: 0.6105, DELTA: 0.60})
    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))
    assert report["decision"] == "stop_generic_locality"


def test_gate_rejects_missing_representation_attribution(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.59, TRUE: 0.61, SHUFFLED: 0.60, DELTA: 0.611})
    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))
    assert report["decision"] == "stop_invp_attribution"


def test_gate_requires_both_seeds_for_medium_promotion(tmp_path: Path) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    _write(seed0, {ANCHOR: 0.60, TRUE: 0.61, SHUFFLED: 0.606, DELTA: 0.605}, seed=0)
    _write(seed1, {ANCHOR: 0.60, TRUE: 0.607, SHUFFLED: 0.604, DELTA: 0.603}, seed=1)
    report = gate_invp_state_matrix_conv2d([seed0, seed1], expected_seeds=(0, 1))
    assert report["decision"] == "promote_medium_65536"
    assert report["next_action"] == "run_65536_per_class_two_seed_medium_confirmation"


def test_gate_rejects_parameter_or_protocol_mismatch(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write(results, {ANCHOR: 0.60, TRUE: 0.61, SHUFFLED: 0.605, DELTA: 0.604})
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    rows[2]["parameter_count"] = 1001
    rows[3]["negative_mode"] = "random_ciphertext"
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = gate_invp_state_matrix_conv2d([results], expected_seeds=(0,))
    assert report["status"] == "fail"
    assert report["decision"] == "invalid_protocol"
    assert any("conv2d_parameter_count_mismatch" in error for error in report["errors"])
    assert any("negative_mode" in error for error in report["errors"])
```

- [x] **Step 2: Run tests and confirm the gate module is missing**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d_gate.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [x] **Step 3: Implement the gate module**

Implement these public constants and function:

```python
MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_invp_state_matrix_conv2d_spn_only",
    "shuffled_p": "present_nibble_shuffled_p_state_matrix_conv2d_spn_only",
    "delta_only": "present_nibble_delta_state_matrix_conv2d_spn_only",
}


def gate_invp_state_matrix_conv2d(
    results_paths: list[Path],
    *,
    progress_paths: list[Path],
    expected_seeds: tuple[int, ...] = (0,),
    samples_per_class: int = 8192,
    epochs: int = 10,
    seed0_topology_margin: float = 0.003,
    seed0_representation_margin: float = 0.003,
    joint_architecture_margin: float = 0.001,
    joint_control_margin: float = 0.002,
) -> dict[str, Any]:
    rows = _load_rows(results_paths)
    errors = _protocol_errors(
        rows,
        expected_seeds=expected_seeds,
        samples_per_class=samples_per_class,
        epochs=epochs,
    )
    if errors:
        return {
            "status": "fail",
            "decision": "invalid_protocol",
            "errors": errors,
            "next_action": "repair_protocol_and_rerun_same_matrix",
            "claim_scope": "invalid strict-protocol architecture evidence",
        }
    by_seed = _rows_by_seed_and_role(rows)
    seed_reports = {
        str(seed): _seed_report(by_seed[seed])
        for seed in expected_seeds
    }
    decision, next_action = _decision(
        seed_reports,
        expected_seeds=expected_seeds,
        seed0_topology_margin=seed0_topology_margin,
        seed0_representation_margin=seed0_representation_margin,
        joint_architecture_margin=joint_architecture_margin,
        joint_control_margin=joint_control_margin,
    )
    conv_counts = {
        role: int(by_seed[expected_seeds[0]][role]["parameter_count"])
        for role in ("candidate", "shuffled_p", "delta_only")
    }
    anchor_count = int(by_seed[expected_seeds[0]]["anchor"]["parameter_count"])
    return {
        "status": "pass",
        "decision": decision,
        "errors": [],
        "expected_seeds": list(expected_seeds),
        "samples_per_class": samples_per_class,
        "models": MODEL_ROLES,
        "seeds": seed_reports,
        "parameter_counts": {
            "anchor": anchor_count,
            **conv_counts,
            "candidate_to_anchor_ratio": conv_counts["candidate"] / anchor_count,
        },
        "next_action": next_action,
        "claim_scope": (
            f"{samples_per_class}/class strict PRESENT r7 architecture-attribution diagnostic; "
            "not formal, paper-scale, or breakthrough evidence"
        ),
    }
```

Execution note: the initial TDD snippets above show the pre-hardening result
fixture calls. The completed public API requires `progress_paths` and validates
terminal cache/create/reuse plus `run_done` evidence; focused tests supply a
synthetic progress fixture for every otherwise-valid result fixture.

Implement `_protocol_errors` to require, for every expected seed, exactly one
row for every `MODEL_ROLES` value and these exact fields:

```python
{
    "rounds": 7,
    "seed": expected_seed,
    "samples_per_class": samples_per_class,
    "pairs_per_sample": 16,
    "feature_encoding": "ciphertext_pair_bits",
    "negative_mode": "encrypted_random_plaintexts",
    "train_key": 0,
    "validation_key": int("11" * 10, 16),
    "key_rotation_interval": 0,
    "sample_structure": "zhang_wang_case2_official_mcnd",
    "difference_profile": "present_zhang_wang2022_mcnd",
    "difference_member": 0,
}
```

The configured `train_key`, `validation_key`, and `key_rotation_interval`
fields above are deterministic plan/cache placeholder identity only for
`zhang_wang_case2_official_mcnd`; they are not the effective encryption key
schedule. Require generated result metadata to report
`training.key_schedule=per_pair_random` and
`validation.key_schedule=per_pair_random`, proving that both datasets use an
independent random PRESENT key for every basic pair. A mismatch in either field
must produce `invalid_protocol` with a field-specific error.

Require training `input_bits=2048`, `pair_bits=128`,
`train_rows=2*samples_per_class`,
`validation_rows=samples_per_class`, `epochs=epochs`, `loss=mse`,
`optimizer=adam`, `learning_rate=0.0001`, `weight_decay=0.00001`,
`lr_scheduler=official_cyclic`, `max_learning_rate=0.002`,
`checkpoint_metric=val_auc`, `restore_best_checkpoint=true`,
`selected_checkpoint=best`, `early_stopping_patience=8`, and
`early_stopping_min_delta=0.0001`. Require one identical non-empty
`dataset_cache_root` per seed and `validation.samples_per_class` equal to half
the training class count.

Require all AUC/accuracy/calibrated-accuracy/loss metrics to be finite. Require
positive integer `parameter_count` and `trainable_parameter_count`; require
the three Conv2D counts to match exactly. Do not require the anchor count to
match the candidate.

Implement `_seed_report` as:

```python
def _seed_report(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aucs = {
        role: float(row["metrics"]["auc"])
        for role, row in rows.items()
    }
    return {
        "aucs": aucs,
        "architecture_margin": aucs["candidate"] - aucs["anchor"],
        "topology_margin": aucs["candidate"] - aucs["shuffled_p"],
        "representation_margin": aucs["candidate"] - aucs["delta_only"],
        "candidate_above_all": aucs["candidate"] > max(
            aucs["anchor"], aucs["shuffled_p"], aucs["delta_only"]
        ),
    }
```

Decision order must be deterministic:

```text
candidate <= anchor           -> stop_conv2d_route
candidate <= shuffled-P       -> stop_generic_locality
candidate <= DeltaC-only      -> stop_invp_attribution
one seed only and margins pass-> promote_seed1
one seed only but submargin   -> weak_or_fragile_no_scale
two seeds and joint gates pass-> promote_medium_65536
two seeds but any gate fails  -> unstable_no_remote_scale
```

- [x] **Step 4: Run gate tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d_gate.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit the gate**

```bash
git add src/blockcipher_nd/planning/invp_state_matrix_conv2d_gate.py tests/test_invp_state_matrix_conv2d_gate.py
git commit -m "feat: gate InvP state-matrix Conv2D evidence"
```

### Task 5: Add CLI, Script, And Frozen Matrices

**Files:**
- Create: `src/blockcipher_nd/cli/gate_invp_state_matrix_conv2d.py`
- Create: `scripts/gate-invp-state-matrix-conv2d`
- Create: `configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_smoke_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_8192_seed0.csv`
- Modify: `tests/test_invp_state_matrix_conv2d_gate.py`

- [x] **Step 1: Add failing CLI and matrix-contract tests**

Append tests that call the CLI `main` and parse both plans through the real
matrix-runner path, `parse_args` plus `build_tasks`:

```python
EXPECTED_MODELS = [ANCHOR, TRUE, SHUFFLED, DELTA]


def test_state_matrix_gate_cli_writes_json(tmp_path: Path) -> None:
    from blockcipher_nd.cli.gate_invp_state_matrix_conv2d import main

    results = tmp_path / "results.jsonl"
    progress = tmp_path / "progress.jsonl"
    output = tmp_path / "gate.json"
    _write(results, {ANCHOR: 0.60, TRUE: 0.61, SHUFFLED: 0.605, DELTA: 0.604})
    status = main([
        "--results", str(results),
        "--progress", str(progress),
        "--output", str(output),
    ])
    assert status == 0
    assert json.loads(output.read_text())["decision"] == "promote_seed1"


def test_frozen_state_matrix_plans_have_exact_protocol() -> None:
    from blockcipher_nd.engine.matrix_runner import parse_args
    from blockcipher_nd.planning.matrix import build_tasks

    root = Path(__file__).resolve().parents[1]
    paths = {
        "smoke": root / "configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_smoke_seed0.csv",
        "r1": root / "configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_8192_seed0.csv",
    }
    expected_samples = {"smoke": 64, "r1": 8192}
    for phase, path in paths.items():
        rows = build_tasks(parse_args(["--plan", str(path)]))
        assert [row["model_key"] for row in rows] == EXPECTED_MODELS
        assert {row["samples_per_class"] for row in rows} == {expected_samples[phase]}
        assert {row["seed"] for row in rows} == {0}
        assert {row["rounds"] for row in rows} == {7}
        assert {row["pairs_per_sample"] for row in rows} == {16}
        assert {row["negative_mode"] for row in rows} == {"encrypted_random_plaintexts"}
        assert {row["sample_structure"] for row in rows} == {"zhang_wang_case2_official_mcnd"}
        assert {row["feature_encoding"] for row in rows} == {"ciphertext_pair_bits"}
        assert all(row["checkpoint_metric"] == "val_auc" for row in rows)
        assert all(row["restore_best_checkpoint"] is True for row in rows)
```

- [x] **Step 2: Run tests and confirm missing CLI/plans**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d_gate.py -q
```

Expected: failures for the missing CLI and CSV files.

- [x] **Step 3: Implement the CLI and script**

The CLI must accept repeated `--results`, repeated required `--progress`,
comma-separated `--expected-seeds`, `--samples-per-class`, `--epochs`,
`--readiness-only`, and `--output`, call the gate, write sorted indented JSON,
and return `0` for protocol-valid reports even when the research decision is
stop; return `1` only for `status=fail`.

Use this parser contract:

```python
parser.add_argument("--results", action="append", required=True, type=Path)
parser.add_argument("--progress", action="append", required=True, type=Path)
parser.add_argument("--expected-seeds", default="0")
parser.add_argument("--samples-per-class", type=int, default=8192)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--readiness-only", action="store_true")
parser.add_argument("--output", required=True, type=Path)
```

Forward `readiness_only` to the public gate. It is valid only for the frozen R0
identity `expected_seeds=(0,)`, `samples_per_class=64`, and `epochs=1`; any
other identity must fail closed as `invalid_protocol`. A valid readiness report
must return `decision=implementation_ready`,
`next_action=run_frozen_r1_seed0_local_diagnostic`,
`claim_scope=implementation readiness only; metrics not interpreted`, and
`research_decision_applied=false`. Normal R1/R2 calls omit the flag and retain
the metric-derived research decision branches.

Parse seeds with:

```python
expected_seeds = tuple(int(value) for value in args.expected_seeds.split(","))
```

Create the script using the same thin wrapper pattern as
`scripts/gate-autond-typed-invp`, then run:

```bash
chmod +x scripts/gate-invp-state-matrix-conv2d
```

- [x] **Step 4: Create both exact four-row CSVs**

Use the standard Innovation 1 header. Every row freezes the protocol in the
design. Anchor options are:

```json
{"spn_mixer_depth":2,"activation":"relu","norm":"layernorm"}
```

Every Conv2D row uses exactly:

```json
{"conv_depth":3,"kernel_size":3,"activation":"relu","norm":"batchnorm2d","dropout":0.0}
```

Use `samples_per_class=64` in the smoke CSV and `8192` in R1. Evidence strings
must explicitly say `SMOKE readiness only` or `8192/class local diagnostic;
not formal, paper-scale, or breakthrough evidence`.

The CSV `train_key=0`, `validation_key=0x11111111111111111111`, and
`key_rotation_interval=0` values are inert deterministic placeholders under
`zhang_wang_case2_official_mcnd`. Do not infer effective fixed-key encryption
from the matrix contract. Effective key behavior is validated only from the
generated train and validation dataset metadata, both of which must serialize
`key_schedule=per_pair_random`.

- [x] **Step 5: Run focused tests and CSV plan alignment parser**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d.py tests/test_invp_state_matrix_conv2d_gate.py -q
```

Expected: pass.

- [x] **Step 6: Commit CLI and plans**

```bash
git add src/blockcipher_nd/cli/gate_invp_state_matrix_conv2d.py scripts/gate-invp-state-matrix-conv2d configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_smoke_seed0.csv configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_8192_seed0.csv tests/test_invp_state_matrix_conv2d_gate.py
git commit -m "experiment: plan InvP state-matrix Conv2D gate"
```

### Task 6: Run R0 End-To-End Readiness

**Files:**
- Generated ignored artifacts: `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/`
- Generated ignored cache: `outputs/local_cache/i1_present_invp_state_matrix_conv2d_smoke_seed0/`
- Modify after success: `docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md`

- [x] **Step 1: Run the four-row CPU readiness matrix**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_smoke_seed0.csv \
  --epochs 1 \
  --batch-size 32 \
  --hidden-bits 32 \
  --device cpu \
  --dataset-cache-root outputs/local_cache/i1_present_invp_state_matrix_conv2d_smoke_seed0 \
  --dataset-cache-chunk-size 64 \
  --dataset-cache-workers 1 \
  --progress-output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/progress.jsonl \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/results.jsonl
```

Expected: four completed rows and `run_done`; metric values are ignored.

- [x] **Step 2: Validate result-plan alignment**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_smoke_seed0.csv \
  --results outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/results.jsonl \
  --expected-rows 4 \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/validation.json
```

Expected: `status=pass`, `result_rows=4`, `errors=[]`.

- [x] **Step 3: Render history and curves**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/curves.svg \
  --history-csv outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/history.csv \
  --title i1_present_invp_state_matrix_conv2d_smoke_seed0
```

Expected: non-empty SVG and CSV.

- [x] **Step 4: Audit cache reuse and parameter metadata**

Use a structured Python check to require 8 cache events (train/validation for
four rows), at least 6 reuse events after the first row, three equal Conv2D
parameter counts, positive gradients already covered by tests, and four finite
metric rows. If the cache event count differs because the runner emits an
additional readiness event, compare exact cache paths and require one unique
train path plus one unique validation path.

Also require every result row to serialize
`training.key_schedule=per_pair_random` and
`validation.key_schedule=per_pair_random`, and require both cache metadata
files to report the same effective schedule. The configured top-level key
fields remain placeholder identity and must not be described as fixed
encryption keys.

Generate the R0 machine-readable readiness artifact without applying a
metric-derived research verdict:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-invp-state-matrix-conv2d \
  --results outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/results.jsonl \
  --progress outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/progress.jsonl \
  --expected-seeds 0 \
  --samples-per-class 64 \
  --epochs 1 \
  --readiness-only \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/readiness_gate.json
```

Require `status=pass`, `decision=implementation_ready`, `errors=[]`,
`research_decision_applied=false`, the exact readiness claim scope, and the
frozen R1 seed0 local next action. Do not use normal decision mode for R0.

- [x] **Step 5: Record R0 readiness in the design document**

Add a dated R0 section containing run id, command, four model keys, validation
status, cache identity, parameter counts/ratio, artifacts, and
`claim_scope=implementation readiness only; metrics not interpreted`.

- [x] **Step 6: Commit and push R0 evidence documentation**

```bash
git add docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md
git commit -m "experiment: validate InvP state-matrix Conv2D readiness"
git push origin experiment/invp-state-matrix-conv2d
```

### Task 7: Run And Adjudicate R1 Seed0

**Files:**
- Generated ignored artifacts: `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/`
- Generated ignored cache: `outputs/local_cache/i1_present_invp_state_matrix_conv2d_8192_seed0/`
- Modify after completion: `docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md`

- [x] **Step 1: Confirm the active AutoND watcher remains independent**

Read only the local monitor health/artifacts. Do not SSH-poll. R1 is CPU-local
and must not modify or stop the paper-scale task.

- [x] **Step 2: Run the exact four-row R1 matrix**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_8192_seed0.csv \
  --epochs 10 \
  --batch-size 256 \
  --hidden-bits 32 \
  --device cpu \
  --dataset-cache-root outputs/local_cache/i1_present_invp_state_matrix_conv2d_8192_seed0 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --progress-output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/progress.jsonl \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/results.jsonl
```

Expected: four completed rows. Do not interpret partial rows.

- [x] **Step 3: Validate and plot R1**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_8192_seed0.csv \
  --results outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/results.jsonl \
  --expected-rows 4 \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/validation.json
```

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/curves.svg \
  --history-csv outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/history.csv \
  --title i1_present_invp_state_matrix_conv2d_8192_seed0
```

Expected: validation passes and artifacts are non-empty.

- [x] **Step 4: Run the strict seed0 gate**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-invp-state-matrix-conv2d \
  --results outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/results.jsonl \
  --progress outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/progress.jsonl \
  --expected-seeds 0 \
  --samples-per-class 8192 \
  --epochs 10 \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/state_matrix_conv2d_gate.json
```

Expected: `status=pass` for a protocol-valid run. Research decision may be
`promote_seed1`, `weak_or_fragile_no_scale`, or one of the stop branches.

- [x] **Step 5: Update the design document with the complete R1 result**

Record all four AUC/accuracy/loss values, three candidate margins, parameter
counts, cache identity, validation status, gate decision, claim scope, artifact
paths, and evidence-backed next action.

Decision handling is mandatory:

```text
promote_seed1:
  create and run an identical seed1 CSV in the next planned task; no remote scale yet

weak_or_fragile_no_scale:
  inspect histories once, document instability, do not launch remote

stop_conv2d_route / stop_generic_locality / stop_invp_attribution:
  stop this Conv2D route and retain the existing InvP token-mixer anchor

invalid_protocol:
  repair the exact invariant and rerun the same R1 matrix without interpreting metrics
```

- [x] **Step 6: Commit and push the R1 adjudication**

```bash
git add docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md
git commit -m "experiment: adjudicate InvP state-matrix Conv2D seed0"
git push origin experiment/invp-state-matrix-conv2d
```

### Task 8: Final Verification And Handoff

**Files:**
- All task-scoped source, tests, configs, scripts, and experiment docs

- [x] **Step 1: Run focused tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d.py tests/test_invp_state_matrix_conv2d_gate.py tests/test_autond_dbitnet2023.py -q
```

Expected: pass.

- [x] **Step 2: Run Ruff on changed Python files**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py \
  src/blockcipher_nd/models/structure/spn/present_invp_state_matrix_conv2d.py \
  src/blockcipher_nd/models/structure/spn/__init__.py \
  src/blockcipher_nd/models/structure/__init__.py \
  src/blockcipher_nd/registry/model_families/spn.py \
  src/blockcipher_nd/engine/modeling.py \
  src/blockcipher_nd/planning/invp_state_matrix_conv2d_gate.py \
  src/blockcipher_nd/cli/gate_invp_state_matrix_conv2d.py \
  tests/test_invp_state_matrix_conv2d.py \
  tests/test_invp_state_matrix_conv2d_gate.py
```

Expected: pass.

- [x] **Step 3: Run the full registered suite and compare with baseline**

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Recorded current environment baseline: `30 failed, 852 passed` (29
Matplotlib 3.11 global-state failures plus one JSON route-alignment failure),
with every new test passing. Any additional failure is a regression and must be
fixed before completion.

- [x] **Step 4: Check repository diffs and status**

```bash
git diff --check
git status --short --branch
```

Expected: no uncommitted task changes after the final adjudication commit;
`experiment/invp-state-matrix-conv2d` matches its origin branch after the
normal push. This implementation was intentionally executed on the isolated
experiment branch rather than directly on `main`.

- [x] **Step 5: Report the actual evidence state**

The final report must state:

```text
hypothesis
exact R0/R1 commands and scale
all four model metrics
three AUC margins
gate decision
artifact paths
focused/full-suite verification
commit hashes and push state
paper-scale AutoND state as a separate task
recommended next action and explicitly stopped actions
```

Do not call the goal complete after R0 or a single `8192/class` seed. The
Innovation 1 objective remains active until the approved evidence ladder yields
a defensible method result or a documented negative result with the next
literature-ranked hypothesis adjudicated.

## Completion Record (2026-07-11)

- Execution branch: `experiment/invp-state-matrix-conv2d` (isolated branch,
  pushed to the matching origin branch; not executed directly on `main`).
- Strict evidence verification source head: `080f5ad` (`fix: bind InvP
  progress to result evidence`).
- Final verification head: `c2ea066` (`docs: refresh bound InvP gate
  artifacts`).
- Runtime: Python `3.10.16`.
- Focused verification: `233 passed` across the strict gate, Conv2D model,
  AutoND implementation/public protocol, and dataset-cache worker tests;
  additionally `4 passed, 447 deselected` for selected cache/progress
  project-structure tests.
- Ruff: format and check passed for the changed gate, CLI, and focused test
  files. `git diff --check` passed.
- Registered-suite environment baseline: exactly `30 failed, 852 passed` (29
  known Matplotlib 3.11 global-state failures plus one known JSON
  route-alignment failure); no new feature failure was introduced.
- R0 replay: existing `64/class`, one-epoch artifacts validated with mandatory
  progress evidence bound by ordered result/progress pair, exact seed,
  normalized result path, and result-declared cache root;
  `decision=implementation_ready`, metrics not interpreted, expanded gate size
  `25561` bytes.
- R1 replay: existing `8192/class`, ten-epoch artifacts validated with mandatory
  bound progress/cache provenance; `decision=stop_conv2d_route`,
  `next_action=keep_token_mixer_anchor_and_do_not_scale_conv2d`, expanded gate
  size `25978` bytes.
- R1 stopped actions: do not run seed1, `65536/class`, `262144/class`, or remote
  scale for the state-matrix Conv2D replacement route.
- H1 topology-residual adapter: planned only. It has not been implemented,
  trained, or run. No conditional seed1 or H1 work is represented as executed
  by the completed Task 1-8 checkboxes above.
