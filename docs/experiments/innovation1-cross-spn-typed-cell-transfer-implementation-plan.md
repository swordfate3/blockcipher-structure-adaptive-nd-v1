# Innovation 1 Cross-SPN Typed Cell Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task in the current workspace. Steps use checkbox (`- [ ]`) syntax for tracking. Project instructions prohibit sub-agent dispatch unless explicitly requested.

**Goal:** Implement and locally adjudicate E4-R0/R1: a cipher-spec-generated typed cell operator with identical trainable state across PRESENT-80 and GIFT-64, plus matched true/shuffled/raw controls and a raw-input GIFT Token-Mixer anchor.

**Architecture:** Every E4 row consumes cached raw ciphertext-pair bits. The typed model computes `DeltaC`, applies a fixed generated inverse-permutation index, separately embeds current and previous-round 4-bit cells, fuses the two semantic roles, mixes 16 cell positions, and pools pair evidence. Only a non-persistent mapping buffer varies across the six typed aliases; the GIFT anchor reconstructs the historical aligned feature internally and delegates to the existing Token-Mixer.

**Tech Stack:** Python 3.10, PyTorch, NumPy-backed disk caches, CSV experiment matrices, pytest, existing JSONL/SVG/gate infrastructure.

---

## Frozen Scope

This plan implements only E4-R0 and E4-R1 from
`docs/experiments/innovation1-cross-spn-typed-cell-transfer-design.md`.

It does not implement checkpoint transfer or E4-R2. E4-R2 becomes authorized
only when the completed R1 PRESENT source gate returns `promote_e4_r2`.

No DDT, trail, beam-search, S-box-table score, negative-definition change,
remote launch, `65536/class`, or larger run belongs in this implementation.

## File Map

- Create `src/blockcipher_nd/models/structure/spn/cross_spn_typed_cell.py`: generated mapping, shared typed operator, fixed aliases, and GIFT raw-input anchor.
- Modify `src/blockcipher_nd/models/structure/spn/__init__.py`: export E4 classes and mapping helper.
- Modify `src/blockcipher_nd/models/structure/__init__.py`: re-export E4 model classes.
- Modify `src/blockcipher_nd/registry/model_families/spn.py`: construct E4 aliases with the existing option helpers.
- Create `src/blockcipher_nd/planning/cross_spn_typed_cell_gate.py`: PRESENT source gate, GIFT protocol gate, and combined decision.
- Create `src/blockcipher_nd/cli/gate_cross_spn_typed_cell.py`: JSON gate CLI.
- Create `scripts/gate-cross-spn-typed-cell`: thin executable wrapper.
- Create `tests/test_cross_spn_typed_cell.py`: semantic, capacity, state-dict, equivalence, and finite-gradient tests.
- Create `tests/test_cross_spn_typed_cell_gate.py`: protocol, decision, CLI, and matrix tests.
- Create four R0/R1 matrices under `configs/experiment/innovation1/`.
- Update `docs/experiments/innovation1-cross-spn-typed-cell-transfer-design.md`: append executed run IDs, artifacts, gate metrics, verdict, and next action only after R1 completes.
- Update `docs/experiments/innovation1-route-verdict-2026-07-09.md`: record only the final E4-R1 adjudication.

### Task 1: Generate Cipher-Spec Mapping Indices

**Files:**
- Create: `src/blockcipher_nd/models/structure/spn/cross_spn_typed_cell.py`
- Test: `tests/test_cross_spn_typed_cell.py`

- [ ] **Step 1: Write failing mapping tests**

Add tests that construct `cipher_inverse_permutation_indices(cipher_key,
mapping_mode)` for `present80` and `gift64` and require:

```python
for cipher_key in ("present80", "gift64"):
    for mode in ("true", "shuffled", "raw"):
        indices = cipher_inverse_permutation_indices(cipher_key, mode)
        assert indices.dtype == torch.long
        assert sorted(indices.tolist()) == list(range(64))

torch.testing.assert_close(
    cipher_inverse_permutation_indices("present80", "true"),
    present_inverse_p_indices("true"),
)
assert torch.equal(
    cipher_inverse_permutation_indices("present80", "shuffled"),
    cipher_inverse_permutation_indices("gift64", "shuffled"),
)
```

For GIFT, apply the generated tensor map to MSB-ordered bits of a
non-symmetric 64-bit integer and compare it with
`Gift64.inverse_permutation_layer` converted back to MSB-ordered bits.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cross_spn_typed_cell.py -q
```

Expected: collection fails because the new module does not exist.

- [ ] **Step 3: Implement the minimal generated mapping helper**

Use the repository cipher factory; do not copy permutation tables:

```python
_SHUFFLED_MAPPING_SEED = 20260627

def cipher_inverse_permutation_indices(
    cipher_key: str, mapping_mode: str
) -> torch.Tensor:
    if mapping_mode == "raw":
        return torch.arange(64, dtype=torch.long)
    if mapping_mode == "shuffled":
        generator = torch.Generator().manual_seed(_SHUFFLED_MAPPING_SEED)
        return torch.randperm(64, generator=generator)
    if mapping_mode != "true":
        raise ValueError(f"unsupported mapping_mode: {mapping_mode}")
    cipher = build_cipher(cipher_key, rounds=1, key=0)
    if cipher.block_bits != 64 or not hasattr(cipher, "inverse_permutation_layer"):
        raise ValueError(f"cipher does not expose a 64-bit inverse permutation: {cipher_key}")
    indices = [-1] * 64
    for source_msb_index in range(64):
        source_state = 1 << (63 - source_msb_index)
        mapped_state = cipher.inverse_permutation_layer(source_state)
        if mapped_state <= 0 or mapped_state & (mapped_state - 1):
            raise ValueError(f"inverse permutation is not one-hot for {cipher_key}")
        target_msb_index = 63 - (mapped_state.bit_length() - 1)
        indices[target_msb_index] = source_msb_index
    if sorted(indices) != list(range(64)):
        raise ValueError(f"inverse permutation is not bijective for {cipher_key}")
    return torch.tensor(indices, dtype=torch.long)
```

- [ ] **Step 4: Re-run mapping tests**

Expected: all mapping tests pass, including exact PRESENT compatibility and
bitwise GIFT equivalence.

### Task 2: Implement The Shared Typed Cell Operator

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/cross_spn_typed_cell.py`
- Test: `tests/test_cross_spn_typed_cell.py`

- [ ] **Step 1: Write failing typed-view and state compatibility tests**

Build all six aliases with raw `pair_bits=128`. Require exact typed views:

```python
pairs = features.reshape(batch, pairs_per_sample, 2, 64)
delta = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
expected_current = delta.reshape(batch, pairs_per_sample, 16, 4)
expected_previous = delta.index_select(2, model.mapping_indices).reshape(
    batch, pairs_per_sample, 16, 4
)
actual_current, actual_previous = model.typed_cell_view(features)
torch.testing.assert_close(actual_current, expected_current)
torch.testing.assert_close(actual_previous, expected_previous)
```

Under the same `torch.manual_seed`, require identical total/trainable counts,
identical `state_dict` keys, shapes, and tensors across PRESENT/GIFT
true/shuffled/raw variants. Require `mapping_indices` to be absent from
`state_dict()` and require finite `[batch, 1]` logits and gradients.

- [ ] **Step 2: Run tests and verify the new classes are missing**

Run the same focused pytest command. Expected: failures identify the missing
typed operator and aliases.

- [ ] **Step 3: Implement one shared model and six fixed aliases**

Implement `CrossSpnTypedCellPairSetDistinguisher` with these fixed tensor
operations and trainable modules:

```text
raw [B,P,2,64]
-> delta and mapped delta [B,P,16,4]
-> separate current/previous Linear(4, token_dim) encoders
-> concatenate and Linear(2*token_dim, token_dim) typed fusion
-> one shared 16-position embedding
-> SpnTokenMixerBlock repeated mixer_depth times
-> mean/max/active cell summary
-> pair projection
-> existing AttentionPooling
-> attention/mean/max pair summary
-> binary classifier
```

Constructor defaults are frozen as:

```python
pair_bits = 128
token_dim = max(16, base_channels * 2)
mixer_depth = 2
token_mlp_ratio = 2
activation = "relu"
norm = "layernorm"
pooling = "attention_mean_max"
dropout = 0.0
```

Register mapping indices with:

```python
self.register_buffer("mapping_indices", indices, persistent=False)
```

Use thin subclasses that set only `cipher_key` and `mapping_mode`, rejecting
conflicting caller values. Do not add cipher IDs, embeddings, round inputs,
S-box/DDT values, or cipher-specific trainable heads.

- [ ] **Step 4: Run typed model tests**

Expected: exact view, alias, state compatibility, invalid-shape, and finite
forward/backward tests pass.

### Task 3: Implement The Raw-Input GIFT Anchor

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/cross_spn_typed_cell.py`
- Test: `tests/test_cross_spn_typed_cell.py`

- [ ] **Step 1: Write the locked logit-equivalence test**

Instantiate `GiftAlignedTokenMixerRawInputDistinguisher` and an external
`SpnTokenMixerPairSetDistinguisher(input_bits=P*256, pair_bits=256)` with the
same delegate state. Build the historical aligned tensor from raw bits:

```python
aligned = torch.cat([c0, c1, delta, mapped_delta], dim=2).reshape(batch, -1)
torch.testing.assert_close(wrapper(raw), external(aligned), rtol=0, atol=0)
```

- [ ] **Step 2: Run the equivalence test and verify it fails**

Expected: the wrapper is not defined.

- [ ] **Step 3: Implement the wrapper**

The wrapper must validate raw 128-bit pairs, generate the true GIFT mapping,
construct `C0 || C1 || DeltaC || InvP_GIFT(DeltaC)` per pair, and delegate to
the existing Token-Mixer using `pair_bits=256`. It must expose no additional
trainable layers and use the historical GIFT options:

```python
mixer_depth = 1
activation = "relu"
norm = "layernorm"
pooling = "topk_logsumexp"
top_k = 2
lse_temperature = 1.0
```

- [ ] **Step 4: Re-run the equivalence test**

Expected: exact logits, identical gradients, and runtime-width checks pass.

### Task 4: Register E4 Models

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/__init__.py`
- Modify: `src/blockcipher_nd/models/structure/__init__.py`
- Modify: `src/blockcipher_nd/registry/model_families/spn.py`
- Test: `tests/test_cross_spn_typed_cell.py`

- [ ] **Step 1: Write failing registry tests**

Require `build_model` to construct these exact keys:

```text
present_cross_spn_typed_cell_true
present_cross_spn_typed_cell_shuffled
present_cross_spn_typed_cell_raw
gift_cross_spn_typed_cell_true
gift_cross_spn_typed_cell_shuffled
gift_cross_spn_typed_cell_raw
gift_cross_spn_aligned_token_mixer_raw_anchor
```

- [ ] **Step 2: Add imports, exports, and one registry dispatch table**

Forward only `token_dim`, `mixer_depth`, `token_mlp_ratio`, `activation`,
`norm`, `pooling`, and `dropout` to typed aliases. Forward historical
Token-Mixer options to the GIFT anchor. Preserve `pair_bits=128` from raw
`ciphertext_pair_bits` data.

- [ ] **Step 3: Run registry/model tests**

Expected: all aliases build through the public factory and reject non-128-bit
raw pair inputs.

### Task 5: Freeze R0/R1 Matrices

**Files:**
- Create: `configs/experiment/innovation1/innovation1_spn_present_cross_spn_typed_cell_smoke_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_cell_smoke_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_present_cross_spn_typed_cell_8192_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_cell_8192_seed0.csv`
- Test: `tests/test_cross_spn_typed_cell_gate.py`

- [ ] **Step 1: Write failing exact-task matrix tests**

Require four rows per matrix in anchor/true/shuffled/raw order. Lock R0 to
`64/class`, seed0, one epoch at runtime; lock R1 to `8192/class`, seed0, ten
epochs. Require raw `ciphertext_pair_bits`, strict encrypted-random-plaintext
negatives, MSE, Adam, `1e-4` learning rate, `1e-5` weight decay, no scheduler,
`val_auc`, restored best checkpoint, and no early stopping.

PRESENT remains r7 Case2 official MCND, 16 pairs, profile member 0, effective
`per_pair_random`. GIFT remains r6 independent pairs, 4 pairs, profile member
0, fixed train/validation keys.

- [ ] **Step 2: Add the four CSV matrices**

PRESENT model order:

```text
present_nibble_invp_only_spn_only
present_cross_spn_typed_cell_true
present_cross_spn_typed_cell_shuffled
present_cross_spn_typed_cell_raw
```

GIFT model order:

```text
gift_cross_spn_aligned_token_mixer_raw_anchor
gift_cross_spn_typed_cell_true
gift_cross_spn_typed_cell_shuffled
gift_cross_spn_typed_cell_raw
```

- [ ] **Step 3: Run matrix tests**

Expected: every parsed task matches the frozen protocol and all four roles in
each matrix share one raw dataset-cache identity.

### Task 6: Implement The Combined E4-R0/R1 Gate

**Files:**
- Create: `src/blockcipher_nd/planning/cross_spn_typed_cell_gate.py`
- Create: `src/blockcipher_nd/cli/gate_cross_spn_typed_cell.py`
- Create: `scripts/gate-cross-spn-typed-cell`
- Test: `tests/test_cross_spn_typed_cell_gate.py`

- [ ] **Step 1: Write failing gate tests**

Use synthetic complete result/progress/cache-terminal fixtures to require:

```text
R0 valid both ciphers -> implementation_ready, metrics ignored
PRESENT true >= .65 and all margins pass -> promote_e4_r2
PRESENT ordered and only one margin misses by <= .002 -> run_present_seed1_fragility
PRESENT/control ordering failure -> reject_e4_shared_operator
either protocol invalid -> invalid_e4_protocol
GIFT weak scratch metrics with valid protocol -> does not block PRESENT promotion
```

Also reject unequal typed capacities, missing histories/checkpoints, wrong
PRESENT effective key schedule, wrong GIFT fixed-key metadata, missing cache
completion/reuse evidence, wrong row count, seeds, sample count, or epochs.

- [ ] **Step 2: Implement separate protocol checks and one combined decision**

Reuse `evaluate_four_role_attribution` where its generic invariants apply.
Define a PRESENT `FourRoleProtocolSpec` from
`PRESENT_CASE2_ATTRIBUTION_PROTOCOL`. Define an explicit GIFT protocol spec
with r6, 4 pairs, raw 512-bit sample width, strict negatives, independent-pair
sampling, fixed key schedule, and the frozen optimizer/runtime fields.

The combined R1 source decision is exactly:

```python
passes = (
    true_auc >= 0.65
    and true_auc - anchor_auc >= -0.01
    and true_auc - shuffled_auc >= 0.003
    and true_auc - raw_auc >= 0.003
)
```

The bounded fragility exception applies only when `true` is above shuffled and
raw, all non-missed conditions pass, and exactly one required margin is short
by at most `0.002`. Otherwise stop E4 before transfer.

- [ ] **Step 3: Implement the CLI and wrapper**

The CLI accepts separate repeatable `--present-results`, `--present-progress`,
`--gift-results`, and `--gift-progress` paths plus `--samples-per-class`,
`--epochs`, `--readiness-only`, and `--output`. It writes newline-terminated,
sorted JSON and exits nonzero only for invalid protocol/status failure.

- [ ] **Step 4: Run focused gate tests**

Expected: semantic, decision, invalid-protocol, CLI, and matrix cases pass.

### Task 7: Run E4-R0 Readiness

**Files:**
- Output only: `outputs/local_smoke/i1_present_cross_spn_typed_cell_r0_seed0/`
- Output only: `outputs/local_smoke/i1_gift64_cross_spn_typed_cell_r0_seed0/`

- [ ] **Step 1: Run focused source tests and static checks**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_cross_spn_typed_cell.py \
  tests/test_cross_spn_typed_cell_gate.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/models/structure/spn/cross_spn_typed_cell.py \
  src/blockcipher_nd/planning/cross_spn_typed_cell_gate.py \
  src/blockcipher_nd/cli/gate_cross_spn_typed_cell.py \
  tests/test_cross_spn_typed_cell.py \
  tests/test_cross_spn_typed_cell_gate.py
```

- [ ] **Step 2: Run PRESENT and GIFT R0 with disk cache**

Use `scripts/train --plan ... --epochs 1 --batch-size 32 --hidden-bits 32
--device cpu --dataset-cache-chunk-size 64 --dataset-cache-workers 1
--train-eval-interval 1`, distinct output/progress/checkpoint directories, and
route-owned cache roots. Re-run each matrix once against the same cache and
verify parameter-matched reuse.

- [ ] **Step 3: Validate and plot each R0 result**

Run `scripts/validate-results` with `--expected-rows 4` for each matrix. Run
`scripts/plot-results` to produce `curves.svg` and `history.csv`, then parse both
SVGs as XML.

- [ ] **Step 4: Replay the neutral combined gate**

Run `scripts/gate-cross-spn-typed-cell` with `--samples-per-class 64 --epochs 1
--readiness-only`. Expected: `status=pass`, `decision=implementation_ready`,
and no interpretation of R0 AUC values.

### Task 8: Run And Adjudicate E4-R1

**Files:**
- Output only: `outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/`
- Output only: `outputs/local_smoke/i1_gift64_cross_spn_typed_cell_r1_seed0/`
- Modify: `docs/experiments/innovation1-cross-spn-typed-cell-transfer-design.md`
- Modify: `docs/experiments/innovation1-route-verdict-2026-07-09.md`

- [ ] **Step 1: Launch R1 automatically after R0 passes**

Run each 8192 matrix locally with `--epochs 10 --batch-size 256
--hidden-bits 32 --device cpu --dataset-cache-chunk-size 512
--dataset-cache-workers 4 --train-eval-interval 1`, distinct progress/result/
checkpoint directories, and reusable disk cache. This is diagnostic only.

- [ ] **Step 2: Validate, plot, and gate R1**

Require four result rows per cipher, complete histories and selected
checkpoints, valid SVG/CSV artifacts, exact plan alignment, and combined gate
`status=pass`. Report all eight AUCs and PRESENT true-minus-anchor/shuffled/raw
margins; report GIFT scratch values without allowing weak GIFT AUC to veto a
valid PRESENT source gate.

- [ ] **Step 3: Apply exactly one next action**

```text
promote_e4_r2:
  freeze and implement the separate checkpoint-transfer plan; do not launch
  transfer from ad hoc code.

run_present_seed1_fragility:
  create only the same-budget PRESENT seed1 matrix and run it locally; do not
  repeat GIFT or scale samples.

reject_e4_shared_operator:
  stop E4 typed transfer, do not implement R2, and consolidate the strongest
  InvP-only route-level evidence as the Innovation 1 method result.

invalid_e4_protocol:
  repair and replay the invalid artifact/protocol; do not interpret metrics.
```

- [ ] **Step 4: Document evidence and executable recommendation**

Append run IDs, exact protocols, artifact paths, metrics, deltas, gate report,
claim scope, stopped actions, and the selected next action to the E4 design and
route-verdict documents. Explicitly retain `8192/class diagnostic`, not formal
or paper-scale language.

### Task 9: Verify, Commit, And Push

**Files:** all task-scoped source, tests, matrices, scripts, and experiment docs.

- [ ] **Step 1: Run full verification**

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache UV_CACHE_DIR=/tmp/uv-cache \
  uv run pytest -q
git diff --check
```

Expected baseline: all tests pass; the pre-E4 baseline was `1199 passed`.

- [ ] **Step 2: Review task-scoped diff and repository state**

Confirm no DDT/trail route, benchmark definition, negative sampling, unrelated
file, or remote launcher changed. Confirm ignored result artifacts remain under
`outputs/`.

- [ ] **Step 3: Make scoped commits and normal pushes**

Commit implementation/readiness separately from the completed R1 adjudication
when both exist. Push each commit to configured `origin/main`; if platform
approval rejects a push, retain the local commit and do not use an alternate
transfer route.

## Plan Self-Review

- Spec coverage: mapping generation, six identical typed variants, GIFT anchor
  equivalence, R0/R1 matrices, cache reuse, combined source gate, controls,
  artifacts, and conditional next action are each assigned to an explicit task.
- Scope: checkpoint transfer and R2 are deliberately excluded until the R1
  gate authorizes them.
- Type consistency: all raw models receive 128 bits/pair; PRESENT uses 16 pairs
  (`2048` input bits), GIFT uses 4 pairs (`512` input bits), while only the
  internal GIFT anchor delegate receives 256 bits/pair.
- Evidence language: both R0 and R1 remain local diagnostics; no formal,
  paper-scale, remote, ceiling, or breakthrough claim is authorized.
