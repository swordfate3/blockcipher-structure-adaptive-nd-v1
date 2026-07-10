# Innovation 1 PRESENT r8 Active-Cell Layout Repair Re-Adjudication

## Status

```text
status = approved design / implementation not started
route = E1-R active-cell semantic-layout repair and re-adjudication
claim_scope = local matched-negative integral diagnostic only
remote_scale = no
```

This plan supersedes the next-action portion of the completed E1 8192/class
gate. It does not delete or rewrite the recorded E1 metrics. Those metrics
remain valid measurements of commit `8c75600`, but source inspection after the
run found that the active-cell graph consumed the encoder output under the
wrong layout contract. E1 therefore did not cleanly adjudicate a
PRESENT-cell-aligned topology architecture.

## Decision Question

```text
After the active-cell model correctly decodes the established global
bit-plane feature layout, does true PRESENT topology beat shuffled topology
and the existing metadata-only control consistently enough to justify one
8192/class confirmation?
```

The repair and the first re-adjudication are one bounded correctness workflow.
They are not a new architecture search and do not authorize a pair-count,
auxiliary-scale, seed, or sample-size sweep.

## Verified Failure Contract

The configured feature encoding is:

```text
present_pair_xor_paligned_sinv_cell_matrix_bits
```

For each ciphertext pair it constructs five 64-bit semantic words:

```text
W0 = ciphertext_a
W1 = ciphertext_b
W2 = ciphertext_a XOR ciphertext_b
W3 = InvP(W2)
W4 = InvS(InvP(ciphertext_a)) XOR InvS(InvP(ciphertext_b))
```

`words_to_present_cell_matrix_bits` emits the resulting 320 bits in global
bit-plane order. With five words and sixteen cells, encoded index `k` has:

```text
bit_plane = k // 80
word_index = (k % 80) // 16
cell_index = k % 16
```

The current `PresentActiveCellGraphPairSetDistinguisher._cell_tokens` instead
reshapes the input directly as `[batch, 5, 16, 4]`. That operation assumes
word-major, cell-major, bit-minor input and mixes multiple semantic cells in
each graph token. Active source roles, P-layer target roles, persistent edges,
topology contrast, and active-relative slot contrast are consequently applied
to positions that are not individual PRESENT cells.

## Repair Design

Keep the encoder and its established cache/input contract unchanged. Repair
only the active-cell graph's model-side decoding.

The model must interpret one pair as:

```python
planes = pair_features.reshape(batch, 4, words_per_pair, 16)
cells = planes.permute(0, 3, 2, 1).reshape(batch, 16, words_per_pair * 4)
```

The semantic axes after the permutation are:

```text
[batch, cell, word, bit]
```

For the E1 encoding, each of the sixteen cell tokens therefore contains
exactly twenty input bits: four bits from each of `W0..W4` at the same cell.
The existing cell encoder, cell-position embedding, graph layers, edge
encoders, pair pooling, contrast branches, auxiliary loss, and classifier stay
unchanged.

The implementation should expose the layout conversion as a small private
method such as `_cell_features`. This gives the semantic contract one directly
testable boundary without introducing a new public abstraction, model key,
feature encoding, or dependency.

## Required Tests

Implementation starts with a failing semantic-layout test.

### Encoder-to-token sentinel test

Construct five deterministic 64-bit sentinel words with nonuniform nibble
patterns, encode them through `words_to_present_cell_matrix_bits`, and pass the
320-bit result through the model's cell-feature layout conversion.

Assert all of the following:

```text
shape = [1, 16, 20]
cell c contains W0[c] bits followed by W1[c] ... W4[c] bits
no token contains bits from a different cell
cell 0 is the most-significant nibble position
cell 15 is the least-significant nibble position
```

### Active-coordinate test

For every active nibble `a` in `0..15`, assert:

```text
source_cells[a] = 15 - a
```

Use a sentinel whose only active semantic nibble is `a` and verify that the
selected source token is the corresponding encoded cell. This test must join
the metadata convention to the repaired layout instead of testing the source
index table in isolation.

### Regression tests

The existing true, shuffled, metadata-only, persistent-edge, consistency,
topology-auxiliary, topology-contrast, and active-relative forward tests must
continue to pass. The repair must not change model dimensions or accepted
configuration values.

Required verification commands use the project environment:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_project_structure.py -k 'present_active_cell_graph or present_cell_matrix' -q

UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q

git diff --check
```

## Dataset Protocol Held Fixed

E1-R deliberately does not redesign the dataset. Each sample still contains:

```text
4 ciphertext pairs * 320 feature bits + 16 active-nibble metadata bits
= 1296 input bits
```

For a sampled active nibble `a`, the Zhang/Wang nibble difference `0x9` is
relocated to `a`. The four positive pairs use variants `[0,1,2,3]` and retain
the aligned `0x9` difference. The matched-negative row cyclically pairs those
variants as `[1,2,3,0]`, producing active-nibble plaintext differences
`[0x8,0xA,0x8,0xA]` before encryption.

Although the matrix metadata says `negative_mode = encrypted_random_plaintexts`,
this sample-structure branch is not an independent-random-plaintext negative
protocol. All E1-R reporting must call it a `matched-negative integral`
diagnostic. No strict-negative or publication-style claim is allowed from this
plan.

## Experiment Stages

### Stage 0: Runtime smoke

After source tests pass, create a three-row seed0 runtime smoke:

```text
routes = true, shuffled, metadata_only
samples_per_class = 64
pairs_per_sample = 4
epochs = 3
batch_size = 64
device = cpu
```

The smoke checks only dataset width, finite forward/backward behavior,
auxiliary loss, result serialization, progress output, and plan alignment. Its
AUC is not evidence and must not be used for route selection.

### Stage 1: E1-R1 2048/class

If Stage 0 passes, run exactly six rows:

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 2048
validation_samples_per_class = 1024
pairs_per_sample = 4
seeds = 0,1
routes = true, shuffled, metadata_only
epochs = 3
batch_size = 64
hidden_bits = 16
device = cpu
```

Keep all other E1 protocol values unchanged:

```text
feature_encoding = present_pair_xor_paligned_sinv_cell_matrix_bits
sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active_metadata
integral_active_nibbles = [0..15]
difference_profile = present_zhang_wang2022_mcnd
difference_member = 0
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
loss = mse
learning_rate = 0.0001
optimizer = adam
weight_decay = 0.00001
checkpoint_metric = val_auc
restore_best_checkpoint = true
token_dim = 32
graph_depth = 2
token_mlp_ratio = 2
pooling = topk_logsumexp
top_k = 4
metadata_bits = 16
edge_mode = persistent
cross_pair_consistency = edge_mean_absdev
active_metadata_fusion = coordinate_only
topology_auxiliary_scale = 0.3
topology_contrast_fusion = true_minus_shuffled
active_relative_contrast_fusion = true_minus_shuffled_slots
```

The six-row matrix changes no research variable relative to the historical
2048/class active-relative contrast matrix. The only intended behavioral
change is the verified model-side layout correction.

### Stage 2: Conditional E1-R2 8192/class

Do not create or run E1-R2 until the E1-R1 gate passes. If allowed, E1-R2
keeps the same six rows and changes only:

```text
samples_per_class: 2048 -> 8192
validation_samples_per_class: 1024 -> 4096
```

E1-R2 remains a local diagnostic. It does not authorize a 16k/32k/65k ladder
or a remote run.

## Frozen E1-R1 Gate

E1-R1 passes only if all conditions hold:

```text
true > shuffled on seed0 and seed1
true > metadata_only on seed0 and seed1
true - shuffled >= +0.020000 on seed0 and seed1
all 6 result rows validate against the plan
no non-finite loss, failed row, or seed-specific control reversal
```

Decision table:

| E1-R1 outcome | Decision |
| --- | --- |
| true loses to shuffled or metadata-only on either seed | stop the active-cell topology route and proceed to E2 |
| ordering holds but either true-shuffled margin is below `+0.02` | classify as corrected but still fragile; stop E1 and proceed to E2 |
| all frozen conditions pass | permit one E1-R2 8192/class confirmation |

The threshold is an adjudication rule for this local matched-negative protocol,
not a statistical significance or formal cryptanalytic threshold.

## E1-R2 Interpretation

If E1-R2 is permitted, it must retain the required ordering and avoid a return
to near-tied true-shuffled margins on both seeds. A mixed or collapsed result
stops the architecture route. A stable result keeps the route alive only as a
matched-negative topology diagnostic.

Before any medium or remote scale, a separate approved plan must add:

```text
strict independent encrypted-random-plaintext negatives
a genuinely topology-free local control
an explicit deterministic baseline appropriate to that protocol
```

Those controls are intentionally not mixed into E1-R because they would change
more than the feature-layout hypothesis.

## Artifacts And Validation

Each executed stage must produce its own run directory containing:

```text
results.jsonl
progress.jsonl
history.csv
curves.svg
```

For E1-R1, result validation must require exactly six rows. The result report
must include per-seed true, shuffled, and metadata-only AUC values; both control
deltas; training versus validation samples per class; and the exact source
commit.

## Documentation Updates

When the repair is implemented, update the completed E1 plan and the route
verdict to state:

```text
historical E1 status = completed implementation-misaligned diagnostic
historical E1 topology verdict = superseded / not adjudicated
next adjudication = E1-R1 active-cell layout repair 2048/class
E2 status = deferred until E1-R1 gate resolves
```

When E1-R1 completes, update this plan and the route verdict in the same turn
with run id, artifacts, metrics, deltas, gate status, claim scope, and exactly
one next action.

## Non-Goals

This workflow does not:

- change `words_to_present_cell_matrix_bits`;
- add a new feature encoding or model key;
- change labels, negative generation, validation data, or AUC computation;
- redesign metadata-only or topology contrast during E1-R1;
- tune graph depth, token size, auxiliary scale, pooling, pair count, or seeds;
- launch a remote job;
- claim that a matched-negative result is a strict PRESENT r8 distinguisher.
