# Innovation 1 PRESENT InvP State-Matrix Conv2D Design

**Date:** 2026-07-11

**Status:** implementation complete; R0 local readiness passed; R1 seed0 local diagnostic approved but not started

**Claim scope:** controlled architecture-attribution experiment under the
strict PRESENT-80 r7 Zhang/Wang Case2 protocol; not a completed experiment,
paper-scale result, formal result, or novelty claim

## Decision

Advance one bounded, non-DDT architecture hypothesis:

> Given the same `InvP(DeltaC)` representation that already has two-seed
> `1000000/class` support, does a small state-matrix Conv2D encoder exploit the
> signal better than the current token mixer, and is any gain tied to the true
> PRESENT inverse-permutation mapping?

This experiment does not resume the stopped E1 graph branch, dense DDT trail
inputs, S-box-prior branch, or public random-ciphertext typed adapter. It does
not change the benchmark to match the running AutoND public-code reproduction.

## Evidence Basis

The strict-protocol InvP route is the strongest current Innovation 1 anchor:

| Evidence | AUC | Status |
| --- | ---: | --- |
| InvP-only seed0, `1000000/class` | `0.797470988906` | retrieved, validated, plan-aligned |
| InvP-only seed1, `1000000/class` | `0.797347588554` | retrieved, validated, plan-aligned |
| DeltaC-only seed0, `1000000/class` | `0.792064879854` | retrieved, validated, plan-aligned |
| shuffled-P seed0, `1000000/class` | `0.793621524954` | retrieved, validated, plan-aligned |

The minimum InvP AUC exceeds the strongest attribution control by
`+0.003726063600`. This supports true `InvP` alignment as a useful input signal,
but it does not show that the current token mixer is the best network for that
signal.

The literature re-audit ranks a controlled typed-representation/network
comparison above additional graph or DDT scaling. State-matrix Conv2D is prior
art as an architecture family, so the experiment is a comparator and method
calibration step. The defensible Innovation 1 contribution remains the
cipher-spec-driven adaptation and explicit same-input/shuffled-structure
attribution procedure, provided the empirical gates support it.

## Frozen Research Protocol

All rows use the existing strict benchmark:

```text
cipher                     = PRESENT-80
rounds                     = 7
difference profile         = present_zhang_wang2022_mcnd
difference member          = 0
pairs per sample           = 16
feature encoding           = ciphertext_pair_bits
negative mode              = encrypted_random_plaintexts
sample structure           = zhang_wang_case2_official_mcnd
configured train key       = 0x00000000000000000000 (inert placeholder)
configured validation key  = 0x11111111111111111111 (inert placeholder)
configured rotation        = 0 (inert placeholder)
effective key schedule     = per_pair_random
effective key scope        = independent random PRESENT key per basic pair
loss                       = mse
optimizer                  = adam
learning rate              = 0.0001
weight decay               = 0.00001
learning-rate scheduler    = official_cyclic
max learning rate          = 0.002
checkpoint metric          = val_auc
restore best checkpoint    = true
early stopping patience    = 8
early stopping min delta   = 0.0001
primary metric             = validation AUC
```

Validation labels, effective key schedule, negative construction, metric
computation, checkpoint selection, and plan-alignment logic must not change.
For this sample structure, effective key behavior must be verified from the
generated train/validation dataset metadata, not inferred from configured key
fields.

## Input Semantics

The model receives the same raw `16 x 128` ciphertext-pair bits as the current
InvP-only anchor. The typed view is constructed inside the model so the dataset
and cache remain identical across rows.

For each ciphertext pair:

```text
(C, C')
  -> DeltaC = C xor C'
  -> mapping(DeltaC)
  -> 16 nibbles x 4 bits
  -> transpose to 4 bit planes x 16 PRESENT cells
```

The three Conv2D variants differ only in `mapping`:

```text
true InvP   = exact PRESENT inverse permutation
shuffled-P  = existing deterministic seed-20260627 pseudo permutation
DeltaC-only = identity mapping
```

The true and shuffled mappings must reuse the existing verified inverse-index
construction. Tests must compare their generated tensors directly with the
current `_PresentNibblePAlignedSpnEncoder` views. No DDT values, plaintext
metadata, active-cell metadata, S-box lookup probabilities, trail summaries,
or public-code random-ciphertext labels enter the model.

## Candidate Architecture

Add one small pair encoder with three mapping modes rather than three copied
implementations.

Per pair:

```text
[batch, 1, 4, 16]
  -> 1x1 Conv2D, BatchNorm2D, ReLU stem to base_channels
  -> three residual state-matrix blocks
  -> each block uses 3x3 Conv2D, BatchNorm2D, ReLU, Dropout2D,
     3x3 Conv2D, BatchNorm2D, residual addition, and ReLU
  -> mean and max pooling over the 4 x 16 state matrix
  -> projection to the existing InvP anchor embedding width
```

Across the 16 pairs:

```text
pair embeddings
  -> mean pooling over pairs
  -> max pooling over pairs
  -> concatenate
  -> classifier with the same widths and activation family as the InvP anchor
  -> one logit
```

Default model options:

```text
base_channels = 32
conv_depth    = 3
kernel_size   = 3
activation    = relu
norm          = batchnorm2d
dropout       = 0.0
```

The implementation must reuse the existing `conv2d_norm` helper with
`batchnorm2d`; it must not add a normalization wrapper, dependency, or
general-purpose framework. The pair projection is
`Linear(2 * base_channels, 4 * base_channels)` followed by ReLU. With the
default `base_channels=32`, every pair becomes a 128-dimensional embedding.
The pair-level classifier is the same shape as the anchor: normalization over
the concatenated 256-dimensional mean/max embedding, `Linear(256, 256)`, ReLU,
dropout, and `Linear(256, 1)`.

The result row must record total trainable parameters for the anchor and all
candidates. The three Conv2D variants must have exactly equal parameter counts;
the Conv2D candidate does not need to equal the token-mixer parameter count,
but the report must state their ratio so capacity is not hidden.

## Model Matrix

The research matrix contains four rows because this is a planned attribution
study. No unrelated architecture is added.

| Row | Model role | Mapping | Purpose |
| --- | --- | --- | --- |
| 1 | existing `present_nibble_invp_only_spn_only` | true InvP | same-budget strongest anchor |
| 2 | new true-InvP state-matrix Conv2D | true InvP | architecture candidate |
| 3 | new shuffled-P state-matrix Conv2D | deterministic shuffled mapping | topology-attribution control |
| 4 | new DeltaC-only state-matrix Conv2D | identity | generic difference/capacity control |

Frozen model keys:

```text
present_nibble_invp_state_matrix_conv2d_spn_only
present_nibble_shuffled_p_state_matrix_conv2d_spn_only
present_nibble_delta_state_matrix_conv2d_spn_only
```

## Scale Ladder

Scale is evidence-driven rather than automatic:

### R0: implementation readiness

```text
samples_per_class = 64
seed              = 0
epochs            = 1
device            = cpu
```

R0 proves tensor semantics, cache compatibility, finite forward/backward
passes, result serialization, plotting, and plan alignment. Its metrics are not
research evidence.

### R1: local architecture-attribution diagnostic

```text
samples_per_class = 8192
seed              = 0
epochs            = 10
rows              = 4
```

`8192/class` is a diagnostic screen, not formal training and not a definitive
route failure. Run seed1 at the same scale only if seed0 passes the promotion
gate below. R1 must reuse one parameter-matched dataset cache across all rows.

### R2: medium confirmation

Only after both R1 seeds pass:

```text
samples_per_class = 65536, then 262144
seeds             = 0 and 1
```

These are medium diagnostics. A remote R2 launch requires a pushed commit,
disk-backed parameter-matched caches, progress JSONL, readiness validation,
and watcher-managed retrieval under `G:\lxy`.

### R3: formal claim gate

Only after the `262144/class` evidence remains positive:

```text
samples_per_class >= 1000000
seeds             >= 2
```

Only completed, retrieved, plan-aligned R3 evidence can support a formal
architecture result. The experiment still cannot claim that Conv2D itself is
novel; it can support the controlled cipher-spec adaptation result.

## R1 Promotion And Stop Gates

Define, for each seed:

```text
architecture_margin = AUC(true Conv2D) - AUC(token-mixer anchor)
topology_margin     = AUC(true Conv2D) - AUC(shuffled-P Conv2D)
representation_margin = AUC(true Conv2D) - AUC(DeltaC-only Conv2D)
```

Seed0 promotes to seed1 only when all conditions hold:

```text
plan alignment passes for 4/4 rows
all metrics are finite
architecture_margin > 0
topology_margin >= +0.003
representation_margin >= +0.003
true Conv2D is above every control
no cache, label, layout, or checkpoint invariant fails
```

After seed1, R2 is allowed only when:

```text
true Conv2D is above all three comparators on both seeds
mean architecture_margin >= +0.001
minimum topology_margin >= +0.002
minimum representation_margin >= +0.002
```

Interpretation branches:

| Observation | Decision |
| --- | --- |
| candidate beats anchor and both controls on both seeds | keep architecture hypothesis; proceed to R2 |
| candidate beats anchor but not shuffled-P | capacity or generic locality, not true PRESENT structure; stop candidate |
| candidate beats shuffled-P but not DeltaC-only | InvP attribution unsupported in this architecture; stop candidate |
| candidate loses to anchor | token mixer remains stronger; stop Conv2D route |
| one seed passes and one fails | unstable diagnostic; no remote scale, inspect training variance once |
| protocol/layout/cache gate fails | invalid experiment; repair and rerun without interpreting metrics |

The numeric margins are promotion thresholds, not scientific effect-size laws.
Reports must include raw per-seed AUCs, all three deltas, accuracy, loss, epoch
histories, and uncertainty language.

## R0 Local Readiness Evidence (2026-07-11)

Run id:

```text
i1_present_invp_state_matrix_conv2d_smoke_seed0
```

The first exact launch exposed a frozen-plan readiness defect before any row
completed: `balanced_per_class` rows redundantly set `train_samples_total` and
`validation_samples_total`, while the dataset validator reserves those fields
for `random_labels_total`. Commit `f45bbc6` blanked those fields in both the R0
and R1 matrices and updated the focused plan-contract test. This did not change
the budgets: `samples_per_class=64` still creates 128 balanced training rows and
the standard half-size validation rule still creates 64 balanced validation
rows. The focused regression suite passed `38/38` after the repair.

The first completed R0 artifacts then exposed a protocol-reporting mismatch
during specification review. The document incorrectly described the configured
train and validation key fields as fixed effective encryption keys, while
`zhang_wang_case2_official_mcnd` intentionally uses an independent random
PRESENT key for every basic pair. Those earlier artifacts were invalid for the
documented protocol-identity claim because result rows did not serialize the
effective dataset schedule. They remain useful only as the evidence that
revealed the mismatch; they have been deleted and replaced by the clean rerun
below. Commit `1023ced` added train/validation `key_schedule` result metadata
and made the attribution gate require `per_pair_random` for both splits.

The corrected implementation then completed a clean-cache rerun with this exact
command:

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

Observed protocol was PRESENT-80 r7, seed 0, `64/class`, 128 total balanced
training rows, 64 total balanced validation rows, one epoch, batch size 32,
hidden bits 32, 16 pairs/sample, strict encrypted-random-plaintext negatives,
effective `per_pair_random` scheduling with an independent random PRESENT key
for every basic pair, and CPU execution. The configured train key
`0x00000000000000000000`, validation key `0x11111111111111111111`, and rotation
interval `0` remained inert deterministic plan/cache placeholders rather than
effective encryption keys. The exact ordered model keys were:

```text
present_nibble_invp_only_spn_only
present_nibble_invp_state_matrix_conv2d_spn_only
present_nibble_shuffled_p_state_matrix_conv2d_spn_only
present_nibble_delta_state_matrix_conv2d_spn_only
```

All four rows completed one finite forward/backward training epoch and emitted
finite result metrics. For reproducibility, the un-interpreted smoke values
were:

| Model key | Validation AUC | Accuracy | Loss |
| --- | ---: | ---: | ---: |
| `present_nibble_invp_only_spn_only` | `0.5771484375` | `0.5` | `0.7021275758743286` |
| `present_nibble_invp_state_matrix_conv2d_spn_only` | `0.498046875` | `0.5` | `0.7061755955219269` |
| `present_nibble_shuffled_p_state_matrix_conv2d_spn_only` | `0.4228515625` | `0.5` | `0.7033177614212036` |
| `present_nibble_delta_state_matrix_conv2d_spn_only` | `0.3828125` | `0.5` | `0.7206352800130844` |

These metric values are recorded only as finite serialization evidence and are
not interpreted as architecture quality or research evidence.

Plan validation passed with `status=pass`, `plan_rows=4`, `result_rows=4`, and
`errors=[]`. Progress contained four `row_done` events and ended with exactly
one `run_done` event. The corrected attribution gate also returned
`status=pass` and `errors=[]` when run with the R0 scale parameters. Its
metric-derived decision is not interpreted at readiness scale. Cache generation
created exactly one train identity and one validation identity:

```text
outputs/local_cache/i1_present_invp_state_matrix_conv2d_smoke_seed0/present80/r7/train/seed-0_b10ab47bffcc5873
outputs/local_cache/i1_present_invp_state_matrix_conv2d_smoke_seed0/present80/r7/validation/seed-10000_88856f69c830db86
```

The first row created the train cache with 128 rows and the validation cache
with 64 rows. Rows 2-4 each reused both exact paths, producing six
`cache_reuse` events. The metadata matched across model rows on cipher, rounds,
seed/split, difference, feature encoding, 16-pair structure, strict negative
definition, label mode, row budget, and effective schedule. All four result
rows serialized `training.key_schedule=per_pair_random` and
`validation.key_schedule=per_pair_random`; both cache metadata files also
reported `key_schedule=per_pair_random`.

The anchor has `128673` total/trainable parameters. Each of the true-InvP,
shuffled-P, and DeltaC-only Conv2D rows has exactly `130881` total/trainable
parameters. All counts are positive and the candidate/anchor capacity ratio is
`130881 / 128673 = 1.0171597771` (approximately `1.01716x`).

R0 artifacts:

| Artifact | Size |
| --- | ---: |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/results.jsonl` | 16282 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/progress.jsonl` | 82406 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/validation.json` | 3750 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/curves.svg` | 55086 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/history.csv` | 1372 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/readiness_gate.json` | 1156 bytes |

`claim_scope=implementation readiness only; metrics not interpreted`.

Recommended next action: proceed to the already frozen R1 seed0 local
architecture-attribution diagnostic with `8192/class`, four identical ordered
rows, 10 epochs, and the same-budget token-mixer anchor plus shuffled-P and
DeltaC-only controls. Change only the evidence scale from R0 to R1. Require
4/4 plan alignment, finite metrics, one train cache plus one validation cache
reused across all rows, equal Conv2D parameter counts, and the existing R1
promotion/stop gates. R1 must also require `per_pair_random` in both serialized
result sections and both cache metadata files. Do not launch remote scale, add
models, change the benchmark, or interpret R0 metrics.

## Artifacts And Adjudication

Every non-smoke stage produces:

```text
results.jsonl
progress.jsonl
history.csv
curves.svg
validation.json
state_matrix_conv2d_gate.json
```

The gate artifact must record:

```text
exact model rows and model options
input tensor semantic checks
parameter counts and candidate/anchor ratio
dataset/cache identity across rows
per-seed metrics and margins
promotion conditions
decision = keep | stop | redesign | invalid
next action and explicitly stopped actions
claim scope
```

When a meaningful result completes, update this document in the same turn with
the run id, artifact paths, protocol gate, metrics, deltas, decision, claim
scope, and recommended next step.

## Testing And Verification

The implementation must add focused tests for:

1. Exact true-InvP tensor equality with the current verified InvP-only view.
2. Deterministic shuffled-P mapping and inequality with true InvP.
3. DeltaC-only identity mapping.
4. Correct `[batch, pair, bit_plane, cell]` layout with no accidental
   `[word, cell, bit]` reinterpretation.
5. Equal parameter counts across all Conv2D controls.
6. Forward shape and finite backward gradients for 16-pair batches.
7. Registry/model-options construction.
8. Four-row plan alignment and gate branch behavior.
9. Existing InvP-only model behavior remains unchanged.

Verification uses project commands with `UV_CACHE_DIR=/tmp/uv-cache` and
`uv run pytest`. The existing full-suite Matplotlib 3.11 and JSON alignment
baseline failures remain separate unless a focused new test exposes a direct
regression.

## Explicit Non-Goals

This design does not:

- use DDT, beam statistics, trail positions, or S-box probability tables;
- resume E1 active-cell or P-layer graph scaling;
- use public-code random-ciphertext negatives;
- change the input difference, pair count, keys, labels, or validation split;
- add attention, U-Net, ECA, NAS, multi-query aggregation, or cross-cipher
  transfer in the same experiment;
- launch remote scale from an R0 smoke or a weak/invalid R1 gate;
- call `8192/class`, `65536/class`, or `262144/class` formal evidence;
- claim Conv2D or InvP as a standalone invention.

## Relationship To The Running AutoND Task

The running public-code paper-scale AutoND reproduction remains independent:

```text
PRESENT-80 r9
single pair
random_labels_total
random_ciphertext negatives
per-row random key
r5 -> r9 curriculum
```

Its result calibrates the public AutoND implementation and published reference.
It is not an anchor row for this strict r7 Case2 matrix. Local R0/R1 work may
proceed after implementation approval, but no new remote Conv2D task launches
until the active paper-scale task is retrieved and adjudicated or an explicit
GPU scheduling exception is documented.

## Approved Next Step

The reviewed implementation plan is:

```text
docs/experiments/
innovation1-present-invp-state-matrix-conv2d-implementation-plan.md
```

Select an approved execution mode, then follow that plan task by task. Source
edits, R0 readiness, and R1 adjudication have not started at this status point.
