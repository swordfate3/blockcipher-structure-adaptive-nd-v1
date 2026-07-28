# Innovation 1 uKNIT-Family Runtime Cell-Semantic Histogram Blueprint

Date: 2026-07-28

## Status And Activation Guard

```text
status = method blueprint frozen; implementation and training not authorized yet
authority = pending K1-U medium adjudication
source mechanism = K1-T deterministic stage/cell histogram residual
primary cipher = uKNIT-BC r5
cell-count stress target = Dialga-128 r4
```

This blueprint defines the next architecture only if both K1-U seeds pass the
preregistered medium gate. K1-U is still running remotely, so this document is
not an experiment launch plan and does not authorize code changes, local
training, remote training, or a transfer claim.

If K1-U does not pass, follow its existing decision tree instead of implementing
this candidate. In particular, a wrong-S-box attribution failure requires a
five-stage contribution audit, a position-erasure failure selects the simpler
invariant branch, and a failed seed requires checkpoint/training-dynamics
inspection with the protocol unchanged.

## Evidence Boundary

K1-T is the first local uKNIT r5 result that made the confirmed cell11 signal
neural and attributable at `2048/class`:

| Seed | Exact cross-key AUC | Wrong S-box | Position erased |
| ---: | ---: | ---: | ---: |
| 3 | `0.713162` | `0.506986` | `0.565424` |
| 4 | `0.748229` | `0.512875` | `0.594048` |

This is a local mechanism diagnostic, not formal scale or family transfer.
K1-U asks whether the same result survives at `65536/class`; no K1-U AUC is
available locally yet.

The next method question is therefore conditional and narrow:

> If the K1-T position residual survives K1-U, can its fixed sixteen-cell
> projection be replaced by one shared, runtime-structure-conditioned
> aggregator whose trainable parameter geometry is identical for 16-cell and
> 32-cell SPNs?

## Current Implementation Audit

The runtime descriptor contract is already variable width. `RuntimeSpnStructure`
derives `block_bits`, `cells` and transition count from externally loaded cell
membership, bit-role, S-box and GF(2) tensors. It validates every four-bit cell,
arbitrary invertible binary linear matrices and per-round/per-cell S-box tables.

The K1-N exact-composition backbone also already has identical state geometry
for uKNIT-64 and Dialga-128:

```text
uKNIT K1-N trainable parameters = 131875
Dialga K1-N trainable parameters = 131875
state-dict names and shapes      = identical
```

The immediate barrier is isolated to
`models/structure/spn/position_histogram_residual.py`:

```text
histogram                   = [batch, 5 stages, C cells, 16 values]
current accepted C          = 16 only
shared value encoder        = Linear(16, 8)
current projection input    = flatten(5 x 16 x 8) = 640
current projection          = Linear(640, 128)
```

The deterministic histogram calculation is otherwise naturally defined for
any `C`. The successor must generalize only the cell aggregation. It must not
replace the supported exact inverse-S-box/inverse-GF(2) composition backbone.

Verified descriptor panel for a two-transition window:

| Descriptor | Bits | Cells | Transitions | Linear fan-in | Distinct cell S-boxes |
| --- | ---: | ---: | ---: | --- | ---: |
| uKNIT rounds 3/4 | 64 | 16 | 2 heterogeneous | 3 | 16 |
| Dialga rounds 2/3 | 128 | 32 | 2 heterogeneous | 3 | 1 |
| SKINNY repeated round | 64 | 16 | 1 homogeneous | 1-3 | 1 |
| GIFT/PRESENT/RECTANGLE repeated round | 64 | 16 | 1 homogeneous | 1 | 1 |

No numeric cell ID is needed to identify uKNIT cells: the two S-box tables plus
local labeled GF(2) incidence signature distinguish all `16/16` cells. Dialga
has `16` structural equivalence classes of size two among its 32 cells. A
four-step labeled neighborhood refinement does not split those pairs. The model
must preserve that real symmetry rather than inject an arbitrary learned cell
number to force uniqueness.

## Difference-Position Gate And Target Choice

The target is Dialga-128 prefix-r4, not prefix-r5.

uKNIT r5 uses the already frozen K1-Q cell11 role1 difference
`0x0000400000000000`. It passed discovery plus untouched seed/key confirmation,
so it must remain fixed for architecture comparison.

Dialga r4 uses the existing D1 `0x40` protocol. D1 is recent, plan-aligned,
strict-negative evidence for this exact cipher, round count, key scopes, pair
count and evaluation protocol:

```text
seed0 correct AUC = 0.958417
seed1 correct AUC = 0.958679
correct-corrupted = +0.022107 / +0.020863
correct-no-topology = +0.461207 / +0.455274
```

That evidence satisfies the project difference-position calibration exception:
fresh signal already exists for the exact target protocol, so a second r4
position sweep would duplicate a completed signal gate. The new experiment
plan must cite the D1 gate explicitly.

Dialga r5 is blocked. D5 screened all 128 Hamming-weight-one positions across
two key panels. Its strongest worst-key AUC was only `0.510098`, and no position
passed the frozen `0.520` per-key gate. Do not combine the new aggregator with
another Dialga r5 difference search, multi-bit search, DDT/trail feature or
sample increase.

## Candidate: Runtime Cell-Semantic Histogram Aggregator

Keep the five exact K1-N/K1-T stages:

```text
ciphertext
inverse linear 1
inverse S-box 1
inverse linear 0
inverse S-box 0
```

For batch size `B` and runtime cell count `C`, build the same deterministic
histogram as K1-T:

```text
H in R[B, 5, C, 16]
```

The candidate replaces only the fixed flatten projection with the following
shared pipeline.

### 1. Shared Value Encoding

Apply the existing encoder independently to every stage/cell histogram:

```text
V[s,c] = ReLU(Linear_16_to_8(H[s,c]))
```

No parameter depends on `C`.

### 2. Runtime Cell-Semantic Descriptor

For each cell, derive a deterministic descriptor from the exact two-transition
runtime structure. For each transition slot include:

```text
64 S-box truth bits
16 incoming counts indexed by target-role x source-role
16 outgoing counts indexed by source-role x target-role
```

The two slots therefore contribute `2 x 96 = 192` values per cell. Apply one
shared descriptor encoder:

```text
Linear(192, 72) -> ReLU -> Linear(72, 32) -> ReLU -> LayerNorm(32)
```

Project the 32-value structural embedding through `Linear(32, 8)` for the
value/structure interaction. Add a five-way stage code only after the cell
descriptor is encoded.

This descriptor contains no cipher name, cipher ID, block width, numeric cell
ordinal, key, label, seed or learned per-cell embedding. It is recomputed from
the runtime S-box and GF(2) tensors. Jointly relabeling cells, the input bits and
the runtime topology must only permute these descriptors.

### 3. Shared Stage/Cell Token

For every `(stage, cell)` create one token from:

```text
encoded value histogram       = 8
encoded runtime cell semantics = 32
five-way stage code            = 5
value x structure interaction  = 8
total token input              = 53
```

Use `Linear(53, 64) -> ReLU -> LayerNorm(64)` as one shared token encoder for
all cells, both ciphers and all five stages.

### 4. Cell-Count-Independent Pooling

For each stage independently, aggregate its `C` 64-wide tokens with one shared
64-wide attention pool and explicit mean/max/RMS summaries. Project the
concatenated 256-wide summary through `Linear(256, 64)`, ReLU and LayerNorm.
Concatenate the five ordered stage embeddings and project the resulting
320-wide tensor through `Linear(320, 128)`, ReLU and LayerNorm to the same
128-wide histogram residual used by K1-T.

```text
C stage/cell tokens
  -> shared attention + mean + max + RMS over cells
  -> fixed-width stage summary
5 ordered stage summaries
  -> fixed Linear(..., 128)
  -> repeat to 384 and add through the existing bounded histogram gate
```

The order of the five physical inverse-composition stages remains meaningful.
Cell order does not. Position information is retained through runtime semantic
descriptors and value/structure binding, not through a 16-slot lookup table.

## Required Controls

Keep the first matrix to three trainable roles per cipher and seed:

| Role | Exact composition | Runtime descriptor binding | Purpose |
| --- | --- | --- | --- |
| exact | correct S-boxes | correct cell semantics | candidate |
| wrong S-box | deterministic wrong S-box semantics | otherwise identical | nonlinear-semantic control |
| position erased | correct S-boxes | histogram averaged over cells before tokenization | native-position control |

All three roles must have identical trainable parameter names, shapes and
counts. They must consume the same cached rows and differ only in the frozen
semantic control transformation.

For homogeneous-S-box Dialga, wrong-S-box attribution is descriptive rather
than a primary gate because every native cell shares one table. Dialga's primary
role is a 32-cell geometry/performance-retention stress test. uKNIT remains the
primary S-box and position-attribution task.

## Zero-Training Readiness Gates

Implementation is authorized only after K1-U passes. Before any optimizer
step, require all of the following:

1. `deterministic_position_histogram` returns `[B,5,16,16]` for uKNIT and
   `[B,5,32,16]` for Dialga without a cell-count branch.
2. uKNIT and Dialga candidate state dictionaries have identical names and
   shapes and exactly equal trainable parameter counts.
3. Candidate, wrong-S-box and position-erased controls have identical geometry.
4. Both 64-bit and 128-bit inputs produce finite `[B,1]` logits.
5. Jointly relabeling runtime cells/topology and corresponding input bits leaves
   logits equal within a frozen numerical tolerance.
6. Relabeling only the descriptor binding, without relabeling histogram values,
   changes logits for a non-degenerate fixture.
7. Wrong S-boxes and position erasure each change the intended deterministic
   tensor while preserving all unrelated tensors.
8. Histogram-branch gradients are finite and nonzero for both cell counts.
9. No parameter or buffer axis is semantically indexed by runtime cell count.
   Ordinary hidden widths of `32` and the fixed nibble-value width of `16` are
   allowed but must be explicitly distinguished from cell-slot axes in model
   metadata and tests.
10. Source inspection and model metadata confirm no cipher-ID table, numeric
    cell embedding, key feature, label feature or active-difference metadata.
11. With the frozen widths above, the histogram branch contains `82554`
    parameters including its scalar gate and the whole model contains `214429`.
    This is exactly `113` parameters (`+0.0527%`) above K1-T's `214316`, well
    inside the one-percent capacity-matching gate.

The readiness artifact must record the descriptor collision audit (`16/16`
uKNIT unique signatures and `16/32` Dialga equivalence classes) rather than
pretend every cell is structurally identifiable.

## Conditional First Training Gate

Only after K1-U and zero-training readiness pass, create a formal experiment
plan under `docs/experiments/`. The one changed variable is the histogram cell
aggregator.

```text
primary task = uKNIT-BC r5 cell11 role1
stress task  = Dialga-128 r4 difference 0x40
train        = 2048/class/cipher
validation   = 1024/class/cipher
pairs        = 4
epochs       = 10
device       = local sub-medium diagnostic
uKNIT seeds  = 3,4 using exact K1-T caches
Dialga seeds = 0,1 using exact D1 caches
roles        = exact, wrong S-box, position erased
negative     = encrypted random plaintexts
checkpoint   = restored best validation AUC
```

This is twelve rows but only three model roles. The two ciphers are optimized
independently in this first gate. Passing it proves shared architecture and
state geometry, not shared learned weights or zero-shot transfer.

Use completed same-budget anchors rather than retraining them:

```text
uKNIT anchor = K1-T exact position histogram residual
Dialga anchor = K1-N exact composition candidate on the D1 data protocol
```

Advance only if both uKNIT seeds independently satisfy:

```text
exact cross-key AUC >= 0.600
exact - wrong S-box >= +0.010
exact - position erased >= +0.030
exact - K1-T exact >= -0.020
```

For both Dialga seeds require:

```text
exact cross-key AUC >= K1-N exact anchor - 0.005
finite nonzero same-checkpoint logit response to position erasure
no protocol, cache, parameter-geometry or relabeling failure
```

Do not require Dialga wrong-S-box separation because its 32 cells use one
homogeneous S-box table. Do not average a failed uKNIT seed away with Dialga's
high absolute AUC.

## Claim Ladder And Stop Routes

If the first gate passes, the supported claim is only:

> One fixed-shape runtime cell-semantic histogram architecture can be trained
> without geometry changes on a 16-cell heterogeneous uKNIT task and a 32-cell
> heterogeneous Dialga task while retaining the uKNIT position mechanism.

It is not yet shared-weight transfer or unseen-cipher generalization. A later
shared-weight or frozen-checkpoint experiment requires its own plan and cannot
be inferred from independently optimized models.

Stop or redirect as follows:

- K1-U fails: do not implement this blueprint; follow the K1-U failure branch.
- uKNIT retention fails: discard the shared aggregator; do not compensate with
  width, epochs, pairs, samples or additional seeds.
- uKNIT position-erased margin fails: prefer the simpler invariant branch.
- Dialga retention fails with valid geometry: inspect whether its real
  two-cell structural equivalence classes are over-collapsed; do not inject
  numeric cell IDs or cipher identity.
- Readiness relabeling fails: repair equivariance before any training.
- Both ciphers pass: freeze the shared state geometry and preregister exactly
  one actual shared-weight/frozen-checkpoint transfer experiment.

Blocked in this phase: MoE, sparse experts, learned cipher routing, arbitrary
cell embeddings, a generic Transformer replacement, DDT/trail features,
Dialga r5 difference reopening, remote scale, `16k/32k/65k` mechanical growth,
and any universal-SPN, attack, SOTA or breakthrough claim.
