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
attention scorer:

```text
Linear(64, 64) -> ReLU -> LayerNorm(64) -> Linear(64, 1)
```

Apply softmax only over the runtime cell axis, then take the weighted sum of
the original 64-wide tokens. Concatenate that attention summary with explicit
mean/max/RMS summaries. Project the resulting 256-wide summary through
`Linear(256, 64)`, ReLU and LayerNorm. Concatenate the five ordered stage
embeddings and project the resulting 320-wide tensor through
`Linear(320, 128)`, ReLU and LayerNorm to the same 128-wide histogram residual
used by K1-T. The scorer is shared across every stage, cell count and cipher;
there is no stage-specific or cipher-specific attention module.

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

The exact frozen branch count is:

| Component | Parameters |
| --- | ---: |
| Shared histogram value encoder `16 -> 8` | `136` |
| Cell descriptor `192 -> 72 -> 32` plus `LayerNorm(32)` | `16296` |
| Value/structure interaction `32 -> 8` | `264` |
| Stage/cell token `53 -> 64` plus `LayerNorm(64)` | `3584` |
| Shared attention scorer `64 -> 64 -> 1` plus `LayerNorm(64)` | `4353` |
| Stage summary `256 -> 64` plus `LayerNorm(64)` | `16576` |
| Ordered-stage projection `320 -> 128` plus `LayerNorm(128)` | `41344` |
| Bounded histogram gate | `1` |
| **New histogram branch total** | **`82554`** |

K1-T's replaced fixed branch contains `82441` parameters: `136` in the shared
value encoder, `82048` in `Linear(640, 128)`, `256` in `LayerNorm(128)` and one
scalar gate. The successor therefore changes the whole-model capacity by only
`82554 - 82441 = 113` parameters. Any implementation with a different scorer,
an extra attention module per stage, or a learned stage/cell lookup fails this
frozen capacity gate even if its total remains below the broad parameter cap.

The readiness artifact must record the descriptor collision audit (`16/16`
uKNIT unique signatures and `16/32` Dialga equivalence classes) rather than
pretend every cell is structurally identifiable.

### Pre-Implementation Formula Audit

The deterministic formulas above were evaluated without adding a model or
running an optimizer. A fixed binary fixture was transformed together with a
random runtime cell relabeling. For each old cell, its four input bits, S-box
table and every source/target endpoint in both inverse GF(2) matrices were moved
to the corresponding new cell before the descriptor and histogram were
recomputed.

```text
uKNIT:
  descriptor shape             = [16, 192]
  histogram shape              = [2, 5, 16, 16]
  descriptor relabel equality  = exact
  histogram relabel equality   = exact

Dialga:
  descriptor shape             = [32, 192]
  histogram shape              = [2, 5, 32, 16]
  descriptor relabel equality  = exact
  histogram relabel equality   = exact
```

This is formula/readiness evidence only. It proves that the proposed
deterministic inputs are jointly cell-equivariant for the two target widths; it
does not prove that the future neural aggregation will learn, retain K1-T AUC,
or transfer weights. The implemented model must repeat this audit end to end on
its logits before training.

## Conditional Compact-Invariant Failure Route

This route activates only if a completed K1-U seed fails the frozen
`exact - position erased >= +0.030` gate while retaining the exact absolute AUC
and wrong-S-box margin. It is not authorized by partial epoch history and no
implementation or optimizer step may start before the six-row K1-U result is
retrieved and adjudicated.

The K1-T/K1-U invariant control does not remove histogram cell slots. It first
averages the exact histogram over native cells, repeats that mean into sixteen
identical slots, and then applies the same fixed `640 -> 128` projection as the
position-preserving candidate. The sixteen repeated slots make most of that
projection algebraically redundant.

For the shared encoded stage histogram

```text
Z[b,s,d] = value_encoder(mean_c H[b,s,c,:])
```

the current invariant branch computes

```text
y[o] = bias[o] + sum_(s,c,d) W_old[o,s,c,d] * Z[s,d]
```

because every repeated cell slot receives the same `Z[s,d]`. Define

```text
W_compact[o,s,d] = sum_c W_old[o,s,c,d]
```

and the branch is exactly the same function with a `5 x 8 = 40` input:

```text
y[o] = bias[o] + sum_(s,d) W_compact[o,s,d] * Z[s,d]
```

This collapse retains the complete K1-N exact inverse-S-box, inverse-GF(2) and
topology-edge backbone. It removes only the redundant repeated histogram slots;
it does not turn the whole network into a structure-free histogram classifier.
The resulting histogram branch is naturally independent of runtime cell count:

```text
H: [B, 5, C, 16]
  -> mean over C
  -> shared Linear(16, 8) + ReLU
  -> flatten [B, 40]
  -> Linear(40, 128) + ReLU + LayerNorm(128)
  -> existing bounded histogram gate
```

The exact parameter geometry is:

| Component | Current repeated invariant | Compact invariant |
| --- | ---: | ---: |
| Shared value encoder | `136` | `136` |
| Histogram projection including bias | `82176` | `5248` |
| LayerNorm | `256` | `256` |
| Bounded gate | `1` | `1` |
| Histogram branch | `82441` | `5641` |
| Whole model | `214316` | `137516` |

The compact route removes `76800` trainable parameters. This is intentional
simplification after a failed position-necessity gate, not a capacity-matched
candidate for the position-preserving hypothesis.

A zero-training float64 formula audit on 2026-07-28 used random
`[7,5,32,16]` histograms and a random current-geometry projection. Summing the
sixteen cell-slot weights produced:

```text
maximum old-versus-compact output error = 3.3306690738754696e-16
maximum 32-cell relabeling error         = 1.6653345369377348e-16
```

This establishes only algebraic collapsibility. If activated, readiness must
repeat the check on every restored K1-U invariant checkpoint and require:

The formula was also audited against both completed K1-T invariant checkpoints
without training or optimizer steps. On a fixed 37-row legal 512-bit fixture,
the complete restored-model logits agreed after folding the sixteen repeated
cell-slot weights into the compact projection:

```text
seed3 maximum full-logit error = 5.960464477539063e-08
seed4 maximum full-logit error = 1.1175870895385742e-07
tolerance                         = 1e-6
original / compact parameters     = 214316 / 137516
```

Both K1-T checkpoints pass. This is stronger than the random formula audit but
remains pre-activation evidence only: it does not select the compact branch,
replace the incomplete K1-U gate or waive the required K1-U checkpoint replay.

The same restored models were then replayed at the frozen batch size of `64`
on their manifest-bound 2048-row cross-key validation caches. The original and
folded models produced exactly equal serialized AUC and fixed-threshold
accuracy on both seeds:

```text
seed3 AUC / accuracy       = 0.565424442 / 0.546875000
seed3 maximum logit error = 1.2665987014770508e-07
seed4 AUC / accuracy       = 0.594047546 / 0.573730469
seed4 maximum logit error = 1.7881393432617188e-07
```

This closes the K1-T metric-preservation precheck, but K1-U still requires the
same replay against its larger restored invariant checkpoints and exact cached
cross-key rows if the compact branch is selected.

A second zero-training audit used the real `uknit64.json` and `dialga128.json`
runtime descriptors. It found one concrete implementation interlock: the
current `deterministic_position_histogram` helper rejects every structure whose
cell count is not sixteen even though its tensor formula is otherwise dynamic.
Removing only that K1-T-specific guard in the temporary audit produced:

```text
uKNIT histogram / parameters = [7,5,16,16] / 137516
Dialga histogram / parameters = [7,5,32,16] / 137516
uKNIT joint-relabel error      = 1.4901161193847656e-07
Dialga joint-relabel error     = 8.940696716308594e-08
```

The two real-runtime prototypes had identical state-dict names and shapes,
strictly loaded one another's state, emitted finite `[7,1]` logits and had
finite nonzero gradients for every trainable histogram parameter. All errors
were below `1e-6`. A shared-weight wrong-S-box intervention remained
non-degenerate after invariant pooling: the maximum histogram/logit changes
were `0.140625/0.002902329` for uKNIT and `0.1015625/0.001866102` for Dialga.
If this branch activates, implementation must therefore generalize that single
geometry guard and add a 32-cell regression test; it must not fork a separate
Dialga histogram implementation.

1. compact and original invariant logits agree within `1e-6` on the exact
   cached cross-key validation rows;
2. AUC and accuracy agree within the metric serialization tolerance;
3. uKNIT and Dialga compact models have identical state-dict names, shapes and
   exactly `137516` trainable parameters;
4. `[B,5,16,16]` and `[B,5,32,16]` histograms produce finite `[B,1]` logits
   without a cell-count branch;
5. joint cell relabeling leaves logits unchanged within the frozen tolerance;
6. wrong S-box semantics change the deterministic pooled tensor and logits
   under a shared learned state;
7. all compact histogram gradients are finite and nonzero;
8. no cipher identity, numeric cell embedding, key, label or active-difference
   metadata is introduced.

If the checkpoint collapse passes, evaluate it before any retraining. Only then
may a separate plan compare independently optimized compact models on the
existing uKNIT r5 K1-T caches and Dialga r4 D1 caches. Do not compensate for the
smaller branch by adding width, depth, epochs, samples, pairs or an auxiliary
loss. A collapse failure authorizes only correction of tensor ordering or
checkpoint conversion; it does not reopen the larger runtime-descriptor
aggregator.

## Conditional Compact-Invariant First Training Gate

This gate exists only for the compact failure route. It activates after the
completed K1-U decision selects `medium_signal_without_position_necessity` and
the restored K1-U checkpoint conversion passes. The research question is not
whether absolute position helps; K1-U has already rejected that hypothesis.
The question is whether the algebraically equivalent compact parameterization
can be optimized independently without losing the existing invariant uKNIT
signal or the adjacent Dialga signal.

Use the same local data protocols and change only the redundant histogram
projection:

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
roles        = compact exact, compact wrong S-box
negative     = encrypted random plaintexts
checkpoint   = restored best validation AUC
```

This is eight rows and two model roles. Both roles must have exactly `137516`
trainable parameters and identical state-dict geometry for uKNIT and Dialga.
There is no separate position-erased row: compact exact is already invariant
over runtime cells, so such a row would be an identical duplicate rather than
a control.

Use completed rows as same-budget anchors:

```text
uKNIT anchor = K1-T invariant histogram residual
Dialga anchor = K1-N exact composition candidate on the D1 data protocol
```

For each uKNIT seed independently require:

```text
compact exact cross-key AUC >= max(0.550, K1-T invariant AUC - 0.020)
compact exact - compact wrong-Sbox AUC >= +0.010
```

The `0.550` floor recognizes that the completed K1-T invariant local anchors
are `0.565424/0.594048`; imposing the rejected position branch's `0.600` floor
would contradict the frozen same-budget evidence. The medium K1-U result, not
this local optimization check, remains the scale evidence.

For each Dialga seed independently require:

```text
compact exact cross-key AUC >= K1-N exact anchor - 0.005
wrong-Sbox pooled histogram and shared-state logits are non-degenerate
```

Dialga wrong-Sbox AUC separation remains descriptive because all native cells
share one S-box table. Across both ciphers also require strict cache reuse,
identical parameter geometry, finite nonzero histogram gradients and joint-cell
relabeling error within `1e-6`. A failed uKNIT seed or Dialga retention seed
holds this architecture; do not add width, data, epochs, pairs or cipher IDs.

Passing this gate proves only that one compact fixed-shape architecture can be
optimized independently on the two tasks. It does not prove shared weights,
zero-shot transfer, a formal-scale result or a universal SPN model.

## Conditional Runtime-Semantic First Training Gate

This gate exists only if both K1-U seeds pass the position-necessity margin and
therefore activate the runtime cell-semantic aggregator described above. It
must not be used after the compact-invariant branch is selected.

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

## Future Third-Cipher Structural Audit

This section ranks a future holdout only. It does not add a row to the first
uKNIT/Dialga gate, authorize implementation, or relax the K1-U interlock.

The preferred third cipher is `Midori64`, used as a homogeneous
component-equivalent holdout after the first variable-cell gate passes. `MANTIS`
is deferred to a later tweakable/reflection-protocol phase.

### Evidence For Midori64

The local uKNIT paper gives two exact identities relevant to the network
boundary:

```text
uKNIT cell S-box = D_(r,c) o S_MANTIS o B_(r,c)
uKNIT linear     = output permutation o L_MIDORI o input permutation
```

Its Appendix C therefore rewrites the complete uKNIT data path as shared
MANTIS substitutions and shared MIDORI linear layers separated by 24 runtime
bit permutations. This is stronger than a visual or family-name similarity: it
is the exact component factorization already checked by the K0 probes.

The local Dialga paper independently defines a Midori-type round as a cell
S-box, a cell permutation, the fixed Midori matrix and key addition. It also
records that the `Sb0` used by Midori64 is the same low-latency involutory
4-bit table used as the MANTIS/uKNIT canonical substitution. Dialga changes the
permutation schedule and, for its 128-bit construction, the cell width; it does
not weaken the reason to use native Midori64 as the direct 64-bit primitive
anchor.

The repository already contains the exact reusable primitives needed for a
zero-training compatibility audit:

```text
MANTIS_SBOX = DIALGA_SBOX
canonical 64-bit linear primitive = midori_linear_layer
runtime cell width = derived from cell_membership, not a model constant
shared S-box descriptor = accepted and expanded over runtime cells
repeated linear descriptors = accepted by RuntimeSpnStructure
```

Consequently, Midori64 changes the schedule regularity while keeping the
canonical primitive vocabulary fixed:

| Property | uKNIT-BC | Midori64 holdout |
| --- | --- | --- |
| State | 64 bits, 16 four-bit cells | 64 bits, 16 four-bit cells |
| S-box vocabulary | per-cell permutations of MANTIS/Midori `Sb0` | shared `Sb0` |
| Linear vocabulary | per-round bit-permutation equivalents of MIDORI | native MIDORI layer |
| Schedule | heterogeneous and non-round-aligned | homogeneous/repeated |
| Intended role | primary attribution task | unseen component-equivalent retention task |

This makes Midori64 a cleaner third-cipher question than adding another
heterogeneous design: can the same fixed-shape network reduce runtime
permutations to the native primitive without cipher identity or a new expert?

### Why MANTIS Is Deferred

MANTIS is highly relevant at the primitive level but is not the next controlled
whole-cipher experiment. Its tweakable reflection structure, middle layer and
tweak/key schedule add protocol variables that the current ordinary prefix-SPN
pair generator does not model explicitly. Adding it together with the new cell
aggregator would mix architecture, cipher protocol and data generation changes.
It remains the preferred later reflection/tweak stress target after native
Midori64 qualification.

`QARMA/QARMAv2` and `CRAFT` remain farther boundary candidates. Their use would
require a separate proof that every loaded transition factors into the frozen
primitive vocabulary; a shared four-bit-cell layout alone is insufficient.

### Missing Qualification Evidence

Midori64 is selected only as the next structural candidate. It is not yet a
loaded or trained project cipher. The repository currently has no Midori64
encryption adapter, runtime JSON descriptor, public-vector regression test,
prefix trace, or calibrated strict-negative differential task. Therefore no
Midori64 AUC, transfer or family-generalization claim is authorized.

After the first uKNIT/Dialga variable-cell gate passes, the required sequence is:

1. Implement a specification-faithful Midori64 adapter and verify public full
   vectors plus at least one intermediate-state trace.
2. Generate a native `midori64.json` descriptor from the verified adapter and
   prove exact forward/inverse S-box and GF(2) reconstruction.
3. Prove that the variable-cell model loads the unchanged uKNIT/Dialga state
   dictionary and remains logit-equivariant under joint Midori cell relabeling.
4. Run a separate, preregistered difference-position calibration at the chosen
   Midori prefix and strict encrypted-random-plaintext negative protocol.
5. Only after that signal gate, preregister one frozen-checkpoint holdout test
   with zero Midori training rows and correct-versus-wrong factorization
   controls.

The official MIDORI and SKINNY/MANTIS bibliographic identities are available in
the local uKNIT and Dialga references. Direct ePrint PDF retrieval for
`2015/1142` and `2016/660` was rechecked on 2026-07-28 but returned Cloudflare
challenge HTML, so this audit does not pretend those challenge files are
papers. The selection above relies only on the two locally stored full texts,
the exact K0 implementation, and the current runtime-structure source.

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
