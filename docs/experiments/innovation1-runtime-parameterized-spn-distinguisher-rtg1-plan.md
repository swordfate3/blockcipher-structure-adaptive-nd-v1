# Innovation 1 Runtime-Parameterized SPN Distinguisher Plan

**Status:** R1 GIFT seed0 not supported; redesign before PRESENT or scale

**Date:** 2026-07-23
**Experiment label:** RTG1

## Method-Level Objective

Design one runtime-parameterized neural distinguisher for 4-bit-cell SPN
ciphers. The model receives ciphertext pairs and an external public structure
description containing cell membership, per-cell S-box truth tables, and
per-round binary linear maps. Its trainable parameter names and tensor shapes
must not depend on the cipher name, block width, number of cells, permutation,
or sparse GF(2) matrix.

The complete method claim requires all of the following:

1. one shared backbone accepts different SPN structures at runtime;
2. one-to-one bit permutations and general invertible GF(2) linear maps are
   both supported;
3. correct topology is attributable against degree-preserving corrupted and
   topology-disabled controls under the same data and budget;
4. source and target ciphers retain identical trainable state geometry;
5. multi-seed, multi-cipher evidence supports the topology ordering before any
   formal or transfer claim.

RTG1 is the implementation/readiness and first local attribution gate. It does
not by itself complete the method-level objective.

## Research Question

Can a cipher-name-free shared processor consume public SPN structure at
runtime and preserve useful topology signal across PRESENT-80, GIFT-64, and a
general-linear-layer SPN without changing its trainable state geometry?

## Same-Protocol Anchors

The current Innovation 1 anchor is the completed E4 typed-cell operator:

```text
input               = raw ciphertext pairs
derived view        = DeltaC and cipher-generated InvP(DeltaC)
source              = PRESENT-80 r7 Case2, 16 pairs/sample
target              = GIFT-64 r6, 4 pairs/sample
negative definition = encrypted random plaintexts
```

RTG1 must not change labels, negative construction, key sampling, validation
data, checkpoint selection, loss, or metric computation to help the new model.

## Runtime Structure Contract

The public structure object contains no cipher identifier, key, DDT, trail,
beam statistic, or label-derived feature. It provides:

```text
block_bits                         N, divisible by 4
cell_membership                    [N], contiguous ids with four bits/cell
bit_role                           [N], a permutation of 0..3 inside each cell
sbox_truth_bits                    [R, cells, 64]
linear_matrices                    [R, N, N], target-by-source over GF(2)
inverse_linear_matrices            [R, N, N], validated exact inverses
```

The structure is passed to `forward`; it is not registered in `state_dict`.
Different structures may have different runtime tensor dimensions while the
model parameters remain unchanged.

## Architecture

Each raw pair is normalized to `[batch, pairs, 2, N]`. The shared model:

```text
C0, C1 -> DeltaC
        -> shared bit-value encoder
        -> shared bit-role embedding
        -> for each selected inverse round:
             exact predecessor = A_inverse * Delta mod 2
             cell context       = mean over runtime cell membership
             graph context      = normalized sparse A_inverse aggregation
             S-box context      = shared encoder of the runtime truth table
             residual update    = shared MLP + LayerNorm
        -> attention/mean/max node pooling
        -> one embedding per ciphertext pair
        -> attention/mean/max pair-set pooling
        -> shared binary head
```

Exact GF(2) propagation and neural graph aggregation are both required. A
normal sum/mean GNN cannot represent XOR semantics by itself.

No fixed-width flattening, absolute position embedding, cipher-specific head,
or cipher-specific trainable adapter is allowed.

## Controls

The readiness implementation must expose equal-capacity modes:

| Role | Structure relation | Purpose |
| --- | --- | --- |
| `true` | exact public cell/S-box/linear structure | candidate |
| `corrupted` | deterministic source-column permutation preserving row degrees and source-degree multiset | topology attribution |
| `independent` | same modules and parameters but zero cell/graph/S-box contexts | no-topology control |

For a permutation matrix, corruption must remain a permutation. For a general
GF(2) matrix, corruption must preserve row degrees and the multiset of column
degrees while changing at least one edge.

## R0: Implementation Readiness

Run on CPU with synthetic ciphertext pairs and structure fixtures for:

```text
PRESENT-like 64-bit permutation
GIFT-like 64-bit permutation
SKINNY-like 64-bit sparse GF(2) map
synthetic 128-bit 32-cell permutation
```

Required checks:

- all structures validate as invertible GF(2) maps;
- forward/backward outputs and gradients are finite;
- output shape is `[batch, 1]` for variable pair counts and block widths;
- all modes and all structures use identical parameter names/shapes/counts;
- runtime structure tensors are absent from `state_dict`;
- true and corrupted structures produce different logits on a non-symmetric
  input under common weights;
- independent mode is not implemented by removing trainable modules;
- a permutation matrix and its index/gather representation produce the same
  exact GF(2) predecessor bits;
- cell relabeling with conjugated topology preserves logits within `1e-6`;
- malformed cells, S-boxes, singular matrices, and shape mismatches fail fast.

R0 metrics are not research evidence.

## R1: Local Attribution Diagnostic

R1 requires a separate config after R0 passes. The planned scale is:

```text
train samples_per_class      = 8192
validation samples_per_class = 4096
epochs                       = 10
seed                         = 0
device                       = local CPU/GPU
```

Use lean within-cipher matrices with the strongest same-protocol anchor and
the necessary true/corrupted/independent roles. Do not combine PRESENT and
GIFT scores into one pooled claim.

R1 authorizes one same-budget seed1 repeat only when both ciphers satisfy:

```text
runtime true >= fixed E4 anchor - 0.005 AUC
runtime true - corrupted       >= +0.005 AUC
runtime true - independent     >= +0.005 AUC
```

A miss of at most `0.002` on exactly one margin is a fragility check; a larger
miss stops local scaling and requires redesign.

## R2: Conditional Remote Medium Diagnostic

Only a two-seed R1 pass can authorize a remote plan:

```text
samples_per_class = 65536
seeds             = 0,1
epochs            = 10
execution         = lxy-a6000
```

This is medium diagnostic evidence, not formal or paper-scale training. The
runner must have parameter-matched disk caches, progress artifacts, resume,
verified pushed source, and watcher-managed retrieval under `G:\lxy`.

## Later General-Linear Gate

After permutation attribution passes, use a verified standard SPN with a
general linear layer, preferably SKINNY-64, without adding cipher-specific
trainable classes. uKNIT is eligible only after its implementation and public
vectors pass an independent cipher-correctness audit.

Formal claims require at least `1000000/class`, multiple seeds, completed
retrieval, and same-budget comparisons against E4, DBitNet, and a literature-
aligned Conv2D/MBConv baseline. No such run is authorized by this plan.

## Stop Boundaries

RTG1 does not authorize:

- DDT, trail, beam, or handcrafted probability features;
- related-key or integral labels in the standard differential matrix;
- simultaneous Conv2D, DBitNet, DRSN, and graph fusion;
- fixed 64-bit flattening disguised as a shared model;
- remote execution from a weak local gate;
- calling `8192/class` or `65536/class` formal evidence;
- claiming stable topology superiority from readiness tensors or one seed.

## Required Artifacts And Recommendation

Every result-producing rung must produce validated JSONL/CSV/SVG/gate
artifacts and refresh `outputs/00_RECENT_RESULTS.md` plus
`outputs/00_RECENT_RESULTS.json` in the same turn. The completed record must
state the observed gate, claim boundary, and an evidence-backed next action.

## Executed R0 Readiness Record

R0 completed locally on 2026-07-23:

```text
run_id          = i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723
training        = no
research metric = no AUC or accuracy comparison
shared params   = 21059
fixtures        = PRESENT-64, GIFT-64, SKINNY-64, synthetic 128-bit
gate            = 15/15 checks passed
decision        = innovation1_runtime_spn_r0_readiness_passed
```

Artifacts:

```text
outputs/local_readiness/i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723/results.jsonl
outputs/local_readiness/i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723/progress.jsonl
outputs/local_readiness/i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723/cells.csv
outputs/local_readiness/i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723/contract.json
outputs/local_readiness/i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723/summary.json
outputs/local_readiness/i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723/gate.json
outputs/local_readiness/i1_rtg1_runtime_parameterized_spn_r0_readiness_seed0_20260723/curves.svg
```

The exact checks covered identical parameter geometry across 64/128-bit
inputs, runtime structure exclusion from `state_dict`, variable pair counts,
finite forward/backward, exact GF(2) inverses, P-layer gather equivalence,
general sparse GF(2) support, degree-preserving corruption, equal-capacity
independent mode, S-box sensitivity, cell-relabel equivariance, and malformed
contract rejection. The SVG was rendered to pixels at 1824 x 993 and passed
the visual QA gate without overlap, clipping, missing glyphs, or ambiguous
research-performance language.

The nonzero differences between random-initialization logits are only an
implementation sensitivity check. They do not show that correct topology is
better than either control.

## R0 Verdict And Executable Next Action

R0 passes and therefore unlocks one R1 seed0 local diagnostic. The next
research question is whether the runtime topology processor preserves the E4
same-budget signal while attributing it to the correct externally supplied
structure. R1 must use the existing E4 PRESENT r7 Case2 and GIFT-64 r6 data
protocols without changing labels, negatives, keys, pairs per sample, loss,
validation, or checkpoint selection.

The single changed variable is the model representation. Within each cipher,
run exactly these four roles from identical seed0 initialization where model
geometry permits:

```text
fixed E4 typed-cell anchor
runtime correct topology
runtime degree-preserving corrupted topology
runtime independent/no-topology
```

Freeze the R1 budget at `8192/class` train, `4096/class` validation, 10 epochs,
seed0, and local CPU because CUDA is unavailable on this host. PRESENT and GIFT
must be adjudicated separately. The advance thresholds remain:

```text
runtime true >= fixed E4 anchor - 0.005 AUC
runtime true - corrupted           >= +0.005 AUC
runtime true - independent         >= +0.005 AUC
```

Only a pass for both ciphers may authorize a same-budget seed1 confirmation.
A miss of at most `0.002` on exactly one margin is a fragility repeat; a larger
miss stops scaling and returns to a one-variable architecture redesign. Do not
launch remote training, add extra model families, increase sample size, or
claim stable topology superiority before the two-cipher, two-seed gate passes.

## Executed R1 GIFT Seed0 Record

The lower-cost GIFT-64 matrix was deliberately run first. It reused the exact
E4 disk-backed r6 data protocol and completed locally on 2026-07-23:

```text
run_id          = i1_rtg1_gift64_runtime_parameterized_r1_8192_seed0_20260723
train           = 8192/class, 16384 total
validation      = 4096/class, 8192 total
epochs          = 10
seed            = 0
runtime params  = 163971 for every relation mode
negative mode   = encrypted_random_plaintexts
checkpoint      = best validation AUC
protocol gate   = 11/11 checks passed
```

Results:

| Role | Validation AUC | Best epoch |
| --- | ---: | ---: |
| fixed E4 typed-cell anchor | `0.551968932` | 6 |
| runtime correct topology | `0.506132305` | 1 |
| runtime degree-preserving corrupted | `0.507886916` | 1 |
| runtime independent/no-topology | `0.507204831` | 10 |

Margins:

```text
runtime true - E4 anchor  = -0.045836627
runtime true - corrupted  = -0.001754612
runtime true - independent = -0.001072526
```

All three research checks failed. This is a valid one-cipher seed0 local
diagnostic, not evidence about multi-seed or multi-cipher stability. It rejects
the current bit-to-pair global-pooling backbone at this budget; it does not
reject runtime structure parameterization as a method class.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_gift64_runtime_parameterized_r1_8192_seed0/results.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_runtime_parameterized_r1_8192_seed0/progress.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_runtime_parameterized_r1_8192_seed0/history.csv
outputs/local_diagnostic/i1_rtg1_gift64_runtime_parameterized_r1_8192_seed0/validation.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_parameterized_r1_8192_seed0/summary.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_parameterized_r1_8192_seed0/gate.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_parameterized_r1_8192_seed0/curves.svg
```

The SVG was rendered to 1776 x 896 pixels and passed visual QA: no overlap,
clipping, missing Chinese glyphs, ambiguous zoom, or hidden negative margins.

## R1 Verdict And Redesign Gate

Decision:

```text
innovation1_runtime_spn_r1_seed0_not_supported
```

PRESENT R1, seed1, larger samples, and remote execution are stopped. The
observed failure is not a small margin miss: the correct runtime model is
`0.04584` AUC below E4 and is worse than both controls. Running the 16-pair
PRESENT matrix would spend substantially more local CPU without being able to
restore the required two-cipher gate.

The next experiment is RTG1-R1a, a small GIFT-only architecture calibration.
Its question is whether preserving cell tokens across pairs before global
pooling recovers the typed-cell signal that the current bit-to-pair pooling
lost. Change exactly one variable:

```text
current:    bit processor -> global node pooling -> pair pooling
candidate:  bit processor -> runtime cell-token interaction -> pair pooling
```

The candidate must keep the same external runtime contract, shared parameter
geometry across ciphers, exact GF(2) path, S-box encoder, training labels, GIFT
r6 difference, encrypted-random-plaintext negatives, keys, loss, optimizer,
and checkpoint rule. Freeze the local calibration at:

```text
train      = 2048/class
validation = 1024/class
epochs     = 5
seed       = 0
rows       = current runtime true, cell-token true, cell-token corrupted
```

Advance back to the full `8192/class` four-role gate only if:

```text
cell-token true >= current runtime true + 0.010 AUC
cell-token true - cell-token corrupted >= +0.005 AUC
cell-token true >= 0.520 AUC
```

A failed calibration discards the cell-token change and triggers a data-flow
audit against E4 before another network design. Do not add absolute cipher IDs,
fixed position embeddings, DDT/trail features, extra model families, more
epochs, seed1, PRESENT training, or remote scale during RTG1-R1a.

## Executed R1a Cell-Token Calibration Record

R1a completed locally on 2026-07-23 with the frozen GIFT-64 r6 calibration
protocol:

```text
run_id          = i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0_20260723
train           = 2048/class, 4096 total
validation      = 1024/class, 2048 total
epochs          = 5
seed            = 0
negative mode   = encrypted_random_plaintexts
checkpoint      = best validation AUC
protocol gate   = 9/9 checks passed
```

Results:

| Role | Validation AUC | Best epoch | Parameters |
| --- | ---: | ---: | ---: |
| old runtime correct topology | `0.502413273` | 1 | `163971` |
| cell-token correct topology | `0.501928329` | 1 | `230787` |
| cell-token corrupted topology | `0.509457111` | 1 | `230787` |

Margins:

```text
cell-token true - old runtime true = -0.000484943
cell-token true - corrupted        = -0.007528782
```

All three research checks failed: the candidate did not improve the old
runtime backbone by `0.010`, did not exceed the corrupted-topology control by
`0.005`, and did not reach `0.520` AUC. Decision:

```text
innovation1_runtime_spn_cell_token_calibration_not_supported
```

This is a GIFT-64 seed0 `2048/class` architecture calibration only. It is not
the full R1 gate, multi-seed evidence, formal training, or evidence that runtime
structure parameterization as a method class is impossible. It specifically
rejects this cell-token pooling redesign at the frozen budget. Repeating it at
`8192/class`, running PRESENT or seed1, and remote scale-up remain blocked.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0/results.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0/progress.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0/history.csv
outputs/local_diagnostic/i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0/validation.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0/summary.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0/gate.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0/curves.svg
```

The final SVG was rendered to 1989 x 993 pixels and passed the
`visual-qa-redraw` pixel gate: Chinese text, titles, locally magnified AUC axis,
negative margins, threshold lines, legends, and footer are readable without
overlap or clipping.

## Corrected E4 Data-Flow Audit And Next Action

Source inspection corrected an earlier hypothesis. E4 does not separately
inverse-transform `C0`, `C1`, and `DeltaC`. Its two views are:

```text
current view  = DeltaC
previous view = InvP(DeltaC)
```

E4 then applies separate current/previous cell encoders, typed fusion, a
learned absolute 16-cell position embedding, fixed 16-cell Token-Mixer blocks,
and pair-set pooling. The runtime models already expose the current/inverse
views, but deliberately omit learned absolute cell positions to preserve
variable-width parameter sharing and cell-relabel equivariance. The next
uncertainty is therefore position identifiability rather than separate
processing of the two ciphertexts.

The next gate is RTG1-R1b, a matched E4 position-identifiability audit:

```text
cipher          = GIFT-64 r6
train           = 2048/class
validation      = 1024/class
epochs          = 5
seed            = 0
data protocol   = reuse the exact R1a disk-backed datasets
anchor          = E4 typed-cell with learned 16-cell position embedding
ablation        = the same E4 model with that embedding fixed to zero
```

The single changed variable is whether cell identity is visible through E4's
learned position embedding. Keep all labels, negatives, keys, pairs, optimizer,
loss, checkpoint rule, widths, encoders, Token-Mixer blocks, and heads fixed.
The audit supports a runtime coordinate redesign only if:

```text
E4 position-enabled AUC                    >= 0.520
E4 position-enabled - position-disabled    >= +0.010 AUC
```

If both pass, the next model change is a parameter-shape-independent runtime
coordinate descriptor, such as fixed Fourier features of externally supplied
cell coordinates followed by a shared encoder. If the ablation gap is below
`0.010`, do not add coordinates; audit E4's current/previous typed fusion next.
R1b does not authorize another generic architecture, PRESENT, seed1, more data,
or remote execution.

## Executed R1b E4 Position-Identifiability Record

R1b completed locally on 2026-07-23 with the exact frozen protocol above:

```text
run_id          = i1_rtg1_gift64_e4_position_identifiability_r1b_2048_seed0_20260723
train           = 2048/class, 4096 total
validation      = 1024/class, 2048 total
epochs          = 5
seed            = 0
data            = reused R1a disk-backed train/validation cache
protocol gate   = 9/9 checks passed
result validate = 2/2 rows, no errors
```

The two models retained identical parameter names, shapes, total count, and
trainable count. The zero-position model retains `position_embedding` in its
parameter geometry and optimizer table but multiplies its contribution by zero
in the forward pass. Therefore the one changed variable is whether the learned
16-cell position tensor reaches the E4 Token-Mixer.

Results:

| Role | Validation AUC | Position margin |
| --- | ---: | ---: |
| E4 learned position | `0.527669907` | `-0.000684261` |
| same E4, position fixed to zero | `0.528354168` | reference |

The signal floor passed (`0.527670 >= 0.520`), but the position contribution
gate failed decisively: the learned-position row was slightly worse, not
`+0.010` better. Decision:

```text
innovation1_runtime_spn_position_identity_not_supported
```

This rejects the hypothesis that E4's learned absolute cell positions explain
the runtime-equivariant model's missing signal at this budget. It does not show
that positions can never matter at larger scales, and it is not evidence that
the runtime model has achieved topology attribution. Adding Fourier coordinates
or another runtime coordinate encoder is therefore blocked by current evidence.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_gift64_e4_position_identifiability_r1b_2048_seed0/results.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_e4_position_identifiability_r1b_2048_seed0/progress.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_e4_position_identifiability_r1b_2048_seed0/history.csv
outputs/local_diagnostic/i1_rtg1_gift64_e4_position_identifiability_r1b_2048_seed0/validation.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_position_identifiability_r1b_2048_seed0/summary.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_position_identifiability_r1b_2048_seed0/gate.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_position_identifiability_r1b_2048_seed0/curves.svg
```

The final SVG passed `visual-qa-redraw` after rendering to 2012 x 967 pixels.
The Chinese title and conclusion, local AUC zoom, negative contribution,
threshold line, and claim boundary are readable without overlap or clipping.

## R1c Executable Next Action

R1c asks whether E4's separate encoders for `DeltaC` and `InvP(DeltaC)` are
responsible for the signal lost by the runtime model. Use the same GIFT-64 r6
cache and budget:

```text
train           = 2048/class
validation      = 1024/class
epochs          = 5
seed            = 0
anchor          = E4 zero-position with separate current/previous encoders
ablation        = E4 zero-position with one shared encoder for both views
```

Absolute position remains fixed to zero in both rows so R1c changes exactly one
factor. Preserve typed fusion, Token-Mixer, pooling, labels, keys, negatives,
optimizer, checkpoint selection, parameter dimensions, and data. The shared
control must retain the second encoder and all of its parameters in the
`state_dict` and optimizer but bypass it in the forward pass. This keeps both
rows exactly parameter-matched while testing only whether distinct current and
previous view functions are needed.

The initial screening gate is:

```text
separate-encoder AUC                       >= 0.520
separate-encoder - shared-encoder AUC      >= +0.010
```

If the gap passes, the runtime redesign should preserve typed view identity
with a shared parameterized encoder plus an external two-value view-role token,
which keeps parameter shapes independent of cipher and block width. If it does
not pass, audit the fixed 16-cell Token-Mixer versus a permutation-equivariant
cell mixer next. Do not run PRESENT, seed1, larger data, or remote training
before this representation mismatch is localized.

## Executed R1c E4 View-Encoder Record

R1c completed locally on 2026-07-24:

```text
run_id          = i1_rtg1_gift64_e4_view_encoder_r1c_2048_seed0_20260724
train           = 2048/class, 4096 total
validation      = 1024/class, 2048 total
epochs          = 5
seed            = 0
data            = reused R1a disk-backed train/validation cache
position        = fixed to zero in both rows
protocol gate   = 9/9 checks passed
result validate = 2/2 rows, no errors
```

Both rows retain identical parameter names, shapes, total count, and trainable
count. The shared-view control keeps the unused previous-view encoder in its
`state_dict` and optimizer, so capacity geometry is not a confound.

Results:

| Role | Validation AUC | Separate-view margin |
| --- | ---: | ---: |
| zero-position E4, separate view encoders | `0.528354168` | `-0.007276058` |
| zero-position E4, shared view encoder | `0.535630226` | reference |

The separate-view anchor cleared `0.520`, but it was worse than the shared
control rather than `+0.010` better. Decision:

```text
innovation1_runtime_spn_typed_view_identity_not_supported
```

The result rejects a second plausible explanation for the runtime model's
failure: E4 does not need separate current and inverse-layer cell encoders to
retain this small-budget GIFT signal. A runtime view-role token is not justified
by the observed evidence. This remains a one-cipher, one-seed, `2048/class`
architecture audit, not topology-attribution or formal evidence.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_gift64_e4_view_encoder_r1c_2048_seed0/results.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_e4_view_encoder_r1c_2048_seed0/progress.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_e4_view_encoder_r1c_2048_seed0/history.csv
outputs/local_diagnostic/i1_rtg1_gift64_e4_view_encoder_r1c_2048_seed0/validation.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_view_encoder_r1c_2048_seed0/summary.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_view_encoder_r1c_2048_seed0/gate.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_view_encoder_r1c_2048_seed0/curves.svg
```

The final SVG passed `visual-qa-redraw` after pixel rendering. The Chinese
title, local AUC scale, negative margin, threshold, and evidence boundary are
readable without overlap or clipping.

## Current RTG1 Verdict And R1d Recommendation

The runtime contract itself remains implemented and verified across PRESENT,
GIFT, SKINNY, permutation layers, general GF(2) layers, 64/128-bit widths, and
variable pair counts. Empirical topology attribution has not passed:

```text
R1  runtime true vs E4 / corrupted / independent       failed
R1a runtime cell-token redesign                         failed
R1b E4 absolute-position contribution                   not supported
R1c E4 separate-view-encoder contribution               not supported
```

The strongest small-budget E4 variant is now the simpler zero-position,
shared-view model at `0.535630` AUC. The next bounded question is whether E4's
fixed 16-cell Token-Mixer, which learns an explicit `16 -> hidden -> 16` map,
retains signal that a permutation-equivariant cell mixer loses.

R1d should compare, on the same frozen GIFT cache and `2048/class`, exactly:

```text
anchor    = zero-position, shared-view E4 with fixed 16-cell Token-Mixer
control   = same frontend/head with a permutation-equivariant cell mixer
```

Use a parameter-matched or parameter-budget-matched control and verify cell
permutation equivariance directly. Keep all data and training fields fixed.
Use the following three-way gate:

```text
fixed mixer signal floor                  >= 0.520
fixed mixer - equivariant mixer           >= +0.010
equivariant mixer signal floor            >= 0.520
```

If the fixed row passes both of its gates, strict cell-relabel equivariance
removes useful interaction; the runtime design then needs externally supplied
functional coordinates or graph positional encodings rather than absolute
cipher IDs. If the fixed advantage fails but the equivariant row reaches
`0.520`, promote the equivariant E4-style frontend itself: port its per-pair
`DeltaC`/exact-inverse cell tokenization, shared view encoder, equivariant mixer,
activity pooling, and pair-set pooling into the runtime model, then rerun
correct/corrupted/no-topology attribution. Only if neither row reaches the
signal floor should E4 component peeling stop in favor of a from-first-principles
runtime message-passing redesign.

PRESENT, seed1, larger samples, and remote GPU scale remain blocked until a
small local candidate beats both corrupted-topology and no-topology controls.

## Executed R1d Cell-Mixer Record

R1d completed locally on 2026-07-24 using the exact R1a/R1b/R1c GIFT-64 r6
cache and training protocol:

```text
run_id          = i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed0_20260724
train           = 2048/class, 4096 total
validation      = 1024/class, 2048 total
epochs          = 5
seed            = 0
protocol gate   = 9/9 checks passed
result validate = 2/2 rows, no errors
```

The fixed and equivariant mixers have exactly equal parameter counts. A direct
numerical test also verifies that the equivariant mixer commutes with arbitrary
cell relabeling and that every one of its parameters participates in forward
computation.

| Role | Validation AUC | Fixed-minus-equivariant |
| --- | ---: | ---: |
| fixed 16-cell Token-Mixer | `0.535630226` | `-0.005233765` |
| cell-relabel-equivariant mixer | `0.540863991` | reference |

The fixed-mixer dependency gate failed while the equivariant signal floor
passed. Decision:

```text
innovation1_runtime_spn_equivariant_e4_backbone_supported
```

This is positive architecture evidence: strict cell-relabel equivariance does
not explain the signal missing from the old runtime backbone. It is not yet
topology attribution because R1d did not compare correct, corrupted, and absent
runtime linear layers.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed0/results.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed0/progress.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed0/history.csv
outputs/local_diagnostic/i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed0/validation.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed0/summary.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed0/gate.json
outputs/local_diagnostic/i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed0/curves.svg
```

The R1d SVG was rendered to pixels and passed the visual QA gate without text
overlap, clipping, missing glyphs, ambiguous scale, or incomplete labels.

## R2a Runtime E4-Equivariant Attribution Plan

Research question: can the runtime model preserve the R1d E4-equivariant signal
while attributing it specifically to the externally supplied linear topology?

R2a ports only the supported R1d data flow into the runtime contract:

```text
per pair DeltaC
  -> exact inverse runtime GF(2) view
  -> runtime cell ordering
  -> shared cell encoder and external S-box context
  -> cell-relabel-equivariant mixer
  -> mean/max/activity pooling
  -> pair-set attention/mean/max
  -> classifier
```

The same runtime class and parameter geometry must support GIFT permutation
layers, SKINNY-style general invertible GF(2) layers, and synthetic 128-bit SPNs.
The calibration changes only the supplied linear relation:

```text
true        = exact external GIFT linear topology
corrupted   = deterministic degree-preserving corruption of that topology
independent = no linear topology; inverse view equals current DeltaC
```

The independent control still receives exactly the same cell partition and
S-box truth table. It removes only the linear topology, avoiding the old
confound where all structure metadata was zeroed.

Freeze R2a at:

```text
cipher      = GIFT-64 r6
train       = 2048/class, 4096 total
validation  = 1024/class, 2048 total
epochs      = 5
seed        = 0
pairs       = 4
negative    = encrypted random plaintexts
cache       = reuse outputs/local_cache/i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0
rows        = runtime E4 true / corrupted / independent
```

Advance only if every gate passes:

```text
runtime true AUC                         >= 0.520
runtime true - R1d equivariant anchor    >= -0.005
runtime true - corrupted                 >= +0.005
runtime true - independent               >= +0.005
```

If R2a passes, the next rung is the same three-control GIFT seed0 experiment at
`8192/class`, 10 epochs. If it fails, redesign the runtime linear-topology
interaction locally. PRESENT, seed1, remote execution, additional model
families, larger samples, DDT/trail features, cipher IDs, and stable-topology
claims remain blocked.

## R2a First Execution: Protocol Invalid

The first R2a execution completed all three training rows, but the post-result
source audit found a coordinate-system mismatch before any scale decision:

```text
project ciphertext_pair_bits = MSB-first within each 64-bit block
runtime GF(2) matrices        = integer LSB bit coordinates
old adapter behavior          = applied the LSB matrix directly to MSB features
```

A deterministic one-hot comparison proved that the old runtime `true` inverse
view was not equal to E4's verified GIFT inverse-P view. Its raw metrics were:

```text
true        = 0.530977726
corrupted   = 0.533804417
independent = 0.473210335
```

These numbers are retained only as protocol-invalid diagnostic history. They do
not show that correct topology is worse than corrupted topology because the
purported correct topology was expressed in the wrong bit coordinates.

The repair changes only the fixed protocol adapter: reshape the project input
into ciphertext blocks and reverse each block from MSB-first to the runtime
structure's LSB coordinates before calling the unchanged runtime backbone. The
result row must record:

```text
input_bit_order = project_msb_to_runtime_lsb
```

The repaired R2a repeats exactly the same config, cache, initialization seed,
budget, and three controls in a new result directory. Its gates and blocked
actions are unchanged. A regression test must first prove that the repaired
runtime GIFT inverse view exactly equals the existing E4 inverse-P gather for
all 64 one-hot inputs.

## Executed Bit-Order-Repaired R2a Record

The repaired R2a completed locally with all ten protocol checks and all three
generic result rows valid:

```text
run_id          = i1_rtg1_gift64_runtime_e4_equivariant_r2a_bitorderfix_2048_seed0_20260724
train           = 2048/class, 4096 total
validation      = 1024/class, 2048 total
epochs          = 5
seed            = 0
runtime params  = 442466 for every control
input bit order = project_msb_to_runtime_lsb
```

Results:

| Role | Validation AUC | True margin |
| --- | ---: | ---: |
| runtime correct topology | `0.534461021` | reference |
| runtime corrupted topology | `0.522696018` | `+0.011765003` |
| runtime no-linear-topology | `0.496503353` | `+0.037957668` |
| R1d equivariant anchor | `0.540863991` | `-0.006402969` |

The signal floor and both topology-attribution gates passed. The anchor
preservation gate missed by only `0.001402969` beyond its allowed loss, so the
overall decision remains:

```text
innovation1_runtime_spn_r2a_topology_attribution_not_supported
```

This is the first protocol-valid positive topology ordering for the runtime
model, but it cannot yet authorize scale because the model was not capacity
matched to R1d: R2a used a 128-dimensional pair embedding and `442466`
parameters, while R1d used a 256-dimensional pair embedding and `733154`
parameters.

## R2b Same-Budget Pair-Embedding Gate

R2b changes only `pair_embedding_dim: 128 -> 256`. The resulting runtime model
has `738786` parameters, within one percent of the R1d anchor; the small excess
is the externally supplied S-box encoder required by the runtime contract.
Reuse the same GIFT cache, data, labels, keys, controls, seed, optimizer, loss,
checkpoint rule, and five-epoch budget.

Run exactly three rows:

```text
runtime E4 true topology, pair_dim=256
runtime E4 corrupted topology, pair_dim=256
runtime E4 no linear topology, pair_dim=256
```

R2b uses the unchanged R2a gates:

```text
runtime true AUC                         >= 0.520
runtime true - R1d equivariant anchor    >= -0.005
runtime true - corrupted                 >= +0.005
runtime true - independent               >= +0.005
```

A full pass authorizes only the same GIFT seed0 `8192/class`, 10-epoch local
gate. A failure stops capacity adjustment and returns to local architectural
analysis. PRESENT, seed1, remote scale, and stable-topology claims remain
blocked.

## Executed R2b Record And Control Audit

R2b completed with valid rows and equal `738786`-parameter controls:

| Role | Validation AUC | True margin |
| --- | ---: | ---: |
| runtime correct topology | `0.524087429` | reference |
| runtime corrupted topology | `0.524861336` | `-0.000773907` |
| runtime no-linear-topology | `0.513041496` | `+0.011045933` |
| R1d equivariant anchor | `0.540863991` | `-0.016776562` |

Pair-embedding capacity did not recover the anchor and removed the earlier
true-over-corrupted margin. The capacity hypothesis is rejected; do not widen
the model again.

The post-result control audit then found that `RuntimeSpnStructure.corrupted()`
was not a genuine shuffled-topology control. It composed each linear matrix
with a one-cell cyclic source shift while preserving the bit role of every
source. For GIFT this retains nibble coherence and a highly regular alternative
permutation, unlike the established E4 shuffled-P control that destroys both
cell and bit-role alignment.

## R2c Full-Bit Corruption Repair

R2c changes only the deterministic corrupted-topology constructor. For each
runtime structure, apply a fixed-seed full-bit source-column permutation to its
own GF(2) matrix. This preserves invertibility, every row degree, and the sorted
column-degree multiset for permutation and general GF(2) layers, while breaking
cell and bit-role alignment. It does not replace a held-out cipher topology
with a train-seen family.

Before training, tests must verify:

```text
same structure + same seed -> identical corrupted matrix
true matrix != corrupted matrix
row degrees unchanged
sorted column degrees unchanged
GIFT source cell and bit-role alignment substantially broken
```

R2c returns to the smaller R2a `pair_embedding_dim=128`, because R2b rejected
the capacity increase. Reuse the same three rows, cache, data, keys, seed,
epochs, and gates. A pass authorizes only the `8192/class`, 10-epoch GIFT seed0
gate. A failure blocks mechanical scaling and requires a new topology
interaction architecture. PRESENT, seed1, remote execution, and stable claims
remain blocked.

## Executed R2c Record And Recommendation

R2c completed with all generic rows valid and all ten protocol checks passing.
The unchanged true and independent rows reproduced R2a exactly to the stored
float, while only the repaired corrupted control changed:

| Role | Validation AUC | True margin |
| --- | ---: | ---: |
| runtime correct topology | `0.534461021` | reference |
| full-bit corrupted topology | `0.493224144` | `+0.041236877` |
| runtime no-linear-topology | `0.496503353` | `+0.037957668` |
| R1d equivariant anchor | `0.540863991` | `-0.006402969` |

The correct topology clears the `0.520` signal floor and both `+0.005`
attribution margins by a wide margin. The overall status remains `hold` because
the anchor-preservation margin still misses its `-0.005` tolerance by
`0.001402969`:

```text
decision = innovation1_runtime_spn_r2a_topology_attribution_not_supported
remote_scale = no
```

R2c is therefore strong single-cipher, single-seed, `2048/class` positive
topology-attribution evidence, not stable or formal evidence. It does not
authorize GIFT `8192/class`, seed1, PRESENT, or remote execution.

The next one-variable local audit should preserve the R2c architecture and
controls while reducing only the additive external S-box context strength. In
GIFT all cells share one S-box, so the current full-scale descriptor is a
cell-constant residual absent from R1d and is the smallest remaining explanation
for the `0.006403` anchor gap. Compare scale `1.0` against a preregistered small
nonzero scale, retain S-box sensitivity, and rerun the full R2c controls only if
the true-topology calibration enters the anchor tolerance. Do not widen the
pair embedding again or relax the gate after observing results.

## Preregistered R2d External S-Box Context Calibration

Research question: is the small R2c-to-R1d anchor gap caused by adding the
full-scale, cell-constant external S-box descriptor before the equivariant
E4 mixer?

R2d changes exactly one forward-pass constant:

```text
R2c S-box context scale = 1.0
R2d candidate scale     = 0.1
```

The candidate remains nonzero so the model still consumes the externally
supplied S-box truth table. `sbox_context_scale` is not a parameter or buffer;
it must not change any trainable parameter name or shape. A deterministic test
must show that changing only the external S-box truth table still changes the
candidate logits at scale `0.1`.

Freeze all other fields to the valid R2c true-topology row:

```text
cipher / rounds      = GIFT-64 / 6
train / validation   = 2048/class / 1024/class
epochs / seed        = 5 / 0
pairs per sample     = 4
negative definition = encrypted random plaintexts
keys                 = train 0x00..00, validation 0x11..11
model                = runtime E4 equivariant true topology
pair embedding       = 128
processor steps      = 2
cache                = reuse i1_rtg1_gift64_runtime_cell_token_r1a_2048_seed0
```

Train only the scale-`0.1` candidate. The scale-`1.0` baseline is the exactly
reproduced R2c true row, so retraining it would add no information. Advance only
if both checks pass:

```text
candidate true AUC >= 0.520
candidate true - R1d equivariant anchor >= -0.005
```

A pass authorizes only a same-budget `2048/class`, seed0 rerun of the complete
true/full-bit-corrupted/no-linear-topology matrix with scale `0.1`. A miss ends
S-box-strength tuning. R2d never authorizes PRESENT, seed1, `8192/class`, remote
execution, or a stable-topology claim by itself. Do not try additional scales,
widen pair embeddings, increase epochs, or relax thresholds after observing
the result.

## Executed R2d Record And Decision

R2d completed locally with its preregistered single candidate. The result row
matched the CSV plan, reused the R2c disk cache, retained the repaired runtime
bit-order adapter, and kept exactly `442466` trainable parameters:

```text
run_id          = i1_rtg1_gift64_runtime_e4_sbox_scale_r2d_2048_seed0_20260724
candidate AUC   = 0.533868790
R2c scale 1.0  = 0.534461021
R1d anchor      = 0.540863991
candidate - R2c = -0.000592232
candidate - R1d = -0.006995201
```

The candidate remained above the `0.520` signal floor but missed the R1d
anchor tolerance by `0.001995201`. All eleven protocol checks passed; the only
failed research check was anchor preservation. The decision is therefore:

```text
decision     = innovation1_runtime_spn_sbox_scale_calibration_not_supported
status       = hold
remote_scale = no
```

Reducing the early additive S-box residual did not recover the E4 anchor and
slightly reduced AUC. Do not try more scalar values and do not run the three
controls at scale `0.1`.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_sbox_scale_r2d_2048_seed0/results.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_sbox_scale_r2d_2048_seed0/progress.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_sbox_scale_r2d_2048_seed0/validation.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_sbox_scale_r2d_2048_seed0/gate.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_sbox_scale_r2d_2048_seed0/history.csv
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_sbox_scale_r2d_2048_seed0/curves.svg
```

The next candidate should test placement rather than another strength. R2e
will preserve the R2c current/inverse-linear cell path through cell pooling and
move the same external S-box descriptor to a late pair-level conditioning
branch. The question is whether keeping cipher metadata out of the topology
extractor preserves the R1d signal while still changing logits when the
external S-box table changes.

R2e must change only the S-box injection location, keep a nonzero fixed
conditioning coefficient, and reuse R2c's data, seed, budget, optimizer, loss,
checkpoint, bit order, and pair dimension. First train only the true-topology
candidate at `2048/class`, seed0, five epochs. Require:

```text
candidate AUC >= 0.520
candidate - R1d anchor >= -0.005
external S-box table changes logits
trainable parameter names and shapes are structure-independent
```

Only a pass may open the full true/full-bit-corrupted/no-topology matrix at the
same budget. R2e must not add PRESENT, seed1, larger data, remote execution, or
new scalar searches. A miss sends the route to a topology/S-box fusion design
review rather than another training run.

## Preregistered R2e Late S-Box Conditioning Calibration

R2e freezes the conditioning coefficient at `1.0` and changes only its
location:

```text
R2c = add encoded S-box context to every cell before the equivariant mixer
R2e = keep the mixer input S-box-free and add the same context after pair pooling
```

For the frozen `hidden_dim=64`, `token_dim=128`, and
`pair_embedding_dim=128`, the late context requires no trainable projection.
For other legal dimensions, a deterministic adaptive average projection keeps
the parameter geometry independent of width and structure. Early and late
modes must have identical trainable parameter names and shapes.

The implementation gate must prove that two structures differing only in their
external S-box truth table produce identical inputs to the first topology
mixer in `late_pair` mode but different final logits. This directly verifies
that R2e preserves the topology extractor while still consuming the S-box.

Train exactly one row:

```text
run id          = i1_rtg1_gift64_runtime_e4_sbox_location_r2e_2048_seed0
cipher / rounds = GIFT-64 / 6
topology         = correct runtime topology
S-box mode       = late_pair
S-box scale      = 1.0
train / val      = 2048/class / 1024/class
epochs / seed    = 5 / 0
pairs            = 4
cache            = reuse the exact R2c disk cache
```

Advance only if:

```text
candidate AUC >= 0.520
candidate - R1d anchor >= -0.005
all source, data, training, bit-order, cache, and parameter-geometry checks pass
```

A pass opens only the same-budget GIFT seed0 true/full-bit-corrupted/no-linear
matrix with frozen `late_pair` conditioning. A miss stops S-box placement
experiments. PRESENT, seed1, `8192/class`, remote execution, and stable claims
remain blocked until the full matrix passes.

## Executed R2e Record And R2f Attribution Plan

R2e completed with all twelve protocol checks and both research checks passing:

```text
run_id             = i1_rtg1_gift64_runtime_e4_sbox_location_r2e_2048_seed0_20260724
late-pair AUC      = 0.537684441
R2c early-add AUC = 0.534461021
R1d anchor AUC     = 0.540863991
late - early       = +0.003223419
late - R1d         = -0.003179550
decision           = innovation1_runtime_spn_sbox_location_calibration_supported
```

Moving S-box conditioning after topology extraction recovered the frozen R1d
anchor tolerance without changing the `442466` parameter geometry. This is a
supported architecture calibration, not topology attribution by itself.

R2f now changes no architecture or benchmark field. It trains exactly three
rows at the same GIFT-64 r6, `2048/class`, seed0, five-epoch budget:

```text
late_pair + correct runtime topology
late_pair + deterministic full-bit corrupted topology
late_pair + no linear topology, retaining cell and S-box metadata
```

All rows must use identical data, keys, optimizer, loss, checkpoint rule,
parameter geometry, and `project_msb_to_runtime_lsb` input conversion. Advance
only if all four research checks pass:

```text
true AUC >= 0.520
true - R1d anchor >= -0.005
true - corrupted >= +0.005
true - independent >= +0.005
```

A pass authorizes a matching R1d seed1 anchor followed by the same frozen
three-control seed1 matrix at `2048/class`. It does not authorize larger data,
PRESENT, remote execution, or stable claims. A miss closes late S-box
conditioning and returns to a non-training fusion audit.

## Executed R2f Seed0 Record And Seed1 Replication Plan

R2f completed locally on 2026-07-24. The three planned rows, plan alignment,
disk-backed data protocol, equal parameter geometry, frozen `late_pair` mode,
and all twelve protocol checks passed:

| Role | Validation AUC | Correct-minus-control |
| --- | ---: | ---: |
| correct runtime topology | `0.537684441` | reference |
| deterministic full-bit corrupted topology | `0.519806862` | `+0.017877579` |
| no linear topology | `0.467815876` | `+0.069868565` |
| R1d equivariant anchor | `0.540863991` | `-0.003179550` |

The signal floor, R1d tolerance, corrupted-topology margin, and no-topology
margin all passed. Decision:

```text
innovation1_runtime_spn_late_attribution_seed0_supported
```

This is the first valid late-conditioned seed0 attribution result: the model
benefits from the correct externally supplied GIFT linear topology, rather
than merely from cell partition or S-box metadata. It remains a single-seed,
single-cipher, `2048/class` local diagnostic. It is not stable, cross-cipher,
formal, paper-scale, or breakthrough evidence.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0/results.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0/progress.jsonl
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0/validation.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0/summary.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0/gate.json
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0/history.csv
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0/curves.svg
```

The next research question is whether the attribution survives a new random
seed without changing architecture, data definition, sample budget, optimizer,
loss, checkpoint selection, or gates. The only experimental variable is
`seed0 -> seed1`.

First run the exact two-row R1d backbone calibration at seed1:

```text
plan       = innovation1_spn_gift64_e4_cell_mixer_r1d_2048_seed1.csv
train/val  = 2048/class / 1024/class
epochs     = 5
pairs      = 4
execution  = local CPU diagnostic with a separate seed1 disk cache
advance    = equivariant R1d seed1 AUC >= 0.520 and all protocol checks pass
stop       = no R2f seed1 matrix if the equivariant anchor misses 0.520
```

If the anchor passes, run exactly three R2f rows at the same seed1 budget:

```text
correct runtime topology
deterministic full-bit corrupted topology
no linear topology, retaining cell and S-box metadata
```

Use the seed1 R1d equivariant row as the same-seed anchor. Advance only if:

```text
true AUC >= 0.520
true - seed1 R1d anchor >= -0.005
true - corrupted >= +0.005
true - independent >= +0.005
```

A full seed1 pass authorizes only a same-budget second-SPN transfer gate. The
preferred next cipher is PRESENT because its one-to-one P layer directly tests
the other required linear-topology family while preserving the runtime model's
trainable parameter names and shapes. Larger samples, more epochs, remote GPU
execution, and stable cross-cipher claims remain blocked until that transfer
passes. A seed1 miss stops replication and returns to a non-training audit of
seed sensitivity and topology/S-box fusion; it must not be repaired by tuning
on seed1.

## Executed Seed1 Replication Record

The seed1 R1d anchor completed first and passed all protocol checks:

```text
fixed cell-mixer AUC       = 0.538514614
equivariant E4 anchor AUC  = 0.553535461
fixed - equivariant        = -0.015020847
decision                   = innovation1_runtime_spn_equivariant_e4_seed1_anchor_supported
```

The frozen R2f seed1 three-control matrix then completed with plan alignment
and all twelve protocol checks passing:

| Role | Validation AUC | Correct-minus-control |
| --- | ---: | ---: |
| correct runtime topology | `0.546117783` | reference |
| deterministic full-bit corrupted topology | `0.503655910` | `+0.042461872` |
| no linear topology | `0.516066074` | `+0.030051708` |
| seed1 R1d equivariant anchor | `0.553535461` | `-0.007417679` |

The true topology again exceeded both controls by far more than `+0.005`, so
the topology-attribution ordering replicated across seed0 and seed1. The true
row also remained above the `0.520` signal floor. However, it missed the
same-seed R1d anchor tolerance by:

```text
required true - anchor >= -0.005000000
observed true - anchor  = -0.007417679
excess loss             =  0.002417679
```

Decision:

```text
innovation1_runtime_spn_late_attribution_seed1_anchor_tolerance_not_met
status = hold
```

Claim boundary: GIFT-64 now has two-seed local diagnostic evidence that the
correct external topology is better than deterministic corrupted topology and
no topology. The full candidate still has not passed the preregistered
same-seed backbone-preservation gate, and no second cipher has been tested.
This is not formal, paper-scale, stable cross-cipher, or breakthrough evidence.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_gift64_e4_cell_mixer_r1d_2048_seed1/
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed1/
```

Do not run PRESENT, increase data, add epochs, tune the anchor tolerance, or
change S-box scale after seeing seed1. The next action is a non-training
semantic-equivalence audit between the R1d equivariant anchor and runtime E4
true-topology path. It must compare, on identical synthetic inputs and copied
weights:

```text
project MSB bits -> runtime LSB coordinates
R1d inverse-permutation view -> runtime exact GF(2) inverse view
cell ordering and within-cell bit ordering
shared current/previous encoder and typed fusion
equivariant mixer, mean/max/activity pooling, and pair pooling
late S-box constant injection
```

The audit must first zero the late S-box branch and use the R1d-matched
`pair_embedding_dim=256`. If copied shared weights then yield identical tokens,
pair embeddings, and logits, the representation is semantically aligned and
the remaining gap is an optimization/capacity interaction; the next candidate
must be justified without tuning on seed1. If equivalence fails, repair only
the first divergent deterministic stage and rerun the same two-seed gate. This
audit, not more training, is what decides whether PRESENT transfer can reopen.

## Executed A1 Semantic-Equivalence Audit And Frozen R2g Repair Gate

The non-training A1 audit first confirmed exact MSB/LSB conversion and exact
GIFT inverse-P behavior, then found its first deterministic mismatch at
`current_delta_cells`: runtime cells used LSB-first bit roles while the R1d
anchor's shared cell encoder received project MSB-first roles. Only this first
divergent representation stage was repaired. The runtime cell factory now
orders roles consistently with the project input, and the runtime E4 forward
uses the same flattened cell-encoder evaluation order as R1d.

After the repair, the audit copied all shared R1d weights into runtime E4,
zeroed the late S-box encoder, and compared fourteen stages in float64. Every
shape matched and every maximum absolute error was below the frozen `1e-6`
tolerance:

```text
maximum stage error = 3.552713678800501e-15
final-logit error   = 2.636779683484747e-16
decision            = innovation1_runtime_spn_deterministic_semantics_equivalent
```

Artifact root:

```text
outputs/local_audits/i1_rtg1_gift64_r1d_runtime_e4_semantic_equivalence_a1_20260724/
```

The result is a deterministic same-weight equivalence proof, not training,
AUC, cross-cipher, scale, or SOTA evidence. Because the repair changes the
runtime training input role order, the pre-repair R2f metrics cannot adjudicate
the repaired model. PRESENT transfer therefore remains blocked pending R2g.

R2g changes exactly one variable relative to R2f: corrected within-cell bit
role ordering. It reuses the existing seed-specific disk-backed datasets and
R1d anchors. Run locally, separately for seed0 and seed1:

```text
models      = correct topology / deterministic full-bit corrupted / no topology
train/val   = 2048/class / 1024/class
pairs       = 4
epochs      = 5
seeds       = 0, 1
negative    = encrypted random plaintexts
checkpoint  = best validation AUC
```

For each seed, all four frozen gates remain unchanged:

```text
correct AUC >= 0.520
correct - same-seed R1d anchor >= -0.005
correct - corrupted >= +0.005
correct - no topology >= +0.005
```

Advance only if both seeds pass. A two-seed pass authorizes one same-budget
PRESENT transfer with the same three controls. Any seed miss keeps the route
on hold and requires a new hypothesis about late S-box/capacity/optimization
interaction; do not increase samples, epochs, relax gates, tune on seed1, or
launch remote training.

## Executed R2g Record And PRESENT T1 Transfer Plan

Both repaired R2g seed gates completed locally with plan-aligned disk-backed
datasets and all protocol checks passing:

| Seed | Correct AUC | Corrupted AUC | No-topology AUC | Correct - R1d |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `0.538176537` | `0.486937046` | `0.514234066` | `-0.002687454` |
| 1 | `0.548832417` | `0.509042740` | `0.504164219` | `-0.004703045` |

Correct-minus-corrupted was `+0.051239491` and `+0.039789677`; correct-minus-
no-topology was `+0.023942471` and `+0.044668198`. Both seeds therefore passed
the signal floor, same-seed R1d tolerance, corrupted-topology margin, and
no-topology margin. These remain two local `2048/class` GIFT diagnostics, not
formal or paper-scale evidence.

T1 now asks whether the unchanged runtime E4 parameter geometry transfers to
PRESENT when only the externally supplied cipher description and dataset are
changed. It uses PRESENT-80 r7 because it is the established local PRESENT
signal-bearing round, not to claim a new round record. The three equal-parameter
rows are correct PRESENT topology, deterministic full-bit corrupted topology,
and no linear topology.

```text
train/val   = 2048/class / 1024/class
pairs       = 16
epochs      = 5
seed        = 0
input       = raw ciphertext-pair bits
sample      = Zhang/Wang Case2 official MCND organization
negative    = encrypted random plaintexts
execution   = local CPU diagnostic with disk-backed cache
```

Advance to a seed1 PRESENT replication only if:

```text
correct AUC >= 0.520
correct - corrupted >= +0.005
correct - no topology >= +0.005
all protocol and equal-geometry checks pass
```

A miss blocks PRESENT replication and all scale-up. Do not switch features,
rounds, pair count, loss, epochs, negatives, or topology corruption after
seeing T1 seed0. A pass authorizes only the identical seed1 local replication,
not remote training or a stable cross-cipher claim.

## Executed PRESENT T1 Two-Seed Record

Both PRESENT T1 seeds completed locally. Plan validation returned `3/3` rows
for each seed, all equal-geometry and protocol checks passed, and the frozen
signal/control gates passed:

| Seed | Correct AUC | Corrupted AUC | No-topology AUC | Correct - corrupted | Correct - no topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.664596081` | `0.570662022` | `0.554435253` | `+0.093934059` | `+0.110160828` |
| 1 | `0.676282406` | `0.561504364` | `0.571587086` | `+0.114778042` | `+0.104695320` |

Decisions:

```text
innovation1_runtime_spn_present_transfer_seed0_supported
innovation1_runtime_spn_present_transfer_seed1_supported
```

Together with repaired GIFT R2g, the current evidence supports this limited
claim: one unchanged runtime E4 parameter geometry produced `correct topology
> corrupted topology` and `correct topology > no topology` on two real
permutation-layer SPNs, with two local seeds per cipher. The network consumes
external cell roles, S-box truth descriptors, and linear maps; it does not use
cipher IDs, keys, DDTs, trails, beam scores, or label-derived features.

This is still local diagnostic evidence. GIFT used r6, 4 pairs and independent
pairs; PRESENT used r7, 16 pairs and Case2 MCND organization. T1 therefore
demonstrates runtime architectural adaptation and topology attribution, not
same-task zero-shot weight transfer, formal scale, a published-protocol
reproduction, SOTA, or universal SPN performance.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_present_runtime_e4_transfer_t1_2048_seed0/
outputs/local_diagnostic/i1_rtg1_present_runtime_e4_transfer_t1_2048_seed1/
```

## Recommended Next Action: T2-A General-GF(2) Data Readiness

The model contract already accepts and tests SKINNY-64's sparse general GF(2)
linear layer, but the Innovation 1 standard cipher factory and differential
dataset path do not yet expose keyed SKINNY-64/64. That missing data adapter,
not neural capacity, is the next concrete blocker.

T2-A is a non-training local readiness task:

```text
question       = can keyed SKINNY-64/64 enter the standard strict differential dataset without semantic drift?
one variable   = add only the SKINNY cipher/data adapter
anchor         = existing standalone SKINNY implementation plus official 32-round public vector
controls       = exact encrypt replay, deterministic seed replay, encrypted-random-plaintext negatives
data gate      = 64/class train and 32/class validation disk-cache fixture
model gate     = true/corrupted/no-topology runtime E4 parameter names and shapes identical
linear gate    = sparse general GF(2) inverse and forward replay exact
execution      = local CPU readiness; no neural training
```

Advance only if the public vector, key schedule, MSB/LSB contract, strict
negative semantics, cache/reuse metadata, deterministic replay, parameter
geometry, and exact GF(2) checks all pass. Then select one literature-backed,
signal-bearing SKINNY round/difference protocol before preregistering a local
`2048/class`, two-seed, three-control T2 training gate. If readiness fails,
repair only the adapter mismatch. Do not guess a difference, start neural
training, reuse Innovation 2 balance labels, scale PRESENT/GIFT, or launch a
remote run before T2-A passes.

### Frozen T2-A Execution Record

T2-A is preregistered as the following local, non-training readiness run:

```text
run_id          = i1_rtg1_skinny64_general_gf2_data_readiness_t2a_20260724
cipher           = SKINNY-64/64, TK1 only
rounds           = 7 (data-path fixture only; not a signal claim)
input difference = 0x0000000000000040 (adapter fixture only)
train fixture    = 64/class, seed0, key 0x0000000000000000
validation       = 32/class, seed1, key 0x1111111111111111
pairs/sample     = 4 independent ciphertext pairs
negative         = encrypted random plaintexts
feature          = raw ciphertext-pair bits
cache            = disk-backed features.npy / labels.npy / metadata.json
training         = none
execution        = local CPU
```

The input difference is deliberately only a data-contract fixture. T2-A must
not use its result to claim that SKINNY r7 is distinguishable. A later training
plan needs a separately verified literature source or a preregistered
same-budget difference/round screen.

T2-A passes only if all of these categories pass:

```text
cipher factory:
  Appendix-B 32-round vector and direct Skinny64/factory replays are exact

strict dataset:
  positive encryption calls preserve the frozen input XOR difference
  negative rows invoke two real plaintext encryptions per pair
  train/validation shapes, labels, keys, seeds and metadata are exact

cache/replay:
  disk arrays equal fresh in-memory generation bit-for-bit
  parameter-matched second load reports cache_status=reused

runtime model:
  true/corrupted/no-topology models have identical parameter names/shapes/counts
  all three accept the SKINNY fixture and produce finite logits
  runtime structure and topology tensors remain outside state_dict

general GF(2):
  the runtime matrix exactly replays SKINNY ShiftRows+MixColumns
  matrix/inverse products equal identity and round-trip random bit states exactly
  at least one output bit depends on multiple input bits
  deterministic corrupted topology changes edges without changing parameter geometry
```

Expected artifacts:

```text
outputs/local_readiness/i1_rtg1_skinny64_general_gf2_data_readiness_t2a_20260724/
  results.jsonl
  progress.jsonl
  metadata.json
  summary.json
  gate.json
  curves.svg
  cache/train/{features.npy,labels.npy,metadata.json}
  cache/validation/{features.npy,labels.npy,metadata.json}
```

If any check fails, repair only the failing adapter or representation contract.
Do not train. If every check passes, perform the literature/protocol selection
as a separate adjudication before creating the `2048/class` two-seed training
matrix.

### T2-A Result And Verdict

T2-A completed locally and passed every frozen readiness check:

| Category | Passed | Total |
| --- | ---: | ---: |
| cipher factory | 3 | 3 |
| strict differential dataset | 11 | 11 |
| disk cache and replay | 5 | 5 |
| runtime model contract | 4 | 4 |
| general GF(2) linear layer | 5 | 5 |
| **Total** | **28** | **28** |

The three runtime controls have the same `442466` trainable parameters, the
same parameter names and shapes, and finite forward outputs. Structure tensors
remain external runtime inputs and are absent from `state_dict`. The exact
SKINNY ShiftRows+MixColumns matrix has row and column degrees up to `3`, so this
fixture exercises a genuine many-source GF(2) layer rather than relabeling a
one-to-one permutation. Its forward and inverse matrices replay the cipher
layer and round-trip random bit states exactly.

Decision:

```text
status   = pass
decision = innovation1_runtime_spn_skinny_general_gf2_data_ready
training = false
empirical_topology_superiority_tested = false
```

Artifacts:

```text
outputs/local_readiness/i1_rtg1_skinny64_general_gf2_data_readiness_t2a_20260724/
```

This establishes only local implementation and data readiness. It contributes
no AUC, no topology-superiority evidence, no formal-scale evidence, no attack,
and no paper reproduction. In particular, the fixture's r7 and `0x40`
difference are not registered as a signal-bearing protocol.

### Recommended Next Action: T2-B Fixed-Key Signal-Anchor Selection

The next research question is whether an ordinary fixed-key SKINNY-64/64
strict differential task provides a learnable anchor for the unchanged runtime
E4 backbone before topology attribution is attempted.

Use one frozen, small candidate panel of externally justified single-cell input
differences and adjacent round counts. Keep ciphertext-pair features, four
independent pairs per sample, encrypted-random-plaintext negatives, validation
construction, optimizer, model geometry, and budget fixed. Change only the
round/difference protocol. Use one selection seed at a local diagnostic scale;
then confirm the selected anchor with a fresh seed. Do not run corrupted or
no-topology controls during selection, because a near-chance task cannot
adjudicate topology.

Only a signal-bearing, fresh-seed-confirmed anchor may open T2-C:

```text
scale      = 2048/class local diagnostic
seeds      = 0, 1
models     = correct topology / deterministic full-bit corrupted / no topology
advance    = per seed: correct - corrupted >= +0.005
             and correct - no topology >= +0.005
controls   = identical data, training, parameter names, shapes and counts
```

If no frozen candidate survives fresh-seed confirmation, stop the SKINNY
training branch and redesign the signal protocol or representation. Do not
mechanically increase samples, guess a paper protocol, substitute a related-key
benchmark, reuse Innovation 2 integral labels, or launch a remote job.
