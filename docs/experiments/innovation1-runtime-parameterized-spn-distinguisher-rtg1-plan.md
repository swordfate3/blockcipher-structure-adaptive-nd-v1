# Innovation 1 Runtime-Parameterized SPN Distinguisher Plan

**Status:** R0 implementation readiness passed; R1 seed0 local attribution authorized

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
