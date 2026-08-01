# Innovation 1 Runtime SPN Source-Bundle Collision K1-BY10

**Date:** 2026-08-01
**Status:** completed / valid diagnostic / no stable locus
**Execution:** local CPU, frozen K1-BY9 models and validation caches

## Research question

K1-BY9 replaced each PRESENT linear-state cell histogram by a fixed blend of
its local histogram and the mean over cells with the same unordered source-cell
set. It repaired every seed3 tap, but reduced seed2's correct-minus-affine
margin below `+0.005` at the permutation expert and cell fusion.

K1-BY10 asks one diagnostic question:

> Is the seed2 loss concentrated at one deterministic target-cell locus whose
> correct and affine equality classes partially overlap but change peers in
> both stages, while the same locus does not harm seed3?

No representation, parameter, checkpoint, sample, label or output changes.
This audit only decomposes the already completed K1-BY9 tensors.

## Frozen source and measurements

The exact K1-BY9 source root, config and seven evidence digests are frozen in
the JSON config. K1-BY9 must remain protocol-valid with its expected valid-miss
decision. Both anchor and candidate models are rebuilt by K1-BY9, using only
the K1-BY3 correct checkpoints and their original runtime buffers.

For every seed, representation, runtime, stage, target cell and cell-preserving
tap, apply the exact K1-BY9 discovery/evaluation split and fixed mean-difference
probe:

```text
2 seeds x 2 representations x 2 runtimes x 2 stages x 16 cells x 3 taps
= 768 probe rows
```

The taps are:

```text
linear_histogram
linear_primitive_expert
cell_fusion
```

`tap_stage=0/1` follows forward hook order, which is reverse program order:
the first captured transition is `program_stage=1`, followed by
`program_stage=0`. Both coordinates are recorded to prevent stage ambiguity.

For every runtime, stage and target cell also record the equality-class peers,
the correct/affine class intersection size and the peers changed by the runtime
swap. Cell numbers are diagnostic coordinates only; they are never supplied to
the model or used to select a checkpoint.

## Frozen per-cell effect

For a fixed seed, stage, cell and tap:

```text
anchor_margin = anchor_correct_probe_auc - anchor_affine_probe_auc
candidate_margin = candidate_correct_probe_auc - candidate_affine_probe_auc
candidate_effect = candidate_margin - anchor_margin
```

The global K1-BY9 observation was a negative candidate effect on seed2 and a
non-negative repair on seed3. K1-BY10 tests whether one cell explains that
cross-seed asymmetry consistently at both learned downstream cell taps.

## Readiness gate

Before complete-cache evaluation require:

1. every frozen K1-BY9 digest and source binding remains exact;
2. the K1-BY9 decision remains the expected protocol-valid representation miss;
3. both seeds rebuild the exact four K1-BY9 intervention cells and parameter
   fingerprints;
4. every captured cell tap has shape `[batch, 2, 16, width]` and is finite;
5. correct and affine equality matrices remain valid idempotent class means;
6. partition metadata is identical across seeds and deterministic across rebuilds;
7. no training, optimizer, data generation or checkpoint selection is possible.

Any failure makes the audit invalid and permits only repair of that invariant.

## Preregistered research gate

A target cell is a supported over-smoothing locus only when all clauses hold:

```text
for both stages and both taps {linear_primitive_expert, cell_fusion}:
  seed2 candidate_effect <= -0.005
  seed3 candidate_effect >=  0.000

for both stages:
  correct/affine equality classes are distinct
  class intersection size is one or two
  at least four peer memberships change under the runtime swap
```

The same target cell must pass in both stages. A seed average, different cell
per stage or histogram-only effect cannot rescue the gate.

## Decision routes

- **Locus identified:** close averaging itself, but retain relative
  source-bundle incidence as a mechanism. Preregister one non-averaging
  per-cell residual that preserves the local histogram and exposes only the
  signed local-minus-bundle deviation. Compare it with the local anchor at the
  same budget before training or scale.
- **No stable locus:** close equality-partition pooling. Return to an
  edge-conditioned residual that preserves individual cells and does not use
  equality-class means.
- **Protocol invalid:** repair only the failed frozen source, shape, partition,
  probe or artifact binding and rerun unchanged.

In all routes, changing `0.5/0.5`, training K1-BY9, adding data, pairs, capacity,
seeds, ciphers or remote execution remains prohibited.

## Required artifacts

```text
run_id = i1_runtime_spn_source_bundle_collision_k1by10_present_r7_seed2_seed3_20260801
```

The audit must emit preflight, 768 probe rows, partition metadata, per-cell
effect CSV, gate, validation, summary, progress and a Chinese SVG. The rendered
SVG must pass `visual-qa-redraw`, and both recent-result indexes must be
refreshed before reporting.

## Completed result

The frozen local audit completed without training or optimizer steps:

```text
result rows       = 768
effect rows       = 192
partition rows    = 32
protocol checks   = pass
source unchanged  = true
research gate     = fail
supported cells   = none
decision          = innovation1_runtime_spn_k1by10_no_stable_partition_locus_identified
```

The correct and affine partitions are stable across both stages. Each contains
24 within-class peer pairs. Four peer pairs are shared and 40 change; per-cell
class intersections have size one or two and each target changes four or six
peer memberships. This validates the intended structural intervention but does
not by itself identify a seed-specific loss locus.

The strongest per-cell effects were distributed across different cells and
changed with stage and learned tap:

| Seed | Capture stage | Tap | Most negative cell | Candidate-minus-anchor margin |
|---:|---:|---|---:|---:|
| 2 | 0 | permutation expert | 1 | `-0.088638` |
| 2 | 0 | cell fusion | 10 | `-0.027336` |
| 2 | 1 | permutation expert | 0 | `-0.012501` |
| 2 | 1 | cell fusion | 3 | `-0.022968` |
| 3 | 0 | permutation expert | 1 | `-0.074486` |
| 3 | 0 | cell fusion | 1 | `-0.100513` |
| 3 | 1 | permutation expert | 0 | `-0.042633` |
| 3 | 1 | cell fusion | 0 | `-0.009460` |

No target cell simultaneously showed the required seed2 loss and non-negative
seed3 effect at both downstream taps in both stages. Several of seed2's largest
losses also harmed seed3, while other cells improved one stage and harmed the
other. The K1-BY9 behavior is therefore not explained by one deterministic
equality-class collision. It is a distributed, checkpoint-dependent response
to averaging local cell statistics.

The Chinese SVG was rendered at `2700 x 1530` pixels and passed
`visual-qa-redraw`. All four heatmaps share a symmetric zero-centered scale;
stage mappings, target-cell labels, seed-specific threshold markers, colorbar,
decision and footer are readable without overlap, clipping or missing glyphs.

Artifacts:

```text
outputs/local_audit/
  i1_runtime_spn_source_bundle_collision_k1by10_present_r7_seed2_seed3_20260801/
```

## Evidence-backed next action

Close equality-partition pooling and do not attempt a signed bundle-deviation
residual. The next representation experiment should return to the full ordered
edge descriptors already present in the compiled primitive program and preserve
every target cell independently:

```text
question = can a non-averaging edge-conditioned residual preserve the K1-BY8
           seed2 access while repairing the seed3 linear-histogram loss?
anchor = exact K1-BY8 local histogram and frozen K1-BY3 correct checkpoints
candidate = local histogram unchanged plus a zero-parameter deterministic
            edge-conditioned modulation derived from existing ordered edge tokens
controls = correct runtime, affine runtime, edge-token-shuffled runtime
changed variable = deterministic per-cell edge modulation only
execution = local CPU, zero training, seeds 2/3, exact validation caches and probes
advance = every candidate tap and final margin >= +0.005 on both seeds, with
          correct final AUC no worse than anchor by 0.005
stop = any seed/tap miss; close deterministic input modulation and move the
       structure intervention after the primitive expert
```

Do not use cell IDs, equality means, blend tuning, new parameters, optimizer
steps, samples, pairs, ciphers or remote execution in that next gate.
