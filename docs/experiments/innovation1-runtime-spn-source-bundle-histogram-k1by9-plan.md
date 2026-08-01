# Innovation 1 Runtime SPN Source-Bundle Histogram K1-BY9

**Date:** 2026-08-01
**Status:** preregistered / implementation pending / zero training
**Execution:** local CPU, frozen checkpoints and validation caches

## Research question

K1-BY8 held every learned parameter fixed and changed only the PRESENT runtime
program. Both seeds preferred the correct runtime at final output, but seed3's
first linear-histogram probe preferred the affine wrong endpoint by `0.005447`
AUC. The learned permutation expert recovered the correct ordering afterward.

K1-BY9 tests one representation repair:

> Can relative source-bundle incidence remove the deterministic seed3 loss at
> the linear-histogram interface without damaging the existing downstream and
> final-output advantages?

This is an inference-only audit. It performs no optimization, data generation,
checkpoint selection or hyperparameter search.

## Single changed variable

The K1-BY8 anchor remains unchanged:

```text
inverse-linear state
  -> one local 16-bin difference histogram per semantic target cell
```

The candidate groups target cells by an equality relation derived only from
the compiled linear program. Two target cells are in one class exactly when
the unordered sets of source cells feeding their four target roles are equal.
For every target cell and stage:

```text
local_histogram = existing 16-bin per-cell difference histogram
bundle_mean     = mean local_histogram over its source-bundle equality class
candidate       = 0.5 * local_histogram + 0.5 * bundle_mean
```

The `0.5/0.5` blend is fixed before observation and cannot be swept. Numerical
source-cell values are never embedded or sorted into features; only equality
of complete source-cell sets is used. A bijective source-cell renaming leaves
the equivalence matrix unchanged, and a target-cell relabeling only conjugates
that matrix. The candidate therefore uses relative incidence rather than
absolute source, target, bit or cipher identity.

The candidate applies only to the linear histogram. S-box histograms, learned
experts, edge descriptors, fusion, attention, recurrence, pooler, classifier
and all learned parameter shapes remain unchanged.

## Frozen four-cell intervention

Only the K1-BY3 correct checkpoint is used for each seed:

| Representation | Runtime program | Role |
|---|---|---|
| existing local histogram | correct | exact K1-BY8 anchor replay |
| existing local histogram | affine wrong endpoint | exact K1-BY8 control replay |
| source-bundle histogram | correct | candidate |
| source-bundle histogram | affine wrong endpoint | candidate runtime control |

The two anchor cells must reproduce K1-BY8's correct-weight cells within
`1e-6`. Candidate cells are not required to reproduce an old checkpoint
output because their deterministic representation has changed. Every cell
must contain byte-identical learned parameters copied with
`named_parameters()` only; runtime buffers remain owned by the target model.

## Frozen protocol

```text
cipher / rounds       = PRESENT-80 / r7
validation rows       = exact K1-BY3 caches, 2048 rows per seed
seeds                 = 2,3
pairs per sample      = 16
input width           = 2048 bits
negative definition   = encrypted random plaintexts
parameter source      = exact K1-BY3 correct best checkpoint per seed
runtime programs      = correct, affine endpoint m5+b1 mod64
taps                   = exact K1-BY8 five-tap forward order
probe discovery       = even validation indices, 512/class
probe evaluation      = odd validation indices, 512/class
probe                  = variance-normalized class-mean difference
batch / device         = 128 / local CPU
epochs / optimizer    = 0 / none
```

The completed audit must produce exactly
`2 seeds x 2 representations x 2 runtimes x 5 taps = 40` probe rows.

## Readiness gate

Before evaluating the complete caches, prove:

1. K1-BY8 source files and digests are exact and its decision is the expected
   same-checkpoint histogram access loss;
2. anchor models retain the old buffer contract and exactly replay both
   K1-BY8 correct-weight fixture outputs;
3. candidate models have the same named parameters and parameter geometry as
   the anchor, with only the candidate source-bundle matrix added as a buffer;
4. correct and affine candidate source-bundle matrices are distinct;
5. every equivalence matrix is finite, symmetric, row-stochastic and
   idempotent, as required for an equality-class mean;
6. a bijective renaming of every source-cell label leaves every candidate
   equivalence matrix byte-identical;
7. the candidate changes the linear-histogram tap for both runtimes while the
   S-box histogram remains the exact existing local representation;
8. all four outputs and five taps are finite with no optimizer step.

Any readiness failure prohibits the audit. Repair only the failed source,
matrix, mode-routing, parameter-transfer, runtime-buffer or hook invariant.

## Preregistered result gate

For each seed independently require all of the following:

```text
anchor correct/affine final AUC replay error <= 1e-6

candidate correct-runtime - candidate affine-runtime:
  every internal tap probe AUC margin >= +0.005
  final output AUC margin             >= +0.005

candidate correct-runtime final AUC
  - anchor correct-runtime final AUC  >= -0.005
```

Both seeds must pass every clause. A mean cannot rescue a failed seed or tap.

## Decision routes

- **Pass:** retain the source-bundle representation as a candidate. Next run a
  separately preregistered same-budget training confirmation against the local
  histogram anchor; do not scale or add ciphers first.
- **Valid miss:** discard this fixed source-bundle mean. Audit whether the
  equality partition collides between correct and affine programs before
  proposing one different representation; do not tune the blend after seeing
  the result.
- **Protocol invalid:** repair only the failed invariant and rerun unchanged.

In every route the broader method remains held. This zero-training audit is
not formal-scale training, cross-cipher transfer, a neural attack or SOTA
evidence.

## Required artifacts

```text
run_id = i1_runtime_spn_source_bundle_histogram_k1by9_present_r7_seed2_seed3_20260801
```

The run must emit preflight, 40 result rows, four-cell final AUCs, gate,
validation, summary, model metadata, comparison CSV, progress JSONL and a
Chinese SVG. The SVG must pass rendered-pixel `visual-qa-redraw`, and both
recent-result indexes must be refreshed before the result is reported.

## Completed result

The frozen zero-training audit completed with every source, parameter,
runtime-buffer, equivalence-matrix, relabeling, probe and artifact check
passing:

```text
result rows                 = 40
neural training performed   = false
optimizer steps             = 0
anchor correct/affine replay= exact on both seeds
source artifacts unchanged  = true
status                      = pass
method status               = hold
research gate               = fail
decision                    = innovation1_runtime_spn_k1by9_source_bundle_histogram_repair_not_supported
```

The candidate margins were:

| Seed | Linear histogram | Permutation expert | Cell fusion | Stage pooling | Pre-classifier | Final output |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `+0.006069` | `+0.002674` | `+0.003471` | `+0.015877` | `+0.033215` | `+0.022757` |
| 3 | `+0.005592` | `+0.021080` | `+0.045727` | `+0.033913` | `+0.034023` | `+0.012746` |

The representation solved the exact seed3 failure that motivated K1-BY9:
its linear-histogram margin changed from K1-BY8's `-0.005447` to
`+0.005592`, and every later seed3 tap passed. Seed3's correct-runtime final
AUC increased from `0.665544` to `0.671487`, a `+0.005944` change.

It did not pass the two-seed gate. Seed2 retained a positive linear-histogram
margin, but the permutation-expert and cell-fusion margins fell to
`+0.002674` and `+0.003471`, below the required `+0.005`. Its correct-runtime
final AUC changed from `0.683737` to `0.681323`, a permitted `-0.002414`, and
its final correct-minus-affine margin remained strong. The failure is thus a
cross-seed internal-access instability rather than loss of final distinguishability.

The narrow conclusion is that equality-class averaging exposes useful
relative source-bundle information for seed3 but over-smooths information used
by the frozen seed2 permutation expert. The fixed `0.5/0.5` representation is
discarded. Its blend must not be tuned after observation and it does not
authorize training, scale, another cipher or a positive method claim.

The final Chinese SVG was rendered at `2700 x 1336` pixels. The first two
renders exposed collisions among close seed labels. The final data-aware
above/below annotation layout passed `visual-qa-redraw`: titles, subtitles,
ticks, legends, threshold lines, values and the hold-colored decision are
readable without overlap, clipping, missing glyphs or misleading scales.

Artifacts:

```text
outputs/local_audit/
  i1_runtime_spn_source_bundle_histogram_k1by9_present_r7_seed2_seed3_20260801/
```

## Evidence-backed next action

Run one zero-training equality-partition collision audit before proposing
another representation:

```text
question        = why does bundle averaging repair seed3 but over-smooth seed2?
source          = exact K1-BY9 correct/affine matrices and per-cell probe tensors
changed variable= none; deterministic attribution only
measurements    = per-stage class membership overlap, changed target-cell pairs,
                  and per-cell contribution to the seed2/seed3 probe margins
execution       = local CPU, zero optimizer steps
advance gate    = identify one deterministic collision or over-smoothing locus
                  shared by both stages and consistent with the observed tap loss
stop gate       = no stable locus; close equality-class pooling and return to an
                  edge-conditioned residual that preserves individual cells
```

Do not sweep the blend, retrain the candidate, add samples, pairs, capacity,
seeds or ciphers, or launch remote work from K1-BY9.
