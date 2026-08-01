# Innovation 1 Runtime SPN Same-Checkpoint Runtime Swap K1-BY8

**Date:** 2026-08-01
**Status:** completed / audit valid / same-checkpoint histogram access loss
**Execution:** local CPU, frozen checkpoints and caches, zero neural training

## Research question

K1-BY6 compared separately trained correct and affine-wrong PRESENT programs.
Seed2 preferred the correct program, while seed3 preferred the affine control.
K1-BY7 localized seed3's first internal reversal to the linear-histogram tap,
but those representations still came from independently trained checkpoints.

K1-BY8 asks the missing causal question:

> With every learned parameter fixed, does changing only the runtime program
> from the correct PRESENT mapping to the affine wrong-endpoint mapping reverse
> the output and internal representation preference?

This is an inference-only mechanism audit. It does not train, calibrate, select
or modify a checkpoint and does not generate data.

## Frozen 2 x 2 intervention

For each seed, evaluate four cells:

| Learned parameter source | Runtime program | Role |
|---|---|---|
| correct K1-BY3 checkpoint | correct | diagonal correct anchor |
| correct K1-BY3 checkpoint | affine wrong endpoint | primary causal swap |
| affine K1-BY6 checkpoint | correct | reciprocal causal swap |
| affine K1-BY6 checkpoint | affine wrong endpoint | diagonal affine anchor |

Only tensors returned by `named_parameters()` are copied from a source
checkpoint into a newly constructed target-runtime model. Runtime matrices,
semantic cell-bit maps, edge descriptors, masks, expert types and the compiled
runtime object remain those of the target model. Within one weight source the
two target models must have byte-identical learned-parameter fingerprints;
within one runtime choice they must have byte-identical runtime-buffer
fingerprints. Correct and affine runtime fingerprints must differ.

## Frozen protocol

```text
cipher / rounds       = PRESENT-80 / r7
validation rows       = exact K1-BY3 caches, 2048 rows per seed
seeds                 = 2,3
pairs per sample      = 16
input width           = 2048 bits
negative definition   = encrypted random plaintexts
checkpoints           = exact K1-BY3 correct and K1-BY6 affine best checkpoints
taps                  = K1-BY7's same five forward-information-flow taps
probe discovery       = even validation indices, 512/class
probe evaluation      = odd validation indices, 512/class
probe                 = variance-normalized class-mean difference
batch / device        = 128 / local CPU
epochs / optimizer    = 0 / none
```

The two diagonal cells must replay their source AUCs within `1e-6`. Source
artifacts and dataset hashes must remain unchanged before and after evaluation.

## Preregistered research gate

For the correct-weight checkpoint on each seed independently require:

```text
final AUC(correct runtime) - final AUC(affine runtime) >= +0.005
linear-histogram probe AUC(correct runtime)
  - linear-histogram probe AUC(affine runtime)         >= +0.005
```

The affine-weight reciprocal cells are explanatory and cannot rescue a failed
correct-weight seed. Seed averaging is prohibited.

## Decision routes

- If both seeds pass both clauses, classify K1-BY6's seed3 reversal as coupled
  independent-training variance. Retain same-checkpoint controls as mandatory
  for future topology attribution; do not redesign the representation from
  K1-BY7 alone.
- If either seed fails first at the linear histogram under correct weights,
  confirm a same-checkpoint runtime-access loss and change exactly the
  state-to-histogram representation to preserve relative source-bundle
  incidence.
- If both linear-histogram margins pass but a later tap or final output fails,
  localize the first same-checkpoint downstream loss and change only that
  interface.
- If protocol validity fails, repair only the source binding, parameter-only
  transfer, runtime preservation, hook, probe or artifact invariant and rerun
  unchanged.

In all routes, keep the method on hold. Do not add samples, pairs, epochs,
width, seeds, ciphers, model training or remote execution.

## Required artifacts

The completed run must produce preflight, 40 internal-probe rows, four-cell
final-AUC replay, gate, validation, summary, comparison CSV, progress JSONL and
a Chinese SVG. The SVG must pass rendered-pixel `visual-qa-redraw` inspection,
and both recent-result indexes must be refreshed before reporting.

## Executable next action

The named-parameter-only swap is implemented and the focused K1-BY7/K1-BY8
suite passes (`12 passed`). Readiness passed every source, four-cell geometry,
parameter-fingerprint, runtime-buffer-fingerprint, diagonal-output and
off-diagonal-intervention check. The exact learned parameter fingerprints are
identical across runtimes within each weight source, while the correct and
affine runtime fingerprints are distinct.

Execute the frozen audit, render and inspect the figure, then record the
observed decision here. Do not change the protocol after observing metrics.

## Completed result

The frozen audit completed with all source, parameter-transfer, runtime-buffer,
hook, probe and artifact checks passing:

```text
result rows                 = 40
neural training performed   = false
optimizer steps             = 0
diagonal source replay      = pass
source artifacts unchanged  = true
status                      = pass
method status               = hold
research gate               = fail
decision                    = innovation1_runtime_spn_k1by8_same_checkpoint_histogram_access_loss
```

Final validation AUCs were:

| Seed | Correct weights + correct runtime | Correct weights + affine runtime | Margin | Affine weights + correct runtime | Affine weights + affine runtime | Margin |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | `0.683737` | `0.662154` | `+0.021583` | `0.663229` | `0.644393` | `+0.018836` |
| 3 | `0.665544` | `0.655450` | `+0.010093` | `0.698355` | `0.692050` | `+0.006305` |

Both correct-weight checkpoints therefore prefer the correct runtime at final
output by more than `+0.005`. K1-BY6's seed3 final reversal is not reproduced
when the learned parameters are held fixed.

The internal same-checkpoint margins tell a more specific story:

| Weight source / seed | Linear histogram | Permutation expert | Cell fusion | Stage pooling | Pre-classifier | Final output |
|---|---:|---:|---:|---:|---:|---:|
| Correct / seed2 | `+0.023502` | `+0.009502` | `+0.014339` | `+0.015549` | `+0.034039` | `+0.021583` |
| Correct / seed3 | `-0.005447` | `+0.015350` | `+0.043461` | `+0.024731` | `+0.034908` | `+0.010093` |
| Affine / seed2 | `+0.023502` | `+0.007900` | `+0.012241` | `+0.008911` | `+0.020031` | `+0.018836` |
| Affine / seed3 | `-0.005447` | `+0.014923` | `+0.015694` | `+0.012722` | `+0.028568` | `+0.006305` |

The linear histogram precedes every learned primitive parameter, so its values
are identical across the two weight sources. Seed3's `-0.005447` reversal at
that tap is therefore a deterministic runtime-representation property rather
than an optimizer or checkpoint artifact. The learned permutation expert then
recovers a positive correct-runtime margin under both weight sources, and all
later taps retain it.

K1-BY8 consequently resolves the two coupled hypotheses from K1-BY7:

1. independently trained weights caused the K1-BY6 seed3 final model ranking
   to reverse;
2. the current state-to-histogram interface still gives the affine wrong
   endpoint slightly stronger raw label association on seed3.

The final Chinese SVG was rendered at `2700 x 1336` pixels. The first render
exposed overlapping near-equal seed labels and a misleading success-colored
decision line. After staggered annotations and a hold-colored decision were
applied, `visual-qa-redraw` found no overlap, clipping, missing glyph, ambiguous
threshold, unreadable series or misleading color.

## Evidence-backed next action

Preregister K1-BY9 as one same-budget representation repair. Keep every
checkpoint, cache, seed, pair, tap, probe, expert, pooler and classifier frozen.
Change only the linear state-to-histogram interface:

```text
K1-BY8 anchor:
  inverse-linear state -> per-cell 16-bin difference histogram

K1-BY9 candidate:
  inverse-linear state
    -> group the four source bits that feed each target cell
    -> preserve the relative source-bundle incidence
    -> shared 16-bin bundle histogram per target cell
```

The candidate must remain invariant to joint cell relabeling and must not use
cipher identity or absolute cell identity. First run a zero-training
deterministic readiness audit proving correct/affine buffer distinction and
shape compatibility. Then evaluate the same two correct checkpoints with
correct and affine runtimes. Advance only if both seeds satisfy linear-
histogram margin `>= +0.005`, all downstream/final margins remain `>= +0.005`,
and both diagonal anchors replay within `1e-6`. Otherwise discard the repair.

Do not retrain weights while selecting the representation, increase data,
change the PRESENT difference, add ciphers or launch remote execution.
