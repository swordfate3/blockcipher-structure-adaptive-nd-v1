# Innovation 1 Runtime SPN Same-Checkpoint Runtime Swap K1-BY8

**Date:** 2026-08-01
**Status:** preregistered / readiness pass / execution authorized
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
