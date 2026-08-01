# Innovation 1 Runtime SPN Paired Runtime Objective K1-BY14

**Date:** 2026-08-01
**Status:** preregistered / implementation pending / no optimizer step authorized
**Execution:** zero-training readiness locally; diagnostic training uses local
CUDA when available, otherwise the established remote-A6000 device-availability
exception after the exact source commit is published and verified

## Research Question

K1-BY8 showed that the same frozen correct PRESENT checkpoint gives a higher
final AUC under the correct runtime than under an affine wrong-endpoint runtime
on both seeds. K1-BY13 then showed that an independently trained post-expert
adapter moves away from zero but does not stably prefer the correct edges.

K1-BY14 changes only the training objective:

> Can the unchanged K1-BY3 architecture preserve its existing PRESENT r7
> signal while a paired primary-versus-counterfactual objective makes the
> correct runtime preferable to both the training-time affine runtime and a
> fresh held-out shuffled runtime?

This is a local-scale objective diagnostic. It is not formal evidence, a scale
experiment, a new attack, a SOTA claim, cross-cipher transfer or proof that an
arbitrary SPN structure can be learned automatically.

## Frozen Precedent And Failure Control

K1-AM previously used a paired semantic loss on Midori64 and showed that an
objective can impose an arbitrary orientation without resolving independently
trained substitutes. K1-BY14 therefore requires all three controls below:

1. a swapped-orientation placebo trained with the identical objective;
2. same-checkpoint evaluation under the correct and affine runtimes;
3. a held-out shuffled runtime not present in either training objective.

A positive correct-versus-affine margin alone is insufficient.

## Frozen Protocol

```text
run_id              = i1_runtime_spn_paired_runtime_objective_k1by14_present_r7_16pair_2048_seed2_seed3_20260801
cipher / rounds      = PRESENT-80 / r7
data protocol        = Zhang/Wang Case2 official MCND
input difference     = 0x0000000000000009
pairs per sample     = 16 independent ciphertext pairs
input width          = 2048 bits
train                = 2048/class = 4096 total rows/seed
cross-key validation = 1024/class = 2048 total rows/seed
seeds                = 2,3
train key            = 0x00000000000000000000
validation key       = 0x11111111111111111111
negative definition  = encrypted random plaintexts
architecture         = unchanged K1-BY3 ordered primitive conditioner
parameters           = 235780; no new trainable parameter
runtime window       = PRESENT rounds 0/1, two compiled stages
epochs / batch       = 10 / 64
loss / optimizer     = MSE / Adam
learning rate        = 1e-4
weight decay         = 1e-5
checkpoint           = restored best cross-key validation AUC
contrast scale       = 0.25
contrast margin      = 0.02
```

The ordinary-MSE anchor is the completed, bound K1-BY3 correct-runtime row for
the same seed and exact protocol. It is not retrained.

## Four-Row Training Matrix

Train exactly two orientations per seed:

| Orientation | Primary runtime | Counterfactual runtime | Purpose |
|---|---|---|---|
| `correct_oriented` | correct PRESENT | affine wrong endpoint `m=5,b=1` | candidate |
| `swapped_orientation` | affine wrong endpoint `m=5,b=1` | correct PRESENT | arbitrary-orientation placebo |

For each sample, let `e_p` and `e_c` be the primary and counterfactual
per-sample MSE values. The only added loss is:

```text
L_aux   = 0.25 * mean(relu(0.02 + e_p - e_c))
L_total = L_primary + L_aux
```

Both forward passes use the same learned parameter tensors. The counterfactual
runtime contributes only alternate deterministic program buffers. The
counterfactual model must not add trainable parameters or enter the optimizer.

## Frozen Evaluation Panel

Restore each of the four best checkpoints and copy named parameters only into
three equal-geometry runtime models:

```text
correct runtime
affine wrong-endpoint runtime used during training
held-out target-binding shuffle with seed 20260814
```

This yields exactly twelve zero-training evaluation rows. Within an
orientation/seed, all three rows must share the same learned-parameter,
validation-feature and validation-label hashes while retaining distinct
runtime-program hashes.

## Readiness Gate

Before any optimizer step require:

1. exact K1-BY3, K1-BY8 and K1-BY13 source decisions and artifact digests;
2. exactly four frozen tasks and identical data/optimization fields;
3. correct and swapped models have identical trainable parameter geometry and
   byte-identical initialization within each seed;
4. both models retain exactly `235780` trainable parameters;
5. the counterfactual forward reuses the primary named parameters and changes
   only runtime buffers;
6. correct, affine and held-out-shuffled program hashes are pairwise distinct;
7. primary and counterfactual logits are finite, unequal on a fixed fixture and
   produce a finite, positive auxiliary loss and nonzero gradient;
8. no cipher identity, absolute cell identity, adapter, DDT or trail feature is
   introduced;
9. local readiness performs zero optimizer steps.

Any failure blocks training. Repair only the failed binding and rerun the same
readiness gate.

## Research Gate

For both seeds independently require:

```text
correct-oriented AUC                              >= 0.550
correct-oriented - ordinary K1-BY3 anchor         >= -0.005
correct-oriented - swapped-orientation primary    >= +0.005
same-checkpoint correct - affine runtime           >= +0.005
same-checkpoint correct - held-out shuffled        >= +0.005
training auxiliary loss                            finite and > 0
training semantic loss gap                         finite
```

No seed averaging may rescue a failed clause.

## Decisions

- **All clauses pass on both seeds:** retain paired runtime preference as a
  local objective result. Next repeat the unchanged four-row protocol on one
  compatible non-PRESENT SPN before any scale increase.
- **Correct same-checkpoint margins pass but orientation placebo fails:** the
  objective imposes an arbitrary preference. Discard it and stop supervised
  topology-preference objectives on this PRESENT surface.
- **Anchor retention fails:** discard the objective because it damages the
  existing distinguisher signal.
- **Held-out shuffle fails:** the objective overfits the single affine
  counterexample. Discard it; do not add a larger negative-runtime bank.
- **Protocol invalid:** repair only the failed implementation or artifact
  binding and rerun unchanged.

If the research gate fails, retain deterministic primitive compilation and
runtime routing as the supported method. Do not tune the contrast scale or
margin, add adapters, increase depth, pairs, samples, epochs or seeds, launch
medium scale, add MoE, or select a favorable wrong runtime after seeing AUC.

## Required Artifacts And Next Action

The completed diagnostic must produce preflight, progress JSONL, four training
rows, four restored checkpoints, twelve same-checkpoint evaluation rows,
validation, gate, summary, comparison/history CSV and a Chinese SVG. The SVG
must pass `visual-qa-redraw`, and both recent-result indexes must be refreshed.

Immediate next action: implement only the paired runtime objective, named-
parameter counterfactual call, held-out evaluation and fail-closed gates; then
run zero-training readiness before any diagnostic training.
