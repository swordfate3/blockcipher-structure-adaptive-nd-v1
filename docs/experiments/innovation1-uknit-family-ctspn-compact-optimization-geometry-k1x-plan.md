# Innovation 1 uKNIT-Family CT-SPN Compact Optimization Geometry K1-X

**Date:** 2026-07-28
**Status:** completed / hold / 16x relation verified but not sufficient
**Execution:** local CPU; inference and gradient inspection only

## Research question

K1-W proved that the sixteen-slot invariant histogram projection and its
compact forty-feature projection are algebraically equivalent after folding,
but fresh compact training lost the uKNIT r5 signal on both seeds. Dialga r4
retained its strong base-model signal, so the failure is specific to learning
the histogram contribution rather than a general forward-pass defect.

K1-X asks one narrow question:

> Does the redundant sixteen-cell parameterization supply a sixteen-fold
> effective projection-weight update that the compact parameterization loses,
> while the failed K1-W checkpoints make only a weak histogram contribution?

K1-X is a post-result mechanism audit. It is not a blind confirmation and may
only authorize a separately planned optimization intervention.

## Frozen sources

Use only the completed uKNIT r5 seed3/4 rows and their existing K1-Q-bound
train and validation caches.

```text
K1-W gate SHA256                 = 8f94cd31798638313d21c632445004ceb9d3fee545b5d3813b1ed6e4b998e338
K1-W results SHA256              = 75a7bdad3fb64b562c92545f4734e14dfad6c2d002b0099c5c02c0a1495a37e7
K1-W seed3 exact checkpoint      = 3a53e7e1c648a5e2998f014285ed61218bae86fd1e5fa6a2216fc32dbbb76821
K1-W seed4 exact checkpoint      = 9f5e6f001e732ae4c2c74ceb300e928e63a0042426d7afa9e1a546258accc5df

K1-T gate SHA256                 = f122f43f4d895a1b68fb696bd81df4e1d362880a3a12d9883933c932dd7f0dbf
K1-T results SHA256              = adafb1217298ade5ad7bda4aff5a53742e951e5c737babb30a449362e948563a
K1-T checkpoint manifest SHA256  = b971fa9e25b70c4a7a76caff608a886e758b356091881fd40ab42ff2e4289bc9
K1-T seed3 invariant checkpoint  = aea483d1438617472216d1f7a70574ae99b0c2bf32ede601fa4a58bd1eed55ae
K1-T seed4 invariant checkpoint  = 21a91a173b3929759971f1ae198f26d3d1e5e0dc802ffa3686183600d635620b
```

The two validation caches are exactly the K1-W/K1-T source caches:

```text
seed3 validation = K1-Q seed-10003_222ac0f458b64b18
seed4 validation = K1-Q seed-10004_f2b02ef8a58bdb97
```

The gradient audit uses only the first frozen batch of 64 rows from the
corresponding K1-Q training cache. No cache may be generated or modified.

## Audit matrix

For each seed independently, strictly load:

1. the restored K1-W compact-exact checkpoint;
2. the restored K1-T invariant checkpoint in the old sixteen-slot model;
3. a compact model obtained by folding that same K1-T state.

On the complete 2048-row validation split measure:

| Checkpoint state | Runtime condition |
|---|---|
| K1-W compact | exact S-box |
| K1-W compact | histogram gate forced to zero |
| K1-W compact | wrong S-box with identical learned tensors |
| folded K1-T compact | exact S-box |
| folded K1-T compact | histogram gate forced to zero |
| folded K1-T compact | wrong S-box with identical learned tensors |

Report AUC, fixed-threshold accuracy, effective histogram gate, full-minus-zero
AUC and exact-minus-wrong-Sbox AUC. Forcing the histogram gate to zero is an
inference-only intervention; restore the original state before the next row.

## Gradient proof

Use the folded K1-T state in float64 and the same frozen 64-row batch for the
old invariant and compact models. Compute the unchanged K1-W training loss:

```text
loss = MSE(sigmoid(logits), labels)
```

Call `backward()` once on each model without constructing or stepping an
optimizer. For the old projection weight, reshape the gradient to
`[128, 5, 16, 8]`. Require:

```text
each of the 16 cell-slot gradients == compact gradient
sum(old cell-slot gradients) == 16 * compact gradient
old and folded-compact logits/losses are equivalent
model state hashes are unchanged after gradient inspection
optimizer_steps = 0
```

The slot equality and folded ratio use relative error at most `1e-9`, with an
absolute floor of `1e-12`. The folded effective update ratio is reported from
the gradient norm and must lie in `[15.999, 16.001]` for both seeds.

## Frozen decision gate

K1-X supports the optimization-geometry mechanism only if all source,
strict-load, replay, state-restoration and zero-step checks pass and, for both
seeds independently:

```text
projection slot-gradient equality relative error <= 1e-9
folded gradient ratio in [15.999, 16.001]
abs(K1-W full AUC - K1-W zero-histogram AUC) <= 0.010
abs(K1-W exact AUC - same-checkpoint wrong-Sbox AUC) <= 0.010
```

The K1-T folded checkpoint contribution is descriptive mechanism context; its
forward AUC must replay the K1-T invariant source within `1e-6`, but no new
post-result benefit threshold is imposed on it.

## Decisions

- **All gates pass:** authorize K1-Y as a separate local diagnostic changing
  only the compact first projection weight's optimizer update scale. Freeze a
  `16x` learning-rate multiplier for that one weight and retain all other K1-W
  data, architecture, pairs, epochs, seeds, loss, optimizer settings and
  wrong-Sbox controls.
- **Gradient ratio fails:** reject the proposed update-geometry explanation.
  Do not tune learning rates; return to a different invariant aggregator.
- **K1-W histogram contribution is not weak:** the compact failure is not
  explained by an under-updated histogram branch. Audit interference or
  semantic mismatch before any training change.
- **Protocol invalid:** repair only the failed source binding, strict load,
  intervention restoration or numerical audit and rerun unchanged.

Blocked in K1-X: training, optimizer steps, sixteen pairs, more data, epochs,
seeds, positions or rounds; remote launch; MoE; learned cipher IDs; changing
the negative definition; and treating a gradient identity as attack or SOTA
evidence.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_compact_optimization_geometry_k1x_20260728
```

Produce source bindings, progress JSONL, inference results JSONL, gradient
results JSONL, comparison CSV, gate, summary, validation and a Chinese SVG.
The rendered SVG must pass `visual-qa-redraw`. Refresh both recent-result
indexes after completion and record the evidence-backed next action here.

## Completed result

All source, strict-load, source-AUC replay, state-restoration and zero-step
checks passed. Both seeds reproduced the algebraic update relation to floating
precision:

| Seed | Folded effective ratio | Slot-gradient relative error | Folded-gradient relative error |
|---:|---:|---:|---:|
| 3 | `16.0000000000` | `4.84e-16` | `4.84e-16` |
| 4 | `16.0000000000` | `4.33e-16` | `4.33e-16` |

The checkpoint interventions were:

| Seed | K1-W exact | K1-W zero histogram | K1-W same-state wrong S-box | Full - zero | Exact - wrong |
|---:|---:|---:|---:|---:|---:|
| 3 | `0.508393288` | `0.501185417` | `0.505091190` | `+0.007207870` | `+0.003302097` |
| 4 | `0.528264046` | `0.506880760` | `0.496545792` | `+0.021383286` | `+0.031718254` |

The folded K1-T checkpoints replayed their invariant anchors exactly. Their
exact/zero/wrong-Sbox AUCs were respectively:

```text
seed3 = 0.565424442 / 0.485805035 / 0.499409676
seed4 = 0.594047546 / 0.517103195 / 0.489146233
```

Seed3 passed both weak-contribution checks. Seed4 failed both: its compact
histogram branch already contributes `+0.0214` AUC and uses correct S-box
semantics by `+0.0317`, despite the full model remaining below the K1-T anchor.
Therefore the frozen decision is:

```text
status       = hold
decision     = innovation1_uknit_family_ctspn_k1x_optimization_geometry_not_sufficient
remote_scale = no
```

This verifies the sixteen-fold parameterization effect, but rejects it as a
sufficient explanation for fresh K1-W failure. K1-Y is not authorized.

The valid output root is:

```text
outputs/local_audit/
  i1_uknit_family_ctspn_compact_optimization_geometry_k1x_20260728/
```

The Chinese SVG was rendered at `1944x1056` pixels and passed
`visual-qa-redraw`: no text overlap, clipping, missing glyphs, ambiguous title
or unreadable close values was observed.

## Evidence-backed next action

Run K1-Z as a zero-training branch-scale/interference audit. Freeze the K1-W
checkpoints and K1-Q caches, select one histogram residual multiplier from a
predeclared grid using only each seed's training split, then evaluate that
frozen multiplier on the untouched cross-key validation split under exact and
same-state wrong-Sbox semantics.

If a train-selected multiplier restores the K1-T invariant anchor on both
validation seeds while retaining an exact-minus-wrong-Sbox margin, redesign
the compact model around explicit late fusion or calibrated gating rather than
projection learning rate. If no multiplier restores the signal, discard the
current learned compact aggregator and test a different invariant statistic
encoder. Do not add pairs, samples or remote scale before this distinction.
