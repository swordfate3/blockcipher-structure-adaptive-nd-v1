# Innovation 1 K1-AP Multi-Cipher Gradient-Conflict Audit

**Status:** completed / pass / gradient-normalization route opened
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_multicipher_gradient_conflict_k1ap_64batch_replica0_replica1_20260729`

## 1. Research Question

K1-AO made correct S-box semantics identifiable in all 12 fresh-data panels,
but uKNIT and Midori lost `0.053-0.086` AUC relative to their independent
anchors while Dialga retained or exceeded its anchor. The next question is:

> Does one equal-batch shared optimizer receive systematically conflicting or
> severely imbalanced gradients from uKNIT-BC, Midori64 and Dialga-128?

This audit does not test another architecture or train another model. It
diagnoses the two already selected K1-AO checkpoints before any optimizer
change is considered.

## 2. Frozen Authority

The source is the completed K1-AO local diagnostic. Bind and rehash its gate,
validation, dataset manifest, checkpoint manifest, both checkpoints and the
training config. The source must retain these facts:

```text
status                       = hold
protocol checks              = 9/9 pass
evaluation rows              = 36
optimizer steps per replica  = 1920
wrong-S-box panels passed    = 12/12
retention panels passed      = 4/12
branch-off panels passed     = 11/12
```

Reload each state dict strictly and require its tensor hash to match the source
checkpoint manifest before measuring a gradient.

## 3. Data And Batch Contract

Reuse only each replica's K1-AO `train_seen` cache:

| Cipher | Replica 0 seed | Replica 1 seed | Rows | Pairs |
|---|---:|---:|---:|---:|
| uKNIT-BC r5 | 3 | 4 | 2048/class | 4 |
| Midori64 r4 | 6 | 7 | 2048/class | 4 |
| Dialga-128 r4 | 0 | 1 | 2048/class | 4 |

For each cipher and replica, deterministically shuffle the 2048 positive and
2048 negative indices separately. Batch `i` contains 32 unused positives and
32 unused negatives. Matching batch indices from the three ciphers form one
triplet, producing 64 non-overlapping triplets and using every row once.

This stratification removes accidental class-ratio variation as an explanation
for gradient direction or norm differences. No data are generated and no
labels, negatives, differences, keys or pairs change.

## 4. Same-Checkpoint Conditions

At every batch triplet, measure three runtime conditions on identical rows and
the same immutable checkpoint:

```text
correct_runtime
wrong_sbox_same_checkpoint
transition_branch_off_same_checkpoint
```

Use the unchanged K1-AO loss:

```text
MSE(sigmoid(logit), binary label)
```

Call automatic differentiation only to read gradients. Do not construct an
optimizer and do not update parameters, buffers or persistent runtime state.
Hash the model state before and after all measurements.

## 5. Parameter Groups And Metrics

Measure two preregistered parameter groups:

- `all_trainable`: every K1-AO trainable tensor in canonical named-parameter
  order.
- `transition_semantic`: `transition_gate`, `sbox_transition_encoder` and
  `transition_projection`, the narrow branch that directly represents the
  runtime S-box transition semantics.

For every condition, replica, parameter group, batch triplet and cipher pair,
record gradient cosine. For each cipher, also record gradient norm. Aggregate:

```text
median pairwise cosine
negative-cosine frequency
median gradient norm per cipher
maximum/minimum median norm ratio
```

The same-budget anchor is the correct-runtime gradient geometry. Wrong S-box
and branch-off are same-state attribution controls: they show whether conflict
is tied to semantic/runtime processing or exists in the shared base path.

## 6. Frozen Decision Gate

A cipher pair has systematic conflict in one replica when both hold:

```text
median correct-runtime cosine <= -0.05
negative-cosine frequency     >= 0.50
```

The conflict-aware optimizer route opens only when the same cipher pair meets
both clauses in both replicas. A separate stable imbalance route opens only
when both replicas have a maximum/minimum median gradient-norm ratio `>= 4.0`
with the same dominant cipher.

- **Systematic conflict:** next compare one minimal gradient-combination rule
  against unchanged K1-AO. Keep model, data, pairs, epochs, seeds, order and
  controls fixed. PCGrad is eligible only for conflicting gradients; do not add
  experts or cipher-specific parameters.
- **Stable norm imbalance only:** next compare one loss/gradient normalization
  rule, not PCGrad and not MoE.
- **Neither:** optimizer competition is not established. Return to the
  transition representation, focusing on the Midori branch-off failure.
- **Protocol failure:** repair only the exact binding, stratification, state
  immutability or row-count defect and rerun without interpreting gradients.

Control conditions are explanatory and cannot rescue a missing correct-runtime
trigger.

## 7. Required Artifacts

Write under `outputs/local_audit/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
checkpoint_manifest.json
gradient_pairs.jsonl
gradient_norms.jsonl
results.jsonl
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
```

The Chinese figure must display pairwise cosine distributions or aggregates,
negative-frequency gates and per-cipher norm ratios without clipped labels or
misleading scales, then pass `visual-qa-redraw`. Refresh both recent-result
indexes after completion.

## 8. Prohibited Mechanical Scale-Up

Do not launch 16 pairs, larger data, more epochs, a wider model or a remote GPU
job from K1-AO's absolute AUC alone. Do not introduce MoE, cipher IDs,
per-cipher heads, adapters or experts. K1-AP changes no trained component; its
sole purpose is to choose between optimizer conflict and representation failure
using the evidence already paid for.

## 9. Completed Result

All seven protocol checks passed:

```text
pairwise-gradient rows       = 2304 / 2304
gradient-norm rows           = 2304 / 2304
aggregate result rows        = 72 / 72
optimizer steps              = 0
checkpoint state mutation    = none
parameter .grad persistence  = none
```

Correct-runtime all-parameter gradient direction was not stable across
replicas:

| Cipher pair | Replica | Median cosine | Negative frequency | Conflict gate |
|---|---:|---:|---:|---|
| uKNIT / Midori | 0 | +0.3027 | 29.69% | fail |
| uKNIT / Dialga | 0 | -0.5266 | 87.50% | pass |
| Midori / Dialga | 0 | -0.0844 | 60.94% | pass |
| uKNIT / Midori | 1 | +0.5657 | 15.62% | fail |
| uKNIT / Dialga | 1 | +0.7515 | 3.12% | fail |
| Midori / Dialga | 1 | +0.6537 | 18.75% | fail |

Replica0 therefore had real conflicting batches, including a strong
uKNIT/Dialga conflict, but replica1 was broadly aligned. No cipher pair met the
preregistered conflict gate in both replicas, so K1-AP does not support PCGrad
as the next controlled change.

Correct-runtime all-parameter gradient magnitudes were stable in a different
way:

| Replica | uKNIT median norm | Midori median norm | Dialga median norm | Max/min |
|---:|---:|---:|---:|---:|
| 0 | 2.9819 | 1.1867 | 5.0818 | 4.2824x |
| 1 | 4.3892 | 1.5310 | 9.2209 | 6.0228x |

Dialga was the dominant cipher in both replicas and both max/min ratios exceeded
the frozen `4.0x` gate. The final decision is therefore:

```text
status   = pass
decision = innovation1_uknit_family_k1ap_stable_gradient_norm_imbalance_supported
```

This identifies a plausible optimization mechanism for K1-AO's observed
pattern: the task that retained its AUC also supplied the largest raw gradients,
while Midori supplied the smallest. It is diagnostic causality evidence only;
the audit does not prove that normalization will recover AUC.

The Chinese `curves.svg` was rendered at `2136 x 1296` pixels and passed the
second `visual-qa-redraw` inspection after clipped pair labels and threshold
legend occlusion were repaired.

## 10. Executable Next Action: K1-AQ

Compare exactly one candidate against K1-AO: fixed inverse-norm loss scaling.
For each replica, derive one fixed scale per cipher from the K1-AP
correct-runtime all-parameter median norms. Normalize the three inverse norms
to geometric mean one. Keep the original K1-AO sequential batch order, one Adam
step per cipher batch and exactly `1920` steps per replica.

```text
question       = does correcting stable gradient magnitude imbalance recover
                 uKNIT/Midori without destroying S-box semantics or Dialga?
baseline       = frozen K1-AO result and its 36 same-checkpoint rows
one variable   = multiply each cipher loss by its frozen inverse-norm scale
data/pairs     = unchanged 2048/class/cipher, 4 pairs
seeds          = unchanged replicas 0/1 and dataset seeds
epochs/steps   = unchanged 10 epochs, 1920 Adam steps per replica
controls       = correct runtime, wrong S-box, branch off at the same checkpoint
execution      = local diagnostic; remote scale remains no
```

The candidate advances only if it improves the uKNIT/Midori correct-runtime AUC
by at least `+0.010` in at least six of their eight fresh panels, does not reduce
any uKNIT/Midori panel or any Dialga panel by more than `0.010`, preserves the
wrong-S-box margin in all 12 panels, and does not worsen the existing `11/12`
branch-off count. Full support still requires the original K1-AO retention and
semantic gates in all panels. Otherwise stop normalization and return to the
transition representation. Do not add PCGrad, dynamic weighting, experts, more
pairs, larger data, more epochs or remote GPUs in the same experiment.
