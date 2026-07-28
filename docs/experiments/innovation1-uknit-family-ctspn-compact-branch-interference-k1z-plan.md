# Innovation 1 uKNIT-Family CT-SPN Compact Branch Interference K1-Z

**Date:** 2026-07-28
**Status:** completed / hold / inference rescaling insufficient
**Execution:** local CPU; frozen-checkpoint inference only

## Research question

K1-X verified the redundant projection's sixteen-fold effective gradient, but
seed4 already had a semantically active compact histogram branch and still
failed to retain the K1-T invariant anchor. K1-Y learning-rate scaling is
therefore not authorized.

K1-Z asks:

> Does the learned compact histogram representation contain recoverable uKNIT
> signal that is suppressed or destructively fused at its learned residual
> scale, or is the compact representation itself inadequate?

## Frozen sources and one intervention

Reuse the exact K1-X-bound K1-W seed3/4 compact checkpoints and K1-Q train and
cross-key validation caches. Strictly load identical learned tensors into the
correct-Sbox and wrong-Sbox runtime controls.

Change only the multiplier applied to the learned histogram residual before it
is added to the unchanged base-plus-edge embedding:

```text
combined = base_plus_edge + alpha * learned_histogram_gate * tanh(histogram)
```

Freeze the candidate grid before any K1-Z metric is observed:

```text
alpha = {-4, -2, -1, -0.5, 0, 0.25, 0.5, 1, 2, 4, 8, 16}
```

Negative values diagnose a learned sign error. `alpha=0` is the K1-X
zero-histogram intervention and `alpha=1` must exactly replay K1-W.

## Discovery and confirmation

For each seed independently:

1. compute exact-Sbox AUC for every alpha on the frozen 4096-row training
   cache;
2. select the highest training AUC, breaking ties by smallest `abs(alpha-1)`
   and then smallest numeric alpha;
3. freeze that alpha;
4. evaluate only the frozen alpha, plus descriptive `alpha=0` and `alpha=1`
   anchors, on the untouched 2048-row cross-key validation cache;
5. evaluate the frozen alpha with the wrong S-box using the identical learned
   state.

Cache the expensive base, edge and histogram embeddings once per
seed/split/semantic condition. Do not train, mutate checkpoints, regenerate
data or use validation AUC to select alpha.

## Frozen gate

Protocol requires exact source digests, two independent train-selected alphas,
`alpha=1` source-AUC replay within `1e-6`, complete finite rows, unchanged state
hashes and `optimizer_steps=0`.

For both seeds independently require:

```text
selected validation exact AUC >= K1-T invariant anchor - 0.020
selected validation exact - selected validation wrong-Sbox AUC >= +0.010
selected alpha is also the preregistered train-split optimum
```

The K1-T invariant anchors remain:

```text
seed3 = 0.565424442
seed4 = 0.594047546
```

## Decisions

- **Both seeds pass:** the compact representation contains recoverable signal,
  but early residual fusion/gate calibration is inadequate. Next compare one
  explicit late-logit fusion design against unchanged K1-W at the same local
  budget; do not tune projection learning rate.
- **A seed fails retention while the exact-versus-wrong-Sbox margin survives:**
  inference rescaling is insufficient, but this does not prove representation
  incapacity because folded K1-T already establishes forward attainability.
  Combine the K1-X gradient identity and K1-Z no-rescue result to preregister a
  projection-optimization intervention; do not change the statistic encoder.
- **A seed loses semantic attribution at every train-selected scale:** hold the
  current learned compact histogram branch and audit the statistic encoder
  before any optimization intervention.
- **Protocol invalid:** repair only source binding, cached-forward equivalence,
  alpha selection or state restoration and rerun unchanged.

Blocked: optimizer steps, learned alpha, validation-selected alpha, sixteen
pairs, more samples/epochs/seeds, remote scale, new differences, MoE, cipher
identity, and averaging seeds to hide a failed confirmation.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_compact_branch_interference_k1z_20260728
```

Produce source bindings, progress JSONL, alpha-grid JSONL, frozen confirmation
JSONL, gate, summary, CSV and Chinese SVG. Apply `visual-qa-redraw`, refresh the
recent-result indexes and append the evidence-backed next action here.

## Interpretation guard

Failure of post-hoc inference rescaling cannot establish that a representation
is incapable of expressing the target. K1-W already folded the trained K1-T
invariant state into the identical compact forward representation and replayed
its AUC exactly. K1-Z can distinguish recoverable gate-scale error from a need
to change learned projection weights, but it cannot distinguish optimization
failure from representation incapacity by itself.

## Completed result

All source, alpha-grid, train-only selection, source-AUC replay, state-integrity
and zero-step checks passed. The train-selected multipliers and untouched
cross-key confirmation were:

| Seed | Selected alpha | Train AUC | Validation exact | Validation wrong S-box | Exact - wrong | K1-T anchor | Exact - anchor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | `4x` | `0.563325524` | `0.525269508` | `0.512797356` | `+0.012472153` | `0.565424442` | `-0.040154934` |
| 4 | `2x` | `0.633269906` | `0.541595459` | `0.492598057` | `+0.048997402` | `0.594047546` | `-0.052452087` |

Both seeds retained the frozen `+0.010` correct-Sbox attribution margin, but
neither reached its anchor-retention threshold (`0.545424442` and
`0.574047546`). The decision is:

```text
status       = hold
decision     = innovation1_uknit_family_ctspn_k1z_inference_rescaling_insufficient_optimization_unresolved
remote_scale = no
```

The valid result root is the corrected clean adjudication:

```text
outputs/local_audit/
  i1_uknit_family_ctspn_compact_branch_interference_k1z_20260728_clean/
```

The Chinese SVG was rendered at `1944x1056` and passed
`visual-qa-redraw` without overlap, clipping, missing glyphs, ambiguous titles
or unreadable curves.

## Evidence-backed next action

K1-Z rules out histogram-gate magnitude as a sufficient repair. It does not
rule out projection optimization: K1-X proved a `16x` effective gradient
geometry change, while folded K1-T constructively proves that the same compact
forward representation can attain the anchors.

Preregister K1-Y as a local same-budget diagnostic changing only the Adam
learning rate for `backbone.histogram_projection.0.weight` from `1e-4` to
`1.6e-3`. Keep every other parameter at `1e-4`, retain weight decay, four
pairs, K1-Q caches, seeds3/4, ten epochs, MSE, strict negatives and exact versus
wrong-Sbox controls. Do not add pairs or remote scale unless K1-Y independently
retains both anchors and semantic margins.
