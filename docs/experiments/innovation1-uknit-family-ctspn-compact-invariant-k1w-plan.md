# Innovation 1 uKNIT-Family CT-SPN Compact Invariant K1-W

**Date:** 2026-07-28
**Status:** completed / hold / uKNIT retention and one semantic gate failed
**Execution:** local CPU; sub-medium architecture diagnostic, not formal training

## Research question

K1-U retained strong uKNIT r5 signal and correct-Sbox attribution at remote
`65536/class`, but the position-preserving histogram was slightly worse than
the invariant control on both seeds. K1-V separately showed that sixteen pairs
help the old exact branch locally, but it did not authorize combining pair
count with an architecture change.

K1-W therefore asks one narrow question:

> Can the algebraically redundant sixteen-cell histogram projection be
> replaced by a compact runtime-cell-count invariant projection without losing
> the established uKNIT r5 or Dialga r4 same-budget signal?

## One changed variable

Keep the K1-N exact inverse-Sbox, inverse-GF(2), topology-edge backbone and
bounded residual gates unchanged. Replace only:

```text
old invariant histogram:
  mean cells -> repeat C=16 -> Linear(5*16*8, 128)

K1-W compact histogram:
  mean cells -> Linear(5*8, 128)
```

The compact weight is the sum of the old projection over the repeated cell
axis. Whole-model trainable parameters change from `214316` to `137516`; no
width, depth, auxiliary loss or metadata branch compensates for the reduction.

## Frozen matrix

Train exactly eight independent rows:

```text
uKNIT-BC r5:   seed3,4 x {compact exact, compact wrong-Sbox}
Dialga-128 r4: seed0,1 x {compact exact, compact wrong-Sbox}
```

| Field | Frozen value |
|---|---|
| Train / validation | `2048/class` / `1024/class` |
| Pairs/sample | `4` |
| uKNIT difference | cell11 role1, `0x0000400000000000` |
| Dialga difference | D1 `0x40` |
| Negative definition | encrypted random plaintexts |
| Sample structure | independent pairs |
| Runtime window | uKNIT rounds 3/4; Dialga rounds 2/3 |
| Hidden / pair embedding | `32` / `128` |
| Histogram value dim | `8` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best validation AUC, restored |
| Device | local CPU |

The exact and wrong-Sbox row for one cipher/seed must reuse identical source
train and validation caches. uKNIT caches are bound to K1-Q/K1-T; Dialga caches
are bound to D1. Dataset regeneration is forbidden in K1-W.

## Source authority

Required anchors and activation evidence:

```text
K1-U gate SHA256 = 79a5f3652b8a6125af8c987cb8b1df075fc8e992e73cdb5dc61dedbfbdb6c3ed
K1-T gate SHA256 = f122f43f4d895a1b68fb696bd81df4e1d362880a3a12d9883933c932dd7f0dbf
K1-N gate SHA256 = e2aed925c5d285f2856be791e1f6450b5e338f10e470572844539d86c1134a4f
D1 gate SHA256   = e113227bbd541a3d5c11502793d5ebb5d75108c4f53e157326b5ac509cc10e67
```

K1-U is fallback-retrieved, not a verified result-branch archive. K1-W may use
its checksummed checkpoints and validation caches for local readiness, but must
retain that qualification and cannot upgrade K1-U to publication-style proof.

## Optimizer authorization gate

Before any K1-W training require all of the following:

1. the frozen CSV contains exactly eight rows and only the projection changes;
2. both K1-U invariant checkpoint and validation-cache payload digests match;
3. each old K1-U invariant checkpoint strictly loads into the old model;
4. folding `W[128,5,16,8]` over cells produces the compact `[128,40]` weight;
5. old and folded compact float64 logits agree within `1e-6` over each exact
   retrieved `65536`-row K1-U cross-key validation cache;
6. production-float32 serialized AUC and fixed-threshold accuracy agree within
   `1e-7`, while its raw maximum logit error remains explicitly reported;
7. uKNIT and Dialga exact/wrong-Sbox models have identical state-dict geometry
   and exactly `137516` trainable parameters;
8. legal `[B,512]` and `[B,1024]` inputs produce finite `[B,1]` logits;
9. joint native-cell relabeling changes logits by at most `1e-6`;
10. shared-weight wrong-Sbox controls change deterministic histograms and logits
    for both ciphers;
11. all compact histogram parameters receive finite nonzero gradients;
12. no cipher ID, numeric cell embedding, key, label or active-difference
    metadata enters the model.

Any failure prohibits optimization. Repair only tensor order, the generalized
cell-count guard, checkpoint folding or source binding and rerun unchanged.

### Pre-optimizer numerical amendment

The first full-cache readiness attempt used raw float32 logits for item 5 and
failed before any optimizer step: maximum errors were `4.7684e-6` (seed3) and
`3.8147e-6` (seed4), although AUC and fixed-threshold accuracy were exactly
preserved. A 1024-row float64 audit reduced the maximum error to `1.0564e-8`,
showing that the formula and tensor ordering were correct and the discrepancy
came from changing finite-precision reduction order from 640 terms to 40.

The gate above is therefore clarified before training: float64 checks algebraic
checkpoint equivalence; production float32 checks decision-metric equivalence
and retains its raw logit delta as a diagnostic. The `1e-6` and `1e-7`
tolerances themselves are unchanged. This amendment does not use or respond to
any K1-W training metric.

## Frozen result gate

Use completed cross-key validation anchors:

```text
uKNIT seed3 K1-T invariant = 0.565424442
uKNIT seed4 K1-T invariant = 0.594047546
Dialga seed0 K1-N exact    = 0.959750175
Dialga seed1 K1-N exact    = 0.954736710
```

For each uKNIT seed independently require:

```text
compact exact AUC >= max(0.550, K1-T invariant AUC - 0.020)
compact exact - compact wrong-Sbox >= +0.010
```

For each Dialga seed independently require:

```text
compact exact AUC >= K1-N exact AUC - 0.005
```

Dialga wrong-Sbox AUC separation is descriptive because Dialga uses one Sbox
table across native cells; its shared-state non-degeneracy is enforced in
readiness instead. A failed seed cannot be hidden by averaging.

## Decisions

- **All retention and semantic gates pass:** retain compact invariant as the
  selected uKNIT-family architecture. Next freeze a compact-only 4-pair versus
  16-pair experiment; do not change anything else.
- **uKNIT retention fails:** hold compact optimization and inspect histogram
  gate dynamics/checkpoint initialization; do not restore redundant slots.
- **uKNIT wrong-Sbox attribution fails:** reject the compact semantic branch;
  more pairs or samples cannot substitute for missing semantic attribution.
- **Dialga retention fails:** hold the family claim and audit whether the
  invariant histogram interferes with the strong K1-N Dialga backbone.
- **Protocol invalid:** repair only the failed invariant and rerun unchanged.

Blocked inside K1-W: sixteen pairs, more samples, epochs, seeds, positions or
rounds; remote launch; MoE; DDT/trails; a learned cipher ID; shared-weight or
zero-shot transfer claims; and post-result threshold changes.

## Artifacts

```text
run_id = i1_uknit_family_ctspn_compact_invariant_k1w_2048_seed_panel_20260728
```

Required artifacts: readiness with K1-U fold replay, source/cache manifest,
progress JSONL, eight best checkpoints, eight result rows, validation, gate,
summary, comparison CSV, history CSV, Chinese SVG, plot report and
`visual_qa_passed.marker`. On completion update this record with exact metrics,
claim scope and the evidence-backed next action, then refresh both recent-result
indexes.

## Completed readiness

The optimizer gate passed before training. All eight K1-T/D1 source caches
matched their frozen dataset digests, all four compact model instances had
identical state geometry and exactly `137516` trainable parameters, and uKNIT
and Dialga joint-cell relabel errors were below `1.5e-7`.

The two retrieved K1-U invariant checkpoints were folded and replayed on each
complete `65536`-row cross-key validation cache:

| Seed | Float64 max logit error | Float32 old/compact AUC | Float32 accuracy |
|---:|---:|---:|---:|
| 3 | `7.1054e-15` | `0.977200513 / 0.977200518` | `0.967559814 / 0.967559814` |
| 4 | `8.8818e-15` | `0.974682369 / 0.974682372` | `0.968276978 / 0.968276978` |

This proves representational equivalence of the fold. It does not prove that
the smaller parameterization will optimize equally from a fresh initialization.

## Completed training result

All eight ten-epoch rows completed with best checkpoints restored and sixteen
exact source-cache reuse events. No dataset was generated or changed.

| Cipher / seed | Compact exact AUC | Compact wrong-Sbox | Historical anchor | Exact - anchor | Exact - wrong-Sbox |
|---|---:|---:|---:|---:|---:|
| uKNIT r5 seed3 | `0.508393288` | `0.501296997` | `0.565424442` | `-0.057031155` | `+0.007096291` |
| uKNIT r5 seed4 | `0.528264046` | `0.505918026` | `0.594047546` | `-0.065783501` | `+0.022346020` |
| Dialga r4 seed0 | `0.960447311` | `0.960926056` | `0.959750175` | `+0.000697136` | `-0.000478745` |
| Dialga r4 seed1 | `0.960405350` | `0.958303452` | `0.954736710` | `+0.005668640` | `+0.002101898` |

All six protocol checks passed. Dialga retained the K1-N anchor on both seeds.
uKNIT failed the retention threshold on both seeds, and seed3 also missed the
`+0.010` wrong-Sbox margin. The frozen result decision is therefore:

```text
status       = hold
decision     = innovation1_uknit_family_ctspn_k1w_semantic_attribution_failed
remote_scale = no
```

The valid output root is:

```text
outputs/local_diagnostic/
  i1_uknit_family_ctspn_compact_invariant_k1w_2048_seed_panel_20260728/
```

`validate-results` passed with `8/8` plan-aligned rows. The Chinese SVG was
rendered to `1944x1056` pixels and passed `visual-qa-redraw` with no overlap,
clipping, missing glyphs, ambiguous titles or unreadable close-value marks.

## Evidence-backed next action

Do not run the previously conditional compact 4-pair versus 16-pair experiment:
K1-W did not select the compact architecture. First run K1-X as an inference-
and-gradient-only optimization-geometry audit on the restored K1-W and folded
K1-T checkpoints.

The exact hypothesis is that the old repeated invariant projection supplies a
sixteen-fold effective update to its folded weight. If every old cell-slot
weight receives the same gradient `g`, then the effective folded weight changes
by `16 * learning_rate * g`; the compact weight changes by only
`learning_rate * g`. K1-X must verify this ratio on both uKNIT seeds, measure
the restored histogram gate and full-minus-zero-histogram AUC, and compare
same-checkpoint exact versus wrong-Sbox semantics on the unchanged K1-T caches.

Only if the sixteen-fold gradient/update relation is exact and the failed
compact checkpoints show weak histogram contribution may a separate K1-Y
training plan change one variable: the compact projection-weight update scale.
Keep data, four pairs, architecture, loss, all other learning rates, epochs and
controls fixed. If K1-X contradicts the mechanism, hold this compact route and
return to a different invariant aggregator rather than tuning learning rates.

### K1-X adjudication

K1-X subsequently verified an exact `16.0x` folded gradient ratio on both
seeds, but seed4's failed K1-W checkpoint did not have a weak histogram branch:
closing it reduced AUC by `0.021383286`, and a same-state wrong S-box reduced
AUC by `0.031718254`. The optimization geometry is therefore real but not a
sufficient explanation. K1-Y is not authorized. The next bounded action is
K1-Z, a train-split-selected and cross-key-confirmed inference-scale audit that
separates branch-fusion interference from an inadequate compact aggregator.
