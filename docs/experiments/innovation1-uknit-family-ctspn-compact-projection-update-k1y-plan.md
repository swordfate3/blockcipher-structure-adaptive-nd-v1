# Innovation 1 uKNIT-Family CT-SPN Compact Projection Update K1-Y

**Date:** 2026-07-28
**Status:** completed / hold / strong improvement with seed3 floor near-miss
**Execution:** local CPU; sub-medium diagnostic, not formal training

## Research question

K1-X proved that folding the old sixteen-slot projection gives an exact
`16x` effective projection-weight gradient, while the compact weight receives
one copy. K1-Z showed that rescaling the already learned residual improves AUC
but cannot restore the K1-T anchors. Folded K1-T nevertheless proves that the
identical compact forward representation can attain those anchors.

K1-Y asks:

> Can restoring the missing sixteen-fold optimizer update for the compact
> first projection weight recover uKNIT r5 anchor retention and correct-Sbox
> attribution without changing the representation, data or training budget?

## One changed variable

Use the exact K1-W compact model and change only the Adam parameter-group
learning rate for:

```text
backbone.histogram_projection.0.weight
```

from `1e-4` to `1.6e-3` (`16x`). Every other parameter, including the
projection bias, value encoder, layer norm, residual gates, base model and
classifier, remains at `1e-4`. Weight decay remains `1e-5` for every group.

The candidate must expose the parameter name and multiplier in model metadata.
Readiness must prove that the K1-Y and K1-W state dictionaries have identical
geometry, parameter count and forward logits after loading the same state.

## Frozen matrix

Train exactly four rows:

```text
uKNIT-BC r5 seed3,4 x {compact exact, compact wrong-Sbox}
```

| Field | Frozen value |
|---|---|
| Train / validation | `2048/class` / `1024/class` |
| Pairs/sample | `4` |
| Difference | cell11 role1, `0x0000400000000000` |
| Negative definition | encrypted random plaintexts |
| Sample structure | independent pairs |
| Runtime window | rounds 3/4 |
| Hidden / pair embedding | `32` / `128` |
| Histogram value dim | `8` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Base LR / projection-weight LR | `1e-4` / `1.6e-3` |
| Weight decay | `1e-5` |
| Checkpoint | restored best validation AUC |
| Device | local CPU |

Reuse the exact K1-Q-bound K1-W train and validation caches through read-only
directory links. Dataset generation is forbidden.

## Readiness gate

Before optimization require:

1. exact K1-X and corrected K1-Z source decisions and digests;
2. exactly four frozen K1-Y tasks and source-cache digests;
3. K1-Y and K1-W state geometry and parameter count are identical;
4. loading one state into both produces logits equal within `1e-7`;
5. the optimizer has exactly one named accelerated parameter and one default
   group, with learning rates `1.6e-3` and `1e-4` respectively;
6. the accelerated tensor is exactly
   `backbone.histogram_projection.0.weight`, shape `[128,40]`;
7. no scheduler, optimizer-state carry, cipher ID, numeric cell embedding or
   dataset change is present.

Any failure blocks optimizer steps. Repair only plan binding, parameter-group
construction, model metadata or source-cache binding and rerun unchanged.

## Frozen result gate

Same-budget K1-W exact anchors:

```text
seed3 = 0.508393288
seed4 = 0.528264046
```

K1-T invariant anchors:

```text
seed3 = 0.565424442
seed4 = 0.594047546
```

For each seed independently require:

```text
K1-Y exact AUC >= max(0.550, K1-T invariant AUC - 0.020)
K1-Y exact - K1-Y wrong-Sbox AUC >= +0.010
K1-Y exact - K1-W exact AUC >= +0.020
```

No averaging may hide a failed seed.

## Decisions

- **All six checks pass:** retain the compact architecture with explicit
  projection optimization. Next compare four versus sixteen pairs inside this
  selected architecture at the same local scale before any remote launch.
- **Anchor retention fails:** a `16x` update is insufficient. Stop optimizer
  multiplier tuning and redesign the parameterization, such as a normalized
  sum-of-experts projection whose forward and backward geometry are both
  runtime-cell-count stable.
- **Semantic margin fails:** reject the optimization route; faster projection
  learning cannot substitute for correct S-box attribution.
- **Protocol invalid:** repair only the failed binding and rerun unchanged.

Blocked: other multipliers, learning-rate sweeps, projection-bias scaling,
more epochs/data/seeds, sixteen pairs, remote launch, MoE, learned cipher ID,
new differences and post-result threshold changes.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_compact_projection_update_k1y_2048_seed3_seed4_20260728
```

Produce readiness, cache/source manifest, progress JSONL, four checkpoints,
four result rows, validation, gate, summary, CSV, history and Chinese SVG.
Apply `visual-qa-redraw`, refresh both recent-result indexes and append the
evidence-backed next action here.

## Completed readiness

All thirteen readiness checks passed. K1-W and K1-Y produced identical logits
after loading one state (`max error = 0` on both seeds), state geometry and
parameter count remained identical, and every optimizer contained exactly:

```text
default group:    132396 parameters, lr = 0.0001
projection group:   5120 parameters, lr = 0.0016
```

The accelerated tensor was exactly
`backbone.histogram_projection.0.weight` with shape `[128,40]`. Four K1-Q
source-cache digests matched and optimization was authorized.

## Completed result

All four ten-epoch rows completed, restored their best validation-AUC
checkpoints and recorded exactly eight source-cache reuse events with no cache
generation.

| Seed | K1-Y exact | K1-Y wrong S-box | K1-W exact | K1-T invariant | Exact - wrong | Exact - K1-W | Exact - K1-T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | `0.548890114` | `0.503664970` | `0.508393288` | `0.565424442` | `+0.045225143` | `+0.040496826` | `-0.016534328` |
| 4 | `0.593873501` | `0.506521702` | `0.528264046` | `0.594047546` | `+0.087351799` | `+0.065609455` | `-0.000174046` |

Both seeds passed the `+0.020` K1-W improvement and `+0.010` correct-Sbox
attribution gates. Seed4 retained the K1-T anchor. Seed3 was within the
`K1-T - 0.020` tolerance but missed the separate `0.550` floor by
`0.001109886`. The frozen decision is therefore:

```text
status       = hold
decision     = innovation1_uknit_family_ctspn_k1y_anchor_retention_failed
remote_scale = no
```

This is a local `2048/class` diagnostic, not a formal failure or ceiling. It
shows that restoring update magnitude is a strong mechanism, but not yet
stable enough to authorize sixteen pairs or remote scale.

The valid output root is:

```text
outputs/local_diagnostic/
  i1_uknit_family_ctspn_compact_projection_update_k1y_2048_seed3_seed4_20260728/
```

The Chinese SVG was rendered at `1944x1056` and passed
`visual-qa-redraw` without text overlap, clipping, missing glyphs, ambiguous
titles or unreadable close values.

## Evidence-backed next action

Stop learning-rate multiplier tuning. K1-Y nearly restored both anchors, but a
fixed `16x` optimizer override is an implementation-specific approximation to
the old redundant geometry rather than a clean family architecture.

Preregister K1-AA to change only the compact projection parameterization:

```text
input: cell-invariant [5,8] encoded histogram
weight: 16 fixed virtual slots x [128,40]
forward: Linear(x, sum(virtual_slot_weights), shared_bias)
```

This keeps a fixed parameter shape for any runtime cipher cell count, exactly
restores sixteen independent Adam states and the folded effective update, and
uses no cipher ID or runtime position. Its total parameter count returns to the
old `214316` without repeating data by native cell. Compare exact and
wrong-Sbox semantics on the same K1-Q caches, seeds, four pairs, ten epochs and
base learning rate. Do not combine it with sixteen pairs or remote scale.
