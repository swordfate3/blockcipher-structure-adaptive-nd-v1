# Innovation 1 uKNIT-Family CT-SPN Virtual-Slot Projection K1-AA

**Date:** 2026-07-28
**Status:** completed / pass / virtual-slot optimizer geometry retained
**Execution:** local CPU; sub-medium mechanism diagnostic, not formal training

## Research question

K1-X proved that the old sixteen-native-slot invariant projection and the
compact K1-W projection implement the same forward function after folding, but
the old parameterization supplies a sixteen-fold effective projection update.
K1-Y restored that update approximately by assigning `16x` learning rate to one
compact weight. It improved uKNIT r5 AUC by `+0.0405/+0.0656`, retained correct
S-box attribution on both seeds, and nearly restored the K1-T anchors, but
seed3 missed the frozen `0.550` floor by `0.0011`.

K1-AA asks:

> Can a fixed sixteen-virtual-slot projection recover stable uKNIT r5 anchor
> retention with ordinary Adam, while keeping the model parameter geometry
> independent of the runtime cipher's actual cell count?

## One changed variable

Retain the exact K1-W compact cell-invariant histogram and replace only its
first compact projection:

```text
K1-W/K1-Y:
  weight [128, 40]

K1-AA:
  virtual_slot_weights [16, 128, 40]
  effective_weight = sum(virtual_slot_weights, dim=0)
  output = Linear(input[40], effective_weight[128,40], shared_bias[128])
```

The sixteen slots are fixed architectural optimizer slots, not runtime native
cell positions. They are never indexed by cipher ID, cell ID or descriptor cell
count. Therefore uKNIT, Dialga and future SPNs use the same parameter shapes.
Each slot has its own Adam state; the forward function remains one compact
linear projection. The special K1-Y learning-rate multiplier is removed and all
parameters use Adam `1e-4`.

The virtual weights and shared bias use the same `fan_in=16*40`
initialization scale as the old K1-T projection. Total trainable parameters must
be exactly `214316`, matching K1-T.

## Frozen matrix

Train exactly four rows:

```text
uKNIT-BC r5 seed3,4 x {virtual-slot exact, virtual-slot wrong-Sbox}
```

| Field | Frozen value |
|---|---|
| Train / validation | `2048/class` / `1024/class` |
| Pairs/sample | `4` |
| Difference | cell11 role1, `0x0000400000000000` |
| Negative definition | encrypted random plaintexts |
| Sample structure | independent pairs |
| Train / validation keys | same K1-Q/K1-W/K1-Y fixed cross-key protocol |
| Runtime window | rounds 3/4 |
| Hidden / pair embedding | `32` / `128` |
| Histogram value dim | `8` |
| Virtual slots | `16` fixed optimizer slots |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| LR / weight decay | `1e-4` / `1e-5` for every parameter |
| Scheduler | none |
| Checkpoint | restored best validation AUC |
| Device | local CPU |

Reuse the exact K1-Q-bound K1-W/K1-Y train and validation caches through
read-only directory links. Dataset generation is forbidden.

## Readiness gate

Before optimization require:

1. the corrected K1-Z and completed K1-Y source decisions and frozen digests;
2. exactly four plan rows matching the matrix above;
3. exact and wrong-S-box models share one state geometry and `214316` trainable
   parameters;
4. the sole virtual tensor is exactly
   `backbone.histogram_projection.0.virtual_slot_weights`, shape
   `[16,128,40]`, with shared bias `[128]`;
5. summing virtual slots and loading the effective weight into a compact K1-W
   model preserves logits within `1e-7`;
6. a zero-training gradient audit proves every virtual slot receives the same
   data gradient within `1e-9`, and their summed effective gradient is exactly
   `16x` the compact gradient within `1e-9`;
7. the optimizer has one ordinary parameter group at `1e-4`; no declarative
   parameter LR multiplier or scheduler is present;
8. no cipher ID, absolute cell/bit identity, runtime native-cell parameter slot
   or dataset change is present.

Any failure blocks optimizer steps. Repair only the plan binding, virtual
projection implementation, metadata or source-cache binding and rerun
unchanged.

## Frozen result gate

Historical K1-T invariant anchors:

```text
seed3 = 0.565424442
seed4 = 0.594047546
```

Same-budget K1-Y exact anchors:

```text
seed3 = 0.548890114
seed4 = 0.593873501
```

For each seed independently require:

```text
K1-AA exact AUC >= max(0.550, K1-T - 0.010, K1-Y - 0.005)
K1-AA exact - K1-AA wrong-Sbox AUC >= +0.010
```

The resulting retention thresholds are `0.555424442` for seed3 and
`0.588873501` for seed4. No averaging may hide a failed seed.

## Decisions

- **All four checks pass:** retain K1-AA as the compact family architecture.
  Next run one separate local same-budget `4 -> 16 pairs/sample` comparison
  inside K1-AA before any remote scale.
- **Retention fails but semantic margins pass:** virtual optimizer geometry is
  insufficiently stable. Audit initialization/checkpoint dynamics without
  changing pairs, data, epochs, seed panel or learning rate.
- **Semantic margin fails:** reject K1-AA; redundant optimizer slots do not
  recover correct S-box attribution.
- **Protocol invalid:** repair only the failed binding and rerun unchanged.

Blocked: sixteen pairs in this run, remote launch, other virtual-slot counts,
learning-rate tuning, more epochs/data/seeds, Dialga rows, MoE, new differences
and post-result threshold changes.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_virtual_slot_projection_k1aa_2048_seed3_seed4_20260728
```

Produce preflight, source-cache manifest, progress JSONL, four checkpoints,
four result rows, validation, gate, summary, comparison CSV, history CSV and a
Chinese SVG. Apply `visual-qa-redraw`, refresh both recent-result indexes, and
append the evidence-backed next action here before reporting completion.

## Completed readiness

All readiness checks passed before optimization:

```text
effective compact forward max error = 0.0 on seed3 and seed4
slot gradient relative error        = 0.0 on seed3 and seed4
effective gradient ratio            = 16.0 on seed3 and seed4
effective ratio error               <= 2.28e-16
trainable parameters                = 214316
optimizer groups                    = one ordinary Adam group at 1e-4
source cache digests                = 4/4 exact
```

No dataset generation occurred. The four K1-Q train/validation caches were
linked read-only and reused exactly eight times.

## Completed result

All four rows completed ten epochs and restored the best validation-AUC
checkpoint:

| Seed | K1-AA correct | K1-AA wrong S-box | K1-T anchor | K1-Y anchor | Correct - wrong | Correct - K1-T | Correct - K1-Y |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | `0.570869923` | `0.503264427` | `0.565424442` | `0.548890114` | `+0.067605495` | `+0.005445480` | `+0.021979809` |
| 4 | `0.590953827` | `0.511684895` | `0.594047546` | `0.593873501` | `+0.079268932` | `-0.003093719` | `-0.002919674` |

Both seeds passed their independent retention thresholds
`0.555424442/0.588873501` and the `+0.010` correct-S-box attribution margin.
The frozen decision is:

```text
status       = pass
decision     = innovation1_uknit_family_ctspn_k1aa_virtual_slots_supported
remote_scale = no
```

This result supports the fixed virtual-slot parameterization as a local
mechanism. It does not establish formal scale, attack performance, SOTA,
cross-cipher transfer or a universal SPN model.

The valid output root is:

```text
outputs/local_diagnostic/
  i1_uknit_family_ctspn_virtual_slot_projection_k1aa_2048_seed3_seed4_20260728/
```

`validate-results` passed `4/4`. The Chinese SVG was rendered to
`1944x1056` pixels and passed `visual-qa-redraw`: no text overlap, clipping,
missing glyphs, ambiguous title or unreadable close values was observed.

## Evidence-backed next action

Retain K1-AA unchanged and preregister K1-AB as a pair-count-only diagnostic.
Use uKNIT r5 cell11, seeds3/4, exact and wrong-S-box controls, `2048/class`
train, `1024/class` cross-key validation, ten epochs and ordinary Adam `1e-4`;
change only `pairs/sample: 4 -> 16`.

Use the completed K1-AA four-pair rows above as the same-architecture anchor.
Advance only if both seeds keep at least `+0.010` correct-S-box attribution and
the sixteen-pair exact branch improves its four-pair AUC by at least `+0.010`.
Otherwise retain four pairs. K1-AB remains a local diagnostic; do not combine
it with more data, new differences, Dialga, remote scale or architectural
changes.
