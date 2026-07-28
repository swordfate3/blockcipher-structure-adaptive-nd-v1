# Innovation 1 uKNIT-Family CT-SPN Dialga Retention K1-AC

**Date:** 2026-07-29
**Status:** completed / held / semantic attribution failed
**Execution:** local CPU; sub-medium family-retention diagnostic, not formal training

## Research question

K1-AA replaced the compact histogram projection with sixteen fixed virtual
optimizer slots whose parameter shape does not depend on the runtime cipher's
native cell count. K1-AB then showed that the unchanged K1-AA network benefits
from sixteen independent ciphertext pairs on uKNIT r5. Before spending a
remote uKNIT medium-scale slot, K1-AC asks:

> Does the selected K1-AA plus sixteen-pair setting retain Dialga-128 r4's
> existing strong signal, and does the correct S-box become distinguishable
> from an equal-capacity wrong-S-box control on both seeds?

Signal retention and correct-structure attribution are separate gates. Strong
Dialga AUC alone does not establish that the network uses the correct S-box.

## Single route change

The accepted uKNIT K1-AB protocol is moved to Dialga-128. Inside the Dialga
panel the candidate is compared with the frozen four-pair K1-W anchor, but the
four-to-sixteen delta is descriptive rather than a pure pair-count ablation:
K1-AC also uses K1-AA's accepted virtual-slot optimization geometry.

No architecture, S-box control, data rows, difference, keys, negative
definition, epochs, optimizer or metric is tuned after result reveal.

## Frozen matrix

Train exactly four rows:

```text
Dialga-128 prefix-r4 seed0,1 x {K1-AA virtual-slot exact, wrong-Sbox}
```

| Field | Frozen value |
|---|---|
| Train / validation | `2048/class` / `1024/class` |
| Pairs/sample | `16` |
| Input width | `4096` bits |
| Difference | calibrated Dialga default `0x40` |
| Negative definition | encrypted random plaintexts |
| Sample structure | independent pairs |
| Train key | all-zero 256-bit key |
| Validation key | all-`1`-nibble 256-bit key |
| Runtime window | rounds 2/3 feeding the r4 task |
| Virtual projection slots | `16` fixed optimizer slots |
| Trainable parameters | `214316` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| LR / weight decay | `1e-4` / `1e-5` |
| Checkpoint | restored best validation AUC |
| Device | local CPU |

Generate four parameter-matched disk caches under the K1-AC output root: one
train and one validation cache per seed. The second model for each seed must
reuse those exact caches. Dataset generation must emit chunk progress before
training.

## Same-budget anchors

The strongest existing Dialga r4 same-row, same-key, same-epoch K1-W anchors
use four pairs/sample:

```text
seed0 K1-W exact       = 0.960447311
seed1 K1-W exact       = 0.960405350
seed0 K1-W wrong-Sbox  = 0.960926056
seed1 K1-W wrong-Sbox  = 0.958303452
```

K1-N exact anchors are `0.959750175/0.954736710`. Because K1-W is stronger on
both seeds, it is the retention authority. The near-equal K1-W exact and wrong
S-box values are prior evidence that Dialga's signal may survive without
correct nonlinear semantics.

## Readiness gate

Before optimization require:

1. exact K1-AB and K1-W result/gate digests and completed source decisions;
2. exactly four frozen Dialga tasks using strict encrypted-random-plaintext
   negatives, `0x40`, cross-key validation and sixteen pairs;
3. exact and wrong-S-box models share one state geometry and exactly `214316`
   trainable parameters;
4. every model accepts `[B,4096]` and decodes sixteen 256-bit pair records;
5. the virtual tensor remains `[16,128,40]`, ordinary Adam has one group at
   `1e-4`, and no model-specific LR override exists;
6. shared-state exact/wrong-S-box logits are finite and observably different;
7. no cipher identity, native-cell slot parameter or absolute runtime-position
   parameter is introduced.

Any failure blocks optimizer steps. Repair only the failed binding and rerun
the unchanged plan.

## Frozen result gate

For each seed independently require:

```text
K1-AC exact16 >= K1-W exact4 - 0.020
K1-AC exact16 - K1-AC wrong-Sbox16 >= +0.010
```

The first check decides signal retention; the second decides S-box semantic
attribution. No seed averaging may hide a failed check.

## Decisions and executable next action

- **Both gates pass on both seeds:** retain K1-AA plus sixteen pairs as the
  cross-cipher local candidate. Next preregister uKNIT r5 `65536/class`,
  seed3/4, exact/wrong-Sbox remote medium diagnostic with disk-backed cache;
  do not call it formal or paper scale.
- **Signal retains but S-box attribution fails:** hold family transfer. Next
  run a zero-training same-checkpoint K1-AD audit that loads each exact K1-AC
  checkpoint into exact and wrong-Sbox descriptors on the identical validation
  cache. Do not add samples, pairs, epochs, MoE or another network first.
- **Any signal-retention gate fails:** keep sixteen pairs for uKNIT only and
  retain Dialga's four-pair protocol. Audit Dialga pair aggregation before any
  family or remote-scale claim.
- **Protocol invalid:** repair only the failed binding and rerun unchanged.

Blocked in K1-AC: remote launch, more samples, additional pairs/seeds/epochs,
new differences, architecture tuning, MoE, seed averaging and publication or
attack claims.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_dialga_retention_k1ac_16pair_2048_seed0_seed1_20260729
```

Produce preflight, progress JSONL, four disk caches, four checkpoints, four
result rows, validation, gate, summary, comparison/history CSV and a Chinese
SVG. Apply `visual-qa-redraw`, refresh both recent-result indexes, then append
the observed metrics, claim boundary and evidence-backed next action here.

## Completed result

```text
status   = hold
decision = innovation1_uknit_family_ctspn_k1ac_semantic_attribution_failed
protocol = pass; 4/4 rows; no failed protocol checks
remote_scale = no
```

| Seed | Correct S-box, 16 pairs | Wrong S-box, 16 pairs | Correct - wrong | K1-W correct, 4 pairs | 16-pair gain |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.999881744` | `0.999889374` | `-0.000007629` | `0.960447311` | `+0.039434433` |
| 1 | `0.999927521` | `0.999859810` | `+0.000067711` | `0.960405350` | `+0.039522171` |

Both seeds retain and substantially increase Dialga's existing four-pair
signal, but both miss the preregistered `+0.010` correct-minus-wrong-S-box
margin. The result therefore supports only signal retention. It does not show
that the near-perfect Dialga score depends on correct nonlinear semantics.

`validate-results` matched all four plan rows with no errors. All four dataset
caches were created once and reused by the paired control; all four best
checkpoints and ten-epoch histories are present. The Chinese SVG was rendered
at 1800 pixels and passed `visual-qa-redraw`: labels, close values, thresholds,
decision text and next action are readable without overlap, clipping or
missing glyphs.

Artifacts:

```text
outputs/local_diagnostic/i1_uknit_family_ctspn_dialga_retention_k1ac_16pair_2048_seed0_seed1_20260729/
```

## Evidence-backed next action

K1-AD is required before any training or scale change. Load only the two exact
K1-AC best checkpoints, evaluate correct and wrong S-box descriptors on the
identical seed-specific validation caches, and perform no optimizer steps. The
gate remains `+0.010 AUC` per seed, with probability changes reported only as
supporting sensitivity evidence. Remote scale, more pairs/samples/epochs and a
new architecture remain blocked.
