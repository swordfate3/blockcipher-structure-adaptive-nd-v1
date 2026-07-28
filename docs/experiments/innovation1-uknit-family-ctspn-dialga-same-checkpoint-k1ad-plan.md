# Innovation 1 uKNIT-Family CT-SPN Dialga Same-Checkpoint K1-AD

**Date:** 2026-07-29
**Status:** completed / held / discriminative S-box use failed
**Execution:** local CPU, inference only; no optimizer steps and no new data

## Research question

K1-AC already has a seed0 result in which independently trained correct- and
wrong-S-box models both reach almost perfect Dialga-128 r4 AUC. That comparison
cannot determine whether the correct checkpoint itself uses the supplied
S-box, because the wrong model may have learned a separate shortcut.

K1-AD asks one narrower causal question:

> When each completed K1-AC correct-S-box best checkpoint is frozen, does
> replacing only its runtime S-box descriptor with the wrong-S-box descriptor
> reduce AUC on the identical validation cache?

## Frozen source and single intervention

Source run:

```text
i1_uknit_family_ctspn_dialga_retention_k1ac_16pair_2048_seed0_seed1_20260729
```

K1-AD may start only after K1-AC has four valid training rows and the persisted
gate exactly matches a recomputation from the source plan, results, progress
and preflight. The required source decision is:

```text
innovation1_uknit_family_ctspn_k1ac_semantic_attribution_failed
```

For each seed, load the exact K1-AC best checkpoint into two identical state
geometries:

| Condition | Weights | Validation data | Runtime S-box |
|---|---|---|---|
| `exact` | exact checkpoint | exact K1-AC cache | correct Dialga descriptor |
| `wrong_sbox` | same exact checkpoint | same cache | deterministic wrong S-box |

Only the runtime S-box condition changes. No checkpoint reselection,
retraining, calibration, new examples, keys, differences, pairs or seeds are
allowed.

## Frozen protocol

```text
cipher                 = Dialga-128
rounds                 = 4
seeds                  = 0,1
validation             = exact K1-AC 1024/class cross-key cache
validation total       = 2048 rows/seed
pairs/sample           = 16
input width            = 4096 bits
difference             = 0x40
negative definition    = encrypted random plaintexts
sample structure       = independent pairs
checkpoint             = seed-specific exact K1-AC best checkpoint
parameters             = 214316
training performed     = false
optimizer steps        = 0
metric                 = AUC
device                 = local CPU
```

## Protocol gate

Require all of the following:

1. exactly four rows: two seeds by two S-box conditions;
2. both conditions within a seed share checkpoint, state-dict, feature, label
   and metadata SHA256 values;
3. seed0 and seed1 use distinct exact K1-AC best checkpoints;
4. both models load the same state dict with `strict=True`, retain `214316`
   parameters and expose distinct runtime-window fingerprints;
5. the exact condition reproduces its K1-AC exact AUC within `1e-7`;
6. cache metadata remains Dialga r4, 16 pairs, 4096 input bits, 2048 total
   validation rows, difference `0x40`, strict negatives and seed `10000+seed`;
7. all AUC and probability-delta metrics are finite;
8. no training or optimizer step occurs.

## Research gate

For each seed independently require:

```text
exact AUC >= 0.950
exact - wrong_sbox AUC >= +0.010
max absolute probability change > 1e-6
```

Probability sensitivity is supporting evidence only. A changed prediction with
no positive AUC margin does not establish useful correct-S-box attribution.
Seed averaging cannot hide a failed gate.

## Decision routes

- **Both seeds pass:** the frozen correct checkpoints functionally use the
  correct S-box, while independently trained wrong models can learn a
  substitute shortcut. Next test one training-time counterfactual attribution
  constraint at the same local budget; keep data and architecture fixed.
- **Predictions change but either AUC margin fails:** the checkpoint reads the
  S-box intervention but does not use it discriminatively. Next run K1-AE, a
  zero-training base-path versus histogram-residual ablation on the same
  checkpoints and caches; do not scale.
- **Prediction sensitivity fails:** the exact checkpoint is functionally
  insensitive to the S-box. Run the same K1-AE branch ablation, then redesign
  the Dialga semantic residual or select a less shortcut-dominated calibration
  task before any family claim.
- **Protocol invalid:** repair only the failed source or artifact binding and
  rerun unchanged.

Blocked: remote scale, additional samples/pairs/epochs/seeds, MoE, new
differences, network replacement, attack/SOTA claims and any broad statement
that the model already understands arbitrary SPNs.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_dialga_same_checkpoint_k1ad_20260729
```

Produce `results.jsonl`, `progress.jsonl`, `validation.json`, `gate.json`,
`summary.json`, `comparison.csv` and a Chinese SVG. Render the SVG to pixels
with `visual-qa-redraw`, refresh both recent-result indexes, then append the
observed metrics, claim boundary and evidence-backed next action here.

## Completed result

```text
status   = hold
decision = innovation1_uknit_family_ctspn_k1ad_discriminative_sbox_use_failed
protocol = pass; 4/4 rows; zero training
remote_scale = no
```

| Seed | Correct S-box | Same weights, wrong S-box | Correct - wrong | Maximum probability change | Mean probability change |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.999881744` | `0.999904633` | `-0.000022888` | `0.269894958` | `0.005660495` |
| 1 | `0.999927521` | `0.999891281` | `+0.000036240` | `0.435594708` | `0.007327503` |

All eleven protocol checks passed. Within each seed both conditions share the
exact checkpoint, pre-intervention state, feature, label and metadata hashes;
the exact condition reproduces K1-AC within the frozen `1e-7` tolerance, and
the runtime structure fingerprints differ as required.

The large probability changes prove that the frozen checkpoints read the S-box
intervention. They do not establish discriminative use: AUC remains nearly
perfect under the wrong S-box, and neither seed reaches the `+0.010` margin.
This separates sensitivity from useful semantic attribution.

The final Chinese SVG was rendered at 1800 pixels and passed
`visual-qa-redraw`; close six-decimal AUC values, per-seed margins, probability
changes, titles and decision text have no overlap, clipping, missing glyphs or
ambiguous scales.

Artifacts:

```text
outputs/local_audits/i1_uknit_family_ctspn_dialga_same_checkpoint_k1ad_20260729/
```

## Evidence-backed next action

Run K1-AE on the same checkpoints and caches with four frozen inference
conditions: full, histogram residual off, exact-composition edge residual off,
and both residuals off. This identifies whether Dialga's near-perfect score is
carried by the GF(2) base or by the S-box-aware branches. No new data, training,
capacity or remote scale is authorized.
