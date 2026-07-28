# Innovation 1 uKNIT-Family CT-SPN Dialga Branch Ablation K1-AE

**Date:** 2026-07-29
**Status:** completed / held / GF(2) base path dominates
**Execution:** local CPU, zero training and exact K1-AC validation caches

## Research question

K1-AD proved that replacing the correct S-box under a frozen K1-AC checkpoint
changes individual probabilities substantially but leaves Dialga-128 r4 AUC
near `0.9999`. The model reads the intervention, but correct nonlinear
semantics are not discriminatively necessary on this task.

K1-AE asks:

> Which frozen K1-AA branch carries Dialga's near-perfect signal: the GF(2)
> base view, the exact-composition edge residual, or the invariant S-box
> histogram residual?

This is a mechanism localization audit, not a new model comparison.

## Frozen source and interventions

Require completed, recomputable sources:

```text
K1-AC decision = innovation1_uknit_family_ctspn_k1ac_semantic_attribution_failed
K1-AD decision = innovation1_uknit_family_ctspn_k1ad_discriminative_sbox_use_failed
```

For each seed-specific K1-AC exact best checkpoint and its exact validation
cache, evaluate four conditions:

| Condition | Edge residual gate | Histogram residual gate | Interpretation |
|---|---:|---:|---|
| `full` | learned | learned | restored exact checkpoint |
| `histogram_off` | learned | `0` | remove invariant S-box histogram only |
| `edge_off` | `0` | learned | remove exact-composition edge residual only |
| `base_only` | `0` | `0` | retain only the GF(2) Boolean base view |

The GF(2) base is structure-aware through the runtime linear operators, but it
does not apply the S-box. Calling it a raw or no-structure bypass would be
incorrect.

## Frozen protocol

```text
cipher / rounds        = Dialga-128 / 4
seeds                  = 0,1
validation             = exact K1-AC 1024/class cross-key cache
pairs / input width    = 16 / 4096 bits
difference             = 0x40
negative definition    = encrypted random plaintexts
checkpoint             = exact K1-AC restored best checkpoint
parameters             = 214316 before intervention
training               = prohibited
optimizer steps        = 0
metric                 = AUC and per-sample probability delta
```

Only the two scalar gate values may change. All learned tensors, runtime
descriptor, validation examples, labels and metric computation stay fixed.

## Protocol gate

1. exactly eight rows: two seeds by four conditions;
2. every condition within a seed binds the same checkpoint, pre-intervention
   state dict, cache files and exact runtime descriptor;
3. the two seeds use distinct exact best checkpoints;
4. all models strictly load `214316` parameters before the declared gate
   intervention;
5. applied gates exactly match the four-condition table, while learned source
   gate values are recorded and finite;
6. `full` reproduces K1-AD exact AUC within `1e-7`;
7. validation geometry and strict-negative protocol remain unchanged;
8. no training, calibration or optimizer step occurs.

## Diagnostic thresholds and decisions

For each seed report:

```text
full - histogram_off AUC
full - edge_off AUC
full - base_only AUC
maximum and mean probability changes from full
```

Use `0.010 AUC` as the branch-necessity threshold.

- **`base_only >= full - 0.010` on both seeds:** classify Dialga r4 with 16
  pairs as GF(2)-base dominated. The task is too shortcut-saturated to judge
  S-box semantics. Keep the uKNIT 16-pair result, but stop using this Dialga
  surface as a semantic family gate. Next select a bounded, already evidenced
  less-saturated Dialga condition before another training run.
- **Removing one residual loses at least `0.010` on both seeds:** that residual
  is functionally necessary. Next redesign only its semantic attribution while
  preserving all other branches and data.
- **Only joint removal loses `0.010`:** classify the residuals as jointly useful
  but individually substitutable. Next test one frozen interaction audit; do
  not add capacity or data first.
- **Protocol invalid:** repair only the audit binding and rerun unchanged.

Blocked: remote scale, new data, retraining, new models, MoE, new differences,
checkpoint selection and attack/SOTA/family-success claims.

## Required artifacts

```text
run_id = i1_uknit_family_ctspn_dialga_branch_ablation_k1ae_20260729
```

Produce result/progress JSONL, validation, gate, summary, comparison CSV and a
Chinese SVG. Apply `visual-qa-redraw`, update the source and result documents,
refresh both recent-result indexes, then make the next action executable.

## Completed result

```text
status   = hold
decision = innovation1_uknit_family_ctspn_k1ae_gf2_base_path_dominates
protocol = pass; 8/8 rows; zero training
remote_scale = no
```

Validation AUC:

| Seed | Full | Histogram off | Edge off | GF(2) base only |
|---:|---:|---:|---:|---:|
| 0 | `0.999881744` | `0.999485016` | `0.999871254` | `0.999432564` |
| 1 | `0.999927521` | `0.998677254` | `0.999928474` | `0.998563766` |

Full-model AUC minus each ablation:

| Seed | Histogram contribution | Edge contribution | Both residuals combined |
|---:|---:|---:|---:|
| 0 | `+0.000396729` | `+0.000010490` | `+0.000449181` |
| 1 | `+0.001250267` | `-0.000000954` | `+0.001363754` |

Every protocol check passed. All conditions within a seed share the same exact
best checkpoint, pre-intervention state, runtime descriptor and validation
cache. Only the declared two scalar gates change, `full` exactly reproduces
K1-AD, and no optimizer or calibration step occurs.

The learned branches do change individual probabilities substantially: turning
both off changes maximum probabilities by `0.713080/0.728326` and mean
probabilities by `0.109790/0.117952`. Nevertheless, AUC remains above `0.9985`
with only the GF(2) base. Neither residual, nor both together, reaches the
`0.010` necessity threshold. Dialga r4 at sixteen pairs is therefore a
linear-diffusion-saturated surface, not a valid S-box semantic adjudicator.

The Chinese SVG was rendered at 1800 pixels and passed `visual-qa-redraw` with
all long branch labels, close values, thresholds, title, decision and next
action fully visible and unambiguous.

Artifacts:

```text
outputs/local_audits/i1_uknit_family_ctspn_dialga_branch_ablation_k1ae_20260729/
```

## Evidence-backed next action

Do not reuse Dialga r4 sixteen-pair AUC as the family S-box gate and do not move
to Dialga r5 `0x40`: D3/D4 already show that fifth-round signal collapses, and
D5 found no eligible alternative among all 128 single-bit differences.

The next bounded question is whether pair aggregation, rather than the cipher
round itself, creates the saturation. Use the existing K1-AC validation caches
and checkpoints for a zero-training single-pair replay audit: select each of
the sixteen constituent pairs in turn, repeat it into the fixed 16-pair input
geometry, and compare exact versus wrong-S-box descriptors using the same
checkpoint. Report per-pair and aggregate AUC without choosing a favorable pair
after inspection. Only if the reduced-query panel becomes non-saturated and
shows a stable correct-S-box margin should a one-pair training matrix be
preregistered. Otherwise keep Dialga as a signal-retention calibration only and
seek family semantics on a different shared-primitive cipher surface.
