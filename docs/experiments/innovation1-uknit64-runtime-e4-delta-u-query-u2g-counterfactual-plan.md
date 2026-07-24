# Innovation 1 uKNIT U2-G Same-Checkpoint Query Counterfactual Plan

Date: 2026-07-24

## Status

```text
stage    = planned local inference audit
run_id   = i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724
training = forbidden
decision = pending
```

## Question

Does each trained U2-F candidate use the runtime delta-U query itself, or did
the separately trained candidate/control margin arise from optimization noise
or a broader S-box-context distribution change?

## Frozen Sources

```text
source run       = i1_rtg1_uknit64_runtime_e4_delta_u_query_u2f_2048_seed0_seed1_20260724
source roles     = correct delta-U candidate only, seeds 0 and 1
checkpoint       = each source row's selected best checkpoint
validation       = each seed's exact 2048-row disk-backed validation cache
runtime window   = uKNIT prefix r4, round_start 2, processor_steps 2
device           = local CPU
training         = none
```

## One Inference Variable

For each seed, evaluate the same checkpoint and examples under:

| Condition | Main structure and edge gate | Third query |
| --- | --- | --- |
| reference | correct | correct runtime `deltaU` |
| ownership control | correct | `deltaU` using shuffled per-cell S-box ownership only |
| identity control | correct | `deltaV`, with no inverse-S-box lookup |

The main structure object remains correct for all three conditions. The
ownership control supplies a second runtime descriptor only to the query
operator; it may not alter inverse linear mapping, cell membership, bit roles,
edge-gate S-box context, mixers, pooling, classifier, or either state-triplet
input.

## Readiness Gate

1. The normal forward path remains numerically unchanged when no audit
   override is supplied.
2. All three conditions use the same checkpoint SHA256, feature SHA256, label
   SHA256, descriptor window and `458850` parameters within each seed.
3. The first two fusion inputs are exactly equal across all three conditions.
4. Correct and shuffled query structures have identical cell membership, bit
   roles and linear operators; only S-box ownership differs.
5. The identity condition bypasses inverse-S-box lookup.
6. Existing runtime-SPN regressions pass.

## Research Gate

For each seed require:

```text
correct deltaU AUC - shuffled-ownership deltaU AUC >= +0.005
correct deltaU AUC - deltaV identity AUC >= +0.005
max absolute probability change versus each control > 1e-6
```

Both seeds must pass. A pass supports a same-checkpoint query-use mechanism
and opens one separately planned same-budget cross-window replication. A miss
classifies U2-F as training-distribution evidence only and closes scale-up.

## Outputs

```text
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724/
```

Require `results.jsonl`, `progress.jsonl`, `validation.json`, `gate.json`,
`summary.json`, `curves.svg`, a visual-QA marker, and a refreshed recent-result
index.

## Blocked Routes

No gradient updates, checkpoint selection, new data, extra seeds, changed
query thresholds, DDT/trail features, remote GPU, or sample scale-up are
allowed in U2-G.
