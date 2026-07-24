# Innovation 1 uKNIT U2-G Same-Checkpoint Query Counterfactual Plan

Date: 2026-07-24

## Status

```text
stage    = completed local inference audit
run_id   = i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724
training = forbidden
decision = innovation1_uknit_delta_u_same_checkpoint_use_supported
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

## Completed Result

The inference-only audit completed from implementation commit `12c1409c` plus
the disk-cache loader fix `f5ca51fb`. It reused each U2-F candidate's selected
best checkpoint and exact 2048-row validation cache. No training or checkpoint
selection occurred. All protocol checks passed, including identical
checkpoint/features/labels within each seed panel, correct main structure in
all conditions, query-only S-box ownership substitution, exact descriptor
window, equal `458850` parameter geometry and complete SHA256 provenance.

| Seed | Correct delta-U | Shuffled-query delta-U | Delta-V identity | Correct - shuffled | Correct - identity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.543138981 | 0.532666683 | 0.525547028 | +0.010472298 | +0.017591953 |
| 1 | 0.554934978 | 0.520118237 | 0.512523651 | +0.034816742 | +0.042411327 |

Maximum absolute probability changes from the correct delta-U reference were:

| Seed | Shuffled-query delta-U | Delta-V identity |
| ---: | ---: | ---: |
| 0 | 0.099709451 | 0.101571321 |
| 1 | 0.043943763 | 0.059160411 |

Both seeds exceeded both `+0.005` AUC gates and both probability vectors
changed by much more than `1e-6`. Therefore the U2-F gain is not explained
only by separately trained model noise: under the same weights, same main
structure, same state-triplet inputs and same examples, the correct runtime
delta-U query is materially better than either query control.

```text
status   = pass
decision = innovation1_uknit_delta_u_same_checkpoint_use_supported
keep     = explicit sample-conditioned runtime inverse-S-box delta-U query
claim    = supported same-checkpoint uKNIT query-use mechanism at local diagnostic scale
```

This does not establish formal scale, a cryptanalytic attack, cross-cipher
generalization, SOTA, or a breakthrough. It does not authorize a larger
dataset or remote GPU run.

Artifacts:

```text
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724/results.jsonl
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724/progress.jsonl
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724/validation.json
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724/gate.json
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724/summary.json
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724/curves.svg
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2g_same_checkpoint_20260724/visual_qa_passed.marker
```

The exact SVG rendered at `2364 x 1085` pixels and passed
`visual-qa-redraw`. The Chinese title/subtitle, decision line, grouped AUC
bars, margin bars, labels, legends, axes and `+0.005` threshold are readable
without overlap, clipping, missing glyphs, or ambiguous scale.

## Evidence-Backed Next Action: U2-H Cross-Window Replication

Preregister one same-budget uKNIT replication that changes only the aligned
cipher prefix and runtime transition window:

```text
question          = does the delta-U query mechanism replicate on the next valid uKNIT window?
candidate         = correct delta-U query
same-budget anchor= delta-V identity query
required control  = shuffled S-box-ownership delta-U query
cipher/window     = uKNIT-BC prefix r5, round_start 3, processor_steps 2
scale             = 2048/class train, 1024/class validation
seeds/epochs      = 0,1 / 10
pairs/sample      = 4
negative          = encrypted_random_plaintexts
execution         = local CPU diagnostic with a new parameter-matched disk cache
```

Use the same `>=0.520`, candidate-minus-anchor `>=+0.005`, and
candidate-minus-shuffled `>=+0.005` gates on both seeds. A pass should be
followed by the same checkpoint query-only audit on the new window. A miss
means the current mechanism is window-specific and must not be scaled. Do not
change data size, epochs, pairs, difference, architecture, loss, optimizer, or
controls in U2-H.
