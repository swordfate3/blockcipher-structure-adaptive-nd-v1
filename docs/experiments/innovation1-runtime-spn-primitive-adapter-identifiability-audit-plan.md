# Innovation 1 Runtime-SPN Primitive Adapter Identifiability Audit Plan

Date: 2026-07-25

## Status

```text
phase = completed frozen-checkpoint audit
training = none
source = completed five-cipher correct checkpoints and immutable train caches
execution = local CPU
remote_scale = prohibited
status = pass
decision = innovation1_runtime_spn_additive_adapter_functionally_weak
```

## Research Question

Does the trained low-rank primitive Adapter make a measurable, route-specific
contribution inside the shared Runtime-E4 model, or is its `0.1` residual
injection too weak or rank-collapsed to be identifiable against the shared
backbone?

The preceding descriptor/gradient audit found two coarse-router collisions,
but same-route adapter gradients remained positively aligned and shared
backbone gradients were not systematically conflicting. Before inventing more
primitive classes, this audit tests whether the current conditional mechanism
has enough functional effect to express any routing benefit at all.

## Frozen Evidence

```text
source run = i1_runtime_spn_primitive_adapter_five_cipher_joint_2048_seed0_seed1_20260725
roles = correct checkpoint only
seeds = 0,1
tasks = GIFT, SKINNY, RECTANGLE, uKNIT, Dialga
split = complete training cache, 4096 total rows/cipher/seed
batch size = 256
model updates = 0
```

Every probe strictly loads the same trained state dictionary. Only a
non-parameter runtime mode or residual multiplier changes.

## Frozen Counterfactuals

| Probe | Router | Residual scale | Purpose |
| --- | --- | ---: | --- |
| disabled | correct | 0.0 | same trained backbone with Adapter contribution removed |
| source | correct | 0.1 | exact completed candidate |
| uniform | uniform | 0.1 | same weights, selection-disabled counterfactual |
| shuffled | shuffled | 0.1 | same weights, wrong-route counterfactual |
| amplified | correct | 0.5 | fixed no-training scale sensitivity probe |

No probe may select a new checkpoint or use validation data. Report per-task
training AUC, mean absolute logit delta, RMS logit delta, delta relative to the
disabled-logit standard deviation and threshold flip rate.

For both trained adapters, also report down/up Frobenius norms and the singular
values/effective rank of the linearized `up.weight @ down.weight` map. This is
a rank proxy only because the actual Adapter contains GELU.

## Frozen Decision Rules

For each seed, define the source Adapter as functionally active when:

```text
median source-vs-disabled relative RMS logit delta >= 0.10
at least 4/5 tasks relative RMS logit delta         >= 0.05
```

Define route specialization when, for each seed:

```text
median source-vs-uniform relative RMS logit delta  >= 0.05
median source-vs-shuffled relative RMS logit delta >= 0.05
```

Define rank collapse when either Adapter has linearized effective rank below
`2.0` in both seeds. Define useful scale sensitivity when amplified correct
routing improves five-cipher training macro AUC over source by at least
`+0.005` in both seeds without reducing any task by more than `0.005`.

Decision order:

```text
not functionally active -> replace weak additive residual with one matched
                           structure-conditioned FiLM/gated modulation
active, not specialized -> retain shared backbone; refine one local descriptor
                           or add a specialization objective, not more rank
active + specialized + useful scale sensitivity
                        -> train one matched scale-0.5 candidate locally
rank collapsed          -> test rank regularization before increasing rank
active + specialized, no scale benefit
                        -> replace Adapter computation with a matched dense
                           conditional basis/FiLM candidate
protocol invalid        -> repair audit only
```

## Required Artifacts

```text
results.jsonl
counterfactual_metrics.csv
adapter_rank.json
validation.json
gate.json
summary.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

The figure must show both seeds, all tasks and the source/route/scale
counterfactual effects with a Chinese title and readable legend. It must pass
`visual-qa-redraw`. Refresh the recent-results index after completion.

## Blocked Actions

- No checkpoint update, optimizer step or validation-set probe.
- No new expert, learned router, scale training or rank increase in this audit.
- No cipher ID, task-specific state or global fingerprint input.
- No remote execution or scale-up.

## Completed Result

All 50 frozen counterfactual rows completed from the two correct checkpoints
and complete training caches. Source alignment, cache metadata, row counts and
finite metrics passed; no model update or optimizer step occurred.

| Audit quantity | seed0 | seed1 | Gate |
| --- | ---: | ---: | ---: |
| source-vs-disabled median relative RMS | 0.112397 | 0.078945 | >= 0.10 |
| tasks above relative RMS 0.05 | 5/5 | 3/5 | >= 4/5 |
| uniform-vs-correct median relative RMS | 0.109726 | 0.058657 | >= 0.05 |
| shuffled-vs-correct median relative RMS | 0.236557 | 0.116516 | >= 0.05 |
| amplified-minus-source macro train AUC | -0.002225 | -0.003805 | >= +0.005 |

The gates are route-sensitive in both seeds, but the source additive effect is
not functionally active on seed1. Increasing the frozen scale from `0.1` to
`0.5` made macro training AUC worse and reduced uKNIT by `0.007061/0.010266`.
The low-rank weights did not collapse: effective ranks were approximately
`7.83--7.89` out of 8 for both adapters and seeds.

```text
functionally_active_both_seeds = false
route_specialized_both_seeds = true
rank_collapsed = false
useful_scale_sensitivity_both_seeds = false
```

Evidence:

```text
outputs/local_audit/i1_runtime_spn_primitive_adapter_identifiability_audit_seed0_seed1_20260725/
```

### Evidence-Backed Next Action

Keep the same two-bin router, rank, scale, data and optimizer and replace only
the weak additive effect with a parameter-matched local multiplicative gate.
That gated redesign was implemented and adjudicated in the subsequent
five-cipher plan; no larger scale was authorized by this audit.
