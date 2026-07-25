# Innovation 1 Runtime-SPN Primitive Descriptor And Gradient Audit Plan

Date: 2026-07-25

## Status

```text
phase = completed post-hold audit
training = none
source = completed five-cipher correct checkpoints and immutable train caches
execution = local CPU
remote_scale = prohibited
status = hold
decision = innovation1_runtime_spn_adapter_identifiability_audit_required
```

## Research Question

Did the deterministic two-adapter route fail mainly because the current
`fan_in_1` versus `multi_source` descriptor collapses materially different SPN
primitives into the same route, or because the five tasks demand conflicting
updates from the shared Runtime-E4 backbone even when their local primitives
are represented correctly?

This is a frozen-checkpoint diagnostic. It does not train, select a checkpoint,
read validation metrics to tune a threshold, or alter the completed result.

## Frozen Evidence

```text
source run = i1_runtime_spn_primitive_adapter_five_cipher_joint_2048_seed0_seed1_20260725
roles = correct only
seeds = 0,1
tasks = GIFT-64 r6, SKINNY-64/64 r7, RECTANGLE-80 r6,
        uKNIT-BC prefix-r5, Dialga-128 prefix-r4
split = complete training cache, 4096 total rows/cipher/seed
batch size = 256
loss = MSE
model state = restored source checkpoint, no optimizer step
```

The audit must verify the source config hash, checkpoint role/mode/seed,
immutable cache metadata, row counts and finite gradients before interpreting
any cosine.

## Descriptor Sufficiency Audit

For each runtime window, record without using the cipher name as a model input:

```text
block bits and cell count (audit metadata only)
current per-round fan_in_1/multi_source cell counts
inverse-linear row fan-in histogram
per-cell inverse source-count pattern
unique S-box truth-table hashes
unique transition count
full transition and window SHA256 fingerprints
```

A coarse-router collision occurs when two cipher windows have the same current
router signature but different full window fingerprints. The audit reports all
such groups; it does not treat block width or a global fingerprint as an
eligible future routing feature.

## Gradient Conflict Audit

For each seed and task, compute the exact mean training-loss gradient over the
complete cached training split. Flatten gradients into four disjoint views:

```text
shared_backbone       = every trainable parameter except primitive adapters
all_adapters          = both primitive adapters together
fan_in_1_adapter      = fan_in_1 low-rank adapter only
multi_source_adapter  = multi_source low-rank adapter only
```

An inactive adapter has zero norm and an undefined cosine; it must be recorded
as `null`, never coerced to zero. Report pairwise cosine, gradient norm and
negative-pair fraction separately for both seeds. Do not combine seeds before
showing their individual matrices.

## Frozen Decision Rules

Descriptor refinement becomes the next priority when all are true:

1. the source joint gate is a valid `hold`;
2. at least two coarse-router collision groups contain distinct full window
   fingerprints;
3. at least one same-route task pair has active-adapter cosine below zero in
   either seed, or its two-seed mean cosine is below `+0.10`.

Shared optimization conflict becomes the next priority when, for both seeds,
the `shared_backbone` task-pair matrix has either mean off-diagonal cosine below
zero or negative-pair fraction at least `0.50`.

Decision order:

```text
descriptor priority only -> refine exactly one local primitive descriptor axis
shared conflict only     -> keep descriptor frozen; test one matched
                            multi-task gradient-conflict treatment
both                     -> refine the worst same-route collision first;
                            optimization remains a required control
neither                  -> audit adapter scale/rank identifiability before any
                            new expert, sample, epoch or remote run
protocol invalid         -> repair audit only; no research interpretation
```

The next descriptor may use local S-box truth-table relations, local
transition identity/type or per-cell source pattern. It may not use cipher ID,
cipher name, block width, key width, round count or a global structure hash.

## Required Artifacts

```text
results.jsonl
descriptor_profiles.json
gradient_cosines.csv
gradient_norms.csv
validation.json
gate.json
summary.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

The plot must render both seed matrices with readable labels and a separate
descriptor-collision panel. It must pass `visual-qa-redraw`. After completion,
refresh `outputs/00_RECENT_RESULTS.md` and JSON in the same turn.

## Blocked Actions

- No retraining or checkpoint selection in this audit.
- No validation-set gradients or validation-driven descriptor selection.
- No remote run, scale increase, extra epochs or full learned MoE.
- No cipher-ID expert, task-specific head or global fingerprint router.

## Completed Result

The audit consumed the complete 4096-row training cache for every cipher and
seed, restored the two correct checkpoints, and performed zero optimizer
steps. Protocol, checkpoint, cache, row-count and finite-gradient validation
all passed.

Two deterministic descriptor collisions were confirmed:

```text
all fan_in_1    = GIFT / RECTANGLE, two distinct full window fingerprints
all multi_source= uKNIT / Dialga,   two distinct full window fingerprints
```

However, the tasks inside each coarse bucket did not request conflicting
Adapter updates:

| Same-route pair | seed0 cosine | seed1 cosine | mean |
| --- | ---: | ---: | ---: |
| uKNIT / Dialga multi-source Adapter | 0.621486 | 0.504165 | 0.562826 |
| GIFT / RECTANGLE fan-in-1 Adapter | 0.238279 | 0.116488 | 0.177383 |

The shared-backbone mean off-diagonal cosine was `0.369807/0.354935`, with
negative-pair fractions `0.20/0.00`; therefore the preregistered shared
gradient-conflict gate also did not trigger.

```text
validation = pass
descriptor_refinement_priority = false
shared_gradient_conflict = false
training_or_optimizer_steps = 0
```

Evidence:

```text
outputs/local_audit/i1_runtime_spn_primitive_descriptor_gradient_audit_seed0_seed1_20260725/
```

### Evidence-Backed Next Action

Do not split the router or add task-conflict optimization yet. Audit whether
the trained additive Adapter has enough functional effect and effective rank
to be identifiable at all. This decision was executed in the subsequent
primitive-Adapter identifiability audit.
