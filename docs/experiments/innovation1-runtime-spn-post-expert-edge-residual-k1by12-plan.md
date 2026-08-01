# Innovation 1 Runtime SPN Post-Expert Edge Residual K1-BY12

**Date:** 2026-08-01
**Status:** completed / held / deterministic interventions exhausted
**Execution:** local CPU, frozen K1-BY3 checkpoints and K1-BY7 validation caches

## Research question

K1-BY8 established that the frozen learned linear primitive expert recovers a
positive correct-runtime margin on both seeds even when seed3's raw linear
histogram prefers the affine control. K1-BY9 through K1-BY11 then tested two
partition-based input repairs and one edge-conditioned input residual. None
preserved correct-structure access through every downstream tap on both seeds.

The evidence closes deterministic input modulation. K1-BY12 asks one new
question at the first already-supported learned access point:

> Can a bounded structural residual placed after the frozen linear primitive
> expert preserve correct-runtime access through cell fusion and final output
> on both seeds, while remaining above affine and edge-shuffled controls?

No input statistic, expert weight, pooler, classifier, data row or checkpoint
changes. This is an inference-only intervention-location audit.

## Frozen post-expert formula

For batch row `b`, target cell `t`, hidden coordinate `h` and actual compiled
incoming edge `e=(target_role, source_cell, source_role)`, let:

```text
X[b,t,h] = frozen selected linear primitive expert output

M[b,t,h] = masked mean_e X[b,source_cell(e),h]

G[t,h]   = tanh(E[t,h])
E[t,h]   = frozen learned embedding of target/source roles and expert type

R[b,t,h] = tanh(M[b,t,h] - X[b,t,h]) * G[t,h]

candidate[b,t,h] = X[b,t,h] + R[b,t,h]
```

The residual is bounded coordinate-wise by `1`, has no parameter and has no
coefficient to tune. It is zero whenever the incoming source-cell mean equals
the target expert output or the frozen role embedding gate is zero. Source
cell numbers are used only for tensor gathering; they are never embedded or
exposed as model features. Jointly relabeling the program cells and the sample
state must relabel the output and otherwise leave it unchanged.

The role embedding is not a newly learned component. It is the existing frozen
`edge_descriptor_encoder` output that the primitive expert already consumes.
K1-BY12 only reuses it as a bounded coordinate gate after expert selection.

## Frozen five-condition matrix

| Condition | Inverse state | Post-expert residual edges | Purpose |
|---|---|---|---|
| anchor local / correct | correct runtime | disabled | exact K1-BY8 anchor replay |
| anchor local / affine | affine runtime | disabled | exact K1-BY8 control replay |
| candidate / correct | correct runtime | correct incoming edges | proposed intervention |
| candidate / affine | affine runtime | affine incoming edges | wrong-runtime control |
| candidate / shuffled | correct runtime | shuffled source-cell bindings | edge-attribution control |

The shuffled control keeps the correct inverse state, target/source roles,
expert types, edge counts, weights, data and all later modules fixed. It changes
only the gather endpoint:

```text
shuffled_source_cell = (7 * source_cell + 3) mod 16
```

## Frozen protocol

```text
cipher / rounds        = PRESENT-80 / r7
weights                = K1-BY3 correct checkpoints only
seeds                  = 2,3
validation             = exact K1-BY7 caches, 2048 rows per seed
pairs / input          = 16 independent pairs / 2048 bits
batch                  = 128
internal taps          = post-expert residual, cell fusion, stage pooling,
                         pre-classifier representation
probe discovery        = even validation indices, 512/class
probe evaluation       = odd validation indices, 512/class
probe                   = frozen variance-normalized class-mean difference
training / optimizer   = none / zero steps
device                 = local CPU
```

Both anchor models must replay K1-BY8 final AUCs within `1e-6`. Every condition
must have byte-identical named learned parameters. Candidate and anchor models
may differ only in deterministic post-expert mode and edge-source buffers.

## Readiness gate

Before complete-cache evaluation require:

1. every frozen K1-BY8 and K1-BY11 source digest and decision remains exact;
2. all five conditions exist and use only K1-BY3 correct named parameters;
3. anchors replay K1-BY8 fixtures exactly;
4. the candidate adds no parameter and does not change input histograms,
   experts, S-box path, cell fusion, pooling or classification;
5. correct, affine and shuffled source bindings are pairwise distinct;
6. the candidate residual is finite and every coordinate has magnitude at most
   `1 + 1e-6`;
7. disabling the residual exactly replays the pre-intervention expert output;
8. joint cell relabeling error is at most `1e-6`;
9. no training, optimizer, checkpoint selection or data generation is reachable.

Any failed item makes the protocol invalid and permits only repair of that
specific invariant.

## Preregistered research gate

For every tap on each seed independently require:

```text
candidate correct - candidate affine   >= +0.005
candidate correct - candidate shuffled >= +0.005
```

The final output must satisfy both margins and retain the K1-BY8 correct anchor:

```text
candidate correct final AUC - anchor correct final AUC >= -0.005
```

Both seeds must pass every clause. A mean, one seed or final-only pass cannot
rescue an internal miss.

## Decision routes

- **Complete pass:** retain the post-expert intervention point and preregister
  one local-GPU, same-budget trainable adapter comparison. The frozen residual
  itself is still mechanism evidence, not the trained candidate.
- **Any seed or tap miss:** stop deterministic frozen-checkpoint interventions.
  Return to a separately preregistered trainable post-expert adapter whose
  initial gate is exactly zero, with the K1-BY8 model as same-budget anchor.
- **Protocol invalid:** repair only the failed source, transfer, hook, residual,
  relabeling, bound or artifact invariant and rerun unchanged.

Do not tune this formula, try a second deterministic residual, change the
checkpoint or data, increase scale, add ciphers or launch remote execution.

## Required artifacts

```text
run_id = i1_runtime_spn_post_expert_edge_residual_k1by12_present_r7_seed2_seed3_20260801
```

The audit must emit preflight, `40` probe rows, five-condition final AUCs,
model metadata, gate, validation, summary, comparison CSV, progress and a
Chinese SVG. The SVG must pass `visual-qa-redraw`, and both recent-result
indexes must be refreshed before the result is reported.

## Completed result

The zero-training audit completed with the frozen `40` internal-probe rows and
five final-output conditions per seed:

```text
protocol status       = pass
failed checks         = []
training              = none
optimizer steps       = 0
research gate passed  = false
decision              = innovation1_runtime_spn_k1by12_deterministic_interventions_exhausted
remote scale          = no
```

Final-output evidence was:

| Seed | Anchor correct | Candidate correct | Candidate affine | Candidate shuffled | Correct-affine | Correct-shuffled | Retention |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `0.683736801` | `0.683066845` | `0.656948566` | `0.683831692` | `+0.026118279` | `-0.000764847` | `-0.000669956` |
| 3 | `0.665543556` | `0.663609505` | `0.649007797` | `0.663092613` | `+0.014601707` | `+0.000516891` | `-0.001934052` |

The candidate retained the pre-existing correct-versus-affine runtime signal
and stayed within the registered `-0.005` anchor-retention allowance. It did
not attribute that signal to the actual incoming edge endpoints. Both seeds
missed the `+0.005` correct-versus-shuffled margin at the first
`post_expert_structural_residual` tap:

```text
seed2 first-tap correct-shuffled = +0.001373291
seed3 first-tap correct-shuffled = +0.000404358
```

The failure persisted downstream. Seed 2's shuffled-edge control exceeded the
candidate at the pre-classifier and final-output taps. Seed 3 reached the
shuffle threshold only at the pre-classifier tap and fell back to
`+0.000516891` at final output. The residual therefore changes the frozen
representation and preserves runtime discrimination, but does not stably use
the real source-cell bindings.

## Evidence-backed decision

Stop frozen-checkpoint deterministic interventions. Do not tune a residual
coefficient, substitute a fourth deterministic formula, increase samples or
pairs, add ciphers or launch remote training.

The next experiment must be a separately preregistered, same-budget trainable
post-expert adapter. Its output must initialize exactly to zero so that the
adapter-off state replays the K1-BY8 anchor. It may add only a small shared
parameter set, may not use cipher or absolute cell identifiers, and must be
tested against correct runtime, affine runtime, correct-state shuffled edges
and an adapter-off or label-shuffled control. Begin at `2048/class` on the
local GPU when CUDA is available; do not authorize remote scale unless both
seeds independently establish correct-versus-affine and
correct-versus-shuffled margins.

## Completed artifacts and visual gate

```text
outputs/local_audit/
i1_runtime_spn_post_expert_edge_residual_k1by12_present_r7_seed2_seed3_20260801/
```

The directory contains the preflight, `40`-row result set, final AUC summary,
model metadata, condition comparison, gate, validation, progress log, Chinese
SVG, plot report and rendered visual-QA evidence. The final `2700 x 1288`
pixel render passed `visual-qa-redraw`: no text overlap, clipping, missing
glyphs, ambiguous title or threshold, misleading axis range, or unreadable
near-equal annotation was found. The right panel explicitly labels its
locally magnified final-AUC axis and the subtitle states that the audit is not
formal-scale evidence.
