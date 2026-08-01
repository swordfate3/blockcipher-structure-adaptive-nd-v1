# Innovation 1 Runtime SPN Edge-Context Covariance K1-BY11

**Date:** 2026-08-01
**Status:** completed / protocol pass / research gate hold
**Execution:** local CPU, frozen K1-BY3 checkpoints and K1-BY7 validation caches

## Research question

K1-BY9 showed that averaging each local linear histogram with an equality-class
mean repaired seed3's first-tap runtime margin but over-smoothed seed2's learned
permutation-expert access. K1-BY10 found no target cell whose partition collision
explained that asymmetry consistently across both stages and downstream taps.
Equality-partition pooling is therefore closed.

K1-BY11 asks one replacement question:

> Can a non-averaging, zero-parameter residual derived from each target cell's
> actual ordered incoming edges repair seed3's linear-histogram margin while
> preserving every seed2 downstream margin and final AUC?

The experiment changes only the linear state-to-histogram representation. It
does not train, tune, add channels, pool cells or change any learned module.

## Frozen edge-conditioned formula

For a target cell `t`, histogram bin `v`, ciphertext-pair index `p` and incoming
compiled edge `e=(target_role, source_cell, source_role)`, define:

```text
V[p,t]       = inverse-linear target-cell difference value in {0,...,15}
H[t,v]       = mean_p 1[V[p,t] = v]

Q[p,e]       = XOR of the three source-cell difference bits whose roles are
               not source_role(e)
S[p,e]       = (-1) ^ Q[p,e]
Z[p,t]       = masked mean of S[p,e] over the actual incoming edges of t

R[t,v]       = E_p[1[V[p,t]=v] * Z[p,t]] - H[t,v] * E_p[Z[p,t]]
candidate[t,v] = H[t,v] + R[t,v]
```

`R` is the empirical covariance between the target bin and the non-transported
three-bit context of its incoming source cells. It has four required properties:

1. it uses the full `(target role, source cell, source role)` edge triple;
2. source/target cell numbers are used only to gather sample values and are
   never encoded as features;
3. it averages edges only within one target cell and never averages target
   cells or equality classes;
4. `sum_v R[t,v] = 0`, so the local histogram mass remains unchanged.

There is no coefficient, learned gate, normalization scan or post-result choice.
The residual is added once with its mathematically fixed unit coefficient.

## Frozen condition matrix

Five inference cells are evaluated under each frozen correct checkpoint:

| Representation | Inverse state | Edge context | Purpose |
|---|---|---|---|
| anchor local | correct runtime | none | exact K1-BY8 anchor replay |
| anchor local | affine runtime | none | exact K1-BY8 wrong-runtime replay |
| candidate covariance | correct runtime | correct edges | proposed representation |
| candidate covariance | affine runtime | affine edges | same-form wrong runtime |
| candidate covariance | correct runtime | shuffled source bindings | edge attribution control |

The shuffled control leaves the correct inverse-linear state, target roles,
source roles, edge count, expert type, data, weights and every downstream module
unchanged. It replaces only each context-gather source cell by:

```text
shuffled_source_cell = (7 * source_cell + 3) mod 16
```

This is a fixed derangement of all sixteen PRESENT cells. It is not selected
from data and is distinct from K1-BY6's affine wrong endpoint `(5s+1) mod 16`.

## Frozen protocol

```text
cipher / rounds        = PRESENT-80 / r7
weights                = K1-BY3 correct checkpoints only
seeds                  = 2,3
validation             = exact K1-BY7 caches, 2048 rows per seed
pairs / input          = 16 independent pairs / 2048 bits
batch                  = 128
taps                   = K1-BY8 five-tap forward order
probe discovery        = even validation indices, 512/class
probe evaluation       = odd validation indices, 512/class
probe                   = frozen variance-normalized class-mean difference
training / optimizer   = none / zero steps
device                  = local CPU
```

The two anchor cells must reproduce K1-BY8's final AUCs within `1e-6`. Every
candidate must have the same named learned parameters and parameter fingerprint
as the frozen correct checkpoint.

## Readiness gate

Before complete-cache evaluation require:

1. all frozen K1-BY8 and K1-BY10 source digests and decisions remain exact;
2. all five conditions exist and share the frozen learned parameters;
3. only candidate cells add deterministic edge-context source buffers;
4. correct, affine and shuffled edge bindings are pairwise distinct;
5. anchor fixture outputs exactly replay K1-BY8;
6. candidate fixture tensors are finite, preserve per-cell histogram mass
   within `1e-6`, and differ from their local anchors;
7. jointly relabeling cells in both the program and sample state changes the
   candidate by at most `1e-6` after undoing the output relabeling;
8. S-box histograms and all learned modules remain unchanged;
9. training, optimizer, checkpoint selection and data generation are impossible.

Any failure makes the experiment invalid and permits only repair of that
specific invariant.

## Preregistered research gate

For each seed independently and for every internal tap:

```text
candidate correct - candidate affine   >= +0.005
candidate correct - candidate shuffled >= +0.005
```

At final output require the same two `+0.005` margins and:

```text
candidate correct final AUC - anchor correct final AUC >= -0.005
```

Both seeds must pass every clause. Means cannot hide a failed seed, and a final
pass cannot rescue an internal tap miss.

## Decision routes

- **Complete pass:** retain edge-context covariance as the deterministic input
  representation and preregister one same-budget local training comparison
  against the local-histogram anchor, affine and shuffled controls.
- **Any tap or seed miss:** close deterministic input modulation. Move the
  structural intervention after the frozen linear primitive expert, where
  K1-BY8 already shows both seeds have positive runtime access.
- **Protocol invalid:** repair only the failed source, buffer, covariance,
  relabeling, hook, probe or artifact binding and rerun unchanged.

In all routes, do not tune the residual, add a second formula, retrain inside
K1-BY11, add data, pairs, seeds, ciphers, capacity or remote execution.

## Required artifacts

```text
run_id = i1_runtime_spn_edge_context_covariance_k1by11_present_r7_seed2_seed3_20260801
```

The audit must emit preflight, 50 probe rows, five-condition final AUCs, model
metadata, gate, validation, summary, comparison CSV, progress and a Chinese SVG.
The rendered SVG must pass `visual-qa-redraw`, and both recent-result indexes
must be refreshed before reporting.

## Completed result

The zero-training audit completed with the frozen `50` probe rows:

```text
protocol status       = pass
failed checks         = []
training              = none
optimizer steps       = 0
research gate passed  = false
decision              = innovation1_runtime_spn_k1by11_input_modulation_not_supported
remote scale          = no
```

The candidate preserved the final output on both seeds and met both final
control margins, but the preregistered gate required every internal tap to pass
as well:

| Seed | Candidate correct AUC | Correct-affine | Correct-shuffled | Retention vs anchor |
|---:|---:|---:|---:|---:|
| 2 | `0.683926582` | `+0.020493507` | `+0.006752014` | `+0.000189781` |
| 3 | `0.663278103` | `+0.008837223` | `+0.007205963` | `-0.002265453` |

Seed 2 first lost the gate immediately after the linear primitive expert:

```text
primitive expert correct-shuffled = -0.007789612
cell fusion correct-shuffled       = -0.002349854
stage pooling correct-shuffled     = -0.006954193
pre-classifier correct-shuffled    = -0.000755310
```

Seed 3 repaired the primitive-expert and cell-fusion margins, but did not fix
the original first-layer failure and again fell below the shuffled margin at
the pre-classifier representation:

```text
linear histogram correct-affine    = -0.005157471
pre-classifier correct-shuffled     = +0.000629425
```

Therefore the final AUC pass cannot rescue the failed internal attribution
gate. The candidate changed the frozen representation without producing stable
correct-edge access through the downstream learned path.

## Evidence-backed decision

Close deterministic input modulation. Do not tune the residual coefficient,
try another input formula, add samples, train the current candidate or move it
to the remote workstation.

The next experiment must place exactly one bounded structural residual after
the frozen `linear_primitive_expert` and before `cell_fusion`. K1-BY8 already
established positive correct-runtime access at that tap on both seeds, so this
intervention point tests whether the accessible signal can be preserved rather
than asking the input histogram to manufacture it. The follow-up must remain a
zero-training, same-checkpoint, same-cache, seed-2/3 local audit with correct,
affine and edge-shuffled controls. Any internal or final margin below `+0.005`
must stop frozen-checkpoint deterministic interventions and route to a
separately preregistered trainable post-expert adapter.

## Completed artifacts and visual gate

```text
outputs/local_audit/
i1_runtime_spn_edge_context_covariance_k1by11_present_r7_seed2_seed3_20260801/
```

The directory contains the preflight, `50`-row result set, final AUC summary,
model metadata, comparison CSV, gate, validation, progress log, Chinese SVG,
plot report and rendered visual-QA evidence. The final `2700 x 1315` pixel
render passed `visual-qa-redraw`: no text overlap, clipping, missing glyphs,
ambiguous threshold, misleading axis range or unreadable near-equal annotation
remained. The final AUC panel is explicitly labeled as a local enlargement.
