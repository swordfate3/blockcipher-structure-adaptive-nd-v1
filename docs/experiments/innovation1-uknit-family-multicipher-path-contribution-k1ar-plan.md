# Innovation 1 K1-AR Shared Path-Contribution Audit

**Status:** completed replay fix / pass / zero-training local audit
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_multicipher_path_contribution_k1ar_replica0_replica1_replay_fix_20260729`

The initial implementation run without the `replay_fix` suffix is protocol
invalid. Its manually decomposed logits matched the model forward exactly, but
it recomputed probabilities with NumPy float64 sigmoid instead of the source
control's PyTorch float32 sigmoid. A few tied ranks changed branch-off AUC by
`4.768e-7`, exceeding the frozen `1e-7` replay tolerance. The repair changes
only probability replay precision, uses a new run ID and preserves every
source, path definition and research gate below.

## 1. Research Question

K1-AQ improved all four Midori panels but damaged uKNIT and Dialga. Its fixed
loss scales changed every learned weight, so final AUC alone does not identify
which shared path moved. K1-AR asks:

> Did inverse-norm training selectively make the S-box-transition residual more
> useful for Midori while leaving the same residual flat or less useful for
> uKNIT and Dialga?

This is a representation-contribution audit, not a new architecture or training
experiment. It determines whether a structure-conditioned fusion gate deserves
one later readiness test without introducing a cipher ID.

## 2. Frozen Sources

Bind and rehash the complete checkpoints, checkpoint manifests, gates,
validations and 36-row controls from:

```text
K1-AO equal-loss shared training
K1-AQ fixed inverse-gradient-norm shared training
```

Reuse K1-AQ's authority loader for the exact same disk-backed fresh datasets,
keys, strict encrypted-random-plaintext negatives, pair count and runtime
descriptors. There are two checkpoint families, two replicas, three ciphers and
two fresh splits, producing `24` summary rows.

## 3. One Observed Variable

For each frozen checkpoint and batch, reproduce the existing K1-AK forward as:

```text
base_embedding
edge_embedding       = tanh(edge_gate) * tanh(edge_residual)
transition_embedding = tanh(transition_gate) * tanh(S-box-transition residual)

pure_base logits     = classifier(base_embedding)
edge_fused logits    = classifier(base_embedding + edge_embedding)
full logits          = classifier(base_embedding + edge_embedding
                                   + transition_embedding)
```

`edge_fused` must exactly replay the existing transition-branch-off control,
and `full` must exactly replay the correct-runtime control within `1e-7` AUC.
The classifier is nonlinear, so contribution is measured by same-checkpoint
intervention, not by assuming additive logits.

Record:

```text
pure-base, edge-fused and full AUC
full-minus-edge AUC gain
edge-minus-base AUC gain
AUC of the transition logit delta
mean signed transition probability improvement
fraction of rows helped by the transition branch
mean MSE reduction from the transition branch
RMS of base, gated edge and gated transition embeddings
gated transition / base and gated transition / edge RMS ratios
effective edge and transition gates
state and output hashes
```

No gradients, optimizer, checkpoint selection, label access during path
construction, or parameter mutation are permitted.

## 4. Frozen Gates

First require all source bindings, row counts, state immutability and exact
full/branch replay checks to pass.

Define each panel's transition gain as:

```text
transition_gain = full_auc - edge_fused_auc
gain_delta       = K1-AQ transition_gain - K1-AO transition_gain
```

The heterogeneous transition-demand hypothesis passes only when:

```text
Midori gain_delta >= +0.010                         in 4/4 panels
uKNIT/Dialga gain_delta <= 0.000                    in at least 6/8 panels
all required directions are represented in both replicas and both splits
```

This gate is deliberately about directionally stable path utility. A large
full AUC or large embedding norm alone cannot pass it.

## 5. Decisions

If the gate passes, the next action is a separately preregistered readiness
test for a bounded structure-derived transition gate. The gate may consume
runtime operator statistics already present in the descriptor, but not a
cipher name, cipher ID, per-cipher head, adapter or expert. It must preserve
the shared parameter geometry and expose mismatched-structure controls.

If Midori does not improve consistently at the transition path, or the
non-Midori tasks improve in the same direction, reject the conditional-gate
hypothesis and audit transition projection geometry instead.

In either case, do not tune K1-AQ loss scales, use PCGrad, increase pairs,
samples, epochs or width, add MoE/experts, or launch remotely.

## 6. Required Artifacts

Write under `outputs/local_audit/<run_id>/`:

```text
preflight.json
checkpoint_manifest.json
results.jsonl
comparison.csv
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

The Chinese figure must distinguish final AUC from incremental path utility,
state the `2048/class/cipher`, `4 pairs`, zero-training claim scope, and pass a
rendered-pixel `visual-qa-redraw` inspection. Refresh the recent-result indexes
after completion.

## 7. Completed Result

The repaired run completed `24/24` frozen panels. All source digests,
checkpoints, datasets, 36-row control sets and inherited K1-AO/K1-AP/K1-AQ
authority checks passed. The manually decomposed full and branch-off forwards
replayed the source controls within `1e-7`; all state hashes were immutable and
the optimizer-step count remained zero.

Transition-path gain deltas were:

| Cipher | Replica | Same-key | Cross-key |
|---|---:|---:|---:|
| uKNIT-BC r5 | 0 | -0.000628 | -0.019046 |
| uKNIT-BC r5 | 1 | -0.093203 | -0.051177 |
| Midori64 r4 | 0 | +0.063750 | +0.063171 |
| Midori64 r4 | 1 | +0.045272 | +0.051010 |
| Dialga-128 r4 | 0 | -0.001811 | -0.004301 |
| Dialga-128 r4 | 1 | -0.038957 | -0.032574 |

Thus the frozen directional gates passed:

```text
Midori gain delta >= +0.010          = 4/4  (required 4/4)
uKNIT/Dialga gain delta <= 0.000     = 8/8  (required >= 6/8)
full and branch forward replay       = 24/24
state immutable                      = 24/24
zero training / zero optimizer step  = 24/24
status                               = pass
decision                             = innovation1_uknit_family_k1ar_heterogeneous_transition_demand_supported
remote_scale                         = no
```

The transition-logit alignment moved in the same explanatory direction. For
Midori it increased from `0.553866-0.584874` to `0.618069-0.665491`; for
Dialga replica1 it fell from `0.966411-0.967270` to `0.878021-0.886292`, and
uKNIT replica1 fell from `0.691026-0.691860` to `0.613027-0.633577`.

The relative residual amplitude is not a sufficient repair. Dialga replica0's
transition/base RMS ratio increased from about `0.66` to `0.72`, yet its
transition AUC gain still decreased on both fresh splits. K1-AQ therefore did
more than simply make a useful branch too small or too large; it changed the
learned projection and its label alignment.

The final Chinese SVG was rendered at `2160 x 1320` and inspected through
`visual-qa-redraw`. All long labels, paired marks, thresholds, legends, numeric
counts and claim-scope text passed without overlap, clipping or ambiguity. The
run directory contains `visual_qa_render_report.json` and
`visual_qa_passed.marker`.

## 8. Interpretation

K1-AR supports heterogeneous transition demand under shared weights. The same
global transition path is strongly useful for uKNIT and Dialga under K1-AO,
but inverse-norm training reallocates its utility toward Midori and away from
the other two ciphers. One global learned scalar cannot express this family
variation, while fixed task-loss scaling only moves the compromise point.

This does not yet prove that a runtime descriptor can choose a correct gate,
and it does not authorize per-cipher parameters. It narrows the next hypothesis:
derive a bounded gate from cryptographic structure statistics already supplied
at runtime, with one shared parameterization and explicit descriptor-mismatch
controls.

## 9. Executable Next Action

Preregister K1-AS as a local readiness gate, not a training result.

```text
question:
  Can one fixed-width, cipher-ID-free descriptor expose stable differences in
  S-box nonlinearity and GF(2) diffusion that a shared bounded gate can consume?

same-budget anchor:
  K1-AO/K1-AQ K1-AK geometry and the K1-AR path decomposition.

one variable:
  replace the single global transition scalar with
  tanh(global_bias + shared_gate_network(runtime_structure_statistics)).

runtime statistics:
  fixed-width S-box DDT/LAT spectral summaries;
  normalized GF(2) row/column weights, rank and transition diversity;
  no cipher name, ID, lookup table, per-cipher head, adapter or expert.

required controls:
  correct descriptor;
  descriptor from another compatible cipher;
  S-box-only mismatch;
  linear-layer-only mismatch;
  descriptor-disabled global scalar.

readiness scale:
  zero training, deterministic synthetic batches plus the existing 24 frozen
  panels; replicas 0/1; local CPU; optimizer steps 0.

advance gate:
  fixed parameter shape across all three ciphers;
  finite bounded gate and nonzero gradient path;
  cell-relabeling-invariant descriptor;
  correct versus mismatched descriptors produce distinct gates;
  no model state mutation and exact global-scalar replay when disabled.

stop gate:
  descriptor collisions across the three structures, dependence on cipher ID,
  failure of relabeling invariance, unbounded gates, or inability to replay the
  K1-AK global-scalar path exactly.
```

Only after K1-AS readiness passes should one local `2048/class/cipher`,
`4 pairs`, replicas `0/1`, ten-epoch training comparison be opened against the
K1-AO equal-loss shared anchor. Do not tune K1-AQ loss scales, use PCGrad,
increase pairs/data/epochs/width, add MoE/experts, or launch remotely.
