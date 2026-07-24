# Innovation 1 uKNIT U2-A Same-checkpoint Counterfactual Plan

Date: 2026-07-24

## Status

```text
stage    = preregistered
run_id   = i1_rtg1_uknit64_runtime_e4_sbox_assignment_u2a_same_checkpoint_20260724
training = none
decision = pending
```

## Question

Does each trained U1 `late_cell` candidate actually use the supplied uKNIT
S-box-to-cell ownership at inference time, or was U1 dominated by separately
trained optimization variance?

U1 trained correct and shuffled models separately. Matched seeds and parameter
geometry constrain that comparison, but their weights diverge during training.
U2-A removes this remaining ambiguity by changing only the runtime structure on
one frozen checkpoint and one frozen validation array.

## Frozen Sources

```text
U1 root       = outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u1_2048_seed0_seed1_20260724
source rows   = seed0/seed1 correct late_cell and correct late_pair
checkpoints   = each source row's selected best checkpoint
validation    = each seed's exact disk-backed U1 validation cache
descriptor    = configs/runtime/spn/uknit64.json
window        = round_start 2, processor_steps 2
shuffle seed  = 20260724
device        = local CPU
```

No fitting, optimizer, scheduler, checkpoint selection, data generation or
threshold tuning is allowed in U2-A.

## Eight Evaluations

For each seed and each source role evaluate the same checkpoint twice:

| Source role | Correct structure | Counterfactual structure |
| --- | --- | --- |
| `late_cell` candidate | true S-box ownership | deterministic shuffled ownership |
| `late_pair` anchor | true S-box ownership | the same deterministic shuffle |

The counterfactual preserves every S2/L2 and S3/L3 GF(2) linear matrix and each
round's S-box multiset. Only S-box-to-cell assignment changes. The source
checkpoint SHA256, validation feature SHA256 and validation label SHA256 must be
identical inside each correct/shuffled pair.

## Protocol Gate

Require:

1. Exactly eight rows: two seeds, two source roles and two structure modes.
2. Every pair uses one identical checkpoint and one identical validation array.
3. All source checkpoints report `selected_checkpoint=best`.
4. The descriptor window remains S2/L2 and S3/L3.
5. Every evaluation contains `2048` validation rows and finite AUC/probabilities.
6. All rows state `training_performed=false` and retain equal parameter geometry.
7. All input/checkpoint/descriptor hashes are present and internally consistent.

Any protocol failure makes the AUC comparison invalid and requires repair
without changing sources or thresholds.

## Research Gate

For each seed define:

```text
candidate_margin = late_cell AUC(correct) - late_cell AUC(shuffled)
anchor_auc_delta = abs(late_pair AUC(correct) - late_pair AUC(shuffled))
anchor_prob_delta= max abs per-sample late_pair probability difference
```

Advance only if all conditions hold:

```text
seed0 candidate_margin >= +0.005
seed1 candidate_margin >= +0.005
seed0 anchor_auc_delta <= 1e-12
seed1 anchor_auc_delta <= 1e-12
seed0 anchor_prob_delta <= 1e-6
seed1 anchor_prob_delta <= 1e-6
```

A pass means trained `late_cell` representations use ownership but separately
trained U1 controls were noisy. It opens a new local paired structure-swap
training objective plan at the same `2048/class` budget.

A miss with valid anchor invariance means the trained representation does not
reliably use ownership. It closes additive `late_cell` injection and opens only
a local edge-conditioned or gated S-box/topology interaction design. A failed
anchor invariance check invalidates the audit implementation and must be fixed
before any model decision.

## Blocked Actions

Do not train a redesign, increase U1 samples/epochs, launch remote GPU, add DDT
or trail features, change the uKNIT window, or claim a uKNIT attack before this
audit is completed and plan-aligned.

## Outputs

```text
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u2a_same_checkpoint_20260724/results.jsonl
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u2a_same_checkpoint_20260724/progress.jsonl
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u2a_same_checkpoint_20260724/validation.json
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u2a_same_checkpoint_20260724/gate.json
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u2a_same_checkpoint_20260724/summary.json
outputs/local_audits/i1_rtg1_uknit64_runtime_e4_sbox_assignment_u2a_same_checkpoint_20260724/curves.svg
```

The final chart must pass `visual-qa-redraw`, and the completed audit must become
the newest entry in both recent-result indexes.

## Protocol Repair Amendment

The first execution on 2026-07-24 was protocol-invalid before research
adjudication. The `late_pair` anchor's S-box context used a float32 row mean.
Correct and shuffled assignments contain the same 16 encoded S boxes, but their
different row order changed the finite-precision reduction by at most
`1.1920928955078125e-07`. That preserved the probability gate but perturbed AUC
by `4.76837158203125e-07` to `9.5367431640625e-07`, violating the preregistered
exact anchor-AUC gate.

Repair only the `late_pair` reduction by accumulating the already encoded
float32 rows in float64 and casting the invariant mean back to the model dtype.
Do not change the `late_cell` candidate path, checkpoints, validation arrays,
shuffle, metrics, thresholds, or eight-row panel. Retain the invalid attempt as
debug evidence and rerun to a fresh output directory before interpreting any
candidate result.
