# Innovation 1 E6-R0 Functional Topology-Margin Source Objective Plan

**Date:** 2026-07-15

**Status:** completed / E6-R0 rejected by the frozen two-target-seed gate

## Research Question

Can a label-preserving source objective that makes the cipher-spec true SPN
topology achieve lower cryptanalytic classification loss than paired shuffled
topologies improve exactly-one-epoch GIFT adaptation beyond the E5 auxiliary-
off transfer anchor?

E6 addresses E5-R0's observed failure mode. The E5 topology-identity BCE fell
from about `0.069` to `0.003`, yet its candidate lost source AUC and did not
beat the off-transfer target anchor. E6 therefore removes topology-identity
prediction from the research hypothesis. It asks a functional question using
the same differential-neural labels and main classification loss.

## Same-Budget Anchor And Reuse

The anchor remains the E5 auxiliary-off source seed0 checkpoint and its GIFT
target seed2/3 transfers:

```text
source anchor = present_cross_spn_typed_cell_e5_off
target anchor = gift_cross_spn_typed_cell_e5_from_present_off
scratch       = gift_cross_spn_typed_cell_e5_scratch
parameters    = 196003 for every source and target role
```

The existing E5 source-off checkpoint is reused only when strict state-dict,
architecture, checkpoint hash, seed, and source-protocol checks pass. The
target scratch, off, candidate, and placebo rows are rerun together in each new
E6 target plan. This preserves one plan-aligned result matrix and one paired
score provenance chain; the current matrix runner cannot safely splice old and
new target rows into a single plan-aligned run. All four target rows still
reuse one new train cache and one new validation cache per target seed.

## One Changed Variable

```text
source functional topology-margin mode
  off                      = true-topology main classification only
  true_vs_shuffled_margin  = E6 candidate
  shuffled_vs_shuffled     = same-compute directional placebo
```

For each labeled PRESENT source batch, all roles retain the true-topology main
classification path. Candidate and placebo roles additionally evaluate two
distinct deterministic shuffled cipher-spec permutations with the same shared
encoder and classifier.

Let `ell(view, y)` be the per-sample MSE classification loss already used by
the source task. The frozen auxiliary terms are:

```text
candidate = mean(relu(0.01 + ell(true, y)
                      - mean(ell(shuffled_a, y), ell(shuffled_b, y))))

placebo   = mean(relu(0.01 + ell(shuffled_a, y)
                      - ell(shuffled_b, y)))

total     = mean(ell(true, y)) + 0.10 * functional_margin_loss
```

The two shuffled permutations, margin `0.01`, and scale `0.10` are fixed before
target training. They may not be tuned from GIFT metrics. All roles retain the
same parameterized E5 auxiliary head for exact parameter-count and strict-load
compatibility, but that head contributes zero loss in E6. Candidate and
placebo use the same number of counterfactual forward paths. The off anchor is
allowed to reuse its frozen E5 result because its mathematical objective is
unchanged; compute matching is enforced between candidate and placebo.

## Fixed Benchmark Protocol

### Source: PRESENT-80 r7

```text
sample structure       = zhang_wang_case2_official_mcnd
difference profile     = present_zhang_wang2022_mcnd
negative               = encrypted_random_plaintexts
key schedule           = per_pair_random
pairs/sample           = 16
main loss              = MSE
optimizer              = Adam
learning rate          = 0.0001
weight decay           = 0.00001
epochs                 = 10
checkpoint metric      = true-topology val_auc
restore best           = true
```

### Target: GIFT-64 r6

```text
sample structure       = independent_pairs
difference profile     = gift64_shen2024_spn_screen
negative               = encrypted_random_plaintexts
key schedule           = fixed train key + distinct fixed validation key
pairs/sample           = 4
optimizer              = Adam, reset at target stage
learning rate          = 0.0001
weight decay           = 0.00001
epochs                 = exactly 1
checkpoint metric      = val_auc
restore best           = true
functional margin loss = disabled
```

No GIFT row, label, score, or metric may affect source training, source
checkpoint selection, loss scale, margin, or shuffled permutations.

## Phase 0: Local Readiness

```text
source train           = 64/class
source validation      = 32/class
source seed            = 0
source epochs          = 3 readiness override
target train           = 64/class
target validation      = 32/class
target seed            = 2
target epochs          = exactly 1
device                 = local CPU
interpret metrics      = no
```

Readiness passes only if:

- all three source roles have exactly 196,003 parameters and compatible state
  dict keys;
- off functional-margin loss is exactly zero;
- candidate/placebo functional-margin losses are finite and participate in
  backpropagation;
- the candidate reads the cipher-spec true permutation plus two distinct fixed
  shuffles, while placebo uses the same two shuffles without a true-vs-shuffled
  margin;
- target strict checkpoint loads pass, the target functional loss is zero, and
  shared target train/validation caches are reused;
- no label, negative definition, validation split, metric, or plan-alignment
  behavior changes.

## Phase 1A: Local Source Seed0 Diagnostic

### Source stage

```text
train                  = 8192/class
validation             = 4096/class
source seed            = 0
new rows               = candidate, shuffled-placebo
frozen reused source   = E5 off checkpoint/result as the source anchor
epochs                 = 10
execution              = local CPU or local CUDA
```

The result must record main loss, functional-margin loss, true/shuffled view
losses, source `val_auc`, checkpoint provenance, and the two deterministic
shuffle identities. Source AUC alone cannot pass or fail E6.

### Target stage

```text
train                  = 8192/class
validation             = 4096/class
target seeds           = 2, 3
rows/seed              = scratch, off-transfer, candidate-transfer,
                         placebo-transfer; all rerun in one E6 target plan
epochs                 = exactly 1
execution              = local CPU or local CUDA
```

All four target roles per seed must use identical cached rows, labels, sample
IDs, key protocol, and score ordering. Export labels, sample IDs, logits,
probabilities, checkpoint hashes, JSONL, history CSV, Chinese SVG, paired-score
CSV, per-seed gate JSON, and a two-seed joint gate.

## Advance Gate

For each target seed independently:

```text
candidate - off-transfer AUC       >= +0.004
candidate - placebo-transfer AUC   >= +0.004
candidate - scratch AUC            >= +0.004
paired bootstrap 95% CI lower      > 0 for all three contrasts
bootstrap replicates               = 10000, label-stratified and paired
source functional losses           = finite and protocol-aligned
```

Phase 1A passes only if both target seeds pass every criterion. A source-loss
margin or source-AUC gain without target adaptation is insufficient.

If and only if Phase 1A passes, repeat the source candidate/placebo with source
seed1 and transfer onto the same target seed2/3 caches. Reuse scratch again.
The independent source seed must pass the same two-target-seed gate before any
remote plan is prepared.

## Conditional Remote Boundary

A complete two-source-seed local pass authorizes design, not automatic launch,
of a `65536/class` target medium diagnostic. That run must execute on
`lxy-a6000`; `65536/class` may not run locally. Before launch it requires a
pushed commit, clean run-owned clone, `cmd.exe /c`, all generated paths under
`G:\lxy`, parameter-matched disk-backed cache/progress/reuse, a bounded start
confirmation, and a local tmux retrieval watcher.

The local pass does not authorize `262144/class` or `1000000/class`.

## Stop Conditions

Stop E6-R0 immediately when any of the following holds:

- candidate fails off-transfer, placebo, or scratch on either target seed;
- candidate/placebo compute or parameter counts differ;
- functional-margin loss changes the main labels, negatives, validation rows,
  key protocol, target epochs, or metric computation;
- target labels influence source objective/checkpoint selection;
- shuffled identities are not deterministic and recorded;
- score pairing or checkpoint provenance fails.

Do not rescue a failed gate with margin/scale tuning, extra target epochs,
extra source seeds, local `65536/class`, remote `65536/class`, `262144/class`,
or `1000000/class`.

## Planned Artifacts

```text
configs/experiment/innovation1/
  innovation1_spn_present_cross_spn_e6_functional_margin_readiness_seed0.csv
  innovation1_spn_present_cross_spn_e6_functional_margin_8192_seed0.csv
  innovation1_spn_gift64_cross_spn_e6_target_readiness_seed2.csv
  innovation1_spn_gift64_cross_spn_e6_target_8192_source_seed0_target_seed2.csv
  innovation1_spn_gift64_cross_spn_e6_target_8192_source_seed0_target_seed3.csv

outputs/local_smoke/i1_cross_spn_e6_functional_margin_readiness/
outputs/local_smoke/i1_cross_spn_e6_target_readiness/
outputs/local_diagnostic/i1_cross_spn_e6_functional_margin_8192_seed0/
outputs/local_diagnostic/i1_cross_spn_e6_target_8192_source_seed0/
```

## Recommended Next Action

Implement only the three frozen E6 source modes, metric recording, and narrow
readiness tests. Run Phase 0 locally. If and only if readiness passes, continue
automatically to the two-row `8192/class` source diagnostic and the two target
seed gates using the reusable E5 anchors. Do not prepare or launch a remote run
until the complete local gate passes.

## Phase 0 Completion

Both local CPU readiness gates passed without interpreting smoke metrics:

```text
source rows                    = 3/3 plan-aligned
source parameter count         = 196003 for every role
source checkpoint state keys   = 59 and identical
off auxiliary loss             = 0
candidate auxiliary loss       = finite, positive
candidate functional loss gap  = +0.000730 -> +0.003772
placebo auxiliary loss         = finite, positive

target rows                    = 4/4 plan-aligned
strict checkpoint loads        = 3/3
target auxiliary loss max      = 0
cache create/reuse              = 2/6
```

```text
source decision = e6_source_functional_margin_readiness_pass
target decision = e6_target_readiness_pass
next action     = run E6 local 8192/class source and two-target-seed gate
```

Artifacts:

```text
outputs/local_smoke/i1_cross_spn_e6_functional_margin_readiness/
outputs/local_smoke/i1_cross_spn_e6_target_readiness/
```

Phase 0 authorizes only the planned local `8192/class` diagnostic. It does not
authorize a local or remote `65536/class` run.

## Phase 1A Source Seed0 Completion

The two new source rows completed at `8192/class`, `4096/class` validation,
16 pairs/sample, and 10 epochs. The frozen E5 off checkpoint supplies the
unchanged source anchor:

```text
off AUC                         = 0.743810147047
functional-margin candidate AUC = 0.743393987417
shuffled-placebo AUC            = 0.742673695087
candidate - off                 = -0.000416159630
candidate - placebo             = +0.000720292330
candidate final functional gap  = +0.078486087499
placebo final functional gap    = +0.003765127854
```

The candidate learned a much larger true-versus-shuffled functional loss gap
without the large source-AUC degradation observed in E5. Source AUC remains a
recorded diagnostic, not the E6 decision gate.

```text
status      = pass, errors=[]
decision    = proceed_to_e6_target_adaptation_gate
state keys  = 59 across off, candidate, and placebo checkpoints
next action = run target seeds2 and 3 at 8192/class, exactly one epoch
```

Artifacts:

```text
outputs/local_diagnostic/i1_cross_spn_e6_functional_margin_8192_seed0/
  results.jsonl
  history.csv
  curves.svg
  progress.jsonl
  gate.json
  initialization_manifest.json
  checkpoints/
```

This source result authorizes only the two predeclared local GIFT target cells.
It does not authorize source seed1 or any `65536/class` experiment.

## Phase 1A Target Completion

Both target cells completed at GIFT-64 r6, `8192/class`, `4096/class`
validation, 4 pairs/sample, and exactly one target epoch. Every role had
196,003 parameters, target auxiliary loss was zero, checkpoint loads were
strict, and the four validation score artifacts within each seed had identical
labels and sample IDs.

| Target seed | Scratch AUC | Off AUC | Candidate AUC | Placebo AUC |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `0.531851321459` | `0.564548403025` | `0.564498603344` | `0.565623760223` |
| 3 | `0.539322316647` | `0.582833975554` | `0.582636594772` | `0.583438843489` |

The 10,000-replicate label-stratified paired bootstrap gate produced:

| Target seed | Candidate - off | 95% CI | Candidate - placebo | 95% CI | Candidate - scratch | 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `-0.000049799681` | `[-0.001978165656, +0.001898437738]` | `-0.001125156879` | `[-0.003453562409, +0.001162195951]` | `+0.032647281885` | `[+0.017801249027, +0.047012680024]` |
| 3 | `-0.000197380781` | `[-0.002259635925, +0.001862239838]` | `-0.000802248716` | `[-0.003121675551, +0.001497677714]` | `+0.043314278126` | `[+0.029751022905, +0.056556243449]` |

The candidate beats scratch on both seeds, confirming that ordinary source
transfer remains useful. It does not beat off-transfer or the same-compute
placebo on either seed. A large source functional gap therefore does not imply
better one-epoch cross-SPN adaptation.

```text
status      = pass, errors=[]
decision    = e6_r0_functional_margin_rejected
gate_pass   = false
next_action = stop_e6_r0_no_source_seed1_or_remote_scale
```

Artifacts:

```text
outputs/local_diagnostic/i1_cross_spn_e6_target_8192_source_seed0/
  gate.json
  summary.csv
  curves.svg
  target_seed2/{results.jsonl,history.csv,curves.svg,gate.json,paired_scores.csv.gz,scores/}
  target_seed3/{results.jsonl,history.csv,curves.svg,gate.json,paired_scores.csv.gz,scores/}
```

## Final Verdict

```text
discard = functional true-topology-vs-shuffled source margin objective
keep    = E4 shared typed SPN representation and ordinary off-transfer anchor
stop    = source seed1 confirmation
stop    = 65536/class remote medium expansion
stop    = 262144/class and 1000000/class expansion
stop    = functional margin/scale tuning and extra target epochs
scope   = local 8192/class diagnostic with one source seed and two target seeds;
          not medium, formal, paper-scale, SOTA, or breakthrough evidence
```

## Evidence-Backed Next Action

Do not create E7 or another source-objective network. Run one no-new-training
E4/E5/E6 synthesis over the verified paired gates:

```text
question            = which Innovation 1 effects survive all source objectives?
same-budget anchor  = target seeds2/3, 8192/class, exactly one epoch
required controls   = scratch, off-transfer, objective-specific placebo
one analysis axis   = off, E5 topology-identity BCE, E6 functional margin
new training        = none
advance decision    = freeze paper-ready method/limitations claim
stop                = no E5/E6 rescue tuning, no source seed1, no remote scale,
                      no additional architecture/source-objective sweep
```

Combine that two-seed objective table with the already verified E4 four-cell
medium topology-attribution synthesis. The supported Innovation 1 result is
expected to be a controlled representation claim, not a new source-objective
or source-seed-robust scratch-efficiency claim.
