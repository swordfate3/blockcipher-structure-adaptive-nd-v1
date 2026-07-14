# Innovation 1 E5-R0 Topology-Counterfactual Source Objective Plan

**Date:** 2026-07-15

**Status:** completed / E5-R0 rejected by the frozen two-target-seed gate

## Research Question

Does an explicit true-versus-shuffled SPN topology auxiliary objective during
PRESENT source training improve exactly-one-epoch GIFT adaptation beyond the
same architecture trained only for source classification?

## Same-Budget Anchor

The E5 anchor is the E4 typed-cell model and protocol with the auxiliary scale
set to zero. Every E5 source role uses the same main true PRESENT operator,
classifier, hidden dimensions, parameterized auxiliary head, data, seed,
optimizer, epochs, and restored-best `val_auc` checkpoint rule.

The target comparison uses the same GIFT typed-true model and exactly one
target epoch for all transferred checkpoints. A same-architecture GIFT scratch
row remains required.

## One Changed Variable

```text
source topology auxiliary mode
  off                    = source-classification anchor
  true_vs_shuffled       = E5 candidate
  shuffled_vs_shuffled   = equal-capacity placebo control
```

The auxiliary head exists in all three source roles. With `off`, its loss is
disabled. The candidate receives the cipher-spec true inverse permutation as
the positive view and one deterministic shuffled permutation as the negative
view. The placebo receives two distinct deterministic shuffled permutations.
All views pass through the same typed-cell encoder; the auxiliary head is not
used by the final target classifier.

The auxiliary scale is frozen at `0.10`. E5-R0 does not tune it on GIFT.

## Fixed Protocol

### Source: PRESENT-80 r7

```text
sample structure       = zhang_wang_case2_official_mcnd
difference profile     = present_zhang_wang2022_mcnd
negative               = encrypted_random_plaintexts
key schedule           = per_pair_random
pairs/sample           = 16
loss                   = mse + 0.10 * auxiliary BCE when enabled
optimizer              = Adam
learning rate          = 0.0001
weight decay           = 0.00001
epochs                 = 10
checkpoint metric      = val_auc
restore best           = true
```

### Target: GIFT-64 r6

```text
sample structure       = independent_pairs
difference profile     = gift64_shen2024_spn_screen
negative               = encrypted_random_plaintexts
key schedule           = fixed train key + distinct fixed validation key
pairs/sample           = 4
optimizer              = Adam, reset for target stage
learning rate          = 0.0001
weight decay           = 0.00001
epochs                 = exactly 1
checkpoint metric      = val_auc
restore best           = true
```

No GIFT row, label, score, or metric may participate in source checkpoint
selection.

## Phase 0: Runtime Readiness

```text
source train           = 64/class
source validation      = 32/class
source seeds           = 0
source epochs          = 3, readiness override only
target train           = 64/class
target validation      = 32/class
target seeds           = 2
target epochs          = 1
device                 = local CPU
interpret metrics      = no
```

Readiness passes only if all roles train, restore and write checkpoints, all
losses are finite, candidate/placebo auxiliary loss is present during source
training, the off role has no auxiliary contribution, initialization into
GIFT is strict and plan-aligned, and shared target caches are reused.

## Phase 1: Local Diagnostic Gate

### Source stage

```text
source train           = 8192/class
source validation      = 4096/class
source seeds           = 0 first; seed 1 only after the seed0 transfer gate
source roles/seed      = off, true_vs_shuffled, shuffled_vs_shuffled
epochs                 = 10
device                 = local CPU or local CUDA
```

`8192/class` is a local diagnostic, not formal evidence. It is below the
project's `65536/class` remote-only boundary.

### Target stage

```text
target train           = 8192/class
target validation      = 4096/class
target seeds           = 2, 3, both initialized from source seed0 first
target roles/seed      = scratch, off-transfer, candidate-transfer,
                         placebo-transfer
epochs                 = exactly 1
device                 = local CPU or local CUDA
```

Each target seed must use one disk-backed train cache and one disk-backed
validation cache reused by all four roles. Export aligned validation sample
IDs, labels, logits, probabilities, and checkpoint hashes for paired gates.

### Local advance gate

For each target seed independently:

```text
candidate - off-transfer AUC       >= +0.004
candidate - placebo-transfer AUC   >= +0.004
candidate - scratch AUC            >= +0.004
paired bootstrap 95% CI lower      > 0 for all three contrasts
bootstrap replicates               = 10000, label-stratified and paired
source true-vs-shuffled attribution = finite and positive
```

Advance from Phase 1A requires all criteria on both target seeds while holding
source seed0 fixed. A source-AUC improvement alone cannot pass E5 because the
hypothesis concerns target adaptation.

If and only if Phase 1A passes, Phase 1B independently retrains the three
PRESENT source roles with source seed1, then repeats the three transfer rows on
the same target seed2 and seed3 caches. The existing scratch scores are reused;
they are not retrained. Phase 1B must pass the same candidate-versus-off,
candidate-versus-placebo, and candidate-versus-scratch paired gates on both
target seeds. This staged crossing prevents source seed and target seed from
being confounded without spending the seed1 matrix after a failed seed0 gate.

## Phase 2: Conditional Remote Medium Diagnostic

Only a complete Phase 1 pass authorizes preparation and launch of:

```text
source train           = reuse frozen Phase 1 8192/class checkpoints
target train           = 65536/class
target validation      = 32768/class
target seeds           = two fresh seeds predeclared before launch
target epochs          = exactly 1
execution              = remote lxy-a6000 only
classification         = medium diagnostic, not formal or paper-scale
```

Before launch, the remote path must pass all project gates: pushed commit,
clean run-owned clone, `cmd.exe /c`, all artifacts under `G:\\lxy`, and
parameter-matched disk-backed cache/progress/reuse. A local tmux watcher must
retrieve results; the main thread must not SSH-poll.

The remote gate repeats the three paired candidate contrasts with the same
thresholds and confidence rule. It does not authorize `262144/class` or
`1000000/class` automatically.

## Stop Conditions

Stop and discard this exact objective when any of these holds:

- the candidate fails either local target seed;
- the shuffled-placebo objective ties or beats the candidate;
- candidate gains appear only in source AUC, not target adaptation;
- score pairing, checkpoint provenance, cache reuse, or key separation fails;
- the auxiliary objective changes labels, negatives, target epochs, metric
  computation, or validation data.

Do not rescue a failed gate with auxiliary-scale tuning, extra target epochs,
more source seeds, `262144/class`, `1000000/class`, MAML/Reptile, or joint
PRESENT+GIFT pretraining.

## Planned Artifacts

```text
configs/experiment/innovation1/
  innovation1_spn_present_cross_spn_e5_source_objective_readiness_seed0.csv
  innovation1_spn_present_cross_spn_e5_source_objective_8192_seed0.csv
  innovation1_spn_present_cross_spn_e5_source_objective_8192_seed1.csv
  innovation1_spn_gift64_cross_spn_e5_target_readiness_seed2.csv
  innovation1_spn_gift64_cross_spn_e5_target_8192_source_seed0_target_seed2.csv
  innovation1_spn_gift64_cross_spn_e5_target_8192_source_seed0_target_seed3.csv
  innovation1_spn_gift64_cross_spn_e5_target_8192_source_seed1_target_seed2.csv
  innovation1_spn_gift64_cross_spn_e5_target_8192_source_seed1_target_seed3.csv

outputs/local_smoke/i1_cross_spn_e5_source_objective_readiness/
outputs/local_diagnostic/i1_cross_spn_e5_source_objective_8192_seed0/
outputs/local_diagnostic/i1_cross_spn_e5_source_objective_8192_seed1/
outputs/local_smoke/i1_cross_spn_e5_target_readiness/
outputs/local_diagnostic/i1_cross_spn_e5_target_8192_source_seed0/
outputs/local_diagnostic/i1_cross_spn_e5_target_8192_source_seed1/
```

Each completed result-producing run must refresh
`outputs/00_RECENT_RESULTS.md` before reporting.

## Completed Evidence

### Phase 0 readiness

Both readiness matrices passed before the diagnostic run:

```text
source readiness rows       = 3/3 plan-aligned
target readiness rows       = 4/4 plan-aligned
parameter count             = 196003 for every role
strict transfer loads       = 3/3
off auxiliary loss          = 0
candidate/placebo aux loss  = finite and positive
target auxiliary loss       = 0
target cache create/reuse   = 2/6
```

Artifacts:

```text
outputs/local_smoke/i1_cross_spn_e5_source_objective_readiness/
outputs/local_smoke/i1_cross_spn_e5_target_readiness/
```

### Source seed0 diagnostic

The three source roles completed the frozen PRESENT-80 r7 protocol at
`8192/class`, `4096/class` validation, 16 pairs/sample, and 10 epochs:

| Source role | Validation AUC | Candidate delta | Auxiliary loss first -> final |
| --- | ---: | ---: | ---: |
| off | `0.743810147047` | `-` | `0 -> 0` |
| true-vs-shuffled candidate | `0.729563534260` | `-0.014246612787` vs off | `0.069015298272 -> 0.002792217792` |
| shuffled-vs-shuffled placebo | `0.734507858753` | `-0.004944324493` candidate vs placebo | `0.068998692790 -> 0.003047403119` |

The auxiliary task was learned quickly, but the candidate source AUC was lower
than both controls. Source AUC alone was not used to stop the run because the
predeclared hypothesis concerned target adaptation.

### Target seed2 and seed3 gates

Each target role used `8192/class` GIFT-64 r6 training, `4096/class`
validation, 4 pairs/sample, and exactly one target epoch. All four roles within
a target seed shared paired labels and sample IDs. The intervals below use
10,000 label-stratified paired bootstrap replicates.

| Target seed | Scratch AUC | Off AUC | Candidate AUC | Placebo AUC |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `0.531851321459` | `0.564548403025` | `0.564159154892` | `0.551355123520` |
| 3 | `0.539322316647` | `0.582833975554` | `0.569158434868` | `0.570697873831` |

| Target seed | Candidate - off | 95% CI | Candidate - placebo | 95% CI | Candidate - scratch | 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `-0.000389248133` | `[-0.007461518794, +0.006784310192]` | `+0.012804031372` | `[+0.004685210437, +0.020708804578]` | `+0.032307833433` | `[+0.017682593316, +0.046767933667]` |
| 3 | `-0.013675540686` | `[-0.020925288647, -0.006522831321]` | `-0.001539438963` | `[-0.009335436672, +0.006125137210]` | `+0.029836118221` | `[+0.016395074874, +0.042713388801]` |

The candidate beats scratch on both target seeds, showing that transfer itself
remains useful at this local budget. It does not beat the stronger off-transfer
anchor on either seed, and it fails the placebo control on seed3. Therefore the
new source objective, not transfer in general, is rejected.

```text
status      = pass, errors=[]
decision    = e5_r0_source_objective_rejected
gate_pass   = false
next_action = stop_e5_r0_no_source_seed1_or_remote_scale
```

Artifacts:

```text
outputs/local_diagnostic/i1_cross_spn_e5_source_objective_8192_seed0/
outputs/local_diagnostic/i1_cross_spn_e5_target_8192_source_seed0/
  gate.json
  summary.csv
  curves.svg
  target_seed2/{results.jsonl,history.csv,curves.svg,gate.json,paired_scores.csv.gz}
  target_seed3/{results.jsonl,history.csv,curves.svg,gate.json,paired_scores.csv.gz}
```

## Verdict And Claim Scope

```text
discard = binary true-vs-shuffled topology auxiliary source objective
keep    = E4 shared typed SPN representation and ordinary source transfer anchor
stop    = source seed1 confirmation
stop    = 65536/class remote medium expansion
stop    = 262144/class and 1000000/class expansion
stop    = auxiliary-scale tuning and extra target epochs
scope   = local 8192/class diagnostic with one source seed and two target seeds;
          not medium, formal, paper-scale, SOTA, or breakthrough evidence
```

The E4 descriptive source-AUC mismatch is not strengthened here. Across the
three E5 source roles, the off role has the highest source AUC and also the
highest target AUC on both target seeds. Three source roles are too few to
validate a transferability estimator, but this ordering removes the immediate
justification for spending the next slot on a label-aware checkpoint proxy.

## Recommended Next Action

Design E6-R0 as a new source objective rather than an E5 rescue. The research
question is whether the actual cryptanalytic classification loss can enforce a
functional margin in favor of the cipher-spec true topology over a paired
shuffled topology, avoiding E5's easy topology-identity head.

```text
same-budget anchor     = E5 off-transfer source seed0 and target seed2/3 scores
required controls      = shuffled-vs-shuffled functional-margin placebo,
                         GIFT scratch, auxiliary-off transfer
one changed variable   = source functional topology-margin objective
source scale           = 8192/class, seed0, 10 epochs, local
target scale           = 8192/class, target seeds2/3, exactly 1 epoch, local
advance gate           = candidate >= +0.004 AUC over off, placebo, and scratch
                         with paired 95% CI lower > 0 on both target seeds
conditional next step  = source seed1 local confirmation; only then may a
                         65536/class remote medium plan be prepared
stop                   = no E5 weight tuning, no extra target epochs, no local
                         65536/class run, no 262144/1000000 mechanical scaling
```

E6-R0 must receive its own frozen plan and readiness gate before code or
training. If a label-preserving functional counterfactual cannot be defined
without changing the benchmark, keep E4 as the final controlled Innovation 1
result instead of adding another architecture sweep.
