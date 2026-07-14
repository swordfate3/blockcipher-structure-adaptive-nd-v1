# Innovation 1 E4-R6 Conditional Formal-Scale Cross-SPN Adaptation Plan

**Date:** 2026-07-15

**Status:** stopped by completed E4-R5 joint gate / no config / no launch

**Claim scope:** held-out multi-source, multi-target confirmation design at
`1000000/class` target training scale; passing the scale alone does not
authorize SOTA, breakthrough, persistent-final-AUC, or end-to-end compute
claims

## Activation Trigger

E4-R6 remains inactive unless every E4-R5 requirement below is proven from
local retrieved artifacts:

```text
seed4 result archive                = verified and plan-aligned
seed5 result archive                = verified and plan-aligned
joint archive                       = verified
seed4 local gate decision           = e4_r5_target_adaptation_efficiency_confirmed
seed5 local gate decision           = e4_r5_target_adaptation_efficiency_confirmed
joint local decision                = e4_r5_source_seed_robustness_confirmed
source seed1 true/shuffled SHA-256  = frozen manifest values
paired score/bootstrap gates        = pass, errors=[]
experiment/design/verdict docs      = updated and pushed
```

If any item is missing, invalid, or fails, do not create E4-R6 CSVs, remote
configs, launchers, or tasks. Retain E4-R4 as conditional-on-source evidence
and follow the E4-R5 stop decision.

The completed E4-R5 evidence activates that stop branch:

```text
seed4 local decision = e4_r5_target_adaptation_signal_unstable
seed5 local decision = e4_r5_target_adaptation_signal_unstable
joint local decision = e4_r5_source_seed_signal_unstable
joint next action     = stop_formal_scale_retain_conditional_e4_r4_result
```

Therefore this document is retained as a predeclared counterfactual protocol
record. Its activation trigger failed; unfinished implementation items are no
longer a work queue.

## Research Question

On source and target seeds not used to discover or confirm E4-R4/E4-R5:

> Does a correctly typed PRESENT-80 source initialization still improve
> GIFT-64 r6 AUC after exactly one target epoch, relative to the same typed
> architecture trained from scratch and source/target shuffled-topology
> controls, at `1000000/class` target training scale and on fresh held-out
> evaluation datasets?

This tests conditional target-adaptation efficiency. It does not test
persistent superiority after many target epochs and does not claim lower
end-to-end compute.

## Literature And Evidence Position

The 2026-07-10 literature re-audit found prior work for inverse-layer
features, SPN state formatting, multi-pair input, input/network co-design,
attention, U-Net/CNN variants, AutoND, and related-key PRESENT frameworks.
Those ingredients are not novel individually.

The narrower gap retained by the audit is:

```text
cipher-spec-generated typed SPN adaptation
+ shared operator geometry across SPN ciphers
+ explicit same-capacity scratch control
+ source-shuffled and target-shuffled topology controls
+ held-out cross-cipher transfer/adaptation testing
+ strict key-independent encrypted-random-plaintext negatives
```

E4-R6 is therefore an attribution and transfer confirmation, not a generic
"new SPN network" comparison.

## Same-Budget Anchor And One Main Variable

The same-budget anchor is the typed GIFT target model trained from scratch for
exactly one epoch on the same target cache. All candidate and control roles
have identical architecture, parameter count, target data, optimizer,
checkpoint policy, and evaluation data.

Relative to E4-R5, the main experimental variable is target training scale:

```text
65536/class -> 1000000/class
```

Held-out source and target seeds are confirmation units, not additional method
changes. Do not alter the representation, architecture, pair count, negative
definition, keys, optimizer, target epochs, or thresholds.

## Phase A: Held-Out Source Checkpoints

Do not reuse source seeds 0 or 1 as E4-R6 confirmation units. They are
development and E4-R5 evidence. Generate two fresh source units:

```text
cipher/rounds                   = PRESENT-80 r7
source seeds                    = 2, 3
train                           = 8192/class = 16384 total rows/source seed
validation                      = 4096/class = 8192 total rows/source seed
sample structure                = Zhang/Wang Case2 official MCND
pairs/sample                    = 16
negative                        = encrypted random plaintexts
effective key policy            = per_pair_random
epochs                          = 10
batch size                      = 256
hidden bits                     = 32
loss/optimizer                  = MSE / Adam
learning rate / weight decay    = 0.0001 / 0.00001
checkpoint                      = restored best validation AUC
disk cache                      = required and parameter matched
```

Run the frozen four-role source matrix independently for seeds 2 and 3:

| Role | Purpose |
| --- | --- |
| InvP-only anchor | strongest same-input PRESENT reference |
| typed true | transferable source candidate |
| typed shuffled | source-topology control |
| typed raw | typed-capacity/mapping control |

Each held-out source seed must independently pass the E4-R1/E4-R5 source gate:

```text
typed_true AUC                  >= 0.65
typed_true - anchor             >= -0.010
typed_true - typed_shuffled     >= +0.003
typed_true - typed_raw          >= +0.003
```

Freeze true and shuffled checkpoint paths, SHA-256 values, source result rows,
and complete provenance in one E4-R6 source manifest. If either source seed
fails, stop E4-R6 before target-scale training. Do not replace the failed seed,
increase source data, add epochs, or tune the architecture.

## Phase B: Formal-Scale Target Grid

Use two fresh GIFT target seeds:

```text
target seeds                    = 6, 7
train                           = 1000000/class = 2000000 total rows/target seed/role
validation                      = 500000/class = 1000000 total rows/target seed/role
pairs/sample                    = 4 independent pairs
input                           = 512 raw ciphertext-pair bits
negative                        = encrypted random plaintexts
train key                       = 0x00000000000000000000000000000000
validation key                  = 0x11111111111111111111111111111111
final-test key                  = 0x22222222222222222222222222222222
target epochs                   = exactly 1
batch size                      = 256
hidden bits                     = 32
loss/optimizer                  = MSE / Adam
learning rate / weight decay    = 0.0001 / 0.00001
LR scheduler                    = none
checkpoint metric              = validation AUC
checkpoint selection           = restored best checkpoint
device                          = target seed6 GPU0, target seed7 GPU1
disk cache                      = required under each G:\\lxy run root
```

### Lean seven-role matrix per target seed

Train scratch once per target seed, then cross both held-out source seeds with
the required controls:

| Row | Source unit | Initialization | Target mapping | Role |
| ---: | --- | --- | --- | --- |
| 0 | none | scratch | true GIFT | shared same-budget anchor |
| 1 | PRESENT seed2 | true | true GIFT | candidate s2 |
| 2 | PRESENT seed2 | shuffled | true GIFT | source-topology control s2 |
| 3 | PRESENT seed2 | true | shuffled GIFT | target-topology control s2 |
| 4 | PRESENT seed3 | true | true GIFT | candidate s3 |
| 5 | PRESENT seed3 | shuffled | true GIFT | source-topology control s3 |
| 6 | PRESENT seed3 | true | shuffled GIFT | target-topology control s3 |

The old aligned GIFT anchor is excluded. The scientific anchor is the
identical typed architecture trained from scratch. Training scratch twice for
the two source units is also excluded because it adds no source-dependent
information.

Total target training budget:

```text
per target seed  = 7 roles x 2000000 rows = 14000000 row exposures
two target seeds = 28000000 row exposures
```

Source pretraining and all control-row costs must be reported separately.

## Fresh Evaluation Protocol

Validation selects/restores the checkpoint only. The formal decision uses five
fresh final-evaluation datasets per target seed:

```text
repeats                         = 5
rows/repeat                     = 1000000 total = 500000/class
evaluation key                  = final-test key, distinct from train/validation
dataset seeds                   = deterministic target-seed-derived fresh seeds
shared cache                    = identical within target seed/repeat across all 7 roles
raw artifacts                   = labels, probabilities, logits, sample_ids, models.json
reported metrics                = AUC, accuracy, calibrated accuracy, loss
```

Every candidate/control comparison must use identical labels and sample IDs.
Store the restored-checkpoint validation metric separately from all fresh-test
metrics.

## Statistical Gate

For each of the four held-out source-target cells:

```text
(source2,target6)
(source2,target7)
(source3,target6)
(source3,target7)
```

compute on each fresh repeat:

```text
candidate - shared scratch
candidate - matching source-shuffled control
candidate - matching target-shuffled control
```

Hard point gates use the mean across five fresh repeats:

```text
mean(candidate - scratch)         >= +0.004
mean(candidate - source-shuffled) >= +0.005
mean(candidate - target-shuffled) >= +0.003
```

Consistency gates:

```text
all four source-target cells pass every mean margin
no fresh repeat reverses candidate versus any required control
primary repeat uses 10000 paired label-stratified bootstrap replicates
paired 95% CI lower(candidate - scratch) > 0 for every source-target cell
source/target topology paired intervals are reported for every cell
```

The primary repeat and bootstrap seed must be frozen before launch. Do not
select the best repeat after seeing metrics.

## Decision Table

```text
both held-out source gates pass,
all four source-target cells pass all fresh-evaluation gates,
and all protocol/archive checks pass:
  decision    = e4_r6_formal_scale_cross_spn_adaptation_supported
  next_action = write_controlled_cross_spn_method_result_and_variance_analysis

exactly one source-target cell misses a point threshold or CI,
while every required ordering remains positive:
  decision    = e4_r6_formal_scale_signal_unstable
  next_action = stop_scale_retain_e4_r4_r5_conditional_result

either held-out source gate fails,
or a required topology control reverses,
or two or more source-target cells miss:
  decision    = e4_r6_cross_spn_generalization_rejected
  next_action = stop_cross_spn_scale_keep_typed_representation_benchmark

any cache, key, initialization, score-pairing, checkpoint, result-alignment,
or archive-integrity error:
  decision    = invalid_e4_r6_protocol
  next_action = repair_protocol_without_interpreting_metrics
```

## Required Implementation Before Any Launch

The current repository is not yet ready for this formal protocol. The
following are launch blockers, not optional improvements. Status was last
audited on 2026-07-15; completed infrastructure does not activate E4-R6:

1. Add a row-level initialization role so one seven-row plan can select source
   seed2/seed3 true/shuffled checkpoints without architecture aliases or
   duplicate scratch training.
2. **Implemented and locally verified, activation still blocked:**
   `final_test_key` parsing, separate dataset cipher construction, cache
   identity, progress/result metadata, validation-key fallback, and
   fail-closed result-alignment tests.
3. **Implemented and locally verified, activation still blocked:** frozen
   score export supports explicit `final_test_1..N` splits with deterministic
   fresh seeds, final-test key, exact row accounting, disk-cache reuse, and
   aligned sample IDs. Out-of-range repeats fail closed.
4. Implement a seven-role, two-source, two-target gate using fresh-test scores
   and the frozen decision table.
5. Add `64/class` readiness plans that prove all source identities, cache
   creation/reuse, seven checkpoints, 35 fresh score artifacts per target
   seed, and exact result alignment without interpreting readiness metrics.
6. Create remote configs only after the readiness gates pass. Require
   disk-backed train/validation/final-test caches, progress JSONL, resumability,
   exact pushed commit, `cmd.exe /c`, run-owned clean clones, and one local
   tmux retrieval watcher.
7. Require complete verified result archives with SHA-256 manifests before any
   formal interpretation.

## Claim Boundary

A passing E4-R6 permits this narrow statement:

> Under the frozen strict real-vs-random protocol, a cipher-spec-generated
> shared typed SPN operator shows held-out, multi-source and multi-target
> support for faster one-epoch PRESENT-to-GIFT adaptation at
> `1000000/class`, beyond same-capacity scratch and shuffled source/target
> topology controls.

It still does not permit:

```text
SOTA or breakthrough
first use of inverse-layer/SPN features
persistent superiority after 10 target epochs
lower end-to-end training compute
universal transfer across SPN ciphers
key-recovery improvement
PRESENT r8/r9 or AutoND reproduction claims
```

## Explicitly Stopped

- no E4-R6 activation before the retrieved E4-R5 joint pass;
- no local `1000000/class` training;
- no intermediate `262144/class` mechanical scale;
- no source-seed replacement after a failed held-out source gate;
- no extra target epochs, source ensembles, cipher IDs, DDT, trails, S-box
  tables, or new negative definitions;
- no reuse of validation AUC as the final formal metric;
- no duplicate scratch rows merely to fit the current manifest API;
- no launch before independent final-test keys and fresh paired scores work
  end to end.

## Current Next Action

```text
E4-R5 verified retrieval = complete
E4-R5 joint decision      = e4_r5_source_seed_signal_unstable
E4-R6                     = stopped / never activated
next                      = consolidate the conditional E4-R4 result and
                            robust typed-topology attribution without new
                            transfer training
```

References:

- `docs/experiments/innovation1-cross-spn-source-seed-robustness-r5-plan.md`
- `docs/experiments/innovation1-cross-spn-target-adaptation-r4-plan.md`
- `docs/experiments/innovation1-cross-spn-typed-cell-transfer-design.md`
- `docs/experiments/innovation1-route-verdict-2026-07-09.md`
- `docs/research/innovation1-spn-literature-reaudit-20260710.md`
