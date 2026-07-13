# PRESENT Same-Input DBitNet Attribution Design

**Status:** R1 seed0 completed; E3-R1 rejected and closed before seed1/scale
**Date:** 2026-07-13  
**Experiment label:** E3-R1  

## Decision Summary

E3-R1 tests whether the published AutoND DBitNet-2023 architecture is a
stronger learner than the current InvP Token-Mixer when both receive the same
strict PRESENT-80 r7 Zhang/Wang Case2 samples and the same key-independent
`InvP(DeltaC)` information. It also tests whether any DBitNet gain is
attributable to the real PRESENT inverse P-layer rather than model capacity or
an arbitrary bit permutation.

This experiment contains no DDT, trail, beam-search, S-box-table, or
handwritten deterministic feature input. It is a literature-aligned comparator
and attribution gate, not itself a novelty claim.

## Evidence That Motivates The Gate

The completed H2 Case 3 residual diagnostic rejected another fixed Conv2D
adapter:

```text
candidate - anchor     = -0.003814578056 AUC
candidate - shuffled-P = -0.002984225750 AUC
candidate - raw-triple = -0.005418211222 AUC
decision               = reject_h2
```

The public-protocol AutoND typed-InvP audit is also closed, but it used a
single-pair random-ciphertext-negative protocol. It did not answer whether a
DBitNet-shaped learner can outperform the strict 16-pair Case2 InvP anchor on
the same data. The literature re-audit explicitly identifies this same-input
comparator as the remaining prerequisite before another new architecture is
designed.

## Alternatives Considered

### A. Recommended: mapped-delta DBitNet on the strict Case2 matrix

Apply true InvP, a deterministic shuffled permutation, or identity to each
pair's 64-bit ciphertext difference, flatten the 16 mapped differences, and
feed the resulting 1024 bits to the existing public-code-aligned DBitNet-2023
backbone. This changes only the learner and mapping role while preserving the
raw dataset and validation protocol.

### B. Feed all 2048 raw ciphertext bits directly to DBitNet

This is a valid generic baseline but it does not provide a matched
true/shuffled/raw structure attribution test. A result could not distinguish
DBitNet capacity from the InvP representation. E3-R1 rejects it as the main
candidate.

### C. Resume residual-focus DDT/trail processing or add another Conv2D

The residual-focus retry failed before producing any adjudication output, and
the user has stopped DDT exploration. H1/H2 already tested fixed Conv2D
residual variants. E3-R1 does not restart either route.

## Frozen Data Protocol

All four rows use:

```text
cipher                    = PRESENT-80
rounds                    = 7
sample_structure          = zhang_wang_case2_official_mcnd
pairs_per_sample          = 16
feature_encoding          = ciphertext_pair_bits
negative_mode             = encrypted_random_plaintexts
dataset_label_mode        = balanced_per_class
effective key schedule    = per_pair_random
input difference          = Zhang/Wang Case2 difference 0x9
loss                      = mse
optimizer                 = adam
learning rate             = 0.0001
weight decay              = 0.00001
schedule                  = official_cyclic, max lr 0.002
checkpoint metric         = val_auc
restore best checkpoint   = true
early stopping patience   = 8
early stopping min delta  = 0.0001
```

Configured key fields remain deterministic cache identities. The result gate
must verify effective `per_pair_random` behavior from generated train and
validation metadata.

## Frozen Four-Role Matrix

| Role | Model input and learner | Purpose |
| --- | --- | --- |
| `anchor` | existing InvP Token-Mixer | strongest strict same-protocol anchor |
| `candidate` | DBitNet-2023 over 16 flattened `InvP(DeltaC)` words | matched prior-art-shaped learner |
| `shuffled_p` | identical DBitNet-2023 over fixed shuffled-P differences | topology attribution control |
| `raw_delta` | identical DBitNet-2023 over raw `DeltaC` words | representation control |

The three DBitNet rows must have identical parameter counts and identical
initial trainable tensors under a common seed. Only the fixed, non-persistent
64-bit mapping buffer may differ.

## Candidate Architecture

For every raw sample:

```text
[batch, 16 pairs, C0||C1]
  -> reshape [batch, 16, 2, 64]
  -> DeltaC = C0 xor C1
  -> mapping(DeltaC), independently for each pair
  -> flatten [batch, 16 * 64] = [batch, 1024]
  -> AutoNDDBitNet2023Distinguisher(input_bits=1024)
  -> one logit
```

The candidate reuses the repository's existing DBitNet-2023 implementation,
including its dilation schedule, Keras-style initialization, batch
normalization, dense classifier, and dense-kernel L2 auxiliary loss. No pair
attention, additional pooling head, widened classifier, residual fusion, or
new hyperparameter is allowed.

## Execution Ladder

### R0: readiness

```text
training samples_per_class   = 64
validation samples_per_class = 32
seed                         = 0
epochs                       = 1
batch_size                   = 32
device                       = cpu
```

R0 checks mapping semantics, three-way capacity/initialization equality,
finite forward/backward, registry construction, dataset cache reuse, effective
key metadata, exact four-row plan alignment, complete histories/checkpoints,
plot generation, and neutral gate replay. R0 metrics are not interpreted.

### R1: local seed0 diagnostic

```text
training samples_per_class   = 8192
validation samples_per_class = 4096
seed                         = 0
epochs                       = 10
batch_size                   = 256
device                       = cpu
```

This is a local diagnostic, not formal SPN/PRESENT training.

### R2: conditional local seed1

Run the identical `8192/class` seed1 matrix only when R1 returns
`promote_seed1`. No `65536/class`, `262144/class`, remote GPU, or formal run is
authorized by a one-seed result.

## Completed R0 Readiness Evidence

Run `i1_present_same_input_dbitnet_smoke_seed0` completed locally on
2026-07-13. It used `64/class` training, `32/class` validation, 16 pairs per
sample, seed0, one epoch, CPU, strict encrypted-random-plaintext negatives, and
effective `per_pair_random` keys verified from train/validation result and
cache metadata.

The exact four-row plan aligned with all four result rows. The strict readiness
gate returned:

```text
status                    = pass
decision                  = implementation_ready
research_decision_applied = false
errors                    = []
next_action               = run_frozen_e3_r1_seed0_local_diagnostic
```

One train cache and one validation cache were created, then reused for all
three DBitNet rows:

```text
create_count        = 2
reuse_count         = 6
control_reuse_count = 6
```

The true-InvP, shuffled-P, and raw-Delta DBitNet rows each had `797633` total
and trainable parameters. The unchanged Token-Mixer anchor had `128673`.
Readiness therefore verifies equal capacity among the three DBitNet attribution
roles without requiring artificial parameter matching to a different published
architecture.

All four rows completed one epoch and restored epoch 1. The generated SVG
parsed as XML and contained four distinct visible model labels. R0 AUCs are
deliberately not interpreted.

Verified artifacts:

```text
outputs/local_smoke/i1_present_same_input_dbitnet_smoke_seed0/results.jsonl
outputs/local_smoke/i1_present_same_input_dbitnet_smoke_seed0/progress.jsonl
outputs/local_smoke/i1_present_same_input_dbitnet_smoke_seed0/validation.json
outputs/local_smoke/i1_present_same_input_dbitnet_smoke_seed0/history.csv
outputs/local_smoke/i1_present_same_input_dbitnet_smoke_seed0/curves.svg
outputs/local_smoke/i1_present_same_input_dbitnet_smoke_seed0/readiness_gate.json
outputs/local_cache/i1_present_same_input_dbitnet_smoke_seed0/
```

This readiness evidence authorizes only the frozen local R1 seed0 diagnostic.
It does not authorize seed1, remote execution, or a larger sample budget.

## Completed R1 Seed0 Adjudication

Run `i1_present_same_input_dbitnet_8192_seed0` completed locally on
2026-07-13. It used the frozen PRESENT-80 r7 Zhang/Wang Case2 protocol with
`8192/class` training, `4096/class` validation, 16 pairs/sample, seed0, 10
epochs, CPU, strict encrypted-random-plaintext negatives, and effective
`per_pair_random` keys verified from result and cache metadata.

Best-checkpoint validation AUCs were:

| Role | AUC |
| --- | ---: |
| InvP Token-Mixer anchor | `0.7505359351634979` |
| true-InvP DBitNet candidate | `0.5168226957321167` |
| shuffled-P DBitNet control | `0.5097089111804962` |
| raw-Delta DBitNet control | `0.5138811767101288` |

The frozen margins were:

```text
candidate - anchor     = -0.23371323943138123
candidate - shuffled-P = +0.007113784551620483
candidate - raw-Delta  = +0.002941519021987915
```

The strict gate returned:

```text
status      = pass
decision    = reject_e3_r1
errors      = []
next_action = stop_dbitnet_component_gate_and_keep_token_mixer_anchor
```

The gate passed every protocol, cache, capacity, history, checkpoint, and
semantic check. One train cache and one validation cache were created and then
reused six times across the remaining rows. The three DBitNet roles each had
`797633` total/trainable parameters and differed only in their fixed mapping.

This is an architecture/generalization rejection, not an interrupted-training
failure. All rows completed 10 epochs and restored their best `val_auc`
checkpoint. The Token-Mixer selected epoch 10 and finished with train AUC
`0.7773294225335121` and validation AUC `0.7505359351634979`. In contrast, the
three DBitNet roles nearly memorized the training set while remaining near
random on validation:

| Role | Best epoch | Final train AUC | Best validation AUC |
| --- | ---: | ---: | ---: |
| true-InvP DBitNet | 2 | `0.9998856782913208` | `0.5168226957321167` |
| shuffled-P DBitNet | 3 | `1.0` | `0.5097089111804962` |
| raw-Delta DBitNet | 7 | `0.9998063743114471` | `0.5138811767101288` |

True InvP is above both DBitNet mapping controls, but its raw-Delta margin
misses the required `+0.003` threshold and it loses to the Token-Mixer anchor
by about `0.234` AUC. Additional epochs cannot address this observed
generalization failure, and this one-seed `8192/class` diagnostic does not
establish a DBitNet or PRESENT ceiling.

Verified artifacts:

```text
outputs/local_smoke/i1_present_same_input_dbitnet_8192_seed0/results.jsonl
outputs/local_smoke/i1_present_same_input_dbitnet_8192_seed0/progress.jsonl
outputs/local_smoke/i1_present_same_input_dbitnet_8192_seed0/validation.json
outputs/local_smoke/i1_present_same_input_dbitnet_8192_seed0/history.csv
outputs/local_smoke/i1_present_same_input_dbitnet_8192_seed0/curves.svg
outputs/local_smoke/i1_present_same_input_dbitnet_8192_seed0/e3_r1_gate.json
outputs/local_smoke/i1_present_same_input_dbitnet_8192_seed0/checkpoints/
outputs/local_cache/i1_present_same_input_dbitnet_8192_seed0/
```

Stopped actions:

```text
seed1            = stopped
65536/class      = stopped
262144/class     = stopped
remote scale     = stopped
```

The next method step is not another PRESENT-only architecture variant. Audit
of the existing three-seed GIFT-64 aligned-input diagnostic shows that a
GIFT-only rerun is also unjustified. The remaining bounded research direction
is a cipher-spec-generated typed adapter with shared operators and an explicit
PRESENT-to-GIFT transfer/attribution gate, or consolidation of Innovation 1 as
a controlled structural-representation methodology result if that design
cannot define a genuinely shared hypothesis.

## Strict Gate

Let AUCs come from restored best-`val_auc` checkpoints. R1 promotes seed1 only
when:

```text
candidate - anchor     >= +0.003 AUC
candidate - shuffled-P >= +0.003 AUC
candidate - raw-Delta  >= +0.003 AUC
```

Decision table:

| Condition | Decision | Next action |
| --- | --- | --- |
| all protocol checks and three margins pass | `promote_seed1` | run identical local seed1 |
| candidate exceeds every row but misses a margin | `weak_or_fragile_no_scale` | stop E3-R1 after one history inspection |
| candidate ties or loses any comparator | `reject_e3_r1` | keep Token-Mixer anchor; stop DBitNet component gate |
| protocol, cache, history, capacity, or metadata mismatch | `invalid_protocol` | repair and rerun the same matrix |

If both seeds later remain above all comparators with at least `+0.001` versus
anchor and `+0.002` versus both mapping controls, E3-R1 may prepare a separate
`65536/class` medium diagnostic plan. That plan still would not be formal
evidence. Formal SPN/PRESENT claims require at least `1000000/class`, multiple
seeds, completed/retrieved/plan-aligned artifacts, and the declared claim gate.

## Required Artifacts

Each executed rung must produce or validate:

```text
results.jsonl
progress.jsonl
validation.json
history.csv
curves.svg
strict gate JSON
disk-backed train and validation caches
cache metadata and progress/completion events
best checkpoints and complete per-row histories
```

## Stop Boundary

E3-R1 does not authorize:

- H2 seed1 or H2 scale-up;
- raw-triple scale-up;
- residual-focus retry or checkpoint repair;
- dense DDT, trail-position, or beamstats reopening;
- pair-count sweeps;
- claims that an `8192/class` or `65536/class` result establishes a PRESENT
  ceiling.

If E3-R1 fails, the next method decision is not another PRESENT-only network
variant. The project must either test cipher-spec-generated typed operators
across more than one SPN under strict attribution controls or consolidate the
current evidence as a controlled structural-representation methodology result.
