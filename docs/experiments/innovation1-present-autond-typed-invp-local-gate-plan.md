# Innovation 1 PRESENT AutoND Typed InvP Local Gate Plan

**Date:** 2026-07-10

**Status:** design frozen; implementation and run pending plan review

**Claim scope:** local protocol and representation adjudication only; not
paper-scale evidence, not strict-negative evidence, and not an Innovation 1
positive result

## Question

Under the exact AutoND public-code data protocol, does the already supported
PRESENT `InvP(DeltaC)` typed representation add controlled signal beyond:

1. the published AutoND/DBitNet-shaped baseline;
2. the same-capacity shuffled inverse-permutation control; and
3. the same-capacity unaligned `DeltaC` control?

This gate is deliberately separate from the running paper-scale AutoND seed0
reproduction. It does not change that run, use its GPU, or interpret its
unfinished artifacts.

## Why This Gate

The literature re-audit ranks controlled SPN component adaptation above a new
uncontrolled architecture. Existing project evidence also constrains the
design:

- `InvP` is supported at PRESENT r7 under the strict Zhang/Wang Case2 protocol
  at `1000000/class` on two seeds.
- inverse-layer features are prior art, so `InvP` alone is not the novelty.
- corrected E1 active-cell topology failed its matched controls.
- dense DDT and trail-value routes are stopped and must not be reopened.
- AutoND/DBitNet is the direct single-pair PRESENT literature baseline that
  must be understood before another architecture claim.

The remaining defensible question is whether a typed, cipher-spec-derived SPN
adapter has reproducible value when the raw data, labels, keys, optimizer,
training schedule, and evaluation are held fixed.

## Alternatives Considered

### A. Four-row local public-protocol gate

Use the existing baseline and three existing typed/control models. This is the
selected approach because it changes no implementation and gives the cheapest
falsifiable attribution result while the remote baseline runs.

### B. Wait for the paper-scale baseline before doing any method work

This minimizes concurrent research, but yields no new method evidence for
multiple days and does not use available local compute.

### C. Launch a strict Case2 DBitNet comparator immediately

This has higher publication relevance, but is a larger training commitment and
would compete with the active remote lifecycle before the public-code baseline
is understood.

## Frozen Protocol

```text
cipher                      = PRESENT-80
target round                = r9
curriculum                  = [5,6,7,8] -> 9
difference                  = 0x000000000d000000
feature encoding            = ciphertext_pair_bits
pairs per row               = 1
input bits                  = 128
dataset label mode          = random_labels_total
negative mode               = random_ciphertext
key rotation interval       = 1
sample structure            = independent_pairs
train samples total         = 16384 per round
validation samples total    = 4096 per round
epochs per round            = 3
pretrain epochs per round   = 3
batch size                  = 256
optimizer                   = Adam + AMSGrad
optimizer transition        = carry_across_stages
learning rate               = 0.001
loss                        = MSE
checkpoint                  = best val_loss
final test repeats          = 3
final test samples total    = 4096 per repeat
seed                        = 0
device                      = CPU
```

`16384` is the total number of random-label rows per training split, not
`16384/class`. Class counts must be reported from the sampled labels.

## Frozen Matrix

| Row | Model | Role |
| ---: | --- | --- |
| 1 | `autond_dbitnet2023` | published-architecture baseline on the exact same input and budget |
| 2 | `present_nibble_invp_only_spn_only` | true typed `InvP(DeltaC)` candidate |
| 3 | `present_nibble_shuffled_paligned_spn_only` | same-capacity shuffled-structure control |
| 4 | `present_nibble_delta_only_spn_only` | same-capacity unaligned-value control |

No model code, data generator, label definition, negative definition,
validation policy, or metric implementation may change in this experiment.

## Required Artifacts

```text
plan CSV
results.jsonl with exactly four rows
progress.jsonl
validation.json
curves.svg
history.csv
typed gate JSON
disk-backed dataset cache and cache-reuse evidence
```

The gate JSON must record all three fresh-test metrics per row, their
accuracy/AUC mean and population standard deviation, exact label counts, and
optimizer-step continuity for r5-r9.

## Readiness Gates

All rows must satisfy:

```text
train rows = 16384 at r5-r9
validation rows = 4096 at r5-r9
both labels present in every split
dataset_label_mode = random_labels_total
negative_mode = random_ciphertext
key schedule = per_row_random
r5 optimizer reused = false and step_before = 0
r6-r9 optimizer reused = true
adjacent optimizer steps are continuous and strictly increasing
checkpoint metric = val_loss
three fresh tests exist with 4096 rows each
result-plan validation = pass with four rows
cache rerun reports reuse for every requested split
```

Any protocol-integrity failure invalidates metric comparison and must be fixed
before rerunning the same frozen matrix.

## Decision Gate

Let `AUC(model)` be the mean AUC over the three fresh test sets.

```text
strong_local_support:
  AUC(InvP) - max(AUC(AutoND), AUC(shuffled-P), AUC(DeltaC)) >= 0.01

weak_or_fragile:
  InvP is above all three controls, but the minimum margin is in (0, 0.01)

stop_public_typed_adapter:
  InvP is tied with or below any required control
```

The individual repeat values must also be reported. A favorable mean caused by
one outlier repeat is weak/fragile rather than strong support.

## Next Actions

### If strong local support

Run the identical frozen matrix at seed1 locally. Do not launch a larger public
random-ciphertext run. If both seeds pass, design a strict
encrypted-random-plaintext comparator gate, because only the strict protocol
can support the main Innovation 1 claim.

### If weak or fragile

Run seed1 only as a bounded variance adjudication. Continue only if both seeds
retain the same ordering. Do not tune the margin on validation data.

### If stopped

Do not modify pair count, widen the network, add DDT inputs, or scale the same
public-protocol adapter. Retain the existing strict InvP result as
representation evidence and focus the contribution on controlled
cross-protocol attribution.

## Stop Rules

- Do not call this formal training or paper-scale evidence.
- Do not compare its accuracy directly with the paper's `0.5092` result.
- Do not claim novelty from `InvP` alone.
- Do not reopen E1 graph scaling or DDT/beamstats input.
- Do not use the active paper-scale remote task's partial artifacts as labels,
  initialization, feature selection, or a gate for this local experiment.
- Do not launch seed1 or remote scale before the four-row seed0 gate is
  complete, validated, documented, and committed.
