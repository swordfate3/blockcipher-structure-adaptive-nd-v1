# PRESENT Case 3 Three-Channel Topology-Residual Design

**Status:** R0 implementation readiness passed; R1 seed0 authorized
**Date:** 2026-07-13  
**Experiment label:** H2  

## Decision Summary

H2 tests whether the Liu Case 3 representation
`(C0, C1, InvP(C0 xor C1))` supplies complementary local information to the
existing PRESENT InvP Token-Mixer. The experiment keeps the H1 backbone,
classifier, residual fusion, data generator, training budget, validation data,
and checkpoint rule fixed. Only the local adapter input changes from one
difference channel to three Case 3 channels.

H2 starts with a local CPU readiness run and then a local `8192/class`, seed0
diagnostic. Seed1 is authorized only if the candidate clears all three frozen
`+0.003` AUC margins. No remote or larger run is authorized by this design.

## Completed R0 Readiness Evidence

Run `i1_present_case3_topology_residual_smoke_seed0` completed locally on
2026-07-13 from pushed feature head `05a5793`. It used `64/class` training,
`32/class` validation, 16 pairs/sample, one epoch, seed0, CPU, strict encrypted
random plaintext negatives, and effective `per_pair_random` keys verified from
the generated cache and result metadata.

The exact four-row plan aligned with all four result rows. The strict readiness
gate returned:

```text
status                    = pass
decision                  = implementation_ready
research_decision_applied = false
errors                    = []
next_action               = run_frozen_h2_seed0_local_diagnostic
```

One train cache and one validation cache were created, then reused for all
three hybrid rows (`create_count=2`, `reuse_count=6`,
`control_reuse_count=6`). The three H2 hybrids each had `137698` total and
trainable parameters; the unchanged anchor had `128673`. The generated SVG
parsed successfully and visibly contained all four distinct role labels.

Verified artifacts:

```text
outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/results.jsonl
outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/progress.jsonl
outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/validation.json
outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/history.csv
outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/curves.svg
outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/readiness_gate.json
outputs/local_cache/i1_present_case3_topology_residual_smoke_seed0/
```

R0 metrics are intentionally not interpreted. This evidence authorizes only
the frozen local `8192/class`, seed0 R1 diagnostic.

## Research Question

H1 added a one-channel `InvP(DeltaC)` state-matrix residual to the strongest
same-protocol Token-Mixer anchor. At `8192/class`, its candidate exceeded the
anchor by `0.002347737551` and the shuffled-P control by `0.002062708139`, but
exceeded the raw-DeltaC control by only `0.000021666288`. The strict gate
therefore returned `weak_or_fragile_no_scale`.

H2 asks a narrower question:

> Does preserving the two original ciphertext states alongside the
> key-cancelled, inverse-permuted difference produce a complementary residual
> signal that cannot be explained by adapter capacity, raw ciphertext content,
> or an arbitrary bit permutation?

This is an architecture-attribution diagnostic. It is not formal training, a
paper reproduction, a PRESENT ceiling result, or breakthrough evidence.

## Paper Semantics And Correction

Liu et al. divide SPN input construction according to the location and coverage
of the last-round key addition. PRESENT performs full-state key addition as the
final round operation, so it follows Case 3. For unknown keys, the individual
ciphertexts cannot be mapped to unbiased previous-round states. XOR cancels the
key only in the ciphertext difference.

The true H2 local tensor is therefore exactly:

```text
channel 0 = C0
channel 1 = C1
channel 2 = InvP(C0 xor C1)
shape     = [batch, pair, 3, 4, 16]
```

H2 must not use `(InvP(C0), InvP(C1), InvP(C0 xor C1))`. That representation
would retain unknown key masks in the first two channels and is especially
incorrect under the effective `per_pair_random` key schedule of the current
Zhang/Wang generator.

## Considered Approaches

### A. Recommended: three-channel residual adapter

Keep the proven Token-Mixer path and replace only H1's local one-channel view
with the Case 3 three-channel tensor. Fuse the resulting pair embedding through
the same learned scalar residual.

This is the smallest attributable change. It directly tests whether Case 3
state context complements the strong anchor and reuses the H1 training and gate
infrastructure.

### B. Replace the Token-Mixer with a pure Case 3 Conv2D

This follows Liu's network more literally, but the project already adjudicated
the pure state-matrix Conv2D route. Replacing the backbone would mix a
representation change with an architecture change and would discard the
strongest local anchor. H2 rejects this option.

### C. Scale H1 or reopen DDT/trail features

H1 failed its frozen promotion margin, while DDT/beam-stat exploration is
closed for this branch. More H1 samples would not isolate the missing
three-channel context. H2 rejects both options.

## Frozen Four-Role Matrix

All rows use equal adapter capacity and common initialization where their
architectures match.

| Role | Local adapter input | Purpose |
| --- | --- | --- |
| `anchor` | none; existing InvP Token-Mixer | strongest same-protocol reference |
| `candidate` | `(C0, C1, InvP(DeltaC))` | Liu Case 3 plus true PRESENT topology |
| `shuffled_p` | `(C0, C1, ShuffledP(DeltaC))` | topology-attribution control |
| `raw_triple` | `(C0, C1, DeltaC)` | representation and capacity control |

`ShuffledP` is one fixed deterministic 64-bit permutation already used by the
H1 shuffled control. It must preserve tensor shape and global bit marginals.
The same mapping is used for every example, split, and run.

## Architecture

The common backbone remains the existing
`present_nibble_invp_only_spn_only` Token-Mixer:

```text
raw 16 ciphertext pairs
  -> existing true-InvP token encoder
  -> existing two-layer pair Token-Mixer
  -> pair embeddings
```

For the three hybrid rows, a parallel local branch performs:

```text
[batch, pair, 3, 4, 16]
  -> shared 1x1 Conv2D stem, 3 -> 16 channels
  -> one 3x3 residual block
  -> spatial mean and max pooling
  -> linear projection to the Token-Mixer pair-embedding width
```

Fusion remains exactly:

```text
fused_pair = token_pair + alpha * local_pair
alpha initialization = 0.1
```

Pair aggregation, classifier, activation, normalization, dropout, and model
selection remain the same as H1. The role-specific mapping is a fixed buffer,
not a learned permutation. No extra loss, auxiliary head, attention block,
channel-specific stem, or enlarged classifier is permitted.

## Frozen Data And Training Protocol

The H2 comparison preserves:

```text
cipher                    = PRESENT-80
rounds                    = 7
sample structure          = zhang_wang_case2_official_mcnd
pairs per sample          = 16
feature encoding          = ciphertext_pair_bits
negative mode             = encrypted_random_plaintexts
effective key schedule    = per_pair_random, verified from cache metadata
loss                      = mse
optimizer                 = adam
learning rate             = 0.0001
weight decay              = 0.00001
schedule                  = official_cyclic, max learning rate 0.002
checkpoint metric         = val_auc
restore best checkpoint   = true
early stopping patience   = 8
early stopping min delta  = 0.0001
epochs                    = 10 for the diagnostic
device                    = local CPU
```

Configured `train_key` and `validation_key` fields are cache/plan identity
placeholders for this generator. The gate must validate the effective
`per_pair_random` behavior from generated dataset metadata rather than infer it
from those fields.

The anchor and all controls must reuse the same parameter-matched train and
validation datasets. Labels, negative construction, train/validation split,
metric computation, optimizer state transitions, and checkpoint selection must
not change from H1.

## Execution Ladder

### R0: readiness only

```text
samples_per_class = 64
seed              = 0
epochs            = 1
rows              = 4
device            = cpu
```

R0 checks exact tensor semantics, registry construction, equal hybrid capacity,
common initialization, finite forward/backward, dataset-cache reuse, metadata,
JSONL serialization, histories, plotting, and neutral strict-gate replay. R0
AUC is not research evidence.

### R1: local seed0 diagnostic

```text
train samples_per_class      = 8192
validation samples_per_class = 4096
seed                         = 0
epochs                       = 10
rows                         = 4
device                       = cpu
```

This run starts automatically only after R0 passes. `8192/class` remains a
diagnostic screen, not formal SPN/PRESENT training.

### R2: conditional seed1

Seed1 at the same `8192/class` budget is authorized only by an R1 promotion
decision. This design does not authorize `65536/class`, `262144/class`, remote
GPU execution, or a formal `>=1000000/class` run.

## Strict Gate

Let AUC values come from each role's restored best `val_auc` checkpoint. R1
passes only when all conditions hold:

```text
candidate_auc - anchor_auc     >= 0.003
candidate_auc - shuffled_p_auc >= 0.003
candidate_auc - raw_triple_auc >= 0.003
```

Decision table:

| Condition | Decision | Next action |
| --- | --- | --- |
| all validation/provenance checks pass and all three margins pass | `promote_seed1` | run same-budget seed1 |
| candidate is above every comparator but any margin is below `0.003` | `weak_or_fragile_no_scale` | inspect histories once; stop H2 scaling |
| candidate does not exceed one or more comparators | `reject_h2` | stop H2 |
| row, protocol, cache, history, checkpoint, or metadata mismatch | `invalid` | repair and rerun the same gate; no evidence claim |

The gate must fail closed on missing/duplicate roles, non-finite metrics,
incorrect sample counts, wrong negative mode, wrong effective key schedule,
unequal validation budgets, incomplete histories, checkpoint mismatch, cache
non-reuse, or plan/result misalignment.

## Artifacts And Result Record

Each run must produce or validate:

```text
results.jsonl
progress.jsonl
history.csv
curves.svg
strict gate JSON
dataset cache features/labels arrays
dataset cache metadata and completion/progress records
best checkpoints and per-role histories
```

The completed meaningful R1 result must update this experiment record with run
ID, artifact paths, AUCs, all three margins, gate decision, claim scope, and the
recommended next action in the same turn.

## Novelty Boundary

The Case 3 three-channel representation and Conv2D processing are Liu et al.
prior art. H2 cannot claim them as the project's innovation. The testable
project contribution is limited to:

1. residual fusion into the stronger cipher-structured Token-Mixer;
2. cipher-spec-derived true and deterministic shuffled PRESENT mappings; and
3. a same-budget anchor/topology/representation attribution protocol.

Any stronger method claim requires successful same-budget controls followed by
completed, retrieved, plan-aligned scale evidence. Formal SPN/PRESENT claims
require at least `1000000/class` and multiple seeds.

## Implementation Acceptance Criteria

Implementation may proceed only if it satisfies all of the following:

1. Unit tests independently reconstruct all three channel tensors from raw
   ciphertext bits and prove only channel 2 is permuted.
2. Candidate, shuffled-P, and raw-triple hybrids have identical parameter
   counts and identical common initialization under the same seed.
3. The anchor's architecture and initialization remain unchanged.
4. R0 uses a dedicated small plan rather than attempting to override an H2 R1
   plan's sample count from the CLI.
5. The strict gate validates effective key behavior from cache metadata.
6. Focused tests, Ruff, `git diff --check`, plan validation, result validation,
   cache/progress audit, SVG parsing, and strict gate replay pass before a result
   is reported.
7. No H2 seed1 or larger/remote run starts unless the preceding gate explicitly
   returns `promote_seed1`.
