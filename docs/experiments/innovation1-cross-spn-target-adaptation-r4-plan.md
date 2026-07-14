# Innovation 1 E4-R4 Cross-SPN Target-Adaptation Confirmation Plan

**Status:** design and local readiness passed; remote launch pending pushed implementation commit
**Date:** 2026-07-15
**Experiment label:** E4-R4

## Research Question

Does a correctly typed PRESENT-80 source checkpoint provide a reproducible,
attributable improvement in GIFT-64 validation AUC after exactly one target
training epoch, relative to typed scratch and source/target topology controls,
on two target seeds not used to discover the hypothesis?

E4-R4 tests target-adaptation efficiency, not persistent final AUC and not
end-to-end compute savings. The hypothesis was discovered post hoc from E4-R3,
so target seeds 0 and 1 are development evidence only. Confirmation uses target
seeds 2 and 3.

## Same-Budget Anchor And One Variable

The same-budget anchor is E4-R3 at `65536/class` target training and
`32768/class` validation with the same data definition, architecture,
optimizer, source checkpoints, and target compute. E4-R4 changes only:

```text
target seeds  = 0,1 -> 2,3
target epochs = inspect post-hoc epoch 1 of 10 -> train exactly 1 epoch
```

No new architecture, feature, negative mode, key policy, source pretraining,
or validation set is introduced.

## Frozen Protocol

```text
cipher                         = GIFT-64
rounds                         = 6
target seeds                   = 2, 3
train                          = 65536/class = 131072 total/seed
validation                     = 32768/class = 65536 total/seed
pairs/sample                   = 4 independent pairs
feature                        = raw ciphertext pair bits
positive                       = fixed-difference encrypted pairs
negative                       = encrypted random plaintext pairs
train key                      = 0x00000000000000000000000000000000
validation key                 = 0x11111111111111111111111111111111
key schedule                   = fixed per split
target epochs                  = exactly 1
batch size                     = 256
hidden bits                    = 32
loss                           = MSE
optimizer                      = Adam
learning rate                  = 0.0001
weight decay                   = 0.00001
LR scheduler                   = none
checkpoint metric              = validation AUC
checkpoint selection           = restored best checkpoint
device                         = remote CUDA, one target seed per A6000
dataset cache chunk/workers    = 512 / 4
```

This remains a remote medium diagnostic. It is not formal or paper-scale.

## Frozen Roles

Use four rows per target seed:

| Role | Model key | Purpose |
| --- | --- | --- |
| typed scratch | `gift_cross_spn_typed_cell_true` | same target architecture without source pretraining |
| true to true | `gift_cross_spn_typed_cell_true_from_present_true` | adaptation candidate |
| shuffled to true | `gift_cross_spn_typed_cell_true_from_present_shuffled` | source-topology attribution control |
| true to shuffled | `gift_cross_spn_typed_cell_shuffled_from_present_true` | target-topology attribution control |

The old GIFT aligned anchor is excluded because E4-R4 asks whether identical
typed target architectures adapt faster; it is not a transfer attribution
control. Removing it reduces the matrix from five to four rows without
changing the core hypothesis.

The frozen PRESENT source checkpoint SHA-256 values remain:

```text
true     = eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1
shuffled = fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22
```

## Required Paired Evidence

For every role, export one score per row on the exact shared validation cache:

```text
sample_index, label, score, target_seed, model_role, checkpoint_sha256
```

The gate must verify identical sample indices and labels across all four roles.
Compute AUC differences with a paired, label-stratified bootstrap that resamples
positive and negative validation rows separately using identical resample
indices for both compared models. Freeze at least `10000` bootstrap replicates
and a deterministic bootstrap seed before remote execution.

Primary comparisons per target seed:

```text
true_to_true - typed_scratch
true_to_true - shuffled_to_true
true_to_true - true_to_shuffled
```

## Frozen Advance Gate

Both new target seeds must satisfy all of:

```text
true_to_true - typed_scratch    >= +0.004
true_to_true - shuffled_to_true >= +0.005
true_to_true - true_to_shuffled >= +0.003
paired 95% CI lower bound for true_to_true - typed_scratch > 0
```

The source-topology and target-topology comparisons must also be positive on
both seeds; their confidence intervals are reported even when not used as the
primary hard CI gate.

Decision table:

```text
both seeds pass all margins and core paired CI:
  decision    = e4_r4_target_adaptation_efficiency_confirmed
  next_action = design_formal_multiseed_adaptation_protocol

ordering positive but a margin or CI misses:
  decision    = e4_r4_target_adaptation_signal_unstable
  next_action = stop_transfer_branch_keep_typed_representation_result

any source/target control reverses or protocol pairing is invalid:
  decision    = e4_r4_target_adaptation_rejected_or_invalid
  next_action = stop_transfer_branch_and_repair_only_if_protocol_invalid
```

Even a positive result supports only conditional target-adaptation efficiency:
given the frozen PRESENT source checkpoints, less GIFT target training reaches
a higher one-epoch AUC. It does not prove lower total compute because the
PRESENT source-pretraining cost must be reported separately.

## Execution Path

1. Implement an E4-R4 four-role gate without changing E4-R2/R3 behavior.
2. Add per-checkpoint score export over the shared disk-backed validation cache
   and deterministic paired stratified-bootstrap AUC differences.
3. Create seed2/seed3 CSVs and `64/class`, one-epoch local readiness matrices.
   Readiness checks implementation/provenance only and does not interpret AUC.
4. Verify four strict source loads, identical typed capacity, one shared
   target cache per seed, four checkpoints, score pairing, and finite metrics.
5. Commit and push code, configs, tests, plan, and remote assets.
6. Launch seed2 on remote A6000 GPU0 and seed3 on A6000 GPU1 from the exact
   pushed commit with `cmd.exe /c`, all files under `G:\lxy`, and local tmux
   monitor/retrieval handoff.
7. After both complete, validate, apply per-seed and joint gates, refresh the
   numbered result index, and update the E4 design and route verdict.

## Explicitly Stopped

- local `65536/class` training or local CPU substitution;
- `262144/class`, `1000000/class`, or more target epochs before E4-R4;
- reuse of target seeds 0/1 as confirmation evidence;
- removal of scratch, source-shuffled, or target-shuffled controls;
- changing data, labels, negatives, keys, architecture, source checkpoints,
  optimizer, or validation policy;
- unpaired confidence intervals or score exports from different validation
  samples;
- claims of formal evidence, SOTA, breakthrough, or end-to-end compute savings;
- reopening DDT, trail, E1, H2, or flattened DBitNet routes.

## 2026-07-15 Implementation And Readiness Record

The E4-R4 four-role gate, paired score export, deterministic label-stratified
bootstrap, compressed long-form score CSV, seed2/seed3 plans, fail-closed
remote readiness checks, Windows runners, and tmux retrieval monitor are
implemented.

Both target seeds completed a `64/class`, exactly-one-epoch CPU readiness run:

```text
seed2 status/decision = pass / implementation_ready
seed3 status/decision = pass / implementation_ready
result rows           = 4 per seed
history rows          = 4 per seed
checkpoints           = 4 restored-best files per seed
score artifacts       = 4 aligned artifacts per seed
score rows            = 64 per role
paired CSV rows       = 256 per seed plus header
typed parameters      = 187426 for all roles
plan validation       = pass, errors=[] for both seeds
remote dry-run        = pass, errors=[], warnings=[] for both configs
```

The gate verified exact source checkpoint SHA-256 provenance, one generated
and three reused train/validation cache events, matching target checkpoint
SHA-256 values in score metadata, identical sample IDs and labels across all
four roles, and score AUC equality with the restored-best result rows. The
readiness AUC values are intentionally not interpreted.

Artifacts:

```text
outputs/local_smoke/i1_gift64_cross_spn_target_adaptation_r4_readiness_seed2/
outputs/local_smoke/i1_gift64_cross_spn_target_adaptation_r4_readiness_seed3/
result index 001/002 = seed3/seed2 readiness
```

The exact 65,536-row paired-bootstrap implementation was benchmarked locally:
100 replicates took approximately `0.97s`, so the frozen 10,000-replicate
remote gate is expected to require roughly 1.5-2 minutes of CPU postprocessing
after score export. This is not a training-time estimate.

Readiness authorizes only the planned remote medium runs after this
implementation, configs, source assets, and remote scripts are committed and
pushed. Seed2 must run on A6000 GPU0 and seed3 on A6000 GPU1 from the exact
pushed commit. Local `65536/class` execution remains prohibited.
