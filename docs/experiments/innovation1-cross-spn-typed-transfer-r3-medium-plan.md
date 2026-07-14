# Innovation 1 E4-R3 Cross-SPN Typed Transfer Medium Diagnostic Plan

**Status:** two-seed readiness passed; medium execution authorized
**Date:** 2026-07-14
**Experiment label:** E4-R3

## Research Question

Does the attributable PRESENT-80 to GIFT-64 typed-cell checkpoint-transfer
advantage observed on both E4-R2 target seeds survive an eight-times larger,
otherwise identical local data budget?

E4-R3 changes one experimental variable only:

```text
E4-R2 train       = 8192/class  = 16384 total/seed
E4-R2 validation  = 4096/class  = 8192 total/seed

E4-R3 train       = 65536/class = 131072 total/seed
E4-R3 validation  = 32768/class = 65536 total/seed
```

This is a medium diagnostic, not formal training or paper-scale evidence.

## E4-R2 Anchor Evidence

| Target seed | True-to-true AUC | vs anchor | vs scratch | vs source-shuffled | vs target-shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.569627493620` | `+0.063060313463` | `+0.017658561468` | `+0.024967253208` | `+0.060677826405` |
| 1 | `0.575072139502` | `+0.023235499859` | `+0.011130839586` | `+0.015329629183` | `+0.057054758072` |

The E4-R2 joint decision was `two_seed_transfer_signal_confirmed`. It
authorizes this plan only; it does not authorize remote or formal claims.

## Frozen Protocol

```text
cipher                         = GIFT-64
rounds                         = 6
target seeds                   = 0, 1
train                          = 65536/class = 131072 total/seed
validation                     = 32768/class = 65536 total/seed
pairs/sample                   = 4 independent pairs
feature                        = raw ciphertext pair bits
positive                       = fixed-difference encrypted pairs
negative                       = encrypted random plaintext pairs
train key                      = 0x00000000000000000000000000000000
validation key                 = 0x11111111111111111111111111111111
key schedule                   = fixed per split
epochs                         = 10
batch size                     = 256
hidden bits                    = 32
loss                           = MSE
optimizer                      = Adam
learning rate                  = 0.0001
weight decay                   = 0.00001
LR scheduler                   = none
checkpoint metric              = validation AUC
checkpoint selection           = restored best checkpoint
train evaluation interval      = 1
device                         = local CPU
dataset cache chunk/workers    = 512 / 4
```

Local CUDA was checked before freezing the execution path. The workstation has
an RTX 5080, but the active project PyTorch environment reports
`torch.cuda.is_available() == False`; E4-R3 therefore uses CPU without changing
the environment or stopping the user's GPU processes.

## Same-Budget Roles And Controls

Each target seed uses exactly five rows and one shared parameter-matched cache:

| Role | Model key | Purpose |
| --- | --- | --- |
| GIFT anchor | `gift_cross_spn_aligned_token_mixer_raw_anchor` | previous same-input GIFT architecture anchor |
| GIFT typed scratch | `gift_cross_spn_typed_cell_true` | same target architecture without source pretraining |
| true to true | `gift_cross_spn_typed_cell_true_from_present_true` | E4 transfer candidate |
| shuffled to true | `gift_cross_spn_typed_cell_true_from_present_shuffled` | source-topology attribution control |
| true to shuffled | `gift_cross_spn_typed_cell_shuffled_from_present_true` | target-topology attribution control |

The source result and checkpoints remain the completed PRESENT-80 r7 E4-R1
seed0 artifacts. Their identities are frozen:

```text
true SHA-256:
  eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1
shuffled SHA-256:
  fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22
manifest:
  configs/experiment/innovation1/innovation1_spn_cross_spn_typed_transfer_seed0_sources.json
```

No source retraining, target architecture change, optimizer change, validation
change, new negative definition, or checkpoint substitution is permitted.

## Frozen Per-Seed Gate

The thresholds are copied unchanged from E4-R2 and frozen before E4-R3
execution:

```text
true_to_true AUC                >= 0.52
true_to_true - gift_anchor      >= +0.003
true_to_true - typed_scratch    >= +0.005
true_to_true - shuffled_to_true >= +0.003
true_to_true - true_to_shuffled >= +0.003
```

The gate must also verify, for each seed:

- exactly five plan-aligned result rows and 50 history epochs;
- strict source state-dict loading and exact source SHA-256 values;
- identical typed parameter count `187426`;
- one non-empty disk-cache root shared by all five roles;
- `features.npy`, `labels.npy`, metadata and progress evidence for train and
  validation splits, with one generation followed by four matched reuse
  events per split;
- `131072` training rows and `65536` validation rows;
- restored-best checkpoints for all five roles;
- finite AUC, accuracy, calibrated accuracy and loss.

## Joint Decision Gate

```text
both seeds pass:
  decision    = e4_r3_two_seed_medium_signal_confirmed
  next_action = design_e4_r4_262144_class_diagnostic_with_remote_readiness

one or both seeds miss an attribution threshold:
  decision    = e4_r3_two_seed_medium_signal_unstable
  next_action = stop_mechanical_scale_and_audit_seed_variance

protocol/provenance/cache mismatch:
  status      = invalid
  next_action = repair_e4_r3_evidence_before_interpretation
```

A positive joint result authorizes only an E4-R4 design and remote-readiness
audit. It does not itself authorize launch, formal evidence, a SOTA claim, or
a breakthrough claim.

## Configs, Caches, And Artifacts

```text
configs:
  configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_65536_seed0.csv
  configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_65536_seed1.csv

caches:
  outputs/local_cache/i1_gift64_cross_spn_typed_transfer_r3_65536_seed0
  outputs/local_cache/i1_gift64_cross_spn_typed_transfer_r3_65536_seed1

result roots:
  outputs/local_diagnostic/i1_gift64_cross_spn_typed_transfer_r3_65536_seed0
  outputs/local_diagnostic/i1_gift64_cross_spn_typed_transfer_r3_65536_seed1

joint gate:
  outputs/local_diagnostic/i1_gift64_cross_spn_typed_transfer_r3_65536_joint_seed0_seed1/gate.json
```

Each result root must contain `results.jsonl`, `progress.jsonl`,
`validation.json`, `history.csv`, `curves.svg`, `gate.json`, and five selected
checkpoints. After readiness and after each completed seed/joint gate, refresh
`outputs/00_RECENT_RESULTS.md` and `outputs/00_RECENT_RESULTS.json`.

## Execution Sequence

1. Add E4-R3 budget/stage support to the existing fail-closed transfer gate;
   preserve all E4-R2 decisions and defaults.
2. Validate both CSVs through task construction and check that only target seed
   differs between them.
3. Re-run the existing `64/class`, one-epoch seed0 and seed1 matrices into new
   E4-R3 readiness roots. Require strict source provenance, complete five-role
   outputs, cache generation/reuse, and `implementation_ready` without
   interpreting AUC.
4. Commit and push the plan, configs, gate, CLI and tests before medium runs.
5. Run seed0 and seed1 at `65536/class` locally on CPU with separate caches.
6. Validate results, render readable Chinese SVGs, export history CSVs, parse
   SVG XML, and apply per-seed then joint gates.
7. Refresh the numbered result index, update the E4 design and route verdict,
   commit/push the adjudication, and report the evidence-backed next action.

## Explicitly Stopped

Do not perform any of the following during E4-R3:

- remote launch or remote dirty overlay;
- `262144/class` or `1000000/class` mechanical scale-up;
- DDT, trail, beamstats, E1, H2, or flattened DBitNet reopening;
- source checkpoint replacement or additional PRESENT pretraining;
- target architecture, feature, optimizer, labels, negatives, keys, epochs, or
  validation-policy changes;
- early interpretation from one completed seed while the other is pending;
- formal, SOTA, paper-scale, ceiling, failure, or breakthrough claims.

## 2026-07-14 Readiness Record

Both target seeds completed the frozen `64/class`, one-epoch, five-role CPU
readiness matrix. Metrics were generated mechanically but were not interpreted.

```text
seed0 status/decision = pass / implementation_ready
seed1 status/decision = pass / implementation_ready
typed parameters      = 187426 on both seeds
source true SHA       = eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1
source shuffled SHA   = fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22
validation            = five plan-aligned rows per seed, errors=[]
history               = five rows per seed, one epoch per role
SVG                    = parsed successfully with Chinese E4-R3 titles
```

Each seed created one train and one validation cache containing
`features.npy`, `labels.npy`, and `metadata.json`; the first role emitted
`cache_done` and the remaining four roles emitted parameter-matched
`cache_reuse` events for both splits. All five selected checkpoints and all
initialization provenance events are present.

```text
seed0 artifacts:
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r3_readiness_seed0/
seed1 artifacts:
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r3_readiness_seed1/
result index at completion:
  001 = seed0 readiness
  002 = seed1 readiness
```

Readiness authorizes the two planned local `65536/class` runs after the plan,
configs, gate, CLI, plotting/index labels, and tests are committed and pushed.
It does not authorize metric interpretation, remote launch, `262144/class`, or
formal claims.
