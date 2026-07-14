# Innovation 1 E4-R3 Cross-SPN Typed Transfer Medium Diagnostic Plan

**Status:** completed remote medium diagnostic; fallback-retrieved, locally re-adjudicated, and archived on post-hoc verified result branches
**Date:** 2026-07-14
**Experiment label:** E4-R3

## Research Question

Does the attributable PRESENT-80 to GIFT-64 typed-cell checkpoint-transfer
advantage observed on both E4-R2 target seeds survive an eight-times larger,
otherwise identical remote-GPU data budget?

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
authorizes this E4-R3 medium diagnostic only. The user corrected the execution
policy on 2026-07-14: `65536/class` is already a medium experiment and must run
on the remote GPU workstation, not local CPU/GPU. This does not authorize a
formal claim or a larger scale.

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
device                         = remote CUDA, one seed per A6000
dataset cache chunk/workers    = 512 / 4
```

Remote execution uses the verified `torch310` environment at
`F:\Anaconda\envs\DWT\torch310\python.exe`. Seed0 runs on remote GPU0 and
seed1 on remote GPU1 as independent processes. Local CUDA availability is
irrelevant to the medium execution decision and must not trigger a local CPU
substitution.

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

remote run roots:
  G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_gift64_cross_spn_typed_transfer_r3_65536_seed0
  G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_gift64_cross_spn_typed_transfer_r3_65536_seed1

remote caches:
  <seed0 run root>\cache
  <seed1 run root>\cache

retrieved result roots:
  outputs/remote_results/i1_gift64_cross_spn_typed_transfer_r3_65536_seed0
  outputs/remote_results/i1_gift64_cross_spn_typed_transfer_r3_65536_seed1

retrieved joint gate:
  outputs/remote_results/i1_gift64_cross_spn_typed_transfer_r3_65536_joint_seed0_seed1/gate.json
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
4. Package the two frozen PRESENT source checkpoints/results as small tracked
   provenance assets so a clean remote clone can reproduce strict loading.
5. Commit and push the plan, configs, gate, CLI, source assets, remote launch
   scripts, monitors, and tests before medium execution.
6. Verify the remote `torch310` CUDA environment, clean run-owned clones, exact
   pushed revision, `cmd.exe /c`, and all paths under `G:\lxy`.
   Pass the pushed commit SHA explicitly to the launcher; both run-owned clones
   must detached-checkout and re-verify that exact SHA before training.
7. Run seed0 on A6000 GPU0 and seed1 on A6000 GPU1 with independent caches.
8. Let local tmux monitors retrieve verified result archives, then validate,
   render/read the Chinese SVGs, and apply the frozen joint gate.
9. Refresh the numbered result index, update the E4 design and route verdict,
   commit/push the adjudication, and report the evidence-backed next action.

## Explicitly Stopped

Do not perform any of the following during E4-R3:

- local `65536/class` training or local CPU/GPU substitution;
- remote dirty overlay or launch from an unpublished commit;
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

Readiness authorizes the two planned remote `65536/class` runs after the plan,
configs, gate, CLI, source checkpoint package, remote assets, and tests are
committed and pushed. It does not authorize metric interpretation,
`262144/class`, or formal claims.

## 2026-07-15 Remote Launch Record

The two medium diagnostics are running from exact pushed commit
`9aa31ddc8f48312ecf3e1d9ea3973a0c4b00542a`:

```text
seed0 task = I1_E4_R3_GIFT64_SEED0_GPU0, remote A6000 GPU0
seed1 task = I1_E4_R3_GIFT64_SEED1_GPU1, remote A6000 GPU1
task mode  = Windows Task Scheduler, cmd.exe /c, SYSTEM, highest privileges
source     = detached clean run-owned clone at exact pushed commit
startup    = both exact started markers present
readiness  = both readiness reports present and passing
progress   = both disk-backed cache progress JSONLs advancing
monitor    = local tmux session i1_e4_r3_remote_monitor
```

Current evidence state is `running`, not completed remotely, retrieved, or
plan-aligned result evidence. No AUC row or gate exists yet, so this launch is
not added to the completed-result index. The tmux monitor will sparsely sync
logs, retrieve both verified result archives and the joint gate, and refresh
the index after completion. The main thread must not SSH-poll the run.

## 2026-07-14 Local Medium Attempt Correction

The first E4-R3 execution plan incorrectly treated a local CPU run as an
acceptable fallback because the local PyTorch environment reported
`cuda=False`. Two local sessions were started and stopped after the user
corrected the scale policy.

```text
seed0 stopped = first model, epoch 5, no completed result row
seed1 stopped = first model, epoch 5, no completed result row
results bytes = 0 for both seeds
gate          = absent for both seeds
joint gate    = absent
index status  = not indexed as completed evidence
```

The partial caches and progress logs are execution diagnostics only. They are
not E4-R3 evidence and must not be resumed or interpreted. The corrected action
is to launch both seeds remotely from a clean clone of the pushed commit.

## 2026-07-15 Completion And Adjudication

Both remote A6000 runs completed all five training rows, 50 epoch-history
records, restored-best checkpoints, disk-cache generation/reuse, and remote
plan validation. The original runners then failed at optional plotting because
the remote `torch310` environment lacked Matplotlib. This was a postprocessing
failure, not a training failure.

The raw run artifacts were retrieved from the approved `G:\lxy` run roots,
validated again locally, rendered into Chinese SVG/history artifacts, and
passed through the corrected frozen E4-R3 gates. Relocatable, hashed post-hoc
result archives were then pushed and retrieved from these branches:

```text
seed0 branch = results/i1_gift64_cross_spn_typed_transfer_r3_65536_seed0
seed0 commit = ecb883b
seed1 branch = results/i1_gift64_cross_spn_typed_transfer_r3_65536_seed1
seed1 commit = be74a83
joint branch = results/i1_gift64_cross_spn_typed_transfer_r3_65536_joint_seed0_seed1
joint commit = 87f582e
```

These branches were constructed from fallback-retrieved evidence after the
remote plotting failure; they were not pushed by the original remote runner.
Each archive preserves that provenance boundary, the unchanged raw remote
progress/results, local validation/re-adjudication copies, source/config
manifests, GPU/Torch/Git evidence, and `SHA256SUMS`. Checkpoints remain in the
local retrieved result roots and are intentionally excluded from Git.

### Final restored-best AUC

| Target seed | Anchor | Scratch | True-to-true | Source-shuffled | Target-shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.579249573871` | `0.588316810317` | `0.588565108832` | `0.587173988577` | `0.507330908440` |
| 1 | `0.578508460429` | `0.585099051706` | `0.586938600987` | `0.584353961516` | `0.510260965209` |

| Target seed | vs anchor | vs scratch | vs source-shuffled | vs target-shuffled |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `+0.009315534960` | `+0.000248298515` | `+0.001391120255` | `+0.081234200392` |
| 1 | `+0.008430140559` | `+0.001839549281` | `+0.002584639471` | `+0.076677635778` |

Both seeds preserve positive ordering versus scratch and source-shuffled, and
both show a large target-topology effect. However, neither seed reaches the
frozen `+0.005` scratch margin or `+0.003` source-topology margin. Therefore:

```text
status      = pass  # evidence/protocol validity, not hypothesis success
decision    = e4_r3_two_seed_medium_signal_unstable
next_action = stop_mechanical_scale_and_audit_seed_variance
```

Mechanical `262144/class` and formal-scale expansion are stopped.

### Post-hoc early-adaptation observation

The complete histories show a different, narrower signal at epoch 1:

| Target seed | True-to-true minus scratch | True-source minus source-shuffled |
| ---: | ---: | ---: |
| 0 | `+0.005607747007` | `+0.009936478920` |
| 1 | `+0.005658070557` | `+0.010892953258` |

This suggests PRESENT pretraining may accelerate early GIFT adaptation while
its final best-AUC advantage decays as scratch training catches up. Because
this hypothesis was discovered after inspecting E4-R3 histories, it is not a
confirmed contribution and cannot reuse seeds 0/1 as confirmation evidence.

### Executable next action

Run E4-R4 as a new-seed, exactly-one-epoch target-adaptation confirmation at
the same `65536/class` target budget. Change target seeds to 2 and 3 and remove
only the irrelevant old architecture anchor; retain typed scratch,
true-to-true, source-shuffled-to-true, and true-to-target-shuffled controls.
Export paired validation scores and require stratified paired-bootstrap AUC
difference intervals. Report PRESENT source-pretraining cost separately and
limit any positive claim to conditional target-adaptation efficiency, not
end-to-end training cost. Do not launch until the E4-R4 plan, score export,
bootstrap gate, cache readiness, and remote assets pass local smoke.
