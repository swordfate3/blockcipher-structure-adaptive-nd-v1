# Innovation 1 E4-R5 Cross-SPN Source-Seed Robustness Plan

**Status:** completed / independent source-seed robustness not confirmed / E4-R6 stopped
**Date:** 2026-07-15
**Experiment label:** E4-R5

## Research Question

Does the E4-R4 one-epoch PRESENT-to-GIFT target-adaptation advantage survive
when the PRESENT true/shuffled source checkpoints are independently retrained
with source seed1, rather than reusing the source-seed0 checkpoints that were
used to discover and confirm E4-R2 through E4-R4?

E4-R4 varied two fresh GIFT target seeds and passed paired confidence gates,
but its source checkpoint remained fixed. E4-R5 audits that exact remaining
variance axis before any paper-scale target run.

## Same-Budget Anchor And One Variable

The same-budget anchor is E4-R4:

```text
source               = PRESENT-80 r7, 8192/class, 10 epochs
target               = GIFT-64 r6, 65536/class, exactly 1 epoch
target validation    = 32768/class
pairs/sample         = 4 independent GIFT pairs
target roles         = scratch, true->true, shuffled->true, true->shuffled
paired bootstrap     = 10000 label-stratified replicates
```

The hypothesis variable is:

```text
PRESENT source checkpoint seed = 0 -> 1
```

Fresh GIFT target seeds 4 and 5 are independent replication units, not a
second method change. Architecture, source/target sample budgets, mappings,
strict negative definition, keys, optimizer, target epochs, validation cache,
checkpoint selection, score pairing, and thresholds remain frozen.

## Phase A: Independent PRESENT Source Gate

Train the exact E4-R1 four-role PRESENT matrix with seed1:

| Role | Model | Purpose |
| --- | --- | --- |
| anchor | `present_nibble_invp_only_spn_only` | strongest same-input PRESENT reference |
| typed true | `present_cross_spn_typed_cell_true` | new transferable source candidate |
| typed shuffled | `present_cross_spn_typed_cell_shuffled` | source-topology control |
| typed raw | `present_cross_spn_typed_cell_raw` | typed-capacity/mapping control |

Frozen source protocol:

```text
cipher/rounds                   = PRESENT-80 r7
source seed                     = 1
train                           = 8192/class = 16384 total rows
validation                      = 4096/class = 8192 total rows
sample structure                = Zhang/Wang Case2 official MCND
pairs/sample                    = 16
effective key policy            = per_pair_random
negative                        = encrypted random plaintexts
epochs                          = 10
batch size                      = 256
hidden bits                     = 32
loss/optimizer                  = MSE / Adam
learning rate / weight decay    = 0.0001 / 0.00001
checkpoint                      = restored best validation AUC
device                          = local CPU
disk cache                      = required and parameter matched
```

The source gate is identical to E4-R1:

```text
typed_true AUC                  >= 0.65
typed_true - anchor             >= -0.010
typed_true - typed_shuffled     >= +0.003
typed_true - typed_raw          >= +0.003
```

Phase A must also verify four equal-capacity typed state geometries where
applicable, complete histories, restored checkpoints, strict key metadata,
one generated then reused train/validation cache, and exact plan alignment.

If Phase A fails any mandatory margin, E4-R5 stops. Do not compensate by
changing source epochs, architecture, source data, or checkpoint metric.

## Phase B: Fresh-Target Source-Seed Gate

Phase B is authorized only after Phase A passes and its true/shuffled source
checkpoint paths and SHA-256 values are frozen in a new initialization
manifest.

Run one target seed per remote A6000:

```text
cipher/rounds                   = GIFT-64 r6
target seeds                    = 4, 5
train                           = 65536/class = 131072 total rows/seed
validation                      = 32768/class = 65536 total rows/seed
pairs/sample                    = 4 independent pairs
input                           = 512 raw ciphertext-pair bits
negative                        = encrypted random plaintexts
train/validation keys           = fixed split keys, unchanged from E4-R4
target epochs                   = exactly 1
batch size                      = 256
hidden bits                     = 32
loss/optimizer                  = MSE / Adam
learning rate / weight decay    = 0.0001 / 0.00001
checkpoint                      = restored best validation AUC
devices                         = seed4 GPU0, seed5 GPU1
disk cache                      = required under each G:\lxy run root
```

Four target roles per seed:

| Role | Initialization | Target mapping |
| --- | --- | --- |
| typed scratch | scratch | true GIFT |
| true to true | PRESENT seed1 true | true GIFT |
| shuffled to true | PRESENT seed1 shuffled | true GIFT |
| true to shuffled | PRESENT seed1 true | shuffled GIFT |

Every role must export one score per row on the exact shared validation cache.
The gate verifies identical sample IDs and labels, checkpoint SHA-256 metadata,
score AUC equality with restored-best results, and one common target cache.

## Frozen Target Gate

For each target seed, run the same deterministic 10,000-replicate paired,
label-stratified bootstrap with bootstrap seed `20260715`.

Both target seeds must satisfy:

```text
true_to_true - typed_scratch    >= +0.004
true_to_true - shuffled_to_true >= +0.005
true_to_true - true_to_shuffled >= +0.003
paired 95% CI lower bound for true_to_true - typed_scratch > 0
```

The source- and target-topology paired intervals must also be reported.

Decision table:

```text
Phase A passes and target seeds 4/5 both pass:
  decision    = e4_r5_source_seed_robustness_confirmed
  next_action = freeze_1000000_class_multisource_multitarget_protocol

Phase A passes but exactly one target seed passes:
  decision    = e4_r5_source_seed_signal_unstable
  next_action = stop_formal_scale_retain_conditional_e4_r4_result

Phase A fails or either source/target control reverses on both target seeds:
  decision    = e4_r5_source_seed_dependence_detected
  next_action = stop_cross_spn_transfer_scale_keep_typed_representation_only

Any provenance, cache, score-pairing, or result-alignment error:
  decision    = invalid_e4_r5_protocol
  next_action = repair_protocol_without_interpreting_auc
```

## Execution Path

1. Create a seed1 PRESENT source CSV by changing only the E4-R1 seed and
   evidence labels.
2. Run a `64/class`, one-epoch source readiness check, then the local
   `8192/class`, 10-epoch Phase A gate.
3. If Phase A passes, freeze source checkpoint hashes in a seed1 initialization
   manifest and add strict-load/provenance tests.
4. Create target seed4/seed5 plans plus `64/class`, one-epoch CPU readiness
   runs. Readiness AUC is not interpreted.
5. Commit and push all source, config, test, plan, runner, and monitor changes.
6. Launch target seed4 on remote GPU0 and seed5 on GPU1 from that exact pushed
   commit using `cmd.exe /c` and run-owned clean sources under `G:\lxy`.
7. Use the local tmux monitor for retrieval, local strengthened gates, plots,
   and numbered result indexing.
8. Update this plan, the E4 design, and the route verdict before reporting the
   completed result.

## Advance And Stop Boundaries

- Local Phase A is an `8192/class` diagnostic, not formal training.
- Phase B is a `65536/class` remote medium diagnostic, not formal evidence.
- Do not run Phase B if the independent source gate fails.
- Do not run local `65536/class` target training.
- Do not run `262144/class`; it does not resolve source-seed variance or meet
  the project's formal scale.
- Do not launch `1000000/class` until E4-R5 is fully retrieved, locally
  re-adjudicated, and both source and target seed gates pass.
- Do not add epochs, source ensembles, multi-task objectives, cipher IDs, DDT,
  trails, S-box tables, or new negative definitions.
- Do not claim lower total compute; source pretraining cost remains separate.
- Do not reopen E1, H2, flattened DBitNet, or stopped deterministic trail
  branches.

## Claim Boundary

A positive E4-R5 result would show that conditional one-epoch cross-SPN target
adaptation survives one independent PRESENT source retraining and two fresh
GIFT targets. It would justify a paper-scale protocol design, not itself prove
formal evidence, SOTA, breakthrough, persistent final-AUC superiority, or
end-to-end compute savings.

## 2026-07-15 Phase A Readiness Record

The seed1 source-only gate, CLI, two frozen CSV matrices, and regression tests
are implemented. The `64/class`, one-epoch CPU readiness run completed:

```text
result rows           = 4
history rows          = 4
restored checkpoints  = 4
typed parameters      = 187426 for true/shuffled/raw
cache events          = 2 create + 6 parameter-matched reuse
effective key policy  = per_pair_random
plan validation       = pass, errors=[]
source readiness gate = pass / implementation_ready
```

Artifacts:

```text
outputs/local_smoke/i1_present_cross_spn_source_seed_r5_readiness_seed1/
```

Readiness metrics are not interpreted. This authorizes only the frozen local
Phase A `8192/class`, 10-epoch source-seed1 run after the implementation and
plan are committed and pushed. It does not authorize remote Phase B yet.

## 2026-07-15 Phase A Completion Record

The frozen local source-seed1 run completed at `8192/class` training and
`4096/class` validation with 16 pairs/sample and 10 epochs:

| Role | Restored-best AUC |
| --- | ---: |
| InvP-only anchor | `0.761801242828` |
| typed true | `0.755739122629` |
| typed shuffled | `0.580704301596` |
| typed raw | `0.608457535505` |

```text
true - anchor    = -0.006062120199  pass >= -0.010
true - shuffled  = +0.175034821033  pass >= +0.003
true - raw       = +0.147281587124  pass >= +0.003
true absolute    =  0.755739122629  pass >=  0.650

status           = pass
decision         = e4_r5_source_seed_gate_pass
next_action      = freeze_source_seed1_hashes_and_prepare_target_seed4_seed5
```

All four rows are plan-aligned, all checkpoints and 40 history rows are
complete, typed roles have identical `187426` parameters, and the Case2 cache
shows two creates plus six parameter-matched reuses.

Frozen source checkpoints:

```text
true checkpoint:
  outputs/local_smoke/i1_present_cross_spn_source_seed_r5_8192_seed1/checkpoints/row0002_present_cross_spn_typed_cell_true_seed1.pt
  SHA-256 = b6eed1f624e5a86d34d444a5f18e5e320447bbb44f2004b059642357543c55b5

shuffled checkpoint:
  outputs/local_smoke/i1_present_cross_spn_source_seed_r5_8192_seed1/checkpoints/row0003_present_cross_spn_typed_cell_shuffled_seed1.pt
  SHA-256 = b22e4a7b34aabc090ca75385389d46a0c866dc5114c3626ec5c233cd4b7c2645
```

Artifacts:

```text
outputs/local_smoke/i1_present_cross_spn_source_seed_r5_8192_seed1/
```

This passes only the source gate. Phase B still requires seed4/seed5 target
plans, strict source-manifest loading, score-pair readiness, remote configs,
and a passing local `64/class` implementation check before launch.

## 2026-07-15 Phase B Readiness Record

The seed4 and seed5 target plans, source-seed1 initialization manifest,
per-seed E4-R5 gate, joint source-seed robustness gate, remote configs,
runner, launcher, and retrieval monitor are implemented. Both local
`64/class`, exactly-one-epoch readiness runs passed:

```text
target seeds                    = 4, 5
result rows                     = 4/seed
restored checkpoints            = 4/seed
score artifacts                 = 4/seed
aligned validation scores       = 64/role/seed
paired score export             = pass
plan validation                 = pass, errors=[]
cache behavior                  = 1 create + 3 reuse per split/seed
source checkpoint seed          = 1
source true SHA-256             = b6eed1f624e5a86d34d444a5f18e5e320447bbb44f2004b059642357543c55b5
source shuffled SHA-256         = b22e4a7b34aabc090ca75385389d46a0c866dc5114c3626ec5c233cd4b7c2645
seed4 gate                      = pass / implementation_ready
seed5 gate                      = pass / implementation_ready
```

Artifacts:

```text
outputs/local_smoke/i1_gift64_cross_spn_source_seed_r5_readiness_seed4/
outputs/local_smoke/i1_gift64_cross_spn_source_seed_r5_readiness_seed5/
```

The two remote configs independently pass fail-closed readiness with
`errors=[]`, `warnings=[]`, and both
`medium_scale_dataset_cache` and `e4_r5_source_seed_protocol_lock`.
The runner and monitor explicitly pass `--experiment-stage e4_r5`; source
asset checks use seed1 filenames; local strengthened re-adjudication uses
target seeds 4 and 5. The launcher uses run-owned clean source clones,
`cmd.exe /c`, SYSTEM/highest scheduled tasks, GPU0/GPU1, and only
`G:\\lxy` project paths.

Readiness AUC values are not interpreted. This readiness authorizes only the
frozen remote Phase B medium diagnostic:

```text
seed4 = 65536/class train, 32768/class validation, GPU0, exactly 1 epoch
seed5 = 65536/class train, 32768/class validation, GPU1, exactly 1 epoch
```

Next action: commit and push the exact source checkpoints, results provenance,
plans, gates, configs, tests, and generated remote assets; then launch both
target seeds from that exact pushed commit and hand monitoring/retrieval to the
local tmux watcher. Do not run `65536/class` locally and do not mechanically
advance to `262144/class`.

The post-pass protocol is predeclared, but inactive, in:

```text
docs/experiments/innovation1-cross-spn-formal-multisource-multitarget-r6-conditional-plan.md
```

That document does not authorize an E4-R6 config or launch. It activates only
after both E4-R5 target seeds and the local joint gate pass from verified,
retrieved archives.

## 2026-07-15 Phase B Completion Record

Both remote target runs completed from exact pushed source commit
`b7bb5edb026dd2096acc34ed06172ee4fcffaf76`. The watcher retrieved verified
seed4, seed5, and joint result archives, performed strengthened local
re-adjudication, and refreshed the numbered result index. All `40 + 40 + 6`
archive entries pass SHA-256 verification after treating the Windows CRLF
manifest as text input; no artifact hash differs.

Frozen target protocol:

```text
target                         = GIFT-64 r6
train                          = 65536/class = 131072 total rows/seed
validation                     = 32768/class = 65536 total rows/seed
target epochs                  = exactly 1
target seeds                   = 4, 5
source                         = PRESENT-80 r7 seed1 checkpoints
negative                       = encrypted random plaintexts
paired bootstrap               = 10000 label-stratified replicates
score rows                     = 65536/role/seed, IDs and labels aligned
```

| Target seed | True AUC | Scratch AUC | True - scratch | Scratch 95% CI | True - source-shuffled | True - target-shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `0.576149932574` | `0.575976267923` | `+0.000173664652` | `[-0.002284564741, +0.002628424205]` | `+0.015344345942` | `+0.069083069451` |
| 5 | `0.573475843295` | `0.569641033188` | `+0.003834810108` | `[+0.001243890217, +0.006373106083]` | `+0.013115312438` | `+0.070617836900` |

Both target seeds preserve strong positive source- and target-topology
attribution. What does not survive is the stronger scratch-efficiency gate:
seed4 is statistically indistinguishable from scratch, while seed5 has a
positive paired interval but misses the predeclared `+0.004` point threshold
by `0.000165189892`.

```text
seed4 decision = e4_r5_target_adaptation_signal_unstable
seed5 decision = e4_r5_target_adaptation_signal_unstable
joint decision = e4_r5_source_seed_signal_unstable
joint status   = pass, errors=[]
next_action    = stop_formal_scale_retain_conditional_e4_r4_result
```

The gate execution status `pass` means protocol, pairing, provenance, and
alignment checks succeeded; it does not mean the research hypothesis passed.
Because source seed and target seeds are not fully crossed, this result must
not be overstated as a causal source-seed failure. It does prove that the
E4-R4 scratch advantage did not generalize under the independently retrained
source plus fresh-target confirmation required by the frozen decision table.

Verified artifacts:

```text
outputs/remote_results/i1_gift64_cross_spn_source_seed_r5_65536_seed4/
outputs/remote_results/i1_gift64_cross_spn_source_seed_r5_65536_seed5/
outputs/remote_results/i1_gift64_cross_spn_source_seed_r5_65536_joint_seed4_seed5/
outputs/00_RECENT_RESULTS.md entry 001
```

### Evidence-backed next action

Stop E4-R6 and all mechanical transfer scaling. Consolidate E4-R4/R5 as a
no-new-training cross-source heterogeneity synthesis:

```text
research question = which typed-operator effects survive across the two
                    independently trained PRESENT source checkpoints?
same-budget anchor = E4-R4 and E4-R5 paired gate artifacts
one comparison axis = source checkpoint seed stratum, with target-seed
                      confounding stated explicitly
data/scale         = existing GIFT target seeds 2..5 at 65536/class
epochs             = existing exactly-one-epoch checkpoints; no retraining
local path         = verified gate JSON and paired-score summaries only
retain             = source/target topology attribution and conditional E4-R4
stop               = 262144/class, 1000000/class, extra seeds/epochs, E4-R6,
                     DDT/trail/E1/H2/flattened-DBitNet reopening
```

This synthesis cannot authorize another training run. A future transfer
experiment would require a new source-objective hypothesis, literature and
same-budget ranking, and a fresh local gate; merely crossing or adding seeds
to rescue the failed claim is not justified.
