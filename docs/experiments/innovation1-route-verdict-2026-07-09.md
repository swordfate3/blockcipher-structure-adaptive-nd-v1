# Innovation 1 Route Verdict 2026-07-09

## Status

status = E4-R5 completed / independent source-seed robustness not confirmed
decision = stop E4 formal scale; retain shared typed representation and conditional E4-R4 result
claim_scope = four GIFT target seeds across two un-crossed PRESENT source-seed strata at 65536/class; medium diagnostic, not formal or paper-scale
next_adjudication = no-new-training E4 result synthesis; no E4-R6 activation

This document changes Innovation 1 from exploration mode to adjudication mode.
It does not claim a PRESENT breakthrough, does not reinterpret diagnostics as
formal evidence, and does not authorize remote scale-up by itself.

## 2026-07-10 E1 Implementation Correction

Source inspection after E1 completion found an encoder/model layout contract
mismatch in `present_active_cell_graph_pairset`. The global bit-plane feature
vector was reshaped as word-major cell groups, so graph nodes did not represent
individual PRESENT cells. E1's recorded AUC values remain historical
measurements, but its topology-route verdict is superseded.

```text
historical E1 status = completed implementation-misaligned diagnostic
historical E1 topology verdict = superseded / not adjudicated
remote_scale = no
repair_adjudication = E1-R1 completed at 2048/class
current_adjudication = E2 trail-position neural residual with deterministic baseline
```

Current matrices:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_active_cell_layout_repair_smoke_seed0.csv
configs/experiment/innovation1/innovation1_spn_present_r8_active_cell_layout_repair_pair4_2048_seed0_seed1.csv
```

E1-R1 has now resolved the implementation block from source commit `feebe27`:

| seed | true AUC | shuffled AUC | metadata-only AUC | true-shuffled | true-metadata |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.499474525 | 0.497209549 | 0.500332355 | +0.002264977 | -0.000857830 |
| 1 | 0.496722221 | 0.522119045 | 0.520334244 | -0.025396824 | -0.023612022 |

The frozen E1-R1 gate fails: seed0 misses the `+0.02` margin and loses to
metadata-only; seed1 loses to both controls. Therefore:

```text
E1-R2 8192/class = not permitted
active-cell topology branch = stopped under this matched-negative protocol
remote_scale = no
next_adjudication = E2 trail-position neural residual with deterministic baseline
```

## Why This Verdict Exists

Innovation 1 has accumulated enough experiments that adding another small
variant now has poor marginal value unless it decides a route. Current project
inventory is already large:

```text
configs/experiment/innovation1 files = 147
csv configs = 129
json configs = 17
docs/experiments innovation1* files = 82
```

The next phase should therefore answer:

```text
Which routes are stopped, which are held as weak evidence, and which few
experiments are allowed to decide whether Innovation 1 continues as architecture,
representation, or controlled negative-result work?
```

## Evidence Standard

The verdict uses the project SPN/PRESENT evidence rules:

- `512/class` through `8192/class` are local smoke or diagnostic evidence.
- `65536/class` is medium diagnostic evidence, not formal training.
- `262144/class` is stronger medium diagnostic evidence, not formal evidence.
- Formal SPN/PRESENT claims require at least `1000000/class`, multiple seeds,
  completed/retrieved/plan-aligned artifacts, and the declared claim gate.
- Strict negatives must be encrypted random plaintexts.
- A route that loses to shuffled topology, metadata-only, deterministic
  baseline, or same-input control is not eligible for scale-up.

## Route Ledger

| Route family | Best current evidence | Verdict | Reason |
| --- | --- | --- | --- |
| 16-pair raw topology aggregation | input reaches `16 * 320 + 16 = 5136` bits; control instability | stop | large input and aggregation did not solve topology-control failures |
| pair-count sweep for raw-prefix topology contrast | pair4 control-clean at 512/class and 2048/class; pair2 fails seed0; pair8 fails seed0 | stop broad sweep | pair4 is the useful minimum for that representation; pair count is not the bottleneck |
| direct active-relative slot summary | 2048/class improves seed0 but seed1 shuffled control exceeds true | stop | active-relative idea useful, direct fusion is not control-stable |
| active-relative true-minus-shuffled slot contrast | corrected E1-R1 at 2048/class: seed0 misses margin and metadata control; seed1 loses both controls | stop | cell-aligned matched-negative gate failed; no E1-R2 or remote scale |
| r7 InvP-only SPN-only anchor | two-seed `1000000/class` positive with attribution controls noted in route recheck; `262144/class` attribution shows InvP-only strongest row | keep as supported anchor | strongest current PRESENT/SPN representation evidence, but it is representation evidence more than new topology architecture |
| InvP deterministic aggregate statistics | SGP, global stats, and group-distribution audits held | stop | broad/grouped signals were too weak or unstable for handwritten statistics |
| trail-position / beamstats position-aware representation | 512/class neural candidate strong vs same-input global-stat control; 2048 split deterministic baseline strong; 65k seed0 medium positive noted | keep with deterministic baseline | strongest non-neighbor r8 representation candidate, but deterministic controls must remain attached |
| near-neighbor diverse ensembles | small improvement below gate; no confirmed low-correlation non-neighbor expert yet | hold | ensemble is not the bottleneck until a controllable non-neighbor expert exists |
| r8/r9 difference screens and curriculum probes | r9 difference screen and curriculum stopped or near random | stop | completed gates do not justify more difference/curriculum variants |
| topology-aware / DDT graph / candidate trail neighbors | weak or stopped in prior route recheck | hold as controls only | useful for comparison, not next main experiment |
| H2 Case 3 topology residual | seed0 `8192/class`: candidate `0.746721`, anchor `0.750536`, shuffled `0.749706`, raw triple `0.752140` | stop this adapter | candidate loses all three same-input comparators; seed1 and scale stopped |
| E3-R1 mapped-delta DBitNet | seed0 `8192/class`: candidate `0.516823`, Token-Mixer anchor `0.750536`; DBitNet final train AUC approaches `1.0` | stop this architecture gate | severe train/validation generalization gap and `-0.233713` AUC versus anchor; seed1 and scale stopped |
| E4-R1 shared typed cell operator | PRESENT true `0.743810` vs anchor `0.745933`, shuffled `0.575898`, raw `0.586375`; GIFT true `0.551969` vs anchor `0.506567`, shuffled `0.500088`, raw `0.501148` | promote local E4-R2 transfer gate | source is anchor-noninferior and clears both attribution controls; GIFT scratch independently clears all controls |
| E4-R2 explicit typed checkpoint transfer | two target seeds at `8192/class`: true-to-true `0.569627`/`0.575072`; all anchor, scratch, source-shuffled, and target-shuffled margins pass | two-seed local transfer signal confirmed | correct PRESENT source topology and correct GIFT target topology contribute beyond scratch and shuffled controls; medium diagnostic plan only |
| E4-R3 remote typed checkpoint transfer | two target seeds at `65536/class`: typed true beats old anchor by `+0.009316`/`+0.008430` and target-shuffled by `+0.081234`/`+0.076678`, but final true-scratch is only `+0.000248`/`+0.001840` | final transfer margin unstable; stop mechanical scale | shared typed representation and target topology remain positive medium evidence, but a persistent 10-epoch transfer advantage is not confirmed |
| E4-R4 one-epoch target adaptation | new target seeds 2/3 at `65536/class`: true-scratch `+0.011248`/`+0.006649`, paired CI lower `+0.008470`/`+0.003761`; source and target topology margins also pass | two-seed conditional target-adaptation efficiency confirmed | predeclared early-adaptation hypothesis passes; source seed remains fixed, so audit source-checkpoint variance before formal scale |
| E4-R5 independent source-seed audit | source-seed1 with fresh targets 4/5: true-scratch `+0.000174`/`+0.003835`; seed4 CI crosses zero and seed5 misses the `+0.004` point gate, while all source/target topology margins remain positive | source-seed robustness not confirmed; stop E4-R6 | typed topology attribution survives, but scratch-efficiency does not generalize under the required independent confirmation |

## 2026-07-13 H2 And E3-R1 Completion Update

H2 and E3-R1 completed as local seed0 `8192/class` diagnostics. Both used the
same strict PRESENT-80 r7 Zhang/Wang Case2 dataset, 16 pairs/sample, encrypted
random plaintext negatives, effective `per_pair_random` keys, 10 epochs, and
best-`val_auc` checkpoint restoration. These are controlled diagnostic results,
not formal training or model-ceiling evidence.

H2 tested a Liu Case 3 three-channel residual adapter:

```text
candidate - anchor     = -0.003814578056 AUC
candidate - shuffled-P = -0.002984225750 AUC
candidate - raw-triple = -0.005418211222 AUC
decision               = reject_h2
```

E3-R1 then replaced that local adapter with the published AutoND/DBitNet-2023
learner over exactly the same mapped 16-pair differences:

| Role | AUC |
| --- | ---: |
| Token-Mixer anchor | `0.7505359351634979` |
| true-InvP DBitNet | `0.5168226957321167` |
| shuffled-P DBitNet | `0.5097089111804962` |
| raw-Delta DBitNet | `0.5138811767101288` |

```text
candidate - anchor     = -0.23371323943138123 AUC
candidate - shuffled-P = +0.007113784551620483 AUC
candidate - raw-Delta  = +0.002941519021987915 AUC
decision               = reject_e3_r1
```

All three DBitNet rows reached final train AUC between `0.999806` and `1.0`
while their best validation AUCs stayed between `0.509709` and `0.516823`.
Therefore the result is not caused by insufficient epochs or a failed run; at
this diagnostic budget the flattened DBitNet learner memorizes the training
set and generalizes far worse than the Token-Mixer anchor.

Both gates explicitly stop seed1, `65536/class`, `262144/class`, and remote
scale. They do not prove that every DBitNet or PRESENT architecture fails.
They do prove that another PRESENT-only fixed-adapter or flattened-DBitNet slot
is not the next justified experiment.

The existing GIFT-64 aligned-input route was separately audited at
`8192/class`, three seeds. Its aligned row had the best mean AUC (`0.505356`)
but only about `+0.0018` over the XOR control and lost that control on seed1.
That exact GIFT-only route remains held and must not be mechanically rerun.

Current bounded next action:

```text
E4 = cipher-spec-generated typed adapter
     + shared operators across PRESENT and GIFT
     + explicit within-cipher true/shuffled/raw attribution
     + cross-cipher transfer test
remote_scale = no until a local same-budget transfer gate passes
```

## 2026-07-13 E4-R1 Completion Update

E4-R0 and E4-R1 completed locally from implementation commit `3a54f75`.
R0 passed implementation readiness with identical `187426`-parameter typed
state across PRESENT/GIFT true, shuffled, and raw variants. R1 then used
`8192/class` training, `4096/class` validation, seed0, 10 epochs, strict
encrypted-random-plaintext negatives, and restored best-`val_auc` checkpoints.

PRESENT source gate:

```text
anchor AUC       = 0.745933175087
typed true AUC   = 0.743810147047
typed shuffled   = 0.575898259878
typed raw        = 0.586375117302

true - anchor    = -0.002123028040  pass >= -0.010
true - shuffled  = +0.167911887169  pass >= +0.003
true - raw       = +0.157435029745  pass >= +0.003
```

GIFT scratch diagnostic:

```text
aligned anchor   = 0.506567180157
typed true       = 0.551968932152
typed shuffled   = 0.500088214874
typed raw        = 0.501148313284

true - anchor    = +0.045401751995
true - shuffled  = +0.051880717278
true - raw       = +0.050820618868
```

Both plan validators and the strict joint gate passed with complete histories,
checkpoints, effective key metadata, disk-cache creation/reuse, and no errors.
The joint decision is:

```text
decision = promote_e4_r2
next_action = freeze_and_implement_e4_r2_checkpoint_transfer
remote_scale = no
```

This is a strong controlled local diagnostic, not formal, paper-scale, or
breakthrough evidence. The next slot changes only checkpoint initialization
and mapping role on the same GIFT target cache/budget. It must compare GIFT
anchor, GIFT typed scratch, PRESENT-true to GIFT-true,
PRESENT-shuffled to GIFT-true, and PRESENT-true to GIFT-shuffled. Do not run
PRESENT seed1, repeat R1, increase samples, or launch remotely before that
five-role transfer gate is adjudicated.

## 2026-07-13 E4-R2 Seed0 Completion Update

E4-R2 was implemented and pushed in commit `cc8fff3`; orchestration-boundary
follow-up `6f3a904` then restored the 160-line task-runner contract with all
`1260` tests passing. The experiment completed locally with the exact R1 GIFT
target cache. The five roles used GIFT-64 r6,
`8192/class` training, `4096/class` validation, four independent pairs/sample,
seed0, 10 epochs, strict encrypted-random-plaintext negatives, fixed split
keys, MSE/Adam, and restored best-`val_auc` checkpoints.

```text
GIFT aligned anchor               = 0.506567180157
GIFT typed scratch                = 0.551968932152
PRESENT true -> GIFT true         = 0.569627493620
PRESENT shuffled -> GIFT true     = 0.544660240412
PRESENT true -> GIFT shuffled     = 0.508949667215

true-to-true - anchor             = +0.063060313463
true-to-true - scratch            = +0.017658561468
true-to-true - source-shuffled    = +0.024967253208
true-to-true - target-shuffled    = +0.060677826405
```

All five frozen thresholds passed. Source-shuffled pretraining was useful but
did not match true-source transfer, while the target-shuffled control fell
close to chance. This seed therefore supports an attributable cross-SPN typed
transfer signal rather than compatible state shapes alone.

The plan validator returned five aligned rows with no errors. The strict gate
verified source checkpoint identity and SHA-256, strict full state-dict loads,
equal typed capacity, one shared reused target cache, complete histories and
restored checkpoints:

```text
status       = pass
decision     = promote_e4_transfer_seed1
remote_scale = no
sample_scale = no
formal_claim = no
```

This remains a local single-seed `8192/class` diagnostic, not formal training,
paper-scale evidence, SOTA, or a breakthrough. The only justified next run is
an identical local target-seed1 repeat: same five roles, source checkpoints
and SHA-256 values, data definition, budget, optimizer, epochs, and thresholds;
change only target seed/cache. If seed1 passes, freeze a separate two-seed
joint gate before deciding any larger diagnostic. If seed1 fails, stop transfer
scale and retain seed0 only as provisional evidence. That seed0-only action is
now superseded by the completed two-seed gate below.

## 2026-07-14 E4-R2 Two-Seed Completion Update

Seed1 repeated the exact five-role local GIFT-64 r6 transfer protocol at
`8192/class` training and `4096/class` validation, changing only the target
seed and its disk-backed cache:

| Target seed | Anchor | Scratch | True-to-true | Source-shuffled | Target-shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.506567180157` | `0.551968932152` | `0.569627493620` | `0.544660240412` | `0.508949667215` |
| 1 | `0.551836639643` | `0.563941299915` | `0.575072139502` | `0.559742510319` | `0.518017381430` |

| Target seed | vs anchor | vs scratch | vs source-shuffled | vs target-shuffled |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `+0.063060313463` | `+0.017658561468` | `+0.024967253208` | `+0.060677826405` |
| 1 | `+0.023235499859` | `+0.011130839586` | `+0.015329629183` | `+0.057054758072` |

Both seeds passed the absolute `0.52` AUC threshold and all four frozen
attribution margins. Both used the same PRESENT source checkpoints:

```text
true SHA-256     = eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1
shuffled SHA-256 = fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22

status      = pass
decision    = two_seed_transfer_signal_confirmed
next_action = design_e4_r3_same_protocol_medium_diagnostic
```

The plan validators returned five aligned rows per seed with no errors. Both
SVGs parsed, each history CSV contains 50 epoch rows, both single-seed gates
passed, and the joint gate has an empty error list:

```text
seed0 = outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/
seed1 = outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1/
joint = outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_joint_seed0_seed1/gate.json
```

Claim boundary: this is repeatable two-seed local `8192/class` diagnostic
evidence, not formal training, paper-scale evidence, remote evidence, SOTA, or
a breakthrough. It authorizes only a separately frozen E4-R3 same-protocol
remote medium diagnostic at `65536/class` after readiness; it does not
authorize mechanical `262144/class` scaling, formal claims, or reopening
DDT/trail/E1/H2.

## 2026-07-15 E4-R3 Remote Medium Completion Update

Both GIFT-64 r6 target seeds completed remotely from exact pushed commit
`9aa31ddc8f48312ecf3e1d9ea3973a0c4b00542a`, one seed per A6000. Each used
`65536/class` training, `32768/class` validation, four independent pairs,
strict encrypted-random-plaintext negatives, 10 epochs, five same-budget
roles, disk-backed cache generation/reuse, and restored-best checkpoints.

Remote validation completed with five plan-aligned rows per seed. The runners
then failed only at Matplotlib plotting, so the evidence was raw-fallback
retrieved, validated and re-adjudicated locally, archived with hashes on three
post-hoc verified result branches, and retrieved into `outputs/remote_results`.
That provenance is preserved; this is not represented as an original remote
runner archive.

```text
seed0 true-to-true AUC = 0.588565108832
seed1 true-to-true AUC = 0.586938600987

true - typed scratch   = +0.000248298515 / +0.001839549281
true - source shuffled = +0.001391120255 / +0.002584639471
true - target shuffled = +0.081234200392 / +0.076677635778

status      = pass
decision    = e4_r3_two_seed_medium_signal_unstable
next_action = stop_mechanical_scale_and_audit_seed_variance
```

The typed operator itself remains positive versus the old same-input GIFT
anchor on both seeds, and target topology remains strongly attributed. What
failed is the stronger claim that PRESENT checkpoint transfer retains a large
advantage after 10 target epochs at the medium budget.

Epoch 1 shows a repeatable but post-hoc pattern: true transfer beats scratch by
about `+0.0056` AUC and source-shuffled transfer by about `+0.010` on both
seeds. The next bounded experiment is E4-R4: new target seeds 2/3, exactly one
target epoch, the same `65536/class` target budget and controls, paired
validation-score export, stratified paired-bootstrap confidence intervals,
and separate accounting of source-pretraining cost. Do not run
`262144/class`, formal scale, DDT/trail/E1/H2, or another 10-epoch transfer
repeat.

## 2026-07-15 E4-R4 Remote Medium Completion Update

E4-R4 completed on new GIFT-64 r6 target seeds 2 and 3 from exact source
commit `a3e0e9d`. Each seed used `65536/class` training, `32768/class`
validation, four independent ciphertext pairs/sample, strict encrypted-random-
plaintext negatives, exactly one target epoch, four equal-capacity typed roles,
and one shared disk-backed validation cache.

```text
seed2 true AUC                    = 0.579260635655
seed2 true - scratch             = +0.011247730348
seed2 paired 95% CI              = [+0.008470458165, +0.014073612948]
seed2 true - source shuffled     = +0.012428269722
seed2 true - target shuffled     = +0.077061415184

seed3 true AUC                    = 0.583190341946
seed3 true - scratch             = +0.006649116985
seed3 paired 95% CI              = [+0.003760749672, +0.009526491968]
seed3 true - source shuffled     = +0.009724145755
seed3 true - target shuffled     = +0.081193963531
```

Both seeds pass the predeclared `+0.004` scratch, `+0.005` source-topology,
and `+0.003` target-topology margins, and both scratch CI lower bounds are
strictly positive. Remote gates and independent local 10,000-replicate paired
bootstrap gates agree with empty error lists:

```text
status      = pass
decision    = e4_r4_two_seed_target_adaptation_efficiency_confirmed
next_action = audit independent source seed, then design formal protocol
```

This is the first repeatable controlled Innovation 1 result that attributes
faster cross-SPN target adaptation to both correct source and target topology.
It remains conditional on one frozen PRESENT source seed and one target epoch.
It is not persistent 10-epoch superiority, lower end-to-end compute, formal
evidence, SOTA, or a breakthrough.

The next bounded experiment is E4-R5: regenerate PRESENT true/shuffled source
checkpoints on an independent source seed under the same source budget, then
run the same four-role one-epoch GIFT gate on fresh target seeds at
`65536/class`. Only if both new target seeds pass the E4-R4 margins and paired
CIs may the project freeze a `1000000/class` multi-source/multi-target plan.
Do not run an intermediate `262144/class`, add epochs, or reopen DDT/trail/E1/
H2/flattened DBitNet.

## 2026-07-15 E4-R5 Remote Medium Completion Update

E4-R5 changed the PRESENT source checkpoint seed from 0 to 1 under the same
`8192/class`, 10-epoch source protocol, then used fresh GIFT target seeds 4
and 5 under the frozen `65536/class`, exactly-one-epoch E4-R4 target protocol.
Both verified result branches and the joint branch were retrieved and locally
re-adjudicated. All archive hashes, plan rows, sample IDs, labels, source
checkpoint hashes, and 10,000-replicate paired bootstrap outputs agree.

```text
seed4 true AUC                    = 0.576149932574
seed4 true - scratch             = +0.000173664652
seed4 paired 95% CI              = [-0.002284564741, +0.002628424205]
seed4 true - source shuffled     = +0.015344345942
seed4 true - target shuffled     = +0.069083069451

seed5 true AUC                    = 0.573475843295
seed5 true - scratch             = +0.003834810108
seed5 paired 95% CI              = [+0.001243890217, +0.006373106083]
seed5 true - source shuffled     = +0.013115312438
seed5 true - target shuffled     = +0.070617836900
```

Neither target seed passes the complete predeclared scratch-efficiency gate:
seed4 is statistically tied with scratch, and seed5 misses the `+0.004` point
margin despite a positive interval. Both seeds still clearly prefer correct
source and target topology over their shuffled controls.

```text
status      = pass, errors=[]
decision    = e4_r5_source_seed_signal_unstable
next_action = stop_formal_scale_retain_conditional_e4_r4_result
```

This closes the E4 scale path. The strongest supported Innovation 1 result is
now a controlled representation statement: a cipher-spec-generated shared
typed SPN operator preserves attributable source- and target-topology value
across four GIFT target seeds and two independently trained PRESENT source
checkpoints. One-epoch superiority over scratch is only conditional E4-R4
evidence, not source-seed-robust, formal, SOTA, breakthrough, persistent, or
end-to-end compute evidence.

The next action is a local, no-new-training E4-R4/R5 synthesis using the four
verified paired gate artifacts. Do not run E4-R6, `262144/class`,
`1000000/class`, extra target epochs/seeds, or a rescue architecture sweep.

## Current Interpretation

Innovation 1 still has a viable method-level story, but the center of gravity
has moved:

```text
from: "keep trying many SPN-aware neural architectures"
to:   "identify controllable SPN-aware representation routes, then test whether
       a neural residual or topology-aware branch adds value beyond strict
       controls"
```

The main open question is no longer whether PRESENT/SPN structure contains a
signal. The evidence says it does. The harder question is whether the project
can turn that signal into a controlled neural distinguisher contribution rather
than a deterministic feature/protocol artifact.

## Historical Adjudication Program

The E1-E3 program below records the experiments that were authorized by the
2026-07-09 verdict. The 2026-07-13 completion update above supersedes it for
new execution: do not repeat or scale these rows. Only the bounded E4 design is
currently eligible for a new readiness plan.

Only the following experiments are allowed before the next verdict update.
They are decision experiments, not exploration experiments.

### E1: Active-Relative Contrast 8192/Class Fragility Gate

Purpose:

```text
Decide whether the topology-contrast architecture route is still alive.
```

Protocol:

- Same 6-row matrix as the completed 4096/class gate.
- `samples_per_class = 8192`.
- Keep `pairs_per_sample = 4`.
- Keep strict encrypted-random-plaintext negatives.
- Keep `true`, `shuffled`, and `metadata_only` controls.
- Do not change feature encoding, labels, metric computation, or sample
  structure.

Gate:

```text
continue only if true > shuffled and true > metadata-only on both seeds,
and true-shuffled margins are materially larger than the 4096/class near-tie.
```

Interpretation:

- If it fails either control, stop this architecture route and redesign before
  any remote scale.
- If it remains ordered but near-tied, classify as weak/fragile and stop scale.
- If margins reopen on both seeds, prepare one medium diagnostic plan; do not
  call it formal evidence.

### 2026-07-09 E1 Completion Update

E1 has completed:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_active_relative_contrast_pair4_8192_seed0_seed1.csv
results = outputs/local_smoke/i1_present_r8_active_relative_contrast_pair4_8192_seed0_seed1/results.jsonl
status = completed local adjudication diagnostic
validation = pass
```

Control deltas:

| seed | true - shuffled AUC | true - metadata-only AUC |
| ---: | ---: | ---: |
| 0 | +0.003980964 | +0.009444445 |
| 1 | +0.008310705 | +0.020662129 |

Verdict:

```text
active-relative topology-architecture branch = verdict superseded by semantic-layout defect
remote_scale = no
repair_adjudication = E1-R1 completed and failed frozen controls
next_adjudication = E2 trail-position neural residual with deterministic baseline
```

The numerical ordering is retained as historical diagnostic evidence, but it
cannot decide the intended topology question because graph tokens were not
semantically cell-aligned. Corrected E1-R1 subsequently failed its frozen
controls, so do not run 16k/32k/65k topology follow-ups; proceed to E2.

### E2: Trail-Position Neural Residual With Deterministic Baseline

Purpose:

```text
Decide whether r8 trail-position statistics provide neural residual value
beyond deterministic split baselines.
```

Protocol:

- Same data construction as the trail-position beamstats route.
- Include same-input global-stat control.
- Include deterministic split baseline or gate artifact in the same report.
- Include neural trail-position candidate.
- Keep strict encrypted-random-plaintext negatives.

Gate:

```text
continue only if the neural candidate beats both the same-input global-stat
control and the declared deterministic baseline on the validation split.
```

Interpretation:

- If neural does not beat deterministic baseline, preserve it as representation
  evidence but do not claim neural architecture gain.
- If neural beats both controls, it becomes the leading r8 non-neighbor
  representation route.

### E3: Same-Budget Route Anchor Comparison

Purpose:

```text
Prevent route selection from comparing different budgets and protocols.
```

Candidates:

- strongest InvP-only/SPN-only anchor available under the target protocol;
- strongest trail-position representation route under matching controls;
- active-relative topology route only if E1 passes.

Gate:

```text
same cipher, rounds, sample structure, negative mode, samples_per_class,
pairs_per_sample where applicable, seeds, training budget, and validation
policy; compare val_auc first, then calibrated accuracy/loss.
```

Interpretation:

- If one route clearly wins under same-budget controls, focus on that route.
- If none wins, convert Innovation 1 into a controlled negative-result and
  methodology contribution rather than continuing broad search.

## Forbidden Next Actions

Do not spend another meaningful slot on:

- adding a new pair count without a new representation hypothesis;
- remote scaling active-relative contrast from the 4096/class near-tie;
- widening near-neighbor ensembles before a non-neighbor expert is confirmed;
- more deterministic InvP aggregate statistics similar to the held audits;
- repeating r8/r9 difference screens or curriculum probes that were already
  stopped;
- reporting any `8k`, `16k`, `32k`, or `65k` SPN/PRESENT run as formal
  training or definitive route failure.

## Stop Criteria

Innovation 1 should stop broad experiment search if the allowed experiments
produce the following:

```text
E1: active-relative contrast remains near-tied or fails controls
E2: trail-position neural candidate fails deterministic baseline
E3: no same-budget route beats its controls by a meaningful margin
```

In that case, the remaining contribution should be framed as:

```text
SPN/PRESENT controlled evidence showing which structure-aware representations
produce signal, which controls invalidate apparent gains, and why naive
architecture/ensemble scaling is insufficient.
```

## Continue Criteria

Innovation 1 may continue as a positive route only if at least one allowed
experiment produces a clear controlled result:

- active-relative contrast reopens true-vs-shuffled margins at 8192/class; or
- trail-position neural residual beats deterministic and global controls; or
- a same-budget comparison identifies a route that beats its baseline and
  controls consistently enough to justify a medium diagnostic ladder.

Even then, the next scale step must be documented as diagnostic, not formal.

## Reporting Template

Future Innovation 1 updates should use this short form:

```text
route = <name>
status = stopped | held | diagnostic-only | medium-candidate | formal-candidate
scale = <samples_per_class/class>
controls = passed | failed | incomplete
best delta = <primary route delta vs strongest same-budget control>
claim_scope = <local diagnostic | medium diagnostic | formal candidate>
next_action = <one concrete action or stop>
```

## Decision

Innovation 1 remains methodologically viable, but broad search and the
PRESENT-only architecture sequence are closed. E1, H2, and E3-R1 have already
served their stop/hold function. The next justified research slot is an E4
design that tests one shared typed operator hypothesis across PRESENT and GIFT
with same-input attribution and cross-cipher transfer. If E4 cannot define or
pass that local gate, consolidate the contribution as a controlled
structural-representation methodology result instead of adding more network
variants.
