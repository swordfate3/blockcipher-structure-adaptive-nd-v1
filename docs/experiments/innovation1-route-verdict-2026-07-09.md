# Innovation 1 Route Verdict 2026-07-09

## Status

status = E4-R2 seed0 completed / identical local seed1 repeat authorized
decision = true-to-true transfer passed all seed0 attribution margins
claim_scope = single-seed 8192/class local diagnostic only
next_adjudication = E4-R2 identical target-seed1 transfer repeat

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
| E4-R2 explicit typed checkpoint transfer | seed0 `8192/class`: true-to-true `0.569627` vs anchor `0.506567`, scratch `0.551969`, source-shuffled `0.544660`, target-shuffled `0.508950` | promote identical local seed1 repeat only | all frozen absolute, anchor, scratch, source-topology, and target-topology margins pass; single seed is not scale evidence |

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
scale and retain seed0 only as provisional evidence. DDT/trail reopening,
architecture changes, `65536/class`, `262144/class`, and remote GPU remain
stopped now.

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
