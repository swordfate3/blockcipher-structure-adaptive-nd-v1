# Innovation 1 Route Verdict 2026-07-09

## Status

status = route verdict / execution mode switch
decision = stop broad exploration; allow only bounded adjudication experiments
claim_scope = project planning and evidence triage only

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

## Allowed Next Experiments

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

Innovation 1 remains methodologically viable, but broad search is over. The
project should now run only adjudication experiments that can stop, hold, or
focus a route. The default next action is E1 if the goal is to adjudicate the
current topology-architecture branch, or E2 if the goal is to maximize the
chance of a usable SPN-aware representation contribution.
