# Innovation 1 SKINNY RTG3-A Runtime E4 1000000/Class Seed0 Plan

Date: 2026-07-25

## Status

```text
stage       = completed remotely, verified-branch retrieved, locally re-adjudicated
run_id      = i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725
execution   = remote lxy-a6000 GPU0 only
dependency  = RTG2-B 262144/class two-seed pass plus completed post-X2 route audit
source      = 131e6387e164fb8679b1bcb9bf46887cf049fca0 (verified origin/main)
launch gate = pass / innovation1_rtg3a_seed0_remote_launch_authorized
started     = 2026-07-25 00:42:34 +08:00
retrieved   = 2026-07-25 06:16:04 +08:00
visual QA   = pass / visual-qa-redraw / 2026-07-25 06:28:33 +08:00
result      = pass / innovation1_rtg3a_skinny_formal_seed0_supported
successor   = seed1 launch gate passed; local monitor started at 06:30 +08:00
claim scope = single-seed project-formal scale evidence only
```

The watcher observed the remote launcher return, confirmed the exact run-owned
`started.marker`, retrieved the immutable verified result branch, verified the
CRLF-normalized archive manifest, replayed all three selected checkpoints,
re-adjudicated the result locally and refreshed the result index. The retrieved
SVG then passed rendered-pixel `visual-qa-redraw` inspection.

## Research Question

Does the SKINNY-64/64 r7 correct runtime general-GF(2) topology advantage
survive at the project's formal evidence floor of `1000000/class`, when the
network, data protocol, controls and five-epoch training budget are frozen from
RTG2-B?

This is a sample-scale replication, not a new architecture, feature route,
negative definition, published-paper reproduction or attack.

## Same-Budget Anchor And One Variable

The exact anchor is the completed RTG2-B matrix:

| Seed | Correct topology | Corrupted topology | No topology |
| ---: | ---: | ---: | ---: |
| 0 | 0.649229395 | 0.603561698 | 0.510189938 |
| 1 | 0.647782881 | 0.602584307 | 0.513038491 |

RTG3-A changes one research field:

```text
samples_per_class = 262144 -> 1000000
```

Run identity and descriptive evidence fields also change. Cipher, rounds,
difference, keys, four-pair input, negatives, model keys, model options,
optimizer, loss, batch size, epochs, topology controls and checkpoint rule do
not change.

## Frozen Protocol

```text
cipher                   = SKINNY-64/64
rounds                   = 7
difference               = 0x2000
train key                = 0x0000000000000000
validation key           = 0x1111111111111111
train                    = 2000000 total = 1000000/class
validation               = 1000000 total = 500000/class
pairs/sample             = 4 independent ciphertext pairs = 512 input bits
negative                 = encrypted random plaintexts
models                   = correct / deterministic corrupted / no topology
parameters               = 442466 per model
epochs                   = 5/model
batch                    = 64
optimizer                = Adam, learning rate 1e-4
loss                     = MSE
weight decay             = 1e-5
checkpoint               = best validation AUC
seed                     = 0
device                   = remote CUDA only
```

All three roles must reuse byte-identical disk-backed train and validation
arrays. Cache chunks, metadata and progress must be durable under the run-owned
`G:\lxy\blockcipher-structure-adaptive-nd-runs` directory before training
finishes. A parameter-matched cache may resume only when its complete protocol
metadata matches.

## Readiness And Launch Gate

Before SSH contact require:

1. The RTG2-B joint gate and validation both pass with exact identities.
2. The post-X2 route decision selects Route A.
3. The formal CSV differs from RTG2-B only in scale and descriptive identity.
4. Remote readiness confirms three rows, CUDA protocol, disk cache, progress,
   checkpoint and `G:\lxy` ownership.
5. The exact source commit contains and byte-matches every plan, gate, run,
   launch and monitor asset.
6. The exact source commit is published and independently verified on
   `origin/main`.
7. Generated Windows scripts contain `cmd.exe /c`, no `cmd.exe /k`, no delayed
   expansion and no `!` characters.

The launcher must use a clean run-owned clone and exact detached source commit.
One bounded read-only startup confirmation must observe the remote
`started.marker`; the local tmux watcher then owns monitoring and retrieval.

## Result Gate

Protocol failure if any of the three rows, sample totals, cache ownership,
five-epoch histories, strict negatives, equal geometry, source revision,
checkpoint replay or immutable archive checks fail.

The result gate now enforces both cross-role equality and an absolute frozen
contract. All three rows must report the exact train/validation keys, balanced
labels, four-pair raw ciphertext input, `442466` trainable parameters, two
loaded runtime transitions, `late_pair` model options, fixed-key schedule,
Adam/MSE optimization fields, `2000000` training rows and `1000000`
validation rows. Validation metadata must independently account for
`500000/class`. This prevents three consistently drifted rows from passing
merely because they agree with one another. Replaying the stronger gate on both
completed RTG2-B result files preserves both prior pass decisions with no failed
protocol checks.

The later two-seed joint gate also revalidates each source gate instead of
trusting its status string. It requires exact per-seed train and validation
totals, recomputes both topology margins from the three AUCs, and verifies that
all research booleans match the frozen `0.55/+0.005` thresholds. Replaying this
joint contract on the two completed RTG2-B source gates preserves the existing
two-seed pass with both new checks true.

Research pass requires:

```text
correct AUC >= 0.55
correct - corrupted AUC >= +0.005
correct - no-topology AUC >= +0.005
```

Decisions:

```text
pass:
  decision    = innovation1_rtg3a_skinny_formal_seed0_supported
  next_action = release the prepared identical seed1 package through its strict
                local checkpoint, visual-QA and publication gate

hold:
  decision    = innovation1_rtg3a_skinny_formal_not_supported
  next_action = stop scale-up and audit fixed-protocol dynamics; do not change
                model, data or thresholds as a rescue

protocol fail:
  decision    = innovation1_rtg3a_skinny_formal_protocol_invalid
  next_action = repair evidence only
```

A seed0 pass is single-seed project-formal evidence. It does not authorize a
two-seed formal conclusion until the identical seed1 result passes and a joint
gate is retrieved, validated, indexed and visually checked.

## Completed Seed0 Result

All three frozen rows completed five epochs with the required disk-backed
cache, strict encrypted-random-plaintext negatives, four independent ciphertext
pairs, equal `442466`-parameter geometry and exact RuntimeE4 model options.
Plan validation, absolute protocol checks, checkpoint replay and the research
gate all report `status = pass`.

| Role | Best validation AUC | Best epoch | Difference from correct |
| --- | ---: | ---: | ---: |
| Correct GF(2) topology | 0.653191631304 | 5 | - |
| Deterministically corrupted topology | 0.607162432806 | 4 | +0.046029198498 |
| No linear topology | 0.511826118586 | 5 | +0.141365512718 |

Relative to the same-protocol RTG2-B seed0 anchor at `262144/class`, the
correct-topology AUC changed from `0.649229395` to `0.653191631304`. More
importantly, its margins remained stable rather than collapsing when only the
sample scale increased: correct-minus-corrupted changed from approximately
`+0.045668` to `+0.046029`, and correct-minus-no-topology changed from
approximately `+0.139039` to `+0.141366`.

The reported AUCs replay the selected best-validation-AUC checkpoints exactly.
All three checkpoint files exist, load strictly into the frozen model geometry,
match the recorded history and final metrics, and have exact archived SHA-256
evidence. The correct model's train-minus-validation AUC at its best epoch is
only `+0.004272`, so the formal-scale advantage is not explained by a large
train/validation gap.

The final chart was rendered to approximately `1976 x 1019` pixels. Chinese
glyphs, title, subtitle, green decision line, legends, value labels, axes,
`0.50` random baseline, `0.55` signal threshold, `+0.005` attribution threshold
and evidence-scope caption are readable with no overlap, clipping or misleading
axis truncation. No redraw was required; `visual_qa_pending.marker` was replaced
by `visual_qa_passed.marker`.

This supports one narrow conclusion: for SKINNY-64/64 r7 and seed0, the same
cipher-name-free RuntimeE4 backbone retains a substantial advantage from the
correct externally supplied general-GF(2) topology at the project's formal
sample floor. It is not yet a two-seed formal conclusion, cross-cipher weight
reuse result, paper reproduction, attack, paper-scale comparison, SOTA or
universal-SPN evidence.

Evidence-backed next action:

```text
next question = does the same formal-scale topology advantage replicate at seed1?
anchor        = this seed0 result plus the completed RTG2-B two-seed matrix
change        = seed only: 0 -> 1
fixed         = cipher, r7, difference, keys, 1000000/class train,
                500000/class validation, 4 pairs, three controls, RuntimeE4,
                442466 parameters, 5 epochs, optimizer, loss and checkpoint rule
execution     = prepared remote GPU seed1 successor from its exact pushed commit
advance gate  = valid archive and checkpoints; correct >= 0.55; both margins >= 0.005
stop gate     = hold on a research miss; repair evidence only on protocol failure
blocked       = no architecture rescue, no extra epochs/pairs, no broad cipher
                matrix and no universal claim before the two-seed joint gate
```

## Prepared Conditional Seed1 Successor

The seed1 matrix, remote configuration, Windows run/launch scripts, result
monitor and a separate local-only successor watcher are prepared under run id
`i1_rtg3a_skinny64_general_gf2_formal_1000000_seed1_20260725`. Preparation is
not launch authorization. The successor consumes only local seed0 artifacts
and cannot contact `lxy-a6000` itself.

After seed0 retrieval and visual QA, the successor generated the strict seed1
launch gate. It passed source publication, seed-only plan equivalence,
unchanged protected training paths, remote disk-cache readiness and complete
seed0 evidence, then started local tmux session
`i1_rtg3a_skinny64_formal_seed1_monitor` at 06:30 +08:00. That monitor owns the
remote launch, bounded started-marker confirmation and later result retrieval;
the main thread does not poll the workstation. Until its start artifact is
confirmed, seed1 is launch-authorized/monitor-started rather than a reported
training result.

It stops without launching when seed0 reports a research hold or protocol
failure. A supported seed0 must additionally supply exactly three locally
validated result rows, fifteen ordered history rows, three strictly replayed
checkpoints, verified-result-branch provenance and
`visual_qa_passed.marker`. Only then does it generate the seed1 publication
gate. That gate must also prove seed-only plan equivalence, unchanged protected
training paths, passing remote readiness and exact source publication before
it starts the independent seed1 tmux monitor.

## Required Artifacts

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>\
  cache\
  checkpoints\
  logs\progress.jsonl
  logs\<run_id>_started.marker
  results\results.jsonl
  source\results_archive\<run_id>\

outputs/remote_results/<run_id>/
  results.jsonl
  validation.local.json
  gate.local.json
  checkpoint-verification.local.json
  history.local.csv
  curves.svg
  visual_qa_passed.marker
```

After retrieval, preserve the remote manifest-owned archive unchanged, write
local re-adjudication to separate `*.local.*` files, run rendered-pixel visual
QA, refresh both recent-result indexes and update this document with exact
metrics and the evidence-backed next action.

## Blocked Routes

Do not launch seed1 before seed0 passes. Do not run Route B concurrently. Do
not change rounds, difference, keys, negative semantics, pair count, topology
corruption, model geometry, optimizer, epochs, threshold or checkpoint rule.
Do not call this paper-scale, an exact reproduction, an attack, SOTA, a
breakthrough or universal-SPN evidence.
