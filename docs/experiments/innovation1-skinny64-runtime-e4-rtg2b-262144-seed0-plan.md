# Innovation 1 SKINNY RTG2-B Runtime E4 262144/Class Plan

Date: 2026-07-24

## Status

```text
stage       = running remotely; bounded startup evidence verified
run_id      = i1_rtg2b_skinny64_general_gf2_scale_262144_seed0_20260724
execution   = remote lxy-a6000 only
source      = 061fd9a3c30cd1089a24e9df241f63964d147d6c (published origin/main)
launch gate = pass / innovation1_rtg2b_seed0_remote_launch_authorized
started     = 2026-07-24 20:23:46 +08:00
monitor     = local tmux i1_rtg2b_skinny64_scale_monitor
claim scope = medium architecture/protocol diagnostic only
```

The watcher records `remote_launcher_returned`, waits for the remote
`started.marker`, and only then writes its local launch marker. The verified
startup evidence is under:

```text
outputs/remote_results_incomplete/i1_rtg2b_skinny64_general_gf2_scale_262144_seed0_20260724_monitor/
```

The first synchronized logs show the exact pinned source revision, a clean
detached source checkout, readiness `status=pass`, PyTorch `2.5.1+cu118`, CUDA
`11.8`, one visible device, and `NVIDIA RTX A6000`. No result or research
decision exists yet; the tmux watcher owns sparse synchronization, verified
retrieval, local re-adjudication, plot generation, visual QA, and index refresh.

## Evidence That Opens This Plan

The frozen RTG2-A `65536/class` protocol passed independently on both seeds:

| Seed | Correct | Corrupted | No topology | Correct - corrupted | Correct - no topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.643590577 | 0.600012660 | 0.510271238 | +0.043577916 | +0.133319339 |
| 1 | 0.644612943 | 0.597460402 | 0.513995145 | +0.047152541 | +0.130617798 |

The joint gate returned:

```text
status      = pass
decision    = innovation1_rtg2a_skinny_medium_two_seed_supported
next_action = freeze an RTG2-B 262144/class seed0 plan that changes only sample scale
```

Seed0 was fallback-retrieved and seed1 came from a verified result branch. The
joint gate preserves those distinct provenance states; it does not turn RTG2-A
into formal, paper-scale or universal-SPN evidence.

## Research Question

Does the correct general-GF(2) runtime topology retain its two-control
advantage when only the SKINNY-64/64 r7 sample scale increases from
`65536/class` to `262144/class`?

This is a sample-scale replication of an already supported architecture, not a
new model search. It must not be combined with the active uKNIT S-box-query
representation branch.

## One Changed Variable

```text
RTG2-A train = 65536/class
RTG2-B train = 262144/class
```

The engine's frozen default validation rule uses half the training
`samples_per_class`, giving:

```text
train      = 524288 total = 262144/class
validation = 262144 total = 131072/class
```

No other data, model or training variable may change.

## Frozen Protocol

```text
cipher                    = SKINNY-64/64
rounds                    = 7
difference profile        = skinny64_gohr2022_single_key
difference member         = 0
input difference          = 0x2000
seed                      = 0
train key                 = 0x0000000000000000
validation key            = 0x1111111111111111
train                     = 262144/class
validation                = 131072/class
pairs/sample              = 4 independent ciphertext pairs
feature                   = ciphertext_pair_bits
negative                  = encrypted_random_plaintexts
models                    = correct / deterministic corrupted / no linear topology
parameters/model          = 442466 trainable
runtime processor steps   = 2
pair embedding            = 128
S-box context             = late_pair
epochs                    = 5
batch                     = 64
optimizer                 = Adam, learning rate 1e-4
loss                      = MSE
weight decay              = 1e-5
learning-rate scheduler   = none
checkpoint                = best validation AUC
train evaluation interval = 1 epoch
device                    = remote CUDA, physical GPU0
```

## Three-Row Matrix

| Role | Model | Purpose |
| --- | --- | --- |
| candidate | `skinny64_runtime_e4_equivariant_true` | exact external general-GF(2) topology |
| topology control | `skinny64_runtime_e4_equivariant_corrupted` | deterministic full-bit topology corruption |
| no-topology control | `skinny64_runtime_e4_equivariant_independent` | retain cell/S-box metadata but remove linear relation |

The rows must use identical parameter geometry, dataset arrays, optimizer,
epochs, batch, validation split and checkpoint selection. Only the runtime
relation mode differs as already frozen in RTG2-A.

Matrix path:

```text
configs/experiment/innovation1/innovation1_spn_skinny64_runtime_e4_scale_rtg2b_262144_seed0.csv
```

## Readiness And Launch Gate

Before any remote contact require:

1. The RTG2-A joint gate exists locally, has the exact run id and decision, and
   all protocol and research checks are true.
2. The new CSV has exactly three rows and differs from RTG2-A seed0 only in
   run identity, evidence text and `samples_per_class`.
3. Every model builds with `442466` parameters and finite forward/backward.
4. Remote configuration reports `expected_rows=3`, CUDA, physical GPU0,
   five epochs, batch 64 and the exact strict-negative protocol.
5. Cache, checkpoints, logs, progress and results are all under
   `G:\lxy\blockcipher-structure-adaptive-nd-runs`.
6. The exact route uses disk-backed `features.npy`, `labels.npy`, metadata,
   chunked progress and parameter-matched reuse/resume before training.
7. Generated Windows scripts use `cmd.exe /c`, contain no delayed expansion or
   `!`, and write no project artifacts outside `G:\lxy`.
8. The launch source is a clean run-owned clone at an exact pushed commit.

A readiness pass authorizes launch; it is not result evidence. A failed push
keeps the remote launch blocked and does not authorize `scp` source overlays.

## Result Gate

All protocol checks must pass, including complete five-epoch histories,
best-checkpoint replay, strict negatives, equal geometry, exact row counts and
disk-backed cache metadata. Then require:

```text
correct AUC >= 0.55
correct AUC - corrupted AUC >= +0.005
correct AUC - no-topology AUC >= +0.005
```

Decisions:

```text
pass = prepare an identical 262144/class seed1 confirmation; do not launch it automatically
hold = stop mechanical scaling and audit scale/training dynamics without changing protocol
fail = repair only invalid evidence; do not interpret AUC
```

Even a seed0 pass remains a single-seed `262144/class` medium diagnostic. It
does not authorize `1000000/class`, formal claims, an attack, SOTA, a
breakthrough or universal-SPN conclusions.

## Outputs And Completion

Remote outputs must include `results.jsonl`, complete progress, three
checkpoints, validation, gate, summary, history, source revision, clean status,
GPU/torch/readiness logs, frozen plan/config and a SHA-256 manifest on a verified
result branch. A local tmux watcher must retrieve and validate the branch,
redraw the chart, execute `visual-qa-redraw`, write
`visual_qa_passed.marker`, refresh both recent-result indexes and update this
record with the metric deltas and evidence-backed next action.

## Blocked Routes

Do not change the difference, round count, key protocol, pairs, feature
encoding, negative definition, topology corruption, no-topology semantics,
model geometry, optimizer, epochs, batch or checkpoint rule. Do not add DDT,
trail, partial-decryption or uKNIT-specific S-box-query features. Do not start
seed1, `1000000/class` or a broad cipher matrix as a rescue.
