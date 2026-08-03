# Innovation 1 Dialga-128 Runtime-E4 DFC1 R4 Formal Plan

Date: 2026-08-02; cancelled 2026-08-03 by author scope decision

## Status

```text
phase = user-cancelled; remote task ended and disabled; watcher stopped
scale = incomplete 1000000/class attempt, excluded from evidence
paper role = none; DMC2 262144/class is the manuscript scale ceiling
```

## Remote Launch Record

```text
launched_at       = 2026-08-02 12:54 CST
source_commit     = b48a13039e373211c83a9d721ec43258512fa351
github_main       = exact SHA match verified before launch
launch_gate       = pass / innovation1_dialga_dfc1_remote_launch_authorized
remote_host       = lxy-a6000 / DESKTOP-BBLPACJ
physical_gpu      = 0
remote_run_root   = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_dialga128_runtime_e4_dfc1_r4_1000000_seed0_seed1_20260802
source_mode       = clean run-owned clone, detached at the exact pushed commit
durable_start     = source_expected_commit.txt, readiness.txt, started.marker
local_monitor     = stopped; tmux:i1_dialga_dfc1_1m_monitor removed
result_status     = user-cancelled; 1/6 rows complete; no comparison claim
```

The bounded prelaunch audit found GPU0 without a compute training process and
the new run root absent. The historical standard clone was on an old branch,
behind its upstream and heavily modified, so it was left untouched. A committed
bootstrap launcher under `G:\lxy\scheduled-runs` created the clean run-owned
clone directly from GitHub. The first and only post-launch remote confirmation
found the exact source pin, readiness evidence and `started.marker`; the
progress file was not yet present at that instant. The main task must not SSH
poll the run. Completion, archive retrieval, local re-adjudication and result
indexing are delegated to the local watcher.

## Cancellation Record

The author stopped DFC1 on 2026-08-03 because the manuscript is scoped as a
structure-mechanism and attribution study whose experiment ceiling is
`262144/class`. The stop targeted only scheduled task
`I1_DIALGA_DFC1_S0S1_GPU0` and this run id. The existing run directory, cache,
checkpoint and logs were retained.

```text
verified_at         = 2026-08-03T10:53:19+08:00
scheduled_task      = Disabled; Enabled=false
matching_python     = NO_MATCH for the exact DFC1 run id
local_watcher       = stopped
completed_rows      = 1 of 6 (correct topology, seed 0)
interrupted_row     = row 2, corrupted topology seed 0, validation start at epoch 6
unstarted_rows      = rows 3 through 6
publication_status  = excluded; incomplete controls prevent adjudication
```

One completed candidate row and five fresh-test repeats exist in the retained
progress log, but the corrupted-topology and AutoND controls and the second
seed are incomplete. These values must not be reported as a DFC1 result, used
to strengthen DMC2, indexed as a completed experiment, or resumed
automatically.

## Research Question

At a preregistered `1000000/class` training scale, does the exact Dialga
Runtime-E4 topology retain its prefix-r4 cross-key and fresh third-key signal,
and does it beat both a deterministic wrong-topology control and a generic
AutoND/DBitNet architecture on every seed and every final-test repeat?

The same-budget anchor is DMC2. DFC1 changes the training/validation scale and
adds the preregistered independent final-test protocol; it does not change the
cipher prefix, difference, pair count, architectures, optimizer, epoch budget,
negative definition, or acceptance margins.

## Authority

DMC2 completed remotely at source commit
`9168cbd8881c539c3db9b5da9a96f83a5c25dc5b`, was raw-fallback retrieved,
checksum verified, and locally re-adjudicated. Both seeds passed:

| seed | correct AUC | corrupted AUC | AutoND AUC | correct-corrupted | correct-AutoND |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.984964147 | 0.967530599 | 0.502369641 | +0.017433548 | +0.482594506 |
| 1 | 0.984212034 | 0.967087931 | 0.501357568 | +0.017124103 | +0.482854466 |

The launch gate is bound byte-for-byte to:

```text
outputs/remote_results_incomplete/
i1_dialga128_runtime_e4_dmc2_r4_262144_seed0_seed1_20260801/gate.local.json
SHA256 = fb92b743157072f8a6aa2209d50ea9cd372f6eec08d8c5f96f479d037f03e1b0
status = pass
decision = innovation1_dialga_dmc2_scale_topology_supported
formal_scale = authorized_dfc1_preregistration
```

This authority is locally closed evidence, but its source archive was retrieved
through the raw fallback because the result branch omitted checkpoints. It is
not relabeled as verified result-branch retrieval.

## Frozen Matrix

Two seeds by three models, exactly six training rows:

| role | model key | purpose |
| --- | --- | --- |
| correct | `runtime_spn_e4_equivariant_true` | exact Dialga Runtime-E4 topology |
| corrupted | `runtime_spn_e4_equivariant_corrupted` | deterministic wrong topology |
| generic | `autond_dbitnet2023` | same-data and same-optimization-budget generic baseline |

```text
cipher / rounds      = Dialga-128 parent / encrypted prefix-r4
input difference     = 0x40
pairs per sample     = 4 independent ciphertext pairs
seeds                = 0, 1
train                = 1000000/class = 2000000 total rows/model/seed
cross-key validation = 250000/class = 500000 total rows/model/seed
fresh final test     = 5 repeats x 1000000 total rows/model/seed
train key            = 0x00 repeated 32 bytes
validation key       = 0x11 repeated 32 bytes
final-test key       = 0x22 repeated 32 bytes
negative mode        = encrypted_random_plaintexts
sample structure     = independent_pairs
epochs / batch       = 10 / 64
loss / optimizer     = MSE / Adam
learning rate        = 1e-4
weight decay         = 1e-5
optimizer transition = reset_each_stage
scheduler            = none
checkpoint           = best validation AUC, restored before final tests
Runtime-E4 window    = round_start 2, rounds 2
execution            = remote A6000 physical GPU0
```

AutoND is evaluated on the identical cached data and optimizer budget. It is
not parameter-matched to Runtime-E4 and this run is not an exact reproduction
of AutoND's public-code `10^7`/`10^6`-row PRESENT protocol.

## Cache And Artifact Contract

The first model for each seed creates seven disk-backed datasets: train,
validation, and `final_test_1` through `final_test_5`. The two later control
models must reuse the exact parameter-matched datasets.

```text
cache creations = 2 seeds x 7 splits = 14
cache reuses    = 14 caches x 2 control models = 28
checkpoints     = 6 nonempty restored-best .pt files
result rows     = 6
final records   = 5 per result row, with frozen seed/key/row counts
```

Every cache must contain `features.npy`, `labels.npy`, `metadata.json`, and
durable progress events under the run-owned `G:\lxy` directory. The result
archive must include all six checkpoints. Because `.pt` is normally ignored,
the result-branch script must use scoped forced staging for the named archive.

## Preregistered Gate

For each seed's cross-key validation independently:

```text
correct AUC >= 0.900
correct - corrupted AUC >= +0.005
correct - AutoND AUC >= +0.010
```

For every one of the five fresh third-key repeats on every seed, apply the same
three thresholds independently. A mean cannot rescue a failed seed or repeat.
Protocol validity additionally requires the exact pushed source revision,
frozen six-row plan, 14 cache creations, 28 cache reuses, six checkpoints,
correct result/final-test structure, and all fixed data and optimizer fields.

## Decisions

- Pass: freeze the exact Dialga prefix-r4 project-formal result for the
  manuscript and stop mechanical scaling.
- Hold: inspect restored-best histories and cache equivalence, report every
  failed seed/repeat, and stop mechanical data or round scaling.
- Invalid: repair only the failed source, plan, cache, checkpoint, archive, or
  result binding and rerun the unchanged protocol under a new unique run id.

Dialga prefix-r5 is blocked in every branch of this gate.

## Remote Execution And Retrieval

```text
run id = i1_dialga128_runtime_e4_dfc1_r4_1000000_seed0_seed1_20260802
host   = lxy-a6000 / DESKTOP-BBLPACJ
GPU    = physical GPU0
root   = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run-id>
source = exact GitHub-pushed commit in a clean run-owned detached clone
launch = Windows Task Scheduler through cmd.exe /c
watch  = local tmux watcher; main task does not SSH-poll
```

Before launch, the fail-closed gate must prove local `HEAD == origin/main`, the
DMC2 authority hash and semantics match, every required source asset is
committed and byte-identical to the worktree, protected paths are clean, and
remote readiness passes. After launch, one bounded read-only check must confirm
the expected run root has source pin, readiness, started marker, or advancing
progress. Retrieval prefers a complete verified result branch and falls back
to the named raw archive only when the branch is unavailable or incomplete.

## Claim Boundary

A passing DFC1 result supports only this project's two-seed, three-model,
`1000000/class` Dialga-128 encrypted prefix-r4 evaluation with five fresh
third-key tests. It is not evidence for full 16/20-round Dialga, key recovery,
SOTA, an exact reproduction of another paper's protocol, a single shared-weight
network that handles all SPNs, or superiority over deterministic cryptanalysis.
Because the six-row matrix was cancelled after one completed row, DFC1 is not a
completed manuscript result and has no publication claim. Its partial files
are retained only for an audit trail.

## Executable Next Action

Do not resume, relaunch, repackage or index DFC1. Preserve its remote and local
logs as cancelled-run audit material. Use the completed DMC2 two-seed
`262144/class` result as the manuscript's highest-scale Dialga evidence, retain
its raw-fallback and no-independent-final-test qualifications, and continue
paper synthesis without additional mechanical scale-up.
