# Innovation 1 Dialga-128 Runtime-E4 DMC2 R4 262144 Plan

Date: 2026-08-01

## Status

```text
phase = completed remotely; raw-fallback retrieved; locally re-adjudicated; pass
scale = 262144/class scale confirmation, not formal evidence
paper role = highest-scale Dialga result in the bounded mechanism manuscript
```

## Remote Launch Record

```text
launched_at       = 2026-08-01 23:42 CST
source_commit     = 9168cbd8881c539c3db9b5da9a96f83a5c25dc5b
github_main       = exact SHA match verified before launch
remote_host       = lxy-a6000 / DESKTOP-BBLPACJ
physical_gpu      = 0
remote_run_root   = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_dialga128_runtime_e4_dmc2_r4_262144_seed0_seed1_20260801
launch_gate       = pass / innovation1_dialga_dmc2_remote_launch_authorized
durable_start     = source_expected_commit.txt, readiness.txt, started.marker, progress.jsonl
local_monitor     = tmux:i1_dialga_dmc2_262k_monitor
retrieval_status  = fallback-retrieved after incomplete result-branch archive
```

The historical standard clone at
`G:\lxy\blockcipher-structure-adaptive-nd` was found dirty and behind its
upstream, so it was excluded without modification. The launch instead created
the run-owned `source` clone directly from GitHub, detached it at the exact
verified commit, and then invoked the committed launcher. The local monitor is
responsible for completion detection, verified-branch retrieval or raw
fallback retrieval, local re-adjudication, result indexing, and the visual-QA
pending marker. The main task must not SSH-poll this run.

## Research Question

Does the Dialga prefix-r4 Runtime-E4 topology margin survive a fourfold increase
from DMC1's `65536/class` to `262144/class` while every model, seed, data,
optimization, and gate field remains frozen?

DMC1 is the same-budget anchor. Its fallback-retrieved remote gate passed every
protocol and research check for both seeds, with correct-minus-corrupted AUC
margins `+0.013189230/+0.014329014`. DMC2 changes only training and validation
scale. It does not change the observed round boundary and does not add final
tests; those belong to DFC1 only after DMC2 passes.

## Frozen Matrix

Two seeds by three models, exactly six rows:

| role | model key | purpose |
| --- | --- | --- |
| correct | `runtime_spn_e4_equivariant_true` | exact Dialga runtime topology |
| corrupted | `runtime_spn_e4_equivariant_corrupted` | deterministic wrong topology |
| generic | `autond_dbitnet2023` | same-protocol generic architecture baseline |

```text
cipher / rounds      = Dialga-128 parent / encrypted prefix-r4
input difference     = 0x40
pairs per sample     = 4 independent ciphertext pairs
train                = 262144/class = 524288 total rows per model/seed
cross-key validation = 65536/class = 131072 total rows per model/seed
fresh final test     = disabled; DFC1-only requirement
seeds                = 0, 1
train key            = 0x00 repeated 32 bytes
validation key       = 0x11 repeated 32 bytes
negative mode        = encrypted_random_plaintexts
sample structure     = independent_pairs
epochs / batch       = 10 / 64
loss / optimizer     = MSE / Adam, lr 1e-4, weight decay 1e-5
checkpoint           = best validation AUC, restored
Runtime-E4 window    = round_start 2, rounds 2
execution            = remote A6000 GPU0, disk-backed cache/progress/reuse
```

The run must create four train/validation caches shared by seed and split,
perform eight parameter-matched cache reuses, restore six nonempty checkpoints,
and emit six result rows with sixty epoch-history rows. All files remain under
`G:\lxy`.

## Preregistered Gate

For each seed independently:

```text
correct AUC >= 0.900
correct - corrupted AUC >= +0.005
correct - AutoND AUC >= +0.010
```

Protocol validity additionally requires the exact pushed source commit, frozen
six-row plan, four disk caches, eight cache reuses, six checkpoints, exact row
counts, fixed keys/difference/negatives, and no independent-final-test fields.
A seed cannot be rescued by averaging.

## Preregistered Decisions (Historical)

- Pass: preregister DFC1 at at least `1000000/class`, multiple seeds, and five
  fresh independent final-test repeats under a third fixed key.
- Hold: stop mechanical scaling and inspect restored-best histories and cache
  equivalence. Do not change rounds, pairs, models, or thresholds.
- Invalid: repair only the failed source, plan, cache, checkpoint, or result
  binding and rerun DMC2 unchanged under a new unique run id if needed.

## Claim Boundary

DMC2 is a two-seed `262144/class` scale confirmation. It is not formal or
paper-scale evidence, not a full-round Dialga result, not a key-recovery result,
not a SOTA claim, and not evidence for a single universal SPN network. Under the
original gate terminology, a subsequent DFC1 run with at least
`1000000/class` and independent final tests would have been required for a
formal project-scale row. The author has instead capped the paper at DMC2 and
removed that stronger claim from scope.

## Executable Next Action

Use DMC2 as the manuscript's highest-scale Dialga evidence, retain its
raw-fallback and no-independent-final-test qualifications, and perform no
further mechanical data scale-up. The historical DFC1 launch was cancelled on
2026-08-03 and must not be resumed automatically.

## Completed Result And Local Re-adjudication (2026-08-02)

The remote A6000 completed all six rows at the frozen source commit. The first
result branch contained the textual evidence and checksum manifest but omitted
the six checkpoint payloads, so it was not accepted as a verified result-branch
archive. The local watcher then retrieved the named raw archive from the
project-owned `G:\lxy` run root. All files listed by the CRLF-normalized
`SHA256SUMS`, including six checkpoints, passed verification before the archive
was exposed under `outputs/remote_results_incomplete/`.

| seed | correct AUC | corrupted AUC | AutoND AUC | correct-corrupted | correct-AutoND |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.984964147 | 0.967530599 | 0.502369641 | +0.017433548 | +0.482594506 |
| 1 | 0.984212034 | 0.967087931 | 0.501357568 | +0.017124103 | +0.482854466 |

The local gate passed all protocol checks: exact six-row matrix, frozen source
revision, four disk-cache creations, eight parameter-matched reuses, six
nonempty checkpoints, fixed data sizes, keys, difference, negative definition,
optimizer and ten complete epochs. Every preregistered research threshold also
passed independently on seed0 and seed1.

```text
status       = pass
decision     = innovation1_dialga_dmc2_scale_topology_supported
formal_scale = authorized_dfc1_preregistration
claim_scope  = two-seed remote 262144/class prefix-r4 scale confirmation
```

The Chinese comparison SVG was rendered at 1920 x 1053 pixels and inspected
with `visual-qa-redraw`. After increasing the upper margin-axis headroom, the
legend no longer occludes the seed1 AutoND-margin annotation; the final title,
protocol text, thresholds, heatmap values, bars, labels and export bounds have
no overlap, clipping, missing glyphs or ambiguity.

## Superseded DFC1 Scale Action

The original gate authorized DFC1 as the only mechanical scale continuation:

```text
cipher / rounds      = Dialga-128 parent / encrypted prefix-r4
models               = correct Runtime-E4, corrupted Runtime-E4, AutoND
seeds                = 0, 1
train                = 1000000/class = 2000000 total rows/model/seed
cross-key validation = 250000/class = 500000 total rows/model/seed
fresh final test     = 5 repeats x 1000000 total rows/model/seed
train / val / test key = 0x00 / 0x11 / 0x22 repeated 32 bytes
pairs / epochs       = 4 / 10
negative mode        = encrypted_random_plaintexts
single variable      = DMC2 to DFC1 data scale plus preregistered final test
execution            = remote A6000 with disk cache, progress and reuse
```

DFC1 must retain the DMC2 validation thresholds on each seed and require every
fresh final-test repeat to satisfy the same correct AUC and two control margins.
A pass permits one formal *project* result row for this exact Dialga prefix-r4
protocol. It still does not imply a full-round result, key recovery, SOTA, a
universal SPN network, or superiority over deterministic cryptanalysis. A hold
or invalid result stops mechanical scaling. Dialga prefix-r5 remains blocked.

## Author Scope Decision (2026-08-03)

The author capped the current paper at `262144/class` and cancelled the running
DFC1 matrix after one of six rows. DFC1 is therefore excluded from all result
claims. DMC2 remains non-formal, non-paper-scale evidence under the original
gate terminology, but it is sufficient for the manuscript's narrower claim:
within this two-seed Dialga prefix-r4 protocol, correct topology consistently
beats the deterministic corrupted-topology control and the same-budget AutoND
baseline through `262144/class`.

The executable next action is manuscript synthesis and evidence audit, not
additional scale-up. Preserve the fallback-retrieval qualification, state that
`final_test_repeats=0`, and do not relabel DMC2 as an AutoND reproduction,
universal-SPN result, full-round attack, SOTA result or formal scale benchmark.
