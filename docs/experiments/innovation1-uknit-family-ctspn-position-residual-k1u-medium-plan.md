# Innovation 1 uKNIT-Family CT-SPN Position Residual K1-U Medium Plan

**Date:** 2026-07-28
**Status:** frozen / launch assets validated / source publication pending
**Execution:** remote A6000 only after exact GitHub source verification

## 1. Research Question

K1-T changed one representation component and passed every two-seed local
mechanism gate. On untouched same-key and cross-key splits, the exact
stage-by-native-cell histogram residual reached AUC `0.713162-0.748229`, beating
the old K1-R path by at least `+0.212928`, wrong S-box semantics by at least
`+0.206177`, and position erasure by at least `+0.134500`.

K1-U asks one narrower question:

> Does that learned signal and its correct-S-box/native-position attribution
> survive when only the uKNIT r5 training and cross-key validation data scale
> increases by 32x?

K1-U is a remote `65536/class` medium diagnostic. It is not formal training,
paper-scale evidence, an attack, SOTA, a breakthrough, transfer to another
cipher, or proof of a universal SPN architecture.

## 2. Authority And Single Variable

Authority:

```text
run_id = i1_uknit_family_ctspn_deterministic_position_residual_
         k1t_2048_seed3_seed4_20260728
gate   = outputs/local_diagnostic/<run_id>/gate.json
status = pass
decision = innovation1_uknit_family_ctspn_k1t_
           deterministic_position_residual_supported
```

K1-U changes only:

```text
train       2048/class -> 65536/class
validation  1024/class -> 32768/class
device      local CPU  -> remote A6000
```

The device change is operational, not a research intervention. Cell11 role1,
the two-transition runtime window, model parameters, controls, keys, seeds,
loss, optimizer, ten epochs, checkpoint selection and all benchmark semantics
remain fixed.

## 3. Frozen Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Input difference | cell11 role1, `0x0000400000000000` |
| Seeds | `3`, `4` |
| Train | `65536/class`, `131072` total rows per seed |
| Cross-key validation | `32768/class`, `65536` total rows per seed |
| Pairs/sample | `4` independent ciphertext pairs |
| Input width | `512` bits/sample |
| Negative definition | encrypted random plaintexts |
| Train/validation keys | exact K1-T seed3/4 key pairs |
| Runtime window | `round_start=3`, `rounds=2` |
| Hidden / pair embedding | `32` / `128` |
| Histogram value width | `8` |
| Initial edge / histogram gates | `0.05` / `0.05` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best cross-key validation AUC, restored |
| Train evaluation interval | every epoch |

The six-row matrix is:

```text
configs/experiment/innovation1/
  innovation1_uknit_family_ctspn_position_residual_
  k1u_medium_65536_seed3_seed4.csv
```

For each seed independently train:

| Condition | Model | Same-budget role |
|---|---|---|
| exact | `runtime_spn_ct_k1t_position_histogram_true` | candidate and strongest local anchor |
| wrong S-box | `runtime_spn_ct_k1t_position_histogram_wrong_sbox` | nonlinear-semantic control |
| invariant | `runtime_spn_ct_k1t_position_histogram_invariant` | native-position-erasure control |

All rows must retain identical `214316`-parameter geometry. No DDT, trail,
cipher identity, active-cell feature, key, label or round-specific learned
shape may enter the model.

## 4. Remote Storage And Cache Gate

Use a run-owned clean clone and run root under:

```text
source = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>\source
run    = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
```

Before launch require:

1. exact local K1-T pass authority;
2. local `HEAD` equals the exact GitHub `main` SHA;
3. the remote source clone is clean and equals that SHA;
4. generated Windows launch commands use `cmd.exe /c` and contain no delayed
   expansion `!` characters;
5. every project-owned path is under `G:\lxy`;
6. cache chunk size/workers are `1024/1`;
7. each of the four parameter-matched seed/split caches writes
   `features.npy`, `labels.npy`, `metadata.json` and durable chunk progress;
8. the later two controls per seed reuse the exact candidate cache rather than
   regenerate data;
9. results, checkpoints, logs, manifests and archives remain under the run root.

The generic disk cache is part of the frozen implementation. It emits durable
`cache_start`, per-class chunk, `cache_flush_start`, `cache_done` and
`cache_reuse` events. K1-U requires four completed cache creations and eight
model-row reuses across two seeds and two splits.

## 5. Result Gate

Protocol validity requires:

- exactly six plan-aligned result rows and six best checkpoints;
- `131072` train and `65536` cross-key validation rows per result;
- disk-backed train/validation storage and `G:\lxy` cache/checkpoint paths;
- ten complete history epochs, finite metrics and restored best AUC checkpoint;
- correct descriptor/window hashes and equal `214316` parameter counts;
- all four cache payloads completed before training and all eight later accesses
  reused the parameter-matched cache;
- recorded source commit equals the launch-pinned GitHub commit.

Apply each research threshold separately to seed3 and seed4:

```text
exact cross-key AUC                    >= 0.600
exact - wrong-S-box cross-key AUC      >= +0.010
exact - invariant cross-key AUC        >= +0.030
```

No average may hide a failed seed.

## 6. Decisions And Next Action

- **Both seeds pass:** retain the position-residual mechanism at medium scale.
  Do not mechanically increase uKNIT data. Next redesign only the fixed
  sixteen-cell flatten projection into a runtime-cell-count parameterized
  shared aggregator, then test one preregistered same-budget transfer to another
  compatible SPN.
- **Exact learns but wrong-S-box margin fails:** hold scale and isolate which of
  the five deterministic stages supplies the medium-scale signal.
- **Exact learns but invariant margin fails:** position preservation is not
  necessary at this scale; prefer the simpler invariant branch.
- **One seed fails:** classify the medium mechanism as seed/key unstable and
  inspect the failed seed's history and checkpoint; do not add seeds or data.
- **Exact is below `0.600` on either seed:** hold the route and audit training
  dynamics versus K1-T; do not change pairs, epochs, difference or rounds.
- **Protocol invalid:** repair only the failed source, cache, checkpoint or
  artifact binding and rerun unchanged.

Blocked inside K1-U: local execution, `262144/class`, more epochs, pairs,
differences, rounds, seeds or keys; MoE; DDT/trails; cipher identity; another
network family; and family-transfer claims before retrieved evidence passes.

## 7. Required Artifacts

```text
run_id = i1_uknit_family_ctspn_position_residual_
         k1u_medium_65536_seed3_seed4_20260728
```

Require remote config, exact source revision, clean status, GPU evidence,
readiness report, four cache metadata bundles, durable progress, six
checkpoints, six results, plan validation, gate, summary, history, archive
manifest and SHA-256 manifest. A local watcher must retrieve the completed
archive, re-adjudicate it, create the Chinese SVG, run `visual-qa-redraw`, update
this document and refresh both recent-result indexes. Preparation or a running
job is not a completed indexed result.

## 8. Preparation Record

The K1-U execution path now has a route-specific result gate, archive packager,
Chinese result plotter, exact-source launch gate, clean-clone Windows runner,
Task Scheduler launcher and local result watcher. The remote readiness report
passes without warnings for six rows and `65536/class`; the focused K1-T/K1-U
regression set passes `17` tests, Ruff passes and `git diff --check` passes.

The synthetic K1-U plot template was rendered to `1920 x 1036` pixels and
inspected with `visual-qa-redraw`. Its title, explanatory text, two heatmaps,
annotations, color bars and export bounds have no overlap, clipping, missing
glyphs or ambiguous labels. This is template QA only. The retrieved result SVG
still requires a fresh rendered-pixel inspection before its pending marker can
be replaced by `visual_qa_passed.marker`.

No remote training has been launched at this preparation checkpoint. Launch
still requires a scoped commit, exact GitHub `main` SHA equality, a passing K1-U
launch gate, one bounded GPU1/run-root check and durable remote start evidence.
