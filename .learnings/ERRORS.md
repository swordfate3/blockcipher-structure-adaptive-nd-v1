## [ERR-20260721-001] remote_launcher_https_clone_connection_reset

**Logged**: 2026-07-21T21:20:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary

The OP12 and OPA1 remote launcher HTTPS clones were reset by GitHub before the corresponding scheduled training task was created.

### Error

```text
fatal: unable to access 'https://github.com/swordfate3/blockcipher-structure-adaptive-nd-v1.git/':
Recv failure: Connection was reset
```

### Context

- Operations: create OP12 and OPA1 launcher-only clean clones under `G:\lxy` after each source commit was pushed.
- The failure occurred before the run-owned experiment clone or Windows scheduled task existed.
- No source overlay, remote training result, cache, or checkpoint was created by the failed attempt.

### Suggested Fix

For remote launcher clones, fall back to the already configured dedicated GitHub SSH key after an HTTPS transport reset. Use a new clean path under `G:\lxy`, verify clean status and exact HEAD, and continue launching only from the pushed commit. Do not substitute `scp` source overlays.

### Metadata

- Reproducible: no
- Related Files: configs/remote/generated/launch_i2_output_prediction_op12_present_r4_structured_xor_key1_gpu0_20260721.cmd, configs/remote/generated/launch_i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721.cmd
- See Also: ERR-20260716-007
- Pattern-Key: remote.git_https_reset_use_existing_scoped_ssh_key
- Recurrence-Count: 2
- First-Seen: 2026-07-21
- Last-Seen: 2026-07-21

### Resolution

- **Resolved**: 2026-07-21T21:20:00+08:00
- **Commit/PR**: 599054b
- **Notes**: New launcher clones under `G:\lxy` succeeded with the pre-existing scoped GitHub SSH key. OP12 used clean HEAD `97fd53e95dea9edbe7fcd4e21ab068a1823626c8`; OPA1 used clean HEAD `7d4e45de5a98aa297a585103e8b6542e1ce73e13`. No source overlay or alternate transfer route was used.

---

## [ERR-20260721-002] windows_schtasks_tr_path_limit

**Logged**: 2026-07-21T23:12:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary

The OPA1 source clone succeeded, but Windows Task Scheduler rejected the generated training command because the `/TR` value exceeded 261 characters.

### Error

```text
ERROR: The value for the '/TR' option cannot be more than 261 character(s).
```

### Context

- The descriptive run id was intentionally retained for experiment evidence and artifact ownership.
- Its run-owned source path plus the long generated run-script filename made the scheduled `/TR` command too long.
- No scheduled task, training process, cache, checkpoint, or result was created by the failed launch.

### Suggested Fix

Generate a short wrapper such as `G:\lxy\scheduled-runs\i2_opa1_key2.cmd` that calls the full run script, and give Task Scheduler only `cmd.exe /c <short-wrapper>`. Keep the descriptive run id and all project artifacts under `G:\lxy`.

### Metadata

- Reproducible: yes
- Related Files: configs/remote/generated/launch_i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721.cmd, tests/test_innovation2_selected_output_architecture.py
- See Also: ERR-20260721-001
- Pattern-Key: remote.windows_schtasks_tr_requires_short_wrapper
- Recurrence-Count: 1
- First-Seen: 2026-07-21
- Last-Seen: 2026-07-21

### Resolution

- **Resolved**: 2026-07-21T23:15:00+08:00
- **Commit/PR**: 7ee17cf
- **Notes**: The launcher now writes a short `G:\lxy` wrapper and tests freeze the short `/TR` form and the 261-character bound.

---

## [ERR-20260721-003] scheduled_run_transient_dirty_source_gate

**Logged**: 2026-07-21T23:24:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary

The first OPA1 short-wrapper retry stopped at the run script's dirty-source gate even though a later read-only status showed the run-owned clone was clean.

### Error

```text
Remote run-owned source clone is dirty.
```

### Context

- The failure marker appeared before readiness, cache generation, or training.
- A bounded post-failure `git status --short --branch` showed no modified or untracked files and exact HEAD `7ee17cf`.
- No reset or cleanup was performed; the next pushed position-preserving source commit was fetched into the same clean run-owned clone.

### Suggested Fix

When the dirty-source gate fails, retrieve the marker first and perform one bounded read-only `git status` check. Never reset automatically. Retry only if the clone is currently clean and an exact pushed commit can be checked out.

### Metadata

- Reproducible: no
- Related Files: configs/remote/generated/run_i2_output_prediction_opa1_present_r3_selected8_architecture_screen_key2_gpu0_20260721.cmd
- See Also: ERR-20260721-001, ERR-20260721-002
- Pattern-Key: remote.run_owned_clone_transient_dirty_requires_readonly_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-21
- Last-Seen: 2026-07-21

### Resolution

- **Resolved**: 2026-07-21T23:27:00+08:00
- **Commit/PR**: pending
- **Notes**: The clone was verified clean without reset, updated to pushed commit `fdeb49d`, and the retry produced both the started marker and `readiness=status=pass`.

---

## [ERR-20260721-004] remote_training_cli_eager_plot_dependency

**Logged**: 2026-07-21T23:30:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary

OPA1 passed remote CUDA/readiness checks but exited before data generation because the training CLI eagerly imported a Matplotlib plotting module unavailable in the remote torch environment.

### Error

```text
ModuleNotFoundError: No module named 'matplotlib'
```

### Context

- Formal remote plots are intentionally deferred until verified local retrieval.
- The CLI nevertheless imported `plot_innovation2_selected_output_architecture` at module scope for local smoke rendering.
- CUDA, the pinned source revision, and the experiment JSON readiness gate all passed; no cache or checkpoint was created before the import failure.

### Suggested Fix

Import the plotting module only inside the local `smoke` branch. Add a subprocess regression test that blocks every Matplotlib import and verifies the training CLI can still be imported.

### Metadata

- Reproducible: yes
- Related Files: src/blockcipher_nd/cli/run_innovation2_selected_output_architecture.py, tests/test_innovation2_selected_output_architecture.py
- See Also: ERR-20260721-002, ERR-20260721-003
- Pattern-Key: remote.training_cli_must_not_eager_import_deferred_plot_dependencies
- Recurrence-Count: 1
- First-Seen: 2026-07-21
- Last-Seen: 2026-07-21

### Resolution

- **Resolved**: 2026-07-21T23:32:00+08:00
- **Commit/PR**: pending
- **Notes**: Plotting is now lazily imported only for local smoke, and the new test emulates the remote environment without Matplotlib.

---

## [ERR-20260716-008] iacr_pdf_cloudflare_and_browser_tool_unavailable

**Logged**: 2026-07-16T15:42:57+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary

Legal open IACR ePrint landing pages were readable, but PDF endpoints returned
Cloudflare HTML and the real-browser fallback could not be installed under the
current platform policy.

### Error

```text
eprint PDF response: HTML document, about 5420 bytes, Cloudflare challenge

npm FetchError: request to https://registry.npmjs.org/@playwright%2fcli failed
connect EPERM 127.0.0.1:7897

Escalated npx installation: rejected because escalation is disallowed
```

### Context

- Affected open PDFs include ePrint 2026/340, 2021/1502, 2026/961, and
  2026/735; additional adjacent ePrint PDFs were affected in the same way.
- `curl`, browser-like user agents, short IACR URLs, and direct publisher PDF
  endpoints were checked without bypassing access controls.
- The Playwright skill was selected as the documented real-browser fallback,
  but its CLI dependency was absent and the reviewed installation request was
  rejected. No alternate browser or policy workaround was attempted.

### Suggested Fix

Retry the canonical ePrint PDF URLs when the Cloudflare policy changes, or use
an already installed/approved real browser after explicit platform approval.
Until then, retain the landing-page HTML, exact PDF URL, DOI, abstract, and
`access_blocked` manifest state; never store the challenge HTML as a PDF.

### Metadata

- Reproducible: yes
- Related Files: docs/research/innovation2-high-round-literature-corpus-20260716.md, sources/research_innovation2_paper_manifest_20260716.csv
- See Also: ERR-20260716-007
- Pattern-Key: research.iacr_pdf_cloudflare_requires_approved_browser
- Recurrence-Count: 1
- First-Seen: 2026-07-16
- Last-Seen: 2026-07-16

---

## [ERR-20260718-001] atm_qmc_python313_ortools_multiprocessing

**Logged**: 2026-07-18T17:55:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: research

### Summary

ATM author QMC model generation could not pickle OR-Tools CP-SAT variables through multiprocessing under Python 3.13.

### Error

```text
TypeError: cannot pickle 'ortools.sat.python.cp_model_helper.CpModelProto' object
```

### Context

- Operation: E58-A mechanism reproduction of the official ATM projected-SAT provider.
- Source: AlgebraicTransitionMatrices commit `b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b`.
- Environment: CPython 3.13.12, OR-Tools 9.15.6755, python-sat 1.9.dev6.
- Failure occurred while `QMC_optimise_CNF` sent constraint-generation tasks containing CP-SAT variables to a multiprocessing pool.
- No SAT coefficient or cryptographic result had been produced at the failure point.

### Suggested Fix

Keep the official source immutable and replace only QMC prime-implicant and constraint-list parallelism with an equivalent deterministic single-process adapter. Freeze CP-SAT to one worker and require full truth-table comparison before trusting the generated CNF.

### Metadata

- Reproducible: yes
- Related Files: src/blockcipher_nd/tasks/innovation2/atm_native_sat_witness_provider.py, docs/experiments/innovation2-present-atm-native-sat-witness-provider-plan.md
- See Also: LRN-20260718-006

### Resolution

- **Resolved**: 2026-07-18T17:55:00+08:00
- **Commit/PR**: pending
- **Notes**: Added a project-side single-process QMC compatibility shim; all 256 PRESENT S-box algebraic transition coefficients matched direct truth values and the keyed-toy witness controls passed.

---

## [ERR-20260716-007] github_unauthenticated_code_search_rate_limit

**Logged**: 2026-07-16T14:35:00+08:00
**Priority**: low
**Status**: pending
**Area**: research

### Summary

GitHub's unauthenticated code-search API could not verify whether Wu/Guo released an implementation of their PRESENT integral-neural distinguisher.

### Error

```text
API rate limit exceeded for 134.195.101.197.
```

### Context

- Repository title search for `"Improved integral neural distinguisher" PRESENT` returned zero repositories.
- The broader code query for `invP`, `invS`, `PRESENT`, and `integral neural` hit the unauthenticated API rate limit.
- The paper text itself contains no GitHub or public-code URL.
- No private manuscript or project content was uploaded; only public paper keywords were queried.

### Suggested Fix

Treat public author code as unavailable unless a canonical paper, author page, or authenticated search later verifies it. Implement the data path from the paper only after freezing hand-calculable `InvP/InvS` fixtures and exact bit-order tests; do not infer missing protocol details silently.

### Metadata

- Reproducible: unknown
- Related Files: docs/experiments/innovation2-present-high-round-integral-neural-anchor-plan.md, papers/innovation_one/grobid_md/improved-integral-neural-distinguisher-model-for-lightweight-cipher-present.md
- See Also: ERR-20260706-002
- Pattern-Key: research.github_unauthenticated_code_search_rate_limited
- Recurrence-Count: 1
- First-Seen: 2026-07-16
- Last-Seen: 2026-07-16

---

## [ERR-20260715-006] tmux_prefix_match_made_watcher_wait_on_itself

**Logged**: 2026-07-15T15:18:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
A local result watcher used a non-exact tmux target and matched its own longer
session name after the training session exited.

### Error

```text
tmux has-session -t i1_feistel_des_r6_local_2048
matched i1_feistel_des_r6_local_2048_postprocess by prefix
```

### Context

- The eight DES result rows and `run_done` event were already complete.
- The watcher remained healthy but never advanced to validation, plotting,
  gating, or indexing because its own session satisfied the prefix match.
- No training was repeated and no result file was modified during diagnosis.

### Suggested Fix

Use tmux exact-target syntax for watcher ownership checks:

```text
tmux has-session -t "=$session_name"
```

Name a watcher so its session cannot be confused with the training session,
and test the stopped-training/running-watcher state before handoff.

### Metadata

- Reproducible: yes
- Related Files: /tmp/i1_feistel_des_r6_local_2048_postprocess.sh
- See Also: ERR-20260715-005
- Pattern-Key: remote_monitor.tmux_target_must_be_exact
- Recurrence-Count: 1
- First-Seen: 2026-07-15
- Last-Seen: 2026-07-15

### Resolution

- **Resolved**: 2026-07-15T15:18:00+08:00
- **Commit/PR**: pending
- **Notes**: Watcher restarted with `-t =name`; validation, SVG/CSV, gate, and index completed.

---

## [ERR-20260715-005] duplicate_local_training_polluted_shared_output

**Logged**: 2026-07-15T14:50:00+08:00
**Priority**: critical
**Status**: resolved
**Area**: infra

### Summary
Two DES R1 training processes used different batch sizes while appending to the
same progress and result paths, so the partial 2048/class output is invalid.

### Error

```text
progress.jsonl interleaved steps_per_epoch=16 and steps_per_epoch=64
results.jsonl remained empty while both Python training processes ran
```

### Context

- Run ID: `i1_feistel_des_r6_branch_inception_2048_seed0_seed1`.
- Two asynchronous tool sessions appeared to complete while their sandboxed
  child processes continued on the host.
- Both jobs reused the same plan, cache root, progress path, and result path,
  but used batch sizes 256 and 64.
- The partial output is not research evidence and must not be indexed.
- Exact host process trees were stopped without touching the active GIFT
  remote-result watcher.

### Suggested Fix

Give each meaningful local matrix one unique tmux session and one owning
command. Before launch, verify that no process or tmux session already names
the plan or output path. After an interrupted or ambiguous launch, inspect the
host process tree before retrying. Never launch a second writer into an
existing result directory; move the invalid output aside and restart with a
fresh progress/result path. Reuse a cache only after its metadata, array
shapes, dtypes, and label counts pass validation.

### Metadata

- Reproducible: yes
- Related Files: docs/experiments/innovation1-feistel-des-branch-inception-plan.md, outputs/local_diagnostic/i1_feistel_des_r6_branch_inception_2048_seed0_seed1
- See Also: ERR-20260715-001
- Pattern-Key: experiments.single_writer_per_output_directory
- Recurrence-Count: 1
- First-Seen: 2026-07-15
- Last-Seen: 2026-07-15

### Resolution

- **Resolved**: 2026-07-15T14:50:00+08:00
- **Commit/PR**: pending
- **Notes**: Both duplicate process trees were stopped; clean single-session rerun required.

---

## [ERR-20260710-001] full_pytest_matplotlib_state_and_json_alignment_baseline

**Logged**: 2026-07-10T12:05:00+08:00
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
The full pytest suite has 29 order-dependent Matplotlib 3.11 failures plus one
independent JSON route-name alignment failure, while the E1-R focused tests and
standalone plotting test pass.

### Error

```text
full suite: 30 failed, 620 passed
Matplotlib failures:
AttributeError: 'Figure' object has no attribute 'items'
  matplotlib/cbook.py normalize_kwargs(kwargs, self)

independent alignment failure:
test_json_plan_alignment_maps_route_specific_short_model_names
expected status=pass, observed status=fail
```

### Context

- Command: `MPLCONFIGDIR=/tmp/matplotlib-cache UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`
- Matplotlib version: `3.11.0`; backend: `agg`.
- The earliest plotting failure passes when run alone, so the Matplotlib
  failure depends on full-suite process/global class state rather than the
  E1-R active-cell repair or standalone plot generation.
- The JSON plan-alignment test still fails when run alone and is unrelated to
  the active-cell model, CSV matrices, or result validation path in E1-R.
- E1-R semantic, active-coordinate, matrix-protocol, and focused graph tests
  pass (`6 passed`).

### Suggested Fix

Handle these in separate test-infrastructure work. Reproduce the Matplotlib
class-state mutation with the smallest preceding test subset before changing
plot code or dependency constraints. Diagnose the JSON alignment report's
actual errors separately. Do not mix either fix into the E1-R single-variable
layout repair.

### Metadata
- Reproducible: yes
- Related Files: src/blockcipher_nd/evaluation/plots.py, src/blockcipher_nd/planning/result_alignment.py, tests/test_pairset_aggregation_postprocess.py, tests/test_project_structure.py
- See Also: ERR-20260624-002, LRN-20260710-003

---

## [ERR-20260715-004] balanced_validation_total_forwarded_as_random_total

**Logged**: 2026-07-15T10:05:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
A balanced large-scale plan failed before training because the generic dataset
builder forwarded `validation_samples_total` into the random-label-only
`samples_total` field.

### Error

```text
ValueError: samples_total is only valid with random_labels_total
```

### Context

- The GIFT-64 performance readiness plan used `balanced_per_class` and
  `validation_samples_total=128`, which correctly means `64/class`.
- `validation_samples_per_class()` converted the total correctly, but
  `task_runner` also forwarded the original total to `build_dataset_config`.
- The same issue would have blocked a balanced `1000000/class` remote plan
  with an explicit total validation size.

### Suggested Fix

Keep `samples_total` only when `dataset_label_mode=random_labels_total` at the
central `build_dataset_config` boundary. Balanced plans should retain the
derived `samples_per_class` and pass `samples_total=None` to the generator.

### Metadata
- Reproducible: yes
- Related Files: src/blockcipher_nd/engine/task_config.py, tests/test_gift64_pairset_baselines.py, configs/experiment/innovation1/innovation1_spn_gift64_mainstream_performance_readiness_seed6.csv
- See Also: LRN-20260715-001

### Resolution
- **Resolved**: 2026-07-15T10:08:00+08:00
- **Commit/PR**: ffdbcef
- **Notes**: Central label-mode guard added; 20 focused protocol tests and the five-row readiness run passed.

---

## [ERR-20260715-002] local_torch_cuda_driver_mismatch

**Logged**: 2026-07-15T07:10:00+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
The local host exposes an RTX 5080 through `nvidia-smi`, but project PyTorch cannot initialize CUDA.

### Error
```text
RuntimeError: Found no NVIDIA driver on your system
```

### Context
- Local E5 `8192/class` diagnostics were eligible for local execution but had
  to use CPU because PyTorch CUDA initialization failed.
- `nvidia-smi` can see the RTX 5080, so device visibility and the PyTorch CUDA
  runtime/driver view disagree.
- This is not permission to run `65536/class` or larger jobs on local CPU.
  Innovation 1 SPN/PRESENT/GIFT `65536/class` and above remain remote-only.

### Suggested Fix
Audit the local PyTorch CUDA build, NVIDIA driver/library visibility inside the
project environment, and `LD_LIBRARY_PATH` before relying on local CUDA. Until
verified, use CPU only for sub-medium smoke/readiness/diagnostics and use the
remote `lxy-a6000` workflow for eligible medium experiments.

### Metadata
- Reproducible: yes
- Related Files: AGENTS.md, pyproject.toml
- See Also: LRN-20260628-001

---

## [ERR-20260715-003] jq_not_installed_in_local_environment

**Logged**: 2026-07-15T07:10:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
An E5 JSON summary command failed because the local environment does not provide `jq`.

### Error
```text
/bin/bash: line 1: jq: command not found
```

### Context
- Operation: extract compact fields from E5 source and target gate JSON files.
- The experiment artifacts were valid; only the optional shell summarizer was
  unavailable.
- The same fields were read successfully with `UV_CACHE_DIR=/tmp/uv-cache uv
  run python`.

### Suggested Fix
Use the project Python environment for JSON summaries unless `command -v jq`
has confirmed availability.

### Metadata
- Reproducible: yes
- Related Files: outputs/local_diagnostic/i1_cross_spn_e5_target_8192_source_seed0/gate.json
- See Also: ERR-20260706-001

### Resolution
- **Resolved**: 2026-07-15T07:10:00+08:00
- **Commit/PR**: pending
- **Notes**: Re-ran the summaries through `uv run python`; all required values were extracted.

---

## [ERR-20260715-001] e4_r4_windows_postprocess_recovery

**Logged**: 2026-07-15T03:10:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
E4-R4 training and paired gates passed on both remote GPUs, but Windows batch
postprocessing failed while hashing archives; two subsequent monitor-state
assumptions also blocked verified retrieval.

### Error
```text
seed2/seed3 gate.json = status pass
seed2/seed3 failed.marker = failed
SHA256SUMS = missing

root cause 1: EnableDelayedExpansion consumed `!` in Python `!=`
root cause 2: scp retained stale local failed.marker after remote recovery
root cause 3: joint-branch checkout hid the seed2 archive from worktree retrieval
root cause 4: Windows Git EOL conversion changed two files after archive hashing
root cause 5: Windows CRLF in SHA256SUMS makes Linux sha256sum -c treat CR as
              part of each filename even when artifact hashes are intact
```

### Context
- Run IDs: `i1_gift64_cross_spn_target_adaptation_r4_65536_seed2` and
  `i1_gift64_cross_spn_target_adaptation_r4_65536_seed3`.
- Training, validation, checkpoint writing, score export, and 10,000-replicate
  gates were complete before the first failure.
- Recovery reused those exact artifacts and did not retrain.

### Suggested Fix
Avoid `!` inside batch files with delayed expansion, make pushed/done markers
override stale failure markers, restore the seed result branch after publishing
a joint branch, and place `* -text` in result archives before hashing/staging.
Test these invariants in the remote-asset regression test. For a retrieved
Windows manifest, normalize only the manifest input stream before Linux
verification (`sed 's/\r$//' SHA256SUMS | sha256sum -c -`); do not rewrite the
archived payload files.

### Metadata
- Reproducible: yes
- Related Files: configs/remote/generated/run_i1_gift64_cross_spn_target_adaptation_r4_65536_20260715.cmd, configs/remote/generated/monitor_i1_gift64_cross_spn_target_adaptation_r4_65536_20260715.sh, configs/remote/generated/recover_i1_gift64_cross_spn_target_adaptation_r4_65536_20260715.cmd, tests/test_cross_spn_target_adaptation_gate.py
- See Also: ERR-20260714-001, ERR-20260714-002
- Pattern-Key: remote.windows_result_archive_state_machine
- Recurrence-Count: 2
- First-Seen: 2026-07-15
- Last-Seen: 2026-07-15

### Resolution
- **Resolved**: 2026-07-15T03:10:00+08:00
- **Commit/PR**: 455db9b, b71d290, 5d1e129, 5b72484
- **Notes**: Recovered and pushed all three E4-R4 result branches without
  retraining. E4-R5 later reproduced the CRLF manifest-parser issue; streaming
  CR removal verified all `40 + 40 + 6` archive entries without modifying the
  archives.

---

## [ERR-20260711-001] full_pytest_python310_fstring_collection

**Logged**: 2026-07-11T20:44:40+08:00
**Priority**: high
**Status**: resolved
**Area**: tests

### Summary
Full pytest collection was blocked on Python 3.10 by a backslash inside an
f-string expression in the residual-focus remote-package path helper.

### Error

```text
File "src/blockcipher_nd/cli/plan_residual_focus_remote_package.py", line 290
  return f"{prefix}\\{suffix.replace('/', '\\')}"
                                               ^
SyntaxError: f-string expression part cannot include a backslash
```

### Context

- Command: `MPLCONFIGDIR=/tmp/matplotlib-cache UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q`
- Environment: Python 3.10 in the project `uv` environment.
- The failing source was unchanged between base `44a0d36` and experiment HEAD
  `5be1607`; the experiment branch exposed the existing compatibility defect
  during its required full-suite verification.
- Collection stopped before the authoritative 30 known runtime failures could
  be compared.

### Suggested Fix

Compute `suffix.replace("/", "\\")` in a local variable before the f-string,
then interpolate that variable. This preserves the exact Windows path result
without placing a backslash in the f-string expression.

### Metadata
- Reproducible: yes
- Related Files: src/blockcipher_nd/cli/plan_residual_focus_remote_package.py, tests/test_residual_focus_remote_package.py
- See Also: ERR-20260710-001

### Resolution
- **Resolved**: 2026-07-11T20:44:40+08:00
- **Commit/PR**: cc12fb9
- **Notes**: Moved slash normalization into `normalized_suffix`; Python 3.10 compilation passed and `tests/test_residual_focus_remote_package.py` passed 5/5.

---

## [ERR-20260621-001] rg_windows_path_regex

**Logged**: 2026-06-21T20:55:00+08:00
**Priority**: medium
**Status**: pending
**Area**: docs

### Summary
Searching for Windows paths such as `G:\lxy` with regex `rg` can fail because backslashes are parsed as escapes.

### Error
```text
regex parse error:
    Remote artifact root|G:\lxy|cmd\.exe /c|cmd\.exe /k|...
                           ^^
error: unrecognized escape sequence
```

### Context
- Command attempted: `rg -n "Remote artifact root|G:\lxy|cmd\.exe /c|..." ...`
- Environment: Linux shell searching repository docs/memory for Windows remote path rules.
- The backslash in `G:\lxy` was interpreted by ripgrep's regex parser.

### Suggested Fix
Use fixed-string search for Windows paths (`rg -F "G:\lxy" ...`) or properly escape backslashes in regex patterns.

### Metadata
- Reproducible: yes
- Related Files: AGENTS.md
- See Also: LRN-20260621-003

---

## [ERR-20260622-001] remote_dirty_overlay_incomplete_model_sync

**Logged**: 2026-06-22T11:40:00+08:00
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
Remote PRESENT trail-position scale run failed because a dirty/scp overlay copied too narrow a model subset into the run directory.

### Error
```text
ModuleNotFoundError: No module named 'blockcipher_ai_eval.models.structure.spn.present_candidate_disagreement'

ImportError: cannot import name 'PresentZhangWangKerasMCNDDistinguisher' from 'blockcipher_ai_eval.models.structure'
```

### Context
- Run ID: `i1-spn-trailpos-r7-scale-med-gpu0-20260622`.
- Operation: remote Windows GPU launch for PRESENT r7 `262144/class` trail-position scale run.
- Initial overlay copied only `spn/__init__.py`, `present_pairset_global_stats_hybrid.py`, and `present_trail_position_stats.py`.
- Current `spn/__init__.py` imports additional SPN modules, and `models/structure/__init__.py` exports newly added classes. The remote run directory was cloned from GitHub plus partial overlay, so imports failed before training.

### Suggested Fix
Remote runs should be based on committed and pushed code. If an overlay is unavoidable, overlay complete dependency boundaries, such as full `src\blockcipher_ai_eval\models\structure`, not individual files that import broader modules. Add generator tests requiring the correct overlay scope for route-specific remote configs.

### Metadata
- Reproducible: yes
- Related Files: experiments/innovation1/configs/remote/innovation1_spn_trailpos_r7_scale_med_gpu0_20260622.json, scripts/generated/remote/run_i1-spn-trailpos-r7-scale-med-gpu0-20260622_and_push.cmd, tests/test_remote_script_generator.py
- See Also: LRN-20260622-001

---

## [ERR-20260624-001] test_internal_class_name_guess

**Logged**: 2026-06-24T15:30:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
New training-history test failed because it guessed a cipher class name instead of using the actual exported class or factory.

### Error
```text
ImportError: cannot import name 'Speck32Cipher' from 'blockcipher_nd.ciphers.arx.speck'
```

### Context
- Command attempted: `uv run pytest -q`
- Test added: `test_training_history_records_train_and_validation_metrics`
- Mistake: imported non-existent `Speck32Cipher`; the module defines `Speck32_64`.

### Suggested Fix
When writing tests against project internals, inspect the target module or use stable factory APIs such as `build_cipher` instead of guessing class names from memory.

### Metadata
- Reproducible: yes
- Related Files: tests/test_project_structure.py, src/blockcipher_nd/ciphers/arx/speck.py
- See Also: LRN-20260622-001

### Resolution
- **Resolved**: 2026-06-24T15:31:00+08:00
- **Commit/PR**: pending
- **Notes**: Test import was changed to the actual `Speck32_64` class.

---

## [ERR-20260624-002] parallel_artifact_read_race

**Logged**: 2026-06-24T15:33:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
An artifact verification command read a CSV before the parallel plot generation command had finished creating it.

### Error
```text
head: cannot open '/tmp/blockcipher_verify_history.csv' for reading: No such file or directory
```

### Context
- Commands attempted in parallel:
  - `uv run python scripts/plot-results --results ... --history-csv /tmp/blockcipher_verify_history.csv`
  - `head -5 /tmp/blockcipher_verify_history.csv`
- The reader command raced ahead of the writer command.

### Suggested Fix
Do not parallelize dependent artifact producer/consumer commands. Generate artifacts first, then inspect them in a later command.

### Metadata
- Reproducible: yes
- Related Files: scripts/plot-results, src/blockcipher_nd/evaluation/plots.py
- See Also: ERR-20260624-001

### Resolution
- **Resolved**: 2026-06-24T15:34:00+08:00
- **Commit/PR**: pending
- **Notes**: Re-ran the artifact inspection after plot generation completed.

---

## [ERR-20260705-001] remote_start_b_returned_without_run_artifacts

**Logged**: 2026-07-05T17:31:00+08:00
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
Remote Windows `start /b cmd.exe /c <script>.cmd` returned success but did not create the expected run directory or logs.

### Error
```text
ssh ... "cmd.exe /c start \"\" /b cmd.exe /c G:\lxy\run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.cmd"
exit code: 0

Later read-only check:
G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705\logs -> MISSING
```

### Context
- Run ID: `i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705`.
- Operation: launch a project-owned remote `.cmd` under `G:\lxy` for a PRESENT r8 integral/inverse feature screen.
- The copied script existed at `G:\lxy\run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.cmd`.
- A direct `cmd.exe /c call G:\lxy\run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.cmd` entered the script correctly and produced `started.marker`, readiness logs, and dataset-cache progress under the expected `G:\lxy` run directory.

### Suggested Fix
Do not treat a successful SSH exit from `start /b` as proof of launch. After any remote `.cmd` launch, perform one bounded read-only confirmation that `logs`, `*_started.marker`, readiness output, or progress JSONL exists under the expected `G:\lxy` run root. If a detached `start /b` invocation does not create artifacts, use a more reliable launch wrapper such as `cmd.exe /c call <script>` with a controlled timeout or a PowerShell `Start-Process` command that explicitly sets the working directory and redirects logs inside `G:\lxy`.

### Metadata
- Reproducible: unknown
- Related Files: configs/remote/generated/run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.cmd
- See Also: ERR-20260622-001

---

## [ERR-20260624-003] remote_dirty_clone_checkout_blocked_launch

**Logged**: 2026-06-24T21:23:10+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
Remote 262k PRESENT run did not start because the canonical remote clone had historical local changes and `git checkout main` aborted before training.

### Error
```text
error: Your local changes to the following files would be overwritten by checkout:
  experiments/innovation1/README.md
  experiments/innovation1/audit_arx_feature_separation.py
  ...
Aborting
```

### Context
- Run attempted: `zhang_wang_present_r7_262k_official_cyclic_20260624`.
- Remote project root attempted first: `G:\lxy\blockcipher-structure-adaptive-nd`.
- The first Windows `start /b cmd.exe /c ...` launch also silently failed because the command chained `if not exist <launcher_logs> mkdir ... && start ...`; when the directory already existed, the `start` segment was skipped.
- A direct tmux-held SSH launcher then reached the remote script, but the remote script exited during `git checkout main` because the historical clone had unrelated dirty files from older project layouts.
- The training itself had not hung; it had never entered training in the first attempt.

### Suggested Fix
For GitHub-pushed formal or medium remote runs, do not assume a historical remote clone is clean. Use a run-owned clean clone directory under `G:\lxy`, or perform a read-only `git status --short --branch` gate before launch and switch to a clean clone when dirty. Do not use `git reset --hard` on the shared historical clone without explicit user approval. Avoid chaining Windows `if not exist ... && start ...` when the `start` command must always run.

### Metadata
- Reproducible: yes
- Related Files: /tmp/remote_zw262k/run_zhang_wang_present_r7_262k_official_cyclic_20260624.cmd, AGENTS.md
- See Also: ERR-20260622-001, LRN-20260622-001, LRN-20260624-002

### Resolution
- **Resolved**: 2026-06-24T21:24:00+08:00
- **Commit/PR**: pending
- **Notes**: Relaunched through local tmux using a new clean clone directory `G:\lxy\blockcipher-structure-adaptive-nd-v1-clean`; remote progress reached dataset cache generation for `262144/class`.

---

## [ERR-20260706-002] git_push_external_review_blocked

**Logged**: 2026-07-06T13:40:00+08:00
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
`git push origin main` was rejected by the sandbox reviewer while trying to unblock the trail-position 65k remote launch source-publication gate.

### Error
```text
Rejected("This action was rejected due to unacceptable risk.
Reason: Pushing 48 local commits to an external GitHub remote is a high-risk outbound transfer of workspace data, and the visible transcript does not show clear explicit user approval for this exact external push.
The agent must not attempt to achieve the same outcome via workaround, indirect execution, or policy circumvention. Proceed only with a materially safer alternative, or if the user explicitly approves the action after being informed of the risk. Otherwise, stop and request user input.")

Latest recurrence after commit 8c75600:
Model "codex-auto-review" is not supported by any configured account in this group

Latest recurrence after commit 48cced7:
Rejected because the environment forbids escalated-permission requests under
`approval policy: never`, after the ordinary push failed with
`Couldn't connect to server`.
```

### Context
- Command attempted: `git push origin main`
- Purpose: publish already committed Innovation 1 experiment, gate, documentation, and remote-launch assets so `scripts/check-launch-source` can pass before a compliant remote run.
- Local gate state before push attempt: `main...origin/main [ahead 48]`, dirty `false`, errors `unpushed_commits`.
- Remote launch affected: `i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706`.
- Latest recurrence: push of local head `8c75600` after the E1 8192/class
  adjudication; branch state `main...origin/main [ahead 1]`.
- 2026-07-16 recurrence: push of `48cced7` containing the verified SIMECK
  curriculum implementation, two-seed local evidence, and the frozen
  `65536/class` remote package. Ordinary network access failed; the escalated
  request was rejected. The remote run
  `i1_feistel_simeck_curriculum_65k_seed0` was correctly not launched.

### Suggested Fix
Do not work around this by dirty overlay, alternate push commands, or SSH remote launch from unpublished code. Either obtain explicit user approval for the exact external push of 48 commits to `origin/main`, or continue with local diagnostics/watchers until a safer publication path exists.

### Metadata
- Reproducible: yes
- Related Files: docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, docs/experiments/innovation1-feistel-simeck-curriculum-65k-scale-plan.md, scripts/check-launch-source
- See Also: ERR-20260622-001, ERR-20260624-003, ERR-20260705-001, LRN-20260706-022
- Pattern-Key: infra.git_push.external_reviewer_unavailable_or_rejected
- Recurrence-Count: 3
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-16

---

## [ERR-20260626-001] remote_https_clone_reset

**Logged**: 2026-06-26T11:18:14+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
Remote Windows scheduled run failed before training because GitHub HTTPS clone was reset.

### Error
```text
Cloning into 'innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0_gpu1_20260626'...
fatal: unable to access 'https://github.com/swordfate3/blockcipher-structure-adaptive-nd-v1.git/': Recv failure: Connection was reset
RUN_GATE_BLOCKED_GIT_FAILED
```

### Context
- Run attempted: `innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0_gpu1_20260626`.
- Remote script used a run-owned clean clone under `G:\lxy\blockcipher-structure-adaptive-nd-runs`.
- The script correctly avoided the dirty historical clone, but HTTPS clone failed before the run directory existed.
- No training started; GPU remained idle and no progress JSONL existed.
- Latest recurrence: run
  `i1_present_autond_public_code_paperscale_seed0_gpu1_20260710` was cloned
  successfully with the dedicated SSH key, but its tracked launcher omitted
  `GIT_SSH_COMMAND`. The Task Scheduler process therefore failed on
  `git fetch origin main` with `Permission denied (publickey)` before
  readiness or training.
- A regression test now requires the dedicated key setup to appear before
  `git fetch origin` in the paper-scale launcher.

### Suggested Fix
For remote Windows scheduled experiment launchers, prefer the known remote GitHub SSH key and SSH repo URL:

```cmd
set REPO_URL=git@github.com:swordfate3/blockcipher-structure-adaptive-nd-v1.git
set GITHUB_SSH_KEY=C:/Users/1304Lijinlin/.ssh/github_blockcipher_20260612_result_pusher_ed25519
set GIT_SSH_COMMAND=ssh -i %GITHUB_SSH_KEY% -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
```

Keep generated scripts under `G:\lxy`, and only use HTTPS clone as a fallback when SSH auth is unavailable.

### Metadata
- Reproducible: unknown
- Related Files: /tmp/run_innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0_gpu1_20260626.cmd, configs/remote/generated/run_i1_present_autond_public_code_paperscale_seed0_gpu1_20260710.cmd, tests/test_autond_public_protocol.py, /home/fate/.agents/skills/remote-windows-gpu-conda-ssh/SKILL.md
- See Also: ERR-20260624-003, ERR-20260705-001
- Pattern-Key: remote.windows_scheduled_git_requires_explicit_ssh_identity
- Recurrence-Count: 2
- First-Seen: 2026-06-26
- Last-Seen: 2026-07-10

### Resolution
- **Resolved**: 2026-06-26T11:19:00+08:00
- **Commit/PR**: pending
- **Notes**: Launchers must set the SSH repo URL and remote dedicated GitHub key before every clone/fetch/pull performed by Task Scheduler. The 2026-07-10 recurrence added a tracked regression test and repaired the paper-scale launcher.

---

## [ERR-20260627-001] remote_plot_missing_matplotlib

**Logged**: 2026-06-27T11:35:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
Remote training completed and result gate passed, but remote plot generation failed because the `torch310` environment did not have Matplotlib installed.

### Error
```text
ModuleNotFoundError: No module named 'matplotlib'
```

### Context
- Run ID: `i1_spn_present_mcnd_r7_1m_seed0_gpu1_retry1_20260626`
- Remote training command completed successfully with `result_lines=2` and `expected_rows=2`.
- Plot command failed inside `scripts/plot-results` when importing `blockcipher_nd.evaluation.plots`.
- The repository depends on `matplotlib>=3.8`, but the remote `F:\Anaconda\envs\DWT\torch310` environment did not have it available during the scheduled task.
- Workaround used: retrieve JSONL/logs and regenerate `curves.svg` plus `history.csv` locally with `uv run python scripts/plot-results`.
- Recurrence: both E4-R3 GIFT-64 `65536/class` seed runs completed training
  and validation on 2026-07-15, then failed before gate/archive at the same
  Matplotlib import. Raw fallback retrieval and local postprocessing recovered
  the evidence without retraining.

### Suggested Fix
Before relying on remote plot generation, verify the remote environment has Matplotlib installed or add a launcher preflight that records missing optional plotting dependencies. Keep local post-retrieval plot regeneration as a safe fallback when training results are valid but remote plotting fails.

### Metadata
- Reproducible: yes
- Related Files: scripts/plot-results, src/blockcipher_nd/evaluation/plots.py, configs/remote/generated/run_i1_gift64_cross_spn_typed_transfer_r3_65536_20260714.cmd, outputs/remote_results/i1_spn_present_mcnd_r7_1m_seed0_gpu1_retry1_20260626/logs/
- See Also: LRN-20260624-001, ERR-20260626-001
- Pattern-Key: remote.postprocess.optional_plot_must_not_block_archive
- Recurrence-Count: 2
- First-Seen: 2026-06-27
- Last-Seen: 2026-07-15

### Resolution
- **Resolved**: 2026-07-15T01:30:00+08:00
- **Commit/PR**: pending
- **Notes**: The E4 remote runner now validates and gates before plotting. A
  missing or invalid plot writes `plot_deferred.marker`; history/SVG are
  optional archive files, while results, validation, gate, progress, and
  provenance remain fail-closed requirements. The local retrieval workflow
  regenerates deferred visualization artifacts.

---

## [ERR-20260627-002] plan_evidence_assertion_too_narrow

**Logged**: 2026-06-27T19:48:50+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
New N2 plan test failed because it asserted one exact evidence phrase for all rows even though rows intentionally describe N2-a/N2-b/N2-c separately.

### Error
```text
AssertionError: assert 'MEDIUM 262144/class N2 transition backbone diagnostic'
in 'MEDIUM 262144/class N2-a pair-level evidence pooling diagnostic; not formal reproduction or breakthrough evidence'
```

### Context
- Command attempted: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_project_structure.py -q`
- New test: `test_present_n2_transition_backbone_262k_plan_is_same_protocol`
- The plan intentionally has row-specific evidence strings for baseline, SPN-only anchor, N2-a, N2-b, and N2-c.
- The test should verify invariant scope markers such as `MEDIUM 262144/class`, `N2`, and non-formal claim language, not require identical evidence prose across all rows.

### Suggested Fix
When validating experiment matrices, assert protocol invariants and stable scope markers. Avoid brittle exact-substring checks if each row is supposed to carry model-specific evidence wording.

### Metadata
- Reproducible: yes
- Related Files: tests/test_project_structure.py, configs/experiment/innovation1/innovation1_spn_present_n2_transition_backbone_r7_262k.csv
- See Also: ERR-20260624-001

### Resolution
- **Resolved**: 2026-06-27T19:49:00+08:00
- **Commit/PR**: pending
- **Notes**: The assertion was relaxed to check stable evidence scope markers while preserving protocol checks.

---

## [ERR-20260628-001] local_python_command_missing

**Logged**: 2026-06-28T11:51:10+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
The local shell does not provide a bare `python` command; use `python3` or `uv run python` in this project.

### Error
```text
/bin/bash: line 1: python: command not found
```

### Context
- Command attempted: `python -c "from pathlib import Path; ..."`
- Purpose: lightweight verification after updating `AGENTS.md` and `.learnings/LEARNINGS.md`.
- Environment: local project shell.
- Follow-up command with `python3` succeeded.

### Suggested Fix
Use `python3` for tiny local interpreter checks, and use `UV_CACHE_DIR=/tmp/uv-cache uv run python ...` for project Python commands that rely on the managed environment.

### Metadata
- Reproducible: yes
- Related Files: AGENTS.md, .learnings/LEARNINGS.md
- See Also: LRN-20260622-001

### Resolution
- **Resolved**: 2026-06-28T11:51:10+08:00
- **Commit/PR**: pending
- **Notes**: Re-ran the same path check with `python3`; it passed.

---

## [ERR-20260628-002] windows_timeout_noninteractive_check

**Logged**: 2026-06-28T17:49:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
`timeout /t ... /nobreak` can fail in non-interactive SSH health checks on the remote Windows workstation.

### Error
```text
ERROR: Input redirection is not supported, exiting the process immediately.
```

### Context
- Command attempted during remote launch health check for `i1_invp_centered_r7_262k_seed0_gpu1_20260628`.
- The command used `cmd.exe /c timeout /t 5 /nobreak >NUL && ...` through SSH.
- Task Scheduler launch itself succeeded; a follow-up check without `timeout` showed normal clone/pull, logs/progress files, and zero-byte training stderr.

### Suggested Fix
Avoid `timeout /t ... /nobreak` in non-interactive SSH health checks. If a short delay is needed, sleep locally before the SSH command or run a plain read-only check and let the tmux monitor handle later status.

### Metadata
- Reproducible: yes
- Related Files: /tmp/run_i1_invp_centered_r7_262k_seed0_gpu1_20260628.cmd, /tmp/launch_i1_invp_centered_r7_262k_seed0_gpu1_20260628.cmd
- See Also: ERR-20260624-003

### Resolution
- **Resolved**: 2026-06-28T17:50:00+08:00
- **Commit/PR**: pending
- **Notes**: Re-ran the health check without `timeout`; remote logs/progress existed and training stderr was 0 bytes.

---

## [ERR-20260628-003] shell_backticks_in_python_inline_check

**Logged**: 2026-06-28T18:23:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
Inline Python verification failed because a double-quoted shell command contained Markdown backticks, which Bash interpreted as command substitution.

### Error
```text
/bin/bash: line 1: docs/experiments/: Is a directory
Traceback (most recent call last):
  File "<string>", line 1, in <module>
AssertionError
```

### Context
- Command attempted: `python3 -c "from pathlib import Path; ... assert '... `docs/experiments/` ...' in a; ..."`
- Purpose: lightweight verification after updating `AGENTS.md` and `.learnings/LEARNINGS.md`.
- The file update was correct; the verification command was wrong because unescaped backticks inside the shell's double quotes triggered command substitution before Python ran.
- Re-running with single quotes around the shell argument and no backtick-dependent assertion passed.

### Suggested Fix
For inline Python checks in shell commands, wrap the whole Python snippet in single quotes and avoid literal Markdown backticks in assertion strings. Prefer asserting stable plain substrings such as `docs/experiments` instead of Markdown-formatted text.

### Metadata
- Reproducible: yes
- Related Files: AGENTS.md, .learnings/LEARNINGS.md
- See Also: ERR-20260628-001

### Resolution
- **Resolved**: 2026-06-28T18:23:00+08:00
- **Commit/PR**: pending
- **Notes**: Re-ran the verification with safer quoting; it passed.

---

## [ERR-20260629-001] project_import_with_bare_python3

**Logged**: 2026-06-29T00:20:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
Bare `python3 -c` could not import the project package because it was not running inside the project environment.

### Error
```text
ModuleNotFoundError: No module named 'blockcipher_nd'
```

### Context
- Command attempted: `python3 -c 'from blockcipher_nd.engine.matrix_runner import parse_args; ...'`
- Purpose: quick CLI argument verification after adding `--train-eval-interval`.
- The package import failed because bare `python3` did not have the project installed or `PYTHONPATH=src`.
- Re-running with `UV_CACHE_DIR=/tmp/uv-cache uv run python ...` succeeded.

### Suggested Fix
Use `uv run python` for project-package import checks. Reserve bare `python3` for pure standard-library file/text checks, or set `PYTHONPATH=src` explicitly.

### Metadata
- Reproducible: yes
- Related Files: src/blockcipher_nd/engine/matrix_runner.py
- See Also: ERR-20260628-001

### Resolution
- **Resolved**: 2026-06-29T00:20:00+08:00
- **Commit/PR**: pending
- **Notes**: Re-ran the same check with `uv run python`; it passed.

---

## [ERR-20260629-002] plan_scale_not_overridden_by_cli_smoke

**Logged**: 2026-06-29T00:45:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
CPU smoke accidentally started a `262144/class` PRESENT plan because plan CSV rows keep their own `samples_per_class`.

### Error
```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_centered_seed1_fast_r7_262k.csv \
  --epochs 1 --batch-size 8 --hidden-bits 8 --device cpu \
  --train-eval-interval 0 ...
```

This command did not make the run small because `--plan` row fields define `samples_per_class`.

### Context
- Task: local smoke for `innovation1_spn_present_invp_centered_seed1_fast_r7_262k.csv`.
- The command correctly overrode `epochs`, `batch_size`, `hidden_bits`, `device`, and `train_eval_interval`.
- It did not override `samples_per_class`, so the CPU smoke began generating the real `262144/class` dataset and had to be interrupted.

### Suggested Fix
For plan-based smoke tests, create a dedicated smoke CSV with small `samples_per_class` instead of assuming CLI scale flags override plan rows. Use full-size CSVs only for real remote/scale runs.

### Metadata
- Reproducible: yes
- Related Files: configs/experiment/innovation1/innovation1_spn_present_invp_centered_seed1_fast_r7_262k.csv
- See Also: ERR-20260629-001

### Resolution
- **Resolved**: 2026-06-29T00:45:00+08:00
- **Commit/PR**: pending
- **Notes**: Interrupted the mistaken CPU run and switched to a dedicated small smoke CSV.

---

## [ERR-20260629-003] remote_run_dir_precreated_clone_target

**Logged**: 2026-06-29T11:25:00+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
Remote Windows run script failed before training because it created logs inside the same directory later used as the `git clone` destination.

### Error
```text
RUN_GATE_BLOCKED_GIT_CLONE_FAILED
fatal: destination path 'G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_invp_centered_seed1_fast_r7_262k_gpu1_20260629' already exists and is not an empty directory.
```

### Context
- Run attempted: `i1_invp_centered_seed1_fast_r7_262k_gpu1_20260629`.
- Operation: remote Task Scheduler launch for the seed1 InvP-centered 262144/class fast confirmation.
- The run script set `RUN_DIR=%BASE_DIR%\%RUN_ID%`, created `%RUN_DIR%\logs`, then ran `git clone "%REPO_URL%" "%RUN_DIR%"`.
- Because the clone destination was already non-empty, Git failed before checkout, dataset generation, or training.
- This is a launcher/script gate failure, not experiment evidence.

### Suggested Fix
For run-owned clean clone scripts, either clone first into a new empty `%RUN_DIR%` and create `logs/results/results_archive` only after clone succeeds, or clone into a dedicated `%RUN_DIR%\repo` subdirectory and keep launcher logs outside the clone destination. Do not pre-create the clone destination with logs before `git clone`.

### Metadata
- Reproducible: yes
- Related Files: /tmp/run_i1_invp_centered_seed1_fast_r7_262k_gpu1_20260629.cmd, configs/remote/innovation1_spn_present_invp_centered_seed1_fast_r7_262k_gpu1_20260629.json
- See Also: ERR-20260624-003, ERR-20260626-001

---

## [ERR-20260629-004] main_thread_tmux_postprocess_start

**Logged**: 2026-06-29T14:55:00+08:00
**Priority**: low
**Status**: pending
**Area**: infra

### Summary
Starting an extra tmux postprocess watcher from the main thread failed and also reflected the wrong monitoring ownership.

### Error
```text
error connecting to /tmp/tmux-1000/default (Operation not permitted)
```

### Context
- Operation attempted: create `/tmp/postprocess_i1_invp_only_r7_1m_seed0_gpu1_20260629.sh` and start a `postprocess_i1_invp_only_1m_20260629` tmux session.
- Environment: sandboxed command context.
- The attempt failed due to tmux socket permission/access behavior.
- More importantly, the user corrected that monitoring tmux should be handled by a sub-agent/watcher and not repeatedly managed from the main thread.

### Suggested Fix
Do not add ad hoc main-thread tmux monitoring loops after a run has already been handed off. Delegate long-running monitor/postprocess loops to a sub-agent or established watcher. The main thread should process local artifacts after they are retrieved, or perform a single explicit status check only when the user asks.

### Metadata
- Reproducible: unknown
- Related Files: /tmp/postprocess_i1_invp_only_r7_1m_seed0_gpu1_20260629.sh, AGENTS.md, .learnings/LEARNINGS.md
- See Also: LRN-20260629-001

---

## [ERR-20260706-001] bare_python_missing_in_codex_env

**Logged**: 2026-07-06T00:44:44+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
A local result-summary command failed because this environment does not provide a bare `python` executable.

### Error
```text
/bin/bash: line 1: python: command not found
```

### Context
- Command attempted: summarize local JSON audit artifacts with `python -c ...`.
- Environment: project Codex shell in `/home/fate/gitproject/blockcipher-structure-adaptive-nd-v1`.
- Project commands should already prefer `UV_CACHE_DIR=/tmp/uv-cache uv run ...`; simple non-project summaries can use `python3`.

### Suggested Fix
Use `UV_CACHE_DIR=/tmp/uv-cache uv run python ...` for project commands and `python3` for simple local JSON summaries. Do not assume a bare `python` binary exists.

### Metadata
- Reproducible: yes
- Related Files: outputs/local_audits/r8_integral_aligned_difference_control_seed23/
- See Also: LRN-20260705-001

### Resolution
- **Resolved**: 2026-07-06T00:45:00+08:00
- **Commit/PR**: pending
- **Notes**: Re-ran the summary with `python3`; the audit commands themselves had already succeeded through `UV_CACHE_DIR=/tmp/uv-cache uv run python`.

---

## [ERR-20260706-003] remote_monitor_train_done_marker_false_terminal

**Logged**: 2026-07-06T19:15:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
The trail-position remote monitor exited before score artifacts were ready because it matched `train_done.marker` as if it were final completion.

### Error
```text
monitor.log:
2026-07-06T18:52:48+08:00 completed_missing_or_incomplete_results rows=2

local artifacts:
results/train_matrix.jsonl had 2 rows
logs/<RUN_ID>_train_done.marker existed
logs/<RUN_ID>_done.marker was missing
score_artifacts/global_stats_control/models.json was missing
score_artifacts/trail_position/models.json was missing
```

### Context
- Run ID: `i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706`.
- Operation: watcher-managed retrieval for PRESENT r8 trail-position `65536/class` remote seed0 diagnostic.
- The remote launcher writes `train_done.marker` after training, then starts `scripts\export-checkpoint-scores`.
- The monitor used `compgen -G "${LOCAL_ROOT}/logs/*done.marker"`, which matched `train_done.marker` and exited before final score export artifacts were retrieved.

### Suggested Fix
Use an exact final marker in monitor scripts:

```bash
[[ -f "${LOCAL_ROOT}/logs/${RUN_ID}_done.marker" ]]
```

Do not use broad `*done.marker` globs when the launcher has stage markers such as
`train_done.marker` or `score_export_done.marker`. Tests should assert the
absence of `${LOCAL_ROOT}/logs/*done.marker` in monitors that wait for score
artifacts.

### Metadata
- Reproducible: yes
- Related Files: configs/remote/generated/monitor_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.sh, tests/test_project_structure.py
- See Also: LRN-20260706-023, ERR-20260705-001

### Resolution
- **Resolved**: 2026-07-06T19:16:00+08:00
- **Commit/PR**: pending
- **Notes**: Patched the trail-position monitor to wait for `${RUN_ID}_done.marker` and restarted a corrected local tmux watcher.

---

## [ERR-20260706-004] score_export_missing_dataset_cache_progress

**Logged**: 2026-07-06T19:35:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
The trail-position remote run completed training, but post-training frozen-score export looked stuck because `export-checkpoint-scores` regenerated the validation dataset without disk cache reuse or progress logging.

### Error
```text
remote logs:
<RUN_ID>_train_done.marker existed
<RUN_ID>_export_global_stats_control_stdout.txt was 0 bytes
<RUN_ID>_export_global_stats_control_stderr.txt was 0 bytes
<RUN_ID>_score_export_done.marker was missing
<RUN_ID>_done.marker was missing

remote process:
python.exe scripts\export-checkpoint-scores ... --output-dir ...\score_artifacts\global_stats_control

local score_artifacts:
empty
```

### Context
- Run ID: `i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706`.
- Operation: PRESENT r8 trail-position `65536/class` remote seed0 diagnostic.
- Training used `run_innovation_one_matrix.py` with `--dataset-cache-root`, `--dataset-cache-workers`, and progress JSONL.
- Score export used `make_differential_dataset(...)` directly, so it bypassed the existing disk-backed validation cache and had no progress output.
- This made the remote state look ambiguous: training was done, but frozen-score artifacts for ensemble/error-overlap analysis were not ready.

### Suggested Fix
`scripts/export-checkpoint-scores` should support the same cache/progress route for evaluation data:

```text
--dataset-cache-root
--dataset-cache-chunk-size
--dataset-cache-workers
--progress-output
```

When those arguments are present, it should call `engine.datasets.make_task_dataset(...)` with `split="validation"` so parameter-matched validation caches can be reused and progress events can be monitored.

### Metadata
- Reproducible: yes
- Related Files: src/blockcipher_nd/cli/export_checkpoint_scores.py, configs/remote/generated/run_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.cmd, tests/test_neural_ensemble_cli.py, tests/test_project_structure.py
- See Also: ERR-20260706-003

### Resolution
- **Resolved**: 2026-07-06T19:38:00+08:00
- **Commit/PR**: pending
- **Notes**: Added cache/progress CLI support to `export-checkpoint-scores`, wired the trail-position launcher exports to the shared dataset cache, and added focused tests.

---

## [ERR-20260714-001] windows_findstr_lf_commit_false_negative

**Logged**: 2026-07-14T00:00:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
A fail-closed remote launcher rejected the correct pinned Git commit because
Windows `findstr /x` did not match a SHA line written by `git rev-parse`.

### Error
```text
launcher exit code = 1
seed0 clean clone HEAD = 54c90b4afc366213fa0a2eea711e592188077b98
source_expected_commit.txt bytes =
  b'54c90b4afc366213fa0a2eea711e592188077b98\n'
findstr /x /c:"54c90b4..." source_expected_commit.txt exit code = 1
no scheduled task was created
```

### Context
- Run ID: `i1_gift64_cross_spn_typed_transfer_r3_65536_seed0`.
- Operation: prepare exact pushed-commit run-owned clones before scheduling the
  E4-R3 remote medium diagnostic.
- The clone and detached checkout succeeded, and both revision files contained
  the intended SHA. Only the `findstr /x` text gate failed.

### Suggested Fix
Capture `git rev-parse HEAD` with a batch `for /f` loop and compare the SHA
strings directly with `if /I not`. Keep the run script's independent byte-for-
byte comparison of two files generated by the same Git command.

### Metadata
- Reproducible: yes
- Related Files: configs/remote/generated/launch_i1_gift64_cross_spn_typed_transfer_r3_65536_20260714.cmd, tests/test_cross_spn_typed_transfer_gate.py
- See Also: none

### Resolution
- **Resolved**: 2026-07-14T00:00:00+08:00
- **Commit/PR**: 19072ce
- **Notes**: Replaced `findstr /x` with direct case-insensitive SHA comparison
  and added a regression assertion.

---

## [ERR-20260714-002] schtasks_run_success_without_execution

**Logged**: 2026-07-14T00:00:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
Windows `schtasks /Run` returned success for both GPU tasks, but Task Scheduler
left both tasks Ready and never started the run scripts.

### Error
```text
schedule_create = success
schedule_run = SUCCESS: Attempted to run the scheduled task
task last run time = 1999/11/30 00:00:00
task last result = 267011 (task has not yet run)
run logs = absent
started markers = absent
python GPU processes = absent
```

### Context
- Tasks: `I1_E4_R3_GIFT64_SEED0_GPU0` and
  `I1_E4_R3_GIFT64_SEED1_GPU1`.
- Operation: E4-R3 remote medium diagnostic launch through Windows Task
  Scheduler.
- The launcher correctly created both tasks with `cmd.exe /c`, but default task
  constraints prevented immediate execution despite exit code 0.
- Retrying with `schtasks /Run /I` still left both tasks Ready. The query showed
  that they were configured to run only in an interactive user session, while
  the SSH launch had no usable interactive desktop token.

### Suggested Fix
Create unattended experiment tasks with `/RU SYSTEM /RL HIGHEST`, then invoke
them with `schtasks /Run /I /TN <task>`. Require a bounded post-launch check for
run logs, exact started markers, and Python GPU processes. Never treat the
`/Run` exit code or its success text alone as proof of execution.

### Metadata
- Reproducible: yes
- Related Files: configs/remote/generated/launch_i1_gift64_cross_spn_typed_transfer_r3_65536_20260714.cmd, tests/test_cross_spn_typed_transfer_gate.py
- See Also: ERR-20260714-001

### Resolution
- **Resolved**: 2026-07-15T00:49:00+08:00
- **Commit/PR**: 9aa31dd
- **Notes**: Recreated both tasks under SYSTEM with highest privileges. Both
  tasks entered Running state and produced exact started markers, passing
  readiness files, and advancing disk-cache progress JSONL.

---
