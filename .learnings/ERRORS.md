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
```

### Context
- Command attempted: `git push origin main`
- Purpose: publish already committed Innovation 1 experiment, gate, documentation, and remote-launch assets so `scripts/check-launch-source` can pass before a compliant remote run.
- Local gate state before push attempt: `main...origin/main [ahead 48]`, dirty `false`, errors `unpushed_commits`.
- Remote launch affected: `i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706`.
- Latest recurrence: push of local head `8c75600` after the E1 8192/class
  adjudication; branch state `main...origin/main [ahead 1]`.

### Suggested Fix
Do not work around this by dirty overlay, alternate push commands, or SSH remote launch from unpublished code. Either obtain explicit user approval for the exact external push of 48 commits to `origin/main`, or continue with local diagnostics/watchers until a safer publication path exists.

### Metadata
- Reproducible: yes
- Related Files: docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, scripts/check-launch-source
- See Also: ERR-20260622-001, ERR-20260624-003, ERR-20260705-001, LRN-20260706-022
- Pattern-Key: infra.git_push.external_reviewer_unavailable_or_rejected
- Recurrence-Count: 2
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-09

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
- Related Files: /tmp/run_innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0_gpu1_20260626.cmd, /home/fate/.agents/skills/remote-windows-gpu-conda-ssh/SKILL.md
- See Also: ERR-20260624-003

### Resolution
- **Resolved**: 2026-06-26T11:19:00+08:00
- **Commit/PR**: pending
- **Notes**: Launcher was patched to use SSH repo URL plus the remote dedicated GitHub key, then re-uploaded for relaunch.

---

## [ERR-20260627-001] remote_plot_missing_matplotlib

**Logged**: 2026-06-27T11:35:00+08:00
**Priority**: medium
**Status**: pending
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

### Suggested Fix
Before relying on remote plot generation, verify the remote environment has Matplotlib installed or add a launcher preflight that records missing optional plotting dependencies. Keep local post-retrieval plot regeneration as a safe fallback when training results are valid but remote plotting fails.

### Metadata
- Reproducible: yes
- Related Files: scripts/plot-results, src/blockcipher_nd/evaluation/plots.py, outputs/remote_results/i1_spn_present_mcnd_r7_1m_seed0_gpu1_retry1_20260626/logs/
- See Also: LRN-20260624-001, ERR-20260626-001

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
