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
