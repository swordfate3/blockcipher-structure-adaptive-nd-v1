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
