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
