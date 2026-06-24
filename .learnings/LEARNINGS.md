## [LRN-20260621-001] correction

**Logged**: 2026-06-21T20:45:00+08:00
**Priority**: critical
**Status**: promoted
**Area**: docs

### Summary
Do not treat SPN/PRESENT small or medium sample runs as formal evidence that a model or route has failed.

### Details
The user corrected an important experimental interpretation error: prior SPN/PRESENT work was discussed too much like formal training, but local completed metrics show PRESENT/SPN results had only reached about `65536` samples per class. Several logs mentioning `131072` or `262144` rows were often total rows across both classes, cache/progress rows, queue plans, or incomplete runs, not completed `>100000/class` formal training.

Correct project distinction:

- Smoke/screen: below `65536/class`.
- Medium diagnostic: `65536/class` through about `524288/class`.
- Formal training: at least `1000000/class`, preferably multi-seed, fixed protocol, completed, retrieved, and plan-aligned.

For SPN/PRESENT, before claiming a route has hit its ceiling or failed, require completed and retrieved scale evidence. A `32k/class` or `65k/class` result may reject only obviously dead variants; it must not be used as a definitive conclusion that the overall route cannot improve.

Current factual baseline from the 2026-06-21 audit:

- ARX/SPECK has completed results above `100000/class`, including `131072/class` and `262144/class`.
- SPN/PRESENT completed metric rows found locally maxed out around `65536/class`.
- Therefore, prior SPN/PRESENT accuracy bottleneck claims were under-supported by large-scale evidence.

### Suggested Action
For future SPN/PRESENT experiments, always state the scale class in reports and labels. Use small runs only as screens. Before making negative claims about accuracy ceilings, run and retrieve at least a medium scale ladder such as `65536/class -> 262144/class`, and reserve "formal result" language for `>=1000000/class` multi-seed completed runs.

### Metadata
- Source: user_feedback
- Related Files: outputs/, experiments/innovation1/plans/, experiments/innovation1/configs/remote/
- Tags: spn, present, experiment-scale, formal-training, accuracy-interpretation
- Pattern-Key: innovation1.spn_present.formal_scale_required
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260621-003] best_practice

**Logged**: 2026-06-21T20:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Promote only durable rules from `memory/` into `AGENTS.md`; keep experiment history and run-specific details in `memory/`.

### Details
The user approved the memory cleanup approach: do not merge all `memory/` content into `AGENTS.md`. Instead, extract compact, stable rules that affect future agent behavior. During the 2026-06-21 cleanup, repeated memory/task-plan rules were promoted for remote Windows GPU hygiene, monitor/retrieval workflow, evidence claim gates, and verification/workspace hygiene.

Specific promoted rule groups:

- Remote artifacts and generated project files must stay under `G:\lxy`.
- Windows remote schedule commands must use `cmd.exe /c`, not `cmd.exe /k`.
- After remote launch/handoff, main thread should not SSH-poll; use tmux/watchers/monitors and controlled gates.
- Result reports must distinguish planned, running, completed remotely, fallback-retrieved, verified-branch retrieved, and plan-aligned.
- Strict SPN/PRESENT claims require claim gates, encrypted-random-plaintext negatives, and explicit qualification for multi-query/application-level evidence.
- Use `uv run pytest ...`; keep project-root `tmp_*` clean.

### Suggested Action
When future memory files grow, periodically scan for repeated durable rules and promote only those concise rules to `AGENTS.md`. Leave detailed run ids, timestamps, and transient experiment states in `memory/` or `progress.md`.

### Metadata
- Source: user_feedback
- Related Files: memory/, task_plan.md, progress.md, AGENTS.md
- Tags: memory, agents, remote-workflow, evidence-gates, workspace-hygiene
- See Also: LRN-20260621-002
- Pattern-Key: workflow.memory_to_agents.promote_only_durable_rules
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260621-002] best_practice

**Logged**: 2026-06-21T20:43:13+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
When reading old conversation, handoff, progress, or memory documents, persist important corrections with the self-improvement workflow.

### Details
The user clarified that previous memory-document reading should use the `self-improvement` skill to store durable memory. Important findings from old dialogue, `memory/`, handoff summaries, `progress.md`, or result audits must not remain only in transient model context. If a finding changes future experimental interpretation, remote workflow, reporting language, or agent behavior, log it to `.learnings/LEARNINGS.md` using the skill format and promote concise operational rules to `AGENTS.md` when broadly applicable.

This is especially important for long-running Innovation 1 work because context-window loss and thread restarts have already caused confusion about what had actually completed, what scale counts as formal, and whether remote results were retrieved.

### Suggested Action
Before and after reading historical memory files for a major task, check whether any conclusion should be persisted. Use `.learnings/LEARNINGS.md` for detailed context and `AGENTS.md` for short rules that future agents should obey immediately.

### Metadata
- Source: user_feedback
- Related Files: memory/, progress.md, task_plan.md, .learnings/LEARNINGS.md, AGENTS.md
- Tags: memory, handoff, self-improvement, context-window, project-rules
- See Also: LRN-20260621-001
- Pattern-Key: workflow.memory_reading.persist_with_self_improvement
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260622-001] correction

**Logged**: 2026-06-22T11:40:00+08:00
**Priority**: critical
**Status**: promoted
**Area**: infra

### Summary
After every completed repository modification, make a scoped git commit and push when a remote exists so the workspace does not accumulate dirty state.

### Details
The user corrected a workflow failure: remote experiments are intended to pull code from GitHub, but the workspace had accumulated many uncommitted changes. To avoid committing unrelated dirty files, a remote run was started with `scp` overlays into `G:\lxy`, which made the run less reproducible than a clean GitHub commit-based launch.

Correct workflow:

- Complete any repository edit, including code, config, scripts, tests, README/docs, `.learnings/`, `AGENTS.md`, generated project files, or memory-rule updates.
- Run appropriate verification.
- Commit only the scoped files for the completed task.
- Push the branch to the remote repository when a remote is configured; if no remote exists, report that push is not possible.
- Keep the workspace clean for agent-authored changes before starting new work or launching remote experiments.
- Remote experiments should default to a GitHub-pushed commit. Dirty/scp overlay launches are emergency-only and must be explicitly labeled as such in status reports and handoff notes.

This rule does not authorize reverting or committing unrelated user changes. If unrelated dirty files already exist, isolate the task's files in a scoped commit and report the remaining unrelated dirty state separately.

### Suggested Action
Promote this to `AGENTS.md` under workspace hygiene and remote workflow. After any file modification, run the relevant verification, make a scoped commit for the files just changed, and push if a remote is configured. Before future remote launches, run `git status --short`, ensure required files are committed and pushed, and avoid relying on scp overlays for normal experiment reproducibility.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/LEARNINGS.md, experiments/innovation1/configs/remote/, scripts/generated/remote/
- Tags: git, commit, push, workspace-hygiene, remote-reproducibility
- See Also: LRN-20260621-003
- Pattern-Key: workflow.git_commit_push_after_code_changes
- Recurrence-Count: 2
- First-Seen: 2026-06-22
- Last-Seen: 2026-06-24
- Promoted: AGENTS.md

### Recurrence Update
- **Updated**: 2026-06-24T15:08:01+08:00
- **Source**: user_feedback
- **Notes**: User explicitly reminded that every modification should be committed, not only code/config/script edits. This includes README/docs, `.learnings/`, `AGENTS.md`, and other memory or documentation updates. Push remains required when a remote is configured; if no remote exists, commit locally and report that push cannot be performed.

---

## [LRN-20260622-002] best_practice

**Logged**: 2026-06-22T13:05:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Treat SPN/PRESENT active-nibble accuracy as an auxiliary trail-activity signal, not as real-vs-random accuracy; promote it into an explicit active-pattern distinguisher route.

### Details
The user identified that the high `val_active_nibble_bit_accuracy` from the multitask active-nibble run should not be left as a side metric. It measures per-position correctness for 16 active/inactive nibble labels per sample, where "active" means a 4-bit differential/trail cell is non-zero. It is not ciphertext-value nonzero-ness, not whole-sample trail accuracy, and not real-vs-random classification accuracy.

Correct interpretation:

- `active nibble` means a nibble of a differential or candidate trail state is non-zero.
- The metric is averaged over all sample-position binary labels, e.g. validation rows times 16 positions.
- It can be inflated by inactive-class imbalance; always compare against all-inactive baseline and report active precision, recall, F1, and per-position rates.
- The research opportunity is to convert this auxiliary structure recognition into explicit real-vs-random evidence: active count, position frequency, candidate-trail disagreement, confidence, margin, pair-set consistency, and trail-family match scores.
- Strict evidence still requires `encrypted_random_plaintexts` negatives and separate reporting of single-sample raw accuracy, AUC, and multi-query aggregation.

### Suggested Action
Implement the active-pattern route as a staged Innovation 1 SPN plan:

1. Add deterministic active-pattern/statistics extraction from existing PRESENT beamstats/candidate-trail feature encodings.
2. Add diagnostics for active-label imbalance and real-vs-random distribution separation.
3. Train active-only and active-plus-candidate-statistics baselines before any large neural model.
4. If small/medium evidence is positive, run the route at 262144/class and then >=1000000/class multi-seed before making formal claims.

### Metadata
- Source: user_feedback
- Related Files: outputs/remote_results_incomplete/innovation1-spn-present-multitask-active-nibble-fast-gate-r7-gpu1-20260618/, src/blockcipher_ai_eval/features/pair_features.py, docs/superpowers/plans/
- Tags: innovation1, spn, present, active-nibble, trail-activity, distinguisher, evidence-gates
- See Also: LRN-20260621-001, LRN-20260621-002
- Pattern-Key: innovation1.spn_present.active_pattern_distinguisher
- Recurrence-Count: 2
- First-Seen: 2026-06-22
- Last-Seen: 2026-06-23

---

## [LRN-20260623-001] research

**Logged**: 2026-06-23T10:23:47+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
SPN/PRESENT active-only 24-dim active-pattern screen failed as a real-vs-random distinguisher despite the earlier high auxiliary active-nibble signal.

### Details
The active-pattern route completed and was retrieved from a verified result branch:

- Run ID: `innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622`
- Result branch: `results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622`
- Result commit: `6c3243137b821d5fc39d266d8aa5f39622ad4fdd`
- Local archive: `outputs/remote_results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622/`
- Gate: `result_lines=2`, `expected_rows=2`, `runner_exit_code=0`, `archive_integrity=pass`
- Config: `rounds=7`, `samples_per_class=65536`, `pairs_per_sample=16`, `negative_mode=encrypted_random_plaintexts`, `sample_structure=zhang_wang_case2_mcnd`
- Mean result: `accuracy=0.500000`, `AUC=0.491578`, `feature_dim=24`
- Seed 0: `val_accuracy=0.5`, `val_auc=0.4894024282693863`
- Seed 1: `val_accuracy=0.5`, `val_auc=0.4937528520822525`

Failure interpretation:

- The auxiliary target and the distinguisher target are different. Active-nibble prediction asks whether each 4-bit trail/difference cell is non-zero; real-vs-random classification asks whether a pair set came from the target differential encryption process or encrypted-random-plaintext negatives.
- The 24-dim active summary is too coarse. It keeps only active position frequencies and aggregate density statistics, but discards value-level S-box/DDT evidence, candidate trail scores, top-k margins, confidence, per-pair ordering, and cross-pair consistency.
- Under this representation, real and random samples can share almost the same active-density and position-frequency distribution, i.e. `P(active-summary | real) ~= P(active-summary | random)`. The retrieved AUC below `0.5` is direct evidence that the current active-only summary gives little or no separation.
- Earlier high `active_nibble_bit_accuracy` may be partly inflated by inactive-class imbalance. Per-position bit accuracy can be high when many positions are inactive, so active precision, recall, F1, balanced accuracy, and all-inactive baselines are required before treating it as strong structural evidence.
- Therefore, active-pattern should be used as auxiliary supervision or one component of richer candidate-trail evidence, not as a standalone final feature family.

This was only a screen/medium diagnostic at `65536/class`, not formal `>=1000000/class` evidence. However, because the active-only baseline is already at chance with strict `encrypted_random_plaintexts` negatives, do not scale this exact 24-dim linear route as the main next experiment.

### Suggested Action
Retire the active-only 24-dim linear baseline as a main scaling route. Keep active-nibble information, but combine it with features that measure whether candidate trails actually support the observed sample:

- Top-1/top-2 trail score and top-k margin
- Candidate score entropy and confidence
- Candidate disagreement across pairs
- Active-pattern-to-top-trail match
- Pair-set trail-family consistency
- Transition-spectrum features and multi-query score aggregation

For future reports, state plainly: high active-nibble auxiliary accuracy proves the model can learn trail-activity propagation patterns; it does not prove real-vs-random distinguishability.

### Metadata
- Source: conversation
- Related Files: src/blockcipher_ai_eval/features/spn_active_pattern.py, experiments/innovation1/run_spn_active_pattern_baseline.py, outputs/remote_results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622/
- Tags: innovation1, spn, present, active-pattern, active-nibble, real-vs-random, failure-analysis, evidence-gates
- See Also: LRN-20260622-002
- Pattern-Key: innovation1.spn_present.active_pattern_distinguisher
- Recurrence-Count: 1
- First-Seen: 2026-06-23
- Last-Seen: 2026-06-23

---

## [LRN-20260623-002] correction

**Logged**: 2026-06-23T21:56:37+08:00
**Priority**: critical
**Status**: promoted
**Area**: infra

### Summary
Remote training that generates datasets or derived features must use disk-backed cache/progress/reuse before launch; do not run large remote jobs with pure in-memory one-shot generation.

### Details
The user corrected a recurring workflow mistake: prior project work had already established that dataset generation should write reusable artifacts such as `features.npy`, `labels.npy`, metadata, CSV/JSONL summaries, and progress logs. The candidate-evidence route violated this principle by launching `65536/class` remotely through a new prototype runner that built `features: list[np.ndarray]` in memory and only wrote a final result after all feature generation, training, and evaluation finished.

Correct rule:

- Any remote training or medium/large screen that generates datasets, feature matrices, candidate-evidence features, trail statistics, or other derived training inputs must have disk-backed cache before launch.
- The cache must be under `G:\lxy` on the remote and normally inside the run directory or approved run cache root.
- Required artifacts include cache metadata, feature/label arrays or equivalent chunked files, progress JSONL/logging, and reuse/resume behavior when parameters match.
- New runners and new feature routes are not exempt. If they bypass `run_innovation_one_matrix.py`, they must implement an equivalent route-specific cache before remote scale-up.
- Smoke-only local experiments may use in-memory generation, but remote launches at `65536/class` or above must not.
- Do not call a remote experiment ready to launch until this cache/progress gate has been checked explicitly.

The immediate failure mode was the candidate-evidence baseline: positive local fast screens led to a remote `65536/class` run, but its feature generation was pure Python/in-memory and produced no progress or reusable cache, making the remote appear stalled and wasting time.

### Suggested Action
Promote this rule to `AGENTS.md` under Remote Windows GPU Rules / Verification. Add cache/progress support to `experiments/innovation1/run_spn_candidate_evidence_baseline.py` before relaunching scaled candidate-evidence experiments:

- `--feature-cache-root`
- `--feature-cache-chunk-size`
- `--progress-output`
- disk-backed `features.npy` / `labels.npy` / `metadata.json`
- cache identity including rounds, seeds, samples_per_class, pairs_per_sample, negative mode, sample structure, difference profile, key rotation, beam width, depth, source, and feature dimension
- chunk progress events and cache reuse

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, experiments/innovation1/run_spn_candidate_evidence_baseline.py, experiments/run_innovation_one_matrix.py, src/blockcipher_ai_eval/data/cache/disk.py
- Tags: remote-training, dataset-cache, feature-cache, progress-logging, innovation1, spn, present, candidate-evidence
- See Also: LRN-20260621-003
- Pattern-Key: remote_training.must_use_disk_cache_for_generated_data
- Recurrence-Count: 1
- First-Seen: 2026-06-23
- Last-Seen: 2026-06-23
- Promoted: AGENTS.md

---

## [LRN-20260624-001] correction

**Logged**: 2026-06-24T16:03:00+08:00
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
Generated result plots must include human-readable coordinate axes, tick labels, and grid lines; bare lines without coordinate values are not acceptable.

### Details
The user corrected the first SVG output from `scripts/plot-results`: it drew train/validation curves, but the plot did not have enough coordinate values to interpret the metric scale and epoch positions like a normal deep learning training curve. A result figure should support human inspection without reading the raw JSON/CSV.

Correct plotting expectations:

- X axis should show epoch tick values, not only first/last labels.
- Y axis should show metric tick values, including intermediate values such as `0.25`, `0.5`, `0.75` for accuracy/AUC.
- Light grid lines should align with tick labels so the reader can estimate values.
- Axis labels such as `epoch`, `accuracy`, `auc`, and `loss` should be visible.
- Train and validation curves should remain visually distinct and annotated.

### Suggested Action
Keep visualization tests that assert generated SVG contains readable axis labels and intermediate tick values. When adding new plots, inspect the rendered artifact or its SVG text for axes, tick labels, and grid lines before calling the visualization complete.

### Metadata
- Source: user_feedback
- Related Files: src/blockcipher_nd/evaluation/plots.py, scripts/plot-results, tests/test_project_structure.py
- Tags: visualization, svg, training-curves, docs, result-reporting
- Pattern-Key: visualization.training_curves.require_readable_axes
- Recurrence-Count: 1
- First-Seen: 2026-06-24
- Last-Seen: 2026-06-24

---
