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

## [LRN-20260624-002] correction

**Logged**: 2026-06-24T20:30:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
Remote experiment launches should not be personally supervised from the main thread; use tmux monitors to watch and retrieve results automatically.

### Details
The user clarified the intended remote workflow: after a remote GPU experiment is launched or handed off, the main agent should not keep personally supervising it through repeated SSH checks or manual polling. The correct pattern is to start a local tmux monitor/watcher that waits for completion artifacts, retrieves the result archive or raw fallback outputs, writes local logs/markers, and then lets the main thread continue or report only from local artifacts.

This strengthens the existing "do not SSH-poll from the main thread" rule. Manual remote contact should be reserved for controlled exceptions such as a local monitor health failure, a dry-run gate that explicitly allows SSH, or a user-requested repair. Normal long-running training should be monitored by tmux and retrieval scripts, not by interactive supervision.

### Suggested Action
For future remote runs, always launch or verify a supported local tmux monitor immediately after the remote command is handed off. Report the monitor session/log/marker paths to the user. Inspect results only after the monitor has pulled them back locally, unless the controlled gate says remote inspection is allowed.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, scripts/monitor_remote_results.py, outputs/remote_results/
- Tags: remote-training, tmux, monitoring, retrieval, ssh, workflow
- See Also: LRN-20260621-003, LRN-20260622-001
- Pattern-Key: remote_training.tmux_monitor_retrieves_results
- Recurrence-Count: 1
- First-Seen: 2026-06-24
- Last-Seen: 2026-06-24
- Promoted: AGENTS.md

---

## [LRN-20260625-001] best_practice

**Logged**: 2026-06-25T15:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Use `blockcipher-auto-research` as the project research workflow and Karpathy-style guidelines as the default implementation discipline for this repository.

### Details
The user explicitly requested that future project work remember and follow the behavior methods from two skills:

- `skills/blockcipher-auto-research/SKILL.md` should drive the project-local research loop: define the question, identify the same-budget baseline, change one hypothesis at a time, run fixed-budget experiments, generate JSONL/CSV/SVG/gate artifacts, apply evidence-scale language, document keep/discard/crash/diagnostic status, and commit/push scoped repository changes after verification.
- `karpathy-guidelines` should guide coding behavior: read relevant code before editing, state uncertainty when evidence is incomplete, prefer simple boring implementations, avoid unnecessary abstractions/dependencies/frameworks, edit precisely, protect user work, validate against observable success criteria, and debug actual failures rather than guessing.

Correct default going forward:

- Treat `blockcipher-auto-research` as the workflow skeleton for Innovation 1 experiments and related reproduction work.
- Treat Karpathy-style guidelines as the execution style for code/config/docs changes inside that workflow.
- Use the combination to avoid untraceable experiment drift: do not change benchmark protocol, validation data, labels, negative-sample definition, metric computation, or plan-alignment logic while also changing model/feature hypotheses unless the user explicitly requests benchmark redesign.
- Keep changes attributable to one research hypothesis whenever possible, then verify, commit, and push.

### Suggested Action
Promote a concise version to `AGENTS.md` under a project research execution section so future agents default to this combined method. Continue using `paper-code-reproducer` for honest paper-reproduction boundaries and `remote-windows-gpu-conda-ssh` for remote A6000 launch/monitor/retrieval rules when those tasks apply.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, skills/blockcipher-auto-research/SKILL.md, /home/fate/.agents/skills/karpathy-guidelines/SKILL.md
- Tags: workflow, research, coding-discipline, innovation1, experiments, karpathy, auto-research
- See Also: LRN-20260622-001, LRN-20260624-002
- Pattern-Key: workflow.blockcipher_auto_research_plus_karpathy_guidelines
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260625-002] best_practice

**Logged**: 2026-06-25T16:10:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Use clear documentation destinations: formal experiment plans in `docs/experiments/`, research blueprints in `docs/research/`, and historical agent plans only in archive-style locations.

### Details
The user clarified the intended documentation organization after discussing whether `$using-superpowers` creates Markdown plans under `docs/superpowers/`. The correct project convention is:

- New formal experiment plans, reproduction records, result analyses, and next-step execution plans belong under `docs/experiments/`.
- New research blueprints, literature syntheses, theory notes, and broad method proposals belong under `docs/research/`.
- Historical agent execution plans may remain under `docs/superpowers/plans/`, or be migrated to `docs/archive/agent-plans/` if the archive is reorganized.

This separates current project-facing research artifacts from historical superpowers/planning workflow outputs. In particular, do not create new current experiment plans under `docs/superpowers/`; that directory should be treated as historical plan archive unless explicitly reorganized.

### Suggested Action
Promote this convention to `AGENTS.md` under documentation organization. When writing new project docs, choose the destination before creating the file:

- `docs/experiments/` for executable experiment plans and result records.
- `docs/research/` for broad research strategy and literature-backed blueprints.
- `docs/superpowers/plans/` or `docs/archive/agent-plans/` only for historical agent execution plans.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, docs/research/, docs/superpowers/plans/
- Tags: docs, documentation-organization, experiments, research, archive, superpowers
- See Also: LRN-20260625-001
- Pattern-Key: docs.destination.experiments_research_agent_archive
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260625-003] correction

**Logged**: 2026-06-25T20:30:07+08:00
**Priority**: critical
**Status**: promoted
**Area**: docs

### Summary
Do not answer project implementation or tooling facts from memory; inspect source, config, logs, or artifacts before reporting them.

### Details
The user corrected a hallucinated implementation claim: the assistant previously stated that project plotting already used Matplotlib, but the actual implementation at that time was hand-written SVG in `src/blockcipher_nd/evaluation/plots.py`. The later visualization change did convert plotting to Matplotlib, but the earlier answer was still wrong because it was made without first checking the code.

Correct behavior:

- Before saying "the project currently uses X" or "this is implemented by Y", inspect relevant files with `rg`, `sed`, config/lockfile reads, tests, logs, or result artifacts.
- This applies especially to dependencies, plotting/rendering libraries, training protocols, remote launch scripts, artifact paths, experiment status, metrics, checkpoint selection, and result gates.
- If the evidence has not been checked in the current turn or handoff, state that the answer is an assumption or check before reporting.
- When a prior statement is found to be wrong, correct it explicitly and separate the old false claim from the newly verified state.
- Do not rely on a recent memory of a change unless the repository state or command output confirms it.

### Suggested Action
Promote a concise factual-reporting rule to `AGENTS.md`. For future implementation/status answers, first cite or consult the relevant source/config/artifact. If verification is not possible, qualify the uncertainty instead of making a definitive claim.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, src/blockcipher_nd/evaluation/plots.py, pyproject.toml, uv.lock, scripts/plot-results
- Tags: hallucination, factual-reporting, source-verification, implementation-facts, visualization
- See Also: LRN-20260624-001, LRN-20260625-001
- Pattern-Key: workflow.factual_reporting.verify_before_claim
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260626-001] best_practice

**Logged**: 2026-06-26T11:07:22+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Update `docs/experiments/` for meaningful experiment lifecycle events, and update `docs/research/` only for research-direction changes.

### Details
The user clarified that future agents should not mechanically write both `docs/experiments/` and `docs/research/` after every action. The correct documentation workflow is:

- New formal or medium-scale experiment plans should be recorded under `docs/experiments/`.
- Completed meaningful experiments should update the relevant `docs/experiments/` record with run id, configuration, evidence gate, metrics, result status, and next action.
- Smoke tests, temporary debug runs, and local implementation checks do not require experiment documentation unless they change an evidence judgment, expose an important failure mode, or become part of the research record.
- `docs/research/` is for research blueprints, theory notes, literature syntheses, method proposals, and major route changes. It should not become a run-by-run log.
- When unsure, choose the smallest durable documentation update that preserves the evidence chain without creating noise.

### Suggested Action
Promote this as an execution rule in `AGENTS.md` under Documentation Organization. Future experiment launches and completions should first decide whether the event is documentation-worthy, then update only the appropriate destination.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, docs/research/
- Tags: docs, experiments, research, documentation-workflow, evidence-chain
- See Also: LRN-20260625-002
- Pattern-Key: docs.update_lifecycle.experiments_vs_research
- Recurrence-Count: 1
- First-Seen: 2026-06-26
- Last-Seen: 2026-06-26
- Promoted: AGENTS.md

---

## [LRN-20260627-001] correction

**Logged**: 2026-06-27T11:35:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
When a meaningful remote experiment result is completed and retrieved, automatically update `docs/experiments/` before reporting; do not ask whether to document it.

### Details
The user corrected the reporting workflow after the `I1-SPN-001` 1M/class single-seed result completed. The assistant initially suggested documenting the result as a next step instead of doing it immediately. For this project, meaningful experiment results are part of the evidence chain and should be written to the relevant experiment document as soon as the result is retrieved and parsed.

Correct behavior:

- If a meaningful experiment reaches a valid result gate and artifacts are retrieved, update the relevant `docs/experiments/` record in the same task turn.
- Include run id, protocol scale, gate status, local/remote artifact paths, metrics, deltas versus baseline, claim scope, and next action.
- Do not ask the user whether to write the result document; ask only if there is genuine ambiguity about which experiment record should own the result.
- Smoke tests, temporary debug checks, and local implementation checks still do not need docs unless they change the evidence judgment or expose an important failure mode.

### Suggested Action
Promote this rule to `AGENTS.md` under Documentation Organization. Future remote-result handling should follow this sequence: retrieve artifacts, parse metrics, generate missing local plots/history if needed, update `docs/experiments/`, verify, commit, push, then report.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, outputs/remote_results/
- Tags: docs, experiments, remote-results, evidence-chain, workflow
- See Also: LRN-20260626-001, LRN-20260624-002
- Pattern-Key: docs.results.auto_update_experiment_record
- Recurrence-Count: 1
- First-Seen: 2026-06-27
- Last-Seen: 2026-06-27
- Promoted: AGENTS.md

---

## [LRN-20260627-002] research

**Logged**: 2026-06-27T19:41:43+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
After the 262k N1-v2 structure ablation, do not scale the current gated-MCND route as the main Innovation 1 path; prioritize SPN-only and transition-aware SPN backbones.

### Details
The completed and locally retrieved run `i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627` tested N0 Zhang/Wang MCND, N1-v1 late fusion, SPN-only, true-P gated MCND, and shuffled-P gated MCND at `262144/class` with strict `encrypted_random_plaintexts` negatives and Zhang/Wang Case2 official MCND sample structure.

Observed AUC:

- `present_zhang_wang_keras_mcnd`: `0.784541`
- `present_nibble_paligned_mcnd`: `0.784299`
- `present_nibble_paligned_spn_only`: `0.791488`
- `present_nibble_paligned_gated_mcnd`: `0.784897`
- `present_nibble_shuffled_paligned_gated_mcnd`: `0.784281`

The true-P gated model was weakly positive against N0, N1-v1, and shuffled control, but it failed all planned continuation gates:

- N1-v2 AUC vs N0: `+0.000356`, below required `+0.002`.
- N1-v2 AUC vs N1-v1: `+0.000598`, below required `+0.001`.
- N1-v2 AUC vs shuffled: `+0.000615`, below required `+0.001`.
- N1-v2 calibrated accuracy vs N0: `-0.000530`.

The strongest diagnostic signal came from `SPN-only`, not MCND fusion. This suggests the PRESENT nibble/P-layer structure view contains useful real-vs-random signal, but the current gate/fusion design dilutes or fails to inject it into the MCND backbone.

This is medium diagnostic evidence, not formal multi-seed `>=1000000/class` evidence. However, it is enough to reject scaling the exact current `present_nibble_paligned_gated_mcnd` design as the main next route.

### Suggested Action
For the next Innovation 1 experiment, design a minimal SPN transition-aware backbone around the SPN-only signal. Keep the benchmark fixed and compare at the same `262144/class` evidence scale:

1. Keep N0 baseline and current SPN-only as anchors.
2. Add N2 transition-aware models that treat 16 PRESENT nibbles and P-layer transitions as the primary representation.
3. Include true-P versus shuffled-P controls for attribution.
4. Do not change validation data, labels, negative mode, metric computation, or Zhang/Wang Case2 sample construction.
5. Only consider 1M/multi-seed scaling if N2 beats SPN-only and the true-P route beats shuffled-P by the predeclared gate.

### Metadata
- Source: conversation
- Related Files: docs/experiments/innovation1-n1v2-structure-ablation-plan.md, outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/
- Tags: innovation1, spn, present, n1v2, spn-only, transition-aware-backbone, evidence-gates
- See Also: LRN-20260621-001, LRN-20260623-001, LRN-20260627-001
- Pattern-Key: innovation1.spn_present.prioritize_transition_backbone_after_n1v2
- Recurrence-Count: 1
- First-Seen: 2026-06-27
- Last-Seen: 2026-06-27

---

## [LRN-20260628-001] correction

**Logged**: 2026-06-28T00:00:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
After a new training experiment smoke test passes, automatically continue to the corresponding non-smoke remote training launch instead of stopping at the smoke report.

### Details
The user corrected the experiment workflow after the SPN-only attribution smoke test. For this project, when the user asks to "推进" a new training experiment and a smoke matrix is part of the setup, the smoke is only a gate. If smoke passes and code/config/docs have been verified, committed, and pushed, the agent should proceed to launch the planned medium/formal remote run using the pushed GitHub commit and the established tmux monitor/retrieval workflow.

Correct behavior:

- Treat local smoke as a readiness gate, not the final deliverable, unless the user explicitly asks for smoke/local verification only.
- After smoke passes, commit and push scoped changes.
- Create or verify the matching remote config/launcher for the non-smoke run.
- Audit remote rules: `cmd.exe /c`, all artifacts under `G:\lxy`, disk-backed dataset cache/progress for `>=65536/class`, GitHub-pushed commit source.
- Launch remote training and start/verify a local tmux monitor for automatic retrieval.
- Report the run as `running`/`planned` with run id and monitor details, not as complete.

### Suggested Action
Promote this to `AGENTS.md` under research execution or remote workflow. Future "推进训练实验" tasks should run the full pipeline: implement -> smoke -> verify -> commit/push -> remote launch -> tmux monitor handoff.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, configs/experiment/innovation1/, configs/remote/, scripts/generated/remote/, docs/experiments/
- Tags: experiments, smoke, remote-training, tmux-monitor, workflow, innovation1
- See Also: LRN-20260622-001, LRN-20260627-001
- Pattern-Key: workflow.training.smoke_then_remote_launch
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260628-002] research

**Logged**: 2026-06-28T17:40:34+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
After SPN-only attribution, prioritize an InvP-centered PRESENT/SPN distinguisher route over generic DeltaC+InvP concatenation.

### Details
The completed remote run `i1_spn_only_attr_r7_262k_seed0_gpu1_20260628` showed that the strongest diagnostic row was `present_nibble_invp_only_spn_only`, not the previous `DeltaC + InvP(DeltaC)` anchor.

At `262144/class`, strict `encrypted_random_plaintexts` negatives, Zhang/Wang Case2 official MCND protocol:

- `present_zhang_wang_keras_mcnd`: AUC `0.783228`
- `present_nibble_delta_only_spn_only`: AUC `0.782918`
- `present_nibble_shuffled_paligned_spn_only`: AUC `0.784487`
- `present_nibble_paligned_spn_only`: AUC `0.790665`
- `present_nibble_invp_only_spn_only`: AUC `0.792536`

Key attribution deltas:

- InvP-only vs baseline: AUC `+0.009308`
- InvP-only vs DeltaC-only: AUC `+0.009617`
- InvP-only vs shuffled-P: AUC `+0.008048`
- InvP-only vs DeltaC+InvP anchor: AUC `+0.001871`

This supports the interpretation that inverse-P aligned `DeltaC` is the dominant useful signal in the current SPN-only family. Raw `DeltaC` concatenation may dilute or fail to improve the signal under the simple token-mixer architecture.

This is medium diagnostic single-seed evidence, not formal `>=1000000/class` multi-seed evidence.

### Suggested Action
Advance Innovation 1 as an InvP-centered route:

1. Add a compact InvP pair-set consistency model that reuses the existing InvP-only encoder and changes only pair aggregation.
2. Compare against baseline, current SPN-only anchor, InvP-only, DeltaC-only, and shuffled-P under the same `262144/class` protocol.
3. Treat local smoke as a launch gate; if smoke passes, automatically commit/push and launch the remote medium diagnostic with tmux monitor retrieval.
4. Only consider 1M/class multi-seed scaling after InvP-only or InvP-centered consistency remains stable across at least one additional 262k seed or clearly beats the current InvP-only anchor.

### Metadata
- Source: conversation
- Related Files: docs/experiments/innovation1-spn-only-attribution-plan.md, outputs/remote_results/i1_spn_only_attr_r7_262k_seed0_gpu1_20260628/
- Tags: innovation1, spn, present, invp, p-layer, attribution, pair-consistency
- See Also: LRN-20260627-002, LRN-20260628-001
- Pattern-Key: innovation1.spn_present.invp_centered_route
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28

---

## [LRN-20260628-003] correction

**Logged**: 2026-06-28T18:20:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Before starting or advancing meaningful training tasks, proactively write or update the experiment plan instead of waiting for the user to request it.

### Details
The user corrected the experiment workflow: future training launches should not depend on the user saying "写计划" first. For this project, a meaningful training task starts with a project-facing plan under `docs/experiments/`, then proceeds through implementation, smoke/readiness validation, scoped commit/push, remote launch from the pushed commit, local tmux monitoring, retrieval, result parsing, experiment-doc update, and final scoped commit/push.

Correct behavior:

- If the task is a meaningful new run, route, scale-up, or ablation, create or update the relevant `docs/experiments/` plan before launching.
- The plan should include the research question, fixed protocol, same-budget baseline, rows/models, scale, evidence gate, artifact paths, cache/progress expectation, remote device/run id expectations, and next action.
- Smoke/local checks remain readiness gates; if they pass and the user did not request smoke-only, continue to the planned non-smoke run.
- Do not mechanically update `docs/research/` for every run; update it only when the broader research route, theory, or method blueprint changes.
- Do not ask whether to write the plan for a meaningful training task; ask only when ownership of the document is genuinely ambiguous.

### Suggested Action
Promote this to `AGENTS.md` under Research Execution Style. Future experiment advancement should follow: plan first -> implement -> smoke -> commit/push -> remote launch -> tmux monitor -> retrieve -> update experiment docs -> commit/push.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, .learnings/LEARNINGS.md
- Tags: experiments, planning, docs, remote-training, workflow
- See Also: LRN-20260628-001, LRN-20260627-001
- Pattern-Key: workflow.training.proactive_experiment_plan_first
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260628-004] correction

**Logged**: 2026-06-28T19:25:00+08:00
**Priority**: high
**Status**: promoted
**Area**: research

### Summary
Do not default every remote training experiment to large multi-model comparison matrices; keep incremental runs lean.

### Details
The user pointed out that repeatedly comparing many models in every remote experiment is cumbersome. The correct workflow is to separate attribution/audit experiments from incremental model-selection experiments:

- Full comparison matrices are useful for stage gates, attribution studies, protocol audits, and checking whether a control invalidates a route.
- Incremental model changes should usually compare only the new candidate against the strongest current same-protocol anchor, plus the minimum necessary baseline/control rows.
- A normal incremental remote matrix should target 2-3 models and rarely exceed 4.
- Do not keep re-running historical weaker controls in every new experiment once the attribution route is stable, unless the research question specifically requires them.
- Existing already-launched remote jobs should normally be allowed to finish; apply the lean-matrix rule to the next planned runs.

For Innovation 1, after InvP attribution has already shown `InvP-only` as the current strongest route, follow-up architecture experiments should usually compare:

```text
new candidate
current strongest InvP anchor
optional Zhang/Wang baseline or one critical control
```

instead of automatically including DeltaC-only, shuffled-P, old DeltaC+InvP, and every previous baseline in each run.

### Suggested Action
Promote this to `AGENTS.md` under Research Execution Style. Future experiment plans should explicitly justify any matrix larger than 3-4 rows and prefer lean comparisons for iteration speed and clarity.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, configs/experiment/innovation1/
- Tags: experiments, model-matrix, remote-training, efficiency, innovation1
- See Also: LRN-20260628-002, LRN-20260628-003
- Pattern-Key: workflow.training.lean_experiment_matrix
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260629-001] correction

**Logged**: 2026-06-29T14:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
Remote tmux monitoring should be delegated to a sub-agent or watcher; the main thread should not repeatedly loop over tmux status.

### Details
The user corrected a workflow drift: after remote GPU launch, I repeatedly checked tmux sessions and monitor logs from the main thread. This is not the intended project workflow. The correct pattern is:

- Main thread launches or verifies exactly enough to hand off the remote run.
- A tmux monitor, watcher, or sub-agent owns the monitoring loop and retrieval.
- Main thread should not repeatedly inspect `tmux ls`, `monitor.log`, or remote progress just to see whether training finished.
- Main thread resumes result processing only when local artifacts have arrived, when the user explicitly asks for a status check, or when a monitor health failure is detected by the delegated watcher/sub-agent.
- If additional post-processing is needed after retrieval, it should also be delegated to a watcher/sub-agent where possible, not manually polled from the main thread.

This refines the existing "do not SSH-poll from the main thread" rule: do not replace SSH polling with main-thread tmux polling. Monitoring loops belong outside the main research/implementation thread.

### Suggested Action
Promote to `AGENTS.md` under Remote Monitoring And Retrieval. Future remote launches should report the monitor/sub-agent handoff and then continue with non-monitoring work or wait for retrieved artifacts. If the user asks "continue" while a remote run is active, avoid repetitive tmux checks; do a single local artifact check if needed, then proceed with planning/implementation that does not require the running result.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/LEARNINGS.md, outputs/remote_results/
- Tags: remote-training, tmux, subagent, monitoring, workflow
- See Also: LRN-20260624-001
- Pattern-Key: remote_training.delegate_tmux_monitoring_to_subagent
- Recurrence-Count: 1
- First-Seen: 2026-06-29
- Last-Seen: 2026-06-29
- Promoted: AGENTS.md

---
