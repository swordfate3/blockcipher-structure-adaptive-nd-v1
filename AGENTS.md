# Agent Notes

## Innovation 1 Experiment Scale

- Do not call SPN/PRESENT `8k`, `16k`, `32k`, or `65k` samples-per-class runs "formal training" or definitive failures. They are smoke/screen or medium diagnostics only.
- For SPN/PRESENT, distinguish total rows from `samples_per_class`: `131072` total rows often means only `65536/class`.
- Before claiming a SPN/PRESENT model or feature route has hit its ceiling, require completed, retrieved, plan-aligned scale evidence. A reasonable ladder is `65536/class -> 262144/class`; formal claims should use at least `1000000/class` and multiple seeds.
- ARX/SPECK has had completed `>=100000/class` results; SPN/PRESENT had not, as of the 2026-06-21 audit. Keep those evidence levels separate in progress reports.

## Memory And Handoff Hygiene

- When reading prior conversations, handoff summaries, `memory/`, `progress.md`, or result audits, persist important corrections through the `self-improvement` workflow instead of leaving them only in chat context.
- Use `.learnings/LEARNINGS.md` for detailed lessons and promote short durable rules to this `AGENTS.md` when they affect future experiment interpretation, remote workflow, or reporting language.
- After context-window loss or a new thread, re-check `.learnings/LEARNINGS.md` and this file before making claims about completed experiments or next steps.

## Research Execution Style

- Default project work should combine `skills/blockcipher-auto-research/SKILL.md` with the Karpathy-style coding discipline from `karpathy-guidelines`: use `blockcipher-auto-research` for the experiment loop and evidence gates, and use Karpathy-style restraint for implementation.
- Frame long-term Innovation 1 goals at the method level: build SPN-structure-adaptive neural distinguishers through better SPN-aware networks or data/feature representations. Do not turn the long-term goal itself into a specific run protocol, seed list, metric gate, or remote SOP; keep those details in `docs/experiments/` plans.
- For research experiments, define the question, identify the same-budget baseline, change one hypothesis at a time, run fixed-budget training, produce JSONL/CSV/SVG/gate artifacts, compare at the same evidence scale, and document keep/discard/crash/diagnostic status.
- Before launching or advancing a meaningful training experiment, proactively create or update the relevant `docs/experiments/` plan. Do this before smoke, remote launch, or handoff; do not wait for the user to ask, unless the user explicitly requests an ad hoc/local-only check.
- Keep training experiment matrices lean by default. For incremental model changes, compare the new candidate against the strongest current same-protocol anchor and only the necessary baseline/control rows, usually 2-3 models and rarely more than 4. Use larger 5-6 row matrices only for planned attribution studies, protocol audits, phase gates, or when a previous result suggests a control may invalidate the conclusion.
- When the user asks to advance a new training experiment, treat local smoke as a readiness gate rather than the endpoint. If smoke passes and the planned non-smoke run exists, continue automatically through scoped commit/push, remote launch from the pushed commit, and local tmux monitor/retrieval handoff, unless the user explicitly asked for local/smoke validation only.
- For code changes, first read the relevant modules, state uncertainty when evidence is incomplete, prefer the smallest boring implementation that satisfies the observable goal, avoid unnecessary abstractions or dependencies, and keep edits scoped to the task.
- Do not mix benchmark changes with model or feature changes unless the user explicitly asks to redesign the benchmark. Preserve validation data, labels, negative-sample definition, metric computation, and plan-alignment logic for comparable experiments.
- Validate with the narrowest meaningful command or test before reporting completion. If validation fails, debug the actual failure rather than guessing or broadening the change.

## Factual Reporting Discipline

- Before answering what the project currently uses or how a feature is implemented, inspect the relevant source, config, dependency files, logs, or artifacts first. Do not answer from expectation or memory for dependencies, plotting/rendering libraries, training protocols, remote scripts, artifact paths, experiment status, checkpoint selection, or metrics.
- If a current-state answer has not been verified in files or command output, either check it before reporting or explicitly label it as an assumption. When a prior statement is wrong, correct it plainly and distinguish the old false claim from the newly verified state.

## Documentation Organization

- New formal experiment plans, reproduction records, result analyses, and next-step execution plans belong under `docs/experiments/`.
- New research blueprints, literature syntheses, theory notes, and broad method proposals belong under `docs/research/`.
- Historical agent execution plans may remain under `docs/superpowers/plans/`, or be migrated to `docs/archive/agent-plans/` if the archive is reorganized. Do not create new current experiment plans under `docs/superpowers/`.
- Do not update both `docs/experiments/` and `docs/research/` mechanically for every run or code change. New meaningful experiment launches and completed meaningful results should update the relevant `docs/experiments/` record with run id, protocol, gate, metrics, status, and next action. Smoke tests, temporary debug runs, and local implementation checks do not need docs unless they change the evidence judgment or expose an important failure mode. Update `docs/research/` only when the research route, theory, literature synthesis, or method blueprint changes.
- When a meaningful remote experiment result completes and is retrieved, update the relevant `docs/experiments/` record automatically in the same turn before the final report. Do not ask whether to document it; ask only if ownership of the result document is genuinely ambiguous. Include run id, artifacts, gate, metrics, deltas, claim scope, and next action.

## Remote Windows GPU Rules

- Remote workstation alias: `lxy-a6000`. Project-owned remote workspace and run artifacts must stay under `G:\lxy`.
- Do not leave generated scripts, schedules, launcher logs, run directories, dataset caches, checkpoints, result archives, or temporary project files under `C:\Users`, Desktop, Documents, Downloads, Temp, or any non-`G:\lxy` path.
- Generated Windows schedule/launch commands must use `cmd.exe /c`, not `cmd.exe /k`, so completed training does not leave stuck command windows.
- Remote experiment roots should use `G:\lxy\blockcipher-structure-adaptive-nd` for the project and `G:\lxy\blockcipher-structure-adaptive-nd-runs` for run outputs.
- Referencing fixed external executables or pre-existing credentials is allowed only as a reference; project files and generated artifacts still must not be written outside `G:\lxy`.
- Remote training that generates datasets or derived features must not use pure in-memory one-shot generation at screen/scale sizes. Before launching a remote run at `65536/class` or above, verify disk-backed cache/progress/reuse exists for the exact data path: `features.npy`/`labels.npy` or equivalent chunk files, metadata, progress JSONL/logging, and parameter-matched reuse/resume behavior. New runners and new feature routes are not exempt; if they bypass `run_innovation_one_matrix.py`, they must implement an equivalent route-specific cache first.

## Remote Monitoring And Retrieval

- After launching or handing off a remote experiment, do not personally supervise it from the main thread. Start or verify a local tmux monitor/watcher that waits for completion artifacts, retrieves results automatically, writes logs/markers, and lets the main thread report from local artifacts.
- Delegate tmux monitoring loops to a sub-agent, watcher, or established monitor process. The main thread must not repeatedly poll `tmux ls`, `monitor.log`, or remote progress as a substitute for SSH polling; it should consume retrieved artifacts, handle explicit user status requests with a single bounded check, or continue non-monitoring research/implementation work.
- Do not SSH-poll from the main thread after a remote launch or handoff is recorded. Leave tmux/watchers/monitors to wait for artifacts.
- Resume interactive remote contact only when a local controlled dry-run says both `should_ssh=true` and `ssh_allowed=true`, or when local monitor health becomes unhealthy.
- Local tmux sessions are for local monitors/result retrieval. If a supported monitor is missing, restart the local tmux monitor only; do not touch remote training unless the controlled gate allows it.
- After a remote Windows `.cmd` launch returns, perform one bounded read-only confirmation that the expected `G:\lxy` run directory has `logs`, `*_started.marker`, readiness output, or progress JSONL before leaving the watcher to wait. Do not treat a successful SSH exit from `start /b` alone as proof that training started.
- Prefer verified Git result branches with complete `results_archive` gates. If the branch is missing or incomplete, use `scripts/monitor_remote_results.py` raw fallback retrieval only from `lxy-a6000:G:/lxy/blockcipher-structure-adaptive-nd-runs`; fallback outputs belong under `outputs/remote_results_incomplete/` and must include the raw retrieval notice.
- In status reports, separate `planned`, `running`, `completed remotely`, `fallback-retrieved`, `retrieved from verified result branch`, and `plan-aligned`. Do not collapse these states into "done".

## Evidence And Claim Gates

- No breakthrough claim unless the claim gate explicitly allows it.
- Current strict reference for PRESENT-80 r7 is Zhang/Wang 2022 Case2 `m=16`, accuracy `0.7205`; say clearly when this has not been reproduced locally.
- Strict negative samples must be encrypted random plaintexts. Random ciphertext negatives are ablation/control evidence only.
- Multi-query aggregation is application-level evidence, not raw single-sample SOTA evidence.
- Provisional route decisions, missing summaries, incomplete archives, or fallback-retrieved raw files are not enough for publication-style claims without explicit qualification.

## Verification And Workspace Hygiene

- Use `uv run pytest ...`, not bare `pytest`, for project test commands.
- Keep the project root clean of `tmp_*`; use `/tmp` for transient local markers and remove or ignore them when no longer needed.
- Before remote launches, audit generated scripts for `cmd.exe /c`, absence of `cmd.exe /k`, and no generated project paths outside `G:\lxy`.
- Before remote launches, audit generated-data paths: any dataset/feature generation for medium or larger remote jobs must write cache/progress under `G:\lxy` and must not wait until after full generation/training to emit the first durable artifact.
- When searching for Windows paths such as `G:\lxy` with `rg`, use fixed-string mode (`rg -F`) or proper escaping to avoid regex backslash parse errors.
- After any repository file modification, run the relevant verification, make a scoped git commit for the files just changed, and push when a remote is configured. This applies to code, config, scripts, tests, README/docs, `.learnings/`, `AGENTS.md`, generated project files, and memory-rule updates. Do not leave agent-authored completed work as an uncommitted pile.
- If no Git remote is configured, still make the local scoped commit and report that push was not possible because no remote exists.
- Do not include unrelated user or historical dirty files in that commit. If the worktree already has unrelated changes, commit only the task-scoped files and report the remaining dirty state separately.
- Remote experiments should normally launch from a GitHub-pushed commit. Avoid scp/dirty-overlay launches except for explicitly labeled emergency repair; if used, record that the run is dirty-overlay evidence and follow up with a proper commit/push immediately.
- Before using an existing remote clone as the source for a GitHub-pushed run, perform a read-only `git status --short --branch` gate. If the clone has historical local changes or cannot fast-forward cleanly, do not reset it without explicit approval; use a run-owned clean clone under `G:\lxy` instead.
