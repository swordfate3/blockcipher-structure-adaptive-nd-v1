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

## Remote Windows GPU Rules

- Remote workstation alias: `lxy-a6000`. Project-owned remote workspace and run artifacts must stay under `G:\lxy`.
- Do not leave generated scripts, schedules, launcher logs, run directories, dataset caches, checkpoints, result archives, or temporary project files under `C:\Users`, Desktop, Documents, Downloads, Temp, or any non-`G:\lxy` path.
- Generated Windows schedule/launch commands must use `cmd.exe /c`, not `cmd.exe /k`, so completed training does not leave stuck command windows.
- Remote experiment roots should use `G:\lxy\blockcipher-structure-adaptive-nd` for the project and `G:\lxy\blockcipher-structure-adaptive-nd-runs` for run outputs.
- Referencing fixed external executables or pre-existing credentials is allowed only as a reference; project files and generated artifacts still must not be written outside `G:\lxy`.
- Remote training that generates datasets or derived features must not use pure in-memory one-shot generation at screen/scale sizes. Before launching a remote run at `65536/class` or above, verify disk-backed cache/progress/reuse exists for the exact data path: `features.npy`/`labels.npy` or equivalent chunk files, metadata, progress JSONL/logging, and parameter-matched reuse/resume behavior. New runners and new feature routes are not exempt; if they bypass `run_innovation_one_matrix.py`, they must implement an equivalent route-specific cache first.

## Remote Monitoring And Retrieval

- Do not SSH-poll from the main thread after a remote launch or handoff is recorded. Leave tmux/watchers/monitors to wait for artifacts.
- Resume interactive remote contact only when a local controlled dry-run says both `should_ssh=true` and `ssh_allowed=true`, or when local monitor health becomes unhealthy.
- Local tmux sessions are for local monitors/result retrieval. If a supported monitor is missing, restart the local tmux monitor only; do not touch remote training unless the controlled gate allows it.
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
- After completing code, config, generated-script, memory-rule, or test edits, run the relevant verification, make a scoped git commit, and push it. Do not leave agent-authored completed work as an uncommitted pile.
- Do not include unrelated user or historical dirty files in that commit. If the worktree already has unrelated changes, commit only the task-scoped files and report the remaining dirty state separately.
- Remote experiments should normally launch from a GitHub-pushed commit. Avoid scp/dirty-overlay launches except for explicitly labeled emergency repair; if used, record that the run is dirty-overlay evidence and follow up with a proper commit/push immediately.
