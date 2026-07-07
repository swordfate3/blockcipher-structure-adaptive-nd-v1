# Scripts

Scripts are thin human-facing entrypoints that call package CLI modules under
`src/blockcipher_nd/cli/`.

Do not put training loops, dataset generation, feature extraction, cache logic,
or result validation implementations here.

Use `scripts/gate-invp-result` after a retrieved InvP-only 1M JSONL result to
write a deterministic seed1-vs-DDT branch report.

Use `scripts/gate-invp-attribution-controls` after retrieved 1M DeltaC-only and
shuffled-P control results to decide whether the completed InvP-only seed0/seed1
advantage remains attributable to true SPN/InvP/P-layer alignment.

Use `scripts/postprocess-invp-attribution-controls` to run the full local
post-retrieval chain for attribution-control runs: plan alignment validation,
curve/history export, attribution gate, summary JSON/Markdown, and optional
experiment-plan Markdown updates via one or more `--update-plan-doc` arguments.

Use `scripts/postprocess-invp-result` to run the full local post-retrieval chain:
plan alignment validation, curve/history export, branch gating, and optional
experiment-plan Markdown update via `--update-plan-doc`.

Use `scripts/monitor-health` for a bounded local health check of a remote-result
monitor directory. It reads local artifacts and optionally checks one tmux
session once; it does not SSH-poll or supervise a remote run.
If the tmux socket check itself fails in a restricted local environment, read
`tmux_interpretation` together with `heartbeat` and
`needs_main_thread_intervention`; a fresh heartbeat with status `running` means
the socket check error alone is not evidence that the watcher stopped.
It emits `postprocess_command` only for `result_ready`, which requires a
retrieved JSONL with at least one non-empty row. `completed_missing_results`
means a done marker exists but the JSONL is missing; `results_empty` means the
JSONL exists but has no non-empty rows. Neither state should be postprocessed.
`remote_artifacts_missing` means repeated local sync attempts could not find
the remote run's `logs/` or `results/` directories; this usually requires a
bounded launch-path check before assuming training is still running.
`postprocessed` means the watcher already ran the route-specific postprocess
successfully, so the next action is to inspect gate/readiness artifacts, update
or commit docs if needed, and branch from the recorded decision rather than
rerunning postprocess. `postprocess_failed` means inspect the monitor
`postprocess_stderr.log` before branching.
Use `--postprocess-kind invp`, `--postprocess-kind invp_attribution`,
`--postprocess-kind ddt_graph`, `--postprocess-kind topology_aware`,
`--postprocess-kind pairset_aggregation`, or `--postprocess-kind
candidate_trail`, `--postprocess-kind transition_spectrum`, or
`--postprocess-kind trail_family`, `--postprocess-kind active_auxiliary`, or
`--postprocess-kind sbox_prior`, `--postprocess-kind
integral_inverse_feature`, `--postprocess-kind r9_weak_probe`, or
`--postprocess-kind r8_pairset_1m` so the emitted command calls the matching
postprocess entrypoint and expected row count.
For feature-cache jobs, `progress_summary` includes cache progress percentages,
row rates when timestamps are available, ETA when computable, and
`cache_rows_remaining` / `cache_class_rows_remaining` for direct status reports.

Use `scripts/watch-high-round` inside a local tmux watcher when high-round
SPN/PRESENT runs have already been launched and handed off. It loops over
`scripts/advance-high-round`, reads only local artifacts, never SSH-polls or
launches remote training, and by default lets route-specific postprocess update
the relevant `docs/experiments/` plan once a result becomes ready. Use
`--max-iterations 1` for a bounded one-shot check.

Use `scripts/check-remote-readiness` before launching a remote config. It checks
the local JSON/CSV invariants only; it does not generate launch scripts, SSH, or
touch the remote workstation. Route-specific entries in `checked_invariants`
are included only when they apply, for example `candidate_trail_protocol_lock`
for candidate-trail configs and `pairset_aggregation_stage_lock` for pair-set
aggregation configs.

Use `scripts/check-launch-source` after local readiness passes and before any
remote launch. It reads `git status --short --branch` only and fails when the
worktree is dirty, lacks an upstream, is behind upstream, or has unpushed
commits. A passing remote-readiness report is therefore not enough to launch:
remote training still requires a clean pushed source commit, generated launch
artifacts, GPU/readiness confirmation, and monitor handoff.

Use `scripts/plan-residual-focus-remote-package` after
`scripts/plan-residual-focus-262k`, `scripts/gate-residual-focus-262k`, and
`scripts/check-launch-source` have written their reports. It translates the
residual-focus action plan into a prepared Windows `.cmd` and local monitor
`.sh` package under `configs/remote/generated/`, while preserving the source
gate as a launch blocker when commits are still unpushed. It does not SSH,
launch, or mark the residual-focus 262144/class gate complete.
The generated `launch_*.sh` wrapper is also gated: it reads the package JSON and
exits with `launch_blocked.marker` before SSH when `launch_allowed` is not true.

Use `scripts/plan-residual-guided-diverse-pool` after
`scripts/gate-residual-focus-262k` has written its report. It is a local-only
guard for the r8 Pool 3 diverse expert route: when the residual-focus gate is
pending it waits, when the gate fails it holds the pool and asks for residual
repair, and only when the gate keeps `focus05` or `focus10` does it mark the
residual-guided fixed-fusion pool as ready. It does not launch remote training,
fit ensemble weights, or make an ensemble claim.

Use `scripts/audit-integral-parity-signal` for local-only audits of
`plaintext_integral_nibble` PRESENT rows before interpreting integral-route
neural metrics. It generates a small deterministic dataset from a plan row and
tests whether the classes are already separable by the hand-coded
pair-xor-parity statistic across the `pairs_per_sample` set. A near-perfect
audit means the run should be interpreted first as an integral/multiset
data-construction signal, not as neural architecture evidence by itself.

Use `scripts/plan-next-action` after a postprocess summary JSON is written. It
reads `next_action`, runs local readiness checks for any declared remote config,
and emits a structured branch/readiness report. It does not launch, SSH, or
touch the remote workstation.
Use `scripts/arbitrate-next-actions` when more than one postprocess summary has
a launchable high-round follow-up. It calls `plan-next-action` for each summary,
keeps the decision local-only, and selects one branch by the documented
high-round priority policy so r8 seed confirmation, r9 seed confirmation,
r9 1M scale-up, curriculum, and difference-screen branches are not launched
blindly in parallel.
DDT graph, topology-aware, r9 weak-probe, and r8 pair-set 1M postprocess also write
`<run_id>_next_action_readiness.json` directly, so watcher-managed DDT or
topology-aware/high-round runs leave both the gate decision and the
next-branch readiness artifact in the run output directory.
Candidate-trail, bit-transition-spectrum, and trail-family postprocess also
write this readiness artifact, including an implementation checklist when the
next branch needs a new plan/config before any remote launch.
These readiness artifacts check both the remote config invariants and the
generated launcher/monitor scripts, so a missing `run_*.cmd` or `monitor_*.sh`
blocks the next remote handoff before launch.
Candidate-trail and bit-transition-spectrum postprocess outputs are therefore
safe handoff points for `scripts/plan-next-action`; trail-family follows the
same readiness-artifact convention.

Use `scripts/summarize-spn-evidence` for a current local Innovation 1 SPN route
overview. It scans retrieved postprocess summaries, reports the strongest
route-level evidence, marks superseded branches, and emits an
`active_recommendation`. For high-round PRESENT work it also tracks the active
r9 weak-probe and r8 pair-set 1M watcher artifacts, emits route-specific
postprocess commands when results are ready, and recommends
`scripts/arbitrate-next-actions` when multiple high-round summaries are
available. After those high-round gates free the branch, it also recognizes the
prepared r9 curriculum, r9 difference-screen, r8 integral/inverse-feature,
pair-mixer, and pair-evidence-pooling follow-up runs, keeping their
running/result-ready state in the same local-only monitor/postprocess workflow.
While candidate-trail is still running, that
recommendation remains `wait_for_candidate_trail_result`; after retrieved
candidate-trail or bit-transition-spectrum results, it follows the newest
gated branch and keeps transition-spectrum decisions ahead of older
candidate-trail decisions. Running or ready candidate-trail recommendations
also include `main_thread_policy`, which lists allowed local actions and the
remote launches or claims that remain forbidden until the gate has produced a
validated postprocess decision.
While candidate-trail is waiting or ready-for-postprocess, the recommendation
also includes `conditional_followup` for the prepared seed1 confirmation or
variance-check config. This field reports the local readiness gate as
`readiness_pass`, but it keeps `should_launch_now = false`; launch remains
forbidden until seed0 has been retrieved, validated, plan-aligned, and
postprocessed with `support_candidate_trail_route` or
`weak_candidate_trail_signal`.

Use `scripts/evaluate-pairset-aggregation` to evaluate a frozen single-pair
checkpoint as an independent score-aggregation control over multi-pair samples.
It does not train or update the scorer; it slices each pair-set sample into
single-pair views, aggregates logits/log-odds, and writes a JSON summary.

Use `scripts/gate-pairset-aggregation` after learned pair-set and frozen
single-pair aggregation artifacts are available. It compares AUC margins against
the predeclared InvP anchor and frozen aggregation control, then emits the
continue/weak/stop decision for the pair-set route.

Use `scripts/postprocess-pairset-aggregation` after learned pair-set JSONL and
frozen aggregation summary artifacts are available. It validates learned results
against the plan, exports learned training curves/history, runs the pair-set
aggregation gate, writes summary artifacts, and can update the experiment plan.

Use `scripts/spn-candidate-evidence-matrix` for candidate-trail JSON plans that
contain `common` plus `rows`. It writes one JSONL result per row so the
candidate-trail gate can compare an external InvP anchor against
`candidate_trail_consistency_linear`, `candidate_trail_consistency_mlp`, and
the shuffled-cell control without hand-concatenating result files.
