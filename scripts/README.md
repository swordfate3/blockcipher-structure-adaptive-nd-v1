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

Use `scripts/plan-residual-focus-repair` after a residual-focus gate, Pool 3
plan, or Pool 3 fixed-fusion evaluation reaches `fail`/`hold`. It maps
machine-readable `repair_hints` such as
`candidate_not_better_than_uniform_control`,
`label_shuffle_control_failed`, and `focus_candidate_metric_failed` into a
local-only repair plan. The plan explicitly forbids Pool 3 launch, 1M/class
scale-up, and candidate claims until the indicated residual objective,
attribution control, or feature-family repair branch is checked. Each repair
branch includes local `command_templates` and `implementation_notes`; these are
handoff aids only and do not SSH, launch remote jobs, or change validation
protocols.

Use `scripts/plan-state-token-residual-expert` to prepare the next PRESENT r8
state-token residual architecture route while the active residual-focus
262144/class job is still pending. It reads only the local residual-focus status
JSON and emits
`outputs/local_audits/i1_present_r8_state_token_residual_expert_plan.json`. When
the residual-focus gate is pending it allows only plan/smoke preparation; when
the gate fails or holds it routes back to residual repair; only after a pass does
it mark local state-token smoke and same-protocol controls as ready. It never
SSHes, launches remote jobs, scales to 1M/class, changes labels or negative
mode, or makes an SPN/PRESENT claim.

Use `scripts/fit-state-token-residual-expert` for the local feature-artifact
smoke behind that route. It fits `present_state_token_residual` on exported
`trail_position_stats` train features and scores held-out validation features,
then writes train/validation frozen-score artifacts plus a JSON report. This is
not a `scripts/train` matrix row: it consumes 3708D feature artifacts, not raw
beamstats pair matrices. It does not SSH, launch remote jobs, change labels,
change negative mode, or make a medium/formal SPN/PRESENT claim. Use
`--shuffle-token-coordinates` as the coordinate attribution control: it keeps
the same span feature values but permutes family/depth/word/cell token ids, so a
passing control means the current model has not proven true coordinate-layout
dependence.

Use `scripts/fit-state-token-residual-correction-expert` for the next local
state-token check: an additive logit correction over two frozen base score
artifacts, usually trail-position plus matched raw117. It consumes the same
3708D `trail_position_stats` feature artifacts, freezes the base logits, trains
only `base_logit_mean + state_token_correction`, and writes corrected
train/validation score artifacts plus a JSON report. Optional
`--residual-focus-fraction` up-weights train rows with high frozen-base residual
loss. It still does not SSH, launch remote jobs, change labels, change negative
mode, or make a medium/formal SPN/PRESENT claim. By default it zero-initializes
the final correction head so step 0 equals the frozen base; use
`--no-zero-init-correction-head` only as an initialization ablation.

Use `scripts/evaluate-residual-guided-diverse-pool` after
`scripts/plan-residual-guided-diverse-pool` reports
`residual_guided_diverse_pool_ready` and all five aligned validation score
artifacts exist: trail-position, raw117, selected residual-focus correction,
uniform residual control, and label-shuffle residual control. It runs only
fixed frozen-score fusions:
trail-position+raw117, trail-position+raw117+residual-focus,
trail-position+raw117+uniform-control, and
trail-position+raw117+label-shuffle-control. It returns `hold` when the
residual-focus fusion does not strictly beat the base/control comparisons, so
upper-level postprocess does not promote a control-failed Pool 3 diagnostic.
It does not fit stacking weights,
generate datasets, SSH, launch remote jobs, or make a formal SPN/PRESENT claim.

Use `scripts/residual-focus-status` for a local-only status summary of the
active residual-focus 262144/class route. It reads the action plan, gate report,
Pool 3 plan, Pool 3 fixed-fusion eval report, repair plan, local monitor log,
and locally retrieved progress/output artifacts to classify the state as running,
outputs-ready, gate-passed, gate-failed, pool-ready, waiting for Pool 3 score
artifacts, pool-control-held, repair-ready, or pool-evaluated. A repair plan is
used only when its `source_summary` matches the current gate/pool artifact, so
stale local repair files do not override a pending or running experiment. It
also emits a `progress_summary` from the latest local progress JSONL row,
including class-level and total-row cache fractions when those counters exist.
It also emits `progress_by_seed_split`, which keeps the latest progress row for
each `(seed, split)` pair so one active seed does not hide another seed's cache
state.
It does not SSH, launch, sync, run gates, or make a result claim.

Use `scripts/advance-residual-focus-results` as a local-only one-shot
postprocess after the residual-focus monitor has retrieved outputs. If outputs
are still missing it writes a pending report and exits without action. If the
planned outputs are all present and the gate is still pending, it runs
`scripts/gate-residual-focus-262k` through the local Python API, then runs
`scripts/plan-residual-guided-diverse-pool`. If that Pool 3 plan is ready and
all per-seed fixed-score artifacts are already present, it also runs the local
Pool 3 fixed-fusion evaluator and writes
`outputs/local_audits/i1_present_r8_residual_guided_diverse_pool_eval.json`.
If the Pool 3 plan is ready but any per-seed score artifact is missing, it
reports `wait_for_pool3_score_artifacts` and remains non-terminal so the local
watcher can continue after later retrieval/sync cycles.
If the residual-focus gate fails, or if the Pool 3 evaluator returns `hold`, it
also writes
`outputs/local_audits/i1_present_r8_residual_focus_repair_plan.json` through
`scripts/plan-residual-focus-repair`, so the next local action is a repair
branch rather than a blind ensemble or scale-up.
It does not SSH, sync, launch remote jobs, or fit ensemble weights.

Use `scripts/watch-residual-focus-results` when a local tmux watcher should keep
running the same local-only postprocess loop. It repeatedly calls
`advance-residual-focus-results`, writes an iteration report after every check,
and exits only when the local postprocess reaches a pass/hold terminal state or
when `--max-iterations` is reached. It is a local watcher; it does not replace
the remote-result retrieval monitor and it does not contact the remote host. Its
watch report includes whether the residual gate, Pool 3 planner, and Pool 3
fixed-fusion evaluator ran in the last iteration.

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
