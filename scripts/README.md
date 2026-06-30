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
`--postprocess-kind ddt_graph`, `--postprocess-kind pairset_aggregation`, or
`--postprocess-kind candidate_trail` so the emitted command calls the matching
postprocess entrypoint and expected row count.

Use `scripts/check-remote-readiness` before launching a remote config. It checks
the local JSON/CSV invariants only; it does not generate launch scripts, SSH, or
touch the remote workstation. Route-specific entries in `checked_invariants`
are included only when they apply, for example `candidate_trail_protocol_lock`
for candidate-trail configs and `pairset_aggregation_stage_lock` for pair-set
aggregation configs.

Use `scripts/plan-next-action` after a postprocess summary JSON is written. It
reads `next_action`, runs local readiness checks for any declared remote config,
and emits a structured branch/readiness report. It does not launch, SSH, or
touch the remote workstation.
DDT graph postprocess also writes `<run_id>_next_action_readiness.json`
directly, so watcher-managed DDT runs leave both the gate decision and the
next-branch readiness artifact in the run output directory.

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
