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
Use `--postprocess-kind invp`, `--postprocess-kind invp_attribution`, or
`--postprocess-kind ddt_graph` so the emitted command calls the matching
postprocess entrypoint and expected row count.

Use `scripts/check-remote-readiness` before launching a remote config. It checks
the local JSON/CSV invariants only; it does not generate launch scripts, SSH, or
touch the remote workstation.

Use `scripts/evaluate-pairset-aggregation` to evaluate a frozen single-pair
checkpoint as an independent score-aggregation control over multi-pair samples.
It does not train or update the scorer; it slices each pair-set sample into
single-pair views, aggregates logits/log-odds, and writes a JSON summary.

Use `scripts/gate-pairset-aggregation` after learned pair-set and frozen
single-pair aggregation artifacts are available. It compares AUC margins against
the predeclared InvP anchor and frozen aggregation control, then emits the
continue/weak/stop decision for the pair-set route.
