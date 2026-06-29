# Scripts

Scripts are thin human-facing entrypoints that call package CLI modules under
`src/blockcipher_nd/cli/`.

Do not put training loops, dataset generation, feature extraction, cache logic,
or result validation implementations here.

Use `scripts/gate-invp-result` after a retrieved InvP-only 1M JSONL result to
write a deterministic seed1-vs-DDT branch report.

Use `scripts/postprocess-invp-result` to run the full local post-retrieval chain:
plan alignment validation, curve/history export, branch gating, and optional
experiment-plan Markdown update via `--update-plan-doc`.

Use `scripts/monitor-health` for a bounded local health check of a remote-result
monitor directory. It reads local artifacts and optionally checks one tmux
session once; it does not SSH-poll or supervise a remote run.
