# Scripts

Scripts are thin human-facing entrypoints that call package CLI modules under
`src/blockcipher_nd/cli/`.

Do not put training loops, dataset generation, feature extraction, cache logic,
or result validation implementations here.

Use `scripts/gate-invp-result` after a retrieved InvP-only 1M JSONL result to
write a deterministic seed1-vs-DDT branch report.

Use `scripts/postprocess-invp-result` to run the full local post-retrieval chain:
plan alignment validation, curve/history export, and branch gating.
