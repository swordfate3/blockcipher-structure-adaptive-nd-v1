# Innovation 1 K1-BR: uKNIT 6-round last-two-round neural scale diagnostic

## Question

Does the frozen K1-U position-histogram residual retain an attributable uKNIT-BC
6-round signal when data rises directly to 262144 samples per class, or was the
near-chance K1-BO/K1-BP evidence primarily a small-data effect?

This is a user-requested data-scarcity diagnostic exception. It is not a
mechanical continuation of K1-U and is not formal or paper-scale evidence.

## Frozen protocol

- Run id: `i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730`
- Cipher and rounds: uKNIT-BC, 6 rounds
- Difference: `0x0000400000000000`, active cell 11, active bit role 1
- Input: 4 independent ciphertext pairs per sample
- Positive/negative labels: chosen plaintext difference versus independently
  sampled plaintexts encrypted under the same fixed key
- Training: 262144/class, 524288 total rows
- Cross-key validation: 65536/class, 131072 total rows
- Fixed keys: train `0x44...44`, validation `0x55...55`
- Seed: 3
- Optimizer: Adam, MSE, learning rate `1e-4`, weight decay `1e-5`
- Budget: 10 epochs, batch 64, restored best validation-AUC checkpoint
- Runtime structure: the exact uKNIT descriptor, last-two-round window
  `runtime_round_start=4`, `runtime_rounds=2`
- Execution: remote physical GPU 1, disk-backed chunked cache under `G:\lxy`

Remote launch repair on 2026-07-30: two clean SSH clone attempts were closed on
GitHub port 22 before checkout. The run-owned clone therefore uses the public
repository's HTTPS transport for read-only source retrieval, then checks out
and verifies the exact GitHub-published commit. This does not transfer a local
source overlay or weaken the source-revision gate. Result-branch push failure
still falls back to raw retrieval from the completed `G:\lxy` run root.

The inherited profile name `uknit64_k1q_cell11_r5` is retained only because it
identifies the frozen difference. The actual cipher round count is 6 and the
explicit model option binds the difference value, so the profile suffix must
not be interpreted as a 5-round dataset.

## Same-budget matrix

1. Exact position histogram residual: candidate.
2. Wrong-S-box position histogram residual: equal-geometry semantic control.
3. Position-invariant histogram residual: tests whether native cell positions
   add value beyond the compact histogram route.

Only the cipher round count and corresponding last-two-round window change
relative to the K1-U architecture/protocol anchor. Data scale is the explicit
diagnostic exception requested for this run.

## Gate

Let `candidate = max(exact AUC, invariant AUC)` and
`attribution margin = candidate - wrong-S-box AUC`.

- Strong attributed candidate: candidate >= 0.55 and margin >= 0.01.
- Weak attributed signal: candidate >= 0.51 and margin >= 0.005.
- Weak but unattributed: candidate >= 0.51 but margin < 0.005.
- No supported positive signal: candidate < 0.51.

A one-seed result cannot support a formal, attack, SOTA, breakthrough, or route
ceiling claim. If the signal is attributed, the next experiment is the same
matrix at seed 4, not 1M/class. If it is unattributed or below 0.51, hold scale
and return to representation/difference diagnosis; do not add epochs, pairs,
capacity, differences, or a deeper runtime window in the same adjudication.

## Required artifacts

The remote run must emit durable cache progress, exactly two completed caches,
four control cache reuses, three checkpoints, three results, validation, gate,
summary, history, archive manifests, and hashes. Retrieval must revalidate and
re-adjudicate locally, refresh the recent-results index, then render and inspect
the Chinese result figure with `visual-qa-redraw` before the figure is complete.

## Recommended next action

Launch this exact single-seed remote diagnostic from a GitHub-verified commit.
Its only purpose is to decide whether a larger-data 6-round signal exists and
is attributable to the correct structural semantics. Do not call it formal
training or compare it directly to paper-scale neural distinguishers.

## Launch record

- Status: completed remotely; fallback-retrieved raw results; local strict gate
  remains invalid because one restored-checkpoint metric exceeded the frozen
  replay tolerance.
- Launch time: 2026-07-30 14:46 Asia/Shanghai.
- Source commit: `d967b02c3221365638189353107eb5b7efb6419f`, verified
  equal to GitHub `main` before launch.
- Remote task: `I1_UKNIT_R6_K1BR_S3_GPU1` on physical GPU 1.
- Remote run root:
  `G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_uknit_r6_last2_neural_scale_k1br_262144_seed3_20260730`.
- Local monitor: `i1_uknit_r6_k1br_262k_s3_monitor`.
- Durable start evidence: run-owned source clone at the exact commit, scheduled
  task in `Running` state, and source/GPU/torch logs under the run root.
- Estimated duration: about 6--8 hours, extrapolated from the retrieved K1-U
  65536/class run (six rows in 3.31 hours) while accounting for three rows,
  four-times larger training data, and two-times larger validation data.

The monitor owns completion waiting, retrieval, local validation, adjudication,
plot generation, and recent-results indexing. After retrieval, run
`visual-qa-redraw` on the rendered result before marking the figure complete,
then append metrics, deltas, decision, claim scope, and the evidence-backed next
action to this record.

## Retrieved result and adjudication

The remote GPU completed all three ten-epoch rows and wrote three result rows.
Plan/result validation passed `3/3`, the progress log contains `run_done`, both
disk caches completed, all four control reuses occurred, and the exact source
commit remained `d967b02c3221365638189353107eb5b7efb6419f`.

The remote postprocessor then failed before archive creation because the gate
wrapper could not import `blockcipher_nd`. The raw result and logs were therefore
retrieved from the completed `G:\lxy` run root into
`outputs/remote_results_incomplete/` and are not a verified result-branch
archive.

| Condition | restored-checkpoint AUC |
|---|---:|
| Correct S-box plus native positions | 0.500422927 |
| Wrong-S-box control | 0.503789498 |
| Position-invariant control | 0.500420902 |

The best intended candidate is the exact-position model at `0.500422927`, below
the preregistered weak floor `0.51`. Its margin over the wrong-S-box control is
`-0.003366571`. Thus this single-seed `262144/class` diagnostic does not observe
a credible six-round signal and does not authorize seed4 or `1M/class` scale.

The local strict gate remains `invalid`, rather than a clean research hold,
because the position-invariant row's restored evaluation AUC differs from its
recorded best-checkpoint AUC by about `1.54e-5`, above the frozen `1e-6`
tolerance. That artifact/protocol mismatch does not create a positive signal,
but it must be resolved before a plan-aligned closure is recorded.

Recommended next action: repair only the remote postprocessing import path and
audit the existing invariant checkpoint's deterministic replay without
retraining. If the unchanged checkpoint passes the frozen protocol and the
three AUCs remain below `0.51`, close K1-BR as a no-supported-positive-signal
larger diagnostic. Do not add another seed, more data, more epochs, more pairs,
another difference, or a deeper runtime window before that audit.

The final Chinese SVG was rendered to `2016x1098` pixels and passed
`visual-qa-redraw`: no text overlap, clipping, missing glyphs, ambiguous title,
misleading scale, or unreadable marker separation was observed.
