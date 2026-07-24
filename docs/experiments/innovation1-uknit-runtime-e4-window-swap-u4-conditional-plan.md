# Innovation 1 uKNIT Runtime-E4 Same-Checkpoint Window-Swap U4 Plan

Date: 2026-07-25

## Status

```text
phase                  = implemented and verified conditional attribution audit
training               = forbidden
execution_authorized   = no
blocking_dependency    = completed, validated, visually checked two-seed U3 pass
source_checkpoint      = each seed's U3 candidate best checkpoint only
execution              = local CPU, 8 inference rows
implementation_tests   = 239 scoped regressions passed, including 6 direct U4 tests
```

## Research Question

If U3 reports that the recurrent heterogeneous-window candidate beats four
separately trained controls, does the candidate's own frozen checkpoint still
depend on the exact two-transition runtime window at inference time?

U4 isolates this question from training variance. It loads one U3 candidate
checkpoint per seed and one frozen validation cache per seed, then changes only
the runtime structure or relation mode. No optimizer, gradient, data generation
or new model initialization is allowed.

## Required U3 Dependency

U4 is authorized only when all of the following evidence exists and replays:

```text
U3 gate status                  = pass
U3 decision                     = innovation1_runtime_spn_recurrent_window_two_seed_supported
U3 protocol checks              = all true
U3 research checks              = all true
U3 result rows                  = 10
U3 plan validation              = pass, expected_rows=10, result_rows=10, errors=[]
U3 candidate checkpoints        = selected best, one for each seed
U3 visual QA marker             = visual_qa_passed.marker
U3 frozen plan SHA256           = 060805c3e1e6793aa11b3e9758ddef738d646c77df596150032c8486b7bbd87f
```

The U4 launcher must recompute the U3 gate from `results.jsonl` and require it
to equal the persisted gate. A copied pass string, partial result panel, missing
checkpoint, failed plan validation or missing visual marker must fail closed.

## Frozen Same-Checkpoint Panel

For each seed, load only the U3 `candidate` checkpoint trained with
`round_window_mode=recurrent_window` and the correct full window:

| Condition | Runtime window | Relation mode | Changed field |
| --- | --- | --- | --- |
| `full_correct` | correct heterogeneous S3/L3, S4/L4 | `true` | reference |
| `repeat_last` | S4/L4 repeated twice | `true` | ordered window content |
| `corrupted` | deterministic seed `20260724` corrupted window | `true` | linear topology |
| `no_topology` | correct descriptor | `independent` | cross-cell linear relation |

The panel contains exactly:

```text
2 seeds x 4 conditions = 8 rows
training_performed     = false
strict state load      = true
checkpoint             = same SHA256 within each seed
validation features    = same SHA256 within each seed
validation labels      = same SHA256 within each seed
```

## Frozen Source Protocol

```text
cipher                  = uKNIT-BC prefix r5
runtime_round_start     = 3
runtime_rounds          = 2
processor_steps         = 2
input difference        = 0x40
pairs_per_sample        = 4
validation              = 1024/class, 2048 total rows
input bits              = 512
seeds                   = 0, 1
parameter count         = 442466
S-box context           = edge_gate
cell input              = state_triplet
negative definition     = encrypted random plaintext pairs
execution               = local CPU inference only
```

The `full_correct` AUC must reproduce the corresponding persisted U3 candidate
AUC within `1e-12`. This is both a source-checkpoint and validation-cache replay
check; a mismatch is a protocol failure, not a negative research result.

## Preregistered Gate

Protocol pass requires the complete two-seed panel, exact source replay, strict
checkpoint loading, frozen geometry, valid SHA256 provenance, no training, and
the expected structure fingerprints:

- `full_correct` and `no_topology` share the correct heterogeneous window;
- `repeat_last` is homogeneous, ends in the same final transition as the full
  window, and has a different ordered-window hash;
- `corrupted` has a deterministic window hash distinct from the full window;
- all four interventions are seed-invariant;
- the two seed checkpoints and validation feature caches are independently
  identified.

Research pass requires all six margins and all six functional-change checks:

```text
for seed0 and seed1:
  full_correct - repeat_last AUC >= +0.005
  full_correct - corrupted AUC   >= +0.005
  full_correct - no_topology AUC >= +0.005

  max probability delta(full_correct, repeat_last) > 1e-6
  max probability delta(full_correct, corrupted)   > 1e-6
  max probability delta(full_correct, no_topology) > 1e-6
```

The probability checks prove that each intervention reaches the frozen
checkpoint's computation. They do not replace the AUC margins.

## Decisions

```text
pass:
  decision    = innovation1_runtime_spn_window_same_checkpoint_attribution_supported
  next_action = preregister X3: freeze the U3 recurrent uKNIT backbone, train
                only the SKINNY output head on the exact X2 target caches and
                budget, and compare full-window/repeat-last/corrupted/random
                source controls; do not scale uKNIT samples or epochs

hold:
  decision    = innovation1_runtime_spn_window_same_checkpoint_attribution_not_supported
  next_action = stop recurrent-window scale-up and redesign the failed local
                structure interaction against the frozen U3/U4 anchors

protocol fail:
  decision    = innovation1_runtime_spn_window_same_checkpoint_protocol_invalid
  next_action = repair evidence loading or replay only; do not change thresholds
```

## Conditional Execution

After U3 is complete and the source gate authorizes execution:

```bash
RUN_ID=i1_uknit64_runtime_e4_window_swap_u4_20260725

UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/audit-runtime-spn-recurrent-window-counterfactual \
  --run-id "${RUN_ID}" \
  --u3-root outputs/local_diagnostic/i1_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1_20260725 \
  --plan configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_recurrent_window_r5_2048_seed0_seed1.csv \
  --output-root "outputs/local_audits/${RUN_ID}" \
  --device cpu \
  --batch-size 256
```

Required outputs are `source_gate.json`, `results.jsonl`, `validation.json`,
`gate.json`, `summary.json`, `progress.jsonl` and `curves.svg`. After rendering,
run `visual-qa-redraw`; only then create `visual_qa_passed.marker` and refresh
`outputs/00_RECENT_RESULTS.md` plus its JSON companion.

The local-only conditional successor is:

```text
configs/remote/generated/monitor_i1_uknit_u4_after_u3_20260725.sh
```

It waits for the completed local U3 evidence, contains no SSH/SCP/Windows/GPU
operation, rechecks the exact pushed U4 implementation before execution,
refreshes the result index, and stops at the visual-QA marker. A U3 hold,
authorization failure, training failure, validation failure or visual failure
records an explicit stop marker instead of trying to rescue the route.

## Conditional Post-U4 X3 Contract

X3 remains unimplemented and unauthorized unless U4 passes both seeds and its
own rendered visual gate. Its research question is whether the heterogeneous-
window representation learned on uKNIT transfers to SKINNY better than the
already completed homogeneous GIFT-source X2 anchor when only the target
classifier is adapted.

The same-budget anchor is X2, not a new scratch run:

```text
seed0 X2 GIFT-source candidate AUC = 0.552013397
seed1 X2 GIFT-source candidate AUC = 0.598568439
seed0 full-target SKINNY AUC       = 0.612733364
seed1 full-target SKINNY AUC       = 0.614543915
```

Freeze five roles per seed:

| Role | Frozen source backbone | SKINNY target structure | Purpose |
| --- | --- | --- | --- |
| candidate | U3 full recurrent-window best checkpoint | correct | X3 hypothesis |
| repeat-source | U3 repeat-last best checkpoint | correct | earlier-window source control |
| corrupted-source | U3 corrupted best checkpoint | correct | source-topology control |
| random-source | deterministic random frozen backbone | correct | learned-source control |
| corrupted-target | U3 full recurrent checkpoint | corrupted | target-topology control |

The single hypothesis variable relative to X2 is the source representation;
reuse X2's exact SKINNY r7 `0x2000` train/validation caches, target keys,
four-pair `512-bit` inputs, encrypted-random-plaintext negatives, deterministic
classifier initialization and training code:

```text
seeds                    = 0, 1
train                    = 2048/class, 4096 total per seed
validation               = 1024/class, 2048 total per seed
trainable parameters     = classifier only, 198401
frozen total geometry    = 442466
epochs                   = 5
batch                    = 256
optimizer                = Adam, lr 1e-4, weight decay 1e-5
loss                     = MSE
checkpoint               = best validation AUC
execution                = local CPU diagnostic
```

Both seeds must satisfy all preregistered advance checks:

```text
candidate AUC >= 0.55
candidate - repeat-source AUC >= +0.005
candidate - corrupted-source AUC >= +0.005
candidate - random-source AUC >= +0.005
candidate - corrupted-target AUC >= +0.005
candidate - same-seed X2 GIFT-source candidate AUC >= +0.005
```

If any check misses, retain U4 only as within-uKNIT window-attribution evidence
and stop recurrent cross-cipher adaptation. Even an X3 pass remains a local
mechanism result; it does not authorize medium scale, a remote run or a
universal-SPN claim.

## Blocked Routes

- Do not execute U4 from a U3 hold, partial panel or copied decision string.
- Do not train separate U4 control models or regenerate validation samples.
- Do not tune the `+0.005` or `1e-6` thresholds after reading U4 results.
- Do not increase uKNIT samples, epochs or pair count as a rescue.
- Do not add DDT, trail, related-key or partial-decryption features.
- Do not claim an attack, SOTA, breakthrough or universal-SPN transfer from U4.

## Recommended Next Action

Keep U4 blocked while RTG3 and U3 are unresolved. Once the verified U3 pass
exists, run this unchanged eight-row audit. A pass advances to a separately
preregistered X3 recurrent-source head-adaptation experiment. X3 must reuse the
completed X2 SKINNY train/validation caches, deterministic classifier
initialization, five-epoch head-only budget and full-target anchor. Its single
new hypothesis is the source backbone: U3 full recurrent uKNIT versus the
repeat-last, corrupted and random-source controls. This does not repeat X1
zero-step transfer or X2 GIFT-source adaptation. A U4 hold closes the recurrent-
window scale/transfer route and returns to a local interaction redesign.
