# Innovation 1 uKNIT-Family Midori64 S-box Transition Residual K1-AK

**Date:** 2026-07-29
**Status:** completed / held after valid local fixed-budget neural diagnostic
**Execution:** local CPU only; no remote scale

## 1. Research Question

K1-AI established usable Midori64 r4 signal, and K1-AJ then showed that the
completed correct checkpoint causally depends on the supplied diffusion layer.
The exact S-box intervention changed predictions and passed three of four fresh
semantic margins, but seed6 same-key fresh reached only `+0.003480` against the
preregistered `+0.005` threshold.

The current K1-AA readout computes five stage/value histograms and immediately
averages over all cells. That retains global value frequencies but discards the
joint question relevant to a nonlinear layer: which 4-bit input difference was
mapped to which 4-bit output difference at each S-box transition?

K1-AK asks:

> Does replacing only that compact value-histogram readout with a shared
> per-cell S-box input/output transition readout produce stable correct-S-box
> discrimination without losing the K1-AI signal or diffusion attribution?

## 2. Frozen Evidence And Same-Budget Anchor

K1-AI is the same-budget neural anchor:

```text
run_id = i1_uknit_family_midori64_neural_attribution_k1ai_2048_seed6_seed7_20260729
status = hold
decision = innovation1_uknit_family_midori64_k1ai_signal_learned_structure_attribution_not_supported
```

K1-AJ is the causal mechanism source:

```text
run_id = i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729
status = hold
decision = innovation1_uknit_family_midori64_k1aj_diffusion_causal_sbox_discrimination_failed
```

Both sources, their validations, exact artifact digests and all six frozen
K1-AH cell8 datasets must pass before optimizer steps are authorized.

## 3. Single Experimental Variable

Retain the K1-AA exact-composition trunk, GF(2) edge residual, classifier,
pair aggregation, hidden width and bounded residual fusion. Replace only:

```text
old: five stage × 16-value histograms -> average cells -> shared encoder
new: two S-box stages × cell × (16 input differences × 16 output differences)
     -> shared per-cell encoder -> average cells -> bounded residual
```

The new branch never receives cipher identity or absolute cell/bit identity.
The same encoder is applied to every cell before invariant mean pooling, so its
parameter shape is independent of block width and cell count. It derives the
transition values from the runtime S-box composition; it does not ingest DDT,
trail, attack-score or hand-ranked position features.

Use a `256 -> 20` shared transition encoder and retain the K1-AA
`16`-virtual-slot `40 -> 128` projection geometry. Candidate trainable
parameters may exceed K1-AA only by the encoder replacement and must remain at
most `1.025 × 214316`.

## 4. Frozen Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | Midori64 r4 |
| Difference | cell8 role1, `0x0000000400000000` |
| Seeds | `6`, `7` |
| Pairs per sample | `4` |
| Train | `2048/class`, `4096` total rows per seed |
| Same-key fresh | `1024/class`, `2048` total rows per seed |
| Cross-key fresh | `1024/class`, `2048` total rows per seed |
| Negative definition | encrypted random plaintexts |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam, `1e-4`, weight decay `1e-5` |
| Checkpoint | best cross-key validation AUC |
| Execution | local CPU |

Train exactly four K1-AK conditions per seed:

| Condition | Runtime S-box | Runtime linear layer |
|---|---|---|
| `correct_structure` | correct | correct |
| `wrong_sbox` | deterministic wrong table | correct |
| `corrupted_linear` | correct | deterministic corruption |
| `no_structure` | disabled | identity |

K1-AI correct-structure AUC is read as a frozen source anchor rather than
retrained, keeping the new matrix to eight training rows.

## 5. Protocol Gate

Require exact K1-AI/K1-AJ source hashes and decisions, exact six dataset
digests, eight plan rows, sixteen train/validation cache reuses, no cache
generation, ten complete epochs, best-checkpoint restoration, `24` fresh-panel
evaluation rows, finite metrics and identical state geometry across the four
K1-AK controls.

The candidate must have no cipher-identity or absolute-cell parameters. Its
transition histograms must be normalized, differ under the fixed wrong S-box,
remain invariant under a consistent cell relabeling, and use no more than
`1.025 ×` K1-AA parameters. Any failed protocol check makes the run invalid.

## 6. Research Gate

Apply every threshold separately to seed6/7 and both fresh splits:

```text
candidate correct AUC                         >= 0.550
candidate correct - K1-AI correct             >= -0.010
candidate correct - wrong S-box               >= +0.005
candidate correct - corrupted linear          >= +0.005
candidate correct - no structure              >= +0.010
```

Training-seen rows are diagnostic only. Do not average seeds or splits to hide
a failure.

## 7. Decisions And Required Next Action

- **All gates pass:** retain the transition readout and run one same-protocol
  uKNIT-BC/Dialga transfer attribution panel before any scale.
- **Signal and diffusion pass but S-box still fails:** discard this readout and
  run a zero-training tap audit of the transition branch before another model.
- **S-box passes but anchor retention fails:** treat the branch as overly
  restrictive; inspect fusion/gate optimization without adding capacity.
- **Protocol invalid:** repair only the failed binding and rerun unchanged.

Do not add pairs, samples, epochs, seeds, positions, rounds, MoE, DDT/trail
inputs, another architecture family or remote execution inside K1-AK.

## 8. Required Artifacts

```text
run_id = i1_uknit_family_midori64_sbox_transition_k1ak_2048_seed6_seed7_20260729

results.jsonl
controls.jsonl
history.csv
checkpoint_manifest.json
dataset_manifest.jsonl
preflight.json
progress.jsonl
gate.json
validation.json
summary.json
comparison.csv
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, append observed metrics, decision and evidence-backed next
action here, then refresh `outputs/00_RECENT_RESULTS.md` and JSON.

## 9. Completed Result And Verdict

The run completed all eight training rows, `80/80` epochs and `24/24`
three-split checkpoint evaluations. All source, cache, model-geometry,
parameter-budget, checkpoint-replay and metric checks passed.

Fresh results were:

| Seed | Fresh split | Correct AUC | K1-AI anchor | Correct - anchor | Correct - wrong S-box | Correct - corrupted linear | Correct - no structure |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | same-key | `0.668338` | `0.635222` | `+0.033115` | `-0.003697` | `+0.123527` | `+0.162691` |
| 6 | cross-key | `0.656132` | `0.626009` | `+0.030123` | `-0.000567` | `+0.121269` | `+0.145361` |
| 7 | same-key | `0.663027` | `0.608102` | `+0.054925` | `+0.004542` | `+0.142382` | `+0.168393` |
| 7 | cross-key | `0.653863` | `0.635971` | `+0.017892` | `-0.015453` | `+0.106547` | `+0.162656` |

The new readout improves every correct-structure fresh AUC over K1-AI by
`+0.017892` to `+0.054925`, preserves all diffusion/no-structure margins and
learns a transition gate of approximately `0.101-0.103`. It therefore adds
usable neural signal. It does not add discriminative S-box attribution:
correct-minus-wrong-S-box misses `+0.005` on all four fresh panels and is
negative on three. Independently trained wrong-S-box models reach AUC
`0.6567-0.6720`, matching or exceeding the correct model.

```text
status       = hold
decision     = innovation1_uknit_family_midori64_k1ak_sbox_transition_discrimination_failed
remote_scale = no
```

The candidate readout is not retained as the family architecture. The result
does not invalidate the Midori64 data route or diffusion-aware trunk; it shows
that an invariant observed-difference transition histogram is still compatible
with a wrong-S-box substitute.

## 10. Recommended Next Action

Run K1-AL as a zero-training same-checkpoint and branch-necessity audit on the
exact K1-AK validation caches and best correct checkpoints. For each seed,
compare the same state dictionary under correct runtime semantics, wrong S-box
runtime semantics and transition branch disabled. Require exact source replay,
identical state/dataset hashes and per-sample probability deltas.

This audit decides between two mechanisms:

- if the correct checkpoint loses AUC under the wrong S-box and when the branch
  is disabled, the branch is causal but independently trained wrong-S-box
  models learn a substitute; test a paired semantic-contrast objective next;
- if AUC remains unchanged, the new branch is not discriminatively causal;
  discard it and redesign the representation rather than adding data or scale.

Do not retrain, change data, increase pairs, add epochs, launch remotely or
start family transfer before K1-AL resolves this mechanism.
