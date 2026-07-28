# Innovation 1 uKNIT-Family Midori64 Same-Checkpoint Replay K1-AJ

**Date:** 2026-07-29
**Status:** completed / held after valid local zero-training causal audit
**Execution:** local CPU, no optimizer steps

## 1. Research Question

K1-AI independently trained four equal-geometry K1-AA models on the confirmed
Midori64 r4 cell8 data route. Correct structure learned a real signal and beat
corrupted-linear/no-structure controls on every fresh seed/split, but failed the
wrong-S-box margin on both same-key fresh rows.

K1-AJ isolates the remaining ambiguity:

> Does a completed correct-structure checkpoint causally depend on the exact
> Midori64 S-box at inference, or did independently trained wrong-S-box models
> merely learn an alternative shortcut with similar AUC?

This is a same-state runtime intervention. It performs no training and does not
change data, labels, checkpoints, model geometry or evaluation metrics.

## 2. Frozen K1-AI Source

```text
source_root = outputs/local_diagnostic/
  i1_uknit_family_midori64_neural_attribution_
  k1ai_2048_seed6_seed7_20260729
```

| Artifact | SHA-256 |
|---|---|
| K1-AI gate | `5f7eca268a26a9f3d3fdf746a0e9beae4552b156c1a832a7f81f02457d32803d` |
| K1-AI validation | `a901d807da281762acbba30d960fc787dedd5df0981ed77499d09cf0589e370e` |
| K1-AI checkpoint manifest | `1afc62124164e21340aac4c2ffe7450f462e341ae9a5be20b380265a104fb327` |
| K1-AI controls | `f2e6a9ba34821f3acd1ccc787befb465ceca4e9f9f90ca58bbc62ca5d87092de` |
| K1-AI dataset manifest | `5525a28f099a21bcca09aafbe05498f0f7951e22e171eaac6db055c174ff35bc` |

The source must retain `status=hold`, decision
`innovation1_uknit_family_midori64_k1ai_signal_learned_structure_attribution_not_supported`,
no failed protocol checks, `remote_scale=no`, and exactly two seed6/7
correct-structure best checkpoints with the manifest-bound checkpoint hashes.
The six source datasets must match their manifest row counts and content
digests.

## 3. Single Experimental Variable

For each seed, load the one K1-AI correct-structure best checkpoint and keep the
state dictionary bit-identical across four runtime conditions:

| Condition | S-box | Linear layer | Purpose |
|---|---|---|---|
| `correct_structure` | correct | correct | exact source replay |
| `wrong_sbox` | fixed wrong table | correct | causal S-box intervention |
| `corrupted_linear` | correct | deterministic corruption | causal diffusion intervention |
| `no_structure` | disabled | identity | no-structure anchor |

All conditions must retain `214316` parameters, identical state-dict geometry
and the exact source state hash. Only external runtime tensors and
`apply_sboxes` may differ.

## 4. Frozen Evaluation Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | Midori64 r4 |
| Difference | cell8 role1, `0x0000000400000000` |
| Seeds | `6`, `7` |
| Pairs per sample | `4` |
| Train-seen rows | `4096` per seed |
| Same-key fresh rows | `2048` per seed |
| Cross-key fresh rows | `2048` per seed |
| Negative definition | encrypted random plaintexts |
| Source checkpoint | K1-AI correct-structure best validation-AUC checkpoint |
| Runtime window | Midori64 homogeneous transition repeated twice |
| Batch / device | `64` / local CPU, exact K1-AI replay geometry |
| Training | none; `optimizer_steps=0`, `epochs=0` |

The audit contains exactly `2 seeds × 3 splits × 4 conditions = 24` inference
rows. For every seed/split, all four rows must share checkpoint hash, state hash,
dataset digest, row count and labels.

## 5. Protocol Gate

Require all of the following before interpreting metrics:

1. exact five K1-AI source artifact hashes and frozen hold decision;
2. K1-AI validation passed with no errors;
3. exactly two correct-structure best checkpoints, one per seed, with live file hashes;
4. six K1-AI datasets pass row-count and content-digest verification;
5. all four model geometries and parameter counts are identical;
6. wrong S-box changes only S-box truth bits, corrupted linear changes only the
   linear matrices, and no-structure disables S-boxes with identity matrices;
7. strict loading preserves the exact source state hash in all 24 rows;
8. correct-structure AUC, checkpoint hash, state hash and dataset digest exactly
   replay the six K1-AI correct rows within `1e-7` AUC tolerance;
9. all probabilities/AUC/deltas are finite and every row is inference-only;
10. runtime fingerprints are distinct across the four conditions.

Any failed protocol check makes the run `invalid` and authorizes only repair
and unchanged replay.

## 6. Research Gate

Apply every threshold separately to seed6/7 on both fresh splits:

```text
correct_structure AUC                    >= 0.550
correct_structure - wrong_sbox           >= +0.005
correct_structure - corrupted_linear     >= +0.005
correct_structure - no_structure         >= +0.010
max probability change per intervention  > 1e-6
```

Training-seen rows are diagnostic only and cannot satisfy a fresh gate.

## 7. Decisions And Required Next Action

- **All S-box and diffusion gates pass:** representation contains causal S-box
  semantics, but K1-AI independent optimization allowed the wrong-S-box model
  to learn a substitute. Test one same-budget paired-initialization semantic
  contrast objective; do not change data or model geometry.
- **Diffusion passes but S-box fails:** current compact invariant representation
  is causally sensitive to diffusion but not discriminatively sensitive to the
  nonlinear table. Replace only the invariant histogram readout with one
  bounded cell-conditional S-box-transition residual at the same budget.
- **Both S-box and diffusion fail:** the learned prediction is dominated by a
  structure-independent path. Run a zero-training branch ablation before any
  architecture redesign.
- **Protocol invalid:** repair source/checkpoint/dataset/runtime binding only.

Do not train, add pairs, samples, epochs, seeds, rounds, positions, MoE,
trail/DDT inputs, another cipher or remote work inside K1-AJ.

## 8. Run ID And Required Artifacts

```text
run_id = i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729
```

Required artifacts are preflight, bound dataset/checkpoint manifests, 24 result
rows, comparison CSV, gate, validation, summary, progress, Chinese SVG and
rendered-pixel visual-QA report/marker. Refresh the recent-results index and
append the observed result plus evidence-backed next action here after
completion.

The first attempted run id
`i1_uknit_family_midori64_same_checkpoint_k1aj_20260729` is retained as invalid
protocol evidence. It used evaluation batch `256`, while the K1-AI source used
batch `64`; seed6 same-key AUC consequently drifted by `4.77e-7`, exceeding the
frozen `1e-7` replay tolerance. No source checkpoint, dataset or research
threshold changed. The corrected run changes only evaluation batching back to
the source value and uses the distinct `replay_fix` run id above.

## 9. Completed Result And Verdict

The corrected replay produced all `24/24` rows with `optimizer_steps=0`,
`epochs=0` and no failed protocol checks. The correct-structure rows reproduce
the six K1-AI source AUCs within the frozen `1e-7` tolerance, and every row in a
seed/split panel shares the exact checkpoint, state dictionary, dataset and
labels.

Fresh same-checkpoint margins were:

| Seed | Fresh split | Correct - wrong S-box | Correct - corrupted linear | Correct - no structure |
|---:|---|---:|---:|---:|
| 6 | same-key | `+0.003480` | `+0.111741` | `+0.131579` |
| 6 | cross-key | `+0.023935` | `+0.100781` | `+0.124036` |
| 7 | same-key | `+0.007615` | `+0.087521` | `+0.101380` |
| 7 | cross-key | `+0.012098` | `+0.130973` | `+0.122338` |

All four correct-structure AUCs exceed `0.550`. Correct diffusion beats the
corrupted-linear control by `+0.087521` to `+0.130973`, and correct structure
beats no structure by `+0.101380` to `+0.131579`. The wrong-S-box intervention
also changes individual probabilities substantially, but seed6 same-key fresh
reaches only `+0.003480`, below the preregistered `+0.005` S-box margin. The
other three fresh panels pass that margin.

```text
status       = hold
decision     = innovation1_uknit_family_midori64_k1aj_diffusion_causal_sbox_discrimination_failed
remote_scale = no
```

The evidence therefore supports a narrow causal claim: the completed K1-AI
checkpoint uses the supplied Midori64 diffusion semantics. It does not yet
support stable discrimination of the exact S-box across both seeds and both
fresh key scopes. It is not evidence for arbitrary-SPN transfer, an attack,
SOTA performance or a scale ceiling.

Required artifacts:

```text
outputs/local_audit/i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729/results.jsonl
outputs/local_audit/i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729/gate.json
outputs/local_audit/i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729/validation.json
outputs/local_audit/i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729/comparison.csv
outputs/local_audit/i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729/curves.svg
outputs/local_audit/i1_uknit_family_midori64_same_checkpoint_k1aj_replay_fix_20260729/visual_qa_passed.marker
```

## 10. Recommended Next Action

The next experiment should replace only the compact invariant histogram
readout with one bounded, cell-conditional S-box-transition residual. It must
use the K1-AI correct-structure model as the same-budget anchor and retain the
wrong-S-box, corrupted-linear and no-structure controls.

Freeze Midori64 r4, cell8 difference `0x0000000400000000`, `4` pairs per
sample, `2048/class`, seeds `6/7`, `10` epochs, same-key and cross-key fresh
splits, encrypted-random-plaintext negatives and local execution. The one
experimental variable is the S-box-conditioned readout. Advance only if the
candidate preserves fresh correct-structure AUC `>=0.550`, beats the wrong
S-box control by `>=+0.005` on all four fresh seed/split panels, and retains the
existing diffusion/no-structure margins. Otherwise discard that readout and
audit its cell/S-box transition representation before another redesign.

Do not increase to `16` pairs, add data, change the difference, change rounds,
switch model families, introduce MoE, or launch remotely from the K1-AJ result.
