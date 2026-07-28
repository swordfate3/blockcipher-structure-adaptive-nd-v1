# Innovation 1 uKNIT-Family Midori64 Transition Causal Audit K1-AL

**Date:** 2026-07-29
**Status:** planned / preregistered zero-training audit
**Execution:** local CPU only; no remote scale

## 1. Research Question

K1-AK replaced the compact value histogram with a shared per-cell S-box
input/output-transition readout. Its correct-structure fresh AUC improved over
K1-AI by `+0.017892` to `+0.054925`, and the correct diffusion layer remained
necessary. Independently trained wrong-S-box models nevertheless matched or
exceeded the correct model on three of four fresh panels.

K1-AL asks one narrower causal question:

> When the exact K1-AK correct checkpoint and dataset are held fixed, do the
> learned predictions depend on the supplied S-box and on the new transition
> branch itself?

This separates a causal but replaceable transition branch from a branch that
the classifier effectively ignores. It does not train or select another model.

## 2. Frozen Source

The only source run is:

```text
run_id = i1_uknit_family_midori64_sbox_transition_k1ak_2048_seed6_seed7_20260729
status = hold
decision = innovation1_uknit_family_midori64_k1ak_sbox_transition_discrimination_failed
```

Required source digests:

```text
gate.json                a8cd9de68a7b4e43a4c8f0793e31cbf8ce87f090c35be6f6821cab282e927f8f
validation.json          2d64a4e27b39a65fda5b44b217226fabb78a954d843573b47abbe34e0070e419
checkpoint_manifest.json 048906c4e9288f9795453d15b4fd5ba476ba54247b22296b5bc745517cabd2f7
controls.jsonl           3b667435eb6c91dfb1c828953e834e9556dedf16c5054b4e70ded1d598e6e04e
dataset_manifest.jsonl   5525a28f099a21bcca09aafbe05498f0f7951e22e171eaac6db055c174ff35bc
```

Correct best-checkpoint payloads:

```text
seed6 ac5364cb2b45d6e5f5dad189b582bfccedc18d29a06626d1ab3d349f12f44ed4
seed7 29d1d8918ed2f6fd0c5345c87cf4b6efe66682540116f8340dc6d3c785996018
```

Any source drift makes K1-AL invalid.

## 3. Single Experimental Variable

Strict-load each seed's K1-AK `correct_structure` best checkpoint into the
same state geometry and evaluate three inference conditions:

| Condition | Runtime S-box | Transition branch | Learned state |
|---|---|---|---|
| `correct_runtime` | correct | enabled | source correct state |
| `wrong_sbox_same_checkpoint` | deterministic wrong table | enabled | same source correct state |
| `transition_branch_off_same_checkpoint` | correct | forward output replaced by zero | same source correct state |

The branch-off intervention must use a forward-time wrapper/hook. It must not
write zero into `transition_gate`, alter a checkpoint tensor, rename state
keys, train, or select a new checkpoint. The underlying state-dict and source
checkpoint hashes must remain identical across all three conditions within a
seed/split.

## 4. Frozen Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | Midori64 r4 |
| Difference | cell8 role1, `0x0000000400000000` |
| Seeds | `6`, `7` |
| Pairs per sample | `4` |
| Train-seen split | `4096` total rows per seed |
| Same-key fresh split | `2048` total rows per seed |
| Cross-key fresh split | `2048` total rows per seed |
| Negative definition | encrypted random plaintexts |
| Batch | `64` |
| Training / optimizer steps / epochs | `0 / 0 / 0` |
| Execution | local CPU |

The complete matrix is `2 seeds x 3 splits x 3 conditions = 18 rows`.

## 5. Protocol Gate

Require:

- exact five K1-AK source digests, held decision and passing validation;
- exactly two correct best checkpoints with the preregistered payload hashes;
- exact replay of all six K1-AK correct-condition AUCs within `1e-7`;
- all six cached datasets loaded with their original payload digests;
- identical checkpoint, underlying state-dict and dataset hashes across the
  three conditions within every seed/split;
- strict state loading, fixed 512-bit input, four pairs, fixed row counts and
  encrypted-random-plaintext negatives;
- `18/18` finite evaluation rows and zero training, epochs and optimizer steps;
- distinct correct/wrong-S-box runtime fingerprints;
- branch-off wrapper verified to preserve every source state tensor.

Any failed protocol check makes the result invalid.

## 6. Research Gate

Apply every threshold independently to seed6/7 and both fresh splits:

```text
correct runtime AUC - wrong S-box AUC       >= +0.005
correct runtime AUC - transition-off AUC    >= +0.005
max per-sample probability delta             > 1e-6
```

Train-seen results are diagnostic only. Do not average seeds or splits to hide
a failed panel.

## 7. Decisions

- **Both interventions pass all four fresh panels:** the new branch and the
  supplied S-box are causal in the correct checkpoint. Independently trained
  wrong-S-box models learn a substitute shortcut. Next test one same-budget
  paired semantic-contrast objective while retaining this representation.
- **Branch-off passes but wrong S-box fails:** retain evidence that the branch
  is used, but discard this representation as non-identifying. Redesign the
  transition representation before another trained model.
- **Branch-off fails:** the new transition path is not a stable causal source of
  the K1-AK improvement. Discard the readout and audit which unchanged base or
  edge path produced the gain before proposing another architecture.
- **Protocol invalid:** repair only the failed source, state, dataset, runtime
  or replay binding and rerun unchanged.

Do not add data, pairs, epochs, seeds, positions, rounds, capacity, DDT/trail
inputs, MoE, family transfer or remote execution inside K1-AL.

## 8. Required Artifacts

```text
run_id = i1_uknit_family_midori64_transition_causal_k1al_20260729

results.jsonl
comparison.csv
checkpoint_manifest.json
dataset_manifest.jsonl
preflight.json
progress.jsonl
gate.json
validation.json
summary.json
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, append metrics, decision and the executable next action here,
then refresh `outputs/00_RECENT_RESULTS.md` and JSON.

## 9. Completed Result And Verdict

K1-AL completed all `18/18` inference rows with zero training, zero epochs and
zero optimizer steps. All source digests, exact checkpoint payloads, cached
dataset digests, strict state loads, replay tolerances, intervention geometry
and state-preservation checks passed.

Fresh results were:

| Seed | Fresh split | Correct runtime AUC | Wrong S-box AUC | Correct - wrong S-box | Transition off AUC | Correct - transition off |
|---:|---|---:|---:|---:|---:|---:|
| 6 | same-key | `0.668338` | `0.633491` | `+0.034847` | `0.579501` | `+0.088837` |
| 6 | cross-key | `0.656132` | `0.632029` | `+0.024103` | `0.580584` | `+0.075548` |
| 7 | same-key | `0.663027` | `0.600177` | `+0.062850` | `0.542057` | `+0.120970` |
| 7 | cross-key | `0.653863` | `0.610801` | `+0.043062` | `0.541830` | `+0.112033` |

Maximum per-sample probability changes were `0.3706-0.5495` for the wrong
S-box intervention and `0.2433-0.3905` for transition-branch removal, far above
the `1e-6` responsiveness floor. Correct-runtime AUC replayed K1-AK exactly.

Every fresh S-box and branch margin passed:

```text
status       = pass
decision     = innovation1_uknit_family_midori64_k1al_transition_and_sbox_causal_use_supported
remote_scale = no
```

The K1-AK transition representation is therefore retained. The earlier
independently trained wrong-S-box AUC does not mean the correct checkpoint
ignores S-box semantics: with weights fixed, changing only the S-box causes a
large and consistent loss. The remaining problem is optimization
identifiability: independent wrong-S-box training can learn a substitute
solution.

This is a local two-seed causal diagnostic, not formal-scale, attack, SOTA,
family-transfer, arbitrary-SPN or ceiling evidence.

## 10. Recommended Next Action: K1-AM

Run one same-budget paired semantic-contrast training diagnostic on the exact
K1-AK Midori64 r4 cell8 datasets.

```text
question:
  can a correct-versus-wrong runtime contrast during training preserve K1-AK
  fresh AUC while making the learned solution more identifiable by S-box?

same-budget anchor:
  frozen K1-AK correct checkpoints and all K1-AL same-checkpoint margins

single variable:
  add one bounded paired semantic-contrast term; keep architecture, data,
  difference, pairs, seeds, epochs, optimizer, batch and checkpoint rule fixed

training matrix:
  seed6/7 x {correct-oriented contrast, swapped-orientation control}
  = 4 trained models, never more than this attribution matrix

scale:
  Midori64 r4, 2048/class, 4 pairs, seeds 6/7, 10 epochs, batch 64, local CPU

evaluation:
  train-seen, same-key fresh and cross-key fresh under correct runtime,
  wrong-S-box same checkpoint and transition branch off

advance gate:
  correct-runtime AUC >= K1-AK anchor - 0.010 on all four fresh panels;
  correct-oriented contrast beats its swapped-orientation control by >= 0.005;
  correct-minus-wrong-S-box and correct-minus-transition-off >= 0.005;
  no parameter increase and no protocol drift

stop gate:
  any fresh anchor-retention or orientation-attribution failure holds the
  objective and returns to representation/optimization analysis

blocked:
  remote scale, family transfer, extra samples/pairs/epochs/seeds, DDT/trail
  inputs, MoE, width changes or mechanical hyperparameter sweeps
```

K1-AM must be preregistered separately before training. K1-AL itself does not
authorize remote execution.
