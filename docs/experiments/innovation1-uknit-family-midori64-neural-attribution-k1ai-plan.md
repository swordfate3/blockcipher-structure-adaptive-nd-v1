# Innovation 1 uKNIT-Family Midori64 Neural Attribution K1-AI

**Date:** 2026-07-29
**Status:** completed / signal learned, S-box attribution held
**Execution:** local CPU fixed-budget diagnostic

## 1. Research Question

K1-AH established that Midori64 r4 cell8, role1 difference
`0x0000000400000000` has a strong deterministic two-stage GF(2) signal on
untouched seed6/7 and same-key/cross-key fresh data. Its exact-statistic AUC was
`0.912907-0.953841`, while the raw anchor remained `0.493369-0.517631`.
K1-AH performed no neural training.

K1-AI asks one question only:

> Can the fixed-geometry K1-AA virtual-slot network learn the confirmed
> Midori64 r4 cell8 signal, and does the correct Midori64 S-box and linear
> diffusion structure outperform independently trained wrong-S-box,
> corrupted-linear and no-structure controls?

This is a local `2048/class` mechanism diagnostic. It is not formal training,
an attack, a SOTA comparison, a family-transfer result or a Midori64 ceiling.

## 2. Frozen K1-AH Source

```text
source_root = outputs/local_audit/
  i1_uknit_family_midori64_difference_position_k1ah_20260729
```

| Artifact | SHA-256 |
|---|---|
| K1-AH gate | `5fb101cd892dcedb849e7a4745996fc9fced8d9450b0449c8f206b53cc786708` |
| K1-AH dataset manifest | `6e7351a132518baa0942431d132d164fd2ef01fe6c12bb75af7fd96b96a7d1c8` |
| K1-AH validation | `e081af654348ee97d65d62756d668d367976150d44449c7fe3e598c7f8f67fb9` |

Execution requires the exact K1-AH pass decision
`innovation1_uknit_family_midori64_k1ah_confirmed_r4_position_supported`,
cell8 in `confirmed_cells`, all source protocol checks true, and the six
seed6/7 cell8 confirmation caches. Cache row counts and dataset digests must
match the source manifest before training.

## 3. Single Experimental Variable

K1-AI freezes cipher, rounds, difference, keys, samples, pairs, labels,
features, optimizer, epochs, checkpoint rule and runtime window. The only
variable is the runtime structure condition consumed by the same K1-AA
fixed-geometry network.

| Condition | S-box stages | Linear stages | Purpose |
|---|---|---|---|
| `correct_structure` | correct Midori64 table | correct repeated Midori64 operator | candidate |
| `wrong_sbox` | fixed wrong truth-table input permutation | correct | nonlinear-semantic control |
| `corrupted_linear` | correct | deterministic invertible corruption | diffusion-semantic control |
| `no_structure` | disabled | identity | same-budget neural anchor |

All four models must have the same `214316` trainable parameters and identical
state-dict geometry. The wrong-S-box control changes only S-box truth bits; the
corrupted-linear control changes only linear matrices; the no-structure control
uses identity linear matrices and disables S-box application.

Midori64's descriptor repeats one homogeneous transition. Reversing the two
loaded transitions is algebraically identical to the correct structure, so
K1-AI explicitly forbids a reversed-round control.

## 4. Frozen Data And Training Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | Midori64 r4 |
| Input difference | cell8 role1, `0x0000000400000000` |
| Difference profile | `midori64_k1ah_cell8_r4` |
| Seeds | `6`, `7` |
| Train samples | `2048/class` (`4096` total rows) |
| Same-key fresh | `1024/class` (`2048` total rows) |
| Cross-key validation | `1024/class` (`2048` total rows) |
| Pairs per sample | `4` |
| Feature input | four raw ciphertext pairs as `512` bits/sample |
| Negative definition | encrypted random plaintexts |
| Keys | exact K1-AH seed6/7 train and validation keys |
| Cipher window | rounds 2 and 3 of the four-round encryption |
| Runtime descriptor window | homogeneous one-round template repeated twice, start `0` |
| Hidden / pair embedding | `32` / `128` |
| Virtual projection slots | `16` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best cross-key validation AUC, restored |
| Device | local CPU |

The matrix contains exactly eight independently trained rows:

```text
seed6/7 x {
  correct_structure,
  wrong_sbox,
  corrupted_linear,
  no_structure
}
```

All training and cross-key validation accesses must reuse K1-AH disk caches.
Same-key fresh data is loaded directly from the bound K1-AH manifest for
post-training evaluation. Every condition for a given seed/split must consume
the same dataset digest.

## 5. Readiness And Protocol Gate

Before interpreting metrics require:

1. exact K1-AH source hashes, pass decision, selected cell8 and validation;
2. exactly eight tasks covering seed6/7 and all four structure conditions;
3. every task binds Midori64 r4, cell8 difference, frozen keys and strict negatives;
4. all models have `214316` parameters and identical state-dict geometry;
5. intervention audits prove each control changes only its declared structure;
6. the homogeneous reverse control is rejected as equivalent and unavailable;
7. six K1-AH source caches pass row-count and content-digest verification;
8. all sixteen matrix cache accesses are reuse events and none regenerate;
9. exactly eight training rows, eight checkpoints and twenty-four three-split evaluations;
10. restored cross-key AUC replays the training result within a tested numerical tolerance;
11. all conditions use the same dataset digest for each seed and split;
12. every row trains ten epochs with best-validation-AUC restoration.

Any failed protocol item makes the run `invalid` and authorizes only repair and
an unchanged rerun.

## 6. Advance Gate

Apply every threshold separately to seed6 and seed7 on both fresh splits. No
average may hide a failed seed or split.

```text
correct_structure AUC                    >= 0.550
correct_structure - no_structure         >= +0.010
correct_structure - wrong_sbox           >= +0.005
correct_structure - corrupted_linear     >= +0.005
```

Training-split metrics diagnose memorization but do not satisfy a fresh gate.

## 7. Decisions And Required Next Action

- **All gates pass:** retain the K1-AA mechanism and separately preregister a
  remote `65536/class` medium diagnostic with only the strongest required
  controls and disk-backed cache/progress/resume. Do not call it formal or
  paper-scale evidence.
- **Correct structure learns signal but misses a control margin:** keep the
  Midori64 cell8 data route, reject semantic attribution, and audit the exact
  shortcut shared by the tied conditions before another architecture or scale.
- **Same-key passes but cross-key fails:** hold scale and test one key-invariance
  change at the same budget.
- **Correct structure remains below `0.550`:** treat neural access to the K1-AH
  statistic as the bottleneck and localize the first destructive representation
  stage before adding capacity, pairs, data or another model family.
- **Protocol invalid:** repair only the failed source, control, cache,
  checkpoint or metric binding and rerun unchanged.

Do not increase pairs, samples, seeds, epochs, width or rounds; do not add MoE,
trail/DDT inputs, another cipher or remote training inside K1-AI.

## 8. Run ID And Required Artifacts

```text
run_id = i1_uknit_family_midori64_neural_attribution_
         k1ai_2048_seed6_seed7_20260729
```

Required artifacts are preflight, filtered dataset manifest, eight checkpoints,
training results, twenty-four evaluation rows, split attribution CSV,
checkpoint manifest, gate, validation, summary, history, progress, Chinese
explanatory SVG and rendered-pixel visual-QA report/marker. After completion,
refresh `outputs/00_RECENT_RESULTS.md` and `outputs/00_RECENT_RESULTS.json` and
record the observed result plus evidence-backed next action in this document.

## 9. Completed Result

K1-AI completed the frozen matrix and all restored-checkpoint evaluations:

```text
training rows = 8 / 8
evaluation rows = 24 / 24
checkpoint rows = 8 / 8
cache reuse events = 16 / 16
cache regeneration = none
failed protocol checks = []
```

Fresh AUC and correct-structure margins were:

| Seed | Split | Correct | Wrong S-box | Corrupted linear | No structure | Correct - wrong S-box | Correct - corrupted linear | Correct - no structure |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | same-key fresh | `0.635222` | `0.630230` | `0.534313` | `0.510911` | `+0.004992` | `+0.100910` | `+0.124311` |
| 6 | cross-key fresh | `0.626009` | `0.602981` | `0.536582` | `0.501007` | `+0.023028` | `+0.089427` | `+0.125001` |
| 7 | same-key fresh | `0.608102` | `0.611664` | `0.531714` | `0.492023` | `-0.003562` | `+0.076388` | `+0.116079` |
| 7 | cross-key fresh | `0.635971` | `0.624208` | `0.542517` | `0.507302` | `+0.011763` | `+0.093454` | `+0.128669` |

Every correct-structure fresh AUC passed `0.550`. Correct structure also beat
the corrupted-linear and no-structure controls by large margins on every seed
and fresh split. This establishes that the current neural path learns a real
Midori64 r4 signal and depends on the correct linear diffusion operator under
this local budget.

The S-box attribution gate did not pass. Seed6 same-key fresh reached only
`+0.004992`, just below the frozen `+0.005` threshold, and seed7 same-key fresh
favoured the wrong S-box by `0.003562`. Cross-key S-box margins were positive,
but the protocol forbids averaging them over the two failed same-key rows.

The frozen decision is:

```text
status = hold
decision = innovation1_uknit_family_midori64_k1ai_signal_learned_structure_attribution_not_supported
remote_scale = no
```

This is stronger than a chance-level failure: linear diffusion semantics are
clearly useful, while correct nonlinear semantics are not yet independently
identifiable. It does not prove that Midori64 S-box information is useless or
that the family-adaptive route has reached a ceiling.

The final Chinese two-heatmap SVG was rendered to `1920x1032` pixels and passed
`visual-qa-redraw` after increasing panel spacing and changing margin labels to
five decimals. The latter prevents the failed `+0.004992` row from being
misread as a passing rounded `+0.0050`. Final inspection found no text overlap,
clipping, missing glyphs, ambiguous scales or unreadable values.

## 10. Recommended Next Action: K1-AJ Same-Checkpoint Semantic Replay

K1-AJ should answer one narrower causal question before any architecture or
scale change:

> Does a completed K1-AI correct-structure checkpoint use the exact Midori64
> S-box semantics at inference, or did independently trained wrong-S-box models
> learn an alternative shortcut with similar AUC?

Reuse only the two completed seed6/7 correct-structure checkpoints and the six
K1-AH cell8 datasets. Perform zero training and change only the runtime
structure loaded under the same state dictionary:

```text
correct checkpoint x {
  correct Midori64 structure,
  wrong S-box + correct linear,
  correct S-box + corrupted linear,
  no S-box + identity linear
}
```

Evaluate all four interventions on train-seen, same-key fresh and cross-key
fresh data. Require strict state-dict geometry/load identity, exact dataset
digests, exact replay of each K1-AI correct-structure AUC, zero optimizer steps
and finite metrics. Apply these research gates separately to seed6/7 and both
fresh splits:

```text
same-checkpoint correct - wrong S-box       >= +0.005
same-checkpoint correct - corrupted linear >= +0.005
same-checkpoint correct - no structure     >= +0.010
```

- If the same-checkpoint S-box margin also fails, the K1-AA representation is
  insufficiently sensitive to nonlinear semantics. Redesign only the compact
  invariant histogram into a bounded cell-conditional S-box transition
  residual, while retaining fixed virtual slots and the same data budget.
- If the same-checkpoint S-box margin passes but independent-training K1-AI
  failed, the representation contains the semantics but optimization permits
  the wrong-S-box shortcut. Test one paired-initialization or semantic-contrast
  training change at the same budget.
- If exact replay fails, mark K1-AJ invalid and repair checkpoint/runtime binding
  only.

K1-AJ is a local, zero-training audit. Do not add pairs, samples, epochs, rounds,
new seeds, MoE, trail/DDT inputs, another cipher or remote scale until this
causal distinction is resolved.
