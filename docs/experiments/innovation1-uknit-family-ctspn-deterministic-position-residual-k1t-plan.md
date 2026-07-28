# Innovation 1 uKNIT-Family CT-SPN Deterministic Position Residual K1-T

**Date:** 2026-07-28
**Status:** completed / pass / remote medium diagnostic authorized
**Execution:** completed local CPU fixed-budget diagnostic after zero-training readiness

## 1. Research Question

K1-S exactly replayed the strong K1-Q uKNIT r5 cell11 deterministic statistic
at AUC `0.806228-0.825591`, while the completed K1-R learned path retained no
tap above the access gate across both seeds and both fresh key scopes. K1-S
therefore held remote scale and selected one representation change:

> Does retaining the exact `stage x native-cell x nibble-value` histogram as a
> bounded residual make the confirmed uKNIT r5 signal learnable and
> attributable to the correct runtime S-box and diffusion semantics?

K1-T is a local `2048/class` mechanism diagnostic. It is not formal training,
an attack, a SOTA comparison, family transfer, or a uKNIT/neural ceiling claim.

## 2. Frozen Authority

### 2.1 K1-S decision

```text
root = outputs/local_audit/
  i1_uknit_family_ctspn_representation_access_
  k1s_seed3_seed4_20260728
```

| Artifact | SHA-256 |
|---|---|
| `gate.json` | `b7b16cef0c14f27c3325b65deaaca4acb206e811397a1839d8f28aca45ecc2e6` |
| `validation.json` | `f264c54343986647594935d0f9b0aee78ec1a986f01eb1d135ef8048b23cdafa` |
| `results.jsonl` | `c9d2b0d899cbe132359755cc19ddc9588e1d97e46dc3e5cdd08ca3e5356be077` |
| `feature_manifest.jsonl` | `b9d62cab840069e34588d6a609fff589ba66d04e26c20a8820036b3b13b36d91` |
| `scorer_manifest.jsonl` | `93e2d9f3d843be91334fe8640d9ce2555dddf3e5635f42bd21e32e1e9894afc9` |
| `checkpoint_manifest.json` | `73758c90a9564fa35b61e0d2bb707ca88bf3b912dd2926de1c457b0c461ee046` |

Require the exact clean hold decision
`innovation1_uknit_family_ctspn_k1s_learned_representation_access_not_supported`,
all protocol checks true, T0 accessible on every fresh seed/split, and T1/T2/T3
not accessible across the full panel.

### 2.2 Frozen data and anchor

Reuse the six exact K1-Q cell11 disk caches and source bindings already frozen
in K1-S. The K1-R exact-composition rows are the completed same-data,
same-optimizer anchor:

```text
seed3 exact checkpoint SHA-256 =
  030d280458654dcbda6a38aafe77f39c3d9f43cdee6ec350742e3d36252071e4
seed4 exact checkpoint SHA-256 =
  a64b3f326795adf955aba6ee87ebc9b9a5b44861322aaa6a7087ea75c9c45e21
```

The anchor may be replayed without retraining only when its source artifact,
dataset, checkpoint, state-dictionary and metric digests remain exact. It is
not counted as a K1-T optimizer row.

## 3. Single Architecture Variable

Retain the complete K1-N exact-composition base, topology residual, bounded
edge gate and classifier. Add one deterministic histogram residual:

1. compute the same five exact inverse stages as K1-Q and K1-S;
2. select the XOR-difference channel;
3. gather each native cell's four ordered bit roles into a nibble value;
4. average the one-hot nibble values across the four pairs, giving
   `[5 stages, 16 cells, 16 values]` per sample;
5. apply one shared `16 -> 8` value encoder at every stage/cell slot;
6. flatten the still ordered `5 x 16 x 8` slots and project to the existing
   `128` pair-embedding width;
7. repeat that `128`-vector across the existing attended/mean/max embedding
   segments and add it through its own bounded scalar gate before the unchanged
   classifier.

The branch may not consume cipher identity, active-cell constants, DDT/trails,
labels, keys or round-specific learned parameter shapes. Its computation is
conditioned only by the runtime cell partition, S-box tables and GF(2) linear
operators already supplied to K1-N.

## 4. Frozen Candidate And Controls

Train these six rows independently:

```text
seed3/4 x {
  exact_position_histogram_residual,
  wrong_sbox_position_histogram_residual,
  invariant_histogram_residual
}
```

| Condition | Runtime composition | Histogram readout | Purpose |
|---|---|---|---|
| exact | correct S-boxes and ordered GF(2) operators | preserve all native cell slots | candidate |
| wrong S-box | deterministic wrong cell/table semantics, same linear operators | preserve all native cell slots | nonlinear semantic control |
| invariant | correct exact composition | replace every cell slot by the mean over cells before projection | position-erasure control |

All three models must have identical trainable parameter names and shapes,
expected count `<=225000`, identical initialization protocol and equal optimizer
budget. The invariant control retains the same `5 x 16 x 16` tensor geometry;
it may not remove parameters or channels. The completed K1-R exact model is a
fourth evaluation condition but receives zero K1-T training steps.

## 5. Frozen Data And Training Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Difference | cell11 role1, `0x0000400000000000` |
| Seeds | `3`, `4` |
| Train | `2048/class`, `4096` total rows |
| Same-key fresh | `1024/class`, `2048` total rows |
| Cross-key | `1024/class`, `2048` total rows |
| Pairs/sample | `4` |
| Negative definition | encrypted random plaintexts |
| Keys | exact K1-Q/K1-R seed3/4 keys |
| Runtime window | `round_start=3`, `rounds=2` |
| Hidden / pair embedding | `32` / `128` |
| Histogram value width | `8` |
| Edge / histogram initial effective gates | `0.05` / `0.05` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best cross-key validation AUC, restored |
| Device | local CPU |

Every condition reuses the exact train/cross-key disk cache for its seed; none
may regenerate data. Same-key fresh evaluation loads the third K1-Q cache
directly. No benchmark field may change from K1-R.

## 6. Readiness Gate

Before any optimizer step require:

1. exact K1-S hashes, clean hold decision and protocol checks;
2. exact K1-Q cache and K1-R anchor/checkpoint bindings;
3. exactly six frozen tasks with the protocol in Sections 4-5;
4. all three model conditions have identical parameter geometry and count at
   most `225000`;
5. exact and wrong-S-box deterministic histograms differ on a fixed binary
   fixture while their linear operators remain equal;
6. invariant histograms equal the exact histogram averaged over native cells
   and repeated to all cell slots, and are not bit-equal to the exact tensor;
7. candidate and both controls produce finite logits with equal shapes;
8. candidate versus wrong-S-box and candidate versus invariant logits are
   observable under a shared state dictionary;
9. both bounded gates are nonzero and strictly inside `(-1, 1)`;
10. one MSE backward fixture gives finite nonzero histogram-branch gradients;
11. candidate ordinary forward is deterministic in evaluation mode;
12. all six source caches and target checkpoint paths are present and digest
    bound.

Any failure prohibits training and authorizes only repair of the failed
readiness invariant followed by an unchanged rerun.

## 7. Result And Advance Gate

Require six completed training rows, six checkpoints and twenty-four
three-split evaluation rows:

```text
2 seeds x 3 splits x {
  exact residual,
  wrong-S-box residual,
  invariant residual,
  frozen K1-R exact anchor
}
```

Apply every threshold separately to both seeds and both fresh splits:

```text
exact residual AUC                         >= 0.600
exact residual - frozen K1-R exact         >= +0.050
exact residual - wrong-S-box residual      >= +0.010
exact residual - invariant residual        >= +0.030
```

Training metrics are descriptive only. Averages may not hide a failed seed or
key scope.

## 8. Decisions And Required Next Action

- **All gates pass:** retain the position-residual mechanism and preregister a
  separate remote `65536/class` medium diagnostic with only exact, strongest
  semantic control and invariant control. Require disk-backed cache/progress/
  reuse and do not call it formal or paper-scale.
- **Exact learns but misses wrong-S-box margin:** the statistic is useful but
  not attributable to correct nonlinear semantics. Hold scale and isolate
  stage contributions without adding data or capacity.
- **Exact learns but misses invariant margin:** position preservation is not
  necessary under this task. Replace the candidate by the simpler invariant
  histogram branch before scale.
- **Same-key passes but cross-key fails:** classify the branch as key-specific
  and test one key-invariance change at the same budget.
- **Exact remains below `0.600`:** reject the current trainable projection/fusion
  despite the deterministic Fisher anchor and audit fixed Fisher initialization
  versus learned random initialization; do not add samples first.
- **Protocol invalid:** repair only the failed source/readiness/artifact binding
  and rerun unchanged.

Blocked inside K1-T: remote launch, more samples, pairs, positions, epochs,
seeds or keys; MoE; DDT/trails; cipher identity; new ciphers; new network
families; and post-result threshold changes.

## 9. Run And Required Artifacts

```text
run_id = i1_uknit_family_ctspn_deterministic_position_residual_
         k1t_2048_seed3_seed4_20260728
```

Required artifacts are preflight/readiness, filtered dataset manifest, six
checkpoints and their manifest, six training rows, twenty-four evaluation rows,
attribution CSV, gate, validation, summary, history, progress, Chinese SVG,
plot report and `visual_qa_passed.marker`. After completion refresh both recent
result indexes, record the observed decision and next executable action here,
then commit and push only K1-T files.

## 10. Completed Result

The frozen run completed on 2026-07-28 under:

```text
outputs/local_diagnostic/
  i1_uknit_family_ctspn_deterministic_position_residual_
  k1t_2048_seed3_seed4_20260728/
```

The independent zero-step readiness gate passed before training. All three
conditions had identical `214316`-parameter geometry, both bounded gates were
open, correct/wrong/invariant semantics produced observable differences under
a shared state, and the histogram branch had finite nonzero gradients. The
runner then reused all twelve frozen train/validation cache bindings, completed
six optimizer rows and restored six best checkpoints. The final evaluation
contained the required twenty-four rows.

### 10.1 Fresh-split metrics

| Seed | Fresh split | Exact position residual | K1-R anchor | Wrong S-box | Position erased | Exact-anchor | Exact-wrong | Exact-invariant |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | same-key | `0.735209` | `0.522282` | `0.511016` | `0.600709` | `+0.212928` | `+0.224194` | `+0.134500` |
| 3 | cross-key | `0.713162` | `0.493238` | `0.506986` | `0.565424` | `+0.219924` | `+0.206177` | `+0.147738` |
| 4 | same-key | `0.738803` | `0.513252` | `0.505732` | `0.594370` | `+0.225551` | `+0.233071` | `+0.144433` |
| 4 | cross-key | `0.748229` | `0.516200` | `0.512875` | `0.594048` | `+0.232029` | `+0.235354` | `+0.154181` |

Every fresh row passed all four frozen research thresholds. All protocol checks
also passed, including source digests, plan alignment, cache reuse, checkpoint
count and strict state replay. `validate-results` returned `status=pass`, six
result rows and no errors.

### 10.2 Decision and claim scope

```text
status = pass
decision =
  innovation1_uknit_family_ctspn_k1t_
  deterministic_position_residual_supported
remote_scale = authorized_65536_per_class
```

K1-T supports the bounded deterministic position-residual mechanism for this
two-seed local uKNIT r5 cell11 diagnostic. It does not yet establish formal
scale, another difference or round, transfer to another uKNIT-family cipher,
an attack, a SOTA result or a family-wide architecture claim.

The observed controls materially narrow the prior failure: the confirmed signal
is not recovered merely by adding equal capacity. It requires the correct S-box
semantics and retaining native cell positions under this protocol. The earlier
K1-R route primarily lost this position-specific statistic inside its learned
compression.

### 10.3 Visual and artifact validation

The Chinese two-panel heatmap was rendered to `1920x1056` pixels and inspected
with `visual-qa-redraw`. The first preview exposed right-panel labels overlapping
the left colorbar; shortened semantic labels removed the conflict. The final
preview had no overlap, clipping, missing glyphs or structural ambiguity.

### 10.4 Executable next action

Preregister a separate remote `65536/class` medium diagnostic. Its question is
whether the K1-T mechanism and attribution margins survive a 32x data increase.
Keep uKNIT r5, cell11 role1, four pairs/sample, runtime window, architecture,
loss, optimizer and ten epochs unchanged. Change only `samples_per_class` from
`2048` to `65536`; use seeds 3 and 4 and train the exact residual, wrong-S-box
semantic control and invariant position-erasure control at equal budget.

Launch only from a pushed commit and only after a route-specific remote
disk-cache readiness gate proves chunked `features.npy`/`labels.npy` or
equivalent payloads, metadata, durable progress and parameter-matched reuse
under `G:\lxy`. Advance beyond this medium run only if both seeds' fresh
cross-key evaluations retain exact AUC `>=0.600`, exact-wrong `>=+0.010` and
exact-invariant `>=+0.030`. Stop mechanical scale-up if any seed fails; inspect
learning curves and the failed attribution control before changing capacity,
epochs, pairs, difference or rounds.
