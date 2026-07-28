# Innovation 1 uKNIT-Family CT-SPN Cell11 Neural Attribution K1-R

**Date:** 2026-07-28
**Status:** completed / neural signal not supported
**Execution:** local CPU fixed-budget diagnostic

## 1. Research Question

K1-Q showed that the old uKNIT-BC r5 input difference `0x40` was a major
confounder: its untouched-seed fresh AUC remained `0.504206-0.518620`, while
moving the same native bit role to cell11 (`0x0000400000000000`) produced
`0.806228-0.825591` across seed3/4 same-key and cross-key splits under the
deterministic exact five-stage Fisher audit. K1-Q did not train a neural model.

K1-R asks one question only:

> Can the K1-N-derived trainable model learn the confirmed cell11 uKNIT r5
> signal, and does correct uKNIT S-box and diffusion structure help more than
> independently trained wrong-S-box, no-S-box and no-topology controls?

This is a local `2048/class` mechanism diagnostic. It is not formal training,
an attack, a SOTA comparison, a uKNIT-family transfer result or a ceiling claim.

## 2. Frozen K1-Q Source

```text
source_root = outputs/local_audit/
  i1_uknit_family_ctspn_difference_position_discovery_
  k1q_seed2_confirm_seed3_seed4_20260728
```

| Artifact | SHA-256 |
|---|---|
| K1-Q gate | `1af79fa865736635d40f729fe6621e677a4378e64c6779fc449756ae48609f8b` |
| K1-Q dataset manifest | `16d9549df5d1a6b2d88fd95e10ceec484e6f5443bd774f11d0f7d68dc85494f2` |
| K1-Q validation | `25b59f9b0eeab8eb894c4b3a40513437306a2c660f0c68f4ab478260689d8059` |

Execution requires the exact K1-Q pass decision
`innovation1_uknit_family_ctspn_k1q_confirmed_r5_difference_position_supported`,
`confirmed_cells` containing cell11, all protocol checks true, and the six
cell11 confirmation caches for seed3/4 and the three frozen splits. Cache
payload row counts and digests must match the K1-Q manifest before training.

## 3. Single Experimental Variable

K1-R freezes cipher, rounds, input difference, keys, samples, pairs, labels,
features, optimizer, epochs, checkpoint rule and runtime window. The only
variable is the runtime structure condition used by the K1-N-derived network.

Unlike K1-N's same-checkpoint zero-step controls, every K1-R condition receives
its own identical ten-epoch training budget. This tests whether the structure
helps optimization and generalization rather than whether changing semantics
after training perturbs one checkpoint.

| Condition | S-box stages | Linear stages | Purpose |
|---|---|---|---|
| `exact_composition` | correct heterogeneous uKNIT tables | correct ordered GF(2) operators | candidate |
| `wrong_sbox_semantics` | deterministic wrong cell/table ownership | correct | S-box semantic control |
| `no_sbox_composition` | identity | correct | nonlinear semantic control |
| `no_topology` | identity | identity | same-budget neural anchor |

All four models retain identical parameter geometry. No condition may receive
extra channels, parameters, epochs, initialization data or validation rows.

## 4. Frozen Data And Training Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Input difference | cell11 role1, `0x0000400000000000` |
| Difference profile | `uknit64_k1q_cell11_r5` |
| Seeds | `3`, `4` |
| Train samples | `2048/class` (`4096` total rows) |
| Same-key fresh | `1024/class` (`2048` total rows) |
| Cross-key validation | `1024/class` (`2048` total rows) |
| Pairs per sample | `4` |
| Feature input | four raw ciphertext pairs as `512` bits/sample |
| Negative definition | encrypted random plaintexts |
| Keys | exact K1-Q seed3/4 train and validation keys |
| Runtime window | `round_start=3`, `rounds=2` |
| Hidden / pair embedding | `32` / `128` |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best cross-key validation AUC, restored |
| Device | local CPU |

The matrix contains exactly eight independently trained rows:

```text
seed3/4 x {
  exact_composition,
  wrong_sbox_semantics,
  no_sbox_composition,
  no_topology
}
```

The train and cross-key caches must be reused for all eight rows without any
regeneration. The same-key fresh caches are loaded directly from the K1-Q
manifest for post-training evaluation. Every condition for a given seed/split
must consume the same dataset digest.

## 5. Readiness And Protocol Gate

Before interpreting metrics require:

1. exact K1-Q source hashes, pass decision and confirmed cell11;
2. exactly eight tasks covering seed3/4 and all four conditions;
3. all tasks bind input difference `0x0000400000000000` and the frozen keys;
4. all models have the same `131875` trainable parameters and strict checkpoint
   loading succeeds;
5. six K1-Q cell11 caches pass row-count and content-digest verification;
6. all sixteen matrix cache accesses are reuse events and none regenerate;
7. exactly eight training rows and twenty-four three-split evaluation rows;
8. every cross-key post-training AUC exactly replays its training-result AUC;
9. validation checkpoint selection is `val_auc` with the best checkpoint
   restored, and every row runs ten epochs;
10. all reported metrics are finite and every compared condition uses the same
    dataset digest for its seed and split.

Any failed item makes the result `invalid` and authorizes only repair and an
unchanged rerun.

## 6. Advance Gate

Apply every threshold separately to seed3 and seed4 on both fresh splits. No
average may hide a failed seed or split.

```text
exact_composition AUC                         >= 0.550
exact_composition - no_topology AUC          >= +0.010
exact_composition - wrong_sbox_semantics AUC >= +0.005
exact_composition - no_sbox_composition AUC  >= +0.005
```

Training-split metrics diagnose memorization but do not satisfy a fresh gate.

## 7. Decisions And Required Next Action

- **All gates pass:** retain the exact-composition mechanism and preregister a
  separate remote `65536/class` medium diagnostic against only the strongest
  same-budget controls. Require disk-backed cache/progress/resume before launch;
  do not call it formal or paper-scale evidence.
- **Exact learns fresh signal but misses structure margins:** keep the cell11
  data route, reject the current claim that K1-N structure causes the gain, and
  redesign one structure interaction locally before scale. Do not increase
  data, pairs, width or experts.
- **Same-key passes but cross-key fails:** classify the result as key-specific;
  hold scale and test one key-invariance change with the same data budget.
- **Exact remains below `0.550` although K1-Q Fisher is strong:** classify the
  neural representation/optimization as the bottleneck and isolate access to
  the five-stage statistic before another architecture family.
- **Protocol invalid:** repair only the failed binding or implementation and
  rerun unchanged.

Do not add cell0, scan more positions or bit roles, add pairs, add samples,
change the negative class, add MoE, DDT/trail inputs, switch architectures or
start remote training inside K1-R.

## 8. Run ID And Required Artifacts

```text
run_id = i1_uknit_family_ctspn_cell11_neural_attribution_
         k1r_2048_seed3_seed4_20260728
```

Required artifacts are preflight, filtered dataset manifest, eight checkpoints,
training results, twenty-four evaluation rows, split attribution CSV, checkpoint
manifest, gate, validation, summary, history, progress, Chinese explanatory SVG
and visual-QA report/marker. After completion, refresh
`outputs/00_RECENT_RESULTS.md` and `outputs/00_RECENT_RESULTS.json` in the same
turn and record the evidence-backed next action here.

## 9. Completed Result

K1-R completed the exact frozen matrix and all post-training evaluation rows:

```text
training rows = 8 / 8
evaluation rows = 24 / 24
checkpoint rows = 8 / 8
validation status = pass
failed protocol checks = []
training/validation cache reuse events = 16 / 16
cache regeneration = none
```

Fresh AUC and exact-structure margins were:

| Seed | Split | Exact AUC | Wrong S-box | No S-box | No topology | Exact - wrong S-box | Exact - no S-box | Exact - no topology |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | same-key fresh | `0.522282` | `0.517742` | `0.518846` | `0.501656` | `+0.004539` | `+0.003436` | `+0.020626` |
| 3 | cross-key | `0.493238` | `0.499823` | `0.499725` | `0.517068` | `-0.006585` | `-0.006486` | `-0.023830` |
| 4 | same-key fresh | `0.513252` | `0.511323` | `0.515407` | `0.469134` | `+0.001929` | `-0.002155` | `+0.044117` |
| 4 | cross-key | `0.516200` | `0.513804` | `0.516692` | `0.512249` | `+0.002396` | `-0.000492` | `+0.003952` |

Every exact fresh AUC missed the preregistered `0.550` floor. Exact structure
also missed both semantic margins in every fresh row, and no-topology exceeded
exact by `0.023830` on seed3 cross-key. The final exact residual gate remained
open at approximately `0.04835`, so this is not another zero-gate result.

The frozen decision is:

```text
status = hold
decision = innovation1_uknit_family_ctspn_k1r_cell11_neural_signal_not_supported
remote_scale = no
```

This result separates two earlier questions. K1-Q proved that cell11 has a
strong deterministic five-stage r5 relation (`0.806228-0.825591` on untouched
fresh seed/split combinations), so the old `0x40` position was indeed a major
confounder. K1-R now shows that merely feeding the same exact inverse stages to
the current K1-N architecture does not make that relation learnable or
structure-attributable at this local budget. It does not prove a uKNIT r5 or
neural ceiling.

The final Chinese heatmap passed rendered-pixel `visual-qa-redraw` at
`1920x1032`. The first rendering was rejected because the right colorbar label
was clipped; the corrected rendering has no text overlap, clipping, missing
glyphs, ambiguous scale or insufficient separation.

## 10. Recommended Next Action: K1-S Representation-Access Audit

K1-S should answer one narrower question before any architecture redesign:

> At which exact stage does the confirmed K1-Q position-preserving signal
> disappear inside the completed K1-R exact-composition model?

Reuse only the completed seed3/4 exact K1-R checkpoints and the six K1-Q cell11
caches. Do not retrain the network. For every sample, extract four frozen taps:

```text
T0 = K1-Q exact five-stage position histogram (deterministic upper anchor)
T1 = composition_bit_encoder output before cell aggregation, position preserved
T2 = topology_delta before invariant_pool, cell position preserved
T3 = invariant_pool(topology_delta) / final residual embedding, position erased
```

Fit the same diagonal Fisher scorer independently at each tap on the frozen
train split and evaluate same-key fresh and cross-key rows. Add a deterministic
label-shuffle scorer at every tap. The only variable is the observation tap;
datasets, checkpoints, labels, scorer, variance floor and fresh splits remain
identical. Execution is local, zero optimizer steps and zero neural epochs.

Apply these gates separately to both seeds and both fresh splits:

```text
T0 exactly replays the K1-Q confirmed AUC/dataset digests
each interpreted tap - its label-shuffle AUC >= +0.030
candidate position-preserved tap AUC            >= 0.550
candidate position-preserved tap - T3 AUC       >= +0.030
```

- If T1 or T2 passes while T3 fails, the invariant readout is the isolated
  bottleneck; only then design an active-relative, runtime-topology-derived
  position-preserving readout.
- If T1 already loses the T0 signal, redesign the exact bit encoder or retain a
  deterministic histogram residual; do not change pooling first.
- If T0 does not replay K1-Q exactly, mark K1-S invalid and repair source/tap
  binding only.
- If T3 retains signal but final logits do not, isolate residual fusion and
  classifier scaling; do not add capacity or data.

Do not start remote training, add samples, pairs, positions, MoE, a new cipher,
or another neural architecture until K1-S localizes the first destructive
representation stage.
