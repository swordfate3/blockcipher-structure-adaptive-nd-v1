# Innovation 1 uKNIT-Family CT-SPN Position/Cell Attribution K1-J

**Date:** 2026-07-28
**Status:** frozen before implementation
**Execution:** local CPU, zero-training mechanism audit

## 1. Decision Context

K1-I repaired the GF(2) information loss exposed by K1-H. On Dialga-128 r4,
the exact Boolean-view candidate recovered `0.954252-0.965361` AUC while the
no-topology control remained at `0.518332-0.531655`. Reversed and corrupted
operators also reached approximately `0.948-0.960`, however, and five of six
Dialga split/seed rows missed the frozen `+0.005` correct-operator margin.
uKNIT-BC r5 fresh-same-key remained at `0.508171/0.495883`.

The K1-I representation computes correct Boolean operators but then applies one
shared bit encoder and immediately pools over bits and cells. K1-J asks:

> Is Dialga's recovered K1-I signal carried by global per-bit marginals,
> correct within-cell co-occurrence, or a joint dependency between those two
> branches, and does the frozen Runtime-E4 anchor retain input-coordinate
> sensitivity that the invariant K1-I representation discards?

This question must be answered before designing another trained network. K1-J
does not test more capacity, another optimizer, an S-box path or more data.

## 2. Frozen Sources

Only Dialga-128 is audited because it supplies a calibrated strong signal under
both source architectures. The sources are:

```text
K1-I root = outputs/local_diagnostic/
  i1_uknit_family_ctspn_gf2_boolean_view_k1i_2048_seed0_seed1_20260728

Runtime-E4 root = outputs/local_diagnostic/
  i1_uknit_family_ctspn_linear_schedule_k1_2048_seed0_seed1_20260727
```

Required source digests at freeze time:

| Artifact | SHA-256 |
|---|---|
| K1-I gate | `e1823155149ce6146358650ae711269b617c93f4f7d48aaaa3e231348bfd675d` |
| K1-I checkpoint manifest | `4def7bc0019d7a258d962c622cfc79db1b69e0f85dc0b491a17bf081683e465f` |
| K1-I dataset manifest | `ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0` |
| Runtime-E4 checkpoint manifest | `517f2fd2eb6d401983ca20c9db136229cf5b011c51ef3f1734cdb46e41967aeb` |

The exact selected Dialga checkpoints are:

| Model | Seed | SHA-256 |
|---|---:|---|
| K1-I exact GF(2) | 0 | `3a192102d4fd2214faf9856ec046ed7577f3d4ec8ac2638b123da9b06257a3d1` |
| K1-I exact GF(2) | 1 | `36a7e1db6342a6cafe79fff454f7260b7c621ceba10bd7787b3591abb3bb9c75` |
| Runtime-E4 | 0 | `5910dd24a360e08a92014275f629772e0ebf215a580566dc7862c3366ea3c812` |
| Runtime-E4 | 1 | `b8a8d49ccdaad026e852fa0542af7b1a7d4af34ca96b9e1aebb513648b1a1ef8` |

## 3. Frozen Data And Baselines

Reuse the six existing Dialga caches without regeneration:

| Seed | Split | Rows |
|---:|---|---:|
| 0, 1 | train seen | `4096` each (`2048/class`) |
| 0, 1 | same-key fresh | `2048` each (`1024/class`) |
| 0, 1 | cross-key validation | `2048` each (`1024/class`) |

The label definition remains strict encrypted-random-plaintext negatives, four
ciphertext pairs per sample, fixed positive input difference `0x40`, and the
same fixed train/cross-key keys used by K1-I. No label, row order, checkpoint or
metric may be regenerated or selected after seeing the audit.

For every seed and split, the native K1-I, no-topology K1-I and native
Runtime-E4 AUC must replay the corresponding K1-I `controls.jsonl` row within
`1e-7` before any attribution result is accepted.

## 4. K1-I Aggregation Interventions

The exact K1-I checkpoint and exact ordered matrices are fixed. Extract the
per-pair bit-hidden tensor immediately after the shared 12-channel bit encoder.
Change only how that frozen tensor is grouped or which pooled branch remains
associated with the current sample:

| Condition | Single intervention | Purpose |
|---|---|---|
| `native` | none | exact baseline |
| `within_cell_role_roll` | cyclically reorder the four hidden bit roles inside every native cell | exact invariance control for cell mean |
| `whole_cell_roll` | cyclically reorder complete four-bit cells | exact invariance control for cell-set pooling |
| `cross_cell_role_mix` | retain every bit and role but source the four roles of one synthetic cell from four different native cells | destroy within-cell co-occurrence while preserving global bit marginals |
| `bit_pool_row_shuffle` | deterministic label-blind sample permutation of the global-bit pooled branch only | remove bit-branch label association |
| `cell_pool_row_shuffle` | the same sample permutation of the cell pooled branch only | remove cell-branch label association |
| `both_pool_row_shuffle` | permute both branches together | chance-level sanity control |

Row shuffling applies one deterministic nonidentity sample permutation to all
four pairs of a sample. It must not inspect labels. The same permutation is
used for both model roles and recorded by digest.

The first two non-native grouping controls must reproduce native probabilities
within `1e-7`. Any larger change means the audit implementation does not match
the declared K1-I pooling computation.

## 5. Runtime-E4 Position Reference

Apply the same four bijective input-coordinate conditions to both frozen K1-I
and Runtime-E4 checkpoints before either model executes its native forward
path:

```text
native_input
within_cell_input_role_roll
whole_cell_input_roll
cross_cell_input_role_mix
```

Each operation reorders both ciphertext endpoints of every pair identically and
preserves all input bits. The runtime descriptor is not relabeled. These rows
are supporting sensitivity evidence: they measure dependence on native input
coordinates, not cryptographic-equivalent joint relabeling. They cannot by
themselves prove correct-operator semantics.

## 6. Zero-Training Readiness Gate

Before accepting the audit require:

1. the K1-I source gate has the exact frozen hold decision and all protocol
   checks pass;
2. all four source manifests have the frozen SHA-256 values above;
3. the two K1-I and two Runtime-E4 Dialga checkpoints strict-load from their
   manifest-bound paths and preserve state-dict hashes before and after audit;
4. exactly six digest-bound Dialga caches load with the frozen row counts;
5. native K1-I, no-topology and Runtime-E4 AUC replay source controls within
   `1e-7` on every seed/split;
6. the manually exposed K1-I pooled path reproduces the normal model forward
   probabilities within `1e-7`;
7. within-cell and whole-cell hidden permutations are bijective, distinct and
   reproduce native probabilities within `1e-7`;
8. cross-cell mixing is bijective, role preserving, globally multiset
   preserving and changes native cell membership;
9. input-coordinate controls are bijective, nonidentity and identical for both
   model roles;
10. row shuffling is deterministic, label blind, nonidentity and digest-bound;
11. every metric is finite and every evaluation row records
    `training_performed=false`, `optimizer_steps=0` and strict checkpoint load;
12. no dataset cache is created or modified and the audit consumes zero new
    training rows.

Any failure permits only repair of the failed binding or intervention. It does
not authorize training or a substitute architecture.

## 7. Attribution Quantities And Gates

For each K1-I intervention define the source gap and explained fraction:

```text
source_gap = native_exact_auc - no_topology_auc
raw_explained_fraction =
  (native_exact_auc - intervention_auc) / source_gap
explained_fraction = clip(raw_explained_fraction, 0, 1)
```

The source gap must be positive. All advance gates apply separately to both
seeds and both fresh splits; train-seen rows are descriptive only.

- **Within-cell interaction supported:** `cross_cell_role_mix` explains at
  least `80%` of the source gap in all four fresh rows.
- **Global-bit branch supported:** `bit_pool_row_shuffle` explains at least
  `80%` in all four fresh rows.
- **Cell branch supported:** `cell_pool_row_shuffle` explains at least `80%` in
  all four fresh rows.
- **Distributed branch interaction supported:** neither single branch reaches
  `80%`, but `both_pool_row_shuffle` reaches `80%` and has AUC in `[0.45, 0.55]`
  in all four fresh rows.

At least one attribution family must pass consistently. No average may hide a
failed seed or split. Runtime-E4 versus K1-I input-coordinate sensitivity is
reported per row but is not allowed to rescue a failed internal attribution
gate.

## 8. Decisions

- **Within-cell interaction passes:** design K1-K as a position-preserving exact
  GF(2) cell-token network. Keep the four bit roles together through a shared
  cell encoder and defer pooling until after transition mixing.
- **Global-bit branch passes alone:** the recovered signal is a global Boolean
  marginal shortcut. Hold K1-I; K1-K must add native endpoint/cell interaction
  controls before any training.
- **Cell branch passes alone:** preserve learned within-cell tuple features and
  test one exact edge-conditioned cell mixer.
- **Distributed interaction passes:** preserve both branches with an explicit
  gated residual whose ablation is preregistered; do not add an unrestricted
  concatenation bypass.
- **No family passes:** close invariant GF(2) pooling at this diagnostic scale
  and make the next local single-variable experiment exact heterogeneous
  S-box/operator composition.
- **Protocol invalid:** repair and rerun K1-J unchanged.

K1-J cannot authorize remote training, additional data, more epochs, width,
pairs, seeds, MoE experts, S-box/DDT/trail features, partial decryption, keys,
cipher IDs or a raw bypass.

## 9. Run ID And Required Artifacts

```text
run_id = i1_uknit_family_ctspn_position_cell_attribution_k1j_20260728
output = outputs/local_audit/
  i1_uknit_family_ctspn_position_cell_attribution_k1j_20260728
```

Required artifacts:

```text
preflight.json
pool_attribution.jsonl
input_position_controls.jsonl
attribution.csv
summary.json
validation.json
gate.json
progress.jsonl
curves.svg
plot_report.json
visual_qa_passed.marker
```

The completed result must refresh `outputs/00_RECENT_RESULTS.md` and JSON, and
this document must record exact metrics, claim scope and the evidence-backed
next action before the audit is considered handled.
