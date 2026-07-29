# Innovation 1 K1-BJ Multi-Shuffle Cell-Joint Null Audit

**Status:** completed / pass
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_multishuffle_cell_joint_null_k1bj_audit_replica0_replica1_20260729`

## 1. Research Question

K1-BI changed only the representation from independent bit means to native
four-bit cell value histograms. Midori-64 r4 increased to `0.918214-0.930943`,
Dialga-128 r4 remained at `0.994392-0.997225`, and every uKNIT-BC r5 panel
remained below `0.502`. The symmetric single-shuffle gate nevertheless failed
because six Midori/Dialga shuffled-label scorers produced reversed AUC below
`0.47`.

K1-BJ asks:

> Is the correct cell-joint scorer's orientation-invariant strength larger
> than a preregistered empirical distribution of shuffled-label scorers, or is
> the apparent Midori/Dialga signal explainable by the same random orientation
> effect?

## 2. Only Changed Variable

K1-BJ reuses the exact K1-BI cell-joint feature, datasets, rounds, differences,
keys, four pairs, splits and diagonal Fisher settings. It changes only the
label-control statistic:

```text
K1-BI: one shuffled-label scorer, require 0.47 <= AUC <= 0.53
K1-BJ: 31 frozen shuffled-label scorers,
       null strength = abs(AUC - 0.5),
       empirical p = (1 + count(null >= correct strength)) / 32
```

For each `replica x cipher`, permutation indices `0..30` use:

```text
seed = 84100 + replica*10000 + cipher_index*100 + permutation_index
cipher order = uknit64, midori64, dialga128
```

Seeds may not be replaced after inspecting results. Each shuffled scorer is fit
on the same 4096-row correct-operator `train_seen` feature matrix and evaluated
on both unchanged 2048-row fresh splits. K1-BI's true-label scorer is replayed
from the same features and must reproduce every source correct AUC within
`1e-7`.

## 3. Budget And Artifacts

```text
2 replicas x 3 ciphers x 31 shuffled scorers = 186 null scorers
6 true-label replay scorers
12 true-label replay results
372 shuffled fresh results
18 correct-feature manifests
neural parameters = 0
optimizer steps = 0
device = local CPU
```

Expected totals:

```text
feature_manifest.jsonl = 18
scorers.jsonl          = 192
results.jsonl          = 384
```

## 4. Frozen Gates

Protocol gates require exact K1-BI config and ten-artifact binding, including
its visual-QA pass; 31 distinct count-preserving permutations per
replica/cipher; exact feature/scorer/result row sets; finite metrics; exact
correct-AUC replay; zero neural updates; and no data generation.

For each of the eight Midori/Dialga fresh panels require:

```text
empirical p <= 0.05
correct strength - null 95th percentile >= 0.10
```

The 95th percentile uses NumPy's frozen `higher` method. For all four uKNIT
panels require only the already-preregistered boundary:

```text
correct AUC < 0.55
```

This audit does not relabel an AUC below `0.5`; both correct and null evidence
use `abs(AUC - 0.5)`.

## 5. Decisions

- **Midori/Dialga attributed and uKNIT below `0.55`:** the linear transport
  boundary is confirmed. Stop linear-only redesign and preregister the
  runtime S-box-aware five-stage native-cell primitive supported by K1-Q/K1-S.
- **Midori/Dialga attribution fails:** hold family-wide structure claims and
  audit Fisher/null validity without changing features or benchmark variables.
- **Any uKNIT panel reaches `0.55`:** re-adjudicate the K1-BI representation
  result before architecture work; this would contradict the bound source row.
- **Protocol failure:** repair only the failed binding, replay, permutation or
  artifact invariant and rerun unchanged.

No outcome authorizes neural training, remote GPU, additional pairs/data,
MoE, cipher-specific modules or difference rescanning.

## 6. Required Visualization

The Chinese figure must show, for every fresh panel, the correct
orientation-invariant strength against the 31-shuffle distribution, empirical
p-values, and margin over the null 95th percentile. It must explain the cipher
rounds and zero-neural scope, render at `2700 x 1800`, and pass
`visual-qa-redraw` before the result is complete.

## 7. Recommended Next Action

Run K1-BJ locally without changing any frozen variable. Use its decision table
to select the next primitive; do not treat a single reversed shuffled AUC as
either proof against the representation or permission to bypass attribution.

## 8. Completed Result

K1-BJ completed with exact source replay and no protocol or research failure:

```text
feature manifests = 18 / 18
scorer rows       = 192 / 192
result rows       = 384 / 384
maximum K1-BI correct-AUC replay delta = 0.0
failed protocol checks = []
failed research checks = []
status   = pass
decision = innovation1_uknit_family_k1bj_linear_transport_boundary_confirmed
```

Every Midori/Dialga panel achieved the smallest empirical p-value possible
with 31 permutations:

```text
empirical p = 1 / 32 = 0.03125
```

Correct orientation-invariant strength minus the shuffled-null 95th
percentile was:

| Cipher / rounds | replica0 same-key | replica0 cross-key | replica1 same-key | replica1 cross-key |
|---|---:|---:|---:|---:|
| Midori-64 r4 | `+0.369180` | `+0.343306` | `+0.358328` | `+0.333563` |
| Dialga-128 r4 | `+0.340734` | `+0.357883` | `+0.350929` | `+0.360096` |

All eight margins exceed the frozen `+0.10` gate by more than a factor of
three. The reversed single-shuffle AUCs in K1-BI were therefore a random
orientation/control-statistic issue, not an explanation of the much stronger
Midori/Dialga correct-operator signal.

The four uKNIT AUCs replayed exactly as:

```text
0.499398 / 0.492171 / 0.469312 / 0.501842
```

All remain below `0.55`. Three panels were plainly inside the multi-shuffle
null (`p=0.96875/0.4375/0.90625`). Replica1 same-key had a weak reversed
orientation-invariant excess (`p=0.03125`), but its flipped-equivalent AUC is
only `0.530688`, still below the frozen signal floor and not replicated on the
cross-key split. This is weak sub-threshold evidence, not a usable uKNIT r5
representation.

## 9. Interpretation And Next Experiment

K1-BJ confirms a representation boundary rather than a universal failure:

```text
exact runtime GF(2) transport + native 4-bit cell categories
  -> strongly attributed for Midori-64 r4
  -> strongly attributed for Dialga-128 r4
  -> insufficient and unstable for uKNIT-BC r5
```

The linear-only response branch is stopped. The next experiment must
preregister a runtime S-box-aware five-stage native-cell primitive. It should
reuse K1-Q/K1-S's verified uKNIT cell11 mechanism, preserve exact transported
cell categories for Midori/Dialga, and introduce only the runtime S-box stage
as the new semantic variable. The same wrong-S-box, wrong-linear, identity and
multi-shuffle controls remain mandatory.

Do not start neural training yet. First prove that one shared deterministic
S-box-aware primitive retains the uKNIT r5 anchor while preserving the
Midori/Dialga K1-BI anchors. Only a passing cross-cipher primitive gate may
authorize a shared position-preserving neural residual readiness design. Pair
expansion, data scaling, remote GPU, MoE and cipher-specific heads remain
blocked.
