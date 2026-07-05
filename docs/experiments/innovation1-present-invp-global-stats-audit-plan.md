# Innovation 1 PRESENT InvP Global Statistics Audit Plan

**Date:** 2026-07-06

**Status:** local audit planned / no neural training / no remote launch

## Question

The SGP raw-axis and grouped-axis audits found broad weak separation in
`InvP(delta)` but no stable localizable axis, cell, or orbit mask. This audit
asks a narrower follow-up:

```text
Can explicit pair/global activity statistics over InvP(delta) produce stable
same-protocol local evidence that is more suitable than a projection mask?
```

This is a deterministic feature audit only. It does not train a neural model
and is not scale evidence.

## Protocol

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_global_stats_audit_r8_local.json
```

Core protocol:

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `8` |
| Samples per class | `2048` |
| Seeds | `0, 1` |
| Key split | `validation` |
| Pairs per sample | `16` |
| Feature source | `ciphertext_xor_spn_paligned_bits` |
| Negative mode | `encrypted_random_plaintexts` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Difference profile | `present_zhang_wang2022_mcnd`, member `0` |

Feature transform:

```text
present_global_pairset_statistics(
  features = InvP(delta) pairset bits,
  pairs_per_sample = 16,
  words_per_pair = 2,
  cells_per_word = 16,
  nibble_bits = 4
)
```

## Gate

Keep the route only if the local audit shows:

```text
top-k oriented statistics composite AUC >= 0.62
top-k statistic-name Jaccard >= 0.35 across seeds
best individual statistic AUC >= 0.55
```

If this passes, the next action is not remote training yet. The next action is
a lean local neural smoke or deterministic-feature baseline comparing the
global-stat representation against existing pairset/global-stats anchors.

If it fails, do not continue the InvP distribution-stat route immediately.

## Claim Scope

This is a local diagnostic. It is not formal PRESENT r8 evidence, not a
breakthrough claim, not an ensemble result, and not a remote launch gate.

## 2026-07-06 Local Result

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --invp-global-stats-config configs/experiment/innovation1/innovation1_spn_present_invp_global_stats_audit_r8_local.json \
  --samples-per-class 2048 \
  --top-k 16 \
  --output outputs/local_audits/i1_present_r8_invp_global_stats_audit_2048.json
```

Result:

```text
decision = invp_global_stats_hold
best_source = invp_delta_global_stats
artifact = outputs/local_audits/i1_present_r8_invp_global_stats_audit_2048.json
```

| Metric | Value |
|---|---:|
| Stat feature dim | `148` |
| Best stat AUC min | `0.5180071592330933` |
| Composite AUC min | `0.5185081958770752` |
| Composite AUC mean | `0.5251665115356445` |
| Top-k Jaccard min | `0.06666666666666667` |

Interpretation:

```text
The existing global activity statistics are too coarse. They average away the
broader cell/bit-role weak signal seen by grouped SGP. This result does not
support a global-stats neural smoke or remote launch.
```

Decision:

```text
do_not_launch_present_pairset_global_stats_from_this_audit
do_not_use_global_stats_as_diverse_expert
```

Next action:

```text
If continuing the statistics route, test a more targeted group-distribution
feature bank over InvP(delta): per scheme cell/word_cell/bit_role/orbit group
activities, distribution variance/span/top-k means, and pair-slot consistency
statistics. This is closer to the grouped SGP evidence than the existing
global-activity branch.
```
