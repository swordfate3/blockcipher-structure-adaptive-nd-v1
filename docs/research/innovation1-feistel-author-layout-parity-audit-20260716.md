# Innovation 1 Feistel Author-Layout Parity Audit

**Date:** 2026-07-16

**Scope:** Lu et al. ordinary-key SIMON/SIMECK data and network mechanism. This
is a source/code audit, not a paper-scale reproduction or performance claim.

## Audit Outcome

The source-verified field equations are correct, but the first project model
did not match the public network's pair-axis layout. The audit also found and
repaired a project key-indexing bug before accepting any experiment result.

## Data And Field Parity

| Item | Lu paper/public code | Corrected project | Verdict |
|---|---|---|---|
| positive row | all 8 pairs use `(0,0x40)` | same | match |
| negative row | second plaintext independently random then encrypted | same strict mode | match |
| key | one random key/sample shared by 8 pairs | global row-indexed random key shared by 8 pairs | match after repair |
| raw split | left word then right word | high 32 bits then low 32 bits | match |
| bit order | MSB first within every word | MSB first | match |
| field order | delta L/R; C0 L/R; C1 L/R; previous deltas | same 8 fields | match |
| round function | SIMON `(rol8 & rol1) xor rol2`; SIMECK `(rol5 & x) xor rol1` | same | match |

The public repository implements executable generators for the 16-bit-word
SIMON32/64 and SIMECK32/64 variants. The 32-bit-word SIMON64/128 and
SIMECK64/128 formulas and results are in the paper, but no corresponding public
64-bit generator file was present at audited commit
`602c664e649a4e3e8e56dc1961efb67400f5c7fb`.

## Corrected Key-Sampling Bug

Before repair, balanced project generation called both classes with row indices
`0..N-1`. Because rotating keys are deterministic functions of row index,
positive row `i` and negative row `i` reused one key. The fix uses:

```text
positive key rows = 0 .. N-1
negative key rows = N .. 2N-1
```

Both memory and disk-backed generators now agree, rotating-key caches are
invalidated by the `global_dataset_row` metadata marker, and the Feistel gate
rejects results missing this marker. All reported SIMON/SIMECK results were
regenerated after this fix.

## Architecture Mismatch

The public effective network reshapes one derived sample as:

```text
[batch, 8 pair positions, 256 derived bits per pair]
```

It applies a pointwise Conv1D stem, two pointwise Dense transitions, five
SE-ResNet blocks across the eight pair positions, then flattens all positions
before two 64-unit Dense layers.

The first project candidate instead uses:

```text
[batch x 8 pairs, 8 word channels, 32 bit positions]
shared per-pair residual encoder
mean + max aggregation across pairs
```

It therefore convolves on the bit axis and deliberately removes pair position.
This is a valid structure-adaptive hypothesis, but it is not a close port of the
source network. The low-round result shows that its correct relation is useful
(`+0.04888` SIMON and `+0.07431` SIMECK over shuffled), while its absolute AUC
remains below the frozen `0.60` calibration gate.

## Decision Unlocked

Implement one source-layout repair that preserves the corrected data protocol
and exact relation fields. Compare correct versus branch-shuffled source-layout
models and the current correct-relation pair encoder at the same easier-round
budget. Do not add samples, remote GPU work, related keys, or another feature
route until that architecture-only comparison is adjudicated.

The comparison was completed: the Lu-layout true models reached only
`0.509408` on SIMON r11 and `0.516449` on SIMECK r14, below the pair-pool
anchors `0.549645` and `0.574093`. The architecture repair was therefore
rejected at `2048/class`; subsequent scale and target-round work retained the
pair-pool encoder.
