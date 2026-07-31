# Innovation 1 K1-BV: uKNIT r6 Pair-Amplification Gate

## Status

`preregistered / zero-training readiness passed / remote launch handed off`

## Research question

Existing uKNIT r6 evidence does not support the four-pair route. K1-BR trained
the exact, wrong-S-box and invariant structure models at `262144/class` with
four ciphertext pairs per sample; their cross-key AUCs were
`0.500423/0.503789/0.500421`. Its fallback local gate was protocol-invalid
because a remote checkpoint path was tested as a local path, but the retrieved
metrics themselves contain no weak positive signal.

The untested hypothesis is pair amplification. At r5, changing only four pairs
to sixteen pairs raised the exact structure expert from AUC
`0.713162/0.748229` to `0.902423/0.932539` on seeds 3/4, while the sixteen-pair
wrong-S-box controls remained near chance. K1-BV asks whether the same query
increase exposes any reproducible r6 signal.

## Frozen matrix

```text
run_id              = i1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4_20260731
cipher / rounds      = uKNIT-BC / 6
difference           = cell11 role1 / 0x0000400000000000
runtime window       = final two rounds / round_start=4, runtime_rounds=2
conditions           = exact-4pair, exact-16pair, wrong-Sbox-16pair
seeds                = 3,4
train                = 2048/class = 4096 total rows per condition/seed
cross-key validation = 1024/class = 2048 total rows per condition/seed
negative definition  = encrypted random plaintexts
sample structure     = independent pairs
epochs / batch       = 10 / 64
loss / optimizer     = MSE / Adam
learning rate        = 1e-4
weight decay         = 1e-5
checkpoint           = restore best validation AUC
execution            = zero-training local readiness, then remote A6000 diagnostic
```

The only research variable is the number of ciphertext pairs. Exact-4pair is
the same-budget anchor. Wrong-Sbox-16pair has the same input width, parameter
geometry and optimizer budget as exact-16pair, so it tests whether any gain
requires the correct public uKNIT nonlinear semantics rather than more input
bits alone. Native-position erasure is omitted because K1-BR already showed no
r6 exact-versus-invariant margin at the much larger four-pair scale.

The eight parameter-matched datasets are two pair-counts x two splits x two
seeds. Exact-16pair and wrong-Sbox-16pair must share their cached train and
validation data, producing exactly four cache reuses. Every cache must contain
`features.npy`, `labels.npy`, `metadata.json` and durable progress events.

## Readiness gate

Before optimization require:

1. exactly six frozen plan rows;
2. exact-4pair decodes 512-bit input as four 128-bit pairs;
3. both sixteen-pair rows decode 2048-bit input as sixteen pairs;
4. all rows retain `214316` trainable parameters and identical state geometry;
5. exact-16pair and wrong-Sbox-16pair differ under a shared state dictionary;
6. outputs and one MSE backward pass are finite;
7. the K1-V r5 pair-amplification source and K1-BR r6 four-pair result are bound
   as context, without treating the protocol-invalid K1-BR gate as authority.

Local CUDA is unavailable in the current environment. Readiness performs no
optimizer step locally; the six-row diagnostic runs on a remote A6000 from an
exact pushed commit.

Readiness completed on 2026-07-31 without training:

```text
plan rows                         = 6
exact-4pair input                 = 512 bits
exact/wrong-16pair input          = 2048 bits
trainable parameters per model    = 214316
identical model state geometry    = pass
shared-state exact/wrong differs  = pass
finite forward and MSE backward   = pass
local training performed          = no
remote configuration              = pass
selected remote device            = physical GPU0 / RTX A6000
```

The bounded pre-launch inspection found both remote A6000 devices idle and the
K1-BV run root absent. This is readiness evidence only; no K1-BV metric exists
until the remote six-row matrix completes and is retrieved.

## Remote handoff

The scoped source commit was published and independently verified on GitHub:

```text
source commit = 1e74787740070ec92c942864c062d5479d4eb158
remote main   = 1e74787740070ec92c942864c062d5479d4eb158
device        = physical GPU0 / RTX A6000
launch        = cmd.exe /c scheduled task from a clean run-owned clone
monitor       = i1_k1bv_2048_monitor
```

The launch command returned successfully and both clean-clone HEAD checks
matched the source commit. Two immediate post-launch SSH attempts failed at the
connection layer before a marker could be read. Therefore the current state is
`launch returned / started marker awaiting monitor confirmation`, not yet
`running confirmed`. The local tmux monitor owns subsequent connection retry,
completion detection, verified-branch or raw fallback retrieval, local
re-adjudication, plotting, and result indexing; the main workflow must not poll
the remote job.

### First-launch readiness failure

The monitor subsequently retrieved a fail marker. The first launch stopped
before dataset generation or optimization:

```text
remote config readiness = pass
torch / CUDA             = 2.5.1+cu118 / CUDA 11.8
visible GPU              = RTX A6000
training rows completed  = 0
failure                  = ModuleNotFoundError: blockcipher_nd
failing entrypoint       = scripts/check-uknit-r6-pair-amplification-k1bv
```

Cause: the new experiment wrappers omitted the standard repository `src/`
bootstrap used by `scripts/train`, `scripts/validate-results`, and
`scripts/check-remote-readiness`. This is a launch-compatibility failure, not a
model, dataset, or r6 evidence result. The repair adds the standard bootstrap to
all K1-BV wrappers and sets `PYTHONPATH=%SOURCE_ROOT%\src` in the remote run
script. The frozen plan, data, keys, pair counts, models, seeds, epochs, and
gates remain unchanged for the retry.

The verified repair commit and retry handoff are:

```text
repair source commit = 1d7a2ea00ad205a39ba3e5295233b399b0bdf6eb
remote main          = 1d7a2ea00ad205a39ba3e5295233b399b0bdf6eb
launch gate          = pass / should_ssh=true / ssh_allowed=true
remote checkout      = detached at 1d7a2ea0
retry launch         = returned successfully on physical GPU0
local monitor        = i1_k1bv_2048_monitor (independent tmux socket)
```

The immediate retry confirmation again failed at the SSH connection layer
before reading a marker. The monitor therefore owns confirmation and retrieval;
the retry remains `launch returned / remote start awaiting monitor evidence`
until a new started, progress, result, or fail artifact is retrieved locally.

## Result gate

For each seed independently calculate:

```text
signal       = exact16 AUC
pair gain    = exact16 AUC - exact4 AUC
semantic gap = exact16 AUC - wrongSbox16 AUC
```

Evidence tiers:

```text
strong candidate = signal >= 0.550, pair gain >= +0.010, semantic gap >= +0.010
weak candidate   = signal >= 0.510, pair gain >= +0.010, semantic gap >= +0.010
unsupported      = either seed misses the weak gate
```

Both seeds must pass the same tier; means cannot hide a failed seed. A strong
two-seed pass authorizes a separately preregistered remote `65536/class`
confirmation. A weak pass authorizes only a fresh-seed sub-medium confirmation.
An unsupported result closes pair amplification for this frozen r6 difference
and network; do not mechanically add data, pairs, epochs or capacity.

## Claim scope and next action

K1-BV is a sub-medium query-budget diagnostic on a public reduced-round cipher.
It is not formal-scale, paper-scale, SOTA, universal-r6, transfer, or topology-
causal evidence. If strong, run the same three conditions at `65536/class` on
seeds 3/4. If weak, confirm seeds 5/6 at `2048/class`. If unsupported, return to
the observed r5-to-r6 boundary and require a new data/representation hypothesis
before another remote slot.
