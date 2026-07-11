# Innovation 1 PRESENT InvP State-Matrix Conv2D Design

**Date:** 2026-07-11

**Status:** implementation complete; R0 local readiness passed; R1 seed0
local diagnostic completed; Conv2D route stopped and token-mixer anchor retained

**Claim scope:** completed `8192/class` controlled architecture-attribution
diagnostic under the strict PRESENT-80 r7 Zhang/Wang Case2 protocol; not
formal training, paper-scale evidence, a breakthrough, or a novelty claim

## Decision

The bounded, non-DDT architecture hypothesis was:

> Given the same `InvP(DeltaC)` representation that already has two-seed
> `1000000/class` support, does a small state-matrix Conv2D encoder exploit the
> signal better than the current token mixer, and is any gain tied to the true
> PRESENT inverse-permutation mapping?

This experiment does not resume the stopped E1 graph branch, dense DDT trail
inputs, S-box-prior branch, or public random-ciphertext typed adapter. It does
not change the benchmark to match the running AutoND public-code reproduction.

R1 seed0 has now adjudicated the hypothesis: true-InvP Conv2D preserved the
representation/topology signal but lost to the same-budget token-mixer anchor.
The Conv2D route is stopped; the token-mixer anchor is retained.

## Evidence Basis

The strict-protocol InvP route is the strongest current Innovation 1 anchor:

| Evidence | AUC | Status |
| --- | ---: | --- |
| InvP-only seed0, `1000000/class` | `0.797470988906` | retrieved, validated, plan-aligned |
| InvP-only seed1, `1000000/class` | `0.797347588554` | retrieved, validated, plan-aligned |
| DeltaC-only seed0, `1000000/class` | `0.792064879854` | retrieved, validated, plan-aligned |
| shuffled-P seed0, `1000000/class` | `0.793621524954` | retrieved, validated, plan-aligned |

The minimum InvP AUC exceeds the strongest attribution control by
`+0.003726063600`. This supports true `InvP` alignment as a useful input signal,
but it does not show that the current token mixer is the best network for that
signal.

The literature re-audit ranks a controlled typed-representation/network
comparison above additional graph or DDT scaling. State-matrix Conv2D is prior
art as an architecture family, so the experiment is a comparator and method
calibration step. The defensible Innovation 1 contribution remains the
cipher-spec-driven adaptation and explicit same-input/shuffled-structure
attribution procedure, provided the empirical gates support it.

## Frozen Research Protocol

All rows use the existing strict benchmark:

```text
cipher                     = PRESENT-80
rounds                     = 7
difference profile         = present_zhang_wang2022_mcnd
difference member          = 0
pairs per sample           = 16
feature encoding           = ciphertext_pair_bits
negative mode              = encrypted_random_plaintexts
sample structure           = zhang_wang_case2_official_mcnd
configured train key       = 0x00000000000000000000 (inert placeholder)
configured validation key  = 0x11111111111111111111 (inert placeholder)
configured rotation        = 0 (inert placeholder)
effective key schedule     = per_pair_random
effective key scope        = independent random PRESENT key per basic pair
loss                       = mse
optimizer                  = adam
learning rate              = 0.0001
weight decay               = 0.00001
learning-rate scheduler    = official_cyclic
max learning rate          = 0.002
checkpoint metric          = val_auc
restore best checkpoint    = true
early stopping patience    = 8
early stopping min delta   = 0.0001
primary metric             = validation AUC
```

Validation labels, effective key schedule, negative construction, metric
computation, checkpoint selection, and plan-alignment logic must not change.
For this sample structure, effective key behavior must be verified from the
generated train/validation dataset metadata, not inferred from configured key
fields.

## Input Semantics

The model receives the same raw `16 x 128` ciphertext-pair bits as the current
InvP-only anchor. The typed view is constructed inside the model so the dataset
and cache remain identical across rows.

For each ciphertext pair:

```text
(C, C')
  -> DeltaC = C xor C'
  -> mapping(DeltaC)
  -> 16 nibbles x 4 bits
  -> transpose to 4 bit planes x 16 PRESENT cells
```

The three Conv2D variants differ only in `mapping`:

```text
true InvP   = exact PRESENT inverse permutation
shuffled-P  = existing deterministic seed-20260627 pseudo permutation
DeltaC-only = identity mapping
```

The true and shuffled mappings must reuse the existing verified inverse-index
construction. Tests must compare their generated tensors directly with the
current `_PresentNibblePAlignedSpnEncoder` views. No DDT values, plaintext
metadata, active-cell metadata, S-box lookup probabilities, trail summaries,
or public-code random-ciphertext labels enter the model.

## Candidate Architecture

Add one small pair encoder with three mapping modes rather than three copied
implementations.

Per pair:

```text
[batch, 1, 4, 16]
  -> 1x1 Conv2D, BatchNorm2D, ReLU stem to base_channels
  -> three residual state-matrix blocks
  -> each block uses 3x3 Conv2D, BatchNorm2D, ReLU, Dropout2D,
     3x3 Conv2D, BatchNorm2D, residual addition, and ReLU
  -> mean and max pooling over the 4 x 16 state matrix
  -> projection to the existing InvP anchor embedding width
```

Across the 16 pairs:

```text
pair embeddings
  -> mean pooling over pairs
  -> max pooling over pairs
  -> concatenate
  -> classifier with the same widths and activation family as the InvP anchor
  -> one logit
```

Default model options:

```text
base_channels = 32
conv_depth    = 3
kernel_size   = 3
activation    = relu
norm          = batchnorm2d
dropout       = 0.0
```

The implementation must reuse the existing `conv2d_norm` helper with
`batchnorm2d`; it must not add a normalization wrapper, dependency, or
general-purpose framework. The pair projection is
`Linear(2 * base_channels, 4 * base_channels)` followed by ReLU. With the
default `base_channels=32`, every pair becomes a 128-dimensional embedding.
The pair-level classifier is the same shape as the anchor: normalization over
the concatenated 256-dimensional mean/max embedding, `Linear(256, 256)`, ReLU,
dropout, and `Linear(256, 1)`.

The result row must record total trainable parameters for the anchor and all
candidates. The three Conv2D variants must have exactly equal parameter counts;
the Conv2D candidate does not need to equal the token-mixer parameter count,
but the report must state their ratio so capacity is not hidden.

## Model Matrix

The research matrix contains four rows because this is a planned attribution
study. No unrelated architecture is added.

| Row | Model role | Mapping | Purpose |
| --- | --- | --- | --- |
| 1 | existing `present_nibble_invp_only_spn_only` | true InvP | same-budget strongest anchor |
| 2 | new true-InvP state-matrix Conv2D | true InvP | architecture candidate |
| 3 | new shuffled-P state-matrix Conv2D | deterministic shuffled mapping | topology-attribution control |
| 4 | new DeltaC-only state-matrix Conv2D | identity | generic difference/capacity control |

Frozen model keys:

```text
present_nibble_invp_state_matrix_conv2d_spn_only
present_nibble_shuffled_p_state_matrix_conv2d_spn_only
present_nibble_delta_state_matrix_conv2d_spn_only
```

## Scale Ladder

Scale is evidence-driven rather than automatic:

### R0: implementation readiness

```text
samples_per_class = 64
seed              = 0
epochs            = 1
device            = cpu
```

R0 proves tensor semantics, cache compatibility, finite forward/backward
passes, result serialization, plotting, and plan alignment. Its metrics are not
research evidence.

### R1: local architecture-attribution diagnostic

```text
samples_per_class = 8192
seed              = 0
epochs            = 10
rows              = 4
```

`8192/class` is a diagnostic screen, not formal training and not a definitive
route failure. Run seed1 at the same scale only if seed0 passes the promotion
gate below. R1 must reuse one parameter-matched dataset cache across all rows.

### R2: medium confirmation

Only after both R1 seeds pass:

```text
samples_per_class = 65536, then 262144
seeds             = 0 and 1
```

These are medium diagnostics. A remote R2 launch requires a pushed commit,
disk-backed parameter-matched caches, progress JSONL, readiness validation,
and watcher-managed retrieval under `G:\lxy`.

### R3: formal claim gate

Only after the `262144/class` evidence remains positive:

```text
samples_per_class >= 1000000
seeds             >= 2
```

Only completed, retrieved, plan-aligned R3 evidence can support a formal
architecture result. The experiment still cannot claim that Conv2D itself is
novel; it can support the controlled cipher-spec adaptation result.

## R1 Promotion And Stop Gates

Define, for each seed:

```text
architecture_margin = AUC(true Conv2D) - AUC(token-mixer anchor)
topology_margin     = AUC(true Conv2D) - AUC(shuffled-P Conv2D)
representation_margin = AUC(true Conv2D) - AUC(DeltaC-only Conv2D)
```

Seed0 promotes to seed1 only when all conditions hold:

```text
plan alignment passes for 4/4 rows
all metrics are finite
architecture_margin > 0
topology_margin >= +0.003
representation_margin >= +0.003
true Conv2D is above every control
no cache, label, layout, or checkpoint invariant fails
```

After seed1, R2 is allowed only when:

```text
true Conv2D is above all three comparators on both seeds
mean architecture_margin >= +0.001
minimum topology_margin >= +0.002
minimum representation_margin >= +0.002
```

Interpretation branches:

| Observation | Decision |
| --- | --- |
| candidate beats anchor and both controls on both seeds | keep architecture hypothesis; proceed to R2 |
| candidate beats anchor but not shuffled-P | capacity or generic locality, not true PRESENT structure; stop candidate |
| candidate beats shuffled-P but not DeltaC-only | InvP attribution unsupported in this architecture; stop candidate |
| candidate loses to anchor | token mixer remains stronger; stop Conv2D route |
| one seed passes and one fails | unstable diagnostic; no remote scale, inspect training variance once |
| protocol/layout/cache gate fails | invalid experiment; repair and rerun without interpreting metrics |

The numeric margins are promotion thresholds, not scientific effect-size laws.
Reports must include raw per-seed AUCs, all three deltas, accuracy, loss, epoch
histories, and uncertainty language.

## R0 Local Readiness Evidence (2026-07-11)

Run id:

```text
i1_present_invp_state_matrix_conv2d_smoke_seed0
```

The first exact launch exposed a frozen-plan readiness defect before any row
completed: `balanced_per_class` rows redundantly set `train_samples_total` and
`validation_samples_total`, while the dataset validator reserves those fields
for `random_labels_total`. Commit `f45bbc6` blanked those fields in both the R0
and R1 matrices and updated the focused plan-contract test. This did not change
the budgets: `samples_per_class=64` still creates 128 balanced training rows and
the standard half-size validation rule still creates 64 balanced validation
rows. The focused regression suite passed `38/38` after the repair.

The first completed R0 artifacts then exposed a protocol-reporting mismatch
during specification review. The document incorrectly described the configured
train and validation key fields as fixed effective encryption keys, while
`zhang_wang_case2_official_mcnd` intentionally uses an independent random
PRESENT key for every basic pair. Those earlier artifacts were invalid for the
documented protocol-identity claim because result rows did not serialize the
effective dataset schedule. They remain useful only as the evidence that
revealed the mismatch; they have been deleted and replaced by the clean rerun
below. Commit `1023ced` added train/validation `key_schedule` result metadata
and made the attribution gate require `per_pair_random` for both splits.

The corrected implementation then completed a clean-cache rerun with this exact
command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_smoke_seed0.csv \
  --epochs 1 \
  --batch-size 32 \
  --hidden-bits 32 \
  --device cpu \
  --dataset-cache-root outputs/local_cache/i1_present_invp_state_matrix_conv2d_smoke_seed0 \
  --dataset-cache-chunk-size 64 \
  --dataset-cache-workers 1 \
  --progress-output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/progress.jsonl \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/results.jsonl
```

Observed protocol was PRESENT-80 r7, seed 0, `64/class`, 128 total balanced
training rows, 64 total balanced validation rows, one epoch, batch size 32,
hidden bits 32, 16 pairs/sample, strict encrypted-random-plaintext negatives,
effective `per_pair_random` scheduling with an independent random PRESENT key
for every basic pair, and CPU execution. The configured train key
`0x00000000000000000000`, validation key `0x11111111111111111111`, and rotation
interval `0` remained inert deterministic plan/cache placeholders rather than
effective encryption keys. The exact ordered model keys were:

```text
present_nibble_invp_only_spn_only
present_nibble_invp_state_matrix_conv2d_spn_only
present_nibble_shuffled_p_state_matrix_conv2d_spn_only
present_nibble_delta_state_matrix_conv2d_spn_only
```

All four rows completed one finite forward/backward training epoch and emitted
finite result metrics. For reproducibility, the un-interpreted smoke values
were:

| Model key | Validation AUC | Accuracy | Loss |
| --- | ---: | ---: | ---: |
| `present_nibble_invp_only_spn_only` | `0.5771484375` | `0.5` | `0.7021275758743286` |
| `present_nibble_invp_state_matrix_conv2d_spn_only` | `0.498046875` | `0.5` | `0.7061755955219269` |
| `present_nibble_shuffled_p_state_matrix_conv2d_spn_only` | `0.4228515625` | `0.5` | `0.7033177614212036` |
| `present_nibble_delta_state_matrix_conv2d_spn_only` | `0.3828125` | `0.5` | `0.7206352800130844` |

These metric values are recorded only as finite serialization evidence and are
not interpreted as architecture quality or research evidence.

Plan validation passed with `status=pass`, `plan_rows=4`, `result_rows=4`, and
`errors=[]`. Progress contained four `row_done` events and ended with exactly
one `run_done` event. The corrected attribution gate also returned
`status=pass` and `errors=[]` in explicit readiness-only mode. Its neutral
machine outcome is `decision=implementation_ready`,
`research_decision_applied=false`,
`next_action=run_frozen_r1_seed0_local_diagnostic`, and
`claim_scope=implementation readiness only; metrics not interpreted`. No
stop/promote research decision was applied to R0 metrics. Cache generation
created exactly one train identity and one validation identity:

```text
outputs/local_cache/i1_present_invp_state_matrix_conv2d_smoke_seed0/present80/r7/train/seed-0_b10ab47bffcc5873
outputs/local_cache/i1_present_invp_state_matrix_conv2d_smoke_seed0/present80/r7/validation/seed-10000_88856f69c830db86
```

The first row created the train cache with 128 rows and the validation cache
with 64 rows. Rows 2-4 each reused both exact paths, producing six
`cache_reuse` events. The metadata matched across model rows on cipher, rounds,
seed/split, difference, feature encoding, 16-pair structure, strict negative
definition, label mode, row budget, and effective schedule. All four result
rows serialized `training.key_schedule=per_pair_random` and
`validation.key_schedule=per_pair_random`; both cache metadata files also
reported `key_schedule=per_pair_random`.

The anchor has `128673` total/trainable parameters. Each of the true-InvP,
shuffled-P, and DeltaC-only Conv2D rows has exactly `130881` total/trainable
parameters. All counts are positive and the candidate/anchor capacity ratio is
`130881 / 128673 = 1.0171597771` (approximately `1.01716x`).

R0 artifacts:

| Artifact | Size |
| --- | ---: |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/results.jsonl` | 16282 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/progress.jsonl` | 82406 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/validation.json` | 3750 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/curves.svg` | 55086 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/history.csv` | 1372 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_smoke_seed0/readiness_gate.json` | 25561 bytes |

`claim_scope=implementation readiness only; metrics not interpreted`.

The regenerated readiness gate now contains machine-verifiable
`protocol_evidence`, `semantic_checks`, `cache_evidence`,
`promotion_conditions`, and `stopped_actions`. It records all four exact model
roles and options, full frozen result identities, checkpoint/history status,
the tested state-matrix semantic contract, two cache creates plus six control
reuses across the exact train/validation paths, and one terminal `run_done`.
The cache evidence is now paired with the supplied result file by argument
order and exact seed. It reports the normalized supplied result path/root and
result-declared cache root, requires normalized `run_done.output` to identify
that result exactly, and requires every normalized terminal cache path to stay
under that cache root.

Recommended next action: proceed to the already frozen R1 seed0 local
architecture-attribution diagnostic with `8192/class`, four identical ordered
rows, 10 epochs, and the same-budget token-mixer anchor plus shuffled-P and
DeltaC-only controls. Change only the evidence scale from R0 to R1. Require
4/4 plan alignment, finite metrics, one train cache plus one validation cache
reused across all rows, equal Conv2D parameter counts, and the existing R1
promotion/stop gates. R1 must also require `per_pair_random` in both serialized
result sections and both cache metadata files. Do not launch remote scale, add
models, change the benchmark, or interpret R0 metrics.

## R1 Seed0 Local Diagnostic Evidence (2026-07-11)

Run id:

```text
i1_present_invp_state_matrix_conv2d_8192_seed0
```

The clean worktree at source commit `30f0c882c7f8714afc6be8373aadbed5be03a9f8`
ran the frozen four-row plan with this exact command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_state_matrix_conv2d_8192_seed0.csv \
  --epochs 10 \
  --batch-size 256 \
  --hidden-bits 32 \
  --device cpu \
  --dataset-cache-root outputs/local_cache/i1_present_invp_state_matrix_conv2d_8192_seed0 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --progress-output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/progress.jsonl \
  --output outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/results.jsonl
```

The observed protocol was PRESENT-80 r7, seed 0, `8192/class`, 16384 total
balanced training rows, 8192 total balanced validation rows (`4096/class`), 10
epochs, batch size 256, hidden bits 32, 16 pairs/sample, strict
encrypted-random-plaintext negatives, Zhang/Wang Case2 official MCND sample
structure, and CPU execution. Each basic pair used an independent random
PRESENT key: all four result rows record
`training.key_schedule=per_pair_random` and
`validation.key_schedule=per_pair_random`, and both cache metadata files record
the same effective schedule. The configured train/validation key fields and
rotation interval remained inert plan/cache identity placeholders.

Plan validation passed exactly (`status=pass`, `plan_rows=4`,
`result_rows=4`, `errors=[]`) in this ordered matrix:

```text
present_nibble_invp_only_spn_only
present_nibble_invp_state_matrix_conv2d_spn_only
present_nibble_shuffled_p_state_matrix_conv2d_spn_only
present_nibble_delta_state_matrix_conv2d_spn_only
```

All primary result metrics are finite. The reported metrics are from the
restored best `val_auc` checkpoint:

| Model role | Validation AUC | Accuracy | Calibrated accuracy | Loss | Best epoch | Epochs ran |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| token-mixer anchor | `0.750535935163` | `0.686035156250` | `0.686889648438` | `0.589239429682` | 10 | 10 |
| true-InvP Conv2D | `0.745510458946` | `0.672729492188` | `0.682739257812` | `0.602004538290` | 10 | 10 |
| shuffled-P Conv2D | `0.555524557829` | `0.505737304688` | `0.541015625000` | `0.712105254643` | 7 | 10 |
| DeltaC-only Conv2D | `0.575832307339` | `0.539794921875` | `0.558715820312` | `0.690229361877` | 8 | 10 |

Every row contains a complete 10-epoch history, uses
`lr_scheduler=official_cyclic` (`0.002` at epoch 1 to `0.0001` at epoch 10),
records `selected_checkpoint=best`, and restored a best checkpoint. All rows
use `optimizer_state_transition=reset_each_stage` and the same validation
budget and metric computation.

The seed0 margins are:

```text
architecture_margin   = 0.745510458946 - 0.750535935163 = -0.005025476217
topology_margin       = 0.745510458946 - 0.555524557829 = +0.189985901117
representation_margin = 0.745510458946 - 0.575832307339 = +0.169678151608
```

The true-InvP mapping therefore carries a large signal relative to the
shuffled-P and DeltaC-only Conv2D controls, but the Conv2D encoder does not
improve on the same-budget token mixer. The strict normal research gate passed
with `research_decision_applied=true`, `candidate_above_all=false`,
`decision=stop_conv2d_route`, and
`next_action=keep_token_mixer_anchor_and_do_not_scale_conv2d`. This is an
architecture-route stop under the frozen diagnostic gate, not a definitive
PRESENT model ceiling or a formal failure claim.

The anchor has `128673` total/trainable parameters. Each Conv2D row has exactly
`130881` total/trainable parameters, so the candidate/anchor capacity ratio is
`1.0171597771x`. The first row created exactly two parameter-matched disk cache
identities:

```text
outputs/local_cache/i1_present_invp_state_matrix_conv2d_8192_seed0/present80/r7/train/seed-0_d0cb1e57a50c7744
outputs/local_cache/i1_present_invp_state_matrix_conv2d_8192_seed0/present80/r7/validation/seed-10000_5b4686c47133da9b
```

The remaining three rows reused both identities, producing six `cache_reuse`
events. Train metadata records 8192 positive plus 8192 negative rows; validation
metadata records 4096 plus 4096. Both identities contain `features.npy`,
`labels.npy`, and metadata with strict negatives, 16 pairs/sample, r7, the
expected split seed, and `key_schedule=per_pair_random`. Progress contains four
`row_done` events and one terminal `run_done` event.

R1 artifacts:

| Artifact | Size |
| --- | ---: |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/results.jsonl` | 33475 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/progress.jsonl` | 807818 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/validation.json` | 3766 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/curves.svg` | 76704 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/history.csv` | 13803 bytes |
| `outputs/local_smoke/i1_present_invp_state_matrix_conv2d_8192_seed0/state_matrix_conv2d_gate.json` | 25978 bytes |

The regenerated R1 gate uses the same expanded evidence schema as R0. Its
`cache_evidence.0.verified=true` report records 2 creates, 6 reuses, 8 exact
role/split terminal events, distinct train/validation identities, and one
`run_done`. It also binds the ordered result/progress pair by exact seed,
matches normalized `run_done.output` to the supplied result path, and reports
trusted normalized result/cache provenance while rejecting cache paths outside
the result-declared root. `protocol_evidence.rows` contains all four frozen
model/options, result/training/validation identities, parameter counts, and
replay-validated checkpoint/history status. `semantic_checks` records the frozen/tested
`[batch, pair, 4, 16]` state-matrix contract without claiming runtime tensor
equality. The stopped-action list explicitly blocks seed1, `65536/class`,
`262144/class`, and remote scaling under `decision=stop_conv2d_route`.

The independent local AutoND watcher-health snapshot was attempted once without
SSH, but local tmux socket access was denied by the execution sandbox. Its
health is therefore recorded as unavailable, not inferred; no remote AutoND
state was touched and this does not affect the complete local R1 artifacts.

Recommended next action: retain `present_nibble_invp_only_spn_only` as the
strongest same-protocol architecture anchor, stop the state-matrix Conv2D
replacement route, and execute the separately gated [H1 topology-residual
adapter plan](#h1-invp-token-mixer-topology-residual-adapter) below. H1 is
planned and has not been implemented or run.

## H1: InvP Token-Mixer Topology-Residual Adapter

**Status:** planned; implementation, R0, and H1 seed0 training have not started

### Research Question And Hypothesis

The completed R1 result creates a specific residual-fusion hypothesis. The
true-InvP Conv2D candidate beat shuffled-P by `+0.189985901117` AUC and
DeltaC-only by `+0.169678151608`, but lost to the token-mixer anchor by
`-0.005025476217`. This suggests that topology-local processing may be
complementary when added as a small residual adapter to the strong true-InvP
token mixer, even though it is weaker when it replaces that mixer.

H1 changes one variable only: the presence and mapping identity of one
lightweight local residual adapter. The token-mixer backbone, classifier, raw
input, dataset, validation, optimizer, scheduler, checkpoint policy, and
training budget remain fixed.

### Frozen Matrix

| Row | Role | Model key | Adapter view |
| --- | --- | --- | --- |
| 1 | same-budget anchor | `present_nibble_invp_only_spn_only` | none |
| 2 | H1 candidate | `present_nibble_invp_topology_residual_spn_only` | true PRESENT InvP |
| 3 | topology control | `present_nibble_shuffled_p_topology_residual_spn_only` | deterministic shuffled P |
| 4 | representation control | `present_nibble_delta_topology_residual_spn_only` | raw DeltaC identity |

The anchor retains exactly these options:

```json
{"spn_mixer_depth":2,"activation":"relu","norm":"layernorm"}
```

All three hybrid rows use the identical true-InvP token-mixer backbone and
classifier. Only the adapter view changes between true InvP, the existing
deterministic shuffled-P mapping, and raw DeltaC. The three hybrids must have
exactly equal total and trainable parameter counts.

### Frozen Hybrid Architecture

The token encoder is exactly the anchor architecture: `base_channels=32`,
128-dimensional pair embedding, mixer depth 2, and token MLP ratio 2. In
parallel, the local adapter receives `[batch, pair, 4, 16]`, applies a
16-channel `1x1` stem, exactly one `3x3` residual Conv2D block, spatial mean and
max pooling, and a linear projection to 128 dimensions.

Pair fusion is:

```text
token_pair_embedding + alpha * local_pair_embedding
```

`alpha` is one learned scalar initialized to `0.1`. The fused pair embeddings
reuse the anchor's mean/max aggregation and unchanged classifier input and
shape. Activation is ReLU, token normalization is LayerNorm, local
normalization is BatchNorm2D, and dropout is zero. Construction must instantiate
the common backbone and classifier before the adapter so the same seed preserves
comparable common initialization.

Exact hybrid options:

```json
{"spn_mixer_depth":2,"token_mlp_ratio":2,"local_channels":16,"local_depth":1,"local_kernel_size":3,"local_residual_scale_init":0.1,"activation":"relu","norm":"layernorm","local_norm":"batchnorm2d","dropout":0.0}
```

### Frozen Protocol

All four rows retain the strict PRESENT-80 r7 Zhang/Wang Case2 `m=16`
protocol:

```text
feature encoding          = ciphertext_pair_bits
negative mode             = encrypted_random_plaintexts
sample structure          = zhang_wang_case2_official_mcnd
effective key schedule    = per_pair_random for train and validation
loss                      = mse
optimizer                 = adam
learning rate             = 0.0001
weight decay              = 0.00001
learning-rate scheduler   = official_cyclic
max learning rate         = 0.002
checkpoint metric         = val_auc
restore best checkpoint   = true
early stopping patience   = 8
early stopping min delta  = 0.0001
```

Labels, input difference, pair count, validation construction, negative
definition, metric computation, and effective key sampling must not change.
Both serialized result sections and both cache metadata files must report
`key_schedule=per_pair_random`.

### R0 Readiness

```text
run id             = i1_present_invp_topology_residual_smoke_seed0
planned config     = configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_smoke_seed0.csv
samples_per_class  = 64
validation         = 32/class, 64 total rows
seed               = 0
epochs             = 1
batch size         = 32
device             = cpu
rows               = 4
```

Planned output/cache paths:

```text
outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/results.jsonl
outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/progress.jsonl
outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/validation.json
outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/curves.svg
outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/history.csv
outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/readiness_gate.json
outputs/local_cache/i1_present_invp_topology_residual_smoke_seed0/
```

R0 is a neutral implementation gate only. It must prove exact tensor semantics,
common initialization order, equal hybrid capacities, finite forward/backward,
complete sequential histories, best-checkpoint consistency, exact 4/4 plan
alignment, one train plus one validation disk cache reused across all rows, and
`decision=implementation_ready` with `research_decision_applied=false`. R0
metrics must not be interpreted.

### H1 Seed0 Local Diagnostic

```text
run id             = i1_present_invp_topology_residual_8192_seed0
planned config     = configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_8192_seed0.csv
samples_per_class  = 8192
training total     = 16384 rows
validation         = 4096/class, 8192 total rows
seed               = 0
epochs             = 10
batch size         = 256
device             = cpu/local
rows               = 4
```

Planned output/cache paths:

```text
outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/results.jsonl
outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/progress.jsonl
outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/validation.json
outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/curves.svg
outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/history.csv
outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/topology_residual_gate.json
outputs/local_cache/i1_present_invp_topology_residual_8192_seed0/
```

The runner must use disk-backed cache generation, chunk progress, progress
JSONL, and exact train/validation cache reuse. This `8192/class` stage is a
local diagnostic only, not formal training, paper-scale evidence, or a
breakthrough claim.

For each seed define:

```text
architecture_margin   = AUC(true residual) - AUC(token-mixer anchor)
topology_margin       = AUC(true residual) - AUC(shuffled-P residual)
representation_margin = AUC(true residual) - AUC(DeltaC residual)
```

Apply these seed0 decisions in order:

| Observation | Decision and action |
| --- | --- |
| protocol, history, count, cache, or schedule invariant fails | `invalid_protocol`; repair and rerun the same matrix without interpreting metrics |
| candidate AUC `<=` anchor | `stop_topology_residual`; retain anchor |
| candidate AUC `<=` shuffled control | `stop_true_topology_attribution`; no seed1 or scale |
| candidate AUC `<=` Delta control | `stop_invp_adapter_attribution`; no seed1 or scale |
| candidate is above all, but any margin is `< +0.003` | `weak_or_fragile_no_scale`; inspect histories once, no remote |
| architecture, topology, and representation margins are all `>= +0.003` | `promote_seed1`; run only the identical local seed1 matrix |

Seed0 plus seed1 may advance to `65536/class` only if the candidate is above all
comparators on each seed, minimum architecture margin is at least `+0.001`, and
minimum topology and representation margins are each at least `+0.002`.
Otherwise the route does not scale. `65536/class` remains a medium diagnostic;
there is no remote or formal claim before later evidence gates and no formal
claim without completed, retrieved, plan-aligned `>=1000000/class` multi-seed
evidence.

### Forbidden H1 Continuations

- Do not run pure Conv2D seed1, R2, `65536/class`, or `262144/class`.
- Do not reopen DDT/beam-stat inputs or E1 graph scaling.
- Do not use a score ensemble as raw single-sample method evidence.
- Do not change data, labels, negatives, keys, validation, or another benchmark
  variable alongside the adapter.
- Do not add another architecture or representation to the frozen four-row
  attribution matrix.

### Claim Boundary

Local/global fusion is established prior art and is not a novelty claim. The
remaining possible method claim is a cipher-spec-generated topology adapter
with same-input, capacity-matched shuffled-P and DeltaC attribution. That claim
is available only if the frozen gates pass at the required later evidence
scale. Until then H1 is a planned, controlled diagnostic hypothesis.

## Artifacts And Adjudication

Every non-smoke stage produces:

```text
results.jsonl
progress.jsonl
history.csv
curves.svg
validation.json
state_matrix_conv2d_gate.json
```

The gate artifact must record:

```text
exact model rows and model options
input tensor semantic checks
parameter counts and candidate/anchor ratio
dataset/cache identity across rows
per-seed metrics and margins
promotion conditions
decision = keep | stop | redesign | invalid
next action and explicitly stopped actions
claim scope
```

When a meaningful result completes, update this document in the same turn with
the run id, artifact paths, protocol gate, metrics, deltas, decision, claim
scope, and recommended next step.

## Testing And Verification

The implementation must add focused tests for:

1. Exact true-InvP tensor equality with the current verified InvP-only view.
2. Deterministic shuffled-P mapping and inequality with true InvP.
3. DeltaC-only identity mapping.
4. Correct `[batch, pair, bit_plane, cell]` layout with no accidental
   `[word, cell, bit]` reinterpretation.
5. Equal parameter counts across all Conv2D controls.
6. Forward shape and finite backward gradients for 16-pair batches.
7. Registry/model-options construction.
8. Four-row plan alignment and gate branch behavior.
9. Existing InvP-only model behavior remains unchanged.

Verification uses project commands with `UV_CACHE_DIR=/tmp/uv-cache` and
`uv run pytest`. The existing full-suite Matplotlib 3.11 and JSON alignment
baseline failures remain separate unless a focused new test exposes a direct
regression.

## Explicit Non-Goals

This design does not:

- use DDT, beam statistics, trail positions, or S-box probability tables;
- resume E1 active-cell or P-layer graph scaling;
- use public-code random-ciphertext negatives;
- change the input difference, pair count, keys, labels, or validation split;
- add attention, U-Net, ECA, NAS, multi-query aggregation, or cross-cipher
  transfer in the same experiment;
- launch remote scale from an R0 smoke or a weak/invalid R1 gate;
- call `8192/class`, `65536/class`, or `262144/class` formal evidence;
- claim Conv2D or InvP as a standalone invention.

## Relationship To The Running AutoND Task

The running public-code paper-scale AutoND reproduction remains independent:

```text
PRESENT-80 r9
single pair
random_labels_total
random_ciphertext negatives
per-row random key
r5 -> r9 curriculum
```

Its result calibrates the public AutoND implementation and published reference.
It is not an anchor row for this strict r7 Case2 matrix. Local R0/R1 work may
proceed after implementation approval, but no new remote Conv2D task launches
until the active paper-scale task is retrieved and adjudicated or an explicit
GPU scheduling exception is documented.

## Completed Decision

The reviewed implementation plan is:

```text
docs/experiments/
innovation1-present-invp-state-matrix-conv2d-implementation-plan.md
```

The implementation plan, R0 readiness gate, and R1 seed0 adjudication are
complete. The applied R1 decision is `stop_conv2d_route`: do not run seed1 or
remote/medium pure-Conv2D scale, and retain the token-mixer anchor. The exact
next experiment is the [planned H1 token-mixer topology-residual adapter](#h1-invp-token-mixer-topology-residual-adapter),
not another pure Conv2D run. H1 has not been implemented or trained.
