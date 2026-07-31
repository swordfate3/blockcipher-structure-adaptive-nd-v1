# Innovation 1 K1-BT: uKNIT r5 Neural Architecture Medium Confirmation

## Status

`preregistered / remote launch pending`

This experiment is the first remote scale step after K1-BS. It is a medium
confirmation, not formal-scale or paper-scale evidence. Its sole research
question is whether the large local advantage of the uKNIT structure expert
over the strongest generic architecture survives at `65536/class`.

## Frozen hypothesis and baseline

K1-BS measured the structure expert at AUC `0.902801514/0.932538986` and the
AutoND DBitNet baseline at `0.511321068/0.526423454` for seeds 3/4 under the
same `2048/class` protocol. AutoND had the strongest two-seed mean among the
three generic models, so it is the only generic architecture retained here.
The single changed variable relative to K1-BS is data scale; the architecture
matrix is reduced before launch to avoid spending remote budget on two weaker
generic rows.

## Frozen protocol

```text
run_id              = i1_uknit_r5_neural_architecture_medium_k1bt_16pair_65536_seed3_seed4_20260731
cipher / rounds      = uKNIT-BC / 5
difference           = cell11 role1 / 0x0000400000000000
models               = uKNIT structure expert, AutoND DBitNet
seeds                = 3,4
train                = 65536/class = 131072 total rows per model/seed
cross-key validation = 16384/class = 32768 total rows per model/seed
pairs per sample     = 16 independent ciphertext pairs
input width          = 2048 bits
negative definition  = encrypted random plaintexts
key policy           = one frozen train key and distinct frozen validation key per seed
epochs / batch       = 10 / 64
loss / optimizer     = MSE / Adam
learning rate        = 1e-4
weight decay         = 1e-5
checkpoint           = restore best validation AUC
execution            = remote A6000 physical GPU1 from exact pushed commit
```

The generated-data path must use parameter-matched disk caches with
`features.npy`, `labels.npy`, `metadata.json`, durable progress JSONL and reuse
between the two architectures for each seed/split. Four cache creations, four
cache reuses, four checkpoints and four result rows are required.

## Gate and interpretation

For each seed independently:

```text
structure expert AUC >= 0.550
structure expert - AutoND DBitNet >= +0.010
```

Both seeds must pass. If they do, authorize a remote `262144/class` confirmation
with the same two models and frozen protocol. If either seed fails, hold scale
and audit scale-dependent optimization/cache equivalence before changing the
model or data. Do not add models, pairs, epochs, differences, keys, or seeds in
this experiment.

Even a pass supports only a two-seed remote medium architecture result. It does
not support a formal ceiling, paper-scale, attack, SOTA, transfer, universal-SPN
or topology-causal claim. Formal claims require at least `1000000/class` and
multiple seeds after the intermediate `262144/class` gate.

## Executable next action

After a plan-aligned retrieved pass, create K1-BU at `262144/class` with the
same expert/AutoND rows, seeds 3/4, 16 pairs, 10 epochs and thresholds. After a
hold, inspect the failed seed's restored-best history and cache manifests; do
not mechanically scale. Every retrieved result must be revalidated locally,
rendered with Chinese labels, inspected through `visual-qa-redraw`, indexed in
both recent-result indexes, and recorded in this document before reporting.
