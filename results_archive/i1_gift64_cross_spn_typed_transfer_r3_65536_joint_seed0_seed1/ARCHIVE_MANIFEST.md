# E4-R3 Two-Seed Joint Result Archive

This branch preserves the locally rerun joint adjudication over the completed
remote E4-R3 seed 0 and seed 1 medium diagnostics. Both remote trainings and
remote validations ran from exact source commit
`9aa31ddc8f48312ecf3e1d9ea3973a0c4b00542a` on separate A6000 GPUs.

The original runners failed at optional Matplotlib plotting before result
branch creation. The raw result and progress evidence was retrieved directly
from the approved `G:\lxy` run roots, validated locally, and passed through the
corrected frozen E4-R3 per-seed and joint gates. This is therefore a post-hoc
verified archive built from fallback-retrieved evidence, not a result branch
pushed by the original remote runner. The per-seed notices preserve that claim
boundary.

Joint adjudication:

```text
seed0 true - scratch          = +0.000248298515
seed0 true - source-shuffled = +0.001391120255
seed1 true - scratch          = +0.001839549281
seed1 true - source-shuffled = +0.002584639471
status                        = pass
decision                      = e4_r3_two_seed_medium_signal_unstable
next_action                   = stop_mechanical_scale_and_audit_seed_variance
```

The result is a valid remote medium diagnostic, not formal or paper-scale
evidence. It stops mechanical `262144/class` scaling. `SHA256SUMS` covers every
archived file except itself.
