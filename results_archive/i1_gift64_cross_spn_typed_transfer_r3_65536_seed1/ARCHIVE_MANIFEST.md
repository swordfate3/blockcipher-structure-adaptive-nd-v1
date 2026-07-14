# E4-R3 Seed 1 Result Archive

This branch preserves the completed remote E4-R3 seed 1 medium diagnostic.
Training and remote validation completed on A6000 GPU1 from source commit
`9aa31ddc8f48312ecf3e1d9ea3973a0c4b00542a`.

The original remote runner failed only during optional Matplotlib plotting,
before it could create a result branch. The files here were therefore built
from the raw fallback retrieval under the project-approved `G:\lxy` run root.
The result JSONL and remote progress/log files are unchanged. Validation,
history/SVG generation, and the corrected E4-R3 gate were rerun locally; this
is a post-hoc verified archive, not an archive pushed by the original remote
runner. See `RAW_RETRIEVAL_NOTICE.md` for the exact claim boundary.

Protocol:

```text
cipher/rounds       = GIFT-64 r6
target seed         = 1
train               = 65536/class = 131072 total
validation          = 32768/class = 65536 total
pairs/sample        = 4 independent pairs
epochs              = 10
negative            = encrypted_random_plaintexts
roles               = 5
claim scope         = remote medium diagnostic, not formal/paper-scale
```

Adjudication:

```text
anchor                  = 0.578508460429
typed scratch           = 0.585099051706
true -> true            = 0.586938600987
source-shuffled -> true = 0.584353961516
true -> target-shuffled = 0.510260965209
decision                = e4_r3_seed_margin_miss
```

The five restored-best checkpoints remain in the local fallback retrieval and
are intentionally excluded from Git. `SHA256SUMS` covers every archived file
except itself.
