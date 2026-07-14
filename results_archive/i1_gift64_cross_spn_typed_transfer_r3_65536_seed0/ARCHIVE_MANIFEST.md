# E4-R3 Seed 0 Result Archive

This branch preserves the completed remote E4-R3 seed 0 medium diagnostic.
Training and remote validation completed on A6000 GPU0 from source commit
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
target seed         = 0
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
anchor                  = 0.579249573871
typed scratch           = 0.588316810317
true -> true            = 0.588565108832
source-shuffled -> true = 0.587173988577
true -> target-shuffled = 0.507330908440
decision                = e4_r3_seed_margin_miss
```

The five restored-best checkpoints remain in the local fallback retrieval and
are intentionally excluded from Git. `SHA256SUMS` covers every archived file
except itself.
