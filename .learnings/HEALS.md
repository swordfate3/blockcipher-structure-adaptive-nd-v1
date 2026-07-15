## [HEAL-20260715-001] gift_gate_entrypoint_source_bootstrap

**Logged**: 2026-07-15T20:43:46+08:00
**Status**: resolved
**Area**: remote-postprocess

### Symptom

Both GIFT-64 `1000000/class` target-seed runs completed five model rows and
primary-score export, then failed in `gate-cross-spn-mainstream-performance`
with `ModuleNotFoundError: No module named 'blockcipher_nd'`.

### Root Cause

The per-seed and joint thin wrappers omitted the repository `src/` bootstrap
used by other source-tree entrypoints. The remote environment ran directly from
a clean clone without an installed `blockcipher_nd` package.

### Verified Fix

Added the standard `Path(__file__).resolve().parents[1] / "src"` insertion to
both wrappers and a regression check. Recovery commit `e7767da` reused the
existing results, checkpoints, caches, and score artifacts; it did not retrain.
Both per-seed gates, the joint gate, archive push, local validation, and paired
re-adjudication passed.

### Prevention

Every new remote-invoked Python wrapper must prove direct source-tree execution
from outside the repository with an empty `PYTHONPATH` before launch.

### Metadata

- Related Files: scripts/gate-cross-spn-mainstream-performance, scripts/gate-cross-spn-mainstream-performance-joint, tests/test_cross_spn_mainstream_performance_gate.py
- Pattern-Key: remote.python_entrypoint.bootstrap_src
- Recurrence-Count: 1
- First-Seen: 2026-07-15
- Last-Seen: 2026-07-15

---

## [HEAL-20260715-002] modern_scp_directory_contents_retrieval

**Logged**: 2026-07-15T20:43:46+08:00
**Status**: resolved
**Area**: remote-retrieval

### Symptom

The repaired result branches and archives existed remotely, but the local
watcher exited while retrieving a directory whose remote path ended in `/.`:
`error: unexpected filename: .`.

### Root Cause

The installed OpenSSH `scp` implementation rejects the historical remote
directory-contents suffix used by the watcher.

### Verified Fix

Commit `0833289` retrieves each complete archive directory into a unique `/tmp`
staging directory, then copies its contents locally with `cp -a`. This retains
hidden files such as `.gitattributes`. Seed6, seed7, and joint archives were
retrieved; normalized-manifest SHA-256 verification passed for every archived
file, and both primary score packs matched their recorded hashes. The same
unsafe suffix was then found during the pre-launch audit of the independent SM4
position-calibration watcher and was repaired before that run launched.

### Prevention

Do not use a remote `/.` suffix with `scp`. Retrieve the named directory into a
temporary local staging directory, preserve dotfiles, verify hashes, and only
then expose the destination as retrieved evidence.

### Metadata

- Related Files: configs/remote/generated/monitor_i1_gift64_mainstream_performance_1m_20260715.sh, configs/remote/generated/monitor_i1_feistel_sm4_position_resnet_calibration_2048_20260715.sh, tests/test_cross_spn_mainstream_performance_gate.py, tests/test_feistel_sm4_innovation.py
- Pattern-Key: remote.scp.modern_directory_retrieval
- Recurrence-Count: 2
- First-Seen: 2026-07-15
- Last-Seen: 2026-07-15
