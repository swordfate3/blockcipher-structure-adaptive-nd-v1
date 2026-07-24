---
name: github-push-recovery
description: Safely publish existing Git commits to a configured GitHub remote when git push is failing, intermittent, or repeatedly timing out. Use for transient GitHub connection failures, connection resets, TLS/HTTP transport errors, uncertain push completion, or requests to keep retrying a normal push until it succeeds. Diagnose authentication, non-fast-forward, protected-branch, repository, and platform-review failures; retry only transient network failures with bounded exponential backoff; verify the exact remote commit after success.
---

# GitHub Push Recovery

Publish an already committed branch without changing repository history, credentials, remotes, or transport. Distinguish failures that can recover through retry from failures that require a human or a repository-state change.

## Safety Contract

- Push only the current committed `HEAD`; never stage, commit, amend, rebase, merge, reset, or force-push as part of recovery.
- Use the configured remote and target branch. Do not silently switch HTTPS to SSH, add another remote, use `scp`, or export a source archive.
- Never print credential-helper output, tokens, private keys, or credential-bearing URLs.
- Allow unrelated dirty files to remain because they are not part of the committed push. Report them and leave them untouched.
- Treat platform approval/reviewer rejection as final for the current attempt. Do not hide the command in another shell or retry through an equivalent transfer path.
- Stop on authentication, authorization, repository-not-found, protected-branch, remote-rejection, and non-fast-forward errors. Repetition cannot repair these conditions.
- Retry only classified transient transport failures. Keep each invocation bounded even when the user says "retry until successful"; a later invocation may resume after the network state changes.

## Workflow

1. Inspect `git status --short --branch`, `git remote -v`, the current branch, and commits ahead of its upstream. Do not expose embedded credentials if a remote URL contains them.
2. Confirm the requested payload: current `HEAD`, configured remote, and destination branch. If the repository has no commit to publish, report that and stop.
3. Run a no-write preflight:

```bash
python skills/github-push-recovery/scripts/retry_push.py \
  --repo . --remote origin --dry-run
```

4. Attempt the normal configured push once through the environment's standard command mechanism. If it succeeds, verify that the destination ref equals the captured local `HEAD`.
5. If the failure is a transient network error and network access is permitted, run the bounded recovery program:

```bash
python skills/github-push-recovery/scripts/retry_push.py \
  --repo . --remote origin --execute \
  --max-attempts 8 --max-elapsed 600 \
  --json-out /tmp/github-push-recovery.json
```

6. If the environment requires approval for network access, request it for the direct recovery command. If approval is rejected, stop. Do not wrap or transform the command to bypass review.
7. On exit `0`, compare the verified remote SHA with the captured local SHA and report both. On any other exit, report the category, attempt count, exact local SHA, and the next legitimate remedy.

## Failure Decisions

| Category | Retry? | Action |
|---|---:|---|
| `transient_network` | Yes | Exponential backoff within attempt/time limits |
| `verification_transient` | Yes | Keep verifying; do not create a new commit |
| `authentication` | No | Repair the configured credential path, then rerun |
| `non_fast_forward` | No | Fetch and inspect divergence; never auto-rebase or force |
| `remote_rejected` | No | Inspect branch protection or server message |
| `repository` | No | Verify configured remote ownership and access |
| `platform_policy` | No | Report the reviewer decision; do not circumvent it |
| `unknown` | No | Preserve output, diagnose first, then decide |

## Recovery Program

Use `scripts/retry_push.py` for deterministic retries. It:

- captures the original local SHA and refuses to continue if `HEAD` changes;
- pushes `HEAD` explicitly to `refs/heads/<branch>` without force;
- disables interactive credential prompts so automation cannot hang;
- sanitizes credential-bearing URLs in logs and JSON;
- uses exponential backoff only for known transient transport messages;
- switches to verification-only mode after Git reports a successful push;
- confirms the remote branch with `git ls-remote` before returning success.

Useful options:

```text
--branch NAME          destination branch; defaults to current branch
--max-attempts N       total push/verification attempts; default 8
--initial-delay SEC    first retry delay; default 5
--max-delay SEC        delay ceiling; default 60
--max-elapsed SEC      overall time ceiling; default 600
--command-timeout SEC  timeout for one Git command; default 120
--json-out PATH        atomic structured diagnostic output
```

Do not set extreme time limits in an interactive agent turn. For longer recovery, use an approved monitor/automation facility and keep the same safety contract.

## Completion Report

State one of:

- `published and verified`: local and remote SHAs match;
- `not published`: a permanent category stopped the attempt;
- `publication uncertain`: Git reported success but remote verification exhausted transiently;
- `temporarily unavailable`: bounded transient retries were exhausted;
- `blocked by platform policy`: the environment rejected the outbound transfer.

Never describe a local commit as published until the remote ref is verified.
