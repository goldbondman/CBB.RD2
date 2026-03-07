# Post-Commit / Pre-Rebase Write Safety Audit

All writes in the commit-and-push step of `cbb_perplexity_models.yml` classified by safety.

## After fix

| Line | Command | Classification | Reason |
|---|---|---|---|
| 493 | `git status --short \| tee debug/pre_stage_status.txt` | SAFE (untracked) | `pre_stage_status.txt` is not tracked; occurs before commit |
| 525 | `POST_COMMIT_STATUS=$(git status --porcelain \|\| true)` | SAFE_STDOUT_ONLY | Captured in shell variable, echoed to stdout; no file write |
| 527 | `grep -c '.' <<< "$POST_COMMIT_STATUS"` | SAFE_STDOUT_ONLY | Reads from variable, no file I/O |
| 536 | `PRE_REBASE_STATUS=$(git status --porcelain \|\| true)` | SAFE_STDOUT_ONLY | Captured in shell variable, echoed to stdout; no file write |
| 538 | `if [[ -n "${PRE_REBASE_STATUS}" ]]` | SAFE_STDOUT_ONLY | In-memory string check, no file I/O |
| 542-551 | `while IFS= read -r line; do ... done <<< "$PRE_REBASE_STATUS"` | SAFE_STDOUT_ONLY | Reads from variable via here-string, no file I/O |
| 557 | `git status --short` | SAFE_STDOUT_ONLY | Stdout only; inside rebase error handler |

## Before fix (for reference)

| Line (old) | Command | Classification | Reason |
|---|---|---|---|
| 525 | `git status --porcelain \| tee debug/post_commit_dirty_worktree.txt` | UNSAFE_TRACKED_WRITE | Writes to tracked file; masked by subsequent cleanup |
| 535 | `git status --porcelain \| tee debug/pre_rebase_status.txt` | UNSAFE_TRACKED_WRITE | **Root cause of failure** — writes to tracked file after cleanup |
