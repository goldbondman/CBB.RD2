# Active Failing Path: pre_rebase_status.txt

## Execution chain

1. **Workflow**: `.github/workflows/cbb_perplexity_models.yml`
2. **Step**: `commit-and-push` (inline shell, not a helper script)
3. **No helper scripts invoked** — all git sync logic is inline in the workflow YAML

## Timing order (before fix)

| Order | Line (old) | Command | Effect |
|---|---|---|---|
| 1 | 522 | `git commit -m "..."` | Commits staged output files |
| 2 | 525 | `git status --porcelain \| tee debug/post_commit_dirty_worktree.txt` | WRITES to tracked file (masked by cleanup) |
| 3 | 531 | `git restore --worktree --staged .` | Restores all tracked files to committed state |
| 4 | 532 | `git clean -fd` | Removes untracked files/dirs |
| 5 | 535 | `git status --porcelain \| tee debug/pre_rebase_status.txt` | **WRITES to tracked file — creates dirty state** |
| 6 | 536 | `if [[ -s debug/pre_rebase_status.txt ]]` | Detects dirty file — FAILS |

## Root cause

Line 535 uses `tee debug/pre_rebase_status.txt` in a pipeline with `git status --porcelain`.

In a bash pipeline, both sides start concurrently:
- `tee` opens `debug/pre_rebase_status.txt` for writing, truncating it from 9 lines of committed documentation content to empty
- `git status --porcelain` scans the worktree; by the time it checks `debug/pre_rebase_status.txt`, `tee` has already truncated it
- `git status` reports `M debug/pre_rebase_status.txt` (modified: content differs from committed version)
- `tee` writes this status line to the file, making it non-empty
- Line 536 checks `-s` (non-empty) → true → workflow fails

**The cleanliness gate creates the very condition it tests for.**

## Why cleanup didn't help

- Cleanup (lines 531-532) correctly restores the worktree to clean state
- But line 535 immediately re-dirties it by writing to a tracked file
- The `tee` redirect is the sole active write path after cleanup
