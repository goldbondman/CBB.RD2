# Pre-Rebase Validation: perplexity_models.yml

## Before fix

```
[INFO] pre-rebase status after cleanup
 M debug/pre_rebase_status.txt
[ERROR] Refusing to pull/rebase with dirty worktree. Remaining files:
 M debug/pre_rebase_status.txt
[INFO] tracked/untracked classification:
  - debug/pre_rebase_status.txt (tracked)
```

**Result**: Workflow fails. Rebase never runs.

## After fix

Expected log output:

```
[INFO] post-commit status (before cleanup)
 M data/some_generated_file.csv
?? debug/some_untracked_file.txt
[INFO] post_commit_dirty_files=2

[INFO] cleaning non-target generated changes before rebase

[INFO] pre-rebase status after cleanup

[OK] push succeeded on attempt 1
```

**Expected `git status --porcelain` before rebase**: empty (no output)

## What changed

| Aspect | Before | After |
|---|---|---|
| Post-commit status capture | `tee debug/post_commit_dirty_worktree.txt` (tracked file write) | `POST_COMMIT_STATUS=$(...)` (shell variable) |
| Pre-rebase status capture | `tee debug/pre_rebase_status.txt` (tracked file write) | `PRE_REBASE_STATUS=$(...)` (shell variable) |
| Cleanliness gate check | `-s debug/pre_rebase_status.txt` (file-based) | `-n "${PRE_REBASE_STATUS}"` (in-memory) |
| Error file listing | `cat` / `read < file` | `echo` / `read <<< variable` |
| `debug/pre_rebase_status.txt` | Tracked in git | Removed from git, added to .gitignore |
| `debug/post_commit_dirty_worktree.txt` | Tracked in git | Removed from git, added to .gitignore |

## Confirmation

- `debug/pre_rebase_status.txt` is no longer a blocker
- No tracked file is modified after commit and before rebase
- Diagnostics remain available via stdout
- The pre-rebase cleanliness gate can no longer create the condition it tests for
