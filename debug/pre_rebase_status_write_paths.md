# Write Paths: pre_rebase_status.txt and post_commit_dirty_worktree.txt

## pre_rebase_status.txt

| File | Line | Snippet | Action | Tracked | Active in perplexity_models.yml |
|---|---|---|---|---|---|
| `.github/workflows/cbb_perplexity_models.yml` | 535 (old) | `git status --porcelain \| tee debug/pre_rebase_status.txt` | WRITE (tee) | YES | YES — **root cause** |
| `.github/workflows/cbb_perplexity_models.yml` | 536 (old) | `if [[ -s debug/pre_rebase_status.txt ]]` | READ | YES | YES |
| `.github/workflows/cbb_perplexity_models.yml` | 538 (old) | `cat debug/pre_rebase_status.txt` | READ | YES | YES |
| `.github/workflows/cbb_perplexity_models.yml` | 549 (old) | `done < debug/pre_rebase_status.txt` | READ | YES | YES |
| `debug/pre_rebase_status.txt` | 4 | self-referencing documentation | N/A | YES | N/A |
| `debug/git_sync_flow.md` | 25 | documentation reference | DOC | YES | N/A |

## post_commit_dirty_worktree.txt

| File | Line | Snippet | Action | Tracked | Active in perplexity_models.yml |
|---|---|---|---|---|---|
| `.github/workflows/cbb_perplexity_models.yml` | 525 (old) | `git status --porcelain \| tee debug/post_commit_dirty_worktree.txt` | WRITE (tee) | YES | YES |
| `.github/workflows/cbb_perplexity_models.yml` | 526 (old) | `grep -c '.*' debug/post_commit_dirty_worktree.txt` | READ | YES | YES |
| `debug/post_commit_dirty_worktree.txt` | 16 | self-referencing documentation | N/A | YES | N/A |
| `debug/git_sync_flow.md` | 20 | documentation reference | DOC | YES | N/A |

## pre_stage_status.txt

| File | Line | Snippet | Action | Tracked | Active in perplexity_models.yml |
|---|---|---|---|---|---|
| `.github/workflows/cbb_perplexity_models.yml` | 493 | `git status --short \| tee debug/pre_stage_status.txt` | WRITE (tee) | NO | YES — SAFE (untracked) |
