# Git Sync Flow (cbb_perplexity_models)

## Workflow + line references
- File: `.github/workflows/cbb_perplexity_models.yml`
- Commit/push block: lines 365-457

## Command sequence in workflow
1. Stage scope manifest (lines 379-386):
   - `data/advanced_metrics.csv`
   - `data/team_snapshot.csv`
   - `data/market_lines_canonical.csv`
   - `data/market_lines_latest_by_game.csv`
   - `data/predictions_joint_latest.csv`
   - `data/predictions_joint_snapshots.csv`
2. Stage only those paths (line 393):
   - `git add --ignore-missing -- "${COMMIT_PATHS[@]}"`
3. Commit staged paths (line 408):
   - `git commit -m "data: refresh perplexity model outputs [skip ci]"`
4. Capture dirty state after commit, before cleanup (stdout only):
   - `POST_COMMIT_STATUS=$(git status --porcelain || true)`
5. Cleanup non-target changes:
   - `git restore --worktree --staged .`
   - `git clean -fd`
6. Pre-rebase cleanliness gate (in-memory, no tracked file writes):
   - `PRE_REBASE_STATUS=$(git status --porcelain || true)`
   - fail if non-empty
7. Sync + push retry loop (lines 439-455):
   - `git fetch origin main`
   - `git rebase origin/main`
   - `git push origin HEAD:main`

## Helper scripts checked for git sync
- Searched `.github/scripts/*.sh` and `.github/scripts/*.py` for git pull/rebase/push/commit.
- No helper script is invoked by this workflow for sync; git sync is fully inline in the workflow block above.

## Dirty-worktree expectation
- Yes, this workflow can be partially dirty after generation and after staged commit.
- Non-target generated files are expected and are intentionally discarded before rebase.
