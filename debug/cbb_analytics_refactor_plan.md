# CBB Analytics Refactor Plan

## Agreed future purpose
cbb_analytics.yml is the post-prediction analytics, backtesting, and results tracking workflow that merges upstream artifacts and produces analytics outputs.

## Changed files
1. `.github/workflows/cbb_analytics.yml`

## Changes made
1. **Step "Download latest rolling ready marker from main"**: Added `continue-on-error: true` so the workflow doesn't hard-fail when this artifact is unavailable
2. **Step "Validate dependency marker"**: Rewritten to check 3 locations for the marker file:
   - Primary: `data/staging/cbb_predictions_rolling_ready.json` (from separate marker artifact)
   - Secondary: `data/staging/ready_markers/cbb_predictions_rolling_ready.json` (from main predictions artifact)
   - Fallback: Synthetic marker generation when INFRA-predictions-rolling downloaded successfully
3. **Dependency chain print**: Updated to note marker is optional

## Validation checklist
- [x] YAML syntax valid
- [x] Workflow structure preserved (same jobs, same step count)
- [x] All required artifacts still required
- [x] Only the redundant marker made optional
- [x] Fallback logic generates valid JSON marker
- [x] Error messaging is clear and actionable

## Rollback plan
Revert the 3 changes:
1. Remove `continue-on-error: true` from marker download step
2. Restore original marker validation (single path check)
3. Restore original dependency chain message

## Why this is the minimal robust change
- Only 1 file modified
- Only 2 steps changed in the workflow
- No scripts modified
- No other workflows modified
- Preserves all existing functionality
- Fixes the root cause (hard-fail on optional artifact)
