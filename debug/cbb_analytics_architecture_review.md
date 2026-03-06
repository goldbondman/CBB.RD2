# CBB Analytics Architecture Review

## Current responsibilities
1. Artifact merge/assembly (ESPN, market lines, predictions → working data)
2. ESPN fallback orchestration (self-healing when Tier A data unavailable)
3. Backtesting engine execution
4. Prediction grading
5. ML training data generation
6. Results tracking
7. Season summaries + accuracy reporting
8. Gate building + segment performance analysis
9. Curated output commit-back to repo

## Unique responsibilities (not duplicated elsewhere)
- Backtesting, grading, results tracking, training data generation
- Gate building, segment performance analysis
- Analytics-specific artifact merge
- ESPN fallback pipeline

## Overlaps with other workflows
- ESPN data download: also done by cbb_predictions_rolling.yml (but different purpose)
- Context overlay: also produced by cbb_predictions_rolling.yml
- Ready marker pattern: shared with cbb_predictions_rolling.yml

## Recommendation: KEEP and SLIM DOWN
### Rationale
cbb_analytics.yml serves a unique and essential role in the project. It is the only workflow that performs backtesting, grading, results tracking, and training data generation. It cannot be removed or merged without losing critical functionality.

### Specific slimming
- **INFRA-ready-cbb-predictions-rolling**: Made optional. The main INFRA-predictions-rolling artifact already contains the ready marker file. The separate marker artifact adds no value and frequently fails due to expiration or non-production.
- **Marker validation**: Now checks 3 locations (separate artifact, main artifact subdirectory, synthetic generation) instead of hard-failing on one.

## Required vs optional artifact matrix (post-fix)
| Artifact | Classification | Rationale |
|---|---|---|
| INFRA-espn-data | Best effort | Fallback pipeline handles missing |
| INFRA-market-lines | Required | Core analytics input |
| INFRA-predictions-with-context | Required | Core analytics input |
| INFRA-predictions-rolling | Required | Core predictions data |
| INFRA-ready-cbb-predictions-rolling | **Optional** | Redundant with main artifact |
