# IDES Process Flows and Dependencies

## Purpose
Canonical runtime flow and dependency guide for the in-repo IDES module at:
- `CBB.RD2/ides_of_march/`

This file defines:
- layer sequencing
- agent ownership
- stage dependencies
- output artifact dependencies
- gate behavior (`PASS`, `WARN`, `FAIL`, `BLOCKED`)

## Layered Runtime Order (Required)
1. Layer 1: Base Team Strength
2. Layer 2: Context Adjustment
3. Layer 3: Situational Intelligence
4. Layer 4: Monte Carlo Probability Engine
5. Layer 5: Agreement Engine
6. Layer 6: Decision Layer

No stage may bypass this order without an architecture update.

## Agent Ownership and Dependencies

| stage_id | owner_agent | depends_on | outputs | gate_on_failure |
|---|---|---|---|---|
| `ides_data_steward` | data_steward_agent | source game/team/player/market inputs | leak-safe matchup frame | `BLOCKED` missing artifacts, `FAIL` on schema/key drift |
| `ides_layer1_base` | base_model_agent | `ides_data_steward` | base spread/probability signals | `FAIL` |
| `ides_layer2_context` | context_adjustment_agent | `ides_layer1_base` | context-adjusted margin | `FAIL` |
| `ides_layer3_situational` | situational_research_agent | `ides_layer2_context` | situational flags/adjustments | `WARN` low support, `FAIL` leakage |
| `ides_layer4_monte_carlo` | monte_carlo_agent | `ides_layer2_context` | MC probabilities and volatility | `WARN` fallback mode, `FAIL` schema break |
| `ides_layer5_agreement` | agreement_analysis_agent | `ides_layer3_situational`, `ides_layer4_monte_carlo` | agreement/conflict buckets | `FAIL` |
| `ides_layer6_decision` | decision_agent | `ides_layer5_agreement` | confidence, recommendations, final outputs | `FAIL` |
| `ides_evaluation` | evaluation_agent | `ides_layer6_decision` | backtest scorecards and bucket reports | `WARN` low sample, `FAIL` contract break |
| `ides_safety_audit` | model_safety_auditor | all prior stages | leakage/overfit checks | `FAIL` leakage, `BLOCKED` missing audit inputs |

## Primary Prediction Flow
1. Hydrate data dependencies.
2. Build leak-safe rolling features (`shift(1)`/prior-game only).
3. Execute L1 -> L6.
4. Validate output schemas.
5. Write artifacts and manifest.

### Prediction Artifact Dependencies
- `data/ides_of_march/predictions_latest.csv` depends on L1-L6 + schema pass.
- `data/ides_of_march/bet_recs.csv` depends on decision layer + market joins.
- `data/ides_of_march/agreement_bucket_report.csv` depends on agreement engine.
- `data/ides_of_march/situational_rulebook.csv` depends on situational research outputs.
- `data/ides_of_march/run_manifest.json` depends on stage status aggregation.

## Backtest / Research Flow
1. Build walk-forward date splits.
2. Run variant matrix across backbone, situational, and MC roles.
3. Produce agreement/situational bucket reports.
4. Emit variant scorecard and calibration diagnostics.

### Research Artifact Dependencies
- `data/ides_of_march/backtest_variant_scorecard.csv` depends on full variant execution.
- `data/ides_of_march/situational_rulebook.csv` depends on signal thresholding/shrinkage.

## Gate Contract
- `PASS`: stage valid.
- `WARN`: non-blocking issue; continue with documented risk.
- `FAIL`: blocking issue; stop downstream stages.
- `BLOCKED`: missing/unreadable required input; route to debugger path.

## Validation Checklist (Before Publish)
1. Required columns present for all output CSV contracts.
2. Join key integrity (`event_id`, `game_id`) preserved.
3. No future leakage in rolling features.
4. Agreement bucket assignment complete and non-null.
5. Confidence outputs calibrated within acceptable error bounds.
6. Manifest reflects real stage statuses and metrics.
