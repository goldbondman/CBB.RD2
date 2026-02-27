# CBB Pipeline — Executive Summary Flow
> Generated 2026-02-27. Reflects repo state as of `main` branch.

---

## High-Level Stage Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INGESTION                                                                  │
│  ESPN Scoreboard API  →  ESPN Summary API  →  ingestion/market_lines.py    │
│  (daily: games, box scores, player stats)    (hourly: betting lines)        │
└───────────────────────────┬─────────────────────────────────┬───────────────┘
                            │                                 │
                            ▼                                 ▼
┌─────────────────────────────────────────┐   ┌──────────────────────────────┐
│  STORAGE (data/*.csv)                   │   │  STORAGE (data/market_lines) │
│  games.csv                              │   │  market_lines.csv            │
│  team_game_logs.csv                     │   │  predictions_with_context.csv│
│  player_game_logs.csv                   │   └──────────────────────────────┘
│  data/raw_json/  (audit copies)         │
└───────────────────────────┬─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FEATURE BUILDS (all pure DataFrame transforms, no I/O)                     │
│                                                                             │
│  Team chain (sequential, each stage feeds the next):                        │
│    espn_metrics.py      → team_game_metrics.csv   (advanced box metrics)    │
│    espn_sos.py          → team_game_sos.csv        (opp-quality context)    │
│    espn_weighted_metrics.py → team_game_weighted.csv (opp-wtd L5/L10)       │
│    espn_tournament.py   → team_tournament_metrics.csv (DNA, UWS, totals)   │
│    espn_tournament.py   → team_pretournament_snapshot.csv (latest per team) │
│    espn_rankings.py     → cbb_rankings.csv, cbb_rankings_by_conference.csv  │
│                                                                             │
│  Player chain (parallel with team chain):                                   │
│    espn_player_metrics.py → player_game_metrics.csv (per-game + L5/L10)    │
│                          → player_rolling_l5.csv                            │
│                          → player_role_splits.csv                           │
│    espn_injury_proxy.py  → player_injury_proxy.csv                          │
│                          → team_injury_impact.csv                           │
│    cbb_player_matchup.py → player_condition_profiles.csv                    │
│                          → player_archetype_matchup_matrix.csv              │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODELS                                                                     │
│                                                                             │
│  Primary model (espn_prediction_runner.py):                                 │
│    Input: team_game_weighted.csv  →  GameData objects per team              │
│    Algo:  CBBPredictionModel  (cbb_prediction_model.py)                     │
│           Recursive bidirectional analysis, 4-factor deltas,                │
│           decay-weighted history, opponent-quality normalization             │
│    Output: predictions_YYYYMMDD.csv, predictions_primary.csv                │
│                                                                             │
│  Ensemble (cbb_ensemble.py — 7 sub-models):                                 │
│    Input: team_pretournament_snapshot.csv → TeamProfile objects             │
│    Sub-models: M1 FourFactors, M2 AdjEfficiency, M3 Pythagorean,           │
│                M4 Momentum, M5 Situational, M6 CAGERankings, M7 Regressed  │
│    Output: ensemble_predictions_YYYYMMDD.csv                                │
│                                                                             │
│  Monte Carlo (cbb_monte_carlo.py):                                          │
│    Input: predictions_combined_latest.csv                                   │
│    Output: predictions_mc_latest.csv (confidence intervals, sim win%)       │
│                                                                             │
│  Player matchup (cbb_player_matchup.py):                                    │
│    Input: player_game_metrics.csv + predictions_combined_latest.csv         │
│    Output: player_context_scores.csv, team_matchup_summary.csv              │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MERGE + EVALUATION                                                         │
│                                                                             │
│  Merge step (workflow inline):                                              │
│    primary + ensemble  →  predictions_combined_latest.csv                  │
│    divergence flag (|gap| ≥ 3 pts)  →  predictions_divergence_latest.csv   │
│                                                                             │
│  Market enrichment (enrichment/predictions_with_context.py):               │
│    combined + market_lines  →  predictions_with_context.csv                 │
│    + alpha evaluation (models/alpha_evaluator.py)                           │
│                                                                             │
│  Results tracker (cbb_results_tracker.py):                                  │
│    Grades yesterday's predictions against actual results                    │
│    Output: results_log_graded.csv                                           │
│                                                                             │
│  Backtester (cbb_backtester.py, weekly):                                    │
│    Full historical replay of all sub-models                                 │
│    Output: data/csv/backtest_*.csv (summary, by_model, weekly, etc.)       │
│                                                                             │
│  Weight optimizer (optimize_weights.py):                                    │
│    Reads backtest results → updates backtest_optimized_weights.json         │
└───────────────────────────┬─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUTS                                                                    │
│  predictions_combined_latest.csv  — primary consumer artifact               │
│  data/csv/predictions_latest.csv  — committed to repo (frontend)           │
│  ensemble_predictions_latest.csv                                            │
│  predictions_mc_latest.csv        — Monte Carlo confidence bands            │
│  predictions_with_context.csv     — market-enriched predictions             │
│  data/csv/bet_recs.csv            — edge ≥ 3.75 betting recommendations    │
│  data/csv/upset_watch.csv         — UWS ≥ 40 games                         │
│  data/csv/matchup_preview.csv     — today's games with full context         │
│  data/csv/player_leaders.csv      — season player leaderboard               │
│  cbb_rankings.csv                 — CAGE power rankings                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Entrypoint Commands / Scripts

| Trigger | Command | Purpose |
|---|---|---|
| Daily 10:00 UTC (auto) | `python espn_pipeline.py --days-back 3` | Full ingestion + feature build |
| After pipeline (auto) | `python espn_prediction_runner.py --date YYYYMMDD` | Primary model predictions |
| After pipeline (auto) | `python cbb_ensemble.py` (via workflow) | 7-model ensemble predictions |
| After predictions | `python cbb_monte_carlo.py --input ... --n-sims 5000` | MC simulation layer |
| After predictions | `python cbb_player_matchup.py --games ...` | Player matchup context |
| After pipeline | `python espn_rankings.py --top 25` | CAGE rankings update |
| Hourly (market window) | `python -m ingestion.market_lines --mode pregame` | Odds capture |
| Hourly (market window) | `python -m enrichment.predictions_with_context` | Line enrichment |
| Daily 8:00 UTC | `python cbb_results_tracker.py` | Grade yesterday's picks |
| Weekly Monday | `python cbb_backtester.py` + `python optimize_weights.py` | Full backtest + tuning |
| On-demand | `python build_derived_csvs.py` | Derived frontend CSVs |
| On-demand | `python field_reconciler.py --check-outputs` | Schema drift audit |
| Backfill | `python espn_pipeline.py --days-back 120` | Full season backfill |

---

## Workflow Schedule

```
market_lines.yml      ─── hourly (6 AM PST open + pregame + 9 PM close)
update_espn_cbb.yml   ─── daily 10:00 AM UTC (2 AM PST)
                               └─ triggers cbb_predictions_rolling.yml on success
cbb_analytics.yml     ─── daily 8:00 AM UTC (results tracker)
                           ─── weekly Monday 10:00 AM UTC (backtester)
```

---

## Key Configuration Files

| File | Purpose |
|---|---|
| `cbb_config.py` | League avg constants, spread/total model weights, SIGMA |
| `espn_config.py` | All output paths (OUT_*), API settings, TZ, DAYS_BACK |
| `config/schemas.py` | Output column schemas (validated at write time) |
| `cbb_output_schemas.py` | `validate_output()` called on team_game_logs + player_game_logs |
| `column_map.json` | ESPN API field → canonical column name mapping |
| `data/backtest_optimized_weights.json` | Live ensemble weights (updated weekly) |
| `data/model_weights.json` | Primary model blend weights |
