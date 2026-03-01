# CSV Data Quality Report
**Generated:** 2026-03-01
**Scope:** 53 canonical (non-archive) CSV files in `data/`
**Checks:** null values, all-zero columns, constant columns

---

## Summary

| Issue Type | Count |
|---|---|
| Columns with ANY nulls | 2,903 |
| Columns that are 100% null | 423 |
| Columns that are all-zero | 104 |
| Columns with constant (single) value | 409 |

---

## 1. Critical Nulls (100% empty — column never populated)

These columns exist in the schema but contain no data at all.

### `backtest_training_data.csv` (24 cols fully null)
Feature groups that were never backfilled:
- ATS/cover tracking: `home_cover_l10`, `home_cover_rate_l10`, `home_cover_rate_season`, `home_ats_margin_l10`, `away_cover_rate_l10`, `away_cover_rate_season`, `away_ats_margin_l10`
- Luck/regression features: `home_luck_score`, `away_luck_score`, `luck_score_delta`, `close_game_luck_delta`, `composite_luck_delta`
- Player injury features: `home_t_top_scorer_efg_l5`, `home_t_bench_pts_share_l5`, `home_t_team_injury_burden`, `home_t_n_injured_starters_l3`, (same away_*)
- Market inputs: `opening_spread`, `pred_spread`, `closing_spread`, `pred_total`
- Feature diffs: `rot_to_swing_diff`, `exec_tax_diff`, `three_pt_fragility_diff`, `clv_delta`

### `market_lines.csv` (17 cols fully null)
Core market data never populated:
- `spread`, `total`, `over_under`, `dk_spread`, `home_spread_open`, `total_open`
- `home_ml`, `away_ml`, `home_money_pct`, `away_money_pct`, `home_tickets_pct`, `away_tickets_pct`
- `home_ats_wins`, `home_ats_losses`, `away_ats_wins`, `away_ats_losses`
- `line_movement`, `rlm_note`, `book_note`, `book_sharp_side`, `rlm_sharp_side`
> **Impact:** `market_lines.csv` is essentially empty for all pricing fields — market data is not being ingested.

### `csv/player_game_metrics.csv` — 100+ rolling stat columns all null
Every `_l5` and `_l10` rolling column is 100% null (e.g., `pts_l5`, `fgm_l5`, `efg_pct_l5`, etc.).
Both the `_l5`/`_l10` naming convention AND the `last_5_`/`last_10_` duplicate columns are null.
> **Root cause:** Rolling window computation is not running or not joining back to this table.

### `clv_report.csv`
- `home_spread_open` (100% null) — opening line not captured
- `clv_vs_open` (100% null) — cannot compute CLV without opening line

### `clv_by_submodel.csv`
- `mean_clv_vs_open` (100% null) — same root cause as above

### `line_movement_features.csv`
- `reverse_line_movement_flag`, `sharp_side`, `spread_move_direction`, `total_move_direction` — all null
> These are the primary signals from line movement; the directional and sharp-money flags are never set.

### `results_log_graded.csv` (many grading columns null)
Per-model graded results (`ens_ats_correct`, `ens_ou_correct`, `ens_wins_game`, all submodel `*_ats`, `*_spread`) are null — grading pipeline not running for ensemble columns.

### Predictions files (shared nulls across `predictions_latest`, `predictions_primary`, `predictions_history`, `predictions_mc_latest`, `predictions_combined_latest`)
- `home_ml`, `away_ml` — money-line odds never populated
- `home_momentum_tier`, `away_momentum_tier` — tier classification not running
- `trap_game_reason` — trap detection not firing reason text
- `bias_corrections_applied` — bias correction pipeline not logging
- All ensemble submodel columns (`adjefficiency_*`, `cagerankings_*`, `atsintelligence_*`, etc.) — not joined from ensemble output

---

## 2. High-Null Columns (>50% missing, not 100%)

### `backtest_training_data.csv`
- `closing_spread` (100%), `composite_luck_delta` (100%), `close_game_luck_delta` (100%), `pred_total` (99%), `away_cover_l10` (100%)
- ATS rotation features (`home_rot_efg_l5`, `rot_efg_delta`, etc.): 68% null
- Player availability deltas (`star_availability_delta`, `lineup_continuity_delta`, `new_starter_flag_*`): 68% null
- `home_rank`, `away_rank`: 92–95% null (ranking data not backfilled historically)

### `games.csv`
- `home_ml`, `away_ml`: 100% null — no ML odds in game log
- `odds_provider`, `odds_details`: 100% null — odds fields empty
- `spread`, `over_under`: 93% null — only 7% of games have market lines
- OT scores (`home_ot1`, `away_ot1`): 95–100% null — OT not captured

### `team_game_metrics.csv`, `team_game_sos.csv`, `team_game_weighted.csv`, `team_tournament_metrics.csv`
- `cover`, `cover_margin`, `ats_push`: 94% null — ATS tracking not populated
- `cover_l5`, `cover_l10`, `cover_margin_l5`, `cover_margin_l10`: 97% null — rolling cover windows empty
- `spread`, `over_under`: 93% null — odds not joined to game-level tables
- `opp_rank`, `opp_wins`, `opp_losses`, `opp_conference`, `conf_id`: 100% null — opponent context missing

### `team_season_summary_latest.csv`
- `luck_score`, `net_rtg_std`, `pyth_win_pct`: 95% null
- `ats_pct`, `ou_pct`, `sos_rank`: 100% null — season-level ATS/SOS tracking not computing

### `predictions_history.csv`
- Ensemble columns (`adjefficiency_*`, `cagerankings_*`, etc.): 87% null — only recent rows have ensemble data
- `_source_date`: 86% null

### `situational_features.csv`
- `prior_matchup_margin`, `prior_matchup_winner`, `revenge_margin`: 75% null
- `prev_loss_margin`: 55% null

### `team_tournament_metrics.csv`
- Player-depth features (`t_bench_pts_share_l5`, `t_top_scorer_efg_l5`): 95% null
- Tournament readiness (`t_readiness_composite`): 82% null
- Star reliance metrics (`t_star_reliance_risk`, `t_star_entropy`, etc.): 81% null

---

## 3. All-Zero Columns (column exists but all values = 0)

These columns are wired but the underlying computation always returns 0:

| File | Zero Columns |
|---|---|
| `backtest_predictions_with_context.csv` | `barthag_diff` |
| `backtest_training_data.csv` | `num_ot`, `home_conf_wins`, `home_conf_losses`, `away_conf_wins`, `away_conf_losses`, `home_cover_streak`, `away_cover_l10`, `home_regression_flag`, `away_regression_flag` |
| `bet_recs.csv` | `expected_roi` |
| `conference_summary.csv` | `conf_w`, `conf_l`, `conf_pct`, `streak` |
| `ensemble_predictions_latest.csv` | `cage_edge`, `barthag_diff` |
| `line_movement_features.csv` | `spread_move`, `spread_move_abs`, `steam_move_flag`, `spread_crossed_key_number`, `total_move`, `total_steam_flag`, `hours_of_movement` |
| `luck_regression_features.csv` | `regression_candidate_flag` |
| `player_game_metrics.csv` | `plus_minus`, `plus_minus_l5`, `plus_minus_l10`, `last_5_plus_minus`, `last_10_plus_minus` |
| `player_injury_proxy.csv` | `appeared`, `sudden_absence` |
| `predictions_combined_latest.csv` | `active_bias_corrections`, `trap_game_flag` |
| `predictions_graded.csv` | `home_off_eff_vs_exp`, `away_off_eff_vs_exp`, `edge_flag`, `active_bias_corrections`, `trap_game_flag`, `revenge_flag`, `home_fatigue_index_x`, `away_fatigue_index_x`, `ens_cage_edge`, `ens_barthag_diff`, `home_cover_streak`, `away_cover_streak` |
| `predictions_latest.csv` | `active_bias_corrections`, `trap_game_flag`, `away_fatigue_index` |
| `results_log.csv` | `neutral_site` |
| `results_log_graded.csv` | `neutral_site`, `primary_edge_flagged`, `ou_roi` |
| `team_ats_profile.csv` | `ats_pushes` |
| `team_game_logs.csv` through `team_tournament_metrics.csv` | `num_ot`, `conf_wins`, `conf_losses`, `ats_push` (across all game-level tables) |
| `team_resume.csv` | `neutral_wins` |
| `team_season_summary.csv` | `neutral_wins`, `neutral_losses` |
| `team_season_summary_latest.csv` | `ats_pushes`, `ortg_trend`, `drtg_trend`, `net_rtg_trend` |

**Key patterns:**
- `barthag_diff` and `cage_edge` are always 0 — these edge metrics are not being computed
- All `line_movement_features` movement columns are 0 — line movement tracking is not working (only 1 snapshot captured per game, so no deltas possible)
- `plus_minus` is always 0 for players — box score not capturing this field
- `conf_wins/conf_losses` always 0 — conference game tracking not running
- `active_bias_corrections` and `trap_game_flag` always 0 — these features not activating

---

## 4. Constant Columns (same value in every row — not metadata)

These are substantively meaningful columns stuck at one value:

### Predictions / Ensemble
- **`ens_total = '144.2'`** in ensemble and prediction files — all rows have the same total prediction (one game's output replicated everywhere, or pipeline not varying the total)
- **`ens_confidence = '0.188'` or `'0.193'`** — confidence score not varying across games
- **`ens_agreement = 'STRONG'`** — always strong, no variation
- All submodel spreads identical: `fourfactors_spread = adjefficiency_spread = momentum_spread = regressedeff_spread = '-3.2'` — models not differentiating
- **`pre_correction_prediction = '-2.89'`** — bias correction not varying

### Game Context
- **`neutral_site`**: Always `'1'` in backtest, always `'False'` in current predictions — venue detection not working
- **`home_field = '1'`** in backtest_training_data — all games coded as home-field regardless
- **`hca = '3.2'`** — home court advantage hardcoded, not varying by team or venue
- **`revenge_team = 'away'`** — revenge game detector always returns 'away', never 'home' or null

### Betting / Market
- **`market_prob = '0.5'`** in `bet_recs.csv` — market probability stuck at 50/50
- **`expected_roi = '0.0'`** in `bet_recs.csv` — ROI not being computed
- **`conf_w = conf_l = conf_pct = streak = '0'`** in `conference_summary.csv` — conference record completely missing

### Rankings
- **`resume_score = '50.0'`** in cbb_rankings files — resume scoring returning default for all teams
- **`luck = '0.0'`, `momentum = '0.0'`, `trend_numeric = '0.0'`** in older ranking snapshots — these metrics zeroed out in early-season files

### Player Data
- **`did_not_play = 'False'`** in `player_game_logs.csv`, `player_game_metrics.csv`, `player_injury_proxy.csv` — DNP flag never set (but these files appear to only contain players who played)
- **`home_pg_starters_count = '5.0'`** — starters count hardcoded to 5

---

## 5. Prioritized Issues by Impact

### P0 — Broken pipelines (data never flowing)
1. **Line movement features** — `spread_move`, `steam_move_flag`, directional flags all 0 or null. Only 1 snapshot per game means no deltas are possible.
2. **Market lines** — `market_lines.csv` has all pricing fields null. Market data ingestion is not working.
3. **`plus_minus`** — Always 0 in player tables. Box score parsing missing this field.
4. **Conference record tracking** — `conf_wins`, `conf_losses`, `conf_pct` always 0 across all game-level and summary tables.
5. **Rolling window stats in `csv/player_game_metrics.csv`** — 100+ columns all null. Rolling join is failing.

### P1 — Features wired but not computing
6. **`barthag_diff` / `cage_edge`** — Always 0. Barttorvik/CageRankings integration not feeding differential.
7. **`active_bias_corrections` / `trap_game_flag` / `revenge_flag`** — Always 0. These detection modules not triggering.
8. **`home_momentum_tier` / `away_momentum_tier`** — Null across all prediction files. Tier classification not running.
9. **`expected_roi`** in bet_recs — Always 0. Kelly criterion / ROI not computed.
10. **`ens_total` / submodel spreads all identical** — Ensemble predictions not varying. Possible data join error returning only 1 game's data for all rows.

### P2 — Missing market/historical data
11. **ML odds** (`home_ml`, `away_ml`) — Missing across all files. Moneyline not being scraped.
12. **ATS/cover tracking** (`cover_l5`, `cover_l10`, `ats_push`) — 94–97% null. ATS history not joining to game tables.
13. **Opening spread** — Missing from CLV report, market lines, predictions. Opening line not captured.
14. **Opponent context** (`opp_rank`, `opp_wins`, `opp_losses`) — 100% null in game tables. Opponent join not working.

### P3 — Partially populated (data exists but sparse)
15. **`cbb_rankings.csv`** — ~40% of teams missing `conference`, `tourney_readiness`, `star_risk`, `close_wpct`. Only teams with ≥10 recent games populated.
16. **ATS situational splits** in `team_season_summary_latest.csv` — `blowout_ats_pct` (93% null), `close_game_ats_pct` (87% null) — sample sizes too small.
17. **`clv_vs_open`** — Null everywhere because opening spread never captured.
