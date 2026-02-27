# CBB Pipeline — Detailed File-Level Flow Map
> Generated 2026-02-27. Reflects repo state as of `main` branch.

---

## Table of Contents
1. [Artifact → Producer → Dependencies Map](#artifact--producer--dependencies-map)
2. [Player-Level Data: Ingestion and Downstream Usage](#player-level-data-ingestion-and-downstream-usage)
3. [Top 3 Lowest-Lift Player Feature Integration Points](#top-3-lowest-lift-player-feature-integration-points)

---

## Artifact → Producer → Dependencies Map

### RAW INGESTION

| Artifact | Producer Script/Function | Input Dependencies | Key Schema Columns |
|---|---|---|---|
| `data/raw_json/scoreboard/YYYYMMDD.json` | `espn_client.fetch_scoreboard()` | ESPN API | Raw event JSON |
| `data/raw_json/summaries/{event_id}.json` | `espn_client.fetch_summary()` | ESPN API | Raw boxscore JSON |
| `data/games.csv` | `espn_pipeline.build_games()` via `espn_parsers.parse_scoreboard_event()` | scoreboard JSON | `game_id, home_team_id, away_team_id, home_conference, away_conference, home_wins, away_wins, spread, over_under, home_ml, away_ml, home_h1, home_h2, away_h1, away_h2, completed, game_datetime_utc, neutral_site` |
| `data/team_game_logs.csv` | `espn_pipeline.build_team_and_player_logs()` via `espn_parsers.summary_to_team_rows()` + `_enrich_team_rows_from_scoreboard()` | summary JSON + games.csv (for conference/standings fill) | `event_id, team_id, opponent_id, home_away, conference, wins, losses, fgm, fga, tpm, tpa, ftm, fta, orb, drb, reb, ast, tov, stl, blk, pf, pts, opp_pts, spread, over_under, h1_pts, h2_pts, game_datetime_utc` |
| `data/player_game_logs.csv` | `espn_pipeline.build_team_and_player_logs()` via `espn_parsers.parse_summary()` → `.players` list | summary JSON | `event_id, team_id, athlete_id, player, jersey, position, starter, did_not_play, min, pts, fgm, fga, tpm, tpa, ftm, fta, orb, drb, reb, ast, stl, blk, tov, pf, plus_minus, game_datetime_utc` |
| `data/market_lines.csv` | `ingestion/market_lines.py` (ESPN API + odds endpoints) | ESPN schedule + odds API | `event_id, capture_type, captured_at_utc, home_spread_open, home_spread_current, line_movement, total_open, total_current, pinnacle_spread, draftkings_spread, home_tickets_pct, steam_flag, rlm_flag, rlm_sharp_side` |
| `data/pipeline_run_log.csv` | `cbb_pipeline_logger.log_pipeline_run()` | pipeline run stats | `run_id, trigger, days_back, games_found, games_parsed, games_failed, team_rows_written, player_rows_written, status, parse_version, ts_utc` |

---

### TEAM FEATURE CHAIN (Sequential — each stage feeds next)

| Artifact | Producer Script/Function | Input Dependencies | Key Schema Columns Added |
|---|---|---|---|
| `data/team_game_metrics.csv` | `espn_metrics.compute_all_metrics()` called from `espn_pipeline.py` | `team_game_logs.csv` | `ortg, drtg, net_rtg, poss, pace, efg_pct, ts_pct, three_pct, tov_pct, orb_pct, drb_pct, ftr, h1_ortg, h2_ortg, margin, margin_capped, ast_tov_ratio, net_rtg_l5, net_rtg_l10, ortg_l5, ortg_l10, drtg_l5, drtg_l10, pace_l5, pace_l10, efg_l5, efg_l10` |
| `data/team_game_sos.csv` | `espn_sos.compute_sos_metrics()` called from `espn_pipeline.py` | `team_game_metrics.csv` | `opp_avg_net_rtg, opp_avg_ortg, opp_avg_drtg, opp_avg_pace, efg_vs_opp, tov_vs_opp, orb_vs_opp, ftr_vs_opp, efg_vs_opp_season, tov_vs_opp_season, opp_avg_net_rtg_season, wab, sos` |
| `data/team_game_weighted.csv` | `espn_weighted_metrics.compute_weighted_metrics()` called from `espn_pipeline.py` | `team_game_sos.csv` | `ortg_wtd_off_l5, ortg_wtd_off_l10, drtg_wtd_def_l5, drtg_wtd_def_l10, efg_pct_wtd_off_l5, net_rtg_wtd_qual_l5, net_rtg_wtd_qual_l10, adj_ortg, adj_drtg, adj_net_rtg, adj_pace` (+ all SOS + metrics columns inherited) |
| `data/team_tournament_metrics.csv` | `espn_tournament.compute_tournament_metrics()` called from `espn_pipeline.py` | `team_game_weighted.csv` + optional `player_game_metrics.csv` | `t_tournament_dna_score, t_suffocation_rating, t_momentum_quality_rating, t_star_reliance_risk, t_offensive_identity_score, t_regression_risk_flag, t_floor_em, t_ceiling_em, game_total_projection, total_confidence, underdog_winner_score (t_uws_total), uwp_upset_probability, game_story` |
| `data/team_pretournament_snapshot.csv` | `espn_tournament.build_pretournament_snapshot()` called from `espn_pipeline.py` | `team_tournament_metrics.csv` (latest game row per team) | All columns from tournament_metrics — one row per team (most recent game) |
| `data/cbb_rankings.csv` | `espn_rankings.run()` called from `espn_pipeline.py` + standalone `python espn_rankings.py --top 25` | `team_pretournament_snapshot.csv` | `rank, team, team_id, conference, cage_em, cage_o, cage_d, cage_t, barthag, wab, eff_grade, cage_power_index, net_rtg_l5, net_rtg_l10, pythagorean_win_pct` |
| `data/cbb_rankings_by_conference.csv` | `espn_rankings.run()` | `cbb_rankings.csv` | `conference, avg_cage_em, top_team, top_em, n_teams` |

---

### SUPPLEMENTAL TEAM FEATURE TABLES

| Artifact | Producer Script/Function | Input Dependencies | Key Schema Columns |
|---|---|---|---|
| `data/team_rolling_l5.csv` | `build_derived_csvs.py` or pipeline | `team_game_metrics.csv` | `team_id, net_rtg_l5, ortg_l5, drtg_l5, pace_l5, efg_l5, tov_l5, game_datetime_utc` |
| `data/team_rolling_l10.csv` | `build_derived_csvs.py` or pipeline | `team_game_metrics.csv` | same columns at L10 window |
| `data/team_weighted_rolling.csv` | `espn_weighted_metrics.py` (secondary output) via `OUT_WEIGHTED_ROLLING` | `team_game_sos.csv` | opponent-weighted L5/L10 subset |
| `data/team_game_halfsplits.csv` | `espn_metrics.py` (half-score section) | `team_game_logs.csv` | `event_id, team_id, h1_ortg, h2_ortg, h1_pts, h2_pts, h1_pts_against, h2_pts_against, h1_margin, h2_margin` |
| `data/team_ats_profile.csv` | `cbb_backtester.py` or `build_derived_csvs.py` | `results_log_graded.csv` + team history | `team_id, cover_rate_season, cover_rate_l10, ats_margin_l10, home_cover_pct, away_cover_pct, cover_streak` |
| `data/team_luck_regression.csv` | `build_derived_csvs.py` | `team_game_metrics.csv` | `team_id, luck_index, pythagorean_win_pct, actual_win_pct, luck_delta, regression_risk_flag` |
| `data/team_situational.csv` | `build_derived_csvs.py` | `team_game_weighted.csv` + `results_log_graded.csv` | `team_id, home_net_rtg, away_net_rtg, rest_split_short_net, rest_split_long_net, back_to_back_margin` |
| `data/team_resume.csv` | `build_derived_csvs.py` | `team_game_sos.csv` | `team_id, wab, q1_record, q2_record, q3_record, bad_loss_count, resume_score` |
| `data/team_injury_impact.csv` | `espn_injury_proxy.compute_team_injury_impact()` | `player_injury_proxy.csv` + `team_game_logs.csv` | `event_id, team_id, team_injury_burden, n_injured_players, top_scorer_injured, injury_impact_rtg_adj` |
| `data/team_matchup_history.csv` | `build_derived_csvs.py` | `team_game_logs.csv` | `team_a_id, team_b_id, h2h_record, avg_margin, last_meeting_date` |
| `data/conference_daily_summary.csv` | `build_derived_csvs.py` | `team_game_weighted.csv` + `cbb_rankings.csv` | `conference, avg_cage_em, avg_net_rtg, avg_pace, n_teams, games_today` |
| `data/team_travel_fatigue.csv` | `cbb_travel_fatigue.py` | `games.csv` + `data/venue_geocodes.csv` | `event_id, team_id, travel_miles, timezone_change, travel_fatigue_index` |
| `data/team_season_summary.csv` | `cbb_season_summaries.py` | `team_game_metrics.csv` + `team_game_sos.csv` | `team_id, season, wins, losses, avg_ortg, avg_drtg, avg_pace, avg_net_rtg, sos` |
| `data/team_season_summary_latest.csv` | `cbb_season_summaries.py` | `team_season_summary.csv` | Same — one row per team, most recent season |
| `data/team_pretournament_snapshot.csv` | `espn_tournament.build_pretournament_snapshot()` | `team_tournament_metrics.csv` | One row per team (latest game), all t_ fields included |

---

### PLAYER FEATURE CHAIN

| Artifact | Producer Script/Function | Input Dependencies | Key Schema Columns |
|---|---|---|---|
| `data/player_game_metrics.csv` | `espn_player_metrics.compute_player_metrics()` | `player_game_logs.csv` + `team_game_metrics.csv` (for team possessions) | `event_id, team_id, athlete_id, player, starter, did_not_play, min, pts, fgm, fga, tpm, tpa, ftm, fta, orb, drb, reb, ast, stl, blk, tov, pf, plus_minus, efg_pct, ts_pct, fg_pct, three_pct, ft_pct, usage_rate, ast_tov_ratio, pts_per_fga, floor_pct, pts_l5, pts_l10, efg_pct_l5, efg_pct_l10, usage_rate_l5, usage_rate_l10, min_l5, min_l10, games_played, efg_std_l10, pts_std_l10, pts_season_avg, min_season_avg, efg_pct_season_avg` |
| `data/player_rolling_l5.csv` | `espn_player_metrics._write_player_splits()` | `player_game_metrics.csv` (subset) | `athlete_id, event_id, team_id, game_datetime_utc` + all `*_l5` columns |
| `data/player_role_splits.csv` | `espn_player_metrics._write_player_splits()` | `player_game_metrics.csv` (subset) | `athlete_id, event_id, starter, pts_starter_l5, pts_bench_l5, min_starter_l5, min_bench_l5, efg_pct_starter_l5, efg_pct_bench_l5, usage_rate_starter_l5, usage_rate_bench_l5` |
| `data/player_injury_proxy.csv` | `espn_injury_proxy.compute_injury_proxy()` | `player_game_logs.csv` + `team_game_logs.csv` | `event_id, athlete_id, team_id, dnp_flag, dnp_prev_game, games_missed_l14, sudden_absence, min_drop_flag, min_drop_severe, starter_to_bench, pts_z_l10, efg_z_l10, usage_z_l10, multi_stat_down, injury_proxy_score, injury_proxy_flag, injury_proxy_severe` |
| `data/player_condition_profiles.csv` | `cbb_player_matchup.py --build-profiles-only` | `player_game_metrics.csv` + `team_game_logs.csv` | `athlete_id, player, team_id, defense_tier_pts, defense_tier_efg, pace_fast_pts, pace_slow_pts, home_pts, away_pts, short_rest_pts, long_rest_pts, pcs_composite` |
| `data/player_archetype_matchup_matrix.csv` | `cbb_player_matchup.py` | `player_game_metrics.csv` + `cbb_rankings.csv` | `archetype, vs_defense_tier, avg_pts, avg_efg_pct, avg_usage, sample_size` |

---

### PREDICTION OUTPUTS

| Artifact | Producer Script/Function | Input Dependencies | Key Schema Columns |
|---|---|---|---|
| `data/predictions_YYYYMMDD.csv` | `espn_prediction_runner.py` → `CBBPredictionModel.predict_game()` | `team_game_weighted.csv` (primary), `games.csv`, `team_pretournament_snapshot.csv` | `game_id, game_datetime_utc, home_team, away_team, home_team_id, away_team_id, pred_spread, pred_total, model_confidence, edge_pts, edge_flag, kelly_fraction, spread_line, total_line, spread_diff_vs_line, rest_days_home, rest_days_away, home_conf, away_conf, neutral_site, breakdown (JSON)` |
| `data/predictions_primary.csv` | `cbb_predictions_rolling.yml` (merge of multi-date outputs) | `predictions_YYYYMMDD.csv` | Same as above — deduped, window-filtered |
| `data/predictions_latest.csv` | `espn_prediction_runner.py` + workflow | Same | Alias — always reflects most recent run |
| `data/ensemble_predictions_YYYYMMDD.csv` | `cbb_ensemble.EnsemblePredictor.predict()` via workflow | `team_pretournament_snapshot.csv` → `TeamProfile` + `predictions_primary.csv` (for game list + rest/lines) | `game_id, ensemble_spread, ensemble_total, ensemble_confidence, model_agreement, spread_std, cage_edge, barthag_diff, edge_flag_spread, edge_flag_total, M1_spread…M7_spread` |
| `data/ensemble_predictions_latest.csv` | workflow | Same | Alias |
| `data/predictions_combined_latest.csv` | `cbb_predictions_rolling.yml` inline merge | `predictions_primary.csv` + `ensemble_predictions_latest.csv` | Primary columns + `ens_*` prefixed ensemble columns + `spread_model_gap` |
| `data/predictions_mc_latest.csv` | `cbb_monte_carlo.py` | `predictions_combined_latest.csv` | `game_id, mc_home_win_pct, mc_spread_mean, mc_spread_std, mc_total_mean, mc_total_std, mc_confidence, mc_cover_pct` |
| `data/predictions_with_context.csv` | `enrichment/predictions_with_context.py` | `predictions_combined_latest.csv` + `market_lines.csv` + ESPN team API | `game_id, home_rest_days, away_rest_days, home_wins, away_wins, home_spread_current, total_current, line_movement, steam_flag, rlm_flag, alpha_edge, home_record, away_record` |
| `data/predictions_divergence_latest.csv` | workflow (inline) | `predictions_combined_latest.csv` | Subset of combined — only rows where `|pred_spread − ens_ensemble_spread| ≥ 3.0` |

---

### EVALUATION & BACKTEST OUTPUTS

| Artifact | Producer Script/Function | Input Dependencies | Key Schema Columns |
|---|---|---|---|
| `data/results_log_graded.csv` | `cbb_results_tracker.py` | `predictions_latest.csv` + ESPN final scores (live API) | `game_id, game_date, home_team, away_team, actual_home_score, actual_away_score, actual_margin, pred_spread, covered_ats, pred_total, hit_over, result_correct, edge_pts, confidence_bucket` |
| `data/csv/backtest_summary.csv` | `cbb_backtester.py` | Full `team_game_weighted.csv` history + `results_log_graded.csv` | `n_games, ats_win_pct, ats_roi, mae_spread, mae_total, brier_score, edge_roi` |
| `data/csv/backtest_by_model.csv` | `cbb_backtester.py` | Same | `model_name, ats_win_pct, mae_spread, n_games` for each sub-model |
| `data/csv/backtest_weekly.csv` | `cbb_backtester.py` | Same | Weekly performance slices |
| `data/csv/backtest_by_conference.csv` | `cbb_backtester.py` | Same | Conference-level accuracy breakdown |
| `data/csv/backtest_calibration.csv` | `cbb_backtester.py` | Same | Confidence bucket calibration curves |
| `data/csv/backtest_by_edge.csv` | `cbb_backtester.py` | Same | ROI by edge_pts bucket |
| `data/csv/backtest_model_matrix.csv` | `cbb_backtester.py` | Same | Full model × date grid |

---

### DERIVED FRONTEND OUTPUTS

| Artifact | Producer Script/Function | Input Dependencies |
|---|---|---|
| `data/csv/bet_recs.csv` | `build_derived_csvs.py` | `predictions_combined_latest.csv` (edge ≥ 3.75) |
| `data/csv/team_form_snapshot.csv` | `build_derived_csvs.py` | `team_game_weighted.csv` |
| `data/csv/matchup_preview.csv` | `build_derived_csvs.py` | `predictions_combined_latest.csv` + `team_pretournament_snapshot.csv` |
| `data/csv/player_leaders.csv` | `build_derived_csvs.py` | `player_game_metrics.csv` |
| `data/csv/upset_watch.csv` | `build_derived_csvs.py` | `predictions_combined_latest.csv` (uws_total ≥ 40) |
| `data/csv/conference_summary.csv` | `build_derived_csvs.py` | `cbb_rankings.csv` + `team_game_weighted.csv` |
| `data/csv/player_context_scores.csv` | `cbb_player_matchup.py` | `player_game_metrics.csv` + `predictions_combined_latest.csv` |
| `data/csv/team_matchup_summary.csv` | `cbb_player_matchup.py` | player context scores + team_game_logs |

---

## Player-Level Data: Ingestion and Downstream Usage

### What Is Ingested

Every completed game triggers a full ESPN boxscore fetch. The parser extracts a player row per athlete per game containing:

```
player_game_logs.csv:
  Raw box scores: min, pts, fgm/fga, tpm/tpa, ftm/fta, orb, drb, reb,
                  ast, stl, blk, tov, pf, plus_minus
  Context: event_id, team_id, athlete_id, starter, did_not_play
```

From this, two downstream player tables are computed every run:

```
player_game_metrics.csv:   Per-game derived stats + rolling L5/L10 for all stats
player_rolling_l5.csv:     L5 rolling subset only
player_role_splits.csv:    Starter vs bench performance splits
player_injury_proxy.csv:   Proxy injury detection (DNP, minute-drop, z-score signals)
team_injury_impact.csv:    Team-level aggregation of injury burden
```

Plus (after predictions run):
```
player_condition_profiles.csv:       How each player performs vs defense tier/pace/rest
player_archetype_matchup_matrix.csv: Archetype-based matchup scoring
player_context_scores.csv:           PCS per game per player
```

### Where Player Data IS Used in Predictions

| Use point | How |
|---|---|
| `espn_tournament.compute_tournament_metrics()` | Reads `player_game_metrics.csv` (optional) to compute `t_star_reliance_risk` — the fraction of team scoring concentrated in top 1-2 players. Flows into `team_tournament_metrics.csv` → `team_pretournament_snapshot.csv` → ensemble `TeamProfile.star_risk`. |
| `cbb_ensemble.py` M6 (CAGERankings) | `star_risk` field on `TeamProfile` is used to penalize confidence: `risk_penalty = (home.star_risk + away.star_risk - 100) / 200` — reduces spread/total confidence (not the spread/total value itself). |
| `cbb_player_matchup.py` | Full player context (PCS, condition profiles, archetype matchups) — appended as columns to `player_context_scores.csv` and `team_matchup_summary.csv`. These are **informational outputs**, not inputs to spread/total calculations. |

### Where Player Data Is NOT Used

| Gap | Details |
|---|---|
| `espn_prediction_runner.py` (primary model) | Does **not** read `player_game_metrics.csv`, `player_injury_proxy.csv`, or `team_injury_impact.csv`. GameData is built entirely from `team_game_weighted.csv`. |
| `CBBPredictionModel` features | All features are team-aggregate box scores and efficiency metrics. No player-level features exist in `GameData`. |
| Ensemble `TeamProfile` | No player-level rolling stats (e.g., top scorer efg_pct_l5, bench depth score) are present. Only `star_risk` (a scalar) touches player data. |
| `team_injury_impact.csv` | Computed and stored, but **not joined** into any prediction input. |
| `player_rolling_l5.csv`, `player_role_splits.csv` | Stored in `data/`, not consumed downstream of ingestion. |

**Summary:** Player box scores are fully ingested, processed into rich metrics, and stored — but they do not materially influence spread or total predictions beyond the `star_reliance_risk` confidence penalty in the ensemble.

---

## Top 3 Lowest-Lift Places to Add Player-Derived Features for Spreads/Totals

### Rank 1 — Add player aggregates to `team_pretournament_snapshot.csv`

**Where:** `espn_tournament.build_pretournament_snapshot()` in `espn_tournament.py`

**Lift:** Low. The snapshot already joins the latest game row per team. Adding a join to `player_game_metrics.csv` here requires ~10 lines of pandas.

**What to add:**
```python
# Join from player_game_metrics.csv grouped by team_id (latest N games)
top_scorer_efg_l5          # top scorer's eFG% over last 5 games
bench_pts_share_l5         # % of team pts from bench players (depth signal)
team_injury_burden         # from team_injury_impact.csv — sum of injury_proxy_score for key players
n_injured_starters_l3      # from player_injury_proxy — starters with injury_proxy_flag in last 3 games
```

**Why it matters for spreads/totals:**
- `top_scorer_efg_l5` is a leading indicator for offensive efficiency vs what adj_ortg captures (adj_ortg lags by full season)
- `team_injury_burden` directly affects scoring output but is currently invisible to all 7 ensemble sub-models
- Once in the snapshot, these fields become automatically available to `load_team_profiles()` and the ensemble with zero additional changes

**Flow change:**
```
player_game_metrics.csv ──┐
team_injury_impact.csv ───┤
                           ├──► build_pretournament_snapshot() ──► TeamProfile
team_tournament_metrics ──┘                                       (new fields available to M1–M7)
```

---

### Rank 2 — Join `team_injury_impact.csv` into `team_game_weighted.csv` upstream

**Where:** `espn_weighted_metrics.compute_weighted_metrics()` in `espn_weighted_metrics.py`, or as a post-step in `espn_pipeline.py` after `compute_weighted_metrics()`

**Lift:** Low. `team_injury_impact.csv` is already keyed on `(event_id, team_id)` — the same keys as `team_game_weighted.csv`. A left-join adds the columns transparently.

**What to add:**
```python
injury_impact_rtg_adj      # from team_injury_impact.csv — estimated pts/100 impact
n_injured_players          # count of players with injury_proxy_flag
top_scorer_injured         # bool — top-scorer on injury proxy
```

**Why it matters:**
- The primary prediction model (`espn_prediction_runner.py`) reads `team_game_weighted.csv` as its core input. Any columns added there become available in `GameData` with minimal code changes to `espn_prediction_runner.py`.
- `injury_impact_rtg_adj` can be used as an additive offset to `adj_ortg`/`adj_drtg` during `GameData` construction — this directly shifts the spread/total projection.

**Flow change:**
```
team_game_sos.csv ──────────────────────────────────┐
team_injury_impact.csv ─── left join on (event_id, team_id) ──► team_game_weighted.csv
                                                      │         (injury columns now available)
                                                      └──► espn_prediction_runner.py → GameData
```

---

### Rank 3 — Add team-level player aggregate columns to `espn_prediction_runner.py` at `GameData` build time

**Where:** `espn_prediction_runner.py`, in the section that builds `GameData` objects from `team_game_weighted.csv` rows

**Lift:** Moderate-low. Requires loading one additional CSV (`player_game_metrics.csv`) in the runner, grouping by `(team_id, game_datetime_utc ≤ cutoff)` to maintain leak-free history, then injecting aggregates into `GameData`.

**What to add:**
```python
# Per-team, looking back from game cutoff:
top2_scorers_efg_l5        # weighted eFG% of top-2 scorers (usage-weighted)
bench_scoring_share_l5     # % of team points from bench last 5 games
roster_usage_concentration # Herfindahl index of usage distribution (star reliance proxy)
minutes_injury_gap_l3      # total star-player minutes below season avg in last 3 games
```

**Why it matters for totals specifically:**
- `bench_scoring_share_l5` predicts low-scoring games better than team avg pace/ortg (shallow benches = lower totals in grind games)
- `roster_usage_concentration` is a better real-time `star_risk` than the season-level `t_star_reliance_risk` already in TeamProfile
- `minutes_injury_gap_l3` is a leading indicator for total suppression even when no official injury is reported

**Flow change:**
```
player_game_metrics.csv ──► (grouped by team_id, cutoff ≤ game_datetime_utc) ──┐
                                                                                 ├──► GameData
team_game_weighted.csv (existing) ──────────────────────────────────────────────┘    (new fields)
                                                                                 └──► CBBPredictionModel
                                                                                      (needs feature hooks in _calculate_matchup)
```

**Trade-off note:** This is the most powerful lift but requires a leak-safe cutoff join (already pattern-established in the runner for `opponent_history`) and a small change to `cbb_prediction_model.py` to consume the new fields. The `GameData` dataclass currently has no player-level fields — adding them requires schema extension.

---

## Cross-Cutting Concerns

### Schema Enforcement
- `cbb_output_schemas.py`: `validate_output()` called on `team_game_logs` and `player_game_logs` at write time
- `config/schemas.py`: column schemas for downstream validation
- `check_schema_drift.py`: detects column additions/removals vs audit baseline (`data/audit_baseline.json`)
- `build_schema_registry.py`: generates `data/schema_registry.json`

### Deduplication Strategy
- All CSVs use `_append_dedupe_write()` in `espn_pipeline.py`
- Deduplication keys: `(event_id, team_id)` for team tables, `(event_id, athlete_id)` for player tables
- Preference order: `completed=True` rows first, then newest `pulled_at_utc`
- Odds columns are preserved across state transitions (pre→live→final) via group-fill

### Artifact Handoff (GitHub Actions)
- Pipeline writes CSVs → uploads as `espn-cbb-csvs` artifact
- Predictions workflow downloads artifact → runs models → uploads `cbb-predictions-rolling-latest`
- Market lines workflow downloads predictions artifact → captures odds → uploads `cbb-market-lines`
- Fallback: `git show HEAD:<file>` for any artifact that failed to download
