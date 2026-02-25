# CBB Pipeline Data Lineage Report
Generated: 2026-02-25

## Executive Summary
1. **`pred_spread` is computed in `CBBPredictionModel._predict_from_stats()` as `predicted_spread = -final_edge` (negative = home favored), then written as `pred_spread` in `espn_prediction_runner.run_predictions()`; later merge steps explicitly protect it, but there are still multiple coercion/recovery paths where nulls can be introduced.** (`cbb_prediction_model.py:_predict_from_stats`; `espn_prediction_runner.py:595-671`; `enrichment/predictions_with_context.py:381-393, 455-489, 1021-1029`)
2. **`ens_ens_spread` origin is not found in Python assignment code in this repo (ORIGIN UNKNOWN in code), but downstream modules treat it as an ensemble spread alias and fallback to it when `pred_spread` is null.** (`enrichment/predictions_with_context.py:383, 815-816, 875-877, 1023-1025`; `cbb_monte_carlo.py:577-581`)
3. **Confirmed net-efficiency alias mismatch risk exists for `cage_em`: profile loader prioritizes `adj_net_rtg/net_eff/net_rtg/cage_em`, logs warning when all teams become 0.0, and defaults silently to 0.0.** (`cbb_ensemble.py:1215-1246, 1290-1304, 1440-1447`)
4. **`spread_diff_vs_line` and CLV columns use `market_line - pred_spread` convention (not `pred - line`), consistent with comments that spread is away-minus-home. This sign choice affects ATS/CLV interpretation everywhere.** (`enrichment/predictions_with_context.py:910-935, 945-950`; `cbb_prediction_model.py:815-817`)
5. **Many merges omit explicit `suffixes=` and rely on defaults (`_x/_y`), and some are `inner` joins (notably Monte Carlo join), creating row-drop and collision risk.** (`enrichment/predictions_with_context.py:559-560, 694, 703`; `cbb_monte_carlo.py:832`)

---

## Data Source Map

### 1) ESPN Scoreboard API (schedules/scores/odds)
- **Endpoint**: `https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?...` (`espn_config.py:212-215`; `ingestion/market_lines.py:143-149`)
- **Caller files**: `espn_client.fetch_scoreboard()` and `ingestion.market_lines.fetch_espn_scoreboard()` (`espn_client.py:48-52`; `ingestion/market_lines.py:140-151`)
- **Raw parsed fields**:
  - Game identifiers/datetime/status/venue, home/away team IDs/names/records/ranks, linescore splits, odds (`espn_parsers.parse_scoreboard_event`) (`espn_parsers.py:340-407`)
- **First rename / first write**:
  - Parsed to flat dict with `game_id`, `home_team_id`, `away_team_id`, `spread`, `over_under`, etc. first lands in `data/games.csv` via pipeline writer (`espn_parsers.py:368-406`; `espn_config.py:20`)
- **Market ingestion path**:
  - Parsed as market row fields (`event_id`, `home_spread_current`, `total_current`) then written to `data/market_lines.csv` (`ingestion/market_lines.py:340-382, 469-552`)

### 2) ESPN Summary API (team + player stats)
- **Endpoint**: `https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={event_id}` (`espn_config.py:216-218`)
- **Caller**: `espn_client.fetch_summary()` (`espn_client.py:55-59`)
- **Raw parsed fields**:
  - Team metadata/stats, per-player box score stats, odds/status (`espn_parsers.parse_summary`) (`espn_parsers.py:413+`)
- **First rename / first write**:
  - Team rows via `summary_to_team_rows()` to team-game logs/metrics chain (`espn_parsers.py:700+`)
  - Player rows emitted with canonical stat aliases (`fgm/fga/tpm/tpa/...`) (`espn_parsers.py` player mapping block)
  - First CSVs: `team_game_logs.csv`, `player_game_logs.csv` (`espn_config.py:21-22`)

### 3) ESPN Team endpoint (team records W-L)
- **Endpoint**: `https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}` (`enrichment/predictions_with_context.py:35-36`)
- **Caller**: `_fetch_team_record()` (`enrichment/predictions_with_context.py:30-50`)
- **Parsed fields**:
  - `wins`, `losses` extracted from `team.record.items[0].stats` (`enrichment/predictions_with_context.py:41-47`)
- **First write**:
  - `data/team_records.csv` cache (`enrichment/predictions_with_context.py:166-168`)
  - Then mapped into `home_wins/home_losses/away_wins/away_losses` in predictions context frame (`enrichment/predictions_with_context.py:170-173`)

### 4) ESPN odds/win-probability/ATS core endpoints (gap fillers)
- **Endpoints**:
  - Core odds: `.../events/{event}/competitions/{comp}/odds`
  - Core probabilities: `.../events/{event}/competitions/{comp}/probabilities`
  - Team ATS: `.../seasons/{year}/types/{season_type}/teams/{team_id}/ats`
  - Pickcenter via summary endpoint (`espn_gap_fillers.py:35-37, 75-78, 105-108, 124-126`)
- **Caller**: `fill_market_row_gaps()` from market ingestion (`ingestion/market_lines.py` import and usage)
- **Parsed fields**:
  - odds (`spread/over_under/home_ml/away_ml`), win probabilities, ATS totals (`espn_gap_fillers.py:57-65, 116-119, 130-135`)
- **First write**:
  - into market row fields (`home_win_prob`, `away_win_prob`, `home_ats_wins/losses`, `away_ats_wins/losses`) then `market_lines.csv` (`ingestion/market_lines.py:546-551, 595`)

### 5) Action Network odds API
- **Endpoint**: `https://api.actionnetwork.com/web/v1/scoreboard/ncaab?...` (`ingestion/market_lines.py:126-127`)
- **Caller**: `fetch_action_network()` (`ingestion/market_lines.py:123-138`)
- **Parsed fields**:
  - open/current spread, totals, ticket/money %, DK/Pinn fallback spreads (`ingestion/market_lines.py:316-334`)
- **First write**: merged into market row builder and written to `market_lines.csv` (`ingestion/market_lines.py:664-691, 693, 595`)

### 6) Pinnacle guest API
- **Endpoints**: `.../matchups`, `.../markets/straight` (`ingestion/market_lines.py:158-166`)
- **Caller**: `fetch_pinnacle_lines()` (`ingestion/market_lines.py:155-188`)
- **Parsed fields**: participant names + spread line (`ingestion/market_lines.py:239-280`)
- **First write**: `pinnacle_spread` in `market_lines.csv` (`ingestion/market_lines.py:528`)

### 7) DraftKings API
- **Endpoint**: `https://sportsbook.draftkings.com/sites/US-SB/api/v1/eventgroups/{id}?format=json` (`ingestion/market_lines.py:192`)
- **Caller**: `fetch_draftkings_lines()` (`ingestion/market_lines.py:190-208`)
- **Parsed fields**: event-level spread lookups (`ingestion/market_lines.py` index helpers)
- **First write**: `draftkings_spread` in `market_lines.csv` (`ingestion/market_lines.py:529`)

### 8) ESPN Rankings endpoint
- **Status**: **No standalone HTTP rankings endpoint call found in `.py` files** from repo grep. Rankings are computed locally from CSV pipeline artifacts (`espn_rankings.py` loads snapshot/log from local files). (`espn_rankings.py:89-151`)

### 9) ESPN Player stats endpoint
- **Status**: **No dedicated separate player endpoint call found**; player stats are obtained from the ESPN game summary payload (`boxscore.players`) parsed in `parse_summary`. (`espn_client.py:55-59`; `espn_parsers.py:413+`)

### 10) Other external URLs found
- ESPN team record endpoint in context enrichment (`enrichment/predictions_with_context.py:35-39`)
- All URLs discovered by `rg -n "requests.get|http|https" --glob "*.py"` are captured above.

---

## Column Lifecycle Traces (20 columns)

### 1) `pred_spread`
- **Origin**: model output `predicted_spread = -final_edge` (`cbb_prediction_model.py:815-817`)
- **Runner write**: mapped to `pred_spread` in row dict (`espn_prediction_runner.py:669`)
- **Normalize stage**: `normalize_column_names()` can map `predicted_spread -> pred_spread`, but runner also re-adds `predicted_spread` alias after normalization (`pipeline_csv_utils.py:126-143`; `espn_prediction_runner.py:1120-1122`)
- **Context stage**:
  - saved before normalize, restored if missing/high-null (`enrichment/predictions_with_context.py:381-393`)
  - market merge backs up/restores `pred_spread` (`enrichment/predictions_with_context.py:455-489`)
  - final fallback from `ens_ens_spread` if null (`enrichment/predictions_with_context.py:1021-1025`)
- **Final output name**: remains `pred_spread` plus alias `predicted_spread` in context compute block (`enrichment/predictions_with_context.py:938-940`)
- **Null/default risks**: `to_numeric(..., errors='coerce')`; missing source columns; failed prediction rows skipped.

### 2) `ens_ens_spread`
- **Origin**: **ORIGIN UNKNOWN in Python assignment code** (no direct producer found by grep in `.py`); appears expected in combined CSV contract and Monte Carlo inputs. (`cbb_monte_carlo.py:577, 729`; `enrichment/predictions_with_context.py:383`)
- **Relation to `pred_spread`**:
  - used as fallback when `pred_spread` missing (`enrichment/predictions_with_context.py:815-816, 875-877, 1023-1025`)
- **Difference cases**: when primary model spread differs from ensemble pipeline artifact (or one missing).

### 3) `cage_em` / `ens_cage_edge`
- **`cage_em` origin**: in `TeamProfile` loader from first available among `adj_net_rtg/net_eff/net_rtg/cage_em` aggregated by team (`cbb_ensemble.py:1215-1237, 1290-1304`)
- **`ens_cage_edge` analog**: ensemble result uses `cage_edge = home.cage_em - away.cage_em` (`cbb_ensemble.py:1081`) then flattened as `cage_edge` (`cbb_ensemble.py:246`)
- **Mismatch risk**: if none of expected columns present, defaults to 0.0 and logs warning (`cbb_ensemble.py:1241-1246, 1440-1447`)

### 4) `projected_total` / `pred_total`
- **Origins**:
  - `pred_total` from main model `predicted_total` (`cbb_prediction_model.py:830`; `espn_prediction_runner.py:670`)
  - `projected_total` from `model_total()` using pace/ortg features (`espn_prediction_runner.py:988-1005`, row write at `671`)
- **Inputs required**: `_home_pace/_away_pace/_home_ortg/_away_ortg` in `model_total`; may fallback if context is sparse.
- **Final output names**: both persisted in predictions files and context output.

### 5) `home_momentum_score`
- **Origin in weighted/team context**: `momentum_score` derived as `net_rtg_wtd_qual_l5 - net_rtg_wtd_qual_l10` (`espn_weighted_metrics.py` momentum block)
- **Flow**:
  - runner uses `_latest_team_context(...).get("momentum_score")` -> `home_momentum_score` (`espn_prediction_runner.py:814+ row field at 751-754`)
  - context merge can override/fill from team snapshot mapping candidates (`enrichment/predictions_with_context.py:601-646, 687-704`)
- **Collision handling**: pre/post `_x/_y` coalescing exists but uses default merge suffix behavior initially (`enrichment/predictions_with_context.py:660-737`)

### 6) `home_win_prob` / `mc_home_win_pct`
- **`home_win_prob` origin**: logistic transform of net-eff diff + HCA in runner (`espn_prediction_runner.py:612-618, 714`)
- **`mc_home_win_pct` origin**: Monte Carlo simulation output (`cbb_monte_carlo.py` write section around `801`)
- **Same or separate?** separate computations (deterministic logistic vs simulation).
- **Final appearance**: `home_win_prob` in predictions/context; `mc_home_win_pct` in MC-enriched outputs.

### 7) `spread_line` / `home_spread_current` / `market_spread`
- **Origin**: market ingestion writes `home_spread_current` from ESPN/AN/book feeds (`ingestion/market_lines.py:349-375, 523-529`)
- **Name variants**:
  - predictions runner uses `spread_line` (from schedule odds) (`espn_prediction_runner.py:583, 697`)
  - context merge uses market `home_spread_current`; can rename `spread -> market_spread` at end (`enrichment/predictions_with_context.py:433, 1029-1033`)
- **Final context name**: generally `home_spread_current` + possibly `market_spread`.

### 8) `spread_diff_vs_line`
- **Origin**:
  - initial compute in runner: `predicted_spread - spread_line` (`espn_prediction_runner.py:586-588`)
  - recomputed in context: `_line - _pred` (overwrites interpretation) (`enrichment/predictions_with_context.py:910-914`)
- **Sign convention in context final**: positive means market line more positive than model spread (`line - pred`), i.e., model more home-favored when spreads are away-minus-home.

### 9) `kelly_fraction` / `kelly_units`
- **Origin**: `evaluate_alpha()` -> `kelly_fraction_calc()` (`models/alpha_evaluator.py:45-97, 334-350`)
- **Inputs required**: `pred_spread`, `spread_line`, `model_confidence`; optional trap/revenge/market context (`models/alpha_evaluator.py:99-109`)
- **Null/default handling**:
  - runner and context both default missing `pred_spread/confidence` to 0/0.55 in some paths (`enrichment/predictions_with_context.py:816-840`)
- **Typical range**: non-negative, often small; capped indirectly by quarter-Kelly and multipliers; `kelly_units = fraction*100`.

### 10) `game_tier` / `conference_tier`
- **Origin**: `get_game_tier(home_conf, away_conf)` in `espn_config` (`espn_config.py:get_game_tier`)
- **Assignments**:
  - runner writes `game_tier` (`espn_prediction_runner.py:740-748`)
  - ensemble metadata also writes both `conference_tier` and `game_tier` (`cbb_ensemble.py:968-970`)
- **Logic**: HIGH/MID/LOW exact or cross-tier labels (`espn_config.py` tier functions).

### 11) `adj_net_rtg`
- **Origin**: computed (not directly ESPN field): `adj_net_rtg = adj_ortg - adj_drtg` in SOS stage (`espn_sos.py:345-363`)
- **Downstream expectation**:
  - ensemble loader checks `adj_net_rtg` first (`cbb_ensemble.py:1215, 1219`)
- **Mismatch evidence**: warning path when absent causes `cage_em=0.0` fallback (`cbb_ensemble.py:1241-1246, 1445-1447`)

### 12) `pythagorean_win_pct`
- **Origin**: metrics stage formula `pts^exp / (pts^exp + opp_pts^exp)` (`espn_metrics.py:176`)
- **Availability**: included in team metrics and therefore can flow to `team_game_weighted.csv` when upstream output retained.

### 13) `home_games_used` / `games_before`
- **Origins**:
  - runner sets `home_games_used` by count before cutoff (`espn_prediction_runner.py:709-713`)
  - `games_before` used in ensemble TeamProfile (`cbb_ensemble.py:116, around load mapping block`)
- **Overwrite behavior in context**:
  - context can overwrite `home_games_used` from season totals (`enrichment/predictions_with_context.py:745-775`)
- **Final**: likely season counts (if context source available).

### 14) `clv_vs_open` / `clv_vs_close`
- **Origin**: computed in context block as `open - pred` and `close - pred` (`enrichment/predictions_with_context.py:926-935`)
- **Availability dependency**: requires `home_spread_open/home_spread_current` from market lines.
- **Current null rate**: **cannot be measured in current repo artifact because `data/predictions_with_context.csv` is absent** (checked locally).

### 15) `ats_wins` / `ats_losses` (team level)
- **Origins**:
  - `cbb_season_summaries.py` computes ATS aggregates and merges into team summary (`cbb_season_summaries.py:306-324`)
  - market gap-fill can fetch ATS from ESPN core team ATS endpoint for per-game market rows (`espn_gap_fillers.py:124-135`, `ingestion/market_lines.py:548-551`)
- **Flow to context**: `_enrich_ats_records()` maps from `team_season_summary.csv` to `home_ats_wins/losses`, `away_ats_wins/losses` (`enrichment/predictions_with_context.py:304-341`)

### 16) `home_form_rating`
- **Origin**: weighted metrics compute `form_rating` (`espn_weighted_metrics.py` form block)
- **Flow**:
  - runner writes from latest team context (`espn_prediction_runner.py:749-750`)
  - context merge may refill from candidate columns (`enrichment/predictions_with_context.py:606-609, 687-704`)
- **Current null rate**: ORIGIN file for final context output missing (`predictions_with_context.csv` absent).

### 17) `model_confidence`
- **Origin**: `_calculate_confidence()` in model (`cbb_prediction_model.py:858-908`)
- **Formula**: `0.60*sample_conf + 0.40*consistency_conf - foul_penalty`, clipped to `[0.05,0.95]`.
- **Range**: 0.05–0.95 by explicit clamp.
- **Accuracy correlation**: repository computes/reporting functions exist but no direct hard-coded correlation in main prediction writer; backtester can evaluate conf vs ATS (`cbb_backtester.py:698-706`).

### 18) `event_id` vs `game_id`
- **Assignments**:
  - scoreboard parser uses `game_id` (`espn_parsers.py:368`)
  - summary/team logs use `event_id` (`espn_parsers.py:690+`)
  - runner writes `game_id` then adds alias `event_id` on write (`espn_prediction_runner.py:661, 1113-1118`)
- **Potential mismatch areas**: any join expecting one only; multiple modules contain fallback logic to use either (`ingestion/market_lines.py:407-408`; `enrichment/predictions_with_context.py:402-408`).

### 19) `home_team_id` / `away_team_id`
- **Origin**: ESPN IDs parsed as strings (`espn_parsers.py:343-370`)
- **Type handling**:
  - many modules force `astype(str).str.strip()` before joins (`enrichment/predictions_with_context.py:551-557, 693, 702`)
  - some scripts use `to_numeric(..., errors='coerce')` for join keys (risk) (`build_derived_csvs.py:121-133`)
- **Join-failure risk**: mixed numeric/string coercions can cause silent misses.

### 20) `actual_margin` / `actual_total` (backtester)
- **Origin**: reconstructed from completed game scores in game-log pivot (`cbb_backtester.py:362-364`) and also in grading merge block (`1806-1807`)
- **Join path**:
  - predictions merged with game outcomes by `game_id` (`cbb_backtester.py:1797-1799`)
- **Current match rate**: no single persisted constant; match count depends on `inner/left` and available completed games at run time.

---

## Column Alias Map

| Concept | Names used | Files where each name appears (sample, exhaustive by grep) |
|---|---|---|
| Net efficiency | `adj_net_rtg`, `net_eff`, `net_rtg`, `cage_em`, `adj_em` | `espn_sos.py`, `espn_prediction_runner.py`, `cbb_ensemble.py`, `espn_rankings.py`, `enrichment/predictions_with_context.py`, CSV headers (`team_game_weighted.csv`, `predictions_latest.csv`) |
| Offensive rating | `ortg`, `adj_ortg`, `off_eff`, `cage_o`, `off_rtg` | `espn_metrics.py`, `espn_sos.py`, `cbb_ensemble.py`, `cbb_prediction_model.py`, rankings + weighted CSVs |
| Defensive rating | `drtg`, `adj_drtg`, `def_eff`, `cage_d`, `def_rtg` | `espn_metrics.py`, `espn_sos.py`, `cbb_ensemble.py`, rankings/weighted CSVs |
| Pace / possessions | `pace`, `adj_pace`, `poss`, `possessions`, `cage_t`, `pace_projected` | `espn_metrics.py`, `espn_sos.py`, `cbb_prediction_model.py`, `espn_prediction_runner.py`, `cbb_ensemble.py` |
| Spread (model) | `pred_spread`, `predicted_spread`, `ens_ens_spread`, `ens_spread`, `model_spread` | `espn_prediction_runner.py`, `pipeline_csv_utils.py`, `enrichment/predictions_with_context.py`, `cbb_monte_carlo.py`, `models/clv_analyzer.py` |
| Spread (market) | `spread`, `spread_line`, `home_spread_current`, `home_spread_open`, `market_spread`, `pinnacle_spread`, `draftkings_spread` | `espn_parsers.py`, `espn_prediction_runner.py`, `ingestion/market_lines.py`, `enrichment/predictions_with_context.py` |
| Game identifier | `game_id`, `event_id` (plus `an_game_id` external) | nearly all pipeline files; parser + runner + enrichers + backtester |
| Team identifier | `team_id`, `home_team_id`, `away_team_id` | parser, metrics, context enrichers, market lines, backtester |

**Search method used**: `rg -n` across `*.py` and CSV header scans.

---

## Merge Inventory

> **Legend:** collision risk HIGH when suffixes omitted + overlapping columns plausible, key-type mismatch, or `inner` join.

| File | Function/area | Left | Right | Key | Type | Collision Risk | Cleaned up? |
|---|---|---|---|---|---|---|---|
| `espn_pipeline.py` | `_enrich_team_rows_from_scoreboard` | team rows | scoreboard ctx | `event_id, team_id` | left | Medium (`suffixes` set) | Yes (`_sb` dropped) |
| `espn_sos.py` | `_build_opponent_lookup` | team-game df | opponent lookup | `opponent_id,_sort_dt` | left | Medium | partial drop `_opp_dt` |
| `espn_sos.py` | `_build_allowed_forced_lookup` | team-game df | allow lookup | `opponent_id,_sort_dt` | left | Medium | partial drop `_allow_dt` |
| `enrichment/predictions_with_context.py` | ATS enrich home | df | `home_map` | `home_team_id` | left | **HIGH** (default suffixes) | No explicit suffix handling here |
| `enrichment/predictions_with_context.py` | ATS enrich away | df | `away_map` | `away_team_id` | left | **HIGH** | No explicit |
| `enrichment/predictions_with_context.py` | market merge | predictions | `market_latest` | `event_id` | left | Medium (pre-drop protected cols) | Yes backup/restore model cols |
| `enrichment/predictions_with_context.py` | situational home/away | predictions | home/away sit | composite keys | left | **HIGH** default suffixes | No explicit per-merge, later context cleanup only |
| `enrichment/predictions_with_context.py` | context home/away | predictions | slim team context | `home_team_id` / `away_team_id` | left | **HIGH** defaults; overlapping likely | Yes `_x/_y` coalesce loop |
| `cbb_monte_carlo.py` | combine with MC | `combined_df` | `mc_df` | `game_id` | **inner** | **HIGH** row-drop risk | No |
| `cbb_monte_carlo.py` | enrich existing | df | `mc_df` | `game_id` | left | Medium | No |
| `cbb_backtester.py` | grading merge | predictions | games actuals | `game_id` | left | Medium key format risk | No explicit suffixes |
| `cbb_results_tracker.py` | matched merge | preds | results | `event_id` etc | left | Medium | function-specific cleanup |
| `cbb_season_summaries.py` | multiple summary merges | summary/base | aggregates | `team_id`/group keys | left | Medium | mostly none |
| `build_derived_csvs.py` | matchup enrich home/away | df | home/away tms | composite keys | left | Medium (numeric coercion on IDs) | No |
| others (see grep list) | various | various | various | mostly left | left | Low-Med | varies |

**All `.merge()` call sites discovered** by static scan include: `espn_tournament.py`, `cbb_player_matchup.py`, `espn_sos.py`, `cbb_backtester.py`, `cbb_season_summaries.py`, `espn_injury_proxy.py`, `cbb_monte_carlo.py`, `build_derived_csvs.py`, `espn_pipeline.py`, `espn_tournament_integration.py`, `espn_player_metrics.py`, `cbb_results_tracker.py`, `cbb_travel_fatigue.py`, `enrichment/predictions_with_context.py`, `models/clv_analyzer.py`.

---

## Null Propagation Analysis

### General null-introducer patterns observed
- `pd.to_numeric(..., errors='coerce')` heavily used in runner, context, ensemble loader, SOS, metrics.
- Left joins with missing right-side matches.
- Column alias normalization + renaming where expected source may not exist.
- `try/except` blocks returning fallback default dicts/zeros.
- Explicit default values in `col(... default=0.0)` patterns in `cbb_ensemble.load_team_profiles()`.

### Pred_spread NULL RISK CHAIN (ordered)
1. `cbb_prediction_model.py:_predict_from_stats` — if upstream stats invalid, prediction may fail before row creation (skipped row). 
2. `espn_prediction_runner.py:567-580` — exception in `model.predict_game()` causes game skipped entirely.
3. `espn_prediction_runner.py:669` — writes `pred_spread`; null if prediction dict malformed.
4. `espn_prediction_runner.py:1108-1122` — `normalize_column_names()` + alias recovery may rename unexpectedly.
5. `enrichment/predictions_with_context.py:381-393` — save/restore logic can fail if shape mismatch or column absent.
6. `enrichment/predictions_with_context.py:474-489` — market merge could collide; backup restore mitigates but depends on index alignment.
7. `enrichment/predictions_with_context.py:879-882` — numeric coercion can convert bad strings to NaN.
8. `enrichment/predictions_with_context.py:1021-1029` — fallback to `ens_ens_spread`/`predicted_spread`; if both missing, remains null.

---

## Sign Convention Findings

### Canonical convention in primary model
- **Convention A in core model**: negative spread means home favored (`predicted_spread = -final_edge`). (`cbb_prediction_model.py:815-817`)

### Usage checks
- Runner picks favorite as `home` when `pred_spread < 0` (`espn_prediction_runner.py:625`) ✅
- Score projection formulas assume spread is `away - home` (`enrichment/predictions_with_context.py:945-950`) ✅
- Backtester converts spread to margin with `pred_margin = -spread` (`cbb_backtester.py:636-637`) ✅
- Results tracker uses same `pred_margin = -pred_spread` logic (`cbb_results_tracker.py:354, 389`) ✅

### Potential inconsistency
- `spread_diff_vs_line` sign differs by module:
  - runner: `pred - line` (`espn_prediction_runner.py:586-588`)
  - context: `line - pred` (`enrichment/predictions_with_context.py:912`)
  This is an interpretation mismatch for edge sign (not necessarily model spread sign mismatch).

### Ensemble sub-model consistency
- `ModelPrediction.spread` docstring states negative = home favored (`cbb_ensemble.py:214`)
- Ensemble output follows same convention (`ens_spread` consumed downstream as spread).
- No direct contradictory `spread = +margin` assignment found in ensemble code sections reviewed.

---

## Recommended Fixes (ranked)
1. **`enrichment/predictions_with_context.py:build_predictions_with_context` — spread diff sign inconsistency with runner — standardize one formula (`pred-line` or `line-pred`) and rename column if needed to preserve semantics.**
2. **`cbb_ensemble.py:load_team_profiles` — cage_em silent default-to-zero on alias miss — require at least one net-eff source column and hard-fail/flag instead of defaulting to 0.0 for all teams.**
3. **`enrichment/predictions_with_context.py` merge blocks — add explicit `suffixes=` and deterministic post-merge coalesce for every merge, not just context merge.**
4. **`cbb_monte_carlo.py:832` — `inner` join on `game_id` can silently drop rows — use `left` + drop report to keep auditability.**
5. **`pipeline_csv_utils.py:COLUMN_ALIASES` — expand alias coverage (see below) to reduce ORIGIN UNKNOWN/null coercion paths for spread/efficiency columns.**

---

## column_map.json Update

Below is a **proposed complete alias coverage block** for spread/efficiency/key concepts discovered in this lineage run.

```json
{
  "canonical_aliases": {
    "event_id": ["game_id", "eventId", "gameId", "espn_game_id"],
    "game_id": ["event_id"],

    "team_id": ["espn_team_id"],
    "home_team_id": ["homeTeamId"],
    "away_team_id": ["awayTeamId"],

    "pred_spread": [
      "predicted_spread",
      "prediction_spread",
      "model_spread",
      "ens_ens_spread",
      "ens_spread",
      "ens_ensemble_spread"
    ],
    "pred_total": ["predicted_total", "model_total"],

    "spread_line": ["home_spread_current", "market_spread", "spread"],
    "home_spread_open": ["opening_spread"],
    "home_spread_current": ["closing_spread", "current_spread"],
    "market_spread": ["spread", "home_spread_current"],

    "adj_net_rtg": ["net_eff", "net_rtg", "cage_em", "adj_em"],
    "adj_ortg": ["ortg", "off_eff", "off_rtg", "cage_o"],
    "adj_drtg": ["drtg", "def_eff", "def_rtg", "cage_d"],
    "adj_pace": ["pace", "poss", "possessions", "cage_t", "pace_factor"],

    "home_win_prob": ["mc_home_win_pct", "win_prob_home"],
    "away_win_prob": ["mc_away_win_pct", "win_prob_away"],

    "clv_vs_open": ["clv_open"],
    "clv_vs_close": ["clv_close"],

    "ats_wins": ["home_ats_wins", "away_ats_wins"],
    "ats_losses": ["home_ats_losses", "away_ats_losses"],

    "model_confidence": ["confidence", "ens_confidence"],
    "projected_total": ["total_projected"],
    "games_before": ["games_used", "home_games_used", "away_games_used", "games_played", "game_number", "n_games"]
  }
}
```

---

## Search Notes / Evidence Gaps
- **ORIGIN UNKNOWN: `ens_ens_spread` producer assignment** in Python code. It is consumed widely but no direct creation assignment was found in scanned `.py`; likely produced by an external step/workflow artifact not present in repository scripts.
- **No direct Supabase table write path found in Python code** during this run (`rg -n "supabase|upsert|insert" --glob "*.py"`), so final resting place in DB cannot be traced from repo code alone.
- **`predictions_with_context.csv` not present in current repo data/**, so requested current null-rate checks for context-only columns (`clv_vs_open`, `clv_vs_close`, `home_form_rating`) cannot be empirically measured from artifact.
