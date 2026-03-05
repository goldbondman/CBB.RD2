# Advanced Metrics Coverage Report

Generated from repository source audit on 2026-03-05.

## Scope + Status Rules
- `EXISTS`: implemented and used in active pipeline outputs.
- `PARTIAL`: implemented but formula differs from requested spec, and/or only exists in standalone/non-integrated code.
- `MISSING`: no implementation found.

Primary code paths audited:
- Legacy feature engine: `pipeline/advanced_metrics/*`
- Backtest metric engine: `backtesting/compute_metrics.py`
- Standalone Codex module: `cbb_advanced_metrics_codex.py` (not wired into `espn_prediction_runner.py` / production prediction flow)

---

## 1) ODI*
**Status:** `PARTIAL`

**Where computed**
- `pipeline/advanced_metrics/metric_library.py::f_ODI`
- `backtesting/compute_metrics.py::compute_per_game_bases` (`odi_g`) and `compute_four_factor_odi` (factor-specific ODI columns)
- `cbb_advanced_metrics_codex.py::compute_odi_star`

**Current formula (today)**
- Legacy engine: `ODI = 0.40*(eFG-opp_eFG) + 0.25*(opp_TOV%-TOV%) + 0.20*(ORB%-opp_ORB%) + 0.15*(FTr-opp_FTr)`
- Backtesting base `ODI`: `odi_g = perf_vs_exp_net` (not four-factor weighted).
- Codex module: league-centered four-factor ODI* with 0.40/0.25/0.20/0.15 weights.

**Required inputs + CSV providers**
- Legacy: `eFG`, `TOV%`, `ORB%`, `FTr` + opponent mirrors; derived from `data/team_game_logs.csv` via `shared_derivations`.
- Backtesting: `perf_vs_exp_net`, `efg_vs_opp_*`, `tov_vs_opp_*`, `orb_vs_opp_*`, `ftr_vs_opp_*` from `data/team_game_weighted.csv`.
- Codex: `efg/tov/orb/ftr` offense + opponent defensive counterparts + league averages (passed in `GameInputs` dictionaries).

**Mismatch vs requested spec**
- Legacy ODI is opponent-edge based, not explicitly league-centered.
- Backtesting `ODI` definition diverges strongly (`perf_vs_exp_net` proxy).
- Exact requested ODI* exists only in standalone Codex module (not integrated in production runner).

---

## 2) PEI
**Status:** `PARTIAL`

**Where computed**
- `pipeline/advanced_metrics/metric_library.py::f_PEI_matchup` (`PEI_matchup`)
- `cbb_advanced_metrics_codex.py::compute_pei`

**Current formula (today)**
- Legacy matchup: `PEI_matchup = PEQ_A - PEQ_B`
- Legacy team PEQ: `PEQ = OffEff * (1 - TOV%)`
- Codex module: `PEI = (ORB_off_adj - ORB_def_adj) + (TOV_def_adj - TOV_off_adj)`

**Required inputs + CSV providers**
- Legacy: `PEQ_A`, `PEQ_B` from `data/matchup_metrics.csv` (built from team metrics).
- Codex: offensive/defensive ORB/TOV adjustments from team stat dictionaries (typically assembled from `team_splits.csv` and weighted/team snapshots).

**Mismatch vs requested spec**
- Legacy PEI is PEQ differential proxy, not the requested ORB/TOV equity equation.
- Exact PEI exists only in standalone Codex module.

---

## 3) POSW
**Status:** `PARTIAL`

**Where computed**
- `pipeline/advanced_metrics/metric_library.py::f_POSW` and `f_POSW_matchup`
- `backtesting/compute_metrics.py` (rolling POSW columns from computed metrics)
- `cbb_advanced_metrics_codex.py::compute_posw`

**Current formula (today)**
- Legacy team POSW: `(ORB% + DRB% + (1 - TOV%)) / 3`
- Legacy matchup POSW: `POSW_A - POSW_B`
- Codex module: `POSW = PEI * projected_pace / 100`

**Required inputs + CSV providers**
- Legacy: `ORB%`, `DRB%`, `TOV%` from team-game boxscore derivations (`data/team_game_logs.csv` -> `team_game_metrics.csv`).
- Codex: `PEI` + `pace_home/pace_away` (from split/team stats inputs, e.g., `team_splits.csv` or weighted stats).

**Mismatch vs requested spec**
- Legacy POSW is possession-quality blend, not PEI*pace.
- Exact formula exists only in standalone Codex module.

---

## 4) SVI
**Status:** `PARTIAL`

**Where computed**
- `pipeline/advanced_metrics/metric_library.py::f_SVI`
- `backtesting/compute_metrics.py::compute_per_game_bases` (`svi_g`)
- `cbb_advanced_metrics_codex.py::compute_svi`

**Current formula (today)**
- Legacy/backtesting SVI: schedule-adjusted win signal `WL * (1 + opp_strength/30)`.
- Codex module SVI: `z(eFG) + 0.5*z(3PA_rate) + 0.5*z(FT_pts/FGA)`.

**Required inputs + CSV providers**
- Legacy: `WL`, `opp_pre_NetRtg_season` from team game rows and pregame baselines.
- Codex: `eFG`, `3PA_rate`, `FTM/FGA` + z-score context (not currently produced as standardized league z-context artifact).

**Mismatch vs requested spec**
- Name collision: legacy SVI = Schedule-adjusted Victory Index; requested SVI = Shot Value Index.
- Requested formula exists only in standalone Codex module.

---

## 5) PXP
**Status:** `PARTIAL`

**Where computed**
- `pipeline/advanced_metrics/metric_library.py::f_PXP`
- `backtesting/compute_metrics.py::compute_per_game_bases` (`pxp_g`)
- `cbb_advanced_metrics_codex.py::compute_pxp`

**Current formula (today)**
- Legacy team engine: `PXP = NetRtg - pre_NetRtg_season`
- Backtesting: `PXP = win - pythagorean_win_pct`
- Codex: `PXP = 0.4*ICS + 0.3*BSI + 0.3*clutch_pct`

**Required inputs + CSV providers**
- Legacy: `NetRtg`, `pre_NetRtg_season` from team-game metrics.
- Backtesting: `win`, `pythagorean_win_pct` from `team_game_metrics.csv`.
- Codex: `returning_minutes`, `rotation_stability`, `star_continuity`, bench components, `clutch_pct` (not emitted as one canonical CSV today).

**Mismatch vs requested spec**
- Two legacy definitions already differ from each other and from requested formula.
- Requested PXP exists only in standalone Codex module.

---

## 6) LNS
**Status:** `PARTIAL`

**Where computed**
- `cbb_advanced_metrics_codex.py::compute_lns`

**Current formula (today)**
- `LNS = sum(minutes_share_i * TOIS_i)`

**Required inputs + CSV providers**
- Needs per-rotation `minutes_share`, `tois` for current game.
- Closest provider: `data/rotation_features.csv` has rotation summaries, but not explicit per-player `minutes_share` + `TOIS` rows.

**Mismatch vs requested spec**
- Formula matches spec in standalone module.
- Not integrated into production prediction/backtest pipelines.

---

## 7) USEF
**Status:** `PARTIAL`

**Where computed**
- `cbb_advanced_metrics_codex.py::compute_usef`

**Current formula (today)**
- `SIE = (star_TS - team_TS) * star_usage`
- `USEF = SIE + weighted_TS_top_usage_cluster` (top-3 usage weighted TS)

**Required inputs + CSV providers**
- Needs player-level usage + TS for top-usage cluster.
- Partial provider: `data/player_game_metrics.csv` includes `usage_rate`, `ts_pct`.
- No active assembler currently feeds these into production model as `USEF`.

**Mismatch vs requested spec**
- Formula intent matches spec.
- Not integrated into production prediction/backtest output.

---

## 8) DPC
**Status:** `PARTIAL`

**Where computed**
- `pipeline/advanced_metrics/metric_library.py::f_DPC`
- `backtesting/compute_metrics.py::compute_per_game_bases` (`dpc_g`)
- `cbb_advanced_metrics_codex.py::compute_dpc`

**Current formula (today)**
- Legacy team engine: `DPC = 0.4*bench_minutes_share + 0.3*REB_rate_bench + 0.3*TS_bench`
- Backtesting: `DPC = league_mean_drtg - drtg`
- Codex: `DPC = BSI + (best_rotation_netrtg - worst_rotation_netrtg)`

**Required inputs + CSV providers**
- Legacy: starter/bench helper outputs from `data/player_game_logs.csv`.
- Backtesting: `drtg` from `data/team_game_metrics.csv`.
- Codex: `bench_minutes`, `bench_ts_rel`, `bench_reb_rel`, rotation net-rating spread (not currently materialized in one canonical CSV).

**Mismatch vs requested spec**
- Legacy definitions differ from requested depth-cushion formula.
- Exact formula only in standalone Codex module.

---

## 9) FII
**Status:** `PARTIAL`

**Where computed**
- `cbb_advanced_metrics_codex.py::compute_fii`

**Current formula (today)**
- `FII = DPC * backup_quality_key_positions + FFC_inverted`

**Required inputs + CSV providers**
- `DPC`, `backup_quality_key_positions`, `ffc_inverted`.
- No production CSV currently emits `backup_quality_key_positions` or `ffc_inverted` as standardized columns.

**Mismatch vs requested spec**
- Formula matches spec in standalone module.
- Not integrated into production flow; upstream columns mostly absent.

---

## 10) SME (Star Matchup Exploit)
**Status:** `PARTIAL`

**Where computed**
- `cbb_advanced_metrics_codex.py::compute_sme`

**Current formula (today)**
- Dot product of star shot profile vector and opponent weakness profile vector.

**Required inputs + CSV providers**
- Star profile: `star_three_pa_rate`, `star_rim_rate`, `star_midrange_rate`, `star_post_ups`.
- Opp weakness: `opp3p_allowed`, `opp_rim_fg_allowed`, `opp_mid_fg_allowed`.
- Partial raw sources exist (`player_game_metrics.csv`, `team_game_weighted.csv`) but no canonical SME-ready table exists.

**Mismatch vs requested spec**
- Formula matches spec in standalone module.
- Not integrated into production/batch pipelines.

---

## 11) SCH
**Status:** `PARTIAL`

**Where computed**
- `cbb_advanced_metrics_codex.py::compute_sch`

**Current formula (today)**
- `|paceA-paceB| + |3PA_rateA-3PA_rateB| + |FTrA-FTrB| + |sizeA-sizeB|`

**Required inputs + CSV providers**
- Pace/FTr from `team_game_weighted.csv` or `team_splits.csv`.
- 3PA rate from `team_game_metrics.csv` / split stats.
- Size (`lineup_height_avg`) is not present in canonical team CSVs.

**Mismatch vs requested spec**
- Formula matches spec in standalone module.
- Missing standardized `size` input in current pipeline artifacts.

---

## 12) VOL
**Status:** `PARTIAL`

**Where computed**
- `pipeline/advanced_metrics/metric_library.py::f_VOL`
- `backtesting/compute_metrics.py` (VOL rollups)
- `cbb_advanced_metrics_codex.py::compute_vol`

**Current formula (today)**
- Legacy: `VOL = abs(NetRtg - pre_NetRtg_season)`
- Codex: `VOL = std(netrtg_adj over last 8)`

**Required inputs + CSV providers**
- Legacy: `NetRtg`, `pre_NetRtg_season` from `team_game_metrics.csv`.
- Codex: `netrtg_adj_l8` sequence (not materialized as a direct list column in canonical CSVs; can be reconstructed from `adj_net_rtg` history in `team_game_weighted.csv`).

**Mismatch vs requested spec**
- Legacy VOL is deviation-from-baseline, not L8 standard deviation.
- Exact spec only in standalone Codex module.

---

## 13) TC
**Status:** `PARTIAL`

**Where computed**
- `pipeline/advanced_metrics/metric_library.py::f_TC`
- `backtesting/compute_metrics.py` (TC rollups)
- `cbb_advanced_metrics_codex.py::compute_tc`

**Current formula (today)**
- Legacy: `TC = poss - opp_pre_poss_season`
- Codex: `TC = -z(mean(|game_pace - team_avg_pace|), last 12)`

**Required inputs + CSV providers**
- Legacy: possessions + opponent pregame possession baseline from `team_game_logs.csv` derivations.
- Codex: game pace series (L12), team season pace average, z-context stats.

**Mismatch vs requested spec**
- Legacy TC definition differs substantially from requested tempo-control stability z-score.

---

## 14) WL (Whistle Leverage)
**Status:** `PARTIAL`

**Where computed**
- Legacy `WL`: `pipeline/advanced_metrics/metric_library.py::f_WL`, `backtesting/compute_metrics.py`
- Requested WL implementation: `cbb_advanced_metrics_codex.py::compute_wl`

**Current formula (today)**
- Legacy WL: binary win indicator (`points_for > points_against`).
- Codex WL: `z(FT_PPP - opp_FT_PPP) + 0.5*z(FTr - opp_FTr)`.

**Required inputs + CSV providers**
- Legacy: `points_for`, `points_against` from `team_game_logs.csv`.
- Codex: `ft_ppp`, `opp_ft_ppp`, `FTr`, `opp_FTr`, z-context.

**Mismatch vs requested spec**
- Major naming collision: production WL is Win-Loss, not Whistle Leverage.
- Requested WL exists only in standalone Codex module.

---

## 15) RFD (Rest & Fatigue Differential)
**Status:** `PARTIAL`

**Where computed**
- Proxy fields: `cbb_situational_features.py` (`rest_delta`, rest flags)
- Proxy fields: `cbb_travel_fatigue.py` (`rest_days`, `is_back_to_back`)
- Exact function: `cbb_advanced_metrics_codex.py::compute_rfd`

**Current formula (today)**
- Codex exact: `days_rest_home - days_rest_away + 0.5*(back2back_away - back2back_home)`.
- Pipeline proxies: rest deltas/flags computed separately; no unified RFD column.

**Required inputs + CSV providers**
- `data/situational_features.csv`: `home_rest_days`, `away_rest_days`, `rest_delta`.
- `data/team_travel_fatigue.csv`: `rest_days`, `is_back_to_back` (file may be absent depending on run).

**Mismatch vs requested spec**
- Exact formula exists but is not currently emitted as a production metric column.

---

## 16) GSR (Game Script Risk)
**Status:** `PARTIAL`

**Where computed**
- `cbb_advanced_metrics_codex.py::compute_gsr`

**Current formula (today)**
- `0.4*tournament_stage + 0.3*elimination_risk + 0.3*spread_magnitude`

**Required inputs + CSV providers**
- No direct canonical columns for all three inputs.
- Closest proxies:
  - `tournament_stage`: could be derived from schedule context/workflow game-type metadata.
  - `elimination_risk`: partial proxies in `situational_features.csv` (`must_win_flag`, `bubble_pressure_flag`).
  - `spread_magnitude`: market spread from `games.csv` / `market_lines.csv`.

**Mismatch vs requested spec**
- Formula exists in standalone module only and is not invoked in prediction pipeline.

---

## 17) ALT (Altitude & Travel)
**Status:** `PARTIAL`

**Where computed**
- `cbb_advanced_metrics_codex.py::compute_alt`
- Related travel pieces: `cbb_travel_fatigue.py`

**Current formula (today)**
- Codex exact: `ALT = elevation_home - elevation_away + cross_country_miles/1000`.
- Existing travel module computes `estimated_travel_miles`, not altitude differential.

**Required inputs + CSV providers**
- `cross_country_miles`: can be proxied from `data/team_travel_fatigue.csv` (`estimated_travel_miles`).
- Elevation fields are not present in canonical outputs (`venue_geocodes.csv` has `lat/lon` only).

**Mismatch vs requested spec**
- Exact formula exists in standalone module, but elevation inputs are missing in current pipeline data.

---

## Summary Findings
- In active pipeline/backtest code, most requested acronyms exist but many are **different metrics under same names** (notably `SVI`, `WL`, `PXP`, `DPC`, `TC`, `VOL`, `POSW`, `PEI`).
- A full requested-formula implementation exists in `cbb_advanced_metrics_codex.py`, but it is currently **standalone/unwired**.
- No requested metric is fully production-integrated end-to-end yet with guaranteed upstream inputs and output columns.
