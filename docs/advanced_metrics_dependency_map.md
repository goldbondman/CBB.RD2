# Advanced Metrics Dependency Map

Generated from repository source audit on 2026-03-05.

## 1) Producer Script -> CSV Output Map

| Script / Module | Key Function(s) | Output CSV(s) | Upstream Inputs |
|---|---|---|---|
| `espn_pipeline.py` | `build_games`, `build_team_and_player_logs`, `compute_all_metrics`, `compute_sos_metrics`, `compute_weighted_metrics`, `compute_tournament_metrics`, `build_pretournament_snapshot` | `data/games.csv`, `data/team_game_logs.csv`, `data/player_game_logs.csv`, `data/team_game_metrics.csv`, `data/team_game_sos.csv`, `data/team_game_weighted.csv`, `data/player_game_metrics.csv`, `data/team_tournament_metrics.csv`, `data/team_pretournament_snapshot.csv` | ESPN scoreboard/summary APIs + existing pipeline tables |
| `pipeline/advanced_metrics/compute_runner.py` | `compute_features` | `data/team_game_metrics.csv`, `data/matchup_metrics.csv`, `data/feature_runs/feature_run_manifest.json` | `data/team_game_logs.csv`, `data/player_game_logs.csv` |
| `backtesting/compute_metrics.py` | `run` (calls merge/compute/pivot steps) | `data/team_game_metrics_advanced.csv`, `data/matchup_metrics.csv` | `data/team_game_metrics.csv`, `data/team_game_weighted.csv` |
| `scripts/build_team_splits.py` | `build_team_splits` | `team_splits.csv` | `data/team_game_metrics.csv` |
| `cbb_rotation_features.py` | `build_rotation_features` | `data/rotation_features.csv` | `data/player_game_logs.csv`, `data/player_game_metrics.csv` |
| `player_availability_features.py` | `build_player_availability_features` | `data/player_availability_features.csv` | `data/player_game_logs.csv`, optional `data/team_injury_impact.csv` |
| `cbb_situational_features.py` | `main` (team-game situational builder) | `data/situational_features.csv` | `data/games.csv`, `data/cbb_rankings.csv` (or fallback `data/team_game_weighted.csv`) |
| `cbb_travel_fatigue.py` | `compute_travel_fatigue` | `data/team_travel_fatigue.csv` | `data/games.csv`, `data/team_game_logs.csv`, `data/venue_geocodes.csv` |
| `build_training_data.py` | `main` | `data/backtest_training_data.csv` | `team_game_weighted`, `rotation_features`, `player_availability_features`, `situational_features`, market/ATS/luck/snapshot tables |

## 2) Metric Builder -> CSV Dependency Map

| Metric Builder | Metrics Family It Produces | Reads | Writes | Where Run |
|---|---|---|---|---|
| `pipeline/advanced_metrics/team_metric_compute.py::compute_team_game_metrics` (via registry + shared derivations) | `ODI`, `PEQ`, `POSW`, `SVI`, `PXP`, `DPC`, `VOL`, `TC`, `WL` (+ rollups and factor ODI columns) | `team_game_logs.csv`, `player_game_logs.csv` | `team_game_metrics.csv` | Explicitly via `pipeline/advanced_metrics/compute_runner.py` and smoke path |
| `pipeline/advanced_metrics/matchup_metric_compute.py::compute_matchup_metrics` | `PEI_matchup`, matchup `POSW`, `ODI_diff/sum`, factor ODI diff/sum | `team_game_metrics.csv` | `matchup_metrics.csv` | same as above |
| `backtesting/compute_metrics.py` | Alternate/backtest versions of `SVI`, `PEQ`, `WL`, `DPC`, `PXP`, `ODI`, factor ODI and rest flags | `team_game_metrics.csv`, `team_game_weighted.csv` | `team_game_metrics_advanced.csv`, `matchup_metrics.csv` | `run_backtest.py` step 1 |
| `cbb_advanced_metrics_codex.py::AdvancedMetricsCodex` | Requested formulas for `ODI*`, `PEI`, `POSW`, `SVI`, `PXP`, `LNS`, `USEF`, `DPC`, `FII`, `SME`, `SCH`, `VOL`, `TC`, `WL`, `RFD`, `GSR`, `ALT` | In-memory `GameInputs` (caller must assemble from CSVs) | In-memory prediction dict | Not wired in production workflows today |

## 3) Requested Metric -> Input CSV Dependency Matrix

| Requested Metric | Current Builder(s) | Required Inputs (Current Code) | CSV(s) Supplying Inputs Today |
|---|---|---|---|
| `ODI*` | Legacy: `f_ODI`; Backtest ODI proxies; Codex exact `compute_odi_star` | Four-factor edges, opponent mirrors, optional league averages | Legacy: `team_game_logs.csv` (derived in team metrics), `team_game_weighted.csv` (vs-opp proxies); Codex caller may use `team_splits.csv` + weighted |
| `PEI` | Legacy: `PEI_matchup` from `PEQ_A-PEQ_B`; Codex exact `compute_pei` | Legacy PEQs or ORB/TOV offense-defense splits | `matchup_metrics.csv` (legacy PEI proxy), `team_splits.csv`/weighted for Codex assembly |
| `POSW` | Legacy: `f_POSW`/`f_POSW_matchup`; Codex exact `compute_posw` | Legacy ORB/DRB/TOV shares or PEI+pace | `team_game_logs.csv` -> `team_game_metrics.csv` (legacy); `team_splits.csv` + weighted pace for Codex |
| `SVI` | Legacy schedule-adjusted win index; Codex shot-value index | Legacy: WL + opponent strength; Codex: eFG, 3PA rate, FT_pts/FGA + z-context | Legacy from `team_game_logs.csv` + pregame baselines; Codex from `team_game_metrics.csv` / `team_splits.csv` + external z-context |
| `PXP` | Legacy `NetRtg-preNetRtg` and backtest `win-pyth`; Codex ICS/BSI/clutch | NetRtg baselines or ICS/BSI/clutch components | Legacy from `team_game_metrics.csv`; Codex needs rotation/team continuity columns not fully materialized |
| `LNS` | Codex only | per-player `minutes_share`, `TOIS` | No canonical CSV with both fields; closest is `rotation_features.csv` but aggregated |
| `USEF` | Codex only | star/team TS, star usage, top-3 usage player TS | Partial in `player_game_metrics.csv`; no existing assembler into model inputs |
| `DPC` | Legacy `f_DPC` + backtest `DRTG mean - drtg`; Codex depth cushion formula | Starter/bench stats or rotation net-rating spread + BSI | Legacy from `player_game_logs.csv` via starter-bench helper; Codex needs extra rotation spread fields not standardized |
| `FII` | Codex only | DPC, backup quality by position, `ffc_inverted` | Not present as canonical columns in current CSV outputs |
| `SME` | Codex only | star shot profile vector + opponent weakness vector | Partial raw ingredients in `player_game_metrics.csv` and `team_game_weighted.csv`; no canonical SME table |
| `SCH` | Codex only | pace, 3PA rate, FTr, size | pace/FTr/3PA from team metrics/splits; `size` (`lineup_height_avg`) not in canonical outputs |
| `VOL` | Legacy `abs(NetRtg-pre_NetRtg)`; Codex `std(L8)` | baseline delta or L8 net rating series | Legacy from `team_game_metrics.csv`; Codex requires reconstructed L8 series from `team_game_weighted.csv` |
| `TC` | Legacy `poss-opp_pre_poss`; Codex `-z(mean abs pace dev L12)` | possessions baseline or L12 pace deviations + z-context | Legacy from `team_game_logs.csv` derivations; Codex requires pace series + z-context artifact |
| `WL` (Whistle Leverage) | Legacy uses WL=Win/Loss; Codex has whistle formula | FT PPP diff + FTr diff + z-context | Partial FT/FTr sources in team metrics; no production whistle-leverage feature output |
| `RFD` | Situational/travel proxies + Codex exact | home/away rest + b2b penalties | `situational_features.csv` (`home_rest_days`, `away_rest_days`, `rest_delta`), `team_travel_fatigue.csv` (`is_back_to_back`) |
| `GSR` | Codex only | tournament stage, elimination risk, spread magnitude | Spread from `games.csv`/`market_lines.csv`; no canonical tournament-stage + elimination-risk columns |
| `ALT` | Travel module proxy + Codex exact | elevation home/away + cross-country miles | `team_travel_fatigue.csv` has miles proxy; `venue_geocodes.csv` has lat/lon but no elevation |

## 4) Workflow Step Mapping (How Inputs Are Generated)

| Workflow Step | Script | Key CSVs Produced/Updated | Metrics Impacted |
|---|---|---|---|
| ESPN ingest + feature chain | `espn_pipeline.py` | `games`, `team_game_logs`, `player_game_logs`, `team_game_metrics`, `team_game_weighted`, `player_game_metrics`, tournament snapshot tables | Supplies base inputs for legacy `ODI/PEQ/POSW/SVI/PXP/DPC/VOL/TC/WL` and many Codex raw ingredients |
| Travel fatigue step | `cbb_travel_fatigue.py` | `team_travel_fatigue.csv` | RFD/ALT proxy inputs |
| Situational step | `cbb_situational_features.py` | `situational_features.csv` | Rest/pressure proxy inputs (RFD/GSR components) |
| Rotation step | `cbb_rotation_features.py` | `rotation_features.csv` | Depth/lineup proxy inputs (LNS/DPC/FII candidates) |
| Availability step | `player_availability_features.py` | `player_availability_features.csv` | Injury/depth proxy inputs (FII candidates) |
| Team split step | `scripts/build_team_splits.py` | `team_splits.csv` | Home/away split inputs for Codex ODI*/PEI/POSW/SVI-style split usage |
| Backtest metric generation | `backtesting/compute_metrics.py` | `team_game_metrics_advanced.csv`, `matchup_metrics.csv` | Alternate (non-spec) definitions used in backtesting |

## 5) Practical Dependency Notes
- Two different metric stacks coexist:
  - Registry team/matchup metrics (`pipeline/advanced_metrics`) with one set of formulas.
  - Backtesting metrics (`backtesting/compute_metrics.py`) with overlapping acronyms but different formulas.
- `cbb_advanced_metrics_codex.py` contains the requested formulas but is currently an isolated module; no workflow calls it.
- `team_travel_fatigue.csv` is conditional in workflow (`cbb_predictions_rolling.yml`) and may be absent when source files are missing.
- Elevation is currently not in canonical venue/team tables, so `ALT` cannot be fully populated without new data enrichment.
