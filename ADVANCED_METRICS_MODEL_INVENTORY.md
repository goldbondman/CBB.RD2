# Advanced Metrics + Model Inventory (Repo-Wide)

_Last updated from repository state on this branch._

## 1) Model collections currently in repo

### Ensemble model family (8-model design)
Source files:
- `cbb_ensemble.py`
- `cbb_config.py`

Models combined in the ensemble:
1. **FourFactors**
2. **AdjEfficiency**
3. **Pythagorean**
4. **Situational**
5. **CAGERankings**
6. **LuckRegression**
7. **Variance**
8. **HomeAwayForm**

Primary feature group used by the ensemble profile object (`TeamProfile`) includes:
- Adjusted efficiency/core rating fields (`cage_em`, `cage_o`, `cage_d`, `cage_t`, `barthag`)
- Four factors + opponent four factors (`efg_pct`, `tov_pct`, `orb_pct`, `drb_pct`, `ftr`, `opp_efg_pct`, `opp_tov_pct`, `opp_ftr`)
- Rolling form (`net_rtg_l5/l10`, `ortg_l5/l10`, `drtg_l5/l10`, `pace_l5/l10`, `efg_l5/l10`, `tov_l5/l10`)
- Volatility/consistency (`net_rtg_std_l10`, `efg_std_l10`, `consistency_score`)
- CAGE composite signals (`suffocation`, `momentum`, `clutch_rating`, `dna_score`, `resume_score`, `cage_power_index`)
- Luck/record context (`luck`, `pythagorean_win_pct`, `home_wpct`, `away_wpct`, `close_wpct`, `win_streak`, `sos`)
- ATS context (`cover_rate_season`, `cover_rate_l10`, `ats_margin_l10`, `cover_margin`, `cover_streak`)

### Standalone primary model (recursive/bidirectional model)
Source file:
- `cbb_prediction_model.py`

Core modeled components:
- Dean Oliver four factors (including `efg`, `tov_pct`, `orb_pct`, `drb_pct`, `ftr`, `ft_pct`)
- Recursive opponent-history normalization
- Tournament multipliers for totals (`regular`, `conf_tournament`, `ncaa_r1`, `ncaa_r2`)
- Foul-rate variance penalty (`pf` integrated into confidence)

---

## 2) Advanced metric families added in pipeline (and where they live)

### A) Per-game + rolling advanced team metrics
Files:
- Computation: `espn_metrics.py`
- Output: `data/team_game_metrics.csv`

Metric families:
- Shooting efficiency: `efg_pct`, `ts_pct`, `fg_pct`, `three_pct`, `ft_pct`
- Shot profile: `three_par`, `ftr`
- Possession + efficiency: `poss`, `ortg`, `drtg`, `net_rtg`, `pace`, `tov_pct`
- Rebounding share: `orb_pct`, `drb_pct`
- Half-split diagnostics: `h1_margin`, `h2_margin`, `h1_ortg`, `h2_ortg`, `h1_drtg`, `h2_drtg`
- Outcome/market diagnostics: `cover`, `cover_margin`, `ats_push`, `dead_spread_flag`, `close_game_flag`, `blowout_flag`
- Luck + pythagorean: `pythagorean_win_pct`, `luck_score`
- Leak-free rolling windows (L5/L10) for the above plus variance/streak/split fields (`efg_std_l10`, `three_pct_std_l10`, `win_streak`, `cover_streak`, `ha_net_rtg_l10`, etc.)

### B) Opponent-context + weighted rolling metrics
Files:
- Computation: `espn_sos.py`, `espn_weighted_metrics.py`
- Output: `data/team_game_weighted.csv`, `data/team_game_sos.csv`

Metric families:
- Opponent baselines (`opp_avg_ortg_*`, `opp_avg_drtg_*`, `opp_avg_net_rtg_*`, `opp_avg_efg_*`, `opp_avg_pace_*`)
- Relative performance vs opponent baseline (`efg_vs_opp_*`, `orb_vs_opp_*`, `drb_vs_opp_*`, `tov_vs_opp_*`, `ftr_vs_opp_*`)
- Adjusted efficiency and pace (`adj_ortg`, `adj_drtg`, `adj_net_rtg`, `adj_pace`)
- Performance-vs-expectation (`perf_vs_exp_ortg`, `perf_vs_exp_drtg`, `perf_vs_exp_def`, `perf_vs_exp_net`)
- Opponent-quality weighted rolling signals (`*_wtd_off_l5/l10`, `*_wtd_def_l5/l10`, `*_wtd_qual_l5/l10`)

### C) Tournament/CAGE composite metrics
Files:
- Computation: `espn_tournament.py`, `espn_tournament_integration.py`
- Output: `data/team_tournament_metrics.csv`, `data/team_pretournament_snapshot.csv`

Key metric groups:
- CAGE base ratings (`cage_o`, `cage_d`, `cage_t`, `cage_em`, `barthag`)
- Composite signals (`suffocation`, `momentum`, `clutch_rating`, `floor_em`, `ceiling_em`, `dna_score`, `resume_score`, `cage_power_index`)

### D) Situational/travel/rotation/player availability/injury/luck/market feature sets
Files:
- Situational: `cbb_situational_features.py` → `data/situational_features.csv`
- Travel fatigue: `cbb_travel_fatigue.py`
- Rotation: `cbb_rotation_features.py` → `data/rotation_features.csv`
- Player availability: `player_availability_features.py` → `data/player_availability_features.csv`
- Injury proxy: `espn_injury_proxy.py` → `data/player_injury_proxy.csv`, `data/team_injury_impact.csv`
- Luck regression: `cbb_luck_regression_features.py` → `data/luck_regression_features.csv`
- Line movement: `cbb_line_movement_features.py` → `data/line_movement_features.csv`

Representative advanced fields added by these modules:
- **Situational**: lookahead/letdown/revenge/bubble pressure/must-win/fatigue/rest extensions/rivalry/neutral indicators
- **Rotation**: `rot_size`, usage concentration (`top2_pused_share`), rotation shooting/TO stability, closer FT quality
- **Availability/Injury**: `minutes_available_pct`, `star_availability_score`, `injury_impact_delta`, `lineup_continuity_l3`, `new_starter_flag`, injury burden/severity fields
- **Luck regression**: pyth-vs-actual deltas, three-point luck, close-game luck, composite luck, regression candidate flags
- **Line movement**: open/close/CLV-style market context fields used in backtest and feature snapshots

### E) Backtest training matrix / model-ready combined feature table
File:
- `data/backtest_training_data.csv`

This file is the combined modeling table and includes:
- Team form deltas (`net_rtg_delta_l5/l10`, `adj_*_delta`, `pace_delta_l5`)
- Rotation/player/injury deltas (`rot_efg_delta`, `rot_to_swing_diff`, `star_availability_delta`, etc.)
- ATS/team trend features (`home_ats_*`, `away_ats_*`)
- Situational flags and line context (`opening_spread`, `closing_spread`, `clv_delta`, revenge/lookahead flags)
- Labels (`actual_margin`, `home_covered_ats`, `actual_total`, `covered_over`)

---

## 3) Backtested performance metrics found in repo

## Source files used
- `data/backtest_model_report_latest.csv`
- `data/backtest_results_20260226.csv`
- `data/csv/backtest_summary.csv`
- `data/results_summary.csv`
- `data/results_model_split.csv`

## 3.1 Submodel + ensemble backtest report (`backtest_model_report_latest.csv`)
Pipeline run id in file: **22441096216**

| model_name | n_games | ATS % | Edge ATS % | Edge ROI (sim units) | Spread MAE | Win % |
|---|---:|---:|---:|---:|---:|---:|
| FourFactors | 3297 | 47.7 | 45.6 | -13.0 | 11.40 | 52.7 |
| AdjEfficiency | 3297 | 45.5 | 46.8 | -10.6 | 17.61 | 58.3 |
| Pythagorean | 3297 | 52.3 | 50.7 | -3.2 | 16.38 | 58.7 |
| Situational | 3297 | 47.7 | 47.9 | -8.5 | 15.18 | 59.1 |
| CAGERankings | 3297 | 45.5 | 49.3 | -5.9 | 14.34 | 58.3 |
| Ensemble | 3297 | 45.5 | 51.4 | -1.9 | 13.69 | 59.2 |

> Note: This report currently contains 6 rows (5 submodels + ensemble), while the current codebase defines 8 ensemble model slots. Treat this as a historical snapshot from that run configuration.

## 3.2 Backtest summary snapshot (`data/csv/backtest_summary.csv`)
Generated at: **2026-02-28T08:36:28Z**

- `season`: 8 ATS-graded games, 5 wins, ATS 62.5%, ATS ROI +1.55 units.
- `l60/l30/l14/l7`: 0 graded games in this file snapshot (insufficient sample in that artifact).

## 3.3 Rolling production results snapshots
From `data/results_summary.csv` (pipeline_run_id `22536680623`):
- `L7` primary: ATS 48.6% (142 games), edge ATS 50.9%
- `L7` ensemble: ATS 51.4% (142 games), edge ATS 50.9%
- `L14` primary: ATS 48.6% (143 games), edge ATS 50.9%

From `data/results_model_split.csv`:
- Per-model split table is present for `fourfactors`, `adjefficiency`, `pythagorean` with ATS and vig-relative values across L7/L30/Season windows.

---

## 4) File map: where to look for each metric/model family

- **Core team advanced metrics**: `espn_metrics.py`, `data/team_game_metrics.csv`
- **SOS/opponent-adjusted + weighted rollups**: `espn_sos.py`, `espn_weighted_metrics.py`, `data/team_game_weighted.csv`, `data/team_game_sos.csv`
- **Tournament/CAGE composites**: `espn_tournament.py`, `espn_tournament_integration.py`, `data/team_tournament_metrics.csv`, `data/team_pretournament_snapshot.csv`
- **Situational context**: `cbb_situational_features.py`, `data/situational_features.csv`
- **Travel fatigue**: `cbb_travel_fatigue.py`
- **Rotation**: `cbb_rotation_features.py`, `data/rotation_features.csv`
- **Player availability/injury proxies**: `player_availability_features.py`, `espn_injury_proxy.py`, `data/player_availability_features.csv`, `data/team_injury_impact.csv`, `data/player_injury_proxy.csv`
- **Luck regression**: `cbb_luck_regression_features.py`, `data/luck_regression_features.csv`
- **Line movement / market context**: `cbb_line_movement_features.py`, `data/line_movement_features.csv`
- **Model training matrix**: `data/backtest_training_data.csv`
- **Primary model logic**: `cbb_prediction_model.py`
- **Ensemble model logic/weights**: `cbb_ensemble.py`, `cbb_config.py`, `data/backtest_optimized_weights.json`, `data/model_weights.json`
- **Backtest outputs/reports**: `cbb_backtester.py`, `data/backtest_results_*.csv`, `data/backtest_model_report_*.csv`, `data/backtest_calibration_*.csv`, `data/csv/backtest_summary.csv`



---

## 5) Appendix: full metric column inventories from generated datasets

### data/team_game_metrics.csv
Column count: **229**

```text
event_id
game_datetime_utc
game_datetime_pst
venue
neutral_site
completed
state
is_ot
num_ot
home_team
away_team
home_team_id
away_team_id
home_conference
away_conference
home_rank
away_rank
spread
over_under
home_ml
away_ml
odds_provider
odds_details
team_id
team
conference
conf_id
home_away
rank
wins
losses
home_wins
home_losses
away_wins
away_losses
conf_wins
conf_losses
win_pct
opponent_id
opponent
opp_conference
opp_rank
opp_wins
opp_losses
points_for
points_against
margin
h1_pts
h2_pts
h1_pts_against
h2_pts_against
fgm
fga
tpm
tpa
ftm
fta
orb
drb
reb
ast
stl
blk
tov
pf
opp_fgm
opp_fga
opp_tpm
opp_tpa
opp_ftm
opp_fta
opp_orb
opp_drb
opp_tov
opp_pf
FGA
FGM
FTA
FTM
TPA
TPM
ORB
DRB
RB
TO
AST
pulled_at_utc
source
parse_version
home_h1
away_h1
home_h2
away_h2
efg_pct
ts_pct
fg_pct
three_pct
ft_pct
three_par
ftr
orb_pct
drb_pct
poss
ortg
drtg
net_rtg
tov_pct
pace
h1_margin
h2_margin
h1_ortg
h2_ortg
h1_drtg
h2_drtg
dead_spread_flag
win
close_game_flag
blowout_flag
margin_capped
cover_margin
cover
ats_push
pythagorean_win_pct
record
home_win_pct
home_record
away_win_pct
away_record
conf_win_pct
conf_record
conf_rank
actual_win_pct_season
pyth_win_pct_season
luck_score
rest_days
games_l7
games_l14
fatigue_index
win_streak
cover_streak
points_for_l5
points_against_l5
margin_l5
margin_capped_l5
efg_pct_l5
ts_pct_l5
three_par_l5
ftr_l5
fg_pct_l5
three_pct_l5
ft_pct_l5
orb_pct_l5
drb_pct_l5
tov_pct_l5
ortg_l5
drtg_l5
net_rtg_l5
poss_l5
pace_l5
h1_pts_l5
h2_pts_l5
h1_pts_against_l5
h2_pts_against_l5
h1_margin_l5
h2_margin_l5
win_l5
cover_l5
cover_margin_l5
stl_l5
blk_l5
ast_l5
points_for_l10
points_against_l10
margin_l10
margin_capped_l10
efg_pct_l10
ts_pct_l10
three_par_l10
ftr_l10
fg_pct_l10
three_pct_l10
ft_pct_l10
orb_pct_l10
drb_pct_l10
tov_pct_l10
ortg_l10
drtg_l10
net_rtg_l10
poss_l10
pace_l10
h1_pts_l10
h2_pts_l10
h1_pts_against_l10
h2_pts_against_l10
h1_margin_l10
h2_margin_l10
win_l10
cover_l10
cover_margin_l10
stl_l10
blk_l10
ast_l10
efg_std_l10
three_pct_std_l10
net_rtg_std_l10
cover_rate_l10
cover_rate_season
ats_margin_l10
close_win_pct_season
close_game_win_pct
ha_ortg_l10
ha_drtg_l10
ha_net_rtg_l10
ha_efg_pct_l10
ha_tov_pct_l10
ha_pace_l10
home_net_rtg_season
away_net_rtg_season
true_orb_pct
true_drb_pct
_improved
conference_name
pipeline_run_id
home_ot1
away_ot1
home_ot2
away_ot2
home_ot3
away_ot3
```

### data/team_game_weighted.csv
Column count: **347**

```text
event_id
game_datetime_utc
game_datetime_pst
venue
neutral_site
completed
state
is_ot
num_ot
home_team
away_team
home_team_id
away_team_id
home_conference
away_conference
home_rank
away_rank
spread
over_under
home_ml
away_ml
odds_provider
odds_details
team_id
team
conference
conf_id
home_away
rank
wins
losses
home_wins
home_losses
away_wins
away_losses
conf_wins
conf_losses
win_pct
opponent_id
opponent
opp_conference
opp_rank
opp_wins
opp_losses
points_for
points_against
margin
h1_pts
h2_pts
h1_pts_against
h2_pts_against
fgm
fga
tpm
tpa
ftm
fta
orb
drb
reb
ast
stl
blk
tov
pf
opp_fgm
opp_fga
opp_tpm
opp_tpa
opp_ftm
opp_fta
opp_orb
opp_drb
opp_tov
opp_pf
FGA
FGM
FTA
FTM
TPA
TPM
ORB
DRB
RB
TO
AST
pulled_at_utc
source
parse_version
home_h1
away_h1
home_h2
away_h2
efg_pct
ts_pct
fg_pct
three_pct
ft_pct
three_par
ftr
orb_pct
drb_pct
poss
ortg
drtg
net_rtg
tov_pct
pace
h1_margin
h2_margin
h1_ortg
h2_ortg
h1_drtg
h2_drtg
dead_spread_flag
win
close_game_flag
blowout_flag
margin_capped
cover_margin
cover
ats_push
pythagorean_win_pct
record
home_win_pct
home_record
away_win_pct
away_record
conf_win_pct
conf_record
conf_rank
actual_win_pct_season
pyth_win_pct_season
luck_score
rest_days
games_l7
games_l14
fatigue_index
win_streak
cover_streak
points_for_l5
points_against_l5
margin_l5
margin_capped_l5
efg_pct_l5
ts_pct_l5
three_par_l5
ftr_l5
fg_pct_l5
three_pct_l5
ft_pct_l5
orb_pct_l5
drb_pct_l5
tov_pct_l5
ortg_l5
drtg_l5
net_rtg_l5
poss_l5
pace_l5
h1_pts_l5
h2_pts_l5
h1_pts_against_l5
h2_pts_against_l5
h1_margin_l5
h2_margin_l5
win_l5
cover_l5
cover_margin_l5
stl_l5
blk_l5
ast_l5
points_for_l10
points_against_l10
margin_l10
margin_capped_l10
efg_pct_l10
ts_pct_l10
three_par_l10
ftr_l10
fg_pct_l10
three_pct_l10
ft_pct_l10
orb_pct_l10
drb_pct_l10
tov_pct_l10
ortg_l10
drtg_l10
net_rtg_l10
poss_l10
pace_l10
h1_pts_l10
h2_pts_l10
h1_pts_against_l10
h2_pts_against_l10
h1_margin_l10
h2_margin_l10
win_l10
cover_l10
cover_margin_l10
stl_l10
blk_l10
ast_l10
efg_std_l10
three_pct_std_l10
net_rtg_std_l10
cover_rate_l10
cover_rate_season
ats_margin_l10
close_win_pct_season
close_game_win_pct
ha_ortg_l10
ha_drtg_l10
ha_net_rtg_l10
ha_efg_pct_l10
ha_tov_pct_l10
ha_pace_l10
home_net_rtg_season
away_net_rtg_season
true_orb_pct
true_drb_pct
_improved
opp_avg_ortg_season
opp_avg_drtg_season
opp_avg_net_rtg_season
opp_avg_efg_season
opp_avg_pace_season
opp_avg_ortg_l5
opp_avg_drtg_l5
opp_avg_net_rtg_l5
opp_avg_efg_l5
opp_avg_pace_l5
opp_avg_ortg_l10
opp_avg_drtg_l10
opp_avg_net_rtg_l10
opp_avg_efg_l10
opp_avg_pace_l10
efg_vs_opp_season
orb_vs_opp_season
drb_vs_opp_season
tov_vs_opp_season
ftr_vs_opp_season
efg_vs_opp_l5
orb_vs_opp_l5
drb_vs_opp_l5
tov_vs_opp_l5
ftr_vs_opp_l5
efg_vs_opp_l10
orb_vs_opp_l10
drb_vs_opp_l10
tov_vs_opp_l10
ftr_vs_opp_l10
adj_pace
adj_ortg
adj_drtg
adj_net_rtg
perf_vs_exp_ortg
perf_vs_exp_drtg
perf_vs_exp_def
perf_vs_exp_net
ortg_wtd_off_l5
efg_pct_wtd_off_l5
ts_pct_wtd_off_l5
fg_pct_wtd_off_l5
three_pct_wtd_off_l5
h1_ortg_wtd_off_l5
h2_ortg_wtd_off_l5
points_for_wtd_off_l5
fgm_wtd_off_l5
fga_wtd_off_l5
tpm_wtd_off_l5
tpa_wtd_off_l5
ftm_wtd_off_l5
fta_wtd_off_l5
ast_wtd_off_l5
three_par_wtd_off_l5
drtg_wtd_def_l5
h1_drtg_wtd_def_l5
h2_drtg_wtd_def_l5
points_against_wtd_def_l5
stl_wtd_def_l5
blk_wtd_def_l5
tov_pct_wtd_def_l5
net_rtg_wtd_qual_l5
margin_wtd_qual_l5
margin_capped_wtd_qual_l5
orb_pct_wtd_qual_l5
drb_pct_wtd_qual_l5
poss_wtd_qual_l5
pace_wtd_qual_l5
h1_margin_wtd_qual_l5
h2_margin_wtd_qual_l5
cover_margin_wtd_qual_l5
win_wtd_qual_l5
cover_wtd_qual_l5
perf_vs_exp_ortg_l5
perf_vs_exp_def_l5
perf_vs_exp_net_l5
ortg_wtd_off_l10
efg_pct_wtd_off_l10
ts_pct_wtd_off_l10
fg_pct_wtd_off_l10
three_pct_wtd_off_l10
h1_ortg_wtd_off_l10
h2_ortg_wtd_off_l10
points_for_wtd_off_l10
fgm_wtd_off_l10
fga_wtd_off_l10
tpm_wtd_off_l10
tpa_wtd_off_l10
ftm_wtd_off_l10
fta_wtd_off_l10
ast_wtd_off_l10
three_par_wtd_off_l10
drtg_wtd_def_l10
h1_drtg_wtd_def_l10
h2_drtg_wtd_def_l10
points_against_wtd_def_l10
stl_wtd_def_l10
blk_wtd_def_l10
tov_pct_wtd_def_l10
net_rtg_wtd_qual_l10
margin_wtd_qual_l10
margin_capped_wtd_qual_l10
orb_pct_wtd_qual_l10
drb_pct_wtd_qual_l10
poss_wtd_qual_l10
pace_wtd_qual_l10
h1_margin_wtd_qual_l10
h2_margin_wtd_qual_l10
cover_margin_wtd_qual_l10
win_wtd_qual_l10
cover_wtd_qual_l10
perf_vs_exp_ortg_l10
perf_vs_exp_def_l10
perf_vs_exp_net_l10
cover_wtd_qual_rate_l5
cover_wtd_qual_rate_l10
momentum_score
form_rating
conference_name
pipeline_run_id
home_ot1
away_ot1
home_ot2
away_ot2
home_ot3
away_ot3
```

### data/team_tournament_metrics.csv
Column count: **376**

```text
event_id
game_datetime_utc
game_datetime_pst
venue
neutral_site
completed
state
is_ot
num_ot
home_team
away_team
home_team_id
away_team_id
home_conference
away_conference
home_rank
away_rank
spread
over_under
home_ml
away_ml
odds_provider
odds_details
team_id
team
conference
conf_id
home_away
rank
wins
losses
home_wins
home_losses
away_wins
away_losses
conf_wins
conf_losses
win_pct
opponent_id
opponent
opp_conference
opp_rank
opp_wins
opp_losses
points_for
points_against
margin
h1_pts
h2_pts
h1_pts_against
h2_pts_against
fgm
fga
tpm
tpa
ftm
fta
orb
drb
reb
ast
stl
blk
tov
pf
opp_fgm
opp_fga
opp_tpm
opp_tpa
opp_ftm
opp_fta
opp_orb
opp_drb
opp_tov
opp_pf
FGA
FGM
FTA
FTM
TPA
TPM
ORB
DRB
RB
TO
AST
pulled_at_utc
source
parse_version
home_h1
away_h1
home_h2
away_h2
efg_pct
ts_pct
fg_pct
three_pct
ft_pct
three_par
ftr
orb_pct
drb_pct
poss
ortg
drtg
net_rtg
tov_pct
pace
h1_margin
h2_margin
h1_ortg
h2_ortg
h1_drtg
h2_drtg
dead_spread_flag
win
close_game_flag
blowout_flag
margin_capped
cover_margin
cover
ats_push
pythagorean_win_pct
record
home_win_pct
home_record
away_win_pct
away_record
conf_win_pct
conf_record
conf_rank
actual_win_pct_season
pyth_win_pct_season
luck_score
rest_days
games_l7
games_l14
fatigue_index
win_streak
cover_streak
points_for_l5
points_against_l5
margin_l5
margin_capped_l5
efg_pct_l5
ts_pct_l5
three_par_l5
ftr_l5
fg_pct_l5
three_pct_l5
ft_pct_l5
orb_pct_l5
drb_pct_l5
tov_pct_l5
ortg_l5
drtg_l5
net_rtg_l5
poss_l5
pace_l5
h1_pts_l5
h2_pts_l5
h1_pts_against_l5
h2_pts_against_l5
h1_margin_l5
h2_margin_l5
win_l5
cover_l5
cover_margin_l5
stl_l5
blk_l5
ast_l5
points_for_l10
points_against_l10
margin_l10
margin_capped_l10
efg_pct_l10
ts_pct_l10
three_par_l10
ftr_l10
fg_pct_l10
three_pct_l10
ft_pct_l10
orb_pct_l10
drb_pct_l10
tov_pct_l10
ortg_l10
drtg_l10
net_rtg_l10
poss_l10
pace_l10
h1_pts_l10
h2_pts_l10
h1_pts_against_l10
h2_pts_against_l10
h1_margin_l10
h2_margin_l10
win_l10
cover_l10
cover_margin_l10
stl_l10
blk_l10
ast_l10
efg_std_l10
three_pct_std_l10
net_rtg_std_l10
cover_rate_l10
cover_rate_season
ats_margin_l10
close_win_pct_season
close_game_win_pct
ha_ortg_l10
ha_drtg_l10
ha_net_rtg_l10
ha_efg_pct_l10
ha_tov_pct_l10
ha_pace_l10
home_net_rtg_season
away_net_rtg_season
true_orb_pct
true_drb_pct
_improved
opp_avg_ortg_season
opp_avg_drtg_season
opp_avg_net_rtg_season
opp_avg_efg_season
opp_avg_pace_season
opp_avg_ortg_l5
opp_avg_drtg_l5
opp_avg_net_rtg_l5
opp_avg_efg_l5
opp_avg_pace_l5
opp_avg_ortg_l10
opp_avg_drtg_l10
opp_avg_net_rtg_l10
opp_avg_efg_l10
opp_avg_pace_l10
efg_vs_opp_season
orb_vs_opp_season
drb_vs_opp_season
tov_vs_opp_season
ftr_vs_opp_season
efg_vs_opp_l5
orb_vs_opp_l5
drb_vs_opp_l5
tov_vs_opp_l5
ftr_vs_opp_l5
efg_vs_opp_l10
orb_vs_opp_l10
drb_vs_opp_l10
tov_vs_opp_l10
ftr_vs_opp_l10
adj_pace
adj_ortg
adj_drtg
adj_net_rtg
perf_vs_exp_ortg
perf_vs_exp_drtg
perf_vs_exp_def
perf_vs_exp_net
ortg_wtd_off_l5
efg_pct_wtd_off_l5
ts_pct_wtd_off_l5
fg_pct_wtd_off_l5
three_pct_wtd_off_l5
h1_ortg_wtd_off_l5
h2_ortg_wtd_off_l5
points_for_wtd_off_l5
fgm_wtd_off_l5
fga_wtd_off_l5
tpm_wtd_off_l5
tpa_wtd_off_l5
ftm_wtd_off_l5
fta_wtd_off_l5
ast_wtd_off_l5
three_par_wtd_off_l5
drtg_wtd_def_l5
h1_drtg_wtd_def_l5
h2_drtg_wtd_def_l5
points_against_wtd_def_l5
stl_wtd_def_l5
blk_wtd_def_l5
tov_pct_wtd_def_l5
net_rtg_wtd_qual_l5
margin_wtd_qual_l5
margin_capped_wtd_qual_l5
orb_pct_wtd_qual_l5
drb_pct_wtd_qual_l5
poss_wtd_qual_l5
pace_wtd_qual_l5
h1_margin_wtd_qual_l5
h2_margin_wtd_qual_l5
cover_margin_wtd_qual_l5
win_wtd_qual_l5
cover_wtd_qual_l5
perf_vs_exp_ortg_l5
perf_vs_exp_def_l5
perf_vs_exp_net_l5
ortg_wtd_off_l10
efg_pct_wtd_off_l10
ts_pct_wtd_off_l10
fg_pct_wtd_off_l10
three_pct_wtd_off_l10
h1_ortg_wtd_off_l10
h2_ortg_wtd_off_l10
points_for_wtd_off_l10
fgm_wtd_off_l10
fga_wtd_off_l10
tpm_wtd_off_l10
tpa_wtd_off_l10
ftm_wtd_off_l10
fta_wtd_off_l10
ast_wtd_off_l10
three_par_wtd_off_l10
drtg_wtd_def_l10
h1_drtg_wtd_def_l10
h2_drtg_wtd_def_l10
points_against_wtd_def_l10
stl_wtd_def_l10
blk_wtd_def_l10
tov_pct_wtd_def_l10
net_rtg_wtd_qual_l10
margin_wtd_qual_l10
margin_capped_wtd_qual_l10
orb_pct_wtd_qual_l10
drb_pct_wtd_qual_l10
poss_wtd_qual_l10
pace_wtd_qual_l10
h1_margin_wtd_qual_l10
h2_margin_wtd_qual_l10
cover_margin_wtd_qual_l10
win_wtd_qual_l10
cover_wtd_qual_l10
perf_vs_exp_ortg_l10
perf_vs_exp_def_l10
perf_vs_exp_net_l10
cover_wtd_qual_rate_l5
cover_wtd_qual_rate_l10
momentum_score
form_rating
t_tournament_dna_score
t_dna_efg_diff
t_dna_tov_diff
t_dna_away_win_pct
t_dna_sos_net_rtg
t_suffocation_rating
t_suf_adj_drtg
t_suf_drb_pct
t_suf_tov_forced_vs
t_momentum_quality_rating
t_mom_three_gap
t_mom_opp_q_trend
t_mom_net_rtg_trend
t_mom_tov_trend
t_regression_risk_flag
t_offensive_identity_score
t_offensive_archetype
t_oi_adj_ortg
t_oi_efg_vs_opp
t_oi_efg_std
t_star_reliance_risk
t_star_danger_flag
t_star_entropy
t_star_top_usage
t_star_top_share
t_star_2nd3rd_share
t_readiness_composite
conference_name
pipeline_run_id
home_ot1
away_ot1
home_ot2
away_ot2
home_ot3
away_ot3
t_top_scorer_efg_l5
t_bench_pts_share_l5
```

### data/team_game_sos.csv
Column count: **263**

```text
event_id
game_datetime_utc
game_datetime_pst
venue
neutral_site
completed
state
is_ot
num_ot
home_team
away_team
home_team_id
away_team_id
home_conference
away_conference
home_rank
away_rank
spread
over_under
home_ml
away_ml
odds_provider
odds_details
team_id
team
conference
conf_id
home_away
rank
wins
losses
home_wins
home_losses
away_wins
away_losses
conf_wins
conf_losses
win_pct
opponent_id
opponent
opp_conference
opp_rank
opp_wins
opp_losses
points_for
points_against
margin
h1_pts
h2_pts
h1_pts_against
h2_pts_against
fgm
fga
tpm
tpa
ftm
fta
orb
drb
reb
ast
stl
blk
tov
pf
opp_fgm
opp_fga
opp_tpm
opp_tpa
opp_ftm
opp_fta
opp_orb
opp_drb
opp_tov
opp_pf
FGA
FGM
FTA
FTM
TPA
TPM
ORB
DRB
RB
TO
AST
pulled_at_utc
source
parse_version
home_h1
away_h1
home_h2
away_h2
efg_pct
ts_pct
fg_pct
three_pct
ft_pct
three_par
ftr
orb_pct
drb_pct
poss
ortg
drtg
net_rtg
tov_pct
pace
h1_margin
h2_margin
h1_ortg
h2_ortg
h1_drtg
h2_drtg
dead_spread_flag
win
close_game_flag
blowout_flag
margin_capped
cover_margin
cover
ats_push
pythagorean_win_pct
record
home_win_pct
home_record
away_win_pct
away_record
conf_win_pct
conf_record
conf_rank
actual_win_pct_season
pyth_win_pct_season
luck_score
rest_days
games_l7
games_l14
fatigue_index
win_streak
cover_streak
points_for_l5
points_against_l5
margin_l5
margin_capped_l5
efg_pct_l5
ts_pct_l5
three_par_l5
ftr_l5
fg_pct_l5
three_pct_l5
ft_pct_l5
orb_pct_l5
drb_pct_l5
tov_pct_l5
ortg_l5
drtg_l5
net_rtg_l5
poss_l5
pace_l5
h1_pts_l5
h2_pts_l5
h1_pts_against_l5
h2_pts_against_l5
h1_margin_l5
h2_margin_l5
win_l5
cover_l5
cover_margin_l5
stl_l5
blk_l5
ast_l5
points_for_l10
points_against_l10
margin_l10
margin_capped_l10
efg_pct_l10
ts_pct_l10
three_par_l10
ftr_l10
fg_pct_l10
three_pct_l10
ft_pct_l10
orb_pct_l10
drb_pct_l10
tov_pct_l10
ortg_l10
drtg_l10
net_rtg_l10
poss_l10
pace_l10
h1_pts_l10
h2_pts_l10
h1_pts_against_l10
h2_pts_against_l10
h1_margin_l10
h2_margin_l10
win_l10
cover_l10
cover_margin_l10
stl_l10
blk_l10
ast_l10
efg_std_l10
three_pct_std_l10
net_rtg_std_l10
cover_rate_l10
cover_rate_season
ats_margin_l10
close_win_pct_season
close_game_win_pct
ha_ortg_l10
ha_drtg_l10
ha_net_rtg_l10
ha_efg_pct_l10
ha_tov_pct_l10
ha_pace_l10
home_net_rtg_season
away_net_rtg_season
true_orb_pct
true_drb_pct
_improved
opp_avg_ortg_season
opp_avg_drtg_season
opp_avg_net_rtg_season
opp_avg_efg_season
opp_avg_pace_season
opp_avg_ortg_l5
opp_avg_drtg_l5
opp_avg_net_rtg_l5
opp_avg_efg_l5
opp_avg_pace_l5
opp_avg_ortg_l10
opp_avg_drtg_l10
opp_avg_net_rtg_l10
opp_avg_efg_l10
opp_avg_pace_l10
efg_vs_opp_season
orb_vs_opp_season
drb_vs_opp_season
tov_vs_opp_season
ftr_vs_opp_season
efg_vs_opp_l5
orb_vs_opp_l5
drb_vs_opp_l5
tov_vs_opp_l5
ftr_vs_opp_l5
efg_vs_opp_l10
orb_vs_opp_l10
drb_vs_opp_l10
tov_vs_opp_l10
ftr_vs_opp_l10
adj_pace
adj_ortg
adj_drtg
adj_net_rtg
conference_name
pipeline_run_id
home_ot1
away_ot1
home_ot2
away_ot2
home_ot3
away_ot3
```

### data/situational_features.csv
Column count: **45**

```text
game_id
team_id
opponent_id
game_date
game_datetime_utc
home_away
season
next_opp_is_ranked
current_opp_rank
lookahead_flag
lookahead_magnitude
prev_opp_was_ranked
prev_game_won
prev_game_margin
letdown_flag
emotional_letdown_score
consecutive_losses
prev_loss_margin
bounce_back_flag
played_earlier_this_season
prior_matchup_winner
prior_matchup_margin
revenge_flag
revenge_margin
days_to_conf_tournament
late_season_flag
is_bubble_team
is_safe_team
bubble_pressure_flag
consecutive_conf_losses
must_win_flag
home_rest_days
away_rest_days
rest_delta
short_rest_flag
extended_rest_flag
optimal_rest_flag
games_in_last_7_days
games_in_last_14_days
fatigue_flag
is_neutral_site
home_field_advantage_applicable
is_conference_game
is_rivalry_game
situational_edge_score
```

### data/rotation_features.csv
Column count: **34**

```text
game_id
team_id
game_datetime_utc
season
rot_size
top2_pused_share
top2_to_rate
rot_efg_l5
rot_efg_l10
rot_to_rate_l5
rot_to_rate_l10
rot_ftrate_l5
rot_3par_l10
rot_stocks_per40_l10
rot_pf_per40_l5
rot_minshare_sd
rot_3p_pct_sd
rot_to_rate_sd
closer_ft_pct
opp_rot_size
opp_top2_pused_share
opp_top2_to_rate
opp_rot_efg_l5
opp_rot_efg_l10
opp_rot_to_rate_l5
opp_rot_to_rate_l10
opp_rot_ftrate_l5
opp_rot_3par_l10
opp_rot_stocks_per40_l10
opp_rot_pf_per40_l5
opp_rot_minshare_sd
opp_rot_3p_pct_sd
opp_rot_to_rate_sd
opp_closer_ft_pct
```

### data/player_availability_features.csv
Column count: **11**

```text
game_id
team_id
game_datetime_utc
season
minutes_available_pct
star_availability_score
injury_impact_delta
lineup_continuity_l3
new_starter_flag
usage_gini
top1_usage_share
```

### data/luck_regression_features.csv
Column count: **31**

```text
game_id
team_id
game_datetime_utc
season
pyth_expected_wins
pyth_actual_win_pct
luck_score
luck_score_l10
team_3p_pct_season_avg
team_3p_pct_l5
three_pt_luck_l5
opp_3p_pct_season_avg
opp_3p_pct_l5
opp_three_pt_luck_l5
close_game_record_l20
close_game_expected_rate
close_game_luck_l20
net_rtg_season_avg
net_rtg_l5
net_rtg_trend
efg_pct_season_avg
efg_pct_l5
efg_luck_l5
to_rate_season_avg
to_rate_l5
to_rate_luck_l5
ft_pct_season_avg
ft_pct_l5
ft_luck_l5
composite_luck_score
regression_candidate_flag
```

### data/line_movement_features.csv
Column count: **25**

```text
game_id
game_date
home_team_id
away_team_id
open_home_spread
open_total
open_source
open_captured_at
close_home_spread
close_total
close_captured_at
spread_move
spread_move_abs
spread_move_direction
steam_move_flag
spread_crossed_key_number
total_move
total_move_direction
total_steam_flag
reverse_line_movement_flag
sharp_side
n_captures
hours_of_movement
line_stability_score
market_confidence_flag
```

### data/team_injury_impact.csv
Column count: **10**

```text
event_id
team_id
players_flagged
players_severe
starters_flagged
team_injury_load
minutes_at_risk_pct
key_player_flag
game_datetime_utc
pipeline_run_id
```

### data/backtest_training_data.csv
Column count: **251**

```text
game_id
event_id
game_date
season
game_datetime_utc
home_team_id
away_team_id
home_team
away_team
home_conference
away_conference
home_rank
away_rank
neutral_site
is_ot
num_ot
home_score_actual
away_score_actual
home_points_for
home_points_against
home_h1_pts
home_h2_pts
home_fgm
home_fga
home_tpm
home_tpa
home_ftm
home_fta
home_orb
home_drb
home_ast
home_stl
home_blk
home_tov
home_pf
away_points_for
away_points_against
away_h1_pts
away_h2_pts
away_fgm
away_fga
away_tpm
away_tpa
away_ftm
away_fta
away_orb
away_drb
away_ast
away_stl
away_blk
away_tov
away_pf
home_h1
away_h1
home_h2
away_h2
home_wins
home_losses
home_win_pct
home_home_wins
home_home_losses
home_away_wins
home_away_losses
home_conf_wins
home_conf_losses
away_wins
away_losses
away_win_pct
away_home_wins
away_home_losses
away_away_wins
away_away_losses
away_conf_wins
away_conf_losses
home_net_rtg_l5
home_net_rtg_l10
home_adj_ortg
home_adj_drtg
home_adj_net_rtg
home_efg_pct_l5
home_efg_pct_l10
home_tov_pct_l5
home_orb_pct_l10
home_ftr_l5
home_pace_l5
home_ortg_l5
home_drtg_l5
home_margin_l5
home_margin_l10
home_cover_l10
home_cover_rate_l10
home_cover_rate_season
home_ats_margin_l10
home_cover_margin
home_rest_days
home_fatigue_index
home_momentum_score
home_form_rating
home_luck_score
home_win_streak
home_cover_streak
away_net_rtg_l5
away_net_rtg_l10
away_adj_ortg
away_adj_drtg
away_adj_net_rtg
away_efg_pct_l5
away_efg_pct_l10
away_tov_pct_l5
away_orb_pct_l10
away_ftr_l5
away_pace_l5
away_ortg_l5
away_drtg_l5
away_margin_l5
away_margin_l10
away_cover_l10
away_cover_rate_l10
away_cover_rate_season
away_ats_margin_l10
away_cover_margin
away_rest_days
away_fatigue_index
away_momentum_score
away_form_rating
away_luck_score
away_win_streak
away_cover_streak
net_rtg_delta_l5
net_rtg_delta_l10
adj_ortg_delta
adj_drtg_delta
adj_net_rtg_delta
efg_delta_l10
to_rate_delta_l5
orb_delta_l10
ftrate_delta_l5
pace_delta_l5
home_field
rest_delta
travel_fatigue_delta
cage_em_diff
cage_t_diff
cage_o_diff
cage_d_diff
home_pg_top_scorer_pts
home_pg_top_scorer_efg
home_pg_bench_pts
home_pg_bench_pts_share
home_pg_starters_count
home_pg_bench_min_share
home_pg_starters_fgm
home_pg_starters_fga
away_pg_top_scorer_pts
away_pg_top_scorer_efg
away_pg_bench_pts
away_pg_bench_pts_share
away_pg_starters_count
away_pg_bench_min_share
away_pg_starters_fgm
away_pg_starters_fga
rot_efg_delta
rot_to_swing_diff
exec_tax_diff
three_pt_fragility_diff
rot_minshare_sd_diff
top2_pused_share_diff
closer_ft_pct_delta
home_rot_size
home_rot_efg_l5
home_rot_efg_l10
home_rot_to_rate_l5
home_closer_ft_pct
away_rot_size
away_rot_efg_l5
away_rot_efg_l10
away_rot_to_rate_l5
away_closer_ft_pct
star_availability_delta
minutes_available_delta
lineup_continuity_delta
usage_gini_delta
new_starter_flag_home
new_starter_flag_away
home_ats_cover_rate_season
home_ats_cover_rate_l10
home_ats_ats_margin_l10
home_ats_cover_rate_l5
home_ats_favorite_cover_rate
home_ats_underdog_cover_rate
away_ats_cover_rate_season
away_ats_cover_rate_l10
away_ats_ats_margin_l10
away_ats_cover_rate_l5
away_ats_favorite_cover_rate
away_ats_underdog_cover_rate
home_t_top_scorer_efg_l5
home_t_bench_pts_share_l5
home_t_star_reliance_risk
home_t_team_injury_burden
home_t_n_injured_starters_l3
away_t_top_scorer_efg_l5
away_t_bench_pts_share_l5
away_t_star_reliance_risk
away_t_team_injury_burden
away_t_n_injured_starters_l3
espn_spread
espn_total
opening_spread
closing_spread
total_line
spread_line
clv_delta
pred_spread
pred_total
home_lookahead_flag
home_letdown_flag
home_bounce_back_flag
home_revenge_flag
home_revenge_margin
away_lookahead_flag
away_letdown_flag
away_bounce_back_flag
away_revenge_flag
home_bubble_pressure_flag
away_bubble_pressure_flag
home_must_win_flag
home_fatigue_flag
away_fatigue_flag
home_extended_rest_flag
away_extended_rest_flag
is_rivalry_game
is_neutral_site
is_conference_game
situational_edge_delta
actual_margin
home_covered_ats
actual_total
covered_over
luck_score_delta
luck_score_l10_delta
three_pt_luck_delta
opp_three_pt_luck_delta
close_game_luck_delta
net_rtg_trend_delta
efg_luck_delta
composite_luck_delta
home_regression_flag
away_regression_flag
data_completeness_tier
created_at
```
