# Metric Library Implementation

## Scope
- Data source scope: team-game and player-game box score tables only.
- No play-by-play features.
- Leak-free windows: all rolling/season features use `shift(1)`.
- Z-scores: computed within season only.
- Starter/bench logic: centralized in one helper.

## Implemented modules
- `pipeline/advanced_metrics/metric_registry.py`
- `pipeline/advanced_metrics/shared_derivations.py`
- `pipeline/advanced_metrics/starter_bench_helper.py`
- `pipeline/advanced_metrics/metric_library.py`
- `pipeline/advanced_metrics/rolling_window_layer.py`
- `pipeline/advanced_metrics/team_metric_compute.py`
- `pipeline/advanced_metrics/matchup_metric_compute.py`
- `pipeline/advanced_metrics/validation_layer.py`

## Shared derivations
- `poss = fga - orb + tov + 0.44*fta`
- `OffEff = 100 * points_for / poss`
- `DefEff = 100 * points_against / opp_poss`
- `NetRtg = OffEff - DefEff`
- `eFG = (fgm + 0.5*tpm) / fga`
- `3PA_rate = tpa / fga`
- `FTr = fta / fga`
- `FT_pts_per_FGA = ftm / fga`
- `FT_pts_per_poss = ftm / poss`
- `ORB% = orb / (orb + opp_drb)`
- `DRB% = drb / (drb + opp_orb)`
- `TOV% = tov / poss`

## Starter/bench helper
File: `pipeline/advanced_metrics/starter_bench_helper.py`

Starter assignment:
1. Use `starter` flag when present and populated.
2. Otherwise mark top-5 by `min` as starters.

Outputs per `(event_id, team_id)`:
- `bench_minutes_share`
- `TS_bench`
- `TS_starters`
- `REB_rate_bench`
- `REB_rate_starters`

## Metric definitions

### Team metrics

| Metric | Inputs | Definition | Status |
|---|---|---|---|
| `ANE` | `NetRtg`, `opp_pre_NetRtg_season` | `NetRtg - opp_pre_NetRtg_season` | ACTIVE |
| `SVI` | `WL`, `opp_pre_NetRtg_season` | `WL * (1 + opp_pre_NetRtg_season/30)` | ACTIVE |
| `PEQ` | `OffEff`, `TOV%` | `OffEff * (1 - TOV%)` | ACTIVE |
| `POSW` | `ORB%`, `DRB%`, `TOV%` | `(ORB% + DRB% + (1-TOV%))/3` | ACTIVE |
| `WL` | `points_for`, `points_against` | `1 if points_for > points_against else 0` | ACTIVE |
| `ODI` | `eFG`, `TOV%`, `ORB%`, `FTr`, opponent mirrors | `0.4*eFG_edge + 0.25*TOV_edge + 0.2*ORB_edge + 0.15*FTr_edge` | ACTIVE |
| `TC` | `poss`, `opp_pre_poss_season` | `poss - opp_pre_poss_season` | ACTIVE |
| `TIN` | `poss`, `pre_poss_season` | `abs(poss - pre_poss_season)` | ACTIVE |
| `VOL` | `NetRtg`, `pre_NetRtg_season` | `abs(NetRtg - pre_NetRtg_season)` | ACTIVE |
| `DPC` | starter/bench helper outputs | `0.4*bench_minutes_share + 0.3*REB_rate_bench + 0.3*TS_bench` | ACTIVE |
| `FFC` | `eFG`, `TOV%`, `ORB%`, `FTr` | `0.4*eFG - 0.25*TOV% + 0.2*ORB% + 0.15*FTr` | ACTIVE |
| `PXP` | `NetRtg`, `pre_NetRtg_season` | `NetRtg - pre_NetRtg_season` | ACTIVE |
| `SCI` | `ODI`, `season`, `game_datetime_utc` | `abs(within-season leak-free zscore(ODI))` | ACTIVE |

### Matchup metrics

| Metric | Inputs | Definition | Status |
|---|---|---|---|
| `PEI_matchup` | `PEQ_home`, `PEQ_away` | `PEQ_home - PEQ_away` | ACTIVE |
| `ODI_diff` | `ODI_home`, `ODI_away` | `ODI_home - ODI_away` | ACTIVE |
| `ODI_sum` | `ODI_home`, `ODI_away` | `ODI_home + ODI_away` | ACTIVE |
| `MTI` | `ANE_home`, `ANE_away` | `(abs(ANE_home) + abs(ANE_away))/2` | ACTIVE |
| `SCI` | `SCI_home`, `SCI_away` | `(SCI_home + SCI_away)/2` | ACTIVE |

## Rolling window layer
For each metric, generated columns are:
- `{metric}_season`
- `{metric}_L4`
- `{metric}_L7`
- `{metric}_L10`
- `{metric}_L12`
- `{metric}_L10_std`
- `{metric}_trend_L4_L10`
- `{metric}_trend_L10_season`

All computed with leak-free `shift(1)` windows.

## Table generation
- Team table builder: `compute_team_game_metrics(...)`
- Matchup table builder: `compute_matchup_metrics(...)`
- CSV generation wrapper: `generate_metric_tables(...)`

Outputs:
- `team_game_metrics`
- `matchup_metrics`

## Runtime blocking behavior
- Input requirements are defined in `metric_registry.py`.
- `validation_layer.py` marks metrics as `BLOCKED` when required inputs are missing.
- Blocked metrics are emitted as null columns and are not renamed.

