# validation_report

## TEST 1: Sample Game Verification (Duke @ Florida, 2026-03-04)
✅ Split lookup resolved for Florida(home)/Duke(away).

| Metric | Florida home used | Duke away used |
|---|---:|---:|
| eFG | 53.406 | 57.950 |
| ORB | 35.428 | 28.825 |
| NetRtg | 25.272 | 28.633 |

✅ 5 sample games: helper used correct home/away split columns (verified against team_splits.csv latest-row lookup).

## TEST 2: Backtest Comparison (100-game holdout)
Holdout size (MAE): 100 games.
- MAE before: 17.445
- MAE after (simulated residual-HCA policy): 17.445
- Delta: +0.000 (negative is better)
- ATS before: 50.00% on 2 lined games (within 100-game holdout).
- ATS after (simulated): 50.00% on 2 lined games.

## TEST 3: Edge Cases
✅ Neutral site: max |after-before| = 0.000 (expected 0.000).
✅ Extreme split scenario sanity: helper returns side-specific fields (`home_efg` / `away_efg`, `home_netrtg` / `away_netrtg`) without cross-side leakage.
✅ Back-to-back road scenario logic: away team values are always sourced from `*_away` columns via `get_team_splits`.

## FINAL CHECK
- Reduced flat HCA constants to residual +1.0 in active prediction paths.
- Split helper used in model/pipeline paths where implemented.
- Intercepts remain learned terms (not rewritten as HCA constants).
