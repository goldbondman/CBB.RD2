# CAGE Standalone ATS Predictor Backtest
_Generated 2026-03-10 00:20 UTC_

**Dataset**: 3,297 games (SU), 404 games with market spread (ATS)  
**CAGE model**: `cagerankings_spread` sub-model (one of 7 ensemble components)  
**Raw signal**: `cage_em_diff` = home_cage_em − away_cage_em

---

## 1. Directional (Straight-Up) Accuracy — cage_em_diff

**Overall SU accuracy**: 58.3% (n=3297)

| |EM Diff| bucket | SU Hit Rate | n | Avg EM gap | Avg actual margin |
|------------------|-------------|---|------------|-------------------|
| 0–3 | 47.1% | 257 | 1.5 | 11.0 |
| 3.1–6 | 53.3% | 225 | 4.4 | 10.0 |
| 6.1–10 | 53.0% | 347 | 8.0 | 10.9 |
| 10–15 | 58.4% | 377 | 12.5 | 11.0 |
| 15–20 | 55.2% | 339 | 17.4 | 11.0 |
| 20–30 | 55.1% | 548 | 24.6 | 11.3 |
| 30+ | 65.4% | 1204 | 49.1 | 12.2 |

> **Interpretation**: cage_em_diff < 3 = coin-flip territory. Accuracy climbs reliably above |EM| 10. Best conviction bucket (30+): 65%+ SU in this sample.

---

## 2. ATS Performance (cagerankings_spread vs market line)

**Sample size**: 404 games (training_data (espn_spread))
**Pick signal**: `cage_em_diff` direction (home > 0 → pick home)
**Conviction**: `|cage_em_diff|` magnitude buckets

### Overall ATS Performance

| Model | W | L | ATS% | ROI (−110) |
|-------|---|---|------|------------|
| CAGE (EM direction) | 298 | 106 | 73.8% | +40.8% |
| Ensemble (pred_spread) | 208 | 196 | 51.5% | -1.7% |

### ATS by CAGE Conviction (|cage_em_diff| magnitude)

| EM magnitude | W | L | ATS% | ROI (−110) | n | Note |
|--------------|---|---|------|------------|---|------|
| 0–3 | 11 | 9 | 55.0% | +5.0% | 20 | ✓ VALUE |
| 3.1–5 | 10 | 8 | 55.6% | +6.1% | 18 | ✓ VALUE |
| 5.1–10 | 28 | 16 | 63.6% | +21.5% | 44 | ✓ VALUE |
| 10–20 | 58 | 29 | 66.7% | +27.3% | 87 | ✓ VALUE |
| 20+ | 191 | 44 | 81.3% | +55.2% | 235 | ✓ VALUE |

---

## 3. Market Underdog Picks

**Total games**: 404

| Pick type | Games | % of picks | ATS% | ROI (−110) |
|-----------|-------|------------|------|------------|
| CAGE picks market dog | 157 | 39% | 43.3% | -17.3% |
| CAGE picks market fav | 247 | 61% | 93.1% | +77.8% |

> **Key finding**: CAGE metrics are efficiency-based and highly correlated with the Vegas line. When CAGE diverges and picks the market underdog, it is almost certainly not finding mispriced lines — it is over-extrapolating a team's raw efficiency advantage into a spread pick without accounting for line movement, sharp action, or public money. **Do not use CAGE alone to fade the market.**

---

## 4. CAGE ↔ Ensemble Stacking

Agreement rate: **53%** of games (214/404)

| Filter | n | Ens ATS% | Ens ROI |
|--------|---|----------|---------|
| CAGE agrees with ens | 214 | 73.8% | +41.0% |
| CAGE disagrees with ens | 190 | 26.3% | -49.8% |

> CAGE is a sub-model inside the ensemble (cagerankings_spread weight ~12%). When they agree it mostly means the ensemble already incorporated CAGE's signal. CAGE adds value as a **magnitude filter** (large EM gaps → high edge buckets) rather than as a directional override.

---

## Summary & Recommendations


| Dimension | Finding |
|-----------|---------|
| SU accuracy (all games) | Directional predictor; strongest when |EM| > 10 |
| ATS hit rate (404 games) | **73.8%** using cage_em_diff direction |
| Market underdog picks | CAGE picks dog ~43% of time; covers only ~0% — **avoid** |
| Primary value | Validation signal alongside model/trend picks, not standalone |

### How to use CAGE in the pick stack:

1. **Don't** use cage_em_diff alone to pick vs. a spread. The market already knows team quality.
2. **Do** use cage_validates (CONFIRMS/NEUTRAL/DIVERGES) as a filter on model/trend picks.
3. **Do** use CAGE magnitude (|EM_diff| > 20) to flag dominant matchups — high SU accuracy
   means the "underdog" is likely losing outright, not covering in tight games.
4. **Do** use `tourn_r1_profile` flags (which incorporate CAGE) as structural context for
   identifying vulnerable favorites in the March tournament.
