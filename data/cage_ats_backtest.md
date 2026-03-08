# CAGE Standalone ATS Predictor Backtest
_Generated 2026-03-08 14:30 UTC_

**Dataset**: 3,297 games (SU), 88 games with market spread (ATS)  
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

**Sample size**: 88 games with market spread lines

### Overall ATS Performance

| Model | W | L | ATS% | ROI (−110) |
|-------|---|---|------|------------|
| CAGERankings | 52 | 36 | 59.1% | +12.8% |
| Ensemble | 52 | 36 | 59.1% | +12.8% |

### ATS by Edge Bucket (CAGE vs market line)

| Edge bucket | W | L | ATS% | ROI (−110) | Note |
|-------------|---|---|------|------------|------|
| 0–3 | 8 | 7 | 53.3% | +1.8% | — |
| 3.1–5 | 4 | 3 | 57.1% | +9.1% | ✓ VALUE |
| 5.1–8 | 4 | 7 | 36.4% | -30.6% | ⚠ FADE |
| 8.1+ | 36 | 19 | 65.5% | +25.0% | ✓ VALUE |

---

## 3. Market Underdog Picks

**Total games**: 88

| Pick type | Games | % of picks | ATS% | ROI (−110) |
|-----------|-------|------------|------|------------|
| CAGE picks market dog | 38 | 43% | 23.7% | -54.8% |
| CAGE picks market fav | 50 | 57% | 86.0% | +64.2% |

> **Key finding**: CAGE metrics are efficiency-based and highly correlated with the Vegas line. When CAGE diverges and picks the market underdog, it is almost certainly not finding mispriced lines — it is over-extrapolating a team's raw efficiency advantage into a spread pick without accounting for line movement, sharp action, or public money. **Do not use CAGE alone to fade the market.**

---

## 4. CAGE ↔ Ensemble Stacking

Agreement rate: **98%** of games (86/88)

| Filter | n | Ens ATS% | Ens ROI |
|--------|---|----------|---------|
| CAGE agrees with ens | 86 | 59.3% | +13.2% |
| CAGE disagrees with ens | 2 | 50.0% | -4.5% |

> CAGE is a sub-model inside the ensemble (cagerankings_spread weight ~12%). When they agree it mostly means the ensemble already incorporated CAGE's signal. CAGE adds value as a **magnitude filter** (large EM gaps → high edge buckets) rather than as a directional override.

---

## Summary & Recommendations


| Dimension | Finding |
|-----------|---------|
| SU accuracy (all games) | **58.3%** — solid directional predictor when |EM| > 10 |
| ATS hit rate (88 games) | **59.1%** — matches ensemble; sample too small for conviction |
| Best ATS edge bucket | **8.1+ edge: 65.5% / +25% ROI** (n=55) — highest value tier |
| Market underdog picks | CAGE picks dog 43% of time but covers only 24% — **avoid** |
| Stacking with ensemble | 98% agreement rate — minimal additive signal |
| Primary value | Confirming signal for large-edge (8.1+) ensemble picks, not standalone |

### How to use CAGE in the pick stack:

1. **Don't** use cage_em_diff alone to pick vs. a spread. The market already knows team quality.
2. **Do** use CAGE edge bucket as a filter: when the ensemble has 8.1+ edge AND CAGE agrees,
   that bucket historically runs at 65%+ ATS.
3. **Do** use CAGE magnitude (|EM_diff| > 30) to flag dominant matchups — 65% SU accuracy
   means the "underdog" is likely just losing outright, not covering in tight games.
4. **Do** use `tourn_r1_profile` flags (which incorporate CAGE) as structural context for
   identifying vulnerable favorites in the March tournament.
