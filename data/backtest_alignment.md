# Signal Alignment Backtest: Model + CAGE + Trend
_Generated 2026-03-09 14:56 UTC_

**Dataset**: 217 games with model edge (|edge| ≥ 1.0 pts)  
**Source**: `backtest_training_data.csv` (253 games with all signals)  
**Signals**: Model (`pred_spread` vs `espn_spread`), CAGE (`cage_em_diff`), Trend (`net_rtg_trend_delta`)  
**Sign convention**: model_sign=+1, trend_sign=+1

---

## 1. ATS Accuracy by Signal Alignment Cohort

CAGE threshold: |cage_em_diff| ≥ 3.0 + same direction as model.  
Trend threshold: |net_rtg_trend_delta| ≥ 1.5 + same direction as model.

| Cohort | W | L | ATS% | ROI (−110) | n | |
|--------|---|---|------|------------|---|-|
| Model alone | 148 | 69 | 68.2% | +30.2% | 217 | ✓ |
| Model + CAGE aligns | 90 | 34 | 72.6% | +38.6% | 124 | ✓ |
| Model + Trend aligns | 77 | 34 | 69.4% | +32.4% | 111 | ✓ |
| Model + CAGE + Trend | 51 | 18 | 73.9% | +41.1% | 69 | ✓ |
| Model + CAGE (|EM|≥10) | 122 | 50 | 70.9% | +35.4% | 172 | ✓ |
| Model + CAGE(≥10) + Trend | 62 | 25 | 71.3% | +36.1% | 87 | ✓ |

> **Interpretation**: Does stacking CAGE and/or Trend confirmation lift ATS% above Model alone?

---

## 2. CAGE Agreement Breakdown (all model-edge games)

| CAGE status | W | L | ATS% | ROI | n | |
|-------------|---|---|------|-----|---|-|
| CAGE CONFIRMS model | 90 | 34 | 72.6% | +38.6% | 124 | ✓ |
| CAGE NEUTRAL (small EM) | 7 | 3 | 70.0% | +33.6% | 10 | ✓ |
| CAGE DIVERGES from model | 51 | 32 | 61.4% | +17.3% | 83 | ✓ |

---

## 3. Trend Alignment Breakdown (all model-edge games)

| Trend status | W | L | ATS% | ROI | n | |
|--------------|---|---|------|-----|---|-|
| Trend ALIGNS with model | 77 | 34 | 69.4% | +32.4% | 111 | ✓ |
| Trend NEUTRAL (flat) | 18 | 6 | 75.0% | +43.2% | 24 | ✓ |
| Trend OPPOSES model pick | 53 | 29 | 64.6% | +23.4% | 82 | ✓ |

---

## 4. Model Edge Magnitude × Alignment

| Model edge | All | +CAGE | +Trend | +Both | n_all | n_both |
|------------|-----|-------|--------|-------|-------|--------|
| 1–3 pts | 47.1% | 70.6% | 62.9% | 83.3% | 68 | 18 |
| 3–6 pts | 71.8% | 79.6% | 94.6% | 92.3% | 78 | 26 |
| 6–10 pts | 79.7% | 97.0% | 89.7% | 100.0% | 59 | 17 |
| 10+ pts | 83.3% | 87.5% | 90.0% | 87.5% | 12 | 8 |

---

## 5. Key Takeaways


| Signal stack | ATS% | vs Model alone |
|---|---|---|
| Model alone | 66.8% | baseline |
| + CAGE aligns | 82.3% | +15.4pp |
| + Trend aligns | 82.9% | +16.1pp |
| + CAGE + Trend | 91.3% | +24.5pp |

Signal thresholds: Model edge ≥ 1.0 pts, CAGE |EM| ≥ 3.0, Trend |delta| ≥ 1.5
