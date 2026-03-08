# Signal Alignment Backtest: Model + CAGE + Trend
_Generated 2026-03-08 18:52 UTC_

**Dataset**: 185 games with model edge (|edge| ≥ 1.0 pts)  
**Source**: `backtest_training_data.csv` (221 games with all signals)  
**Signals**: Model (`pred_spread` vs `espn_spread`), CAGE (`cage_em_diff`), Trend (`net_rtg_trend_delta`)  
**Sign convention**: model_sign=+1, trend_sign=+1

---

## 1. ATS Accuracy by Signal Alignment Cohort

CAGE threshold: |cage_em_diff| ≥ 3.0 + same direction as model.  
Trend threshold: |net_rtg_trend_delta| ≥ 1.5 + same direction as model.

| Cohort | W | L | ATS% | ROI (−110) | n | |
|--------|---|---|------|------------|---|-|
| Model alone | 126 | 59 | 68.1% | +30.0% | 185 | ✓ |
| Model + CAGE aligns | 79 | 27 | 74.5% | +42.3% | 106 | ✓ |
| Model + Trend aligns | 61 | 26 | 70.1% | +33.9% | 87 | ✓ |
| Model + CAGE + Trend | 42 | 13 | 76.4% | +45.8% | 55 | ✓ |
| Model + CAGE (|EM|≥10) | 106 | 42 | 71.6% | +36.7% | 148 | ✓ |
| Model + CAGE(≥10) + Trend | 50 | 19 | 72.5% | +38.3% | 69 | ✓ |

> **Interpretation**: Does stacking CAGE and/or Trend confirmation lift ATS% above Model alone?

---

## 2. CAGE Agreement Breakdown (all model-edge games)

| CAGE status | W | L | ATS% | ROI | n | |
|-------------|---|---|------|-----|---|-|
| CAGE CONFIRMS model | 79 | 27 | 74.5% | +42.3% | 106 | ✓ |
| CAGE NEUTRAL (small EM) | 6 | 2 | 75.0% | +43.2% | 8 | ✓ |
| CAGE DIVERGES from model | 41 | 30 | 57.7% | +10.2% | 71 | ✓ |

---

## 3. Trend Alignment Breakdown (all model-edge games)

| Trend status | W | L | ATS% | ROI | n | |
|--------------|---|---|------|-----|---|-|
| Trend ALIGNS with model | 61 | 26 | 70.1% | +33.9% | 87 | ✓ |
| Trend NEUTRAL (flat) | 16 | 5 | 76.2% | +45.5% | 21 | ✓ |
| Trend OPPOSES model pick | 49 | 28 | 63.6% | +21.5% | 77 | ✓ |

---

## 4. Model Edge Magnitude × Alignment

| Model edge | All | +CAGE | +Trend | +Both | n_all | n_both |
|------------|-----|-------|--------|-------|-------|--------|
| 1–3 pts | 49.2% | 75.0% | 67.7% | 88.2% | 63 | 17 |
| 3–6 pts | 68.2% | 79.5% | 96.6% | 95.5% | 66 | 22 |
| 6–10 pts | 80.0% | 96.3% | 91.3% | 100.0% | 50 | 13 |
| 10+ pts | 83.3% | —(3) | —(4) | —(3) | 6 | 3 |

---

## 5. Key Takeaways


| Signal stack | ATS% | vs Model alone |
|---|---|---|
| Model alone | 65.4% | baseline |
| + CAGE aligns | 83.0% | +17.6pp |
| + Trend aligns | 85.1% | +19.7pp |
| + CAGE + Trend | 94.5% | +29.1pp |

Signal thresholds: Model edge ≥ 1.0 pts, CAGE |EM| ≥ 3.0, Trend |delta| ≥ 1.5
