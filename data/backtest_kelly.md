# Fractional Kelly Backtest: Model + CAGE + Trend
_Generated 2026-03-08 23:09 UTC_

**Dataset**: 221 games with all signals  
**Bets placed**: 149 / 221 (edge ≥ 2.0 pts)  
**Elite spots (5u)**: 16 total  
**Unit scale**: 0.5u – 5u (signal-weighted fractional Kelly)

---

## 1. Overall Performance

| Metric | Flat betting | Kelly-weighted |
|--------|-------------|----------------|
| Bets | 149 | 149 |
| Win rate | 69.1% | 69.1% |
| ROI | +32.0% | +49.3% |
| Total units wagered | — | 381.0u |
| Net unit P&L | — | +187.9u |

---

## 2. Performance by Kelly Unit Tier

| Tier | Bets | Win% | Unit P&L | ROI | Avg edge |
|------|------|------|----------|-----|----------|
| 0.5u | 9 | 11.1% | -3.5u | -78.8% | 2.7 pts |
| 1u | 6 | 50.0% | -0.3u | -4.5% | 3.1 pts |
| 1.5u | 29 | 41.4% | -9.1u | -21.0% | 3.7 pts |
| 2u | 17 | 94.1% | +27.1u | +79.7% | 3.2 pts |
| 2.5u | 27 | 59.3% | +8.9u | +13.1% | 5.8 pts |
| 3u | 22 | 95.5% | +54.3u | +82.2% | 5.8 pts |
| 3.5u | 14 | 71.4% | +17.8u | +36.4% | 7.7 pts |
| 4u | 12 | 91.7% | +36.0u | +75.0% | 8.1 pts |
| 4.5u | 5 | 100.0% | +20.5u | +90.9% | 9.2 pts |
| 5u | 8 | 100.0% | +36.4u | +90.9% | 9.9 pts |

---

## 3. Elite Spots (5u plays)

**Criteria**: edge ≥ 6 pts + CAGE CONFIRMS + Trend aligns  
**Count**: 16 games (7.2% of sample)

| Metric | Value |
|--------|-------|
| Win rate | 100.0% |
| Unit P&L | +65.5u |
| ROI (flat) | +90.9% |
| Avg edge | 8.5 pts |

---

## 4. Signal Stack Comparison (unit-weighted)

| Cohort | Bets | Win% | Unit P&L | Unit ROI |
|--------|------|------|----------|----------|
| Model only (all edges) | 149 | 69.1% | +187.9u | +49.3% |
| + CAGE CONFIRMS | 90 | 83.3% | +178.5u | +68.0% |
| + Trend aligns | 70 | 88.6% | +153.5u | +75.0% |
| + CAGE + Trend | 47 | 95.7% | +126.8u | +84.5% |
| CAGE DIVERGES (red flag) | 54 | 46.3% | +9.4u | +8.7% |

---

## 5. Edge Tier × Signal Stack

| Edge | n | All win% | All P&L | +Both signals win% | +Both P&L |
|------|---|----------|---------|--------------------|-----------|
| 2–4 pts | 47 | 57.4% | +19.9u | 93.3% | +23.5u |
| 4–6 pts | 46 | 67.4% | +44.6u | 93.8% | +37.9u |
| 6–8 pts | 35 | 77.1% | +60.3u | 100.0% | +29.1u |
| 8+ pts | 21 | 85.7% | +63.1u | 100.0% | +36.4u |

---

## 6. Unit Sizing Rules (reference)


| Edge (pts) | Base | +CAGE CONFIRMS | +Trend | Both | CAGE DIVERGES |
|------------|------|----------------|--------|------|---------------|
| 2–4        | 1u   | 1.5u           | 1.5u   | 2u   | 0.5u          |
| 4–6        | 2u   | 2.5u           | 2.5u   | 3u   | 1.5u          |
| 6–8        | 3u   | 3.5u           | 3.5u   | 4u   | 2.5u          |
| 8–10       | 4u   | 4.5u           | 4.5u   | 5u ★ | 3.5u         |
| 10+        | 4u   | 4.5u           | 4.5u   | 5u ★ | 3.5u         |

★ = Elite spot. 5u requires edge ≥ 6 pts + CAGE CONFIRMS + Trend aligns.
Max 5u. Min 0.5u (even with CAGE DIVERGES penalty applied).
