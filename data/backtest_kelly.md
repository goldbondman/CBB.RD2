# Fractional Kelly Backtest: Model + CAGE + Trend
_Generated 2026-03-10 00:20 UTC_

**Dataset**: 253 games with all signals  
**Bets placed**: 177 / 253 (edge ≥ 2.0 pts)  
**Elite spots (5u)**: 25 total  
**Unit scale**: 0.5u – 5u (signal-weighted fractional Kelly)

---

## 1. Overall Performance

| Metric | Flat betting | Kelly-weighted |
|--------|-------------|----------------|
| Bets | 177 | 177 |
| Win rate | 71.2% | 71.2% |
| ROI | +35.9% | +51.1% |
| Total units wagered | — | 472.5u |
| Net unit P&L | — | +241.5u |

---

## 2. Performance by Kelly Unit Tier

| Tier | Bets | Win% | Unit P&L | ROI | Avg edge |
|------|------|------|----------|-----|----------|
| 0.5u | 10 | 20.0% | -3.1u | -61.8% | 2.9 pts |
| 1u | 8 | 50.0% | -0.4u | -4.5% | 3.1 pts |
| 1.5u | 30 | 43.3% | -7.8u | -17.3% | 3.7 pts |
| 2u | 19 | 94.7% | +30.7u | +80.9% | 3.2 pts |
| 2.5u | 32 | 62.5% | +15.5u | +19.3% | 5.8 pts |
| 3u | 26 | 92.3% | +59.5u | +76.2% | 5.7 pts |
| 3.5u | 15 | 73.3% | +21.0u | +40.0% | 7.7 pts |
| 4u | 16 | 87.5% | +42.9u | +67.0% | 8.2 pts |
| 4.5u | 6 | 100.0% | +24.5u | +90.9% | 9.1 pts |
| 5u | 15 | 93.3% | +58.6u | +78.2% | 11.8 pts |

---

## 3. Elite Spots (5u plays)

**Criteria**: edge ≥ 6 pts + CAGE CONFIRMS + Trend aligns  
**Count**: 25 games (9.9% of sample)

| Metric | Value |
|--------|-------|
| Win rate | 96.0% |
| Unit P&L | +95.0u |
| ROI (flat) | +83.3% |
| Avg edge | 9.9 pts |

---

## 4. Signal Stack Comparison (unit-weighted)

| Cohort | Bets | Win% | Unit P&L | Unit ROI |
|--------|------|------|----------|----------|
| Model only (all edges) | 177 | 71.2% | +241.5u | +51.1% |
| + CAGE CONFIRMS | 106 | 84.0% | +222.8u | +67.9% |
| + Trend aligns | 91 | 86.8% | +195.9u | +70.3% |
| + CAGE + Trend | 60 | 93.3% | +160.6u | +78.7% |
| CAGE DIVERGES (red flag) | 64 | 50.0% | +14.2u | +11.0% |

---

## 5. Edge Tier × Signal Stack

| Edge | n | All win% | All P&L | +Both signals win% | +Both P&L |
|------|---|----------|---------|--------------------|-----------|
| 2–4 pts | 50 | 58.0% | +20.7u | 93.8% | +25.3u |
| 4–6 pts | 56 | 71.4% | +60.7u | 89.5% | +40.4u |
| 6–8 pts | 40 | 77.5% | +71.0u | 100.0% | +36.4u |
| 8+ pts | 31 | 83.9% | +89.1u | 93.3% | +58.6u |

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
