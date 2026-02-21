# ML Architecture Audit & Improvement Roadmap

## College Basketball Prediction System — CBB.RD2

---

## 1. System Architecture Summary

### Pipeline Flow

```
ESPN Scoreboard API ─→ espn_client.py ─→ espn_parsers.py ─→ espn_pipeline.py
                                                                    │
                              ┌─────────────────────────────────────┘
                              ▼
                    ┌──── Raw CSVs ────┐
                    │  games.csv       │
                    │  team_game_logs  │
                    │  player_game_logs│
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────────┐
              ▼              ▼                   ▼
        espn_metrics    espn_sos       espn_player_metrics
        espn_weighted   espn_injury    espn_tournament
              │              │                   │
              └──────────────┼───────────────────┘
                             ▼
                   team_game_weighted.csv
                   team_pretournament_snapshot.csv
                             │
              ┌──────────────┴──────────────────┐
              ▼                                  ▼
    espn_prediction_runner.py          cbb_ensemble.py
    (Primary model — v2.1)             (7-model ensemble)
    cbb_prediction_model.py            ├─ M1 FourFactors
              │                        ├─ M2 AdjEfficiency
              │                        ├─ M3 Pythagorean
              │                        ├─ M4 Momentum
              │                        ├─ M5 Situational
              │                        ├─ M6 CAGERankings
              │                        └─ M7 RegressedEff
              ▼                                  ▼
    predictions_latest.csv       ensemble_predictions_latest.csv
              │                                  │
              └────────────┬─────────────────────┘
                           ▼
              predictions_combined_latest.csv
                           │
              ┌────────────┴──────────────┐
              ▼                           ▼
    cbb_results_tracker.py      cbb_backtester.py
    (daily performance log)     (weekly weight optimizer)
              ▼                           ▼
    results_log.csv             backtest_optimized_weights.json
    results_summary.csv         backtest_results_*.csv
    results_alerts.csv          backtest_calibration_*.csv
```

### Automation (GitHub Actions)

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `update_espn_cbb.yml` | Daily 10:00 UTC | ESPN data ingest + rankings + primary predictions |
| `cbb_predictions_rolling.yml` | After ESPN update | Primary + ensemble predictions, merge, divergence checks |
| `cbb_analytics.yml` | Daily 8 AM / Weekly Mon 10 AM | Results tracking (daily) + backtesting with weight optimization (weekly) |

---

## 2. Model Evaluation Audit

### Model Inventory

| # | Model | Type | Key Features | Target | Feature Overlap | Signal Diversity |
|---|-------|------|-------------|--------|-----------------|------------------|
| P | **Primary (v2.1)** | Recursive bidirectional four-factor | Raw box scores → 3-layer normalized baselines → vs-expectation deltas | Spread + Total | Unique: recursive opponent history, game-level box data | **HIGH** — only model with raw box-score input |
| M1 | **FourFactors** | Heuristic (weighted composite) | `efg_vs_opp`, `tov_vs_opp`, `orb_vs_opp`, `ftr_vs_opp` | Spread + Total | Medium: shares vs_opp features with Primary | Medium — opponent-adjusted four factors only |
| M2 | **AdjEfficiency** | Heuristic (linear) | `cage_em`, `cage_o`, `cage_d`, `cage_t`, `net_rtg_std` | Spread + Total | **HIGH overlap with M7** | Low vs M7 — same core signal |
| M3 | **Pythagorean** | Statistical (normal CDF) | `pythagorean_win_pct`, `luck`, `cage_o`, `cage_t` | Spread + Total | Low overlap | **HIGH** — fundamentally different approach |
| M4 | **Momentum** | Heuristic (recency-weighted) | `net_rtg_l5`, `net_rtg_l10`, `ortg_l5/l10` | Spread + Total | Medium: rolling windows shared with ensemble | **HIGH** — captures trend/streakiness |
| M5 | **Situational** | Heuristic (multi-factor) | `rest_days`, `home_wpct`, `away_wpct`, `close_wpct`, `win_streak` | Spread + Total | Low overlap | **HIGH** — non-efficiency signals |
| M6 | **CAGERankings** | Composite rating | `cage_power_index`, `suffocation`, `clutch_rating`, `resume_score`, `dna_score` | Spread + Total | Low: CAGE composites are unique | **HIGH** — tournament-calibrated metrics |
| M7 | **RegressedEff** | Mean-regressed linear | `cage_em`, `cage_o`, `consistency_score` | Spread + Total | **HIGH overlap with M2** | Low — intentionally conservative M2 |

### Ensemble Interaction

```
M1 ──┬──(w=0.12)──┐
M2 ──┤──(w=0.22)──┤
M3 ──┤──(w=0.14)──┤
M4 ──┤──(w=0.16)──├──→ Weighted Average ──→ Ensemble Spread/Total
M5 ──┤──(w=0.10)──┤                            │
M6 ──┤──(w=0.18)──┤                      Agreement Score
M7 ──┘──(w=0.08)──┘                      Confidence
                                          Edge Flags
```

- **Aggregation**: Weighted linear average (separate weight vectors for spread/total)
- **Training**: Models are **independent** — no sequential dependencies
- **Weight optimization**: Weekly backtester finds optimal weights via Nelder-Mead
- **Agreement scoring**: STRONG (≥70% same side) / MODERATE (≥55%) / SPLIT

### Model Redundancy Findings

#### ⚠️ High Redundancy: M2 (AdjEfficiency) ↔ M7 (RegressedEff)

- **Correlation**: Expected >0.92 — M7 is a mean-regressed version of M2
- **Feature overlap**: Both primarily use `cage_em`, `cage_o`, `cage_t`
- **Combined weight**: 0.30 (0.22 + 0.08) — 30% of ensemble on correlated signals
- **Impact**: Reduces effective diversity; weights partially wash out
- **Recommendation**: Replace M7 with a fundamentally different model (e.g., conference-adjusted or shooting-profile model) or merge M2/M7 into one with built-in regression

#### ✅ Strong Diversity: M3 (Pythagorean) and M5 (Situational)

- M3 uses win-percentage statistics (different signal source)
- M5 uses non-efficiency signals (rest, splits, streaks)
- Both add genuine signal diversity to the ensemble

#### ✅ Strong Diversity: M6 (CAGERankings)

- Uses proprietary CAGE composites unavailable to other models
- Tournament-calibrated metrics provide unique signal

#### 📋 Primary Model Is Siloed

- The primary model (`cbb_prediction_model.py`) operates independently
- Its predictions are **not** part of the 7-model ensemble weighted average
- They're only merged at the CSV output stage
- **Opportunity**: Include primary model output as an 8th input to ensemble

---

## 3. ML Improvement Opportunities (No DB Required)

### Priority Matrix

| Improvement | Impact | Complexity | Runs in GH Actions? | Needs DB? |
|-------------|--------|------------|---------------------|-----------|
| **1. Calibration layer (Platt/isotonic)** | HIGH | Low | ✅ | ❌ |
| **2. Stacking meta-model** | HIGH | Medium | ✅ | ❌ |
| **3. Dynamic model weighting by recency** | HIGH | Low | ✅ | ❌ |
| 4. Variance-aware ensemble blending | Medium | Low | ✅ | ❌ |
| 5. Conference-adjusted features | Medium | Medium | ✅ | ❌ |
| 6. Opponent-adjusted rolling stats | Medium | Medium | ✅ | ❌ |
| 7. Feature importance pruning | Medium | Low | ✅ | ❌ |
| 8. Replace M7 with a diverse model | Medium | Medium | ✅ | ❌ |
| 9. Include primary model in ensemble | Low | Low | ✅ | ❌ |
| 10. Bayesian model averaging | Low | High | ✅ | ❌ |

---

## 4. Top 3 MVP Recommendations

### MVP 1: Calibration Layer (Platt Scaling)

**Problem**: Model confidence scores are heuristic (rule-based), not calibrated to empirical win rates. A model that says "70% confidence" should win ATS ~70% of the time.

**Solution**: After backtesting, fit a logistic (Platt) calibration curve mapping raw ensemble confidence → actual ATS win rate. Apply this transform to live predictions.

**Implementation**:
```
Location:  cbb_backtester.py (training) + cbb_ensemble.py (inference)
Complexity: LOW
Expected impact: +1-3% ATS accuracy on edge-flagged games
Pipeline fit: Backtester trains calibrator weekly → saves to JSON → ensemble loads at predict-time
```

**Steps**:
1. In `cbb_backtester.py`, after computing backtest results, fit `sklearn.linear_model.LogisticRegression` on `[ens_confidence, spread_std, cage_edge]` → ATS outcome
2. Save calibration coefficients to `data/calibration_params.json`
3. In `cbb_ensemble.py`, load calibration params and transform raw confidence

### MVP 2: Stacking Meta-Model

**Problem**: Weighted averaging treats model errors as independent. In reality, models make correlated errors (e.g., both M2 and M7 overestimate the same teams).

**Solution**: Train a simple linear regression (ridge) on the 7 model outputs to learn optimal combination weights that account for correlation structure.

**Implementation**:
```
Location:  cbb_backtester.py (training) + EnsemblePredictor (inference)
Complexity: MEDIUM
Expected impact: +1-2% ATS, -0.5 MAE reduction
Pipeline fit: Backtester trains stacker weekly → saves coefficients → ensemble uses at predict-time
```

**Steps**:
1. In backtester, after collecting all per-model spread predictions, fit `Ridge(alpha=1.0)` on `[m1_spread, ..., m7_spread, cage_edge, barthag_diff]` → actual_margin
2. Save coefficients to `data/stacking_coefficients.json`
3. In ensemble, optionally use stacking coefficients instead of simple weighted average

### MVP 3: Dynamic Model Weighting by Recent Accuracy

**Problem**: Fixed weights don't adapt to mid-season model performance changes. If M4 (Momentum) is running hot in February, it should get more weight.

**Solution**: Track per-model L14/L30 ATS accuracy in `cbb_results_tracker.py`. Weight models proportionally to their recent accuracy.

**Implementation**:
```
Location:  cbb_results_tracker.py (tracking) + cbb_ensemble.py (weighting)
Complexity: LOW
Expected impact: +0.5-1.5% ATS in volatile periods (mid-conference play)
Pipeline fit: Results tracker computes L14 accuracy daily → saves weights → ensemble loads
```

**Steps**:
1. In results tracker, compute per-model L14 ATS% from `results_log.csv`
2. Convert to weights: `w_i = max(0.02, ats_pct_i / sum(ats_pct_all))`
3. Blend with backtest-optimized weights: `final = 0.7 * backtest + 0.3 * dynamic`
4. Save to `data/dynamic_model_weights.json`
5. In `EnsembleConfig.from_optimized()`, load and blend dynamic weights

---

## 5. Implementation Roadmap

### Phase 1 — Immediate (This PR)
- [x] Complete `cbb_ensemble.py` (was a 3-line stub)
- [x] Implement all 7 sub-models with distinct analytical approaches
- [x] Add 62 tests covering all models, ensemble, config, and integration
- [x] Verify all existing tests still pass (90/90)
- [x] Document full architecture audit

### Phase 2 — Next Sprint (Low-Lift)
- [ ] **MVP 1**: Add Platt calibration layer to backtester + ensemble
- [ ] **MVP 3**: Add dynamic model weighting to results tracker
- [ ] Reduce M2/M7 redundancy (consider merging or replacing M7)
- [ ] Include primary model output as 8th ensemble input

### Phase 3 — Following Sprint (Medium-Lift)
- [ ] **MVP 2**: Implement stacking meta-model in backtester
- [ ] Add conference-adjusted performance features
- [ ] Implement variance-aware ensemble blending (weight by inverse spread_std)
- [ ] Add feature importance analysis to backtester reports

### Phase 4 — Future (Infrastructure)
- [ ] Supabase integration for persistent model state
- [ ] Model registry for version tracking
- [ ] A/B testing framework for model comparison
- [ ] Real-time line movement integration

---

## Appendix: Feature Overlap Matrix

```
                      Primary  M1-FF  M2-AdjE  M3-Pyth  M4-Mom  M5-Sit  M6-CAGE  M7-Reg
Raw box scores           ✓
vs-expectation deltas    ✓       ✓
cage_em                          (via)   ✓                                  (via)    ✓
cage_o / cage_d                  (via)   ✓                                  (via)    ✓
pyth_win_pct                                      ✓
L5/L10 rolling                                             ✓
rest_days                                                           ✓
home/away splits                                                    ✓
CAGE composites                                                              ✓
consistency_score                                                                    ✓
luck                                              ✓
```

*Legend: ✓ = primary input, (via) = indirect dependency*
