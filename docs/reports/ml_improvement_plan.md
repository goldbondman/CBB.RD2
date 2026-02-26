# CBB 8-Model System Review and ML Upgrade Plan

## Scope
This document maps the current primary + 7-model ensemble architecture and proposes high-ROI upgrades that preserve current pipeline outputs.

## Step 1 — Current model map (specific)

| Model | File + entry point | Target(s) predicted | Main feature families | Training vs rule logic | Vegas usage | Output fields + merge path |
|---|---|---|---|---|---|---|
| Primary model | `cbb_prediction_model.py` → `CBBPredictionModel.predict_game()`; orchestrated by `espn_prediction_runner.py` → `run_predictions()` | `predicted_spread`, `predicted_total`, confidence (win prob not explicitly produced) | Recursive opponent-adjusted game history; L5/L10 decayed windows; Four Factors deltas (`efg/tov/orb/drb/ftr`); pace; net efficiency vs expectation; foul-rate variance | Deterministic formula model with fixed coefficients in `ModelConfig`; no fitted training artifact in repo | Uses lines only after prediction in runner for edge calc (`spread_diff_vs_line`, `total_diff_vs_line`, picks/flags) | Runner writes `pred_spread`, `pred_total`, projected scores, confidence, line diffs and metadata to `data/predictions_*.csv` and `predictions_latest.csv`; workflow copies to `predictions_primary.csv` and later merges into `predictions_combined_latest.csv` |
| M1 FourFactors | `cbb_ensemble.py` → `FourFactorsModel.predict()` | spread + total | Four Factors raw and opponent-context deltas (`efg_vs_opp`, `tov_vs_opp`, `orb_vs_opp`, `ftr_vs_opp`), pace | Rule-based weighted formula | Optional line passed only to ensemble combiner; model itself ignores Vegas | Exported as `fourfactors_spread/total/conf` via `to_flat_dict()`; merged into combined CSV with `ens_` prefix in workflow |
| M2 AdjEfficiency | `cbb_ensemble.py` → `AdjustedEfficiencyModel.predict()` | spread + total | `cage_em`, `cage_o`, `cage_d`, `cage_t` | Rule-based efficiency differential | Same as above | `adjefficiency_spread/total/conf` |
| M3 Pythagorean | `cbb_ensemble.py` → `PythagoreanModel.predict()` | spread + total | `barthag` Log5 win rate; normal inverse to spread; pace + implied ortg/drtg from barthag/net | Rule-based probabilistic conversion | Same as above | `pythagorean_spread/total/conf` |
| M4 Momentum | `cbb_ensemble.py` → `MomentumModel.predict()` | spread + total | L5/L10 net/ortg/drtg/pace trends, trend deltas, three-point regression risk, momentum score | Rule-based recency/trend model | Same as above | `momentum_spread/total/conf` |
| M5 Situational | `cbb_ensemble.py` → `SituationalModel.predict()` | spread + total | rest days, schedule density, fatigue factor, home/away split effects, streak, clutch score | Rule-based adjustment model | Same as above | `situational_spread/total/conf` |
| M6 CAGERankings | `cbb_ensemble.py` → `CAGERankingsModel.predict()` | spread + total | CAGE composite: `cage_em`, power index, resume score, suffocation, barthag | Rule-based weighted composite | Same as above | `cagerankings_spread/total/conf` |
| M7 RegressedEff | `cbb_ensemble.py` → `RegressedEfficiencyModel.predict()` | spread + total | luck-adjusted efficiencies, consistency score, floor/ceiling EM regression, pace | Rule-based shrinkage model | Same as above | `regressedeff_spread/total/conf` |
| Ensemble combiner | `cbb_ensemble.py` → `EnsemblePredictor.predict()` | `ensemble_spread`, `ensemble_total`, `ensemble_confidence`, ranges/std/agreement | Weighted blend of 7 model outputs; confidence-based weight attenuation | Rule-based static weights (or backtest-loaded JSON) | Optional `spread_line` only for `line_value` text; does not alter numeric prediction | `to_flat_dict()` emits ensemble + per-model fields; workflow writes `ensemble_predictions_latest.csv` then left-joins to primary output into `predictions_combined_latest.csv` with `ens_` prefix |

## Step 2 — Redundancy and failure mode diagnosis (what can be concluded now)

### Structural overlap likely high
- **M2 AdjEfficiency, M6 CAGERankings, M7 RegressedEff** share heavy CAGE efficiency DNA, so they are likely strongly correlated.
- **M1 FourFactors** and the **primary model** both rely on Four-Factor-like efficiency composition.
- **M4 Momentum** and **M5 Situational** are the most orthogonal context models, but still use some shared pace/efficiency scaffolding.

### Current measurable diagnostics already supported in repo
- `cbb_backtester.py` can compute per-model MAE, ATS%, O/U%, Brier, calibration curves, and optimize ensemble weights.
- It already avoids leakage with per-game pre-tip cutoffs.

### What is missing right now
- No historical `data/*.csv` artifacts were present in this repo snapshot, so no empirical correlation matrix/split metrics could be computed in this review pass.

## Step 3 — Recommended ensemble upgrade (highest ROI)

### Choose **B) Stacking model** (best fit to current codebase)
Why this over dynamic-only weights or pure gating:
1. Repo already emits all 7 sub-model outputs + primary outputs per game, so meta-features are naturally available.
2. Current combiner is static weighted average; stacking can directly learn de-correlation and context interactions.
3. Easy to keep auditable using **regularized linear/logistic models** and walk-forward retraining.

### Proposed stackers
- **Win probability stacker**: logistic regression on model-implied win probs + stabilizers (`spread_line`, home/away, pace proxy, conference flag).
- **Margin stacker**: ridge regression on model spreads + same stabilizers.
- **Total stacker**: ridge regression on model totals + total_line + tempo proxy.

### Leakage-safe protocol
- Strict date-based walk-forward (train through day D-1, predict day D).
- Persist daily model coefficients and data window metadata.

## Step 4 — Optional new model (only if incremental)

Add **one** new component: **Vegas residual model** (recommended over tree booster initially).
- Target: `actual_margin - market_spread`.
- Features: existing pre-tip team and model outputs (no post-game leakage).
- Output: residual-adjusted spread and edge score.

Rationale: this fills a gap not directly modeled by current 8 outputs (market inefficiency residual), while preserving compatibility with existing fields and betting logic.

## Step 5 — Calibration layer

Apply two calibration layers:
1. Primary model implied win probability (derive from spread via normal CDF first, then calibrate).
2. Stacked ensemble win probability.

Method:
- Start with **Platt scaling** for stability; compare against isotonic on enough sample size.
- Store calibration artifact by retrain date and evaluation Brier/log-loss deltas.

## Step 6 — Validation/backtesting framework aligned to betting

Enhance current backtester outputs with:
- Segment slices: early/mid/late season, conference/non-conference, home/away, ranked/unranked, spread bins, tempo tiers.
- CLV proxy tracking where open/close lines exist.
- Confidence bucket ROI for staking validation.
- Explicit pre-tip feature audits (already mostly covered by cutoff logic, but add assertions for shifted rolling windows).

## Ranked upgrade list (expected impact)
1. **Stacked ensemble with walk-forward retraining + segment-aware features** (largest expected gain in MAE/Brier/ROI).
2. **Probability calibration layer (primary + stacked)** (largest gain in staking reliability and risk control).
3. **Vegas residual addon model (single new model only)** (incremental edge capture vs market).

## Concrete implementation plan (no code changes yet)

### Files to change
- `cbb_backtester.py`
  - Add correlation matrix export for model spreads/totals.
  - Add segment split reporting module.
  - Add walk-forward stacker training/eval path.
  - Add calibration evaluation (Platt/isotonic) and Brier deltas.
- `cbb_ensemble.py`
  - Add optional stacker inference path replacing static weighted blend when artifact exists.
  - Keep fallback to current weighted blend.
- `espn_prediction_runner.py`
  - Add primary implied win-prob output and calibrated variant.
  - Preserve current columns; append new fields only.
- `.github/workflows/cbb_predictions_rolling.yml`
  - Load latest stacker/calibration artifacts before prediction merge.
  - Add artifact integrity checks.

### New modules
- `ml/stacking.py` — train/load/predict for logistic + ridge stackers.
- `ml/calibration.py` — Platt/isotonic fit/apply utilities.
- `ml/segmentation.py` — game segmentation helpers for diagnostics.
- `ml/vegas_residual.py` — optional residual model train/infer.

### Guardrails/tests
- Unit tests: no leakage in walk-forward splits, stable feature ordering, deterministic artifact IO.
- Regression tests: stacker fallback behavior when artifact missing.
- Integrity checks: reject predictions if required base model columns missing; audit row counts and null rates.
