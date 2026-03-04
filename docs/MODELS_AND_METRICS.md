# CBB.RD2 — Models & Metrics Reference

> All models, composite metrics, feature columns, and analysis tools added to the CBB.RD2 pipeline.
> PEI (Performance vs Expectation Index) is used as the canonical example format throughout.

---

## Table of Contents

1. [Prediction Models](#1-prediction-models)
   - [Primary Model (v2.1)](#primary-model-v21)
   - [Ensemble Sub-Models (M1–M8)](#ensemble-sub-models-m1m8)
2. [CAGE Proprietary Metrics](#2-cage-proprietary-metrics)
3. [Performance vs Expectation (PEI)](#3-performance-vs-expectation-pei)
4. [Opponent-Weighted Rolling Metrics](#4-opponent-weighted-rolling-metrics)
5. [Strength of Schedule (SOS) Metrics](#5-strength-of-schedule-sos-metrics)
6. [Core Per-Game Metrics](#6-core-per-game-metrics)
7. [Tournament Composite Metrics](#7-tournament-composite-metrics)
8. [Analysis & Intelligence Tools](#8-analysis--intelligence-tools)
9. [Supplemental Feature Sets](#9-supplemental-feature-sets)

---

## 1. Prediction Models

### Primary Model (v2.1)

**File:** `cbb_prediction_model.py` → `CBBPredictionModel.predict_game()`
**Orchestrator:** `espn_prediction_runner.py`

**Philosophy:** Beat the market through normalized performance vs expectation.

**Architecture:** Recursive Bidirectional Four-Factor Analysis with three-layer normalized baselines.

| Layer | Description |
|-------|-------------|
| Layer 1 | Raw box-score stats (fgm, fga, tpm, tpa, ftm, fta, orb, drb, tov, pf) |
| Layer 2 | Opponent-weighted rolling averages (L5/L10 with smooth exponential decay) |
| Layer 3 | Schedule-adjusted baselines via `NormalizedOpponentBaseline` |

**Key Parameters (ModelConfig):**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `vs_exp_weight` | 0.70 | Weight on vs-expectation signal vs raw stats |
| `raw_weight` | 0.30 | Weight on raw rolling stats |
| `efg_weight` | 0.28 | Four-factor weight for eFG% |
| `tov_weight` | 0.22 | Four-factor weight for TOV% |
| `orb_weight` | 0.18 | Four-factor weight for ORB% |
| `ftr_weight` | 0.16 | Four-factor weight for FTR |
| `drb_weight` | 0.16 | Four-factor weight for DRB% |
| `default_hca` | 3.2 pts | Home court advantage |

**Tournament Multipliers** (applied to `predicted_total` only):

| Game Type | Multiplier | Rationale |
|-----------|-----------|-----------|
| `regular` | 1.000 | No adjustment |
| `conf_tournament` | 0.964 | Defensive preparation, neutral court |
| `ncaa_r1` | 0.958 | Heightened pressure, fewer fast breaks |
| `ncaa_r2` | 0.951 | Maximum tournament environment drag |

**Outputs:** `pred_spread`, `pred_total`, `predicted_home_score`, `predicted_away_score`, `model_confidence`, `spread_diff_vs_line`, `total_diff_vs_line`

---

### Ensemble Sub-Models (M1–M8)

**File:** `cbb_ensemble.py`
**Entry Point:** `EnsemblePredictor.predict()` → `EnsembleResult`

All models consume a `TeamProfile` (one row per team from `team_pretournament_snapshot.csv`) and produce `ModelPrediction(spread, total, confidence)`.

---

#### M1 — FourFactors

**Class:** `FourFactorsModel`

**Type:** Heuristic weighted composite

**Formula:** Predicts spread from opponent-adjusted four-factor differentials.

| Feature | Role | Dean Oliver Weight |
|---------|------|--------------------|
| `efg_vs_opp` | eFG% above what opponent typically allows | 40% |
| `tov_vs_opp` | TOV% below what opponent typically forces | 25% |
| `orb_vs_opp` | ORB% above what opponent typically allows | 20% |
| `ftr_vs_opp` | FTR above what opponent typically allows | 15% |

**Signal diversity:** Medium — opponent-adjusted four factors only.

---

#### M2 — AdjustedEfficiency

**Class:** `AdjustedEfficiencyModel`

**Type:** Heuristic linear

**Key features:** `cage_em`, `cage_o`, `cage_d`, `cage_t`, `net_rtg_std`

**Formula:** Efficiency edge × pace factor → projected margin.

**Signal diversity:** Low vs M7 — high structural overlap (both use `cage_em`).

---

#### M3 — Pythagorean

**Class:** `PythagoreanModel`

**Type:** Statistical (normal CDF)

**Key features:** `pythagorean_win_pct`, `luck`, `cage_o`, `cage_t`, `barthag`

**Formula:** Log5 win probability from Pythagorean expectations → normal inverse → implied spread.

```
barthag = cage_o^11.5 / (cage_o^11.5 + cage_d^11.5)
win_prob_h_vs_a = (barthag_h * (1 - barthag_a)) /
                  (barthag_h * (1 - barthag_a) + barthag_a * (1 - barthag_h))
spread = Φ⁻¹(win_prob) × σ
```

**Signal diversity:** HIGH — fundamentally different statistical approach.

---

#### M4 — Momentum

**Class:** `MomentumModel`

**Type:** Heuristic recency-weighted

**Key features:** `net_rtg_l5`, `net_rtg_l10`, `ortg_l5`, `ortg_l10`, `drtg_l5`, `momentum_score`

**Formula:** L5 vs L10 trend delta weighted by recency decay.

**Signal diversity:** HIGH — captures trend and streakiness.

---

#### M5 — Situational

**Class:** `SituationalModel`

**Type:** Heuristic multi-factor context model

**Key features:** `rest_days`, `home_wpct`, `away_wpct`, `close_wpct`, `win_streak`, schedule density

**Formula:** Rest advantage + home/away split differential + streak momentum.

**Rest Adjustment Logic:**
- 0 rest days (back-to-back): penalty applied
- 1 day: slight penalty
- 2–3 days: optimal window
- 4+ days: mild rust penalty

**Signal diversity:** HIGH — non-efficiency signals orthogonal to all other models.

---

#### M6 — CAGERankings

**Class:** `CAGERankingsModel`

**Type:** Composite rating (CAGE proprietary)

**Key features:** `cage_power_index`, `suffocation`, `clutch_rating`, `resume_score`, `dna_score`, `barthag`

**Formula:** CAGE Power Index differential → scaled spread + efficiency confirmation.

**Signal diversity:** HIGH — uses proprietary CAGE composites unique to this system.

---

#### M7 — LuckRegression

**Class:** `LuckRegressionModel`

**Type:** Mean-regression model

**Key features:** `luck`, `pythagorean_win_pct`, `actual_win_pct`, `cage_em`, `consistency_score`

**Formula:** Regresses efficiency toward luck-adjusted expectation, then derives spread.

**Signal diversity:** Low — intentional conservative complement to M2 (overlapping CAGE efficiency signals).

---

#### M8 — Variance

**Class:** `VarianceModel`

**Type:** Volatility-aware efficiency moderator

**Key features:** `efg_std_l10`, `three_pct_std_l10`, `consistency_score`, `net_rtg`

**Formula:** Moderates spread confidence when shooting variance is high; adjusts total prediction for high-variance matchups.

**Signal diversity:** UNIQUE — only model explicitly modeling prediction uncertainty from volatility.

---

#### Ensemble Aggregation

**Default weights (equal-start baseline):**

| Model | Spread Weight | Total Weight |
|-------|--------------|-------------|
| M1 FourFactors | 0.125 | 0.125 |
| M2 AdjEfficiency | 0.125 | 0.125 |
| M3 Pythagorean | 0.125 | 0.125 |
| M4 Momentum | 0.125 | 0.125 |
| M5 Situational | 0.125 | 0.125 |
| M6 CAGERankings | 0.125 | 0.125 |
| M7 LuckRegression | 0.125 | 0.125 |
| M8 Variance | 0.125 | 0.125 |

**Live weights:** Loaded from `data/active_weights.json` → `data/backtest_optimized_weights.json` (weekly Nelder-Mead optimization via `cbb_backtester.py`).

**Agreement scoring:**
- `STRONG`: ≥70% of models on same side
- `MODERATE`: ≥55% on same side
- `SPLIT`: no majority

---

## 2. CAGE Proprietary Metrics

> CAGE = **C**omposite **A**djusted **G**rade **E**ngine — our answer to KenPom / Bart Torvik / Haslametrics.

**File:** `espn_rankings.py`
**Output:** `data/cbb_rankings.csv`

---

### cage_em — CAGE Efficiency Margin

**Category:** Core Efficiency (KenPom AdjEM equivalent)

**Formula:**
```
cage_em = adj_ortg - adj_drtg
```

**Range:** -25 to +30 pts/100 possessions

**Grading:**

| Grade | cage_em | Tier |
|-------|---------|------|
| A+  | ≥ 25    | National title contender (top 5) |
| A   | 18–25   | Elite (top 10–15) |
| A-  | 12–18   | Very good (top 25) |
| B+  | 7–12    | Tournament team (4–7 seed range) |
| B   | 2–7     | Tournament bubble (8–10 seed) |
| B-  | 0–2     | Bubble |
| C+  | -3–0    | NIT contender |
| C   | -7–-3   | Mediocre |
| C-  | -12–-7  | Poor |
| D   | -18–-12 | Bottom of conference |
| F   | < -18   | Historically bad |

**Primary sort column for all rankings.**

---

### cage_o / cage_d / cage_t — Adjusted Efficiency Ratings

| Metric | Description | Range | KenPom Equivalent |
|--------|-------------|-------|-------------------|
| `cage_o` | Adjusted Offensive Rating — points scored per 100 possessions, adjusted for opponent D quality | 95–125 | AdjO |
| `cage_d` | Adjusted Defensive Rating — points allowed per 100 possessions, adjusted for opponent O quality. **Lower is better.** | 85–115 | AdjD |
| `cage_t` | Adjusted Tempo — possessions per 40 minutes, pace-adjusted | 60–80 | AdjT |

---

### barthag — Win Probability vs Average D1

**Category:** Win Probability (Torvik BARTHAG equivalent)

**Formula:**
```
barthag = cage_o^11.5 / (cage_o^11.5 + cage_d^11.5)
```

**Range:** 0.000–1.000

**Interpretation:**
- ≥ 0.90: Elite (title contender)
- ≈ 0.50: Bubble team
- < 0.20: Bottom tier D1

---

### wab — Wins Above Bubble

**Category:** Win Probability (Torvik WAB equivalent)

**Definition:** How many more wins this team earned vs what a bubble team (cage_em = 0) would have earned against the identical schedule.

**Range:** -5 to +12

---

### cage_power_index — CAGE Composite Power Index

**Category:** CAGE Composite (Master ranking metric)

**Formula:** Weighted blend of five components:

| Component | Weight |
|-----------|--------|
| `cage_em` | 35% |
| `barthag` | 20% |
| `suffocation` | 15% |
| `momentum` | 12% |
| `resume` | 10% |
| `clutch` | 8% |

**Range:** 0–100 (normalized to D1 distribution)

**Interpretation:**
- 100: Historically elite (2012 Kentucky tier)
- 85+: Title contender
- 70–85: Top-5 seed
- 55–70: Tournament team
- 40–55: Bubble
- <40: NIT

---

### resume_score — Quality-Wins Composite

**Category:** CAGE Composite (Torvik resume equivalent)

**Formula:** Rewards Q1 wins, win rate vs good teams, road success. Penalizes bad losses.

**Range:** 0–100

**Interpretation:**
- >75: Top-10 resume
- >60: Comfortable tournament profile
- 50: Bubble-quality
- <40: Questionable résumé

---

### suffocation — Defensive Composite

**Category:** CAGE Composite

**Formula:** Composite of opponent eFG% suppression, defensive rebounding rate, and adjusted DRTG.

**Range:** 0–100

**Interpretation:**
- >75: Elite shutdown defense (low-scoring game predictor)
- 50: Average D1 defense
- <35: Porous defense

---

### clutch_rating — Close-Game Excellence

**Category:** CAGE Composite

**Definition:** Performance in games decided by ≤5 points, adjusted for luck in close-game outcomes.

**Range:** 0–100

**Interpretation:**
- >70: Consistently closes games (valuable in tournament settings)
- 50: Average
- <35: Struggles in close games

---

### consistency_score — Reliability Rating

**Category:** CAGE Composite

**Formula:** Inverse of net efficiency variance and shooting variance across the season.

**Range:** 0–100

**Interpretation:**
- 100: Machine-like consistency (Virginia under Bennett)
- 50: Average
- <35: Boom-or-bust (dangerous to pick in single-elimination)

---

### floor_em / ceiling_em — Performance Range

**Category:** Range

| Metric | Formula | Description |
|--------|---------|-------------|
| `floor_em` | `cage_em − 1.5σ` | Downside performance estimate — the lower bound of the range that contains ~87% of game outcomes |
| `ceiling_em` | `cage_em + 1.5σ` | Upside performance estimate — the upper bound of that same range |

**Gap interpretation:** Wide gap = boom-or-bust team. Narrow gap = reliable performer.

---

### trend_arrow — Directional Momentum

**Category:** Trend

**Formula:**
```
delta = net_rtg_l5 - net_rtg_l10
```

| Arrow | Delta | Meaning |
|-------|-------|---------|
| ↑↑ SURGE | +5 or more | Dramatically improving |
| ↑ UP | +2 to +5 | Trending upward |
| → FLAT | ±2 | No significant trend |
| ↓ DOWN | -2 to -5 | Trending downward |
| ↓↓ SLIDE | -5 or worse | Significant decline |

---

### luck — Luck Score

**Category:** Schedule

**Formula:**
```
luck = actual_win_pct - pythagorean_win_pct
```

**Range:** -0.15 to +0.15

**Interpretation:** Positive = winning more than efficiency predicts (lucky). Teams with high luck tend to regress toward Pythagorean expectation. Primary input to `LuckRegressionModel` (M7).

---

## 3. Performance vs Expectation (PEI)

> **PEI = Performance vs Expectation Index** — the canonical example metric for how this system documents new analytics.
> Measures how much a team exceeded or fell short of what the opponent typically allows/forces.

**File:** `espn_weighted_metrics.py` → `add_performance_vs_expectation()`
**Output columns:** `perf_vs_exp_ortg`, `perf_vs_exp_drtg`, `perf_vs_exp_def`, `perf_vs_exp_net`

---

### perf_vs_exp_ortg — Offensive PEI

**Formula:**
```
perf_vs_exp_ortg = ortg - opp_avg_drtg_season
```

**Interpretation:**
- Positive: Team scored more efficiently than the opponent typically allows → genuine offensive performance
- Negative: Team scored below what the opponent typically concedes → poor offensive day

**Range:** -20 to +25 pts/100 possessions

---

### perf_vs_exp_def — Defensive PEI

**Formula:**
```
perf_vs_exp_def = -(drtg - opp_avg_ortg_season)
```
*(Sign-flipped so positive = good defense)*

**Interpretation:**
- Positive: Team held opponent below what they typically score → genuine defensive performance
- Negative: Opponent scored above their typical rate against this team

**Range:** -20 to +20 pts/100 possessions

---

### perf_vs_exp_net — Net PEI

**Formula:**
```
perf_vs_exp_net = net_rtg - opp_avg_net_rtg_season
```

**Interpretation:**
- The single best "did this team beat the opponent they faced?" metric.
- Rolling average (`perf_vs_exp_net_l5`) is a key input to `form_rating`.

**Range:** -25 to +25 pts/100 possessions

---

### Rolling PEI Windows

All three PEI columns have rolling means computed:

| Column | Description |
|--------|-------------|
| `perf_vs_exp_ortg_l5` | 5-game rolling avg of offensive PEI |
| `perf_vs_exp_ortg_l10` | 10-game rolling avg of offensive PEI |
| `perf_vs_exp_def_l5` | 5-game rolling avg of defensive PEI |
| `perf_vs_exp_def_l10` | 10-game rolling avg of defensive PEI |
| `perf_vs_exp_net_l5` | 5-game rolling avg of net PEI |
| `perf_vs_exp_net_l10` | 10-game rolling avg of net PEI |

All rolling windows use `shift(1)` (leak-free — game N uses only games 1..N-1).

---

### form_rating — Schedule-Adjusted Form

**Formula:**
```
form_rating = 0.6 × net_rtg_wtd_qual_l5 + 0.4 × perf_vs_exp_net_l5
```

**Interpretation:** High score = performing well AND doing it against quality opponents. Best single pre-game form indicator.

---

### momentum_score — Trend Composite

**Formula:**
```
momentum_score = net_rtg_wtd_qual_l5 - net_rtg_wtd_qual_l10
```

**Interpretation:** Positive = improving trend. Negative = declining. Directly feeds `MomentumModel` (M4).

---

## 4. Opponent-Weighted Rolling Metrics

**File:** `espn_weighted_metrics.py` → `add_weighted_rolling()`
**Output:** `data/team_game_weighted.csv`

**Philosophy:** Equal-weight rolling averages treat all games equally. Scoring 110 ORTG against a 95-DRTG defense is NOT the same as 110 against a 115-DRTG defense. Quality-weighted rolling surfaces true performance signal.

### Weight Schemes

Three weight schemes applied per metric, per rolling window:

| Suffix | Weight Formula | Applied To |
|--------|---------------|------------|
| `_wtd_off_l{N}` | `w = LEAGUE_AVG_DRTG / opp_DRTG` | Offensive metrics |
| `_wtd_def_l{N}` | `w = opp_ORTG / LEAGUE_AVG_ORTG` | Defensive metrics |
| `_wtd_qual_l{N}` | `w = (opp_NetRTG + 20) / 20` | Two-way / general |

Weights are clipped to [0.5, 2.0] and normalized within each window.
Dead-spread games are down-weighted to 0.3 to reduce garbage-time noise.

### Offensive Metrics (weighted by opponent defensive quality)

`ortg`, `efg_pct`, `ts_pct`, `fg_pct`, `three_pct`, `h1_ortg`, `h2_ortg`, `points_for`, `fgm`, `fga`, `tpm`, `tpa`, `ftm`, `fta`, `ast`, `three_par`

Example: `ortg_wtd_off_l5`, `efg_pct_wtd_off_l10`

### Defensive Metrics (weighted by opponent offensive quality)

`drtg`, `h1_drtg`, `h2_drtg`, `points_against`, `stl`, `blk`, `tov_pct`

Example: `drtg_wtd_def_l5`, `drtg_wtd_def_l10`

### Quality Metrics (weighted by overall opponent quality)

`net_rtg`, `margin`, `margin_capped`, `orb_pct`, `drb_pct`, `poss`, `pace`, `h1_margin`, `h2_margin`, `cover_margin`, `win`, `cover`

Example: `net_rtg_wtd_qual_l5`, `net_rtg_wtd_qual_l10`

### cover_wtd_qual_rate

**Formula:** Weighted cover rate where covering vs ranked opponents counts more.

| Column | Description |
|--------|-------------|
| `cover_wtd_qual_rate_l5` | Quality-weighted ATS rate over last 5 games |
| `cover_wtd_qual_rate_l10` | Quality-weighted ATS rate over last 10 games |

---

## 5. Strength of Schedule (SOS) Metrics

**File:** `espn_sos.py` → `compute_sos_metrics()`
**Output:** `data/team_game_sos.csv`

### Opponent Profile Columns

Computed for three windows each: `_season`, `_l5`, `_l10`

| Column | Description |
|--------|-------------|
| `opp_avg_ortg_{window}` | Average ORTG of opponents faced |
| `opp_avg_drtg_{window}` | Average DRTG of opponents faced |
| `opp_avg_net_rtg_{window}` | Average NetRTG of opponents faced — **primary SOS number** |
| `opp_avg_efg_pct_{window}` | Average eFG% of opponents faced |
| `opp_avg_pace_{window}` | Average pace of opponents faced |

### Opponent-Context Performance Columns

Team performance relative to what opponents allow/force. Computed for `_season`, `_l5`, `_l10`:

| Column | Formula | Description |
|--------|---------|-------------|
| `efg_vs_opp_allow` | team_efg − avg_efg_opponents_allow | Shooting above defensive expectation |
| `orb_vs_opp_allow` | team_orb − avg_orb_opponents_allow | Offensive rebounding above expectation |
| `drb_vs_opp_allow` | team_drb − avg_drb_opponents_allow | Defensive rebounding above expectation |
| `tov_vs_opp_force` | team_tov − avg_tov_opponents_force | Turnover rate vs opponent forcing ability |
| `ftr_vs_opp_allow` | team_ftr − avg_ftr_opponents_allow | Foul drawing above defensive expectation |

### Adjusted Ratings

| Column | Formula | Description |
|--------|---------|-------------|
| `adj_ortg` | ortg × (LEAGUE_AVG_DRTG / opp_DRTG) | ORTG adjusted for opponent defensive quality |
| `adj_drtg` | drtg × (LEAGUE_AVG_ORTG / opp_ORTG) | DRTG adjusted for opponent offensive quality |
| `adj_net_rtg` | adj_ortg − adj_drtg | Net adjusted efficiency margin |
| `adj_pace` | pace × (LEAGUE_AVG_PACE / opp_pace) | Pace normalized by opponent tempo context |

### Quad Records

NCAA Tournament committee-style opponent quality tiers:

| Quad | Opponent net_rtg threshold | Meaning |
|------|---------------------------|---------|
| Q1 | ≥ +8.0 | Elite opponents (top ~25% of D1) |
| Q2 | 0 to +8.0 | Above average |
| Q3 | -8.0 to 0 | Below average |
| Q4 | < -8.0 | Weak bottom tier |

---

## 6. Core Per-Game Metrics

**File:** `espn_metrics.py` → `compute_all_metrics()`
**Output:** `data/team_game_metrics.csv`

### Efficiency Metrics (per game)

| Metric | Formula | Description |
|--------|---------|-------------|
| `poss` | `fga − orb + tov + 0.44×fta` (averaged with opponent) | Possessions — both teams averaged to reconcile discrepancies |
| `ortg` | `(points_for / poss) × 100` | Offensive rating — points per 100 possessions |
| `drtg` | `(points_against / poss) × 100` | Defensive rating — lower is better |
| `net_rtg` | `ortg − drtg` | Net efficiency |
| `pace` | `poss / (minutes / 40)` | Possessions per 40 minutes |

### Shooting Metrics (per game)

| Metric | Formula | Description |
|--------|---------|-------------|
| `efg_pct` | `(fgm + 0.5×tpm) / fga` | Effective FG% — accounts for 3pt bonus |
| `ts_pct` | `pts / (2 × (fga + 0.44×fta))` | True shooting % — includes free throws |
| `fg_pct` | `fgm / fga` | Raw FG% |
| `three_pct` | `tpm / tpa` | 3-point % |
| `ft_pct` | `ftm / fta` | Free throw % |
| `three_par` | `tpa / fga` | 3-point attempt rate |
| `ftr` | `fta / fga` | Free throw rate |

### Rebound Metrics (per game)

| Metric | Formula | Description |
|--------|---------|-------------|
| `orb_pct` | `orb / (orb + opp_drb)` | Offensive rebound % of available boards |
| `drb_pct` | `drb / (drb + opp_orb)` | Defensive rebound % of available boards |
| `tov_pct` | `tov / poss × 100` | Turnover rate per 100 possessions |

### Half-Split Metrics

| Metric | Description |
|--------|-------------|
| `h1_ortg` / `h2_ortg` | First/second half offensive rating |
| `h1_drtg` / `h2_drtg` | First/second half defensive rating |
| `h1_margin` / `h2_margin` | Scoring margin by half |

### Win/Cover Metrics

| Metric | Description |
|--------|-------------|
| `win` | Binary win flag (1 = won) |
| `cover` | Binary ATS cover flag (1 = covered the spread) |
| `cover_margin` | Margin of victory minus spread |
| `ats_result` | `COVER`, `PUSH`, or `LOSS` |

### Schedule/Fatigue Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `rest_days` | Days since prior game | Primary input to SituationalModel |
| `games_l7` | Games in last 7 days | Schedule density indicator |
| `games_l14` | Games in last 14 days | Medium-term load |
| `fatigue_index` | Derived from rest_days + density | Composite scheduling fatigue signal |

### Streak Metrics (rolling, leak-free)

| Metric | Description |
|--------|-------------|
| `win_streak` | Current consecutive win count (0 if on losing streak) |
| `lose_streak` | Current consecutive loss count |
| `cover_streak` | Current consecutive ATS cover count |

### Game Flags

| Flag | Condition | Use |
|------|-----------|-----|
| `blowout_flag` | `abs(margin) ≥ 15` | Down-weight in efficiency calculations |
| `close_game_flag` | `abs(margin) ≤ 5` | Input to clutch metrics |
| `dead_spread_flag` | Garbage time detected | Reduced weight (0.3×) in weighted rolling |

### Pythagorean Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `pythagorean_win_pct` | `pts^11.5 / (pts^11.5 + opp_pts^11.5)` | Expected win% from scoring ratios |
| `luck_score` | `actual_win_pct − pythagorean_win_pct` | Luck indicator per game |

### Rolling Windows (L5 / L10)

All per-game metrics above have rolling window variants: `{metric}_l5` and `{metric}_l10`.

All use `shift(1)` within grouped (`team_id`) rolling — fully leak-free. No future data leaks into any rolling column.

---

## 7. Tournament Composite Metrics

**File:** `espn_tournament.py` → `compute_tournament_metrics()`
**Output prefix:** `t_` (distinguishes from raw box-score columns)

---

### t_tournament_dna_score — Tournament DNA Index

**Alias in rankings:** `dna_score`

**Definition:** Composite capturing historical behavioral markers that predict tournament success: road wins, close wins, elite opponent wins.

**Formula:** Weighted composite of Q1 road wins, close-game win rate, top-opponent performance, and defensive efficiency.

**Range:** 0–100

**Interpretation:**
- >70: Demonstrated performance specifically in tournament-like conditions
- 50: Average
- <35: No evidence of surviving tournament pressure

---

### t_suffocation_rating — Defensive Suffocation

**Alias in rankings:** `suffocation`

**Definition:** How completely this defense shuts down opponents.

**Formula:** Z-score composite of `opp_efg_pct` (suppression), `drb_pct`, and `adj_drtg`.

**Range:** 0–100

---

### t_momentum_quality_rating — Momentum Quality

**Alias in rankings:** `momentum`

**Definition:** Recent form weighted by opponent quality. A 5-game win streak vs weak opponents scores lower than 3 wins vs top-25.

**Formula:** Blended z-score of recent net rating trend + opponent quality context.

**Range:** 0–100

**Interpretation:**
- >80: Surging through elite competition
- >65: Meaningfully hot
- 50: Neutral

---

### t_star_reliance_risk — Star Player Fragility

**Alias in rankings:** `star_risk`

**Definition:** Roster fragility if top player struggles or fouls out.

**Formula:** Uses player-level minutes/usage concentration when available; falls back to box-score-derived estimate.

**Range:** 0–100

**Interpretation:**
- >70: One player carries the team — high tournament variance
- 50: Balanced
- <30: Deep, reliable roster

---

### t_offensive_identity_score — Offensive System Cohesion

**Alias in rankings:** `off_identity`

**Definition:** Clarity and execution of offensive system. High = team executes a clear, repeatable offensive scheme.

**Formula:** Composite of assist rate, three-point rate consistency, and free throw generation pattern.

**Range:** 0–100

---

### Game Total Projection

**Function:** `project_game_total()`

**Formula:**
```
projected_total = (home_adj_pace + away_adj_pace) / 2 
                × (home_adj_ortg + away_adj_drtg) / (2 × 100)
                × (home_adj_drtg + away_adj_ortg) / (2 × 100)
                × tournament_multiplier
```

Slow team's pace weighted 53% vs fast team's 47% (defensive tempo drag).

**Outputs:**
- `game_total_projection`: Projected combined points
- `total_confidence`: Confidence interval (±pts) based on variance flags

---

### t_underdog_winner_score (UWS) — Upset Probability

**Definition:** Composite 0–70 score for the lower-seeded / higher-ML team predicting upset likelihood.

**Components:**

| Component | Max Points | Description |
|-----------|-----------|-------------|
| Seed matchup | 10 | Favorable seed tier for upset history |
| Recent form | 15 | Hot streak relative to favorite |
| Defensive suffocation | 15 | Can they limit a superior offense? |
| Tournament DNA | 15 | Historical tournament performance markers |
| Star risk | 8 | Favorite's fragility |
| Scheduling | 7 | Rest/fatigue differential |

**Thresholds:**
- ≥55: `STRONG_UPSET_ALERT`
- ≥45: `LEGITIMATE_UPSET_THREAT`
- ≥35: `MILD_UPSET_THREAT`

**Derived:** `uwp_upset_probability` — implied probability from UWS score.

---

## 8. Analysis & Intelligence Tools

### Alpha Evaluator

**File:** `models/alpha_evaluator.py` → `evaluate_alpha()`

**Purpose:** Single source of truth for edge detection, alpha classification, and Kelly stake sizing.

**Called at two pipeline points:**
1. `espn_prediction_runner.py` — at prediction time (base alpha, no market data)
2. `predictions_with_context.py` — after market data merge (final alpha, overrides #1)

#### Kelly Fraction

**Formula:** Quarter-Kelly using edge-adjusted win probability:

```
edge_pts → base_win_prob mapping:
  0.0 pts → 50.0%
  1.0 pts → 51.5%
  2.0 pts → 53.0%
  3.0 pts → 54.5%
  5.0 pts → 57.0%
  7.0+ pts → 59.0% (intentional model cap — conservative calibration)

Kelly_f = max(0, (win_prob - (1-win_prob)/decimal_odds)) / 4
Scaled_f = Kelly_f × model_confidence × multiplier
```

**Output:** `kelly_fraction` — recommended stake size as fraction of bankroll.

#### Edge Classification

| Alpha Level | Condition |
|-------------|-----------|
| `STRONG` | Edge ≥ 3.5 pts AND confidence ≥ 65% |
| `MODERATE` | Edge ≥ 2.0 pts AND confidence ≥ 55% |
| `MARGINAL` | Edge ≥ 1.0 pts |
| `NO_EDGE` | Below thresholds |

---

### CLV Analyzer — Closing Line Value

**File:** `models/clv_analyzer.py`
**Outputs:** `data/clv_report.csv`, `data/clv_by_submodel.csv`

**Definition:** Closing Line Value — the gold standard for evaluating whether a model's predictions beat the market's final consensus.

**Metrics produced:**

| Metric | Description |
|--------|-------------|
| `clv_vs_open` | Model spread − opening line |
| `clv_vs_close` | Model spread − closing line |
| `pct_positive_clv` | % of games where model beat the close |
| `mean_abs_error` | Average absolute prediction error |
| `correlation_clv_to_outcome` | How predictive CLV was for actual ATS outcomes |

**Sub-model CLV** computed separately for each ensemble model (`ens_fourfactors_spread`, `ens_adjefficiency_spread`, etc.).

---

### Bias Detector

**File:** `models/bias_detector.py`

**Purpose:** Detect systematic over/under-prediction by dimension (conference, game type, spread bucket, opponent tier).

**Dimensions analyzed:**
- Conference
- Game type (home/away/neutral)
- Spread bucket (size of opening line)
- Opponent quality tier (Q1–Q4)

**Key outputs:**

| Column | Description |
|--------|-------------|
| `mean_signed_error` | Direction of systematic bias |
| `correction` | Recommended adjustment (capped at `MAX_CORRECTION = 3.0 pts`) |
| `actionable` | 1 if enough sample (≥20 games) AND large enough error (≥1.5 pts) |
| `direction_consistent` | Whether bias is stable across first/second half of season |

**Application:** Bias corrections are applied in `_apply_bias_corrections()` within `EnsemblePredictor`.

---

### Player Matchup Overlay

**File:** `models/player_matchup_overlay.py` → `PlayerMatchupOverlay`

**Purpose:** Adjust ensemble spread and total predictions using rotation-level player efficiency data.

**Key input features:**

| Feature | Source | Description |
|---------|--------|-------------|
| `rot_efg_l5` | `rotation_features.csv` | Rotation eFG% last 5 games |
| `rot_to_rate_l5` | `rotation_features.csv` | Rotation turnover rate last 5 |
| `rot_ftrate_l5` | `rotation_features.csv` | Rotation free throw rate last 5 |
| `rot_3par_l10` | `rotation_features.csv` | Rotation 3PA rate last 10 |
| `top2_pused_share` | Derived | Usage share of top 2 players |
| `closer_ft_pct` | Derived | FT% of team's primary closer |

**Adjustment outputs:**
- `total_adj` — total adjustment (pts)
- `spread_adj` — spread adjustment (pts)
- `total_std_adj` — uncertainty added to total
- `spread_std_adj` — uncertainty added to spread

All adjustments capped at `max_total_adj_pts = 10.0` and `max_spread_adj_pts = 8.0`.

---

### Weight Optimizer

**File:** `models/weight_optimizer.py`

**Purpose:** Weekly Nelder-Mead optimization of ensemble sub-model weights from graded backtest results.

**Optimization targets:**
- Minimize ATS MAE (spread)
- Minimize O/U MAE (total)

**Constraints:**
- All weights ≥ `MIN_WEIGHT = 0.05`
- Weights sum to 1.0

**Walk-forward protocol:** Train on first 70% of sample, evaluate on last 30%.

**Output:** `data/backtest_optimized_weights.json` → loaded by `EnsembleConfig.from_optimized()`

---

## 9. Supplemental Feature Sets

### Rotation Features

**File:** `cbb_rotation_features.py`
**Output:** `data/rotation_features.csv`

Per-team per-game aggregation of rotation-level efficiency, compiled from `player_game_logs.csv` + `player_game_metrics.csv`.

| Column | Description |
|--------|-------------|
| `rot_efg_l5` / `_l10` | Rotation effective FG% rolling avg |
| `rot_to_rate_l5` / `_l10` | Rotation turnover rate rolling avg |
| `rot_ftrate_l5` | Rotation free throw rate rolling avg |
| `rot_3par_l10` | Rotation 3-point attempt rate rolling avg |
| `rot_stocks_per40_l10` | Rotation steals + blocks per 40 min |
| `rot_pf_per40_l5` | Rotation personal fouls per 40 min |
| `rot_minshare_sd` | Std dev of minute share across rotation |
| `rot_3p_pct_sd` | Std dev of 3pt% within rotation |
| `rot_to_rate_sd` | Std dev of turnover rate within rotation |

---

### Player Availability Features

**File:** `player_availability_features.py`
**Output:** `data/player_availability_features.csv`

Quantifies how available a team's rotation is for the upcoming game.

| Column | Description |
|--------|-------------|
| `minutes_available_pct` | Team minutes vs prior 5-game baseline |
| `top_player_usage_share` | Fraction of possessions owned by top player |
| `usage_gini` | Gini coefficient of usage distribution (high = concentrated) |
| `n_eligible_rotation` | Players with >0 minutes and not DNP |
| `injury_burden_proxy` | Usage-weighted impact of potentially unavailable players |

---

### Luck Regression Features

**File:** `cbb_luck_regression_features.py`
**Output:** `data/luck_regression_features.csv`

Quantifies how much a team's results deviate from efficiency-implied expectations.

| Column | Description |
|--------|-------------|
| `luck_index` | Rolling pythagorean luck (actual − expected wins) |
| `three_luck_l10` | Three-point percentage vs prior season baseline |
| `ft_luck_l10` | Free throw luck vs baseline |
| `net_rtg_zscore` | Current net rating z-score vs expanding season mean |
| `regression_probability` | Estimated probability of mean reversion |

---

### Line Movement Features

**File:** `cbb_line_movement_features.py`
**Output:** `data/line_movement_features.csv`

One-row-per-game summary of betting market movement from opening to close.

| Column | Description |
|--------|-------------|
| `spread_open` | Opening line |
| `spread_close` | Closing line |
| `spread_move` | Close minus open (positive = moved toward favorite) |
| `crossed_key_number` | 1 if line crossed 3, 6, 7, or 10 (key spread numbers) |
| `steam_flag` | 1 if rapid sharp-money line movement detected |
| `rlm_flag` | 1 if reverse line movement (public on one side, line moved opposite) |
| `sharp_side` | `home`/`away` — side sharp money is on |
| `public_side` | `home`/`away` — side with majority of tickets |

---

### Calibration

**File:** `calibrate_confidence.py` → `fit_calibration()`
**Output:** `data/calibration_params.json`

Fits isotonic regression calibration curve mapping raw model confidence (50–100%) → empirically observed ATS win rate.

| Output field | Description |
|--------------|-------------|
| `mean_overconfidence` | How many percentage points the model overstates confidence on average |
| `calibration_table` | Grid of raw_confidence → calibrated_probability values |
| `iso_x_thresholds` / `iso_y_thresholds` | Isotonic regression knot points for `np.interp()` application |

**Application:** `apply_calibration()` in `cbb_ensemble.py` transforms raw ensemble confidence using saved knots.

---

*Last updated: auto-generated from pipeline source. See individual source files for full implementation details.*
