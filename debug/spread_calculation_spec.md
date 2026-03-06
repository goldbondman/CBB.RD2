# Spread Calculation Confirmation

Model: `joint_v1.0`
Perspective: HOME (positive margin means home favored).

## Exact Formula
1. `signal_raw = sum((feature_i / scale_i) * weight_i) / sum(weight_i_present)`
2. `allocation_raw = tanh(signal_raw) * 0.08`
3. `allocation_final = clip(allocation_raw, -0.2, 0.2)`
4. `pred_margin_raw = pred_total * 2 * allocation_final + intercept + hca_points`
5. `pred_margin_final = pred_margin_raw` unless a guardrail blocks the row.
Legacy (pre-fix) reference: `pred_margin_legacy = pred_total * 2 * clip(unscaled_signal_raw, -0.2, 0.2)`.

## Intercept / HCA
- intercept = 0.0
- hca_points = 0.0

## Feature Scales
- `away_efg_diff` scale = 0.08
- `lns_diff` scale = 20.0
- `odi_star_diff` scale = 0.15
- `posw` scale = 0.35
- `pxp_diff` scale = 0.2
- `rfd` scale = 3.0
- `sme_diff` scale = 0.15
- `vol_diff` scale = 12.0

## Percent Scaling Convention
- Percent-like source columns are read as either `0-1` or `0-100`.
- Normalization rule before metric construction: any value `> 1.5` is divided by `100`.

## Notes
- No hard-coded spread fallback constants are used.
- BLOCKED rows keep `pred_margin` as blank/NaN.
