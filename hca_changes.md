# HCA Changes Audit (Agent 4)

## Summary
Home/away splits now provide the primary venue signal, so flat HCA values were reduced to a **residual +1.0** where HCA was still hard-coded.

## Files updated

1. **`cbb_config.py`**
   - Old HCA: `3.2`
   - New HCA: `1.0`
   - Change: `HCA = 1.0  # Residual HCA after home/away splits`

2. **`cbb_prediction_model.py`**
   - Old default HCA: `default_hca = 3.2`
   - New default HCA: `default_hca = 1.0`
   - Flat fallback changed from hard-coded `3.2` to `cfg.default_hca` with comment:
     - `# Residual HCA after home/away splits`

3. **`espn_prediction_runner.py`**
   - Old fallback HCA: `hca = 0.0 if neutral_site else 3.5`
   - New fallback HCA: `hca = 0.0 if neutral_site else 1.0`
   - Comment added:
     - `# Residual HCA after home/away splits`

4. **`cbb_ensemble.py`**
   - Old flat adders: `margin += HCA * 0.5` (two locations)
   - New adders: `margin += HCA` (two locations)
   - Since `HCA` is now `1.0`, these become a +1.0 residual adjustment.
   - Comment added:
     - `# Residual HCA after home/away splits`

## Intercepts reviewed
`intercept` values in logistic/calibration code were reviewed and left unchanged because they are learned model parameters, not fixed HCA constants.

## Rationale
- Home/away splits capture most location effects.
- Keep only a conservative residual HCA term to avoid double-counting location.

## TODO
- Backtest residual HCA calibration (`0.5`, `1.0`, `1.5`) by season and market type.
- Track calibration drift in `predictions` by `model_version` after rollout.
