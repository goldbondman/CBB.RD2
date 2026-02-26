import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

MIN_SAMPLE = 100

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # graceful fallback in lightweight environments
    IsotonicRegression = None


def fit_calibration(df: pd.DataFrame) -> dict:
    sub = df[["model_confidence", "home_covered_pred"]].dropna().copy()
    sub["model_confidence"] = pd.to_numeric(sub["model_confidence"], errors="coerce")
    sub["home_covered_pred"] = pd.to_numeric(sub["home_covered_pred"], errors="coerce")
    sub = sub.dropna()

    if IsotonicRegression is None:
        log.warning("scikit-learn not installed — returning identity calibration")
        return {
            "calibrated": False,
            "sample_n": len(sub),
            "note": "scikit-learn missing — no correction applied",
            "identity": True,
        }

    if len(sub) < MIN_SAMPLE:
        log.warning("Only %s samples (need %s) — returning identity calibration", len(sub), MIN_SAMPLE)
        return {
            "calibrated": False,
            "sample_n": len(sub),
            "note": "Insufficient data — no correction applied",
            "identity": True,
        }

    X = sub["model_confidence"].values.astype(float)
    y = sub["home_covered_pred"].values.astype(float)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X, y)

    conf_grid = np.arange(50, 100, 5).astype(float)
    cal_values = iso.predict(conf_grid)

    calibration_table = [
        {
            "raw_confidence": int(c),
            "calibrated_probability": round(float(p) * 100, 1),
            "overconfidence_gap": round(float(c) - float(p) * 100, 1),
        }
        for c, p in zip(conf_grid, cal_values)
    ]

    mean_raw = float(X.mean())
    mean_cal = float(iso.predict(X).mean() * 100)
    overconf = round(mean_raw - mean_cal, 2)

    log.info("Calibration fit on %s samples", len(sub))
    log.info("Mean raw confidence: %.1f%%", mean_raw)
    log.info("Mean calibrated: %.1f%%", mean_cal)
    log.info("Model is %s by %.1fpp on average", "OVERCONFIDENT" if overconf > 0 else "UNDERCONFIDENT", abs(overconf))

    return {
        "calibrated": True,
        "sample_n": len(sub),
        "mean_overconfidence": overconf,
        "calibration_table": calibration_table,
        "iso_x_thresholds": iso.X_thresholds_.tolist(),
        "iso_y_thresholds": [round(v * 100, 2) for v in iso.y_thresholds_.tolist()],
        "fitted_at_utc": pd.Timestamp.now('UTC').isoformat(),
        "note": "Apply via np.interp(raw_50_100, iso_x_thresholds, iso_y_thresholds)",
    }


def apply_calibration_example(raw_confidence: float, cal_data: dict) -> float:
    """Example of applying calibration. Input raw_confidence is 0-1 scale."""
    if not cal_data.get("calibrated"):
        return raw_confidence

    iso_x = cal_data["iso_x_thresholds"]
    iso_y = cal_data["iso_y_thresholds"]

    # Input may be 0.65 (0-1) or 65 (50-100). Standardize to 50-100 for interpolation.
    raw_scaled = raw_confidence * 100 if raw_confidence <= 1.0 else raw_confidence

    cal_scaled = float(np.interp(raw_scaled, iso_x, iso_y))
    return cal_scaled / 100


def main() -> None:
    try:
        from espn_config import OUT_PREDICTIONS_GRADED as graded_path, OUT_CONFIDENCE_CALIBRATION as cal_path
    except ImportError:
        graded_path = Path("data/predictions_graded.csv")
        cal_path = Path("data/confidence_calibration.json")

    graded_path = Path(graded_path)
    cal_path = Path(cal_path)

    if not graded_path.exists():
        log.warning("predictions_graded.csv not found — skipping")
        return

    df = pd.read_csv(graded_path, dtype={"event_id": str})

    if "graded" in df.columns:
        graded = df[df["graded"] == True].copy()
    else:
        graded = df.copy()

    log.info("Graded predictions loaded: %s", len(graded))

    cal_data = fit_calibration(graded)
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps(cal_data, indent=2))
    log.info("confidence_calibration.json → %s", cal_path)


if __name__ == "__main__":
    main()
