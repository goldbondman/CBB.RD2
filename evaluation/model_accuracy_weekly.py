import logging
from pathlib import Path

import pandas as pd

from espn_config import (
    OUT_ACCURACY_BY_VERSION,
    OUT_ACCURACY_WEEKLY,
    OUT_PREDICTIONS_COMBINED,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

DATA_DIR = Path("data")


def build_accuracy_outputs(predictions_path: Path = OUT_PREDICTIONS_COMBINED) -> None:
    if not predictions_path.exists():
        log.warning("predictions file missing: %s", predictions_path)
        return

    df = pd.read_csv(predictions_path)
    if "game_datetime_utc" not in df.columns:
        log.warning("game_datetime_utc column missing; skipping weekly accuracy")
        return

    if "abs_spread_error" not in df.columns and "spread_error" in df.columns:
        df["abs_spread_error"] = pd.to_numeric(df["spread_error"], errors="coerce").abs()

    required = ["home_covered_pred", "abs_spread_error", "game_datetime_utc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning("Missing required columns for accuracy outputs: %s", missing)
        return

    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], errors="coerce", utc=True)
    df["home_covered_pred"] = pd.to_numeric(df["home_covered_pred"], errors="coerce")
    df["abs_spread_error"] = pd.to_numeric(df["abs_spread_error"], errors="coerce")
    df = df.dropna(subset=required)

    if df.empty:
        log.warning("No graded rows available after cleaning")
        return

    weekly = (
        df.assign(week_start=df["game_datetime_utc"].dt.to_period("W").dt.start_time)
        .groupby("week_start", as_index=False)
        .agg(
            predictions_n=("home_covered_pred", "size"),
            ats_pct=("home_covered_pred", lambda s: round(s.mean() * 100, 1)),
            mae=("abs_spread_error", lambda s: round(s.mean(), 2)),
        )
    )
    weekly.to_csv(OUT_ACCURACY_WEEKLY, index=False)
    log.info("Wrote weekly accuracy: %s", OUT_ACCURACY_WEEKLY)

    if "model_version_hash" in df.columns:
        version_accuracy = []
        for version, grp in df.groupby("model_version_hash"):
            if len(grp) < 10:
                continue
            version_accuracy.append({
                "model_version_hash": version,
                "predictions_n": len(grp),
                "ats_pct": round(grp["home_covered_pred"].mean() * 100, 1),
                "mae": round(grp["abs_spread_error"].mean(), 2),
                "date_range": (
                    f"{grp['game_datetime_utc'].min().isoformat()[:10]} to "
                    f"{grp['game_datetime_utc'].max().isoformat()[:10]}"
                ),
            })
        if version_accuracy:
            pd.DataFrame(version_accuracy).to_csv(OUT_ACCURACY_BY_VERSION, index=False)
            log.info("Accuracy by model version:")
            for v in version_accuracy:
                log.info(
                    "  v%s: ATS=%s%% MAE=%s n=%s (%s)",
                    v["model_version_hash"],
                    v["ats_pct"],
                    v["mae"],
                    v["predictions_n"],
                    v["date_range"],
                )


if __name__ == "__main__":
    build_accuracy_outputs()
