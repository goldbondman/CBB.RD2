from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.logging_config import get_logger

log = get_logger(__name__)

CRITICAL_OUTPUT_SCHEMAS: dict[str, list[str]] = {
    "team_game_metrics": ["event_id", "team_id", "efg_pct", "fgm", "fga", "tpm", "tpa"],
    "predictions_combined_latest": [
        "event_id",
        "predicted_spread",
        "model_confidence",
        "home_team_id",
        "away_team_id",
        "home_conference",
        "away_conference",
        "model1_schedule_pred",
        "model2_four_factors_pred",
    ],
    "predictions_latest": ["event_id", "predicted_spread", "model_confidence", "home_team_id", "away_team_id"],
    "results_log": ["event_id", "predicted_spread", "actual_margin"],
}


def validate_and_write(
    df: pd.DataFrame,
    path: Path,
    required_cols: list[str],
    label: str,
    *,
    index: bool = False,
) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log.error(f"{label} missing required columns: {missing}")
        raise ValueError(f"Schema validation failed for {label}")

    null_critical = [c for c in required_cols if df[c].isna().all()]
    if null_critical:
        log.warning(
            f"{label}: these required columns are ALL null: "
            f"{null_critical} — possible upstream failure"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    log.info(f"✓ {label} → {path} ({len(df)} rows, {len(df.columns)} columns)")
