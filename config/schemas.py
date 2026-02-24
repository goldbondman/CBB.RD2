from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

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
    from pipeline_csv_utils import normalize_column_names

    df = normalize_column_names(df)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log.error(f"{label} missing required columns: {missing}")
        dq_path = Path("data") / "dq_audit.csv"
        dq_path.parent.mkdir(parents=True, exist_ok=True)
        dq_row = pd.DataFrame(
            [
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "entity_type": label,
                    "severity": "error",
                    "reason_codes": "missing_required_columns",
                    "details": f"missing={missing}",
                    "target_path": str(path),
                }
            ]
        )
        if dq_path.exists():
            dq_existing = pd.read_csv(dq_path)
            dq_row = pd.concat([dq_existing, dq_row], ignore_index=True)
        dq_row.to_csv(dq_path, index=False)
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
