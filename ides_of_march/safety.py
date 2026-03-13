from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SafetyStatus:
    status: str
    issues: list[str]
    metrics: dict[str, Any]


def run_data_integrity_audit(
    upcoming_games: pd.DataFrame,
    historical_games: pd.DataFrame,
    *,
    min_market_coverage: float = 0.6,
) -> SafetyStatus:
    issues: list[str] = []
    status = "PASS"

    metrics = {
        "upcoming_rows": int(len(upcoming_games)),
        "historical_rows": int(len(historical_games)),
    }

    if upcoming_games.empty:
        # Empty upcoming horizon is a valid no-slate state (off days / windows
        # between game blocks). Keep this non-blocking so contract stages can
        # complete and downstream outputs can remain empty by design.
        status = "WARN"
        issues.append("No upcoming games found in active horizon.")
        metrics["no_upcoming_horizon"] = True
        return SafetyStatus(status=status, issues=issues, metrics=metrics)

    required_cols = ["event_id", "game_id", "home_team_id", "away_team_id", "game_datetime_utc"]
    missing = sorted(set(required_cols) - set(upcoming_games.columns))
    if missing:
        status = "BLOCKED"
        issues.append(f"Upcoming frame missing required columns: {missing}")

    market_coverage = float(pd.to_numeric(upcoming_games.get("market_spread"), errors="coerce").notna().mean())
    metrics["market_coverage"] = market_coverage
    if market_coverage < min_market_coverage:
        if status == "PASS":
            status = "WARN"
        issues.append(
            f"Market line coverage below threshold: {market_coverage:.3f} < {min_market_coverage:.3f}"
        )

    duplicate_event_ids = int(upcoming_games.get("event_id", pd.Series(dtype=str)).duplicated().sum())
    metrics["duplicate_event_ids"] = duplicate_event_ids
    if duplicate_event_ids > 0:
        status = "FAIL"
        issues.append(f"Upcoming frame has duplicate event_id rows: {duplicate_event_ids}")

    if historical_games.empty or len(historical_games) < 120:
        if status == "PASS":
            status = "WARN"
        issues.append("Historical sample is thin; calibration and situational stats may be unstable.")

    return SafetyStatus(status=status, issues=issues, metrics=metrics)


def run_model_safety_audit(predictions: pd.DataFrame) -> SafetyStatus:
    issues: list[str] = []
    status = "PASS"

    confidence = pd.to_numeric(predictions.get("confidence_score"), errors="coerce")
    edge_abs = pd.to_numeric(predictions.get("edge_home"), errors="coerce").abs()
    win_prob = pd.to_numeric(predictions.get("win_prob_home"), errors="coerce")

    metrics = {
        "rows": int(len(predictions)),
        "confidence_gt90": int((confidence > 90).sum()) if len(confidence) else 0,
        "mean_confidence": float(confidence.mean()) if len(confidence) else np.nan,
    }

    if len(predictions) == 0:
        return SafetyStatus(status="BLOCKED", issues=["No predictions to audit."], metrics=metrics)

    if ((confidence < 0) | (confidence > 100)).any():
        status = "FAIL"
        issues.append("Confidence values fall outside [0, 100].")

    if ((win_prob < 0) | (win_prob > 1)).any():
        status = "FAIL"
        issues.append("Win probabilities fall outside [0, 1].")

    overconf = (confidence >= 85) & (edge_abs <= 1.0)
    overconf_rate = float(overconf.mean()) if len(overconf) else 0.0
    metrics["overconfidence_rate"] = overconf_rate
    if overconf_rate > 0.10:
        status = "WARN" if status == "PASS" else status
        issues.append(f"Potential overconfidence detected: {overconf_rate:.1%} of rows have high confidence with low edge.")

    if "home_Last5_AdjEM" in predictions.columns:
        null_rate = float(predictions["home_Last5_AdjEM"].isna().mean())
        metrics["home_Last5_AdjEM_null_rate"] = null_rate
        if null_rate > 0.5:
            status = "WARN" if status == "PASS" else status
            issues.append("High null rate in rolling form features; verify data depth before release.")

    return SafetyStatus(status=status, issues=issues, metrics=metrics)
