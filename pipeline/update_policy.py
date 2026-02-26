from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class UpdatePolicyConfig:
    N_BAD_WINDOW_RUNS: int = 5
    N_BAD_REQUIRED: int = 3
    MIN_GAMES_SINCE_LAST_UPDATE: int = 150
    COOLDOWN_DAYS: int = 7
    ROI_DROP_ABS: float = 0.03
    CLV_SPREAD_DROP_PTS: float = 0.25
    BRIER_WORSEN: float = 0.01
    SPREAD_MAE_WORSEN: float = 0.35
    TOTAL_MAE_WORSEN: float = 1.25


DEFAULTS = UpdatePolicyConfig()


def tag_run(eval_row: dict, baseline: dict, cfg: UpdatePolicyConfig = DEFAULTS) -> tuple[str, dict]:
    roi_bad = (eval_row.get("roi_ats", 0.0) < (baseline.get("roi_ats", 0.0) - cfg.ROI_DROP_ABS)) or (
        eval_row.get("roi_ats", 0.0) < -0.02
    )
    clv_bad = (
        eval_row.get("median_clv_spread", 0.0) < -cfg.CLV_SPREAD_DROP_PTS
        or eval_row.get("brier_score", 0.0) > baseline.get("brier_score", 0.0) + cfg.BRIER_WORSEN
    )
    err_bad = (
        eval_row.get("spread_mae", 0.0) > baseline.get("spread_mae", 0.0) + cfg.SPREAD_MAE_WORSEN
        or eval_row.get("total_mae", 0.0) > baseline.get("total_mae", 0.0) + cfg.TOTAL_MAE_WORSEN
    )
    signals = {"roi": roi_bad, "clv_calibration": clv_bad, "error": err_bad}
    bad_count = sum(int(v) for v in signals.values())
    return ("BAD" if bad_count >= 2 else "GOOD"), signals


def load_last_update(path: Path) -> dict:
    if not path.exists():
        return {
            "model_version": None,
            "promotion_timestamp": "1970-01-01T00:00:00+00:00",
            "games_since_update": 0,
            "update_reason": "bootstrap",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_update_eligibility(
    *,
    run_history_path: Path,
    last_update_path: Path,
    force_update: bool,
    override_reason: str | None,
    cfg: UpdatePolicyConfig = DEFAULTS,
) -> dict:
    history = pd.read_csv(run_history_path) if run_history_path.exists() else pd.DataFrame(columns=["run_tag", "graded_games"])
    last_update = load_last_update(last_update_path)

    recent = history.tail(cfg.N_BAD_WINDOW_RUNS)
    bad_recent = int((recent.get("run_tag") == "BAD").sum()) if not recent.empty else 0
    graded_games = int(history.get("graded_games", pd.Series(dtype=int)).sum())

    promoted_at = datetime.fromisoformat(last_update["promotion_timestamp"])
    cooldown_ok = datetime.now(timezone.utc) - promoted_at >= timedelta(days=cfg.COOLDOWN_DAYS)

    eligible = (
        bad_recent >= cfg.N_BAD_REQUIRED
        and graded_games >= cfg.MIN_GAMES_SINCE_LAST_UPDATE
        and cooldown_ok
    )

    reason = "policy_not_met"
    if eligible:
        reason = "policy_met"
    if force_update:
        eligible = True
        reason = f"force_update:{override_reason or 'unspecified'}"

    return {
        "eligible": bool(eligible),
        "reason": reason,
        "bad_runs_recent": bad_recent,
        "window_runs": int(len(recent)),
        "graded_games_since_last_update": graded_games,
        "cooldown_ok": bool(cooldown_ok),
        "config": cfg.__dict__,
    }
