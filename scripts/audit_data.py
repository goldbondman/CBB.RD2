#!/usr/bin/env python3
"""Agent 1: Data Auditor with Gate 1 enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FILES = {
    "advanced_metrics": Path("data/advanced_metrics.csv"),
    "team_snapshot": Path("data/team_snapshot.csv"),
    "market_lines": Path("data/market_lines_latest_by_game.csv"),
    "predictions_joint": Path("data/predictions_joint_latest.csv"),
}

BOX_SCORE_CANDIDATES = {
    "FGA": ["FGA", "fga", "field_goals_attempted"],
    "FGM": ["FGM", "fgm", "field_goals_made"],
    "3PA": ["3PA", "three_pa", "fg3a", "three_point_attempts", "tpa"],
    "3PM": ["3PM", "three_pm", "fg3m", "three_point_made", "tpm"],
    "FTA": ["FTA", "fta", "free_throw_attempts"],
    "FTM": ["FTM", "ftm", "free_throws_made"],
    "OREB": ["OREB", "oreb", "offensive_rebounds", "orb"],
    "DREB": ["DREB", "dreb", "defensive_rebounds", "drb"],
    "TO": ["TO", "to", "turnovers", "tov"],
    "PTS": ["PTS", "pts", "points", "score", "points_for"],
    "OffEff": ["OffEff", "OffRtg", "off_rtg", "offensive_rating", "pts_per_100"],
    "DefEff": ["DefEff", "DefRtg", "def_rtg", "defensive_rating", "opp_pts_per_100"],
    "eFG": ["eFG_pct", "efg", "efg_pct", "effective_fg"],
    "Pace": ["Pace", "pace", "possessions_per_40"],
    "NetRtg": ["NetRtg", "net_rtg", "net_rating"],
}


def _pick_case_insensitive(cols: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in cols}
    for item in candidates:
        hit = lookup.get(item.lower())
        if hit:
            return hit
    return None


def _to_jsonable_sample(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    sample = df.head(2).copy()
    return sample.where(pd.notna(sample), None).to_dict()


def run_audit() -> tuple[dict[str, object], dict[str, bool]]:
    audit: dict[str, object] = {}

    for name, path in FILES.items():
        if not path.exists():
            audit[name] = {"exists": False}
            continue

        df = pd.read_csv(path, low_memory=False)
        cols = list(df.columns)
        audit[name] = {
            "exists": True,
            "rows": int(len(df)),
            "cols": cols,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample": _to_jsonable_sample(df),
            "date_cols": [
                c for c in cols if any(x in c.lower() for x in ["date", "time", "when", "utc"])
            ],
            "team_cols": [
                c for c in cols if any(x in c.lower() for x in ["team", "home", "away", "matchup"])
            ],
            "id_cols": [c for c in cols if any(x in c.lower() for x in ["game_id", "event_id", "id", "key"])],
            "score_cols": [
                c
                for c in cols
                if any(
                    x in c.lower()
                    for x in [
                        "pts",
                        "score",
                        "points",
                        "poss",
                        "eff",
                        "rtg",
                        "efg",
                        "tov",
                        "orb",
                        "ftr",
                        "pace",
                    ]
                )
            ],
            "market_cols": [
                c for c in cols if any(x in c.lower() for x in ["spread", "total", "line", "over", "under", "close"])
            ],
        }

    resolved: dict[str, dict[str, str]] = {}
    for metric, candidates in BOX_SCORE_CANDIDATES.items():
        for file_key in ["advanced_metrics", "team_snapshot", "predictions_joint"]:
            file_audit = audit.get(file_key, {})
            if not isinstance(file_audit, dict) or not file_audit.get("exists"):
                continue
            cols = file_audit.get("cols", [])
            if not isinstance(cols, list):
                continue
            hit = _pick_case_insensitive(cols, candidates)
            if hit:
                resolved[metric] = {"file": file_key, "col": hit}
                break
        if metric in resolved:
            continue
    audit["resolved_columns"] = resolved

    market_path = FILES["market_lines"]
    market_df = pd.read_csv(market_path, low_memory=False) if market_path.exists() else pd.DataFrame()
    result_col_candidates = [
        "home_score",
        "away_score",
        "final_score",
        "result",
        "actual_total",
        "home_pts",
        "away_pts",
    ]
    result_col = _pick_case_insensitive(list(market_df.columns), result_col_candidates)

    detection_mode = "result_col"
    if result_col:
        upcoming = market_df[market_df[result_col].isna()]
        completed = market_df[market_df[result_col].notna()]
    else:
        detection_mode = "date_fallback"
        date_col = _pick_case_insensitive(list(market_df.columns), ["game_datetime_utc", "date", "line_timestamp_utc"])
        if date_col:
            dt = pd.to_datetime(market_df[date_col], errors="coerce", utc=True)
            now_utc = pd.Timestamp.now(tz="UTC")
            upcoming_mask = dt.isna() | (dt >= now_utc)
            completed_mask = dt.notna() & (dt < now_utc)
            upcoming = market_df[upcoming_mask]
            completed = market_df[completed_mask]
        else:
            detection_mode = "all_upcoming"
            upcoming = market_df
            completed = pd.DataFrame(columns=market_df.columns)

    audit["upcoming_games"] = {
        "count": int(len(upcoming)),
        "result_col": result_col,
        "detection_mode": detection_mode,
        "completed": int(len(completed)),
        "sample_upcoming": _to_jsonable_sample(upcoming) if len(upcoming) else {},
    }

    gate = {
        "has_team_game_data": any(
            isinstance(v, dict) and v.get("exists") and int(v.get("rows", 0)) > 50
            for v in audit.values()
        ),
        "has_any_efficiency": any(k in audit.get("resolved_columns", {}) for k in ["NetRtg", "OffEff", "eFG"]),
        "has_market_lines": bool(audit.get("market_lines", {}).get("exists"))
        and int(audit.get("market_lines", {}).get("rows", 0)) > 0,
        "has_spread_col": any(
            "spread" in str(c).lower() for c in audit.get("market_lines", {}).get("cols", [])
        ),
        "upcoming_games_found": int(audit.get("upcoming_games", {}).get("count", 0)) > 0,
    }
    return audit, gate


def main() -> int:
    audit, gate = run_audit()

    out_dir = Path("data/internal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_report.json"
    out_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    print("=== GATE_1 RESULTS ===")
    for check, result in gate.items():
        print(f"  {'PASS' if result else 'FAIL'}  {check}")

    if not all(gate.values()):
        failed = [k for k, v in gate.items() if not v]
        print(f"[STOP] Gate 1 failed: {failed}")
        return 1

    print(f"[OK] Gate 1 passed. Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
