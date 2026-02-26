from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_csv

SEGMENTS = ["OVERALL", "HIGH_MAJOR", "MID_MAJOR", "LOW_MAJOR"]
SEGMENT_PRECEDENCE = ["segment", "conference_bucket", "conference_tier", "game_tier"]
SEGMENT_MAP = {
    "HIGH": "HIGH_MAJOR",
    "MID": "MID_MAJOR",
    "LOW": "LOW_MAJOR",
    "HIGH_MAJOR": "HIGH_MAJOR",
    "MID_MAJOR": "MID_MAJOR",
    "LOW_MAJOR": "LOW_MAJOR",
}
MODEL_ID_CANDIDATES = ["model_id", "sub_model", "model", "model_name"]
GAME_ID_CANDIDATES = ["game_id", "event_id"]
SPREAD_CLOSE_COLUMNS = ["closing_spread", "spread_line", "market_spread_home"]


class BacktestOutputError(RuntimeError):
    pass


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _assert_no_market_snapshot_ambiguity(df: pd.DataFrame) -> None:
    present = [c for c in SPREAD_CLOSE_COLUMNS if c in df.columns]
    if len(present) <= 1:
        return
    base = pd.to_numeric(df[present[0]], errors="coerce")
    for col in present[1:]:
        cur = pd.to_numeric(df[col], errors="coerce")
        mismatch = (base.notna() & cur.notna() & (base.round(4) != cur.round(4))).any()
        if mismatch:
            raise BacktestOutputError(
                "Ambiguous spread market snapshot detected across columns "
                f"{present}. Add DECISIONS_NEEDED.md guidance for canonical grading line."
            )


def classify_segment(row: pd.Series) -> str:
    candidates: list[str] = []
    for col in SEGMENT_PRECEDENCE:
        if col in row.index:
            val = row.get(col)
            if pd.isna(val):
                continue
            mapped = SEGMENT_MAP.get(str(val).strip().upper())
            if mapped:
                candidates.append(mapped)
    if not candidates:
        return "UNKNOWN"
    if len(set(candidates)) > 1:
        raise BacktestOutputError(
            f"Conflicting segment values for row game_id={row.get('game_id')}: {candidates}. "
            "Set explicit precedence in config or source one canonical tier column."
        )
    return candidates[0]


def _pct(w: int, l: int) -> float | None:
    den = w + l
    if den == 0:
        return None
    return round(w / den, 4)


def _roi(series: pd.Series) -> float | None:
    if series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce")
    if not s.notna().any():
        return None
    return round(float(s.fillna(0).sum()), 4)


def _ensure_required_game_level(df: pd.DataFrame) -> pd.DataFrame:
    model_col = _find_column(df, MODEL_ID_CANDIDATES)
    game_col = _find_column(df, GAME_ID_CANDIDATES)
    if model_col is None or game_col is None:
        raise BacktestOutputError("Missing model_id/sub_model or game_id/event_id required for backtest outputs.")

    out = df.copy()
    out["model_id"] = out[model_col].fillna("PRIMARY").astype(str)
    out["game_id"] = out[game_col].astype(str)

    key_dups = out.duplicated(subset=["model_id", "game_id"], keep=False)
    if key_dups.any():
        raise BacktestOutputError(
            f"Expected one row per (game_id, model_id); found duplicates={int(key_dups.sum())}."
        )

    _assert_no_market_snapshot_ambiguity(out)

    out["segment"] = out.apply(classify_segment, axis=1)
    return out


def _normalize_result(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(index=df.index, dtype=object)
    return df[col].astype(str).str.upper().map({"WIN": "W", "LOSS": "L", "PUSH": "PUSH", "W": "W", "L": "L", "P": "PUSH"})


def _get_market_spread(df: pd.DataFrame) -> pd.Series:
    col = _find_column(df, SPREAD_CLOSE_COLUMNS)
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _build_game_level(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_required_game_level(df)
    out["su_result"] = np.where(pd.to_numeric(out.get("winner_correct"), errors="coerce") == 1, "W", "L")
    out["ats_result"] = _normalize_result(out, "ats_result")
    if out["ats_result"].isna().all():
        out["ats_result"] = _normalize_result(out, "primary_ats_result")
    out["total_result"] = _normalize_result(out, "ou_result")
    if out["total_result"].isna().all():
        out["total_result"] = _normalize_result(out, "total_result")

    out["market_spread_home"] = _get_market_spread(out)
    out["market_ml_home"] = pd.to_numeric(out.get("market_ml_home", out.get("home_ml")), errors="coerce")
    out["market_ml_away"] = pd.to_numeric(out.get("market_ml_away", out.get("away_ml")), errors="coerce")

    out["model_pick_su"] = out.get("predicted_winner", "").astype(str).str.lower().map({"home": "HOME", "away": "AWAY"})
    if out["model_pick_su"].isna().all() and "pred_spread" in out.columns:
        ps = pd.to_numeric(out["pred_spread"], errors="coerce")
        out["model_pick_su"] = np.where(ps < 0, "HOME", np.where(ps > 0, "AWAY", np.nan))

    spread = out["market_spread_home"]
    mlh = out["market_ml_home"]
    mla = out["market_ml_away"]
    out["market_underdog_side"] = np.where(
        spread.notna(),
        np.where(spread > 0, "HOME", np.where(spread < 0, "AWAY", np.nan)),
        np.where(mlh.notna() & mla.notna(), np.where(mlh > mla, "HOME", np.where(mla > mlh, "AWAY", np.nan)), np.nan),
    )
    out["picked_market_underdog_to_win_flag"] = out["model_pick_su"].eq(out["market_underdog_side"])

    out["su_roi_u"] = pd.to_numeric(out.get("su_roi_u"), errors="coerce")
    out["ats_roi_u"] = pd.to_numeric(out.get("ats_roi", out.get("ats_roi_u")), errors="coerce")
    out["total_roi_u"] = pd.to_numeric(out.get("ou_roi", out.get("total_roi_u")), errors="coerce")
    return out


def _segment_views(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    views = [("OVERALL", df)]
    for segment in ["HIGH_MAJOR", "MID_MAJOR", "LOW_MAJOR"]:
        views.append((segment, df[df["segment"] == segment]))
    return views


def build_backtest_model_summary(df_game_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model_id, model_df in df_game_level.groupby("model_id", dropna=False):
        for segment, seg_df in _segment_views(model_df):
            su_w = int((seg_df["su_result"] == "W").sum())
            su_l = int((seg_df["su_result"] == "L").sum())
            ats_w = int((seg_df["ats_result"] == "W").sum())
            ats_l = int((seg_df["ats_result"] == "L").sum())
            ats_push = int((seg_df["ats_result"] == "PUSH").sum())
            total_w = int((seg_df["total_result"] == "W").sum())
            total_l = int((seg_df["total_result"] == "L").sum())
            total_push = int((seg_df["total_result"] == "PUSH").sum())
            su_roi = _roi(seg_df["su_roi_u"])
            ats_roi = _roi(seg_df["ats_roi_u"])
            total_roi = _roi(seg_df["total_roi_u"])
            known_rois = [x for x in [su_roi, ats_roi, total_roi] if x is not None]
            overall_roi = round(sum(known_rois), 4) if known_rois else None

            rows.append(
                {
                    "model_id": model_id,
                    "segment": segment,
                    "games": int(len(seg_df)),
                    "su_w": su_w,
                    "su_l": su_l,
                    "su_push": 0,
                    "su_win_pct": _pct(su_w, su_l),
                    "ats_w": ats_w,
                    "ats_l": ats_l,
                    "ats_push": ats_push,
                    "ats_win_pct": _pct(ats_w, ats_l),
                    "total_w": total_w,
                    "total_l": total_l,
                    "total_push": total_push,
                    "total_win_pct": _pct(total_w, total_l),
                    "su_roi_u": su_roi,
                    "ats_roi_u": ats_roi,
                    "total_roi_u": total_roi,
                    "overall_roi_u": overall_roi,
                }
            )
    return pd.DataFrame(rows)


def build_upset_picks_summary(df_game_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model_id, model_df in df_game_level.groupby("model_id", dropna=False):
        for segment, seg_df in _segment_views(model_df):
            dogs = seg_df[seg_df["picked_market_underdog_to_win_flag"].fillna(False)].copy()
            n_games = len(seg_df)
            dogs_n = len(dogs)
            rows.append(
                {
                    "model_id": model_id,
                    "segment": segment,
                    "dogs_picked": int(dogs_n),
                    "games_dogs_picked_pct": round(dogs_n / n_games, 4) if n_games else None,
                    "dogs_su_w": int((dogs["su_result"] == "W").sum()),
                    "dogs_su_l": int((dogs["su_result"] == "L").sum()),
                    "dogs_su_win_pct": _pct(int((dogs["su_result"] == "W").sum()), int((dogs["su_result"] == "L").sum())),
                    "dogs_ats_w": int((dogs["ats_result"] == "W").sum()),
                    "dogs_ats_l": int((dogs["ats_result"] == "L").sum()),
                    "dogs_ats_push": int((dogs["ats_result"] == "PUSH").sum()),
                    "dogs_ats_win_pct": _pct(int((dogs["ats_result"] == "W").sum()), int((dogs["ats_result"] == "L").sum())),
                    "dogs_avg_spread": round(float(pd.to_numeric(dogs["market_spread_home"], errors="coerce").mean()), 4) if dogs_n else None,
                    "dogs_median_spread": round(float(pd.to_numeric(dogs["market_spread_home"], errors="coerce").median()), 4) if dogs_n else None,
                    "dogs_su_roi_u": _roi(dogs["su_roi_u"]),
                    "dogs_ats_roi_u": _roi(dogs["ats_roi_u"]),
                }
            )
    return pd.DataFrame(rows)


def build_leaderboard(df_summary: pd.DataFrame, df_upsets: pd.DataFrame) -> pd.DataFrame:
    metrics = ["su_win_pct", "ats_win_pct", "total_win_pct", "overall_roi_u"]
    rows: list[dict] = []
    for segment in SEGMENTS:
        seg_summary = df_summary[df_summary["segment"] == segment]
        for metric in metrics:
            row = {"segment": segment, "metric": metric}
            for _, rec in seg_summary.iterrows():
                row[str(rec["model_id"])] = rec.get(metric)
            rows.append(row)

        seg_upsets = df_upsets[df_upsets["segment"] == segment]
        for metric in ["dogs_su_win_pct", "dogs_ats_win_pct"]:
            row = {"segment": segment, "metric": metric}
            for _, rec in seg_upsets.iterrows():
                row[str(rec["model_id"])] = rec.get(metric)
            rows.append(row)
    return pd.DataFrame(rows)


def write_backtest_outputs(run_dir: Path, graded_df: pd.DataFrame) -> dict:
    game_level = _build_game_level(graded_df)
    summary = build_backtest_model_summary(game_level)
    upsets = build_upset_picks_summary(game_level)
    leaderboard = build_leaderboard(summary, upsets)

    detail_cols = [
        "game_date", "game_id", "segment", "model_id", "home_team", "away_team",
        "market_spread_home", "total_line", "market_ml_home", "market_ml_away",
        "model_pick_su", "home_win_prob", "pred_spread", "pred_total",
        "su_result", "ats_result", "total_result", "picked_market_underdog_to_win_flag",
    ]
    available_detail_cols = [c for c in detail_cols if c in game_level.columns]
    game_detail = game_level[available_detail_cols].rename(columns={
        "game_date": "date",
        "total_line": "market_total",
        "home_win_prob": "model_win_prob_home",
        "pred_spread": "model_spread",
        "pred_total": "model_total",
    })

    write_csv(summary, run_dir / "backtest/summary/backtest_model_summary.csv")
    write_csv(upsets, run_dir / "backtest/summary/backtest_upset_picks_summary.csv")
    write_csv(leaderboard, run_dir / "backtest/summary/backtest_model_leaderboard.csv")
    if not game_detail.empty:
        write_csv(game_detail, run_dir / "backtest/detail/backtest_game_level.csv")

    unknown_segments = int((game_level["segment"] == "UNKNOWN").sum())
    return {
        "rows_game_level": int(len(game_level)),
        "rows_summary": int(len(summary)),
        "rows_upset_summary": int(len(upsets)),
        "rows_leaderboard": int(len(leaderboard)),
        "unknown_segment_rows": unknown_segments,
        "detail_columns_included": list(game_detail.columns),
        "win_pct_formula": "W / (W + L), pushes excluded",
        "market_underdog_definition": "closing spread (+spread side), fallback to ML longer odds if spread missing",
        "roi_policy": "Uses explicit ROI columns only; no assumed -110 injected by this module.",
    }
