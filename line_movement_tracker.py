from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ides_of_march.orchestrator import MODEL_VERSION as IDES_MODEL_VERSION
from ides_of_march.utils import canonical_id, safe_read_csv, utc_now_iso


PST = ZoneInfo("America/Los_Angeles")
IDES_RUN_PREFIXES = ("ides_predict_", "ides_backtest_", "ides_audit_")


@dataclass(frozen=True)
class LineMovementConfig:
    game_id: str = "game_id"
    game_date_pst: str = "game_date_pst"
    team_a: str = "team_a"
    team_b: str = "team_b"
    model_version: str = "model_version"
    phase: str = "phase"
    market_spread: str = "market_spread"
    market_total: str = "market_total"
    market_moneyline_team_a: str = "market_moneyline_team_a"
    market_moneyline_team_b: str = "market_moneyline_team_b"
    line_open_spread: str = "line_open_spread"
    line_open_total: str = "line_open_total"
    line_open_ml_team_a: str = "line_open_ml_team_a"
    line_open_ml_team_b: str = "line_open_ml_team_b"
    line_close_spread: str = "line_close_spread"
    line_close_total: str = "line_close_total"
    line_close_ml_team_a: str = "line_close_ml_team_a"
    line_close_ml_team_b: str = "line_close_ml_team_b"
    spread_movement: str = "spread_movement"
    total_movement: str = "total_movement"
    ml_movement_team_a: str = "ml_movement_team_a"
    ml_movement_team_b: str = "ml_movement_team_b"
    public_pct_team_a: str = "public_pct_team_a"
    public_pct_team_b: str = "public_pct_team_b"
    reverse_line_movement_team_a: str = "reverse_line_movement_team_a"
    reverse_line_movement_team_b: str = "reverse_line_movement_team_b"
    sharp_action_indicator: str = "sharp_action_indicator"
    clv_spread_team_a: str = "clv_spread_team_a"
    clv_spread_team_b: str = "clv_spread_team_b"
    clv_total_over: str = "clv_total_over"
    clv_total_under: str = "clv_total_under"
    line_fetched_at_utc: str = "line_fetched_at_utc"
    updated_at_utc: str = "updated_at_utc"
    is_ncaa_tournament: str = "is_ncaa_tournament"
    is_conference_tournament: str = "is_conference_tournament"
    run_id: str = "run_id"
    model_spread_team_a: str = "model_spread_team_a"
    model_spread_team_b: str = "model_spread_team_b"
    model_total: str = "model_total"
    bet_type: str = "bet_type"
    bet_side: str = "bet_side"
    market_line: str = "market_line"
    model_line: str = "model_line"

    # market source columns
    source_event_id: str = "event_id"
    source_home_team_name: str = "home_team_name"
    source_away_team_name: str = "away_team_name"
    source_opening_spread: str = "opening_spread"
    source_closing_spread: str = "closing_spread"
    source_spread_line: str = "spread_line"
    source_opening_total: str = "opening_total"
    source_closing_total: str = "closing_total"
    source_total_line: str = "total_line"
    source_moneyline_home: str = "moneyline_home"
    source_moneyline_away: str = "moneyline_away"
    source_line_timestamp: str = "line_timestamp_utc"
    source_capture_time: str = "captured_at_utc"
    source_home_tickets_pct: str = "home_tickets_pct"
    source_away_tickets_pct: str = "away_tickets_pct"
    source_capture_type: str = "capture_type"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return _to_num(series).fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _ides_mask(df: pd.DataFrame, cfg: LineMovementConfig) -> tuple[pd.Series, str]:
    if df.empty:
        return pd.Series(dtype=bool), "empty_frame"
    if cfg.model_version in df.columns:
        model_mask = df[cfg.model_version].astype(str).str.strip().eq(IDES_MODEL_VERSION)
        if model_mask.any():
            return model_mask, "model_version"
    if cfg.run_id in df.columns:
        run = df[cfg.run_id].astype(str).str.strip()
        run_mask = pd.Series(False, index=df.index, dtype=bool)
        for pref in IDES_RUN_PREFIXES:
            run_mask = run_mask | run.str.startswith(pref)
        if run_mask.any():
            return run_mask, "run_id_prefix"
    return pd.Series(False, index=df.index, dtype=bool), "none"


def _safe_game_id(series: pd.Series) -> pd.Series:
    return series.map(canonical_id)


def _resolve_source_file(data_dir: Path) -> Path:
    p = data_dir / "market_lines_latest_by_game.csv"
    if p.exists():
        return p
    q = data_dir / "market_lines_latest.csv"
    if q.exists():
        return q
    return data_dir / "market_lines.csv"


def fetch_opening_lines(
    game_ids: list[str],
    game_dates: list[str],
    data_dir: str = "data/",
) -> pd.DataFrame:
    """
    Fetch opening lines for listed games.
    Reuses project market-line source files populated by ingestion pipeline.
    """
    cfg = LineMovementConfig()
    root = Path(data_dir)
    latest = safe_read_csv(root / "market_lines_latest_by_game.csv")
    hist = safe_read_csv(root / "market_lines.csv")

    out = pd.DataFrame({cfg.game_id: pd.Series(game_ids).map(canonical_id).dropna().unique()})
    if out.empty:
        return out

    if not latest.empty:
        if cfg.source_event_id in latest.columns:
            latest[cfg.game_id] = _safe_game_id(latest[cfg.source_event_id])
        elif cfg.game_id in latest.columns:
            latest[cfg.game_id] = _safe_game_id(latest[cfg.game_id])
        else:
            latest[cfg.game_id] = ""
        latest = latest[latest[cfg.game_id].isin(set(out[cfg.game_id]))].copy()
        lsel = latest[[cfg.game_id]].copy()
        lsel[cfg.line_open_spread] = _to_num(latest.get(cfg.source_opening_spread))
        lsel[cfg.line_open_total] = _to_num(latest.get(cfg.source_opening_total))
        lsel[cfg.line_open_ml_team_a] = _to_num(latest.get(cfg.source_moneyline_home))
        lsel[cfg.line_open_ml_team_b] = _to_num(latest.get(cfg.source_moneyline_away))
        lsel[cfg.line_fetched_at_utc] = latest.get(cfg.source_line_timestamp)
        out = out.merge(lsel, on=cfg.game_id, how="left")

    need = out[cfg.line_open_spread].isna() if cfg.line_open_spread in out.columns else pd.Series(True, index=out.index)
    if need.any() and not hist.empty:
        if cfg.source_event_id in hist.columns:
            hist[cfg.game_id] = _safe_game_id(hist[cfg.source_event_id])
        elif cfg.game_id in hist.columns:
            hist[cfg.game_id] = _safe_game_id(hist[cfg.game_id])
        else:
            hist[cfg.game_id] = ""
        hist = hist[hist[cfg.game_id].isin(set(out[cfg.game_id]))].copy()
        if cfg.source_capture_time in hist.columns:
            hist["_ts"] = pd.to_datetime(hist[cfg.source_capture_time], utc=True, errors="coerce")
        else:
            hist["_ts"] = pd.NaT
        if cfg.source_capture_type in hist.columns:
            open_mask = hist[cfg.source_capture_type].astype(str).str.lower().str.contains("open", regex=False)
            open_hist = hist[open_mask].copy()
        else:
            open_hist = hist.copy()
        open_hist = open_hist.sort_values(["_ts"]).drop_duplicates([cfg.game_id], keep="first")
        patch = open_hist[[cfg.game_id]].copy()
        patch[cfg.line_open_spread] = _to_num(open_hist.get("home_spread_open"))
        patch[cfg.line_open_total] = _to_num(open_hist.get("total_open"))
        patch[cfg.line_open_ml_team_a] = _to_num(open_hist.get(cfg.source_moneyline_home))
        patch[cfg.line_open_ml_team_b] = _to_num(open_hist.get(cfg.source_moneyline_away))
        patch[cfg.line_fetched_at_utc] = open_hist.get(cfg.source_capture_time)
        out = out.merge(patch, on=cfg.game_id, how="left", suffixes=("", "_hist"))
        for col in [cfg.line_open_spread, cfg.line_open_total, cfg.line_open_ml_team_a, cfg.line_open_ml_team_b, cfg.line_fetched_at_utc]:
            hist_col = f"{col}_hist"
            if hist_col in out.columns:
                out[col] = _to_num(out[col]).where(_to_num(out[col]).notna(), _to_num(out[hist_col])) if "ml_" in col or "spread" in col or "total" in col else out[col].fillna(out[hist_col])
                out = out.drop(columns=[hist_col], errors="ignore")

    for col in [cfg.line_open_spread, cfg.line_open_total, cfg.line_open_ml_team_a, cfg.line_open_ml_team_b]:
        if col in out.columns:
            out[col] = _to_num(out[col])
    return out


def fetch_current_lines(
    game_ids: list[str],
    data_dir: str = "data/",
) -> pd.DataFrame:
    """
    Fetch current (most recent) lines for listed game IDs.
    """
    cfg = LineMovementConfig()
    root = Path(data_dir)
    source_file = _resolve_source_file(root)
    df = safe_read_csv(source_file)
    if df.empty:
        return pd.DataFrame(columns=[cfg.game_id, cfg.line_close_spread, cfg.line_close_total, cfg.line_close_ml_team_a, cfg.line_close_ml_team_b, cfg.line_fetched_at_utc])

    if cfg.source_event_id in df.columns:
        df[cfg.game_id] = _safe_game_id(df[cfg.source_event_id])
    elif cfg.game_id in df.columns:
        df[cfg.game_id] = _safe_game_id(df[cfg.game_id])
    else:
        return pd.DataFrame(columns=[cfg.game_id, cfg.line_close_spread, cfg.line_close_total, cfg.line_close_ml_team_a, cfg.line_close_ml_team_b, cfg.line_fetched_at_utc])

    wanted = {canonical_id(g) for g in game_ids if canonical_id(g)}
    if wanted:
        df = df[df[cfg.game_id].isin(wanted)].copy()
    if df.empty:
        return pd.DataFrame(columns=[cfg.game_id, cfg.line_close_spread, cfg.line_close_total, cfg.line_close_ml_team_a, cfg.line_close_ml_team_b, cfg.line_fetched_at_utc])

    ts_col = cfg.source_line_timestamp if cfg.source_line_timestamp in df.columns else cfg.source_capture_time
    df["_ts"] = pd.to_datetime(df.get(ts_col), utc=True, errors="coerce")
    if "_ts" in df.columns:
        df = df.sort_values(["_ts"]).drop_duplicates([cfg.game_id], keep="last")

    close_spread_col = cfg.source_spread_line if cfg.source_spread_line in df.columns else cfg.source_closing_spread
    close_total_col = cfg.source_total_line if cfg.source_total_line in df.columns else cfg.source_closing_total

    out = df[
        [
            cfg.game_id,
            close_spread_col,
            close_total_col,
            cfg.source_moneyline_home if cfg.source_moneyline_home in df.columns else close_spread_col,
            cfg.source_moneyline_away if cfg.source_moneyline_away in df.columns else close_spread_col,
            ts_col if ts_col in df.columns else "_ts",
        ]
    ].copy()
    out.columns = [
        cfg.game_id,
        cfg.line_close_spread,
        cfg.line_close_total,
        cfg.line_close_ml_team_a,
        cfg.line_close_ml_team_b,
        cfg.line_fetched_at_utc,
    ]
    for col in [cfg.line_close_spread, cfg.line_close_total, cfg.line_close_ml_team_a, cfg.line_close_ml_team_b]:
        out[col] = _to_num(out[col])
    return out


def fetch_public_betting_percentages(
    game_ids: list[str],
    data_dir: str = "data/",
) -> pd.DataFrame:
    """
    Fetch public split percentages from market capture history if available.
    """
    cfg = LineMovementConfig()
    hist = safe_read_csv(Path(data_dir) / "market_lines.csv")
    if hist.empty:
        return pd.DataFrame(columns=[cfg.game_id, cfg.public_pct_team_a, cfg.public_pct_team_b, cfg.line_fetched_at_utc])
    if cfg.source_event_id in hist.columns:
        hist[cfg.game_id] = _safe_game_id(hist[cfg.source_event_id])
    elif cfg.game_id in hist.columns:
        hist[cfg.game_id] = _safe_game_id(hist[cfg.game_id])
    else:
        return pd.DataFrame(columns=[cfg.game_id, cfg.public_pct_team_a, cfg.public_pct_team_b, cfg.line_fetched_at_utc])

    wanted = {canonical_id(g) for g in game_ids if canonical_id(g)}
    hist = hist[hist[cfg.game_id].isin(wanted)].copy()
    if hist.empty:
        return pd.DataFrame(columns=[cfg.game_id, cfg.public_pct_team_a, cfg.public_pct_team_b, cfg.line_fetched_at_utc])

    ts_col = cfg.source_capture_time if cfg.source_capture_time in hist.columns else cfg.source_line_timestamp
    hist["_ts"] = pd.to_datetime(hist.get(ts_col), utc=True, errors="coerce")
    hist = hist.sort_values(["_ts"]).drop_duplicates([cfg.game_id], keep="last")
    out = hist[[cfg.game_id]].copy()
    out[cfg.public_pct_team_a] = _to_num(hist.get(cfg.source_home_tickets_pct))
    out[cfg.public_pct_team_b] = _to_num(hist.get(cfg.source_away_tickets_pct))
    out[cfg.line_fetched_at_utc] = hist["_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for col in [cfg.public_pct_team_a, cfg.public_pct_team_b]:
        s = out[col]
        out[col] = np.where(s > 1.0, s / 100.0, s)
    return out


def compute_line_movement(
    opening: pd.DataFrame,
    current: pd.DataFrame,
    public_pcts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute spread/total/ML movements and reverse-line movement flags.
    """
    cfg = LineMovementConfig()
    if opening.empty and current.empty:
        return pd.DataFrame()

    base = opening.copy() if not opening.empty else pd.DataFrame({cfg.game_id: current[cfg.game_id].drop_duplicates()})
    base = base.merge(current, on=cfg.game_id, how="outer", suffixes=("", "_cur"))
    if public_pcts is not None and not public_pcts.empty:
        base = base.merge(public_pcts, on=cfg.game_id, how="left", suffixes=("", "_pub"))
    else:
        base[cfg.public_pct_team_a] = np.nan
        base[cfg.public_pct_team_b] = np.nan

    for col in [
        cfg.line_open_spread,
        cfg.line_open_total,
        cfg.line_open_ml_team_a,
        cfg.line_open_ml_team_b,
        cfg.line_close_spread,
        cfg.line_close_total,
        cfg.line_close_ml_team_a,
        cfg.line_close_ml_team_b,
    ]:
        if col in base.columns:
            base[col] = _to_num(base[col])
        else:
            base[col] = np.nan

    base[cfg.spread_movement] = base[cfg.line_close_spread] - base[cfg.line_open_spread]
    base[cfg.total_movement] = base[cfg.line_close_total] - base[cfg.line_open_total]
    base[cfg.ml_movement_team_a] = base[cfg.line_close_ml_team_a] - base[cfg.line_open_ml_team_a]
    base[cfg.ml_movement_team_b] = base[cfg.line_close_ml_team_b] - base[cfg.line_open_ml_team_b]

    public_a = _to_num(base[cfg.public_pct_team_a])
    public_b = _to_num(base[cfg.public_pct_team_b])
    mov = _to_num(base[cfg.spread_movement])
    base[cfg.reverse_line_movement_team_a] = np.where(public_a.notna(), (public_a > 0.55) & (mov > 0), np.nan)
    base[cfg.reverse_line_movement_team_b] = np.where(public_b.notna(), (public_b > 0.55) & (mov < 0), np.nan)

    abs_m = mov.abs()
    base[cfg.sharp_action_indicator] = np.where(
        abs_m >= 3.0,
        "steam",
        np.where(
            (abs_m >= 1.5) & (mov > 0),
            "team_b",
            np.where((abs_m >= 1.5) & (mov < 0), "team_a", "none"),
        ),
    )

    now = utc_now_iso()
    if cfg.line_fetched_at_utc not in base.columns:
        base[cfg.line_fetched_at_utc] = now
    base[cfg.updated_at_utc] = now
    return base


def compute_clv(
    bet_recs: pd.DataFrame,
    closing_lines: pd.DataFrame,
    cfg: LineMovementConfig,
) -> pd.DataFrame:
    """
    Compute CLV metrics for current recommendations.
    """
    if bet_recs.empty or closing_lines.empty:
        return pd.DataFrame(columns=[cfg.game_id, cfg.clv_spread_team_a, cfg.clv_spread_team_b, cfg.clv_total_over, cfg.clv_total_under])

    recs = bet_recs.copy()
    recs[cfg.game_id] = _safe_game_id(recs[cfg.game_id])
    close = closing_lines.copy()
    close[cfg.game_id] = _safe_game_id(close[cfg.game_id])
    merged = recs.merge(close, on=cfg.game_id, how="left", suffixes=("", "_close"))
    merged[cfg.bet_type] = merged.get(cfg.bet_type, "").astype(str).str.lower()
    merged[cfg.bet_side] = merged.get(cfg.bet_side, "").astype(str).str.lower()

    model_line = _to_num(merged.get(cfg.model_line, np.nan))
    close_spread = _to_num(merged[cfg.line_close_spread])
    close_total = _to_num(merged[cfg.line_close_total])

    clv_spread = np.where(
        merged[cfg.bet_type].eq("spread"),
        np.where(merged[cfg.bet_side].eq("team_a"), model_line - close_spread, close_spread - model_line),
        np.nan,
    )
    clv_total_over = np.where((merged[cfg.bet_type].eq("total")) & (merged[cfg.bet_side].eq("over")), model_line - close_total, np.nan)
    clv_total_under = np.where((merged[cfg.bet_type].eq("total")) & (merged[cfg.bet_side].eq("under")), close_total - model_line, np.nan)

    out = merged[[cfg.game_id]].copy()
    out[cfg.clv_spread_team_a] = clv_spread
    out[cfg.clv_spread_team_b] = np.where(pd.notna(clv_spread), -clv_spread, np.nan)
    out[cfg.clv_total_over] = clv_total_over
    out[cfg.clv_total_under] = clv_total_under
    return out.groupby(cfg.game_id, as_index=False).agg(
        {
            cfg.clv_spread_team_a: "mean",
            cfg.clv_spread_team_b: "mean",
            cfg.clv_total_over: "mean",
            cfg.clv_total_under: "mean",
        }
    )


def write_line_movement_csv(
    movement_df: pd.DataFrame,
    output_path: str = "data/plumbing/line_movement.csv",
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    movement_df.to_csv(output_path, index=False)


def update_master_with_clv(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    movement_df: pd.DataFrame | None = None,
    cfg: LineMovementConfig | None = None,
) -> None:
    """
    Update IDES master predictions with closing lines + CLV fields.
    """
    config = cfg or LineMovementConfig()
    preds = safe_read_csv(Path(predictions_path))
    if preds.empty or movement_df is None or movement_df.empty:
        return

    preds[config.game_id] = _safe_game_id(preds[config.game_id])
    mask, _ = _ides_mask(preds, config)
    if not mask.any():
        return
    scoped = preds.loc[mask].copy()
    scoped["_orig_idx"] = scoped.index

    for col in [
        config.line_close_spread,
        config.line_close_total,
        config.line_close_ml_team_a,
        config.line_close_ml_team_b,
        config.clv_spread_team_a,
        config.clv_spread_team_b,
        config.clv_total_over,
        config.clv_total_under,
        config.updated_at_utc,
    ]:
        if col not in preds.columns:
            preds[col] = np.nan

    mv = movement_df.copy()
    mv[config.game_id] = _safe_game_id(mv[config.game_id])
    keep_cols = [
        c
        for c in [
            config.game_id,
            config.line_close_spread,
            config.line_close_total,
            config.line_close_ml_team_a,
            config.line_close_ml_team_b,
            config.clv_spread_team_a,
            config.clv_spread_team_b,
            config.clv_total_over,
            config.clv_total_under,
        ]
        if c in mv.columns
    ]
    scoped = scoped.merge(mv[keep_cols].drop_duplicates([config.game_id], keep="last"), on=config.game_id, how="left", suffixes=("", "_mv"))
    scoped[config.updated_at_utc] = utc_now_iso()

    updatable = [c for c in keep_cols + [config.updated_at_utc] if c in preds.columns and c != config.game_id]
    for col in updatable:
        preds.loc[scoped["_orig_idx"], col] = scoped[col].values

    preds.to_csv(predictions_path, index=False)


def run_line_movement_tracker(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    bet_recs_path: str = "data/actionable/bet_recommendations.csv",
    data_dir: str = "data/",
    mode: str = "pre_game",
) -> bool:
    """
    Run line movement tracker in pre-game or post-game mode.
    """
    cfg = LineMovementConfig()
    preds = safe_read_csv(Path(predictions_path))
    if preds.empty:
        print("[WARN] predictions master missing/empty.")
        return False

    preds[cfg.game_id] = _safe_game_id(preds[cfg.game_id])
    mask, method = _ides_mask(preds, cfg)
    if not mask.any():
        print("[FAIL] Could not identify IDES rows; refusing to run.")
        return False
    scoped = preds.loc[mask].copy()

    today_pst = pd.Timestamp.now(tz=PST).strftime("%Y-%m-%d")
    if cfg.game_date_pst in scoped.columns:
        if mode == "pre_game":
            scoped = scoped[scoped[cfg.game_date_pst].astype(str).eq(today_pst)].copy()
        else:
            scoped = scoped[pd.to_datetime(scoped[cfg.game_date_pst], errors="coerce") <= pd.Timestamp.now(tz=PST).normalize().tz_localize(None)].copy()
    if scoped.empty:
        print("[INFO] no IDES games in selected mode window.")
        write_line_movement_csv(pd.DataFrame(columns=[cfg.game_id]), output_path="data/plumbing/line_movement.csv")
        return True

    game_ids = scoped[cfg.game_id].astype(str).tolist()
    game_dates = scoped[cfg.game_date_pst].astype(str).tolist() if cfg.game_date_pst in scoped.columns else []

    opening = fetch_opening_lines(game_ids=game_ids, game_dates=game_dates, data_dir=data_dir)
    current = fetch_current_lines(game_ids=game_ids, data_dir=data_dir)
    public = fetch_public_betting_percentages(game_ids=game_ids, data_dir=data_dir)
    movement = compute_line_movement(opening, current, public)

    if cfg.team_a in scoped.columns and cfg.team_b in scoped.columns:
        movement = movement.merge(scoped[[cfg.game_id, cfg.team_a, cfg.team_b]].drop_duplicates(cfg.game_id), on=cfg.game_id, how="left")
    write_line_movement_csv(movement, output_path="data/plumbing/line_movement.csv")

    if mode == "post_game":
        bets = safe_read_csv(Path(bet_recs_path))
        if not bets.empty and cfg.game_id in bets.columns:
            bets[cfg.game_id] = _safe_game_id(bets[cfg.game_id])
            bmask, _ = _ides_mask(bets, cfg)
            bets = bets.loc[bmask].copy()
            clv = compute_clv(bets, movement, cfg)
            enriched = movement.merge(clv, on=cfg.game_id, how="left")
            update_master_with_clv(predictions_path=predictions_path, movement_df=enriched, cfg=cfg)
            avg_clv = pd.to_numeric(clv.get(cfg.clv_spread_team_a), errors="coerce")
            pos_clv = float((avg_clv > 0).mean()) if avg_clv.notna().any() else np.nan
            print(
                f"[INFO] mode={mode} scope={method} games={len(scoped)} "
                f"avg_clv={float(avg_clv.mean()):.3f}" if avg_clv.notna().any() else f"[INFO] mode={mode} scope={method} games={len(scoped)} avg_clv=n/a"
            )
            if pd.notna(pos_clv):
                print(f"[INFO] positive_clv_rate={pos_clv:.2%}")
        return True
    else:
        sharp = movement[cfg.sharp_action_indicator].astype(str).isin({"steam", "team_a", "team_b"}).sum() if cfg.sharp_action_indicator in movement.columns else 0
        rlm_a = _to_bool(movement[cfg.reverse_line_movement_team_a]).sum() if cfg.reverse_line_movement_team_a in movement.columns else 0
        rlm_b = _to_bool(movement[cfg.reverse_line_movement_team_b]).sum() if cfg.reverse_line_movement_team_b in movement.columns else 0
        print(f"[INFO] mode={mode} scope={method} games={len(scoped)} sharp_signals={int(sharp)} rlm_team_a={int(rlm_a)} rlm_team_b={int(rlm_b)}")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="IDES line movement tracker")
    parser.add_argument("--predictions-path", default="data/reports/game_predictions_master.csv")
    parser.add_argument("--bet-recs-path", default="data/actionable/bet_recommendations.csv")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--mode", default="pre_game", choices=["pre_game", "post_game"])
    args = parser.parse_args()
    ok = run_line_movement_tracker(
        predictions_path=args.predictions_path,
        bet_recs_path=args.bet_recs_path,
        data_dir=args.data_dir,
        mode=args.mode,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
