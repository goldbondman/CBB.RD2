from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_pes_columns import discover_ides_rows
from ides_of_march.utils import safe_read_csv, utc_now_iso
from layer_registry import get_active_layers, load_registry, save_registry


@dataclass(frozen=True)
class MonitorConfig:
    game_id: str = "game_id"
    game_date_pst: str = "game_date_pst"
    team_a: str = "team_a"
    season: str = "season"
    covered: str = "covered"
    ml_won: str = "ml_won"
    total_covered: str = "total_covered"
    over_hit: str = "over_hit"
    under_hit: str = "under_hit"
    final_margin: str = "final_margin"
    model_version: str = "model_version"
    phase: str = "phase"
    is_ncaa_tournament: str = "is_ncaa_tournament"
    is_conference_tournament: str = "is_conference_tournament"
    updated_at_utc: str = "updated_at_utc"
    rolling_window_games: int = 30
    demotion_threshold: float = 0.50
    watchlist_threshold: float = 0.52
    min_rolling_sample: int = 15

    # Existing columns needed to derive hits from master
    final_score_team_a: str = "final_score_team_a"
    final_score_team_b: str = "final_score_team_b"
    covered_team_a: str = "covered_team_a"
    covered_team_b: str = "covered_team_b"
    actual_winner: str = "actual_winner"
    model_pick: str = "model_pick"
    total_result: str = "total_result"
    total_pick: str = "total_pick"
    situational_layer_applied: str = "situational_layer_applied"
    run_id: str = "run_id"


@dataclass
class RollingPerformanceReport:
    layer_name: str
    market: str
    all_time_hit_rate: float
    all_time_n: int
    rolling_hit_rate: float
    rolling_n: int
    rolling_trend: str
    games_since_promotion: int
    last_result_date: str
    consecutive_misses: int


def _num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _bool(s: Any) -> pd.Series:
    x = pd.Series(s)
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False)
    if pd.api.types.is_numeric_dtype(x):
        return _num(x).fillna(0).astype(float) != 0
    return x.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_layers(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    delim = "|"
    if ";" in text:
        delim = ";"
    elif "," in text:
        delim = ","
    return {t.strip() for t in text.split(delim) if t.strip()}


def _derive_market_hit(frame: pd.DataFrame, market: str, cfg: MonitorConfig) -> pd.Series:
    m = str(market).lower()
    if "ats" in m:
        if cfg.covered in frame.columns:
            return _num(frame[cfg.covered])
        model_on_team_a = frame.get(cfg.model_pick, "").astype(str).eq(frame.get(cfg.team_a, ""))
        c_a = _num(frame.get(cfg.covered_team_a))
        c_b = _num(frame.get(cfg.covered_team_b))
        return pd.Series(np.where(model_on_team_a, c_a, c_b), index=frame.index)

    if "ml" in m or "su" in m:
        if cfg.ml_won in frame.columns:
            return _num(frame[cfg.ml_won])
        return frame.get(cfg.actual_winner, "").astype(str).eq(frame.get(cfg.model_pick, "")).astype(float)

    if "over" in m:
        if cfg.over_hit in frame.columns:
            return _num(frame[cfg.over_hit])
        if cfg.total_covered in frame.columns:
            return _num(frame[cfg.total_covered])
        return frame.get(cfg.total_result, "").astype(str).str.lower().eq("over").astype(float)

    if "under" in m:
        if cfg.under_hit in frame.columns:
            return _num(frame[cfg.under_hit])
        if cfg.total_covered in frame.columns:
            return 1.0 - _num(frame[cfg.total_covered])
        return frame.get(cfg.total_result, "").astype(str).str.lower().eq("under").astype(float)

    if cfg.total_covered in frame.columns:
        return _num(frame[cfg.total_covered])
    return pd.Series(np.nan, index=frame.index)


def _rolling_trend(rolling: float, all_time: float) -> str:
    if pd.isna(rolling) or pd.isna(all_time):
        return "stable"
    if rolling < all_time - 0.04:
        return "degrading"
    if rolling > all_time + 0.04:
        return "improving"
    return "stable"


def _consecutive_misses(hit_series: pd.Series) -> int:
    s = _num(hit_series).dropna()
    streak = 0
    for val in reversed(s.tolist()):
        if float(val) >= 0.5:
            break
        streak += 1
    return streak


def load_post_promotion_games(
    predictions_path: str,
    registry: pd.DataFrame,
    cfg: MonitorConfig,
) -> dict[str, pd.DataFrame]:
    """
    Loads post-promotion games where each active layer fired and has final results.
    """
    preds = safe_read_csv(Path(predictions_path))
    if preds.empty or registry.empty:
        return {}
    try:
        mask, _ = discover_ides_rows(preds, strict=True)
    except ValueError:
        return {}
    scoped = preds.loc[mask].copy()
    if scoped.empty:
        return {}

    if cfg.game_date_pst in scoped.columns:
        scoped["_game_date"] = pd.to_datetime(scoped[cfg.game_date_pst], errors="coerce")
    else:
        scoped["_game_date"] = pd.NaT
    scoped["_has_result"] = _num(scoped.get(cfg.final_score_team_a)).notna() & _num(scoped.get(cfg.final_score_team_b)).notna()
    scoped = scoped[scoped["_has_result"]].copy()
    if scoped.empty:
        return {}

    active = get_active_layers(registry)
    if active.empty:
        return {}
    active = active.sort_values("last_backtest_date").drop_duplicates(["layer_name", "market"], keep="last")

    out: dict[str, pd.DataFrame] = {}
    for _, layer in active.iterrows():
        lname = str(layer.get("layer_name", "")).strip()
        if not lname:
            continue
        promo_ts = pd.to_datetime(layer.get("last_backtest_date"), utc=True, errors="coerce")
        if pd.isna(promo_ts):
            promo_date = pd.Timestamp.min.tz_localize("UTC")
        else:
            promo_date = promo_ts

        fired = scoped[cfg.situational_layer_applied].apply(_parse_layers).apply(lambda x: lname in x) if cfg.situational_layer_applied in scoped.columns else pd.Series(False, index=scoped.index)
        local = scoped[fired].copy()
        if local.empty:
            continue
        game_dt = pd.to_datetime(local[cfg.updated_at_utc], utc=True, errors="coerce") if cfg.updated_at_utc in local.columns else pd.to_datetime(local[cfg.game_date_pst], errors="coerce").dt.tz_localize("UTC")
        local = local[game_dt >= promo_date].copy()
        if local.empty:
            continue
        key = f"{lname}||{layer.get('market', '')}"
        out[key] = local.sort_values(cfg.game_date_pst)
    return out


def compute_rolling_performance(
    layer_games: pd.DataFrame,
    layer_name: str,
    market: str,
    cfg: MonitorConfig,
) -> RollingPerformanceReport:
    """
    Computes all-time and rolling post-promotion performance stats.
    """
    hits = _derive_market_hit(layer_games, market, cfg)
    valid = hits.dropna()
    all_n = int(len(valid))
    all_hr = float(valid.mean()) if all_n else np.nan
    window = int(max(1, cfg.rolling_window_games))
    rolling_slice = valid.tail(window)
    roll_n = int(len(rolling_slice))
    roll_hr = float(rolling_slice.mean()) if roll_n else np.nan
    trend = _rolling_trend(roll_hr, all_hr)
    last_date = ""
    if cfg.game_date_pst in layer_games.columns and not layer_games.empty:
        last_date = str(layer_games[cfg.game_date_pst].astype(str).iloc[-1])
    return RollingPerformanceReport(
        layer_name=layer_name,
        market=market,
        all_time_hit_rate=all_hr,
        all_time_n=all_n,
        rolling_hit_rate=roll_hr,
        rolling_n=roll_n,
        rolling_trend=trend,
        games_since_promotion=all_n,
        last_result_date=last_date,
        consecutive_misses=_consecutive_misses(valid),
    )


def evaluate_demotion_candidates(
    performance_reports: dict[str, RollingPerformanceReport],
    registry: pd.DataFrame,
    cfg: MonitorConfig,
) -> pd.DataFrame:
    """
    Produces status recommendations (flagged_for_review/watchlist/active).
    """
    rows: list[dict[str, Any]] = []
    for key, rep in performance_reports.items():
        if rep.rolling_n >= int(cfg.min_rolling_sample) and pd.notna(rep.rolling_hit_rate) and rep.rolling_hit_rate < float(cfg.demotion_threshold):
            rec_status = "flagged_for_review"
            reason = f"rolling_hit_rate_{rep.rolling_hit_rate:.3f}_below_{cfg.demotion_threshold:.3f}"
        elif (
            rep.rolling_n >= int(cfg.min_rolling_sample)
            and pd.notna(rep.rolling_hit_rate)
            and float(cfg.demotion_threshold) <= rep.rolling_hit_rate < float(cfg.watchlist_threshold)
            and rep.rolling_trend == "degrading"
        ):
            rec_status = "watchlist"
            reason = f"rolling_between_thresholds_and_{rep.rolling_trend}"
        else:
            rec_status = "active"
            reason = "clear_to_remain_active"
        rows.append(
            {
                "key": key,
                "layer_name": rep.layer_name,
                "market": rep.market,
                "recommended_status": rec_status,
                "reason": reason,
                "rolling_hit_rate": rep.rolling_hit_rate,
                "rolling_n": rep.rolling_n,
                "all_time_hit_rate": rep.all_time_hit_rate,
                "all_time_n": rep.all_time_n,
                "rolling_trend": rep.rolling_trend,
                "consecutive_misses": rep.consecutive_misses,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[out["recommended_status"].isin(["flagged_for_review", "watchlist"])].copy()


def apply_demotion_flags(
    demotion_candidates: pd.DataFrame,
    registry_path: str = "data/layer_registry.csv",
    auto_demote: bool = False,
) -> dict[str, Any]:
    """
    Applies status updates (flagged_for_review/watchlist) or writes review file.
    """
    summary = {"applied": 0, "flagged_for_review": 0, "watchlist": 0, "output": ""}
    if demotion_candidates.empty:
        return summary

    today = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    if not auto_demote:
        out_path = Path(f"data/layer_demotion_candidates_{today}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        demotion_candidates.to_csv(out_path, index=False)
        summary["output"] = str(out_path)
        return summary

    reg = load_registry(registry_path)
    if reg.empty:
        return summary
    latest = reg.sort_values("last_backtest_date").drop_duplicates(["layer_name", "scenario", "market"], keep="last").copy()
    changes = 0
    flag_count = 0
    watch_count = 0
    for _, row in demotion_candidates.iterrows():
        mask = latest["layer_name"].astype(str).eq(str(row["layer_name"])) & latest["market"].astype(str).eq(str(row["market"]))
        if not mask.any():
            continue
        new_status = row["recommended_status"]
        if latest.loc[mask, "status"].astype(str).eq(new_status).all():
            continue
        latest.loc[mask, "status"] = new_status
        latest.loc[mask, "notes"] = str(row["reason"])
        latest.loc[mask, "last_backtest_date"] = utc_now_iso()
        changes += int(mask.sum())
        if new_status == "flagged_for_review":
            flag_count += int(mask.sum())
        if new_status == "watchlist":
            watch_count += int(mask.sum())
    if changes:
        save_registry(latest, path=registry_path)
    summary["applied"] = changes
    summary["flagged_for_review"] = flag_count
    summary["watchlist"] = watch_count
    return summary


def generate_performance_report(
    performance_reports: dict[str, RollingPerformanceReport],
    demotion_summary: dict[str, Any],
    cfg: MonitorConfig,
    report_dir: str = "data/reports/",
) -> None:
    """
    Writes a formatted layer performance report file.
    """
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now(tz="UTC")
    p = out_dir / f"layer_performance_report_{ts.strftime('%Y%m%d')}.txt"

    healthy: list[RollingPerformanceReport] = []
    watch: list[RollingPerformanceReport] = []
    flagged: list[RollingPerformanceReport] = []
    low_sample: list[RollingPerformanceReport] = []
    for rep in performance_reports.values():
        if rep.rolling_n < int(cfg.min_rolling_sample):
            low_sample.append(rep)
        elif pd.notna(rep.rolling_hit_rate) and rep.rolling_hit_rate < float(cfg.demotion_threshold):
            flagged.append(rep)
        elif pd.notna(rep.rolling_hit_rate) and rep.rolling_hit_rate < float(cfg.watchlist_threshold):
            watch.append(rep)
        else:
            healthy.append(rep)

    def _line(rep: RollingPerformanceReport) -> str:
        arrow = "↓" if rep.rolling_trend == "degrading" else ("↑" if rep.rolling_trend == "improving" else "→")
        return f"  {rep.layer_name} | {rep.market} | Rolling: {rep.rolling_hit_rate:.2%} ({rep.rolling_n}) | Trend: {arrow}"

    lines: list[str] = []
    lines.append(f"LAYER PERFORMANCE MONITOR — {ts.strftime('%Y-%m-%d')}")
    lines.append("═" * 62)
    lines.append(f"ACTIVE LAYERS MONITORED: {len(performance_reports)}")
    lines.append(f"ROLLING WINDOW: {cfg.rolling_window_games} games")
    lines.append("")
    lines.append(f"HEALTHY ({len(healthy)} layers):")
    lines.extend([_line(r) for r in healthy] or ["  none"])
    lines.append("")
    lines.append(f"WATCHLIST ({len(watch)} layers):")
    lines.extend([_line(r) for r in watch] or ["  none"])
    lines.append("")
    lines.append(f"FLAGGED FOR DEMOTION REVIEW ({len(flagged)} layers):")
    lines.extend([_line(r) for r in flagged] or ["  none"])
    lines.append("")
    lines.append(f"INSUFFICIENT ROLLING SAMPLE ({len(low_sample)} layers):")
    lines.extend([_line(r) for r in low_sample] or ["  none"])
    lines.append("")
    missers = [r for r in performance_reports.values() if r.consecutive_misses >= 5]
    lines.append("CONSECUTIVE MISS STREAKS:")
    lines.extend([f"  {r.layer_name} | {r.market} | misses={r.consecutive_misses}" for r in missers] or ["  none"])
    lines.append("═" * 62)
    lines.append(
        f"DEMOTION UPDATE: applied={demotion_summary.get('applied', 0)} "
        f"flagged={demotion_summary.get('flagged_for_review', 0)} watchlist={demotion_summary.get('watchlist', 0)}"
    )
    if demotion_summary.get("output"):
        lines.append(f"Candidates file: {demotion_summary['output']}")
    p.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def run_performance_monitor(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    registry_path: str = "data/layer_registry.csv",
    signals_path: str = "data/plumbing/situational_layer_signals.csv",
    auto_demote: bool = True,
    report_dir: str = "data/reports/",
) -> dict[str, Any]:
    """
    End-to-end monitor for post-promotion layer degradation.
    """
    del signals_path  # Reserved for future per-layer fire logs.
    cfg = MonitorConfig()
    registry = load_registry(registry_path)
    active = get_active_layers(registry)
    if active.empty:
        print("[INFO] No active layers found in registry.")
        return {"active_layers": 0, "reports": 0}

    post_games = load_post_promotion_games(predictions_path=predictions_path, registry=active, cfg=cfg)
    perf: dict[str, RollingPerformanceReport] = {}
    for key, frame in post_games.items():
        parts = key.split("||", 1)
        lname = parts[0]
        market = parts[1] if len(parts) > 1 else "ATS"
        perf[key] = compute_rolling_performance(layer_games=frame, layer_name=lname, market=market, cfg=cfg)

    demotion_candidates = evaluate_demotion_candidates(performance_reports=perf, registry=active, cfg=cfg)
    demotion_summary = apply_demotion_flags(demotion_candidates, registry_path=registry_path, auto_demote=auto_demote)
    generate_performance_report(perf, demotion_summary, cfg=cfg, report_dir=report_dir)
    return {
        "active_layers": len(active),
        "reports": len(perf),
        "demotion_candidates": len(demotion_candidates),
        "demotion_summary": demotion_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="IDES layer performance monitor")
    parser.add_argument("--predictions-path", default="data/reports/game_predictions_master.csv")
    parser.add_argument("--registry-path", default="data/layer_registry.csv")
    parser.add_argument("--signals-path", default="data/plumbing/situational_layer_signals.csv")
    parser.add_argument("--auto-demote", default="true")
    parser.add_argument("--report-dir", default="data/reports/")
    args = parser.parse_args()

    summary = run_performance_monitor(
        predictions_path=args.predictions_path,
        registry_path=args.registry_path,
        signals_path=args.signals_path,
        auto_demote=_parse_bool(args.auto_demote),
        report_dir=args.report_dir,
    )
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
