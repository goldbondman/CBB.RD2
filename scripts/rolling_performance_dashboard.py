from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase_common import ensure_parent, resolve_paths, safe_read_csv, to_num


log = logging.getLogger("rolling_performance_dashboard")
logging.basicConfig(level=logging.INFO, format="[PHASE4] %(levelname)-8s %(message)s")


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return to_num(series).fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _confidence_tier(series: pd.Series) -> pd.Series:
    vals = to_num(series)
    return pd.cut(
        vals,
        bins=[-0.001, 0.60, 0.70, 0.78, 0.85, 1.01],
        labels=["<60%", "60-70%", "70-78%", "78-85%", "85%+"],
    ).astype(str).replace({"nan": "unknown"})


def _seed_matchup(frame: pd.DataFrame) -> pd.Series:
    a = to_num(frame.get("team_a_seed", pd.Series(np.nan, index=frame.index)))
    b = to_num(frame.get("team_b_seed", pd.Series(np.nan, index=frame.index)))
    low = np.minimum(a, b)
    high = np.maximum(a, b)
    text = np.where(low.notna() & high.notna(), low.astype("Int64").astype(str) + "v" + high.astype("Int64").astype(str), "unknown")
    return pd.Series(text, index=frame.index)


def _split_layers(series: pd.Series) -> pd.Series:
    def _norm(v: Any) -> str:
        if pd.isna(v):
            return "none"
        txt = str(v).strip()
        if not txt:
            return "none"
        for delim in ["|", ";", ","]:
            if delim in txt:
                return delim.join([t.strip() for t in txt.split(delim) if t.strip()]) or "none"
        return txt

    return series.apply(_norm)


def _derive_hits(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    df["game_datetime_utc"] = pd.to_datetime(df.get("game_start_datetime_utc"), errors="coerce", utc=True)
    if "game_datetime_utc" not in df.columns:
        df["game_datetime_utc"] = pd.to_datetime(df.get("game_datetime_utc"), errors="coerce", utc=True)
    if "phase" not in df.columns:
        df["phase"] = "unknown"

    covered_a = to_num(df.get("covered_team_a", pd.Series(np.nan, index=df.index)))
    covered_b = to_num(df.get("covered_team_b", pd.Series(np.nan, index=df.index)))
    spread_pick = df.get("spread_pick", pd.Series("", index=df.index)).astype(str)
    team_a = df.get("team_a", pd.Series("", index=df.index)).astype(str)
    model_on_a = spread_pick.eq(team_a)
    df["spread_hit"] = np.where(model_on_a, covered_a, covered_b)
    df["spread_hit"] = to_num(pd.Series(df["spread_hit"], index=df.index))

    total_pick = df.get("total_pick", pd.Series("", index=df.index)).astype(str).str.lower()
    over_hit = to_num(df.get("over_hit", pd.Series(np.nan, index=df.index)))
    under_hit = to_num(df.get("under_hit", pd.Series(np.nan, index=df.index)))
    total_result = df.get("total_result", pd.Series("", index=df.index)).astype(str).str.lower()
    total_result_known = total_result.isin({"over", "under"})
    df["total_hit"] = np.where(
        total_pick.str.contains("over"),
        np.where(
            over_hit.notna(),
            over_hit,
            np.where(total_result_known, (total_result == "over").astype(float), np.nan),
        ),
        np.where(
            total_pick.str.contains("under"),
            np.where(
                under_hit.notna(),
                under_hit,
                np.where(total_result_known, (total_result == "under").astype(float), np.nan),
            ),
            np.nan,
        ),
    )
    df["total_hit"] = to_num(pd.Series(df["total_hit"], index=df.index))

    model_pick = df.get("model_pick", df.get("prediction_winner", pd.Series("", index=df.index))).astype(str)
    actual_winner = df.get("actual_winner", pd.Series("", index=df.index)).astype(str)
    margin = to_num(df.get("final_margin", pd.Series(np.nan, index=df.index)))
    ml_hit = np.where(
        actual_winner.astype(str).str.len() > 0,
        (model_pick == actual_winner).astype(float),
        np.where(
            margin.notna(),
            np.where(
                model_pick == team_a,
                (margin > 0).astype(float),
                (margin < 0).astype(float),
            ),
            np.nan,
        ),
    )
    df["ml_hit"] = to_num(pd.Series(ml_hit, index=df.index))

    df["confidence_tier"] = _confidence_tier(df.get("spread_confidence", pd.Series(np.nan, index=df.index)))
    df["total_conf_tier"] = _confidence_tier(df.get("total_confidence", pd.Series(np.nan, index=df.index)))
    df["situational_layer"] = _split_layers(df.get("situational_layer_applied", pd.Series("none", index=df.index)))
    df["pes_tier"] = df.get("pes_quadrant", pd.Series("unknown", index=df.index)).astype(str).replace({"nan": "unknown"})
    df["seed_matchup"] = _seed_matchup(df)
    return df


def _long_bets(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    common_cols = [
        "game_id",
        "game_datetime_utc",
        "season",
        "phase",
        "confidence_tier",
        "situational_layer",
        "pes_tier",
        "seed_matchup",
        "spread_edge_for_pick",
        "final_margin",
        "market_spread",
    ]
    for col in common_cols:
        if col not in df.columns:
            df[col] = np.nan

    spread = df.copy()
    spread["market"] = "spread"
    spread["bet_type"] = "ats"
    spread["hit"] = to_num(spread.get("spread_hit", pd.Series(np.nan, index=spread.index)))
    spread["confidence_bucket"] = spread["confidence_tier"]
    rows.append(spread[common_cols + ["market", "bet_type", "hit", "confidence_bucket"]])

    total = df.copy()
    total["market"] = "total"
    total["bet_type"] = "ou"
    total["hit"] = to_num(total.get("total_hit", pd.Series(np.nan, index=total.index)))
    total["confidence_bucket"] = total["total_conf_tier"]
    rows.append(total[common_cols + ["market", "bet_type", "hit", "confidence_bucket"]])

    moneyline = df.copy()
    moneyline["market"] = "moneyline"
    moneyline["bet_type"] = "ml"
    moneyline["hit"] = to_num(moneyline.get("ml_hit", pd.Series(np.nan, index=moneyline.index)))
    moneyline["confidence_bucket"] = moneyline["confidence_tier"]
    rows.append(moneyline[common_cols + ["market", "bet_type", "hit", "confidence_bucket"]])

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out[out["hit"].isin([0, 1])].copy()
    out = out.sort_values("game_datetime_utc")
    return out


def _window_rows(frame: pd.DataFrame, label: str, subset: pd.DataFrame, dimension: str, value: str) -> dict[str, Any]:
    n = int(len(subset))
    wins = int(to_num(subset["hit"]).sum()) if n else 0
    return {
        "report_date_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dimension": dimension,
        "dimension_value": value,
        "window": label,
        "n": n,
        "wins": wins,
        "hit_rate": (wins / n) if n else np.nan,
    }


def build_performance_tracker(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["report_date_utc", "dimension", "dimension_value", "window", "n", "wins", "hit_rate"])

    windows = {
        "last_10": 10,
        "last_25": 25,
        "last_50": 50,
        "last_100": 100,
    }

    rows: list[dict[str, Any]] = []
    dimensions = [
        ("overall", pd.Series("all", index=df_long.index)),
        ("market", df_long["market"].astype(str)),
        ("bet_type", df_long["bet_type"].astype(str)),
        ("confidence_tier", df_long["confidence_bucket"].astype(str)),
        ("situational_layer", df_long["situational_layer"].astype(str)),
        ("pes_tier", df_long["pes_tier"].astype(str)),
        ("seed_matchup", df_long["seed_matchup"].astype(str)),
        ("phase", df_long["phase"].astype(str)),
    ]

    for dim_name, dim_values in dimensions:
        for value, part in df_long.groupby(dim_values, dropna=False):
            part = part.sort_values("game_datetime_utc")
            for label, nrows in windows.items():
                rows.append(_window_rows(df_long, label, part.tail(nrows), dim_name, str(value)))
            # season-to-date by latest season in subset
            if "season" in part.columns and part["season"].notna().any():
                latest_season = int(to_num(part["season"]).dropna().max())
                season_part = part[to_num(part["season"]) == latest_season]
                rows.append(_window_rows(df_long, "season_to_date", season_part, dim_name, str(value)))
            else:
                rows.append(_window_rows(df_long, "season_to_date", pd.DataFrame(columns=part.columns), dim_name, str(value)))
            rows.append(_window_rows(df_long, "all_time", part, dim_name, str(value)))

    out = pd.DataFrame(rows)
    return out.sort_values(["dimension", "dimension_value", "window"]).reset_index(drop=True)


def build_edge_vs_actual(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["spread_edge_for_pick"] = to_num(work.get("spread_edge_for_pick", pd.Series(np.nan, index=work.index)))
    work["market_spread"] = to_num(work.get("market_spread", pd.Series(np.nan, index=work.index)))
    work["final_margin"] = to_num(work.get("final_margin", pd.Series(np.nan, index=work.index)))
    work["spread_hit"] = to_num(work.get("spread_hit", pd.Series(np.nan, index=work.index)))
    valid = work[
        work["spread_edge_for_pick"].notna() & work["market_spread"].notna() & work["final_margin"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame([{"status": "BLOCKED", "reason": "Missing spread edge/market/outcome columns"}])
    valid["edge_bucket"] = pd.cut(
        valid["spread_edge_for_pick"].abs(),
        bins=[-0.001, 2, 4, 6, 10, 1000],
        labels=["0-2", "2-4", "4-6", "6-10", "10+"],
    )
    out = (
        valid.groupby("edge_bucket", dropna=False)
        .agg(
            n=("spread_hit", "size"),
            spread_hit_rate=("spread_hit", "mean"),
            avg_abs_edge=("spread_edge_for_pick", lambda s: float(np.nanmean(np.abs(s)))),
            avg_realized_margin=("final_margin", "mean"),
        )
        .reset_index()
    )
    return out


def build_clv_tracker(paths) -> pd.DataFrame:
    clv = safe_read_csv(paths.rd2_data / "plumbing" / "line_movement.csv")
    if clv.empty:
        return pd.DataFrame(
            [
                {
                    "status": "PROVISIONAL",
                    "reason": "line_movement.csv unavailable",
                    "required_columns": "clv_spread_team_a, clv_spread_team_b, clv_total_over, clv_total_under",
                }
            ]
        )
    out_rows = []
    for col, market in [
        ("clv_spread_team_a", "spread"),
        ("clv_spread_team_b", "spread"),
        ("clv_total_over", "total"),
        ("clv_total_under", "total"),
    ]:
        if col not in clv.columns:
            continue
        vals = to_num(clv[col]).dropna()
        if vals.empty:
            continue
        out_rows.append(
            {
                "market": market,
                "metric": col,
                "n": int(len(vals)),
                "mean_clv": float(vals.mean()),
                "median_clv": float(vals.median()),
                "positive_clv_rate": float((vals > 0).mean()),
            }
        )
    if not out_rows:
        return pd.DataFrame([{"status": "PROVISIONAL", "reason": "CLV columns present but all null"}])
    return pd.DataFrame(out_rows)


def write_weekly_summary(path: Path, tracker: pd.DataFrame, edge_df: pd.DataFrame, clv_df: pd.DataFrame) -> None:
    ensure_parent(path)
    lines = [
        f"Weekly Performance Summary ({pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')})",
        "=" * 72,
        "",
    ]
    if tracker.empty:
        lines.append("No graded bets available in performance tracker.")
    else:
        focus = tracker[(tracker["dimension"] == "overall") & (tracker["dimension_value"] == "all")]
        for w in ["last_10", "last_25", "last_50", "last_100", "season_to_date", "all_time"]:
            row = focus[focus["window"] == w]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(f"{w:>13}: n={int(r['n']):>4d} | hit_rate={r['hit_rate']:.3f}")
    lines.extend(["", "Edge vs Actual Margin"])
    if edge_df.empty:
        lines.append("- unavailable")
    else:
        lines.append(edge_df.head(10).to_string(index=False))
    lines.extend(["", "CLV Tracking"])
    if clv_df.empty:
        lines.append("- unavailable")
    else:
        lines.append(clv_df.to_string(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_layer_monitor_hook(paths) -> tuple[str, str]:
    script = paths.rd2_root / "layer_performance_monitor.py"
    if not script.exists():
        return "SKIPPED", f"{script} missing"
    cmd = [
        sys.executable,
        str(script),
        "--predictions-path",
        str(paths.predictions_master_csv),
        "--registry-path",
        str(paths.layer_registry_csv),
        "--signals-path",
        str(paths.rd2_data / "plumbing" / "situational_layer_signals.csv"),
        "--auto-demote",
        "true",
        "--report-dir",
        str(paths.rd2_data / "reports"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return "PASS", "layer_performance_monitor.py executed"
    return "WARN", f"layer_performance_monitor.py exit={proc.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 rolling performance dashboard and tracker")
    parser.add_argument("--predictions-path", default="data/reports/game_predictions_master.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--run-layer-monitor", default="true")
    args = parser.parse_args()

    paths = resolve_paths()
    pred_path = Path(args.predictions_path)
    pred_path = pred_path if pred_path.is_absolute() else paths.rd2_root / pred_path
    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else paths.rd2_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = safe_read_csv(pred_path)
    if preds.empty:
        log.error("Predictions master missing/empty: %s", pred_path)
        return 2

    derived = _derive_hits(preds)
    long_bets = _long_bets(derived)
    tracker = build_performance_tracker(long_bets)
    tracker_path = out_dir / "performance_tracker.csv"
    tracker.to_csv(tracker_path, index=False)

    edge_df = build_edge_vs_actual(derived)
    edge_path = out_dir / "edge_vs_actual_margin.csv"
    edge_df.to_csv(edge_path, index=False)

    clv_df = build_clv_tracker(paths)
    clv_path = out_dir / "clv_tracker.csv"
    clv_df.to_csv(clv_path, index=False)

    summary_path = out_dir / f"weekly_performance_summary_{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}.txt"
    write_weekly_summary(summary_path, tracker, edge_df, clv_df)

    if str(args.run_layer_monitor).strip().lower() in {"1", "true", "t", "yes", "y"}:
        status, note = run_layer_monitor_hook(paths)
        log.info("Layer monitor hook: %s | %s", status, note)

    log.info("performance_tracker -> %s", tracker_path)
    log.info("weekly_summary -> %s", summary_path)
    log.info("edge_vs_actual -> %s", edge_path)
    log.info("clv_tracker -> %s", clv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
