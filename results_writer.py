from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ides_of_march.orchestrator import MODEL_VERSION as IDES_MODEL_VERSION
from ides_of_march.utils import canonical_id, safe_read_csv, utc_now_iso

try:
    # Reuse existing completion logic used by the IDES data pipeline.
    from ides_of_march.data_steward import _to_bool_completed as steward_completed_mask
except Exception:  # pragma: no cover
    steward_completed_mask = None


PST = ZoneInfo("America/Los_Angeles")
IDES_RUN_PREFIXES = ("ides_predict_", "ides_backtest_", "ides_audit_")


@dataclass(frozen=True)
class ResultsConfig:
    game_id: str = "game_id"
    game_date_pst: str = "game_date_pst"
    team_a: str = "team_a"
    team_b: str = "team_b"
    model_spread_team_a: str = "model_spread_team_a"
    model_spread_team_b: str = "model_spread_team_b"
    market_spread: str = "market_spread"
    market_spread_team_a: str = "market_spread_team_a"
    market_total: str = "market_total"
    model_total: str = "model_total"
    prediction_winner: str = "prediction_winner"
    final_score_team_a: str = "final_score_team_a"
    final_score_team_b: str = "final_score_team_b"
    final_margin: str = "final_margin"
    actual_winner: str = "actual_winner"
    covered_team_a: str = "covered_team_a"
    covered_team_b: str = "covered_team_b"
    total_result: str = "total_result"
    over_hit: str = "over_hit"
    under_hit: str = "under_hit"
    result_written_at_utc: str = "result_written_at_utc"
    result_source: str = "result_source"
    updated_at_utc: str = "updated_at_utc"
    season: str = "season"
    is_ncaa_tournament: str = "is_ncaa_tournament"
    is_conference_tournament: str = "is_conference_tournament"
    model_version: str = "model_version"
    run_id: str = "run_id"
    phase: str = "phase"
    game_status: str = "game_status"

    # Source columns in data/games.csv (same source used by ESPN updater).
    source_game_id: str = "game_id"
    source_event_id: str = "event_id"
    source_game_datetime_utc: str = "game_datetime_utc"
    source_game_datetime_pst: str = "game_datetime_pst"
    source_date: str = "date"
    source_home_team: str = "home_team"
    source_away_team: str = "away_team"
    source_home_score: str = "home_score"
    source_away_score: str = "away_score"
    source_completed: str = "completed"
    source_state: str = "state"
    source_status_desc: str = "status_desc"
    source_name: str = "source"


@dataclass
class ResultsWriteReport:
    games_updated: int = 0
    games_not_found: list[str] = field(default_factory=list)
    games_already_had_results: int = 0
    push_count: int = 0
    errors: list[str] = field(default_factory=list)


def _num(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _is_nan(value: Any) -> bool:
    return pd.isna(value) or (isinstance(value, float) and np.isnan(value))


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _ides_mask(df: pd.DataFrame, cfg: ResultsConfig) -> tuple[pd.Series, str]:
    if df.empty:
        return pd.Series(dtype=bool), "empty_frame"

    if cfg.model_version in df.columns:
        mv = df[cfg.model_version].astype(str).str.strip()
        model_mask = mv.eq(IDES_MODEL_VERSION)
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


def _pst_date_from_source(games: pd.DataFrame, cfg: ResultsConfig) -> pd.Series:
    if cfg.source_game_datetime_pst in games.columns:
        dt = pd.to_datetime(games[cfg.source_game_datetime_pst], errors="coerce")
        return dt.dt.strftime("%Y-%m-%d")
    if cfg.source_game_datetime_utc in games.columns:
        dt = pd.to_datetime(games[cfg.source_game_datetime_utc], utc=True, errors="coerce").dt.tz_convert(PST)
        return dt.dt.strftime("%Y-%m-%d")
    if cfg.source_date in games.columns:
        d = games[cfg.source_date].astype(str).str.strip()
        dt = pd.to_datetime(d, format="%Y%m%d", errors="coerce")
        return dt.dt.strftime("%Y-%m-%d")
    return pd.Series(np.nan, index=games.index, dtype=object)


def fetch_final_scores(
    game_ids: list[str],
    game_dates: list[str],
    data_dir: str = "data/",
) -> pd.DataFrame:
    """
    Fetches final scores for completed games.
    Uses data/games.csv, which is populated by the existing ESPN update pipeline.
    """
    cfg = ResultsConfig()
    games_path = Path(data_dir) / "games.csv"
    games = safe_read_csv(games_path)
    if games.empty:
        return pd.DataFrame(
            columns=[
                cfg.game_id,
                cfg.final_score_team_a,
                cfg.final_score_team_b,
                cfg.final_margin,
                cfg.actual_winner,
                cfg.result_source,
                cfg.source_home_team,
                cfg.source_away_team,
                cfg.source_status_desc,
                cfg.source_state,
            ]
        )

    if cfg.source_game_id not in games.columns:
        if cfg.source_event_id in games.columns:
            games[cfg.source_game_id] = games[cfg.source_event_id]
        else:
            return pd.DataFrame()

    games[cfg.source_game_id] = games[cfg.source_game_id].map(canonical_id)
    wanted_ids = {canonical_id(g) for g in game_ids if canonical_id(g)}
    if wanted_ids:
        games = games[games[cfg.source_game_id].isin(wanted_ids)].copy()
    if games.empty:
        return pd.DataFrame()

    games["_date_pst"] = _pst_date_from_source(games, cfg)
    wanted_dates = {str(d) for d in game_dates if str(d)}
    if wanted_dates:
        games = games[games["_date_pst"].isin(wanted_dates)].copy()
    if games.empty:
        return pd.DataFrame()

    if steward_completed_mask is not None:
        completed_mask = steward_completed_mask(games)
    elif cfg.source_completed in games.columns:
        completed_mask = _to_bool_series(games[cfg.source_completed])
    else:
        completed_mask = pd.Series(False, index=games.index, dtype=bool)

    status_text = (
        games.get(cfg.source_status_desc, pd.Series("", index=games.index)).fillna("").astype(str)
        + " "
        + games.get(cfg.source_state, pd.Series("", index=games.index)).fillna("").astype(str)
    ).str.lower()
    blocked_status = status_text.str.contains("postpon|cancel|suspend|abandon|delayed", regex=True)
    final_text = status_text.str.contains("final", regex=False)
    complete = (completed_mask | final_text) & (~blocked_status)

    games[cfg.source_home_score] = pd.to_numeric(games.get(cfg.source_home_score), errors="coerce")
    games[cfg.source_away_score] = pd.to_numeric(games.get(cfg.source_away_score), errors="coerce")
    has_scores = games[cfg.source_home_score].notna() & games[cfg.source_away_score].notna()
    final_games = games[complete & has_scores].copy()
    if final_games.empty:
        return pd.DataFrame()

    final_games[cfg.final_margin] = final_games[cfg.source_home_score] - final_games[cfg.source_away_score]
    final_games[cfg.actual_winner] = np.where(
        final_games[cfg.final_margin] > 0,
        final_games.get(cfg.source_home_team, ""),
        np.where(final_games[cfg.final_margin] < 0, final_games.get(cfg.source_away_team, ""), "tie"),
    )
    source_label = (
        "data/games.csv"
        + np.where(
            final_games.get(cfg.source_name, pd.Series("", index=final_games.index)).fillna("").astype(str).ne(""),
            ":" + final_games.get(cfg.source_name, "").astype(str),
            "",
        )
    )
    final_games[cfg.result_source] = source_label

    return final_games[
        [
            cfg.source_game_id,
            cfg.source_home_team,
            cfg.source_away_team,
            cfg.source_home_score,
            cfg.source_away_score,
            cfg.final_margin,
            cfg.actual_winner,
            cfg.result_source,
            cfg.source_status_desc,
            cfg.source_state,
        ]
    ].rename(
        columns={
            cfg.source_game_id: cfg.game_id,
            cfg.source_home_score: cfg.final_score_team_a,
            cfg.source_away_score: cfg.final_score_team_b,
        }
    )


def compute_betting_outcomes(
    predictions_row: pd.Series,
    scores_row: pd.Series,
    cfg: ResultsConfig,
) -> dict:
    """
    Given one prediction row and one score row, compute ATS/SU/total outcomes.
    """
    team_a_name = str(predictions_row.get(cfg.team_a, ""))
    team_b_name = str(predictions_row.get(cfg.team_b, ""))
    src_home = str(scores_row.get(cfg.source_home_team, ""))
    src_away = str(scores_row.get(cfg.source_away_team, ""))

    score_a = _num(scores_row.get(cfg.final_score_team_a))
    score_b = _num(scores_row.get(cfg.final_score_team_b))
    if (
        team_a_name
        and team_b_name
        and src_home
        and src_away
        and team_a_name == src_away
        and team_b_name == src_home
    ):
        score_a, score_b = score_b, score_a

    final_margin = score_a - score_b
    actual_winner = team_a_name if score_a > score_b else (team_b_name if score_b > score_a else "tie")

    market_spread = pd.to_numeric(pd.Series([predictions_row.get(cfg.market_spread)]), errors="coerce").iloc[0]
    if _is_nan(market_spread) and cfg.market_spread_team_a in predictions_row.index:
        market_spread = pd.to_numeric(pd.Series([predictions_row.get(cfg.market_spread_team_a)]), errors="coerce").iloc[0]

    covered_team_a: float | bool = np.nan
    covered_team_b: float | bool = np.nan
    spread_push = False
    if not _is_nan(market_spread):
        if final_margin == float(market_spread):
            spread_push = True
        else:
            covered_team_a = bool(final_margin > float(market_spread))
            covered_team_b = bool(not covered_team_a)

    market_total = pd.to_numeric(pd.Series([predictions_row.get(cfg.market_total)]), errors="coerce").iloc[0]
    actual_total = score_a + score_b
    total_result = "no_line"
    over_hit: float | int = np.nan
    under_hit: float | int = np.nan
    if not _is_nan(market_total):
        if actual_total == float(market_total):
            total_result = "push"
            over_hit = 0
            under_hit = 0
        elif actual_total > float(market_total):
            total_result = "over"
            over_hit = 1
            under_hit = 0
        else:
            total_result = "under"
            over_hit = 0
            under_hit = 1

    model_side_a = np.nan
    model_spread = pd.to_numeric(pd.Series([predictions_row.get(cfg.model_spread_team_a)]), errors="coerce").iloc[0]
    if not _is_nan(model_spread) and not _is_nan(market_spread):
        model_side_a = bool(float(model_spread) > float(market_spread))
    ats_hit = np.nan
    if not _is_nan(model_side_a) and not _is_nan(covered_team_a) and not spread_push:
        ats_hit = int((model_side_a and bool(covered_team_a)) or ((not model_side_a) and bool(covered_team_b)))

    total_hit = np.nan
    model_total = pd.to_numeric(pd.Series([predictions_row.get(cfg.model_total)]), errors="coerce").iloc[0]
    if not _is_nan(model_total) and not _is_nan(market_total) and total_result in {"over", "under"}:
        model_over = bool(float(model_total) > float(market_total))
        total_hit = int((model_over and total_result == "over") or ((not model_over) and total_result == "under"))

    return {
        cfg.final_score_team_a: int(round(score_a)),
        cfg.final_score_team_b: int(round(score_b)),
        cfg.final_margin: float(final_margin),
        cfg.actual_winner: actual_winner,
        cfg.covered_team_a: covered_team_a,
        cfg.covered_team_b: covered_team_b,
        cfg.total_result: total_result,
        cfg.over_hit: over_hit,
        cfg.under_hit: under_hit,
        cfg.result_source: scores_row.get(cfg.result_source, "data/games.csv"),
        "_spread_push": spread_push,
        "_ats_hit": ats_hit,
        "_total_hit": total_hit,
    }


def identify_games_needing_results(
    predictions_df: pd.DataFrame,
    cfg: ResultsConfig,
    lookback_days: int = 3,
) -> pd.DataFrame:
    """
    Identify IDES rows in the past window with missing result fields.
    """
    if predictions_df.empty:
        return predictions_df.copy()

    out = predictions_df.copy()
    ides_mask, _ = _ides_mask(out, cfg)
    if ides_mask.any():
        out = out.loc[ides_mask].copy()
    else:
        return out.iloc[0:0].copy()

    if cfg.game_date_pst not in out.columns:
        return out.iloc[0:0].copy()

    game_dates = pd.to_datetime(out[cfg.game_date_pst], errors="coerce")
    today_pst = pd.Timestamp.now(tz=PST).normalize().tz_localize(None)
    lower_bound = today_pst - pd.Timedelta(days=max(int(lookback_days), 0))
    in_past = game_dates < today_pst
    within_window = game_dates >= lower_bound

    if cfg.final_score_team_a in out.columns:
        score_a = pd.to_numeric(out[cfg.final_score_team_a], errors="coerce")
    else:
        score_a = pd.Series(np.nan, index=out.index)
    missing_score = score_a.isna() | score_a.eq(0)

    if cfg.game_status in out.columns:
        status = out[cfg.game_status].fillna("").astype(str).str.lower()
        not_blocked = ~status.str.contains("postpon|cancel", regex=True)
    else:
        not_blocked = pd.Series(True, index=out.index, dtype=bool)

    mask = in_past & within_window & missing_score & not_blocked
    return out.loc[mask].copy()


def _result_columns(cfg: ResultsConfig) -> list[str]:
    return [
        cfg.final_score_team_a,
        cfg.final_score_team_b,
        cfg.final_margin,
        cfg.actual_winner,
        cfg.covered_team_a,
        cfg.covered_team_b,
        cfg.total_result,
        cfg.over_hit,
        cfg.under_hit,
        cfg.result_written_at_utc,
        cfg.result_source,
        cfg.updated_at_utc,
    ]


def _row_has_confirmed_results(row: pd.Series, cfg: ResultsConfig) -> bool:
    score_a = pd.to_numeric(pd.Series([row.get(cfg.final_score_team_a)]), errors="coerce").iloc[0]
    score_b = pd.to_numeric(pd.Series([row.get(cfg.final_score_team_b)]), errors="coerce").iloc[0]
    actual_winner = str(row.get(cfg.actual_winner, "")).strip().lower()
    return pd.notna(score_a) and pd.notna(score_b) and actual_winner not in {"", "nan", "none"}


def write_results_to_master(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    outcomes: pd.DataFrame | None = None,
    cfg: ResultsConfig | None = None,
    dry_run: bool = False,
) -> ResultsWriteReport:
    """
    Write computed outcomes to IDES rows in game_predictions_master.csv.
    """
    config = cfg or ResultsConfig()
    report = ResultsWriteReport()
    master_path = Path(predictions_path)
    master = safe_read_csv(master_path)
    if master.empty:
        report.errors.append(f"missing_or_empty_master:{master_path}")
        return report

    if config.game_id not in master.columns:
        report.errors.append(f"missing_column:{config.game_id}")
        return report

    master[config.game_id] = master[config.game_id].map(canonical_id)
    ides_mask, ides_method = _ides_mask(master, config)
    if not ides_mask.any():
        report.errors.append("ides_scope_not_identifiable:model_version_or_run_id_missing")
        return report

    for col in _result_columns(config):
        if col not in master.columns:
            master[col] = np.nan

    if outcomes is None or outcomes.empty:
        return report

    outcomes_local = outcomes.copy()
    outcomes_local[config.game_id] = outcomes_local[config.game_id].map(canonical_id)
    ides_master = master.loc[ides_mask].copy()
    idx_by_game = ides_master.groupby(config.game_id).groups

    now_utc = utc_now_iso()
    update_cols = _result_columns(config)
    for _, out_row in outcomes_local.iterrows():
        gid = str(out_row.get(config.game_id, "")).strip()
        if not gid:
            continue
        if gid not in idx_by_game:
            report.games_not_found.append(gid)
            continue

        for idx in list(idx_by_game[gid]):
            current = master.loc[idx]
            if _row_has_confirmed_results(current, config):
                report.games_already_had_results += 1
                continue

            for col in update_cols:
                if col == config.result_written_at_utc:
                    master.at[idx, col] = now_utc
                elif col == config.updated_at_utc:
                    master.at[idx, col] = now_utc
                elif col in out_row.index:
                    master.at[idx, col] = out_row[col]

            spread_push = bool(out_row.get("_spread_push", False))
            total_push = str(out_row.get(config.total_result, "")).lower() == "push"
            if spread_push or total_push:
                report.push_count += 1
            report.games_updated += 1

    if dry_run:
        print(f"[DRY-RUN] scope={ides_method} games_updated={report.games_updated}")
        return report

    master.to_csv(master_path, index=False)
    print(f"[INFO] scope={ides_method} wrote={report.games_updated} -> {master_path}")
    return report


def run_results_writer(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    data_dir: str = "data/",
    lookback_days: int = 3,
    dry_run: bool = False,
) -> ResultsWriteReport:
    """
    End-to-end writer:
      1) load master, 2) filter IDES rows, 3) find missing results,
      4) fetch finals, 5) compute outcomes, 6) write to master.
    """
    cfg = ResultsConfig()
    master = safe_read_csv(Path(predictions_path))
    report = ResultsWriteReport()
    if master.empty:
        report.errors.append("missing_or_empty_master")
        return report

    if cfg.game_id not in master.columns:
        report.errors.append(f"missing_column:{cfg.game_id}")
        return report

    master[cfg.game_id] = master[cfg.game_id].map(canonical_id)
    ides_mask, method = _ides_mask(master, cfg)
    if not ides_mask.any():
        report.errors.append("ides_scope_not_identifiable:model_version_or_run_id_missing")
        print("[FAIL] Could not identify IDES rows; refusing to update.")
        return report

    ides_rows = master.loc[ides_mask].copy()
    needed = identify_games_needing_results(ides_rows, cfg, lookback_days=lookback_days)
    if needed.empty:
        print("[INFO] No IDES games need results.")
        return report

    game_ids = needed[cfg.game_id].astype(str).tolist()
    game_dates = needed[cfg.game_date_pst].astype(str).tolist() if cfg.game_date_pst in needed.columns else []
    score_rows = fetch_final_scores(game_ids=game_ids, game_dates=game_dates, data_dir=data_dir)
    if score_rows.empty:
        report.errors.append("no_completed_scores_found")
        print("[WARN] No completed scores returned for games needing results.")
        return report

    score_lookup = score_rows.drop_duplicates(subset=[cfg.game_id], keep="last").set_index(cfg.game_id)
    outcomes_records: list[dict[str, Any]] = []
    for _, pred_row in needed.iterrows():
        gid = str(pred_row.get(cfg.game_id, ""))
        if gid not in score_lookup.index:
            continue
        score_row = score_lookup.loc[gid]
        if isinstance(score_row, pd.DataFrame):
            score_row = score_row.iloc[-1]
        outcome = compute_betting_outcomes(pred_row, score_row, cfg)
        outcome[cfg.game_id] = gid
        outcomes_records.append(outcome)

    if not outcomes_records:
        report.errors.append("no_outcomes_computed")
        print("[WARN] No outcomes computed from fetched final scores.")
        return report

    outcomes = pd.DataFrame(outcomes_records)
    report = write_results_to_master(
        predictions_path=predictions_path,
        outcomes=outcomes,
        cfg=cfg,
        dry_run=dry_run,
    )

    ats_series = pd.to_numeric(outcomes.get("_ats_hit"), errors="coerce")
    total_series = pd.to_numeric(outcomes.get("_total_hit"), errors="coerce")
    ats_hit_rate = float(ats_series.mean()) if ats_series.notna().any() else np.nan
    total_hit_rate = float(total_series.mean()) if total_series.notna().any() else np.nan

    print("RESULTS WRITER SUMMARY")
    print("=" * 60)
    print(f"scope={method}")
    print(f"games_needing_results={len(needed)}")
    print(f"completed_scores_found={len(score_rows)}")
    print(f"outcomes_computed={len(outcomes)}")
    print(f"games_updated={report.games_updated}")
    print(f"games_already_had_results={report.games_already_had_results}")
    print(f"games_not_found={len(report.games_not_found)}")
    print(f"push_count={report.push_count}")
    print(f"ats_hit_rate_new={ats_hit_rate:.2%}" if pd.notna(ats_hit_rate) else "ats_hit_rate_new=n/a")
    print(f"totals_hit_rate_new={total_hit_rate:.2%}" if pd.notna(total_hit_rate) else "totals_hit_rate_new=n/a")
    if report.errors:
        print(f"errors={report.errors}")
    print("=" * 60)

    return report


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "t", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Write final game results into IDES game_predictions_master.csv")
    parser.add_argument("--predictions-path", default="data/reports/game_predictions_master.csv")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--lookback-days", type=int, default=3)
    parser.add_argument("--dry-run", default="false")
    args = parser.parse_args()

    report = run_results_writer(
        predictions_path=args.predictions_path,
        data_dir=args.data_dir,
        lookback_days=args.lookback_days,
        dry_run=_parse_bool(args.dry_run),
    )
    if report.errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
