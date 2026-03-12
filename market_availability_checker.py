from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_pes_columns import discover_ides_rows
from ides_of_march.utils import canonical_id, safe_read_csv, utc_now_iso
from line_movement_tracker import fetch_current_lines


@dataclass(frozen=True)
class MarketConfig:
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
    market_available: str = "market_available"
    market_available_at_utc: str = "market_available_at_utc"
    market_suppressed: str = "market_suppressed"
    suppression_reason: str = "suppression_reason"
    final_bet_flag: str = "final_bet_flag"
    recommended_stake_units: str = "recommended_stake_units"
    bet_rank: str = "bet_rank"
    updated_at_utc: str = "updated_at_utc"
    is_ncaa_tournament: str = "is_ncaa_tournament"
    is_conference_tournament: str = "is_conference_tournament"
    stale_line_threshold_hours: float = 12.0

    # Existing columns used by this checker
    run_id: str = "run_id"
    bet_reason_short: str = "bet_reason_short"
    notes: str = "notes"

    # Columns produced by line_movement_tracker
    line_close_spread: str = "line_close_spread"
    line_close_total: str = "line_close_total"
    line_close_ml_team_a: str = "line_close_ml_team_a"
    line_close_ml_team_b: str = "line_close_ml_team_b"
    line_fetched_at_utc: str = "line_fetched_at_utc"
    spread_movement: str = "spread_movement"

    # Reversible suppression memory columns
    pre_suppression_final_bet_flag: str = "pre_suppression_final_bet_flag"
    pre_suppression_recommended_stake_units: str = "pre_suppression_recommended_stake_units"
    pre_suppression_bet_rank: str = "pre_suppression_bet_rank"


@dataclass
class SuppressionReport:
    total_games_checked: int = 0
    available_count: int = 0
    stale_count: int = 0
    unavailable_count: int = 0
    suppressed_count: int = 0
    warned_count: int = 0
    unsuppressed_count: int = 0
    suppressed_game_ids: list[str] = field(default_factory=list)
    warned_game_ids: list[str] = field(default_factory=list)
    came_online_game_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bool(series: Any) -> pd.Series:
    s = pd.Series(series)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return _num(s).fillna(0).astype(float) != 0
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "t", "yes", "y"}


def _append_reason(existing: Any, prefix: str, reason: str) -> str:
    current = "" if pd.isna(existing) else str(existing).strip()
    tag = f"{prefix}{reason}"
    if not current:
        return tag
    if tag in current:
        return current
    return f"{tag} | {current}"


def check_line_availability(
    game_ids: list[str],
    data_dir: str = "data/",
) -> pd.DataFrame:
    """
    Checks whether current lines exist and are fresh for each game.
    """
    cfg = MarketConfig()
    now = pd.Timestamp.now(tz="UTC")
    ids = [canonical_id(g) for g in game_ids if canonical_id(g)]
    base = pd.DataFrame({cfg.game_id: pd.Series(ids).drop_duplicates()})
    if base.empty:
        return pd.DataFrame(columns=[cfg.game_id, "market_status", cfg.market_available, cfg.market_available_at_utc, "details"])

    lines = fetch_current_lines(game_ids=ids, data_dir=data_dir)
    if lines.empty:
        out = base.copy()
        out[cfg.market_available] = False
        out["market_status"] = "unavailable"
        out[cfg.market_available_at_utc] = np.nan
        out["details"] = "no_line_rows_found"
        return out

    lines = lines.copy()
    lines[cfg.game_id] = lines[cfg.game_id].map(canonical_id)
    out = base.merge(lines, on=cfg.game_id, how="left")

    spread_ok = _num(out.get(cfg.line_close_spread)).notna() & _num(out.get(cfg.line_close_spread)).ne(0)
    total_ok = _num(out.get(cfg.line_close_total)).notna() & _num(out.get(cfg.line_close_total)).ne(0)
    ml_a_ok = _num(out.get(cfg.line_close_ml_team_a)).notna()
    ml_b_ok = _num(out.get(cfg.line_close_ml_team_b)).notna()
    has_line = spread_ok & total_ok & ml_a_ok & ml_b_ok

    fetched = pd.to_datetime(out.get(cfg.line_fetched_at_utc), utc=True, errors="coerce")
    age_hours = ((now - fetched).dt.total_seconds() / 3600.0).astype(float)
    stale = has_line & fetched.notna() & (age_hours > float(cfg.stale_line_threshold_hours))

    out[cfg.market_available] = has_line & (~stale)
    out[cfg.market_available_at_utc] = fetched.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["market_status"] = np.where(~has_line, "unavailable", np.where(stale, "stale", "available"))
    out["details"] = np.select(
        [~has_line, stale, out[cfg.market_available]],
        [
            "missing_spread_total_or_ml",
            f"line_stale_gt_{cfg.stale_line_threshold_hours}h",
            "line_available",
        ],
        default="unknown",
    )
    return out[[cfg.game_id, "market_status", cfg.market_available, cfg.market_available_at_utc, "details"]]


def identify_suppression_candidates(
    availability: pd.DataFrame,
    bet_recs: pd.DataFrame,
    cfg: MarketConfig,
) -> pd.DataFrame:
    """
    Flags unavailable-market recommendations for suppression and stale/moved lines for warning.
    """
    if availability.empty:
        return pd.DataFrame()
    av = availability.copy()
    av[cfg.game_id] = av[cfg.game_id].map(canonical_id)

    recs = bet_recs.copy()
    if not recs.empty and cfg.game_id in recs.columns:
        recs[cfg.game_id] = recs[cfg.game_id].map(canonical_id)
    if cfg.final_bet_flag in recs.columns:
        recs["_final_flag"] = _bool(recs[cfg.final_bet_flag])
    else:
        recs["_final_flag"] = False

    joined = av.merge(recs[[cfg.game_id, "_final_flag"]].drop_duplicates(cfg.game_id), on=cfg.game_id, how="left")
    joined["_final_flag"] = joined["_final_flag"].fillna(False)
    suppress = joined["market_status"].eq("unavailable")
    warn = joined["market_status"].eq("stale")
    joined[cfg.market_suppressed] = suppress
    joined[cfg.suppression_reason] = np.select(
        [suppress & joined["_final_flag"], suppress, warn],
        [
            "market_unavailable_final_bet_suppressed",
            "market_unavailable_no_actionable_bet",
            "market_stale_review_line_freshness",
        ],
        default="",
    )

    line_move_path = Path("data/plumbing/line_movement.csv")
    if line_move_path.exists():
        lm = safe_read_csv(line_move_path)
        if not lm.empty and cfg.game_id in lm.columns:
            lm[cfg.game_id] = lm[cfg.game_id].map(canonical_id)
            move_cols = [c for c in [cfg.game_id, cfg.spread_movement] if c in lm.columns]
            joined = joined.merge(lm[move_cols].drop_duplicates(cfg.game_id), on=cfg.game_id, how="left", suffixes=("", "_lm"))
            moved = _num(joined.get(cfg.spread_movement)).abs().ge(2.0)
            joined[cfg.suppression_reason] = np.where(
                (~suppress) & moved,
                np.where(joined[cfg.suppression_reason].astype(str).eq(""), "line_moved_ge_2_points_review", joined[cfg.suppression_reason].astype(str) + "|line_moved_ge_2_points_review"),
                joined[cfg.suppression_reason],
            )
    return joined[[cfg.game_id, "market_status", cfg.market_available, cfg.market_available_at_utc, cfg.market_suppressed, cfg.suppression_reason]].drop_duplicates(cfg.game_id)


def apply_suppressions(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    bet_recs_path: str = "data/actionable/bet_recommendations.csv",
    candidates: pd.DataFrame | None = None,
    cfg: MarketConfig | None = None,
    dry_run: bool = False,
) -> SuppressionReport:
    """
    Applies market suppressions to predictions + bet recommendations for IDES scope only.
    """
    config = cfg or MarketConfig()
    report = SuppressionReport()
    if candidates is None or candidates.empty:
        return report

    preds_path = Path(predictions_path)
    recs_path = Path(bet_recs_path)
    preds = safe_read_csv(preds_path)
    recs = safe_read_csv(recs_path)

    if preds.empty:
        report.errors.append("missing_predictions_master")
        return report
    if recs.empty:
        report.errors.append("missing_bet_recommendations")
        return report
    if config.game_id not in preds.columns or config.game_id not in recs.columns:
        report.errors.append("missing_game_id_column")
        return report

    preds[config.game_id] = preds[config.game_id].map(canonical_id)
    recs[config.game_id] = recs[config.game_id].map(canonical_id)
    cands = candidates.copy()
    cands[config.game_id] = cands[config.game_id].map(canonical_id)

    try:
        p_mask, p_method = discover_ides_rows(preds, strict=True)
        r_mask, r_method = discover_ides_rows(recs, strict=True)
    except ValueError as exc:
        report.errors.append(f"ides_scope_not_identifiable:{exc}")
        return report

    for col in [config.market_available, config.market_available_at_utc, config.market_suppressed, config.suppression_reason, config.updated_at_utc, config.notes]:
        if col not in preds.columns:
            preds[col] = pd.Series([np.nan] * len(preds), dtype=object)
    for col in [
        config.market_suppressed,
        config.suppression_reason,
        config.updated_at_utc,
        config.bet_reason_short,
        config.pre_suppression_final_bet_flag,
        config.pre_suppression_recommended_stake_units,
        config.pre_suppression_bet_rank,
    ]:
        if col not in recs.columns:
            recs[col] = pd.Series([np.nan] * len(recs), dtype=object)

    for col in [config.market_available, config.market_suppressed]:
        if col in preds.columns:
            preds[col] = preds[col].astype(object)
    if config.market_suppressed in recs.columns:
        recs[config.market_suppressed] = recs[config.market_suppressed].astype(object)

    cand_map = cands.drop_duplicates(config.game_id).set_index(config.game_id)
    now = utc_now_iso()

    for idx, row in preds.loc[p_mask].iterrows():
        gid = row[config.game_id]
        if gid not in cand_map.index:
            continue
        crow = cand_map.loc[gid]
        preds.at[idx, config.market_available] = bool(crow.get(config.market_available, False))
        preds.at[idx, config.market_available_at_utc] = crow.get(config.market_available_at_utc)
        preds.at[idx, config.market_suppressed] = bool(crow.get(config.market_suppressed, False))
        reason = str(crow.get(config.suppression_reason, "")).strip()
        if reason:
            preds.at[idx, config.suppression_reason] = reason
            if bool(crow.get(config.market_suppressed, False)):
                preds.at[idx, config.notes] = _append_reason(preds.at[idx, config.notes], "SUPPRESSED: ", reason)
            elif "stale" in reason or "line_moved" in reason:
                preds.at[idx, config.notes] = _append_reason(preds.at[idx, config.notes], "MARKET WARN: ", reason)
        preds.at[idx, config.updated_at_utc] = now

    for idx, row in recs.loc[r_mask].iterrows():
        gid = row[config.game_id]
        if gid not in cand_map.index:
            continue
        crow = cand_map.loc[gid]
        status = str(crow.get("market_status", "")).strip()
        suppress = bool(crow.get(config.market_suppressed, False))
        reason = str(crow.get(config.suppression_reason, "")).strip()
        recs.at[idx, config.market_suppressed] = suppress
        recs.at[idx, config.suppression_reason] = reason
        recs.at[idx, config.updated_at_utc] = now

        if suppress:
            if pd.isna(recs.at[idx, config.pre_suppression_final_bet_flag]):
                recs.at[idx, config.pre_suppression_final_bet_flag] = row.get(config.final_bet_flag)
            if pd.isna(recs.at[idx, config.pre_suppression_recommended_stake_units]):
                recs.at[idx, config.pre_suppression_recommended_stake_units] = row.get(config.recommended_stake_units)
            if pd.isna(recs.at[idx, config.pre_suppression_bet_rank]):
                recs.at[idx, config.pre_suppression_bet_rank] = row.get(config.bet_rank)
            recs.at[idx, config.final_bet_flag] = False
            recs.at[idx, config.recommended_stake_units] = 0.0
            recs.at[idx, config.bet_rank] = np.nan
            recs.at[idx, config.bet_reason_short] = _append_reason(row.get(config.bet_reason_short), "SUPPRESSED: ", reason)
            report.suppressed_count += 1
            report.suppressed_game_ids.append(str(gid))
        else:
            if status == "stale":
                report.warned_count += 1
                report.warned_game_ids.append(str(gid))
            prior_suppressed = _bool(pd.Series([row.get(config.market_suppressed)])).iloc[0]
            if prior_suppressed:
                recs.at[idx, config.final_bet_flag] = row.get(config.pre_suppression_final_bet_flag, row.get(config.final_bet_flag))
                recs.at[idx, config.recommended_stake_units] = row.get(config.pre_suppression_recommended_stake_units, row.get(config.recommended_stake_units))
                recs.at[idx, config.bet_rank] = row.get(config.pre_suppression_bet_rank, row.get(config.bet_rank))
                report.unsuppressed_count += 1
                report.came_online_game_ids.append(str(gid))

    if dry_run:
        return report

    preds.to_csv(preds_path, index=False)
    recs.to_csv(recs_path, index=False)
    return report


def run_market_checker(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    bet_recs_path: str = "data/actionable/bet_recommendations.csv",
    line_movement_path: str = "data/plumbing/line_movement.csv",
    data_dir: str = "data/",
    dry_run: bool = False,
) -> SuppressionReport:
    """
    End-to-end market availability guardrail for IDES model rows.
    """
    cfg = MarketConfig()
    report = SuppressionReport()

    preds = safe_read_csv(Path(predictions_path))
    recs = safe_read_csv(Path(bet_recs_path))
    if preds.empty:
        report.errors.append("missing_predictions_master")
        return report
    if recs.empty:
        report.errors.append("missing_bet_recommendations")
        return report

    preds[cfg.game_id] = preds[cfg.game_id].map(canonical_id)
    recs[cfg.game_id] = recs[cfg.game_id].map(canonical_id)
    try:
        p_mask, p_method = discover_ides_rows(preds, strict=True)
        r_mask, r_method = discover_ides_rows(recs, strict=True)
    except ValueError as exc:
        report.errors.append(f"ides_scope_not_identifiable:{exc}")
        print(f"[FAIL] {exc}")
        return report

    today = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d")
    recs_scope = recs.loc[r_mask].copy()
    if cfg.game_date_pst in recs_scope.columns:
        recs_scope = recs_scope[recs_scope[cfg.game_date_pst].astype(str).eq(today)].copy()
    if recs_scope.empty:
        print("[INFO] No IDES recommendations in today's slate.")
        return report

    game_ids = recs_scope[cfg.game_id].dropna().astype(str).unique().tolist()
    availability = check_line_availability(game_ids=game_ids, data_dir=data_dir)
    if availability.empty:
        report.errors.append("no_availability_rows")
        return report

    report.total_games_checked = len(availability)
    report.available_count = int((availability["market_status"] == "available").sum())
    report.stale_count = int((availability["market_status"] == "stale").sum())
    report.unavailable_count = int((availability["market_status"] == "unavailable").sum())

    candidates = identify_suppression_candidates(availability=availability, bet_recs=recs_scope, cfg=cfg)
    if not Path(line_movement_path).exists():
        Path(line_movement_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[cfg.game_id]).to_csv(line_movement_path, index=False)
    outcome = apply_suppressions(
        predictions_path=predictions_path,
        bet_recs_path=bet_recs_path,
        candidates=candidates,
        cfg=cfg,
        dry_run=dry_run,
    )

    report.suppressed_count = outcome.suppressed_count
    report.warned_count = outcome.warned_count
    report.unsuppressed_count = outcome.unsuppressed_count
    report.suppressed_game_ids = sorted(set(outcome.suppressed_game_ids))
    report.warned_game_ids = sorted(set(outcome.warned_game_ids))
    report.came_online_game_ids = sorted(set(outcome.came_online_game_ids))
    report.errors.extend(outcome.errors)

    print("MARKET AVAILABILITY SUMMARY")
    print("=" * 60)
    print(f"scope_predictions={p_method} scope_bets={r_method}")
    print(f"games_checked={report.total_games_checked}")
    print(f"available={report.available_count} stale={report.stale_count} unavailable={report.unavailable_count}")
    print(f"suppressed={report.suppressed_count} warned={report.warned_count} unsuppressed={report.unsuppressed_count}")
    if report.suppressed_game_ids:
        print(f"suppressed_game_ids={report.suppressed_game_ids}")
    if report.warned_game_ids:
        print(f"warned_game_ids={report.warned_game_ids}")
    if report.came_online_game_ids:
        print(f"came_online_since_last_run={report.came_online_game_ids}")
    if report.errors:
        print(f"errors={report.errors}")
    print("=" * 60)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="IDES market availability checker")
    parser.add_argument("--predictions-path", default="data/reports/game_predictions_master.csv")
    parser.add_argument("--bet-recs-path", default="data/actionable/bet_recommendations.csv")
    parser.add_argument("--line-movement-path", default="data/plumbing/line_movement.csv")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--dry-run", default="false")
    args = parser.parse_args()

    report = run_market_checker(
        predictions_path=args.predictions_path,
        bet_recs_path=args.bet_recs_path,
        line_movement_path=args.line_movement_path,
        data_dir=args.data_dir,
        dry_run=_parse_bool(args.dry_run),
    )
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
