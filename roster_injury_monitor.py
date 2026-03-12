from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_pes_columns import discover_ides_rows
from ides_of_march.utils import canonical_id, safe_read_csv, utc_now_iso


@dataclass(frozen=True)
class InjuryConfig:
    game_id: str = "game_id"
    game_date_pst: str = "game_date_pst"
    team_a: str = "team_a"
    team_b: str = "team_b"
    model_version: str = "model_version"
    phase: str = "phase"
    is_ncaa_tournament: str = "is_ncaa_tournament"
    is_conference_tournament: str = "is_conference_tournament"
    injury_flag: str = "injury_flag"
    injury_details: str = "injury_details"
    injury_stake_override: str = "injury_stake_override"
    updated_at_utc: str = "updated_at_utc"
    significant_player_threshold: float = 0.15
    check_window_hours: int = 18

    # Existing columns touched by this checker
    run_id: str = "run_id"
    game_start_datetime_utc: str = "game_start_datetime_utc"
    prediction_winner: str = "prediction_winner"
    model_pick: str = "model_pick"
    notes: str = "notes"
    bet_reason_short: str = "bet_reason_short"
    recommended_stake_units: str = "recommended_stake_units"
    created_at_utc: str = "created_at_utc"

    # Source file columns
    src_team: str = "team"
    src_player_name: str = "player"
    src_player_id: str = "athlete_id"
    src_did_not_play: str = "did_not_play"
    src_minutes: str = "min"
    src_injury_flag: str = "injury_proxy_flag"
    src_injury_severe: str = "injury_proxy_severe"
    src_pulled_at_utc: str = "pulled_at_utc"
    src_game_datetime_utc: str = "game_datetime_utc"
    src_logs_team: str = "team"
    src_logs_player_id: str = "athlete_id"


@dataclass
class InjuryImpact:
    game_id: str
    injury_flag: bool
    injury_details: str
    injury_stake_override: float | None
    suppress_bet: bool
    max_tier: str


@dataclass
class InjuryWriteReport:
    games_checked: int = 0
    changes_detected: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    games_flagged: int = 0
    stake_overrides_applied: int = 0
    bets_suppressed: int = 0
    errors: list[str] = field(default_factory=list)


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


def _append_text(existing: Any, prefix: str, text: str) -> str:
    cur = "" if pd.isna(existing) else str(existing)
    token = f"{prefix}{text}"
    if token in cur:
        return cur
    if not cur.strip():
        return token
    return f"{token} | {cur}"


def fetch_roster_status(
    teams: list[str],
    game_date: str,
    data_dir: str = "data/",
) -> pd.DataFrame:
    """
    Fetches roster availability using the project's existing injury proxy dataset.
    """
    cfg = InjuryConfig()
    injury_path = Path(data_dir) / "player_injury_proxy.csv"
    logs_path = Path(data_dir) / "player_game_logs.csv"
    injury = safe_read_csv(injury_path)
    logs = safe_read_csv(logs_path)

    if injury.empty:
        return pd.DataFrame(
            columns=[
                cfg.src_team,
                cfg.src_player_name,
                "status",
                "minutes_share_season_avg",
                "is_significant",
                "status_changed_since",
                "source",
            ]
        )

    out = injury.copy()
    out[cfg.src_team] = out.get(cfg.src_team, "").astype(str).str.strip()
    out[cfg.src_player_id] = out.get(cfg.src_player_id, "").astype(str).str.strip()
    if teams:
        out = out[out[cfg.src_team].isin(set(map(str, teams)))].copy()
    if out.empty:
        return out

    pulled = pd.to_datetime(out.get(cfg.src_pulled_at_utc), utc=True, errors="coerce")
    game_dt = pd.to_datetime(out.get(cfg.src_game_datetime_utc), utc=True, errors="coerce")
    out["_ts"] = pulled.fillna(game_dt)
    out = out.sort_values(["_ts", cfg.src_game_datetime_utc]).drop_duplicates([cfg.src_team, cfg.src_player_id], keep="last")

    minutes_share = pd.Series(np.nan, index=out.index, dtype=float)
    if not logs.empty:
        l = logs.copy()
        l[cfg.src_logs_team] = l.get(cfg.src_logs_team, "").astype(str).str.strip()
        l[cfg.src_logs_player_id] = l.get(cfg.src_logs_player_id, "").astype(str).str.strip()
        l["min_num"] = _num(l.get("min")).fillna(0.0)
        target_year = pd.to_datetime(pd.Series([game_date]), errors="coerce").dt.year.iloc[0]
        if pd.notna(target_year):
            gdt = pd.to_datetime(l.get("game_datetime_utc"), utc=True, errors="coerce")
            l = l[gdt.dt.year.eq(int(target_year))].copy()
        agg = l.groupby([cfg.src_logs_team, cfg.src_logs_player_id], dropna=False)["min_num"].mean().rename("avg_minutes").reset_index()
        agg["minutes_share_season_avg"] = (agg["avg_minutes"] / 200.0).clip(lower=0.0, upper=1.0)
        out = out.merge(
            agg[[cfg.src_logs_team, cfg.src_logs_player_id, "minutes_share_season_avg"]],
            left_on=[cfg.src_team, cfg.src_player_id],
            right_on=[cfg.src_logs_team, cfg.src_logs_player_id],
            how="left",
        )
        minutes_share = _num(out["minutes_share_season_avg"])
    else:
        minutes_share = (_num(out.get(cfg.src_minutes)).fillna(0.0) / 200.0).clip(0.0, 1.0)
        out["minutes_share_season_avg"] = minutes_share

    dnp = _bool(out.get(cfg.src_did_not_play))
    severe = _bool(out.get(cfg.src_injury_severe))
    flagged = _bool(out.get(cfg.src_injury_flag))
    out["status"] = np.select(
        [dnp | severe, flagged],
        ["out", "questionable"],
        default="available",
    )
    out["minutes_share_season_avg"] = minutes_share.fillna(0.0)
    out["is_significant"] = out["minutes_share_season_avg"] > float(cfg.significant_player_threshold)
    out["status_changed_since"] = out["_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["source"] = "data/player_injury_proxy.csv"

    return out[
        [
            cfg.src_team,
            cfg.src_player_name,
            cfg.src_player_id,
            "status",
            "minutes_share_season_avg",
            "is_significant",
            "status_changed_since",
            "source",
        ]
    ]


def load_model_run_timestamp(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    cfg: InjuryConfig | None = None,
) -> str:
    """
    Gets latest model output timestamp for today's IDES games.
    """
    config = cfg or InjuryConfig()
    preds = safe_read_csv(Path(predictions_path))
    if preds.empty:
        return utc_now_iso()
    try:
        mask, _ = discover_ides_rows(preds, strict=True)
    except ValueError:
        return utc_now_iso()
    scoped = preds.loc[mask].copy()
    if scoped.empty:
        return utc_now_iso()
    today = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d")
    if config.game_date_pst in scoped.columns:
        scoped = scoped[scoped[config.game_date_pst].astype(str).eq(today)].copy()
    if scoped.empty:
        return utc_now_iso()
    timestamps: list[pd.Series] = []
    if config.updated_at_utc in scoped.columns:
        timestamps.append(pd.to_datetime(scoped[config.updated_at_utc], utc=True, errors="coerce"))
    if config.created_at_utc in scoped.columns:
        timestamps.append(pd.to_datetime(scoped[config.created_at_utc], utc=True, errors="coerce"))
    if not timestamps:
        return utc_now_iso()
    merged = pd.concat(timestamps, axis=0).dropna()
    if merged.empty:
        return utc_now_iso()
    return merged.max().strftime("%Y-%m-%dT%H:%M:%SZ")


def identify_significant_changes(
    roster_status: pd.DataFrame,
    model_run_timestamp: str,
    cfg: InjuryConfig,
) -> pd.DataFrame:
    """
    Identifies post-model roster updates for significant players.
    """
    if roster_status.empty:
        return roster_status.copy()
    out = roster_status.copy()
    model_ts = pd.to_datetime(model_run_timestamp, utc=True, errors="coerce")
    change_ts = pd.to_datetime(out["status_changed_since"], utc=True, errors="coerce")
    changed_after_model = change_ts > model_ts
    significant = _bool(out["is_significant"])

    out["tier"] = np.select(
        [
            changed_after_model & out["status"].astype(str).eq("out") & (_num(out["minutes_share_season_avg"]) >= 0.18),
            changed_after_model & out["status"].astype(str).eq("out") & significant,
            changed_after_model & out["status"].astype(str).eq("questionable") & significant,
            changed_after_model,
        ],
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default="NONE",
    )
    return out[out["tier"].ne("NONE")].copy()


def compute_injury_impact(
    changes: pd.DataFrame,
    game_row: pd.Series,
    cfg: InjuryConfig,
) -> InjuryImpact:
    """
    Computes game-level injury impact and optional stake override recommendation.
    """
    gid = str(game_row.get(cfg.game_id, ""))
    ta = str(game_row.get(cfg.team_a, "")).strip()
    tb = str(game_row.get(cfg.team_b, "")).strip()
    pred_winner = str(game_row.get(cfg.prediction_winner, game_row.get(cfg.model_pick, ""))).strip()
    if changes.empty:
        return InjuryImpact(gid, False, "", None, False, "NONE")

    local = changes[changes[cfg.src_team].astype(str).isin({ta, tb})].copy()
    if local.empty:
        return InjuryImpact(gid, False, "", None, False, "NONE")

    tier_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    local["_tier_rank"] = local["tier"].map(tier_order).fillna(0)
    top_rank = int(local["_tier_rank"].max())
    inv_tier = {v: k for k, v in tier_order.items()}
    max_tier = inv_tier.get(top_rank, "NONE")

    injury_flag = max_tier in {"CRITICAL", "HIGH"}
    suppress_bet = False
    if max_tier == "CRITICAL":
        critical_teams = set(local[local["tier"].eq("CRITICAL")][cfg.src_team].astype(str))
        suppress_bet = pred_winner in critical_teams if pred_winner else False

    details_parts: list[str] = []
    for _, r in local.sort_values(["_tier_rank", "minutes_share_season_avg"], ascending=[False, False]).head(4).iterrows():
        details_parts.append(
            f"{r[cfg.src_team]}:{r.get(cfg.src_player_name,'player')}={r['status']}({r['tier']},share={float(r['minutes_share_season_avg']):.2f})"
        )
    details = "; ".join(details_parts)

    override = None
    current_stake = pd.to_numeric(pd.Series([game_row.get(cfg.recommended_stake_units)]), errors="coerce").iloc[0]
    if pd.notna(current_stake):
        if max_tier == "CRITICAL":
            override = float(current_stake) * 0.5
        elif max_tier == "HIGH":
            override = float(current_stake) * 0.75

    return InjuryImpact(
        game_id=gid,
        injury_flag=injury_flag,
        injury_details=details,
        injury_stake_override=override,
        suppress_bet=suppress_bet,
        max_tier=max_tier,
    )


def apply_injury_flags(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    bet_recs_path: str = "data/actionable/bet_recommendations.csv",
    impacts: pd.DataFrame | None = None,
    cfg: InjuryConfig | None = None,
    dry_run: bool = False,
) -> InjuryWriteReport:
    """
    Writes injury flags to predictions + bet recommendations (IDES rows only).
    """
    config = cfg or InjuryConfig()
    report = InjuryWriteReport()
    if impacts is None or impacts.empty:
        return report

    preds_path = Path(predictions_path)
    recs_path = Path(bet_recs_path)
    preds = safe_read_csv(preds_path)
    recs = safe_read_csv(recs_path)
    if preds.empty or recs.empty:
        report.errors.append("missing_predictions_or_bets")
        return report

    preds[config.game_id] = preds[config.game_id].map(canonical_id)
    recs[config.game_id] = recs[config.game_id].map(canonical_id)
    impacts = impacts.copy()
    impacts[config.game_id] = impacts[config.game_id].map(canonical_id)
    impact_map = impacts.drop_duplicates(config.game_id).set_index(config.game_id)

    try:
        p_mask, _ = discover_ides_rows(preds, strict=True)
        r_mask, _ = discover_ides_rows(recs, strict=True)
    except ValueError as exc:
        report.errors.append(f"ides_scope_not_identifiable:{exc}")
        return report

    for col in [config.injury_flag, config.injury_details, config.injury_stake_override, config.updated_at_utc, config.notes]:
        if col not in preds.columns:
            preds[col] = pd.Series([np.nan] * len(preds), dtype=object)
    for col in [config.injury_flag, config.injury_details, config.injury_stake_override, config.updated_at_utc, config.bet_reason_short, config.recommended_stake_units]:
        if col not in recs.columns:
            recs[col] = pd.Series([np.nan] * len(recs), dtype=object)

    if config.injury_flag in preds.columns:
        preds[config.injury_flag] = preds[config.injury_flag].astype(object)
    if config.injury_flag in recs.columns:
        recs[config.injury_flag] = recs[config.injury_flag].astype(object)

    now = utc_now_iso()
    for idx, row in preds.loc[p_mask].iterrows():
        gid = row[config.game_id]
        if gid not in impact_map.index:
            continue
        imp = impact_map.loc[gid]
        preds.at[idx, config.injury_flag] = bool(imp.get(config.injury_flag, False))
        preds.at[idx, config.injury_details] = imp.get(config.injury_details, "")
        preds.at[idx, config.injury_stake_override] = imp.get(config.injury_stake_override, np.nan)
        if bool(imp.get(config.injury_flag, False)):
            preds.at[idx, config.notes] = _append_text(preds.at[idx, config.notes], "INJURY FLAG: ", str(imp.get(config.injury_details, "")))
            report.games_flagged += 1
        preds.at[idx, config.updated_at_utc] = now

    for idx, row in recs.loc[r_mask].iterrows():
        gid = row[config.game_id]
        if gid not in impact_map.index:
            continue
        imp = impact_map.loc[gid]
        flag = bool(imp.get(config.injury_flag, False))
        detail = str(imp.get(config.injury_details, ""))
        override = pd.to_numeric(pd.Series([imp.get(config.injury_stake_override)]), errors="coerce").iloc[0]
        suppress = bool(imp.get("suppress_bet", False))

        recs.at[idx, config.injury_flag] = flag
        recs.at[idx, config.injury_details] = detail
        recs.at[idx, config.updated_at_utc] = now
        if pd.notna(override):
            recs.at[idx, config.injury_stake_override] = float(override)
            recs.at[idx, config.recommended_stake_units] = float(max(0.0, override))
            report.stake_overrides_applied += 1
        if suppress:
            recs.at[idx, config.bet_reason_short] = _append_text(recs.at[idx, config.bet_reason_short], "INJURY FLAG: ", detail)
            report.bets_suppressed += 1

    if dry_run:
        return report

    preds.to_csv(preds_path, index=False)
    recs.to_csv(recs_path, index=False)
    return report


def run_injury_monitor(
    predictions_path: str = "data/reports/game_predictions_master.csv",
    bet_recs_path: str = "data/actionable/bet_recommendations.csv",
    data_dir: str = "data/",
    dry_run: bool = False,
) -> InjuryWriteReport:
    """
    End-to-end roster/injury monitor.
    """
    cfg = InjuryConfig()
    report = InjuryWriteReport()

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

    now_utc = pd.Timestamp.now(tz="UTC")
    scoped = preds.loc[p_mask].copy()
    if cfg.game_start_datetime_utc in scoped.columns:
        gstart = pd.to_datetime(scoped[cfg.game_start_datetime_utc], utc=True, errors="coerce")
        window_end = now_utc + pd.Timedelta(hours=int(cfg.check_window_hours))
        scoped = scoped[gstart.between(now_utc - pd.Timedelta(hours=6), window_end, inclusive="both")].copy()
    if scoped.empty:
        print("[INFO] No IDES games in injury monitor window.")
        return report

    report.games_checked = int(scoped[cfg.game_id].nunique())
    teams = sorted(set(scoped[cfg.team_a].astype(str)).union(set(scoped[cfg.team_b].astype(str))))
    game_date = str(scoped[cfg.game_date_pst].astype(str).min()) if cfg.game_date_pst in scoped.columns else pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d")
    model_ts = load_model_run_timestamp(predictions_path=predictions_path, cfg=cfg)
    roster_status = fetch_roster_status(teams=teams, game_date=game_date, data_dir=data_dir)
    changes = identify_significant_changes(roster_status=roster_status, model_run_timestamp=model_ts, cfg=cfg)

    report.changes_detected = len(changes)
    report.critical_count = int((changes.get("tier", pd.Series(dtype=str)) == "CRITICAL").sum())
    report.high_count = int((changes.get("tier", pd.Series(dtype=str)) == "HIGH").sum())
    report.medium_count = int((changes.get("tier", pd.Series(dtype=str)) == "MEDIUM").sum())
    report.low_count = int((changes.get("tier", pd.Series(dtype=str)) == "LOW").sum())

    recs_scope = recs.loc[r_mask].copy()
    impacts: list[dict[str, Any]] = []
    for _, g in scoped.iterrows():
        gid = g[cfg.game_id]
        rec_row = recs_scope[recs_scope[cfg.game_id].astype(str).eq(str(gid))]
        base = g.copy()
        if not rec_row.empty:
            for col in [cfg.recommended_stake_units]:
                if col in rec_row.columns:
                    base[col] = rec_row.iloc[0][col]
        imp = compute_injury_impact(changes=changes, game_row=base, cfg=cfg)
        impacts.append(
            {
                cfg.game_id: imp.game_id,
                cfg.injury_flag: imp.injury_flag,
                cfg.injury_details: imp.injury_details,
                cfg.injury_stake_override: imp.injury_stake_override,
                "suppress_bet": imp.suppress_bet,
                "max_tier": imp.max_tier,
            }
        )

    impacts_df = pd.DataFrame(impacts)
    out = apply_injury_flags(
        predictions_path=predictions_path,
        bet_recs_path=bet_recs_path,
        impacts=impacts_df,
        cfg=cfg,
        dry_run=dry_run,
    )

    report.games_flagged = out.games_flagged
    report.stake_overrides_applied = out.stake_overrides_applied
    report.bets_suppressed = out.bets_suppressed
    report.errors.extend(out.errors)

    print("ROSTER / INJURY MONITOR SUMMARY")
    print("=" * 60)
    print(f"scope_predictions={p_method} scope_bets={r_method}")
    print(f"games_checked={report.games_checked}")
    print(f"changes_detected={report.changes_detected} (critical={report.critical_count}, high={report.high_count}, medium={report.medium_count}, low={report.low_count})")
    print(f"games_flagged={report.games_flagged}")
    print(f"stake_overrides_applied={report.stake_overrides_applied}")
    print(f"bets_suppressed={report.bets_suppressed}")
    if report.errors:
        print(f"errors={report.errors}")
    print("=" * 60)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="IDES roster/injury monitor")
    parser.add_argument("--predictions-path", default="data/reports/game_predictions_master.csv")
    parser.add_argument("--bet-recs-path", default="data/actionable/bet_recommendations.csv")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--dry-run", default="false")
    args = parser.parse_args()

    report = run_injury_monitor(
        predictions_path=args.predictions_path,
        bet_recs_path=args.bet_recs_path,
        data_dir=args.data_dir,
        dry_run=_parse_bool(args.dry_run),
    )
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
