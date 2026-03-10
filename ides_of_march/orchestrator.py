from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import DATA_DIR, output_paths
from .csv_contracts import CSV_HEADERS, conform_to_contract, write_template
from .data_steward import build_data_steward_frame
from .evaluation import run_variant_backtest
from .layer1_base_strength import apply_base_strength
from .layer2_context import apply_context_adjustments
from .layer3_situational import apply_situational_layer, discover_situational_rules
from .layer4_monte_carlo import apply_monte_carlo_layer
from .layer5_agreement import apply_agreement_layer, summarize_agreement_buckets
from .layer6_decision import apply_decision_layer, fit_direct_win_model
from .safety import run_data_integrity_audit, run_model_safety_audit
from .schemas import validate_agreement, validate_backtest_summary, validate_bet_recs, validate_predictions
from .utils import ensure_dir, pretty_exception, utc_now_iso, write_json


MODEL_VERSION = "ides_of_march_v1"


@dataclass
class StageRecord:
    agent: str
    status: str
    notes: list[str]
    metrics: dict[str, Any]


@dataclass
class RunResult:
    ok: bool
    status: str
    stages: list[StageRecord]
    outputs: dict[str, str]
    error: str | None


def _num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bool(series: Any) -> pd.Series:
    return pd.Series(series).fillna(False).astype(bool)


def _time_fields(frame: pd.DataFrame, source_col: str = "game_datetime_utc") -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    utc = pd.to_datetime(frame.get(source_col), utc=True, errors="coerce")
    pst = utc.dt.tz_convert("America/Los_Angeles")
    out["game_start_datetime_utc"] = np.where(utc.notna(), utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ"), np.nan)
    out["game_start_datetime_pst"] = np.where(pst.notna(), pst.dt.strftime("%Y-%m-%d %H:%M:%S%z"), np.nan)
    out["game_start_time_pst"] = np.where(pst.notna(), pst.dt.strftime("%H:%M"), np.nan)
    out["game_date_pst"] = np.where(pst.notna(), pst.dt.strftime("%Y-%m-%d"), np.nan)
    out["season"] = np.where(pst.notna(), np.where(pst.dt.month >= 7, pst.dt.year + 1, pst.dt.year), np.nan)
    out["day_of_week_pst"] = np.where(pst.notna(), pst.dt.day_name(), np.nan)
    return out


class IDESOrchestrator:
    def __init__(self, *, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.paths = output_paths()
        self._ensure_templates()

    def _ensure_templates(self) -> None:
        file_to_path = {
            "games_schedule_master.csv": self.paths.games_schedule_master,
            "team_game_boxscores.csv": self.paths.team_game_boxscores,
            "player_game_boxscores.csv": self.paths.player_game_boxscores,
            "team_rolling_features.csv": self.paths.team_rolling_features,
            "game_matchup_features.csv": self.paths.game_matchup_features,
            "game_totals_features.csv": self.paths.game_totals_features,
            "game_context_adjustments.csv": self.paths.game_context_adjustments,
            "situational_signals_game_level.csv": self.paths.situational_signals_game_level,
            "game_monte_carlo_outputs.csv": self.paths.game_monte_carlo_outputs,
            "game_predictions_master.csv": self.paths.game_predictions_master,
            "bet_recommendations.csv": self.paths.bet_recommendations,
            "watchlist_games.csv": self.paths.watchlist_games,
            "no_bet_explanations.csv": self.paths.no_bet_explanations,
            "daily_card_summary.csv": self.paths.daily_card_summary,
            "agreement_analysis_results.csv": self.paths.agreement_analysis_results,
            "backtest_model_summary.csv": self.paths.backtest_model_summary,
            "pipeline_run_log.csv": self.paths.pipeline_run_log,
        }
        for file_name, path in file_to_path.items():
            write_template(path, file_name)

    def _write_contract_csv(self, file_name: str, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        out = conform_to_contract(frame, file_name)
        ensure_dir(path)
        out.to_csv(path, index=False)
        return out

    def _build_frame(
        self,
        *,
        base: pd.DataFrame,
        file_name: str,
        mapping: dict[str, str | float | int | Callable[[pd.DataFrame], Any]],
        source_col: str = "game_datetime_utc",
        now_utc: str | None = None,
    ) -> pd.DataFrame:
        headers = CSV_HEADERS[file_name]
        out = pd.DataFrame(index=base.index)
        tf = _time_fields(base, source_col=source_col)
        for col in ("game_date_pst", "game_start_time_pst", "game_start_datetime_pst", "game_start_datetime_utc", "season", "day_of_week_pst"):
            if col in headers:
                out[col] = tf.get(col)
        for col, spec in mapping.items():
            if callable(spec):
                out[col] = spec(base)
            elif isinstance(spec, str):
                value = base.get(spec)
                if isinstance(value, pd.DataFrame):
                    value = value.iloc[:, 0]
                out[col] = value
            else:
                out[col] = spec
        if "created_at_utc" in headers and "created_at_utc" not in out.columns:
            out["created_at_utc"] = now_utc or utc_now_iso()
        if "updated_at_utc" in headers and "updated_at_utc" not in out.columns:
            out["updated_at_utc"] = now_utc or utc_now_iso()
        return conform_to_contract(out, file_name)

    def _append_pipeline_log(self, run_id: str, stages: list[StageRecord], error: str | None = None) -> None:
        rows: list[dict[str, Any]] = []
        now = utc_now_iso()
        for idx, stage in enumerate(stages, start=1):
            rows.append(
                {
                    "run_id": run_id,
                    "model_version": MODEL_VERSION,
                    "stage_name": stage.agent,
                    "stage_order": idx,
                    "status": stage.status,
                    "severity": stage.status,
                    "attempt": 1,
                    "started_at_utc": now,
                    "ended_at_utc": now,
                    "duration_seconds": np.nan,
                    "rows_written": stage.metrics.get("rows", np.nan),
                    "error_code": "runtime_exception" if error else np.nan,
                    "error_message": error or (" | ".join(stage.notes) if stage.notes else np.nan),
                    "created_at_utc": now,
                }
            )
        update = conform_to_contract(pd.DataFrame(rows), "pipeline_run_log.csv")
        if self.paths.pipeline_run_log.exists() and self.paths.pipeline_run_log.stat().st_size > 0:
            prior = pd.read_csv(self.paths.pipeline_run_log, low_memory=False)
            out = pd.concat([prior, update], ignore_index=True)
        else:
            out = update
        ensure_dir(self.paths.pipeline_run_log)
        out.to_csv(self.paths.pipeline_run_log, index=False)

    def audit(self, *, as_of: pd.Timestamp, hours_ahead: int = 48) -> RunResult:
        stages: list[StageRecord] = []
        run_id = f"ides_audit_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}"
        try:
            ds = build_data_steward_frame(data_dir=self.data_dir, as_of=as_of, hours_ahead=hours_ahead)
            integrity = run_data_integrity_audit(ds.upcoming_games, ds.historical_games)
            stages.append(StageRecord("Data Integrity Auditor", integrity.status, integrity.issues, integrity.metrics))
            self._append_pipeline_log(run_id, stages)
            return RunResult(ok=integrity.status in {"PASS", "WARN"}, status=integrity.status, stages=stages, outputs={}, error=None)
        except Exception as exc:
            self._append_pipeline_log(run_id, stages, pretty_exception(exc))
            return RunResult(ok=False, status="FAIL", stages=stages, outputs={}, error=pretty_exception(exc))

    def _score_history_for_reports(self, historical: pd.DataFrame, *, mc_mode: str) -> pd.DataFrame:
        if historical.empty:
            return historical
        hist = apply_base_strength(historical)
        hist = apply_context_adjustments(hist)
        rulebook = discover_situational_rules(hist)
        hist = apply_situational_layer(hist, rulebook)
        hist = apply_monte_carlo_layer(hist, mode=mc_mode, n_sims=250)
        direct = fit_direct_win_model(hist)
        hist = apply_decision_layer(hist, direct_win_model=direct, mc_mode=mc_mode)
        return apply_agreement_layer(hist)

    def predict(self, *, as_of: pd.Timestamp, mc_mode: str = "confidence_only", hours_ahead: int = 48) -> RunResult:
        stages: list[StageRecord] = []
        outputs: dict[str, str] = {}
        run_id = f"ides_predict_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}"
        now_utc = utc_now_iso()
        try:
            ds = build_data_steward_frame(data_dir=self.data_dir, as_of=as_of, hours_ahead=hours_ahead)
            integrity = run_data_integrity_audit(ds.upcoming_games, ds.historical_games)
            stages.append(StageRecord("Data Integrity Auditor", integrity.status, integrity.issues, integrity.metrics))
            if integrity.status in {"FAIL", "BLOCKED"}:
                self._append_pipeline_log(run_id, stages, "integrity_gate_failed")
                return RunResult(ok=False, status=integrity.status, stages=stages, outputs=outputs, error="integrity_gate_failed")

            upcoming = apply_base_strength(ds.upcoming_games)
            upcoming = apply_context_adjustments(upcoming)
            hist_scored = self._score_history_for_reports(ds.historical_games, mc_mode=mc_mode)
            rulebook = discover_situational_rules(hist_scored)
            upcoming = apply_situational_layer(upcoming, rulebook)
            upcoming = apply_monte_carlo_layer(upcoming, mode=mc_mode, n_sims=500)
            direct_model = fit_direct_win_model(hist_scored)
            upcoming = apply_decision_layer(upcoming, direct_win_model=direct_model, mc_mode=mc_mode)
            upcoming = apply_agreement_layer(upcoming)
            stages.append(StageRecord("Layer Stack", "PASS", [], {"rows": int(len(upcoming))}))

            safety = run_model_safety_audit(upcoming)
            stages.append(StageRecord("Model Safety Auditor", safety.status, safety.issues, safety.metrics))

            schedule = self._build_frame(
                base=ds.upcoming_games,
                file_name="games_schedule_master.csv",
                now_utc=now_utc,
                mapping={
                    "game_id": "game_id",
                    "phase": lambda d: d.get("phase", pd.Series("regular_season", index=d.index)),
                    "round_name": lambda d: d.get("round_name", np.nan),
                    "is_postseason": lambda d: _bool(d.get("is_postseason")),
                    "is_conference_tournament": lambda d: _bool(d.get("is_conference_tournament")),
                    "is_ncaa_tournament": lambda d: _bool(d.get("is_ncaa_tournament")),
                    "conference_tournament_name": np.nan,
                    "site_type": lambda d: np.where(_bool(d.get("is_neutral")), "neutral", "home"),
                    "venue": np.nan,
                    "city": np.nan,
                    "state": np.nan,
                    "team_a": "home_team",
                    "team_b": "away_team",
                    "team_a_conference": np.nan,
                    "team_b_conference": np.nan,
                    "team_a_seed": np.nan,
                    "team_b_seed": np.nan,
                    "team_a_is_home": lambda d: _bool(d.get("home_bonus_eligible")).astype(int),
                    "team_b_is_home": 0,
                    "team_a_is_neutral": lambda d: _bool(d.get("is_neutral")).astype(int),
                    "team_b_is_neutral": lambda d: _bool(d.get("is_neutral")).astype(int),
                    "market_spread_team_a_open": lambda d: _num(d.get("market_spread")),
                    "market_spread_team_a_close": lambda d: _num(d.get("market_spread")),
                    "market_spread_team_b_open": lambda d: -_num(d.get("market_spread")),
                    "market_spread_team_b_close": lambda d: -_num(d.get("market_spread")),
                    "market_total_open": lambda d: _num(d.get("market_total")),
                    "market_total_close": lambda d: _num(d.get("market_total")),
                    "market_moneyline_team_a": lambda d: _num(d.get("moneyline_home")),
                    "market_moneyline_team_b": lambda d: _num(d.get("moneyline_away")),
                    "game_status": "scheduled",
                },
            )
            self._write_contract_csv("games_schedule_master.csv", schedule, self.paths.games_schedule_master)

            team_hist = ds.team_history.rename(columns={"event_id": "game_id", "team_id": "team", "opponent_id": "opponent"})
            team_box = self._build_frame(
                base=team_hist,
                file_name="team_game_boxscores.csv",
                source_col="game_datetime_utc",
                now_utc=now_utc,
                mapping={
                    "game_id": "game_id",
                    "team": "team",
                    "opponent": "opponent",
                    "conference": np.nan,
                    "phase": "historical",
                    "round_name": np.nan,
                    "site_type": np.nan,
                    "is_home": np.nan,
                    "is_away": np.nan,
                    "is_neutral": np.nan,
                    "pace": lambda d: _num(d.get("pace")),
                    "off_eff": lambda d: 110.0 + (_num(d.get("adj_em")) / 2.0),
                    "def_eff": lambda d: 110.0 - (_num(d.get("adj_em")) / 2.0),
                    "efg_pct": lambda d: _num(d.get("efg_pct")),
                    "ft_pct": lambda d: _num(d.get("ft_pct")),
                    "ftr": lambda d: _num(d.get("ftr")),
                    "to_pct": lambda d: _num(d.get("tov_pct")),
                    "oreb_pct": lambda d: _num(d.get("orb_pct")),
                    "dreb_pct": lambda d: _num(d.get("drb_pct")),
                    "previous_game_ot_flag": 0,
                },
            )
            self._write_contract_csv("team_game_boxscores.csv", team_box, self.paths.team_game_boxscores)
            self._write_contract_csv("player_game_boxscores.csv", pd.DataFrame(), self.paths.player_game_boxscores)
            rolling = self._build_frame(
                base=team_hist,
                file_name="team_rolling_features.csv",
                source_col="game_datetime_utc",
                now_utc=now_utc,
                mapping={
                    "game_id": "game_id",
                    "team": "team",
                    "opponent": "opponent",
                    "conference": np.nan,
                    "phase": "historical",
                    "round_name": np.nan,
                    "site_type": np.nan,
                    "games_played_last_5": 5,
                    "games_played_last_12": 12,
                    "last5_adj_em": lambda d: _num(d.get("adj_em_l5")),
                    "last12_adj_em": lambda d: _num(d.get("adj_em_l12")),
                    "form_delta_adj_em": lambda d: _num(d.get("Form_Delta")),
                    "last5_efg_pct": lambda d: _num(d.get("efg_pct_l5")),
                    "last12_efg_pct": lambda d: _num(d.get("efg_pct_l12")),
                    "last5_to_pct": lambda d: _num(d.get("tov_pct_l5")),
                    "last12_to_pct": lambda d: _num(d.get("tov_pct_l12")),
                    "last5_oreb_pct": lambda d: _num(d.get("orb_pct_l5")),
                    "last12_oreb_pct": lambda d: _num(d.get("orb_pct_l12")),
                    "last5_ftr": lambda d: _num(d.get("ftr_l5")),
                    "last12_ftr": lambda d: _num(d.get("ftr_l12")),
                    "last5_ft_pct": lambda d: _num(d.get("ft_pct_l5")),
                    "last12_ft_pct": lambda d: _num(d.get("ft_pct_l12")),
                    "last5_ft_scoring_pressure": lambda d: _num(d.get("ft_scoring_pressure_l5")),
                    "last12_ft_scoring_pressure": lambda d: _num(d.get("ft_scoring_pressure_l12")),
                    "last5_pace": lambda d: _num(d.get("pace_l5")),
                    "last12_pace": lambda d: _num(d.get("pace_l12")),
                    "days_since_last_game": lambda d: _num(d.get("days_rest")),
                    "sos_last12": lambda d: _num(d.get("sos_pre")),
                    "previous_game_ot_flag": 0,
                    "top5_minutes_share_previous_game": np.nan,
                    "lineup_continuity_score": np.nan,
                    "adj_em_nonhome": np.nan,
                    "adj_em_away": np.nan,
                    "adj_em_neutral": np.nan,
                    "travelability_delta": np.nan,
                },
            )
            rolling["form_delta_efg_pct"] = _num(rolling["last5_efg_pct"]) - _num(rolling["last12_efg_pct"])
            rolling["form_delta_to_pct"] = _num(rolling["last5_to_pct"]) - _num(rolling["last12_to_pct"])
            rolling["form_delta_oreb_pct"] = _num(rolling["last5_oreb_pct"]) - _num(rolling["last12_oreb_pct"])
            rolling["form_delta_ftr"] = _num(rolling["last5_ftr"]) - _num(rolling["last12_ftr"])
            rolling["form_delta_ft_pct"] = _num(rolling["last5_ft_pct"]) - _num(rolling["last12_ft_pct"])
            rolling["form_delta_ft_scoring_pressure"] = _num(rolling["last5_ft_scoring_pressure"]) - _num(rolling["last12_ft_scoring_pressure"])
            rolling["pace_form_delta"] = _num(rolling["last5_pace"]) - _num(rolling["last12_pace"])
            self._write_contract_csv("team_rolling_features.csv", rolling, self.paths.team_rolling_features)

            matchup = self._build_frame(
                base=upcoming,
                file_name="game_matchup_features.csv",
                now_utc=now_utc,
                mapping={
                    "game_id": "game_id",
                    "phase": lambda d: d.get("phase", pd.Series("regular_season", index=d.index)),
                    "round_name": lambda d: d.get("round_name", np.nan),
                    "site_type": lambda d: np.where(_bool(d.get("is_neutral")), "neutral", "home"),
                    "team_a": "home_team",
                    "team_b": "away_team",
                    "team_a_base_power": lambda d: _num(d.get("model_b_margin")),
                    "team_b_base_power": lambda d: -_num(d.get("model_b_margin")),
                    "base_rating_diff": lambda d: _num(d.get("base_margin_blend")),
                    "team_a_adj_em": lambda d: _num(d.get("home_adj_em_l12")),
                    "team_b_adj_em": lambda d: _num(d.get("away_adj_em_l12")),
                    "adj_em_diff": lambda d: _num(d.get("adj_em_margin_l12")),
                    "efg_margin_diff": lambda d: _num(d.get("efg_margin_l5")),
                    "to_margin_diff": lambda d: _num(d.get("to_margin_l5")),
                    "oreb_margin_diff": lambda d: _num(d.get("oreb_margin_l5")),
                    "ft_scoring_pressure_diff": lambda d: _num(d.get("ft_scoring_pressure_margin_l5")),
                    "form_diff": lambda d: _num(d.get("form_delta_diff")),
                    "sos_diff": lambda d: _num(d.get("sos_diff")),
                    "rest_diff": lambda d: _num(d.get("rest_diff")),
                    "home_edge_team_a": lambda d: _bool(d.get("home_bonus_eligible")).astype(float),
                    "neutral_flag": lambda d: _bool(d.get("is_neutral")).astype(int),
                },
            )
            self._write_contract_csv("game_matchup_features.csv", matchup, self.paths.game_matchup_features)

            totals = self._build_frame(
                base=upcoming,
                file_name="game_totals_features.csv",
                now_utc=now_utc,
                mapping={
                    "game_id": "game_id",
                    "phase": lambda d: d.get("phase", pd.Series("regular_season", index=d.index)),
                    "round_name": lambda d: d.get("round_name", np.nan),
                    "site_type": lambda d: np.where(_bool(d.get("is_neutral")), "neutral", "home"),
                    "team_a": "home_team",
                    "team_b": "away_team",
                    "projected_possessions": lambda d: _num(d.get("expected_pace")),
                    "projected_pace_team_a": lambda d: _num(d.get("home_pace_l5")),
                    "projected_pace_team_b": lambda d: _num(d.get("away_pace_l5")),
                    "pace_control_score": lambda d: _num(d.get("home_pace_l5")) - _num(d.get("away_pace_l5")),
                    "pace_variance": lambda d: _num(d.get("mc_volatility")),
                    "projected_ppp_team_a": lambda d: (110.0 + _num(d.get("home_adj_em_l12")) / 2.0) / 100.0,
                    "projected_ppp_team_b": lambda d: (110.0 + _num(d.get("away_adj_em_l12")) / 2.0) / 100.0,
                    "base_total_raw": lambda d: _num(d.get("projected_total_ctx")),
                    "projected_to_rate_team_a": lambda d: _num(d.get("home_tov_pct_l5")),
                    "projected_to_rate_team_b": lambda d: _num(d.get("away_tov_pct_l5")),
                    "projected_oreb_rate_team_a": lambda d: _num(d.get("home_orb_pct_l5")),
                    "projected_oreb_rate_team_b": lambda d: _num(d.get("away_orb_pct_l5")),
                    "projected_ft_scoring_pressure_team_a": lambda d: _num(d.get("home_ft_scoring_pressure_l5")),
                    "projected_ft_scoring_pressure_team_b": lambda d: _num(d.get("away_ft_scoring_pressure_l5")),
                    "projected_three_pt_pressure_team_a": lambda d: _num(d.get("home_three_par_l5")),
                    "projected_three_pt_pressure_team_b": lambda d: _num(d.get("away_three_par_l5")),
                    "empty_possession_pressure": lambda d: (_num(d.get("home_tov_pct_l5")) + _num(d.get("away_tov_pct_l5")) - _num(d.get("home_orb_pct_l5")) - _num(d.get("away_orb_pct_l5"))),
                },
            )
            self._write_contract_csv("game_totals_features.csv", totals, self.paths.game_totals_features)
            context = self._build_frame(
                base=upcoming,
                file_name="game_context_adjustments.csv",
                now_utc=now_utc,
                mapping={
                    "game_id": "game_id",
                    "phase": lambda d: d.get("phase", pd.Series("regular_season", index=d.index)),
                    "round_name": lambda d: d.get("round_name", np.nan),
                    "team_a": "home_team",
                    "team_b": "away_team",
                    "home_court_adjustment": lambda d: _num(d.get("context_hca")),
                    "neutral_court_adjustment": lambda d: np.where(_bool(d.get("is_neutral")), -_num(d.get("context_hca")), 0.0),
                    "sos_diff": lambda d: _num(d.get("sos_diff")),
                    "sos_adjustment": lambda d: _num(d.get("context_sos")),
                    "form_diff": lambda d: _num(d.get("form_delta_diff")),
                    "form_adjustment": lambda d: _num(d.get("context_form")),
                    "rest_diff": lambda d: _num(d.get("rest_diff")),
                    "rest_adjustment": lambda d: _num(d.get("context_rest")),
                    "context_spread_adjustment_total": lambda d: _num(d.get("context_adjustment")),
                    "spread_after_context": lambda d: -_num(d.get("margin_ctx_blend")),
                    "total_after_context": lambda d: _num(d.get("projected_total_ctx")),
                },
            )
            self._write_contract_csv("game_context_adjustments.csv", context, self.paths.game_context_adjustments)

            active = upcoming.get("situational_active_rules", pd.Series("", index=upcoming.index)).astype(str)
            situ = self._build_frame(
                base=upcoming,
                file_name="situational_signals_game_level.csv",
                now_utc=now_utc,
                mapping={
                    "game_id": "game_id",
                    "phase": lambda d: d.get("phase", pd.Series("regular_season", index=d.index)),
                    "round_name": lambda d: d.get("round_name", np.nan),
                    "team_a": "home_team",
                    "team_b": "away_team",
                    "home_rest_form_signal_team_a": active.str.contains("home_rest_form_edge", na=False).astype(int),
                    "home_rest_form_signal_team_b": 0,
                    "slow_dog_to_edge_signal_team_a": active.str.contains("road_favorite_short_rest", na=False).astype(int),
                    "slow_dog_to_edge_signal_team_b": 0,
                    "oreb_mismatch_signal_team_a": active.str.contains("home_oreb_vs_weak_dreb", na=False).astype(int),
                    "oreb_mismatch_signal_team_b": 0,
                    "situational_score_team_a": lambda d: _num(d.get("situational_score")),
                    "situational_score_team_b": lambda d: -_num(d.get("situational_score")),
                    "situational_lean": lambda d: np.where(_num(d.get("situational_score")) >= 0, "team_a", "team_b"),
                    "situational_confidence_adjustment": lambda d: _num(d.get("situational_confidence_boost")),
                },
            )
            self._write_contract_csv("situational_signals_game_level.csv", situ, self.paths.situational_signals_game_level)

            mc = self._build_frame(
                base=upcoming,
                file_name="game_monte_carlo_outputs.csv",
                now_utc=now_utc,
                mapping={
                    "game_id": "game_id",
                    "phase": lambda d: d.get("phase", pd.Series("regular_season", index=d.index)),
                    "round_name": lambda d: d.get("round_name", np.nan),
                    "team_a": "home_team",
                    "team_b": "away_team",
                    "simulation_count": 500,
                    "mean_margin_team_a": lambda d: _num(d.get("projected_margin_pre_mc")),
                    "median_margin_team_a": lambda d: _num(d.get("projected_margin_pre_mc")),
                    "margin_std_dev": lambda d: _num(d.get("mc_volatility")),
                    "margin_p10": lambda d: _num(d.get("mc_margin_p10")),
                    "margin_p50": lambda d: _num(d.get("projected_margin_pre_mc")),
                    "margin_p90": lambda d: _num(d.get("mc_margin_p90")),
                    "mean_total": lambda d: _num(d.get("projected_total_ctx")),
                    "median_total": lambda d: _num(d.get("projected_total_ctx")),
                    "total_std_dev": lambda d: _num(d.get("mc_volatility")),
                    "total_p10": lambda d: _num(d.get("mc_total_p10")),
                    "total_p50": lambda d: _num(d.get("projected_total_ctx")),
                    "total_p90": lambda d: _num(d.get("mc_total_p90")),
                    "team_a_win_probability_mc": lambda d: _num(d.get("mc_home_win_prob")),
                    "team_b_win_probability_mc": lambda d: _num(d.get("mc_away_win_prob")),
                    "team_a_cover_probability_mc": lambda d: _num(d.get("mc_home_cover_prob")),
                    "team_b_cover_probability_mc": lambda d: _num(d.get("mc_away_cover_prob")),
                    "volatility_score": lambda d: _num(d.get("mc_volatility")),
                    "fragility_score": lambda d: _num(d.get("mc_volatility")) / 20.0,
                },
            )
            mc["over_probability_mc"] = (1.0 / (1.0 + np.exp(-(_num(upcoming.get("projected_total_ctx")) - _num(upcoming.get("market_total"))) / 6.0))).clip(0.01, 0.99)
            mc["under_probability_mc"] = 1.0 - _num(mc["over_probability_mc"])
            self._write_contract_csv("game_monte_carlo_outputs.csv", mc, self.paths.game_monte_carlo_outputs)

            preds = self._build_frame(
                base=upcoming,
                file_name="game_predictions_master.csv",
                now_utc=now_utc,
                mapping={
                    "run_id": run_id,
                    "model_version": MODEL_VERSION,
                    "game_id": "game_id",
                    "phase": lambda d: d.get("phase", pd.Series("regular_season", index=d.index)),
                    "round_name": lambda d: d.get("round_name", np.nan),
                    "is_postseason": lambda d: _bool(d.get("is_postseason")),
                    "is_conference_tournament": lambda d: _bool(d.get("is_conference_tournament")),
                    "is_ncaa_tournament": lambda d: _bool(d.get("is_ncaa_tournament")),
                    "site_type": lambda d: np.where(_bool(d.get("is_neutral")), "neutral", "home"),
                    "team_a": "home_team",
                    "team_b": "away_team",
                    "market_spread_team_a": lambda d: _num(d.get("market_spread")),
                    "market_spread_team_b": lambda d: -_num(d.get("market_spread")),
                    "market_spread": lambda d: _num(d.get("market_spread")),
                    "market_fav": lambda d: np.where(
                        _num(d.get("market_spread")) < 0,
                        d.get("home_team"),
                        np.where(_num(d.get("market_spread")) > 0, d.get("away_team"), "pickem"),
                    ),
                    "market_total": lambda d: _num(d.get("market_total")),
                    "market_moneyline_team_a": lambda d: _num(d.get("moneyline_home")),
                    "market_moneyline_team_b": lambda d: _num(d.get("moneyline_away")),
                    "model_spread_team_a": lambda d: _num(d.get("projected_spread")),
                    "model_spread_team_b": lambda d: -_num(d.get("projected_spread")),
                    "prediction_winner": lambda d: np.where(
                        _num(d.get("projected_spread")) < 0,
                        d.get("home_team"),
                        np.where(_num(d.get("projected_spread")) > 0, d.get("away_team"), "pickem"),
                    ),
                    "model_total": lambda d: _num(d.get("projected_total_ctx")),
                    "team_a_win_probability": lambda d: _num(d.get("win_prob_home")).clip(0.0, 1.0),
                    "team_b_win_probability": lambda d: (1.0 - _num(d.get("win_prob_home"))).clip(0.0, 1.0),
                    "team_a_cover_probability": lambda d: _num(d.get("ats_cover_prob_home")).clip(0.0, 1.0),
                    "team_b_cover_probability": lambda d: (1.0 - _num(d.get("ats_cover_prob_home"))).clip(0.0, 1.0),
                    "spread_edge_team_a": lambda d: _num(d.get("edge_home")),
                    "spread_edge_team_b": lambda d: -_num(d.get("edge_home")),
                    "spread_confidence": lambda d: _num(d.get("confidence_score")).clip(0.0, 100.0),
                    "base_model_name": "ides_base_blend",
                    "context_layer_applied": True,
                    "situational_layer_applied": True,
                    "monte_carlo_layer_applied": True,
                    "agreement_bucket": lambda d: d.get("agreement_bucket"),
                    "final_bet_flag": lambda d: d.get("bet_recommendation", pd.Series("PASS", index=d.index)).astype(str).ne("PASS"),
                    "notes": lambda d: d.get("context_summary"),
                },
            )
            preds["over_probability"] = (1.0 / (1.0 + np.exp(-(_num(preds["model_total"]) - _num(preds["market_total"])) / 6.0))).clip(0.01, 0.99)
            preds["under_probability"] = 1.0 - _num(preds["over_probability"])
            preds["total_edge_over"] = _num(preds["model_total"]) - _num(preds["market_total"])
            preds["total_edge_under"] = -_num(preds["total_edge_over"])
            preds["spread_pick"] = np.where(_num(preds["spread_edge_team_a"]) >= 0, preds["team_a"], preds["team_b"])
            preds["total_pick"] = np.where(_num(preds["over_probability"]) >= 0.5, "over", "under")
            preds["total_confidence"] = (_num(preds["total_edge_over"]).abs() * 8.0).clip(0.0, 100.0)
            preds["spread_bet_flag"] = preds["final_bet_flag"].astype(bool)
            total_recommend_mask = (
                (_num(preds["total_edge_over"]).abs() >= 2.0)
                & (_num(preds["total_confidence"]) >= 55.0)
                & (_num(preds["over_probability"]).sub(0.5).abs() >= 0.03)
            )
            if mc_mode == "confidence_filter" and "mc_filter_pass" in preds.columns:
                total_recommend_mask = total_recommend_mask & preds["mc_filter_pass"].astype(bool)
            preds["total_bet_flag"] = total_recommend_mask.astype(bool)
            preds["final_bet_flag"] = preds["spread_bet_flag"] | preds["total_bet_flag"]

            spread_bets = preds[preds["spread_bet_flag"]].copy()
            spread_bets["bet_type"] = "spread"
            spread_bets["bet_side"] = np.where(spread_bets["spread_pick"] == spread_bets["team_a"], "team_a", "team_b")
            spread_bets["market_line"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["market_spread_team_a"], spread_bets["market_spread_team_b"])
            spread_bets["model_line"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["model_spread_team_a"], spread_bets["model_spread_team_b"])
            spread_bets["edge"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["spread_edge_team_a"], spread_bets["spread_edge_team_b"])
            spread_bets["win_probability"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["team_a_win_probability"], spread_bets["team_b_win_probability"])
            spread_bets["cover_probability"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["team_a_cover_probability"], spread_bets["team_b_cover_probability"])
            spread_bets["confidence"] = spread_bets["spread_confidence"]
            spread_bets["bet_reason_short"] = "spread edge + confidence"

            total_bets = preds[preds["total_bet_flag"]].copy()
            total_bets["bet_type"] = "total"
            total_bets["bet_side"] = total_bets["total_pick"].str.lower()
            total_bets["market_line"] = _num(total_bets["market_total"])
            total_bets["model_line"] = _num(total_bets["model_total"])
            total_bets["edge"] = np.where(total_bets["bet_side"].eq("over"), _num(total_bets["total_edge_over"]), _num(total_bets["total_edge_under"]))
            total_bets["win_probability"] = np.where(total_bets["bet_side"].eq("over"), _num(total_bets["over_probability"]), _num(total_bets["under_probability"]))
            total_bets["cover_probability"] = np.nan
            total_bets["confidence"] = _num(total_bets["total_confidence"])
            total_bets["bet_reason_short"] = "total edge + confidence"

            bets = pd.concat([spread_bets, total_bets], ignore_index=True, sort=False)
            bets["situational_support"] = bets["agreement_bucket"].astype(str).str.contains("Situational", na=False)
            bets["monte_carlo_support"] = bets["agreement_bucket"].astype(str).str.contains("Monte Carlo", na=False)
            bets["recommended_stake_units"] = ((_num(bets["confidence"]) - 50.0) / 20.0).clip(0.0, 3.0).round(2)
            bets = bets.sort_values(["confidence", "edge"], ascending=[False, False], kind="mergesort")
            bets["bet_rank"] = np.arange(1, len(bets) + 1)
            bets["created_at_utc"] = now_utc

            preds = self._write_contract_csv("game_predictions_master.csv", preds, self.paths.game_predictions_master)
            bets = self._write_contract_csv("bet_recommendations.csv", bets, self.paths.bet_recommendations)

            bet_game_ids = set(bets["game_id"].astype(str)) if len(bets) else set()
            watch = preds[(~preds["game_id"].astype(str).isin(bet_game_ids)) & ((_num(preds["spread_edge_team_a"]).abs() >= 1.0) | (_num(preds["total_edge_over"]).abs() >= 2.0))].copy()
            watch["watchlist_reason"] = "edge near threshold"
            watch["current_market_spread"] = watch["market_spread_team_a"]
            watch["current_market_total"] = watch["market_total"]
            watch["model_spread"] = watch["model_spread_team_a"]
            watch["spread_edge"] = watch["spread_edge_team_a"]
            watch["total_edge"] = watch["total_edge_over"]
            watch["trigger_to_promote"] = "line movement or confidence increase"
            watch["created_at_utc"] = now_utc
            watch = self._write_contract_csv("watchlist_games.csv", watch, self.paths.watchlist_games)

            no_bet = preds[(~preds["game_id"].astype(str).isin(bet_game_ids)) & (~preds["game_id"].isin(watch["game_id"]))].copy()
            no_bet["market_spread"] = no_bet["market_spread_team_a"]
            no_bet["model_spread"] = no_bet["model_spread_team_a"]
            no_bet["spread_edge"] = no_bet["spread_edge_team_a"]
            no_bet["total_edge"] = no_bet["total_edge_over"]
            no_bet["no_bet_reason"] = "thresholds_not_met"
            no_bet["primary_blocker"] = "confidence_or_edge"
            no_bet["secondary_blocker"] = np.nan
            no_bet["created_at_utc"] = now_utc
            no_bet = self._write_contract_csv("no_bet_explanations.csv", no_bet, self.paths.no_bet_explanations)

            summary = pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "model_version": MODEL_VERSION,
                        "season": preds["season"].iloc[0] if len(preds) else np.nan,
                        "game_date_pst": preds["game_date_pst"].iloc[0] if len(preds) else np.nan,
                        "games_on_card": int(len(preds)),
                        "spread_bets_count": int((bets.get("bet_type", pd.Series(dtype=str)) == "spread").sum()) if len(bets) else 0,
                        "total_bets_count": int((bets.get("bet_type", pd.Series(dtype=str)) == "total").sum()) if len(bets) else 0,
                        "watchlist_count": int(len(watch)),
                        "highest_confidence_bet": (
                            (
                                f"{bets.iloc[0]['team_a']} vs {bets.iloc[0]['team_b']} {bets.iloc[0]['bet_type']}:{bets.iloc[0]['bet_side']}"
                            )
                            if len(bets)
                            else np.nan
                        ),
                        "highest_edge_bet": (
                            (
                                f"{bets.sort_values('edge', ascending=False).iloc[0]['team_a']} vs {bets.sort_values('edge', ascending=False).iloc[0]['team_b']} "
                                f"{bets.sort_values('edge', ascending=False).iloc[0]['bet_type']}:{bets.sort_values('edge', ascending=False).iloc[0]['bet_side']}"
                            )
                            if len(bets)
                            else np.nan
                        ),
                        "avg_spread_edge": float(_num(preds["spread_edge_team_a"]).abs().mean()) if len(preds) else np.nan,
                        "avg_total_edge": float(_num(preds["total_edge_over"]).abs().mean()) if len(preds) else np.nan,
                        "avg_confidence": float(_num(bets["confidence"]).mean()) if len(bets) else np.nan,
                        "notes": "ides run summary",
                        "created_at_utc": now_utc,
                    }
                ]
            )
            summary = self._write_contract_csv("daily_card_summary.csv", summary, self.paths.daily_card_summary)

            agreement = summarize_agreement_buckets(hist_scored).rename(
                columns={"su_accuracy": "straight_up_win_pct", "ats_accuracy": "ats_win_pct", "avg_edge": "avg_spread_edge", "confidence_mean": "avg_spread_confidence"}
            )
            agreement["run_id"] = run_id
            agreement["model_version"] = MODEL_VERSION
            agreement["phase"] = "all"
            agreement["round_name"] = "all"
            agreement["over_win_pct"] = np.nan
            agreement["under_win_pct"] = np.nan
            agreement["avg_total_edge"] = np.nan
            agreement["avg_total_confidence"] = np.nan
            agreement["roi_spread"] = np.nan
            agreement["roi_total"] = np.nan
            agreement["notes"] = "historical agreement summary"
            agreement["created_at_utc"] = now_utc
            agreement = self._write_contract_csv("agreement_analysis_results.csv", agreement, self.paths.agreement_analysis_results)

            outputs = {
                "games_schedule_master": str(self.paths.games_schedule_master),
                "team_game_boxscores": str(self.paths.team_game_boxscores),
                "player_game_boxscores": str(self.paths.player_game_boxscores),
                "team_rolling_features": str(self.paths.team_rolling_features),
                "game_matchup_features": str(self.paths.game_matchup_features),
                "game_totals_features": str(self.paths.game_totals_features),
                "game_context_adjustments": str(self.paths.game_context_adjustments),
                "situational_signals_game_level": str(self.paths.situational_signals_game_level),
                "game_monte_carlo_outputs": str(self.paths.game_monte_carlo_outputs),
                "game_predictions_master": str(self.paths.game_predictions_master),
                "bet_recommendations": str(self.paths.bet_recommendations),
                "watchlist_games": str(self.paths.watchlist_games),
                "no_bet_explanations": str(self.paths.no_bet_explanations),
                "daily_card_summary": str(self.paths.daily_card_summary),
                "agreement_analysis_results": str(self.paths.agreement_analysis_results),
            }

            pred_schema = validate_predictions(preds)
            bet_schema = validate_bet_recs(bets)
            agr_schema = validate_agreement(agreement)
            schema_issues: list[str] = []
            if not pred_schema.ok:
                schema_issues.append(f"game_predictions_master missing columns: {pred_schema.missing_columns}")
            if not bet_schema.ok:
                schema_issues.append(f"bet_recommendations missing columns: {bet_schema.missing_columns}")
            if not agr_schema.ok:
                schema_issues.append(f"agreement_analysis_results missing columns: {agr_schema.missing_columns}")
            stages.append(StageRecord("Contract Validation", "PASS" if not schema_issues else "FAIL", schema_issues, {"rows": int(len(preds))}))

            final_status = "PASS"
            if safety.status == "FAIL" or schema_issues:
                final_status = "FAIL"
            elif safety.status == "WARN" or integrity.status == "WARN":
                final_status = "WARN"

            manifest = {
                "run_type": "predict",
                "run_id": run_id,
                "model_version": MODEL_VERSION,
                "run_at_utc": now_utc,
                "as_of_utc": as_of.isoformat(),
                "mc_mode": mc_mode,
                "status": final_status,
                "stages": [asdict(s) for s in stages],
                "outputs": outputs,
            }
            write_json(self.paths.run_manifest, manifest)
            outputs["run_manifest"] = str(self.paths.run_manifest)
            self._append_pipeline_log(run_id, stages)
            return RunResult(ok=final_status in {"PASS", "WARN"}, status=final_status, stages=stages, outputs=outputs, error=None if final_status in {"PASS", "WARN"} else "predict_failed")
        except Exception as exc:
            self._append_pipeline_log(run_id, stages, pretty_exception(exc))
            return RunResult(ok=False, status="FAIL", stages=stages, outputs=outputs, error=pretty_exception(exc))

    def backtest(
        self,
        *,
        as_of: pd.Timestamp,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> RunResult:
        stages: list[StageRecord] = []
        outputs: dict[str, str] = {}
        run_id = f"ides_backtest_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}"
        now_utc = utc_now_iso()
        try:
            ds = build_data_steward_frame(data_dir=self.data_dir, as_of=as_of, hours_ahead=48)
            hist = ds.historical_games.copy()
            if start_date:
                start_ts = pd.to_datetime(start_date, utc=True, errors="coerce")
                hist = hist[pd.to_datetime(hist.get("game_datetime_utc"), utc=True, errors="coerce") >= start_ts]
            if end_date:
                end_ts = pd.to_datetime(end_date, utc=True, errors="coerce")
                hist = hist[pd.to_datetime(hist.get("game_datetime_utc"), utc=True, errors="coerce") <= end_ts]
            if hist.empty:
                self._append_pipeline_log(run_id, stages, "no_historical_games")
                return RunResult(ok=False, status="BLOCKED", stages=stages, outputs=outputs, error="no_historical_games")

            scorecard = run_variant_backtest(hist)
            backtest = pd.DataFrame(index=scorecard.index)
            backtest["run_id"] = run_id
            backtest["model_version"] = MODEL_VERSION
            backtest["variant_name"] = scorecard.get("variant_id")
            backtest["phase"] = "all"
            backtest["games_tested"] = scorecard.get("sample_size")
            backtest["spread_mae"] = scorecard.get("spread_mae")
            backtest["spread_rmse"] = scorecard.get("spread_rmse")
            backtest["winner_accuracy"] = scorecard.get("winner_accuracy")
            backtest["ats_win_pct_all"] = scorecard.get("ats_accuracy")
            backtest["ats_win_pct_edge_gt_1"] = scorecard.get("ats_win_pct_edge_gt_1")
            backtest["ats_win_pct_edge_gt_2"] = scorecard.get("ats_win_pct_edge_gt_2")
            backtest["ats_win_pct_edge_gt_3"] = scorecard.get("ats_win_pct_edge_gt_3")
            backtest["totals_mae"] = scorecard.get("totals_mae")
            backtest["over_under_win_pct_all"] = scorecard.get("over_under_win_pct_all")
            backtest["total_win_pct_edge_gt_1"] = scorecard.get("total_win_pct_edge_gt_1")
            backtest["total_win_pct_edge_gt_2"] = scorecard.get("total_win_pct_edge_gt_2")
            backtest["total_win_pct_edge_gt_3"] = scorecard.get("total_win_pct_edge_gt_3")
            backtest["win_probability_brier_score"] = scorecard.get("calibration_brier")
            backtest["win_probability_calibration_error"] = np.nan
            backtest["mc_probability_calibration_error"] = np.nan
            backtest["avg_spread_edge"] = scorecard.get("avg_spread_edge")
            backtest["avg_total_edge"] = scorecard.get("avg_total_edge")
            backtest["roi_spread"] = np.nan
            backtest["roi_total"] = np.nan
            backtest["notes"] = "ides variant backtest"
            backtest["created_at_utc"] = now_utc
            backtest = self._write_contract_csv("backtest_model_summary.csv", backtest, self.paths.backtest_model_summary)
            outputs["backtest_model_summary"] = str(self.paths.backtest_model_summary)

            schema = validate_backtest_summary(backtest)
            issues = [] if schema.ok else [f"backtest_model_summary missing columns: {schema.missing_columns}"]
            stages.append(StageRecord("Evaluation Agent", "PASS" if not issues else "FAIL", issues, {"rows": int(len(backtest))}))

            status = "PASS" if not issues else "FAIL"
            manifest = {
                "run_type": "backtest",
                "run_id": run_id,
                "model_version": MODEL_VERSION,
                "run_at_utc": now_utc,
                "as_of_utc": as_of.isoformat(),
                "start_date": start_date,
                "end_date": end_date,
                "status": status,
                "stages": [asdict(s) for s in stages],
                "outputs": outputs,
            }
            write_json(self.paths.run_manifest, manifest)
            outputs["run_manifest"] = str(self.paths.run_manifest)
            self._append_pipeline_log(run_id, stages)
            return RunResult(ok=status == "PASS", status=status, stages=stages, outputs=outputs, error=None if status == "PASS" else "backtest_failed")
        except Exception as exc:
            self._append_pipeline_log(run_id, stages, pretty_exception(exc))
            return RunResult(ok=False, status="FAIL", stages=stages, outputs=outputs, error=pretty_exception(exc))
