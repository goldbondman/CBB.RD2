from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .calibration import (
    build_backtest_calibration_policy,
    load_backtest_calibration_policy,
    resolve_spread_thresholds,
)
from .config import (
    DATA_DIR,
    ENABLE_UPSET_LAYER,
    SPREAD_BET_ATS_PROB_EDGE_MIN,
    SPREAD_BET_CONFIDENCE_MIN,
    SPREAD_BET_EDGE_MIN,
    output_paths,
)
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
from .utils import ensure_dir, home_spread_to_margin, pretty_exception, utc_now_iso, write_json


MODEL_VERSION = "ides_of_march_v1"

SPREAD_HALF_COLUMNS_GAME_PRED = [
    "market_spread_team_a",
    "market_spread_team_b",
    "market_spread",
    "model_spread_team_a",
    "model_spread_team_b",
    "spread_edge_team_a",
    "spread_edge_team_b",
    "spread_edge_for_pick",
]

PROB_COLUMNS_GAME_PRED = [
    "team_a_win_probability",
    "team_b_win_probability",
    "team_a_cover_probability",
    "team_b_cover_probability",
    "team_a_win_probability_mc",
    "team_b_win_probability_mc",
    "team_a_cover_probability_mc",
    "team_b_cover_probability_mc",
    "over_probability",
    "under_probability",
]

PROB_COLUMNS_BET_RECS = [
    "win_probability",
    "cover_probability",
    "mc_win_probability",
    "mc_cover_probability",
    "over_probability",
    "under_probability",
]


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


def _american_to_implied_prob(series: Any) -> pd.Series:
    odds = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=odds.index, dtype=float)
    neg = odds < 0
    pos = odds > 0
    out.loc[neg] = (-odds.loc[neg]) / ((-odds.loc[neg]) + 100.0)
    out.loc[pos] = 100.0 / (odds.loc[pos] + 100.0)
    return out


def _prob_to_fair_american(series: Any) -> pd.Series:
    p = pd.to_numeric(series, errors="coerce").clip(0.01, 0.99)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    fav = p >= 0.5
    dog = p < 0.5
    out.loc[fav] = -100.0 * p.loc[fav] / (1.0 - p.loc[fav])
    out.loc[dog] = 100.0 * (1.0 - p.loc[dog]) / p.loc[dog]
    return out


def _round_series_to_half(series: Any) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return np.round(vals * 2.0) / 2.0


def _round_series_to_hundredth(series: Any) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.round(2)


def _apply_output_rounding(file_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if file_name == "game_predictions_master.csv":
        for col in SPREAD_HALF_COLUMNS_GAME_PRED:
            if col in out.columns:
                out[col] = _round_series_to_half(out[col])
        for col in PROB_COLUMNS_GAME_PRED:
            if col in out.columns:
                out[col] = _round_series_to_hundredth(out[col])
        return out

    if file_name == "bet_recommendations.csv":
        spread_mask = out.get("bet_type", pd.Series("", index=out.index)).astype(str).eq("spread")
        for col in ("market_line", "model_line", "edge"):
            if col in out.columns:
                rounded = _round_series_to_half(out[col])
                out[col] = np.where(spread_mask, rounded, out[col])
        for col in PROB_COLUMNS_BET_RECS:
            if col in out.columns:
                out[col] = _round_series_to_hundredth(out[col])
    return out


def _compute_blowout_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    adj_abs = _num(frame.get("adj_em_margin_l12")).abs()
    to_abs = _num(frame.get("to_margin_l5")).abs()
    oreb_abs = _num(frame.get("oreb_margin_l5")).abs()
    rest_abs = _num(frame.get("rest_diff")).abs()
    mc_home = _num(frame.get("mc_home_win_prob", pd.Series(np.nan, index=frame.index)))
    if not isinstance(mc_home, pd.Series):
        mc_home = pd.Series(mc_home, index=frame.index)
    win_home = _num(frame.get("win_prob_home", pd.Series(np.nan, index=frame.index)))
    if not isinstance(win_home, pd.Series):
        win_home = pd.Series(win_home, index=frame.index)
    mc_win = mc_home.fillna(win_home)

    out["blowout_score_adj"] = ((adj_abs - 8.0) / 10.0).clip(lower=0.0) * 1.2
    out["blowout_score_to"] = ((to_abs - 0.012) / 0.01).clip(lower=0.0) * 0.08
    out["blowout_score_oreb"] = ((oreb_abs - 0.03) / 0.01).clip(lower=0.0) * 0.10
    out["blowout_score_rest"] = ((rest_abs - 1.0).clip(lower=0.0)) * 0.05
    out["blowout_score_mcw"] = ((mc_win.sub(0.5).abs() - 0.10).clip(lower=0.0)) * 1.3
    out["blowout_score_total"] = (
        _num(out["blowout_score_adj"])
        + _num(out["blowout_score_to"])
        + _num(out["blowout_score_oreb"])
        + _num(out["blowout_score_rest"])
        + _num(out["blowout_score_mcw"])
    )
    return out


def _apply_spread_flip_logic(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    out["model_pick_raw"] = np.where(_num(out.get("spread_edge_team_a")) >= 0, out.get("team_a"), out.get("team_b"))
    raw_market_line = np.where(out["model_pick_raw"].eq(out["team_a"]), _num(out.get("market_spread_team_a")), _num(out.get("market_spread_team_b")))
    raw_edge = np.where(out["model_pick_raw"].eq(out["team_a"]), _num(out.get("spread_edge_team_a")), _num(out.get("spread_edge_team_b")))
    raw_dog = _num(pd.Series(raw_market_line, index=out.index)) > 0
    raw_line = _num(pd.Series(raw_market_line, index=out.index))
    raw_edge_abs = _num(pd.Series(raw_edge, index=out.index)).abs()
    blowout_total = _num(out.get("blowout_score_total"))

    flip_dog_5_to_7_5 = raw_dog & raw_line.between(5.0, 7.5, inclusive="both") & ((raw_edge_abs >= 5.0) | (blowout_total >= 1.10))
    # Broader blowout override for larger market dogs when volatility/mismatch is elevated.
    flip_dog_8_plus_blowout = raw_dog & (raw_line >= 8.0) & (blowout_total >= 1.20) & (raw_edge_abs >= 3.0)
    out["model_pick_flipped"] = (flip_dog_5_to_7_5 | flip_dog_8_plus_blowout).astype(bool)
    out["model_pick_flip_reason"] = np.select(
        [flip_dog_5_to_7_5, flip_dog_8_plus_blowout],
        ["dog_5_to_7_5_inverse", "dog_8_plus_blowout_inverse"],
        default="",
    )
    out["model_pick"] = np.where(out["model_pick_flipped"], np.where(out["model_pick_raw"].eq(out["team_a"]), out["team_b"], out["team_a"]), out["model_pick_raw"])

    out["spread_pick"] = out["model_pick"]
    pick_edge = pd.Series(
        np.where(out["model_pick"].eq(out["team_a"]), _num(out.get("spread_edge_team_a")), _num(out.get("spread_edge_team_b"))),
        index=out.index,
    )
    out["spread_edge_for_pick"] = _num(pick_edge).abs()
    return out


def _apply_totals_flip_logic(preds: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    out["total_pick_raw"] = np.where(_num(out.get("over_probability")) >= 0.5, "over", "under")

    market_total = _num(out.get("market_total"))
    blowout_total = _num(out.get("blowout_score_total"))
    mc_vol = _num(frame.get("mc_volatility"))
    expected_pace = _num(frame.get("expected_pace"))
    threepa_sum = _num(frame.get("home_three_par_l5")) + _num(frame.get("away_three_par_l5"))

    # In high-total/high-variance spots, modeled unders can be too aggressive.
    flip_under_chaos = (
        out["total_pick_raw"].eq("under")
        & (market_total >= 145.0)
        & (blowout_total >= 1.10)
        & (threepa_sum >= 0.78)
        & (mc_vol >= 11.0)
        & (expected_pace >= 66.0)
    )

    out["total_pick_flipped"] = flip_under_chaos.astype(bool)
    out["total_pick_flip_reason"] = np.where(out["total_pick_flipped"], "under_high_total_chaos_inverse", "")
    out["total_pick"] = np.where(out["total_pick_flipped"], "over", out["total_pick_raw"])
    return out


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


def _build_recent_predictions_results_frame(
    scored_history: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    run_id: str,
    now_utc: str,
) -> pd.DataFrame:
    if scored_history.empty:
        return pd.DataFrame()

    hist = scored_history.copy()
    game_dt_utc = pd.to_datetime(hist.get("game_datetime_utc"), utc=True, errors="coerce")
    game_dt_pst = game_dt_utc.dt.tz_convert("America/Los_Angeles")
    game_day_pst = game_dt_pst.dt.normalize()
    as_of_pst = as_of.tz_convert("America/Los_Angeles")
    window_start = as_of_pst.normalize() - pd.Timedelta(days=2)
    window_end = as_of_pst.normalize() + pd.Timedelta(days=2)

    state_text = hist.get("state", pd.Series("", index=hist.index)).fillna("").astype(str).str.lower()
    status_text = hist.get("status_desc", pd.Series("", index=hist.index)).fillna("").astype(str).str.lower()
    completed_flag = (
        _bool(hist.get("completed"))
        | state_text.isin({"post", "final", "completed"})
        | status_text.str.contains("final", na=False)
        | status_text.str.contains("complete", na=False)
    )
    completed_flag = completed_flag & _num(hist.get("actual_margin")).notna() & _num(hist.get("actual_total")).notna()
    in_window = game_day_pst.between(window_start, window_end, inclusive="both")
    mask = in_window & completed_flag & game_dt_utc.notna()
    if not bool(mask.any()):
        return pd.DataFrame()

    frame = hist.loc[mask].copy()
    tf = _time_fields(frame)

    market_spread = _num(frame.get("market_spread"))
    model_spread = _num(frame.get("projected_spread"))
    spread_edge = _num(frame.get("edge_home"))
    market_total = _num(frame.get("market_total"))
    model_total = _num(frame.get("projected_total_ctx"))
    win_prob_home = _num(frame.get("win_prob_home")).clip(0.0, 1.0)
    cover_prob_home = _num(frame.get("ats_cover_prob_home")).clip(0.0, 1.0)
    over_prob = (1.0 / (1.0 + np.exp(-(model_total - market_total) / 6.0))).clip(0.01, 0.99)

    prediction_winner = np.where(
        model_spread < 0,
        frame.get("home_team"),
        np.where(model_spread > 0, frame.get("away_team"), "pickem"),
    )
    predicted_spread_pick = np.where(cover_prob_home >= 0.5, frame.get("home_team"), frame.get("away_team"))
    predicted_total_pick = np.where(model_total >= market_total, "over", "under")

    actual_margin = _num(frame.get("actual_margin"))
    actual_total = _num(frame.get("actual_total"))
    home_won = _num(frame.get("home_won"))
    actual_winner = np.where(home_won > 0.5, frame.get("home_team"), frame.get("away_team"))
    actual_home_covered = actual_margin > home_spread_to_margin(market_spread)

    total_result = actual_total - market_total
    total_pick_correct = np.where(
        market_total.notna() & actual_total.notna() & (total_result != 0.0),
        np.where(predicted_total_pick == "over", total_result > 0.0, total_result < 0.0),
        np.nan,
    )
    spread_pick_correct = np.where(
        market_spread.notna() & actual_margin.notna(),
        np.where(predicted_spread_pick == frame.get("home_team"), actual_home_covered, ~actual_home_covered),
        np.nan,
    )
    winner_pick_correct = np.where(home_won.notna(), prediction_winner == actual_winner, np.nan)

    home_score = _num(frame.get("home_score"))
    away_score = _num(frame.get("away_score"))
    derived_home = (actual_total + actual_margin) / 2.0
    derived_away = (actual_total - actual_margin) / 2.0
    home_score = home_score.where(home_score.notna(), derived_home)
    away_score = away_score.where(away_score.notna(), derived_away)

    game_status = frame.get("game_status", frame.get("status_desc", frame.get("state", pd.Series("completed", index=frame.index))))
    game_status = pd.Series(game_status, index=frame.index).fillna("completed")

    out = pd.DataFrame(index=frame.index)
    out["run_id"] = run_id
    out["model_version"] = MODEL_VERSION
    out["as_of_utc"] = as_of.isoformat()
    out["window_start_pst"] = window_start.strftime("%Y-%m-%d")
    out["window_end_pst"] = window_end.strftime("%Y-%m-%d")
    out["game_id"] = frame.get("game_id")
    out["event_id"] = frame.get("event_id", frame.get("game_id"))
    out["season"] = tf.get("season")
    out["game_date_pst"] = tf.get("game_date_pst")
    out["game_start_time_pst"] = tf.get("game_start_time_pst")
    out["game_start_datetime_pst"] = tf.get("game_start_datetime_pst")
    out["game_start_datetime_utc"] = tf.get("game_start_datetime_utc")
    out["team_a"] = frame.get("home_team")
    out["team_b"] = frame.get("away_team")
    out["game_status"] = game_status
    out["line_source_used"] = frame.get("line_source_used")
    out["historical_odds_source"] = frame.get("historical_odds_source")
    out["market_spread_team_a"] = market_spread
    out["model_spread_team_a"] = model_spread
    out["spread_edge_team_a"] = spread_edge
    out["market_total"] = market_total
    out["model_total"] = model_total
    out["total_edge_over"] = model_total - market_total
    out["team_a_win_probability"] = win_prob_home
    out["team_a_cover_probability"] = cover_prob_home
    out["over_probability"] = over_prob
    out["prediction_winner"] = prediction_winner
    out["predicted_spread_pick"] = predicted_spread_pick
    out["predicted_total_pick"] = predicted_total_pick
    out["team_a_score_actual"] = home_score.round(0)
    out["team_b_score_actual"] = away_score.round(0)
    out["actual_winner"] = actual_winner
    out["actual_margin"] = actual_margin
    out["actual_total"] = actual_total
    out["actual_home_covered"] = actual_home_covered.astype(object)
    out["winner_pick_correct"] = pd.Series(winner_pick_correct, index=frame.index).astype(object)
    out["spread_pick_correct"] = pd.Series(spread_pick_correct, index=frame.index).astype(object)
    out["total_pick_correct"] = pd.Series(total_pick_correct, index=frame.index).astype(object)
    out["completed_flag"] = True
    out["created_at_utc"] = now_utc
    out["updated_at_utc"] = now_utc

    return out.sort_values(["game_start_datetime_utc", "game_id"], ascending=[False, False], kind="mergesort")


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
            "recent_predictions_results.csv": self.paths.recent_predictions_results,
            "agreement_analysis_results.csv": self.paths.agreement_analysis_results,
            "backtest_model_summary.csv": self.paths.backtest_model_summary,
            "backtest_edge_band_summary.csv": self.paths.backtest_edge_band_summary,
            "backtest_bet_ledger.csv": self.paths.backtest_bet_ledger,
            "backtest_kelly_summary.csv": self.paths.backtest_kelly_summary,
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
                # String mappings can represent either:
                # 1) source-column references (when the source column exists), or
                # 2) literal constants (run_id/model_version/static labels).
                if spec in base.columns:
                    value = base.get(spec)
                    if isinstance(value, pd.DataFrame):
                        value = value.iloc[:, 0]
                    out[col] = value
                else:
                    out[col] = spec
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

    @staticmethod
    def _default_spread_thresholds() -> dict[str, float]:
        return {
            "edge_min": float(SPREAD_BET_EDGE_MIN),
            "confidence_min": float(SPREAD_BET_CONFIDENCE_MIN),
            "ats_prob_edge_min": float(SPREAD_BET_ATS_PROB_EDGE_MIN),
        }

    def audit(self, *, as_of: pd.Timestamp, hours_ahead: int = 48, hours_back: int = 1) -> RunResult:
        stages: list[StageRecord] = []
        run_id = f"ides_audit_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}"
        try:
            ds = build_data_steward_frame(data_dir=self.data_dir, as_of=as_of, hours_ahead=hours_ahead, hours_back=hours_back)
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

    def predict(self, *, as_of: pd.Timestamp, mc_mode: str = "confidence_only", hours_ahead: int = 48, hours_back: int = 1) -> RunResult:
        stages: list[StageRecord] = []
        outputs: dict[str, str] = {}
        run_id = f"ides_predict_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}"
        now_utc = utc_now_iso()
        try:
            ds = build_data_steward_frame(data_dir=self.data_dir, as_of=as_of, hours_ahead=hours_ahead, hours_back=hours_back)
            integrity = run_data_integrity_audit(ds.upcoming_games, ds.historical_games)
            stages.append(StageRecord("Data Integrity Auditor", integrity.status, integrity.issues, integrity.metrics))
            if integrity.status in {"FAIL", "BLOCKED"}:
                self._append_pipeline_log(run_id, stages, "integrity_gate_failed")
                return RunResult(ok=False, status=integrity.status, stages=stages, outputs=outputs, error="integrity_gate_failed")
            if ds.upcoming_games.empty:
                hist_scored = self._score_history_for_reports(ds.historical_games, mc_mode=mc_mode)
                stages.append(
                    StageRecord(
                        "Layer Stack",
                        "PASS",
                        [f"upcoming_rows=0 upset_layer_enabled={ENABLE_UPSET_LAYER}"],
                        {"rows": 0, "upset_layer_enabled": int(ENABLE_UPSET_LAYER)},
                    )
                )
                stages.append(
                    StageRecord(
                        "Model Safety Auditor",
                        "WARN",
                        ["No upcoming games: model safety audit skipped."],
                        {"rows": 0},
                    )
                )

                self._write_contract_csv("games_schedule_master.csv", pd.DataFrame(), self.paths.games_schedule_master)
                self._write_contract_csv("team_game_boxscores.csv", pd.DataFrame(), self.paths.team_game_boxscores)
                self._write_contract_csv("player_game_boxscores.csv", pd.DataFrame(), self.paths.player_game_boxscores)
                self._write_contract_csv("team_rolling_features.csv", pd.DataFrame(), self.paths.team_rolling_features)
                self._write_contract_csv("game_matchup_features.csv", pd.DataFrame(), self.paths.game_matchup_features)
                self._write_contract_csv("game_totals_features.csv", pd.DataFrame(), self.paths.game_totals_features)
                self._write_contract_csv("game_context_adjustments.csv", pd.DataFrame(), self.paths.game_context_adjustments)
                self._write_contract_csv("situational_signals_game_level.csv", pd.DataFrame(), self.paths.situational_signals_game_level)
                self._write_contract_csv("game_monte_carlo_outputs.csv", pd.DataFrame(), self.paths.game_monte_carlo_outputs)

                preds = self._write_contract_csv(
                    "game_predictions_master.csv",
                    _apply_output_rounding("game_predictions_master.csv", pd.DataFrame()),
                    self.paths.game_predictions_master,
                )
                bets = self._write_contract_csv(
                    "bet_recommendations.csv",
                    _apply_output_rounding("bet_recommendations.csv", pd.DataFrame()),
                    self.paths.bet_recommendations,
                )
                watch = self._write_contract_csv("watchlist_games.csv", pd.DataFrame(), self.paths.watchlist_games)
                no_bet = self._write_contract_csv("no_bet_explanations.csv", pd.DataFrame(), self.paths.no_bet_explanations)

                as_of_pst = as_of.tz_convert("America/Los_Angeles")
                season = int(as_of_pst.year + 1) if as_of_pst.month >= 7 else int(as_of_pst.year)
                summary = pd.DataFrame(
                    [
                        {
                            "run_id": run_id,
                            "model_version": MODEL_VERSION,
                            "season": season,
                            "game_date_pst": as_of_pst.strftime("%Y-%m-%d"),
                            "games_on_card": 0,
                            "spread_bets_count": 0,
                            "total_bets_count": 0,
                            "watchlist_count": 0,
                            "highest_confidence_bet": np.nan,
                            "highest_edge_bet": np.nan,
                            "avg_spread_edge": np.nan,
                            "avg_total_edge": np.nan,
                            "avg_confidence": np.nan,
                            "notes": "no upcoming games in active horizon",
                            "created_at_utc": now_utc,
                        }
                    ]
                )
                summary = self._write_contract_csv("daily_card_summary.csv", summary, self.paths.daily_card_summary)

                agreement = summarize_agreement_buckets(hist_scored).rename(
                    columns={
                        "su_accuracy": "straight_up_win_pct",
                        "ats_accuracy": "ats_win_pct",
                        "avg_edge": "avg_spread_edge",
                        "confidence_mean": "avg_spread_confidence",
                    }
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
                recent_results = _build_recent_predictions_results_frame(
                    hist_scored,
                    as_of=as_of,
                    run_id=run_id,
                    now_utc=now_utc,
                )
                recent_results = self._write_contract_csv(
                    "recent_predictions_results.csv",
                    recent_results,
                    self.paths.recent_predictions_results,
                )

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
                    "recent_predictions_results": str(self.paths.recent_predictions_results),
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

                final_status = "WARN" if integrity.status == "WARN" else "PASS"
                if schema_issues:
                    final_status = "FAIL"

                manifest = {
                    "run_type": "predict",
                    "run_id": run_id,
                    "model_version": MODEL_VERSION,
                    "run_at_utc": now_utc,
                    "as_of_utc": as_of.isoformat(),
                    "mc_mode": mc_mode,
                    "hours_back": int(hours_back),
                    "status": final_status,
                    "stages": [asdict(s) for s in stages],
                    "outputs": outputs,
                }
                write_json(self.paths.run_manifest, manifest)
                outputs["run_manifest"] = str(self.paths.run_manifest)
                self._append_pipeline_log(run_id, stages)
                return RunResult(
                    ok=final_status in {"PASS", "WARN"},
                    status=final_status,
                    stages=stages,
                    outputs=outputs,
                    error=None if final_status in {"PASS", "WARN"} else "predict_failed",
                )

            upcoming = apply_base_strength(ds.upcoming_games)
            upcoming = apply_context_adjustments(upcoming)
            hist_scored = self._score_history_for_reports(ds.historical_games, mc_mode=mc_mode)
            rulebook = discover_situational_rules(hist_scored)
            upcoming = apply_situational_layer(upcoming, rulebook)
            upcoming = apply_monte_carlo_layer(upcoming, mode=mc_mode, n_sims=500)
            direct_model = fit_direct_win_model(hist_scored)
            upcoming = apply_decision_layer(upcoming, direct_win_model=direct_model, mc_mode=mc_mode)
            upcoming = apply_agreement_layer(upcoming)
            stages.append(
                StageRecord(
                    "Layer Stack",
                    "PASS",
                    [f"upset_layer_enabled={ENABLE_UPSET_LAYER}"],
                    {"rows": int(len(upcoming)), "upset_layer_enabled": int(ENABLE_UPSET_LAYER)},
                )
            )

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
                    "home_rest_form_signal_team_a": active.str.contains("home_rest_form_stack", na=False).astype(int),
                    "home_rest_form_signal_team_b": active.str.contains("away_rest_form_stack", na=False).astype(int),
                    "slow_dog_to_edge_signal_team_a": active.str.contains("home_pace_mismatch_fast_edge", na=False).astype(int),
                    "slow_dog_to_edge_signal_team_b": active.str.contains("away_pace_mismatch_fast_edge", na=False).astype(int),
                    "oreb_mismatch_signal_team_a": active.str.contains("home_oreb_edge_3pct", na=False).astype(int),
                    "oreb_mismatch_signal_team_b": active.str.contains("away_oreb_edge_3pct", na=False).astype(int),
                    "high_three_pt_dependence_warning_team_a": active.str.contains("fade_home_threept_vs_slow", na=False).astype(int),
                    "high_three_pt_dependence_warning_team_b": active.str.contains("fade_away_threept_vs_slow", na=False).astype(int),
                    "weak_sos_form_inflation_warning_team_a": active.str.contains("fade_home_weak_sos_form_pop", na=False).astype(int),
                    "weak_sos_form_inflation_warning_team_b": active.str.contains("fade_away_weak_sos_form_pop", na=False).astype(int),
                    "conference_tourney_bye_signal_team_a": active.str.contains("conference_bye_edge_home", na=False).astype(int),
                    "conference_tourney_bye_signal_team_b": active.str.contains("conference_bye_edge_away", na=False).astype(int),
                    "conference_tourney_fatigue_signal_team_a": active.str.contains("conference_fatigue_fade_home", na=False).astype(int),
                    "conference_tourney_fatigue_signal_team_b": active.str.contains("conference_fatigue_fade_away", na=False).astype(int),
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
                    "team_a_win_probability_mc": lambda d: _num(d.get("mc_home_win_prob")).clip(0.0, 1.0),
                    "team_b_win_probability_mc": lambda d: _num(d.get("mc_away_win_prob")).clip(0.0, 1.0),
                    "team_a_cover_probability_mc": lambda d: _num(d.get("mc_home_cover_prob")).clip(0.0, 1.0),
                    "team_b_cover_probability_mc": lambda d: _num(d.get("mc_away_cover_prob")).clip(0.0, 1.0),
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
            blowout = _compute_blowout_scores(upcoming)
            for col in blowout.columns:
                preds[col] = blowout[col]

            preds = _apply_spread_flip_logic(preds)
            preds = _apply_totals_flip_logic(preds, upcoming)
            preds["total_confidence"] = (_num(preds["total_edge_over"]).abs() * 8.0).clip(0.0, 100.0)
            preds["spread_model_pick"] = preds["spread_pick"]
            preds["total_model_pick"] = preds["total_pick"]

            default_spread_thresholds = self._default_spread_thresholds()
            spread_policy = load_backtest_calibration_policy(self.paths.backtest_calibration_policy)
            spread_thresholds = resolve_spread_thresholds(
                preds,
                policy=spread_policy,
                mc_mode=mc_mode,
                default_thresholds=default_spread_thresholds,
            )
            preds["spread_edge_threshold_used"] = _num(spread_thresholds.get("edge_min"))
            preds["spread_confidence_threshold_used"] = _num(spread_thresholds.get("confidence_min"))
            preds["spread_prob_edge_threshold_used"] = _num(spread_thresholds.get("ats_prob_edge_min"))
            preds["spread_calibration_scope"] = spread_thresholds.get("policy_scope", pd.Series("defaults", index=preds.index)).astype(str)
            preds["spread_calibration_variant"] = spread_thresholds.get("policy_variant", pd.Series("", index=preds.index)).astype(str)

            spread_prob_for_pick = np.where(
                preds["model_pick"].eq(preds["team_a"]),
                _num(preds["team_a_cover_probability"]),
                _num(preds["team_b_cover_probability"]),
            )
            spread_line_ok = _num(preds["market_spread"]).notna() & _num(preds["model_spread_team_a"]).notna()
            spread_edge_ok = _num(preds["spread_edge_for_pick"]) >= _num(preds["spread_edge_threshold_used"])
            spread_conf_ok = _num(preds["spread_confidence"]) >= _num(preds["spread_confidence_threshold_used"])
            spread_prob_ok = _num(pd.Series(spread_prob_for_pick, index=preds.index)).sub(0.5).abs() >= _num(preds["spread_prob_edge_threshold_used"])
            spread_recommend_mask = spread_line_ok & spread_edge_ok & spread_conf_ok & spread_prob_ok
            spread_mc_blocked = pd.Series(False, index=preds.index)
            if mc_mode == "confidence_filter" and "mc_filter_pass" in preds.columns:
                spread_mc_blocked = ~preds["mc_filter_pass"].astype(bool)
                spread_recommend_mask = spread_recommend_mask & preds["mc_filter_pass"].astype(bool)
            preds["spread_bet_flag"] = spread_recommend_mask.astype(bool)
            preds["spread_bet_recommended"] = preds["spread_bet_flag"].astype(bool)
            preds["spread_bet_reason"] = np.select(
                [
                    preds["spread_bet_flag"] & preds["model_pick_flipped"].astype(bool),
                    preds["spread_bet_flag"],
                    ~spread_line_ok,
                    spread_mc_blocked,
                    ~spread_edge_ok,
                    ~spread_conf_ok,
                    ~spread_prob_ok,
                ],
                [
                    "recommended:flip_gate+edge_confidence_probability",
                    "recommended:edge_confidence_probability",
                    "pass:missing_market_or_model_spread",
                    "pass:mc_filter_blocked",
                    "pass:edge_lt_calibrated_min",
                    "pass:confidence_lt_calibrated_min",
                    "pass:ats_prob_edge_lt_calibrated_min",
                ],
                default="pass:thresholds_not_met",
            )

            scope_counts = preds["spread_calibration_scope"].value_counts(dropna=False).to_dict()
            scope_note = ",".join([f"{str(k)}={int(v)}" for k, v in scope_counts.items()]) if scope_counts else "none"
            active_variant = preds["spread_calibration_variant"].replace("", np.nan).dropna().astype(str).mode()
            policy_branch = (spread_policy or {}).get("spread_thresholds_by_mc_mode", {}).get(mc_mode, {}) if isinstance(spread_policy, dict) else {}
            policy_status = "PASS" if spread_policy and isinstance(policy_branch, dict) and policy_branch else "WARN"
            policy_notes = [
                f"policy_file={self.paths.backtest_calibration_policy}",
                f"policy_loaded={int(spread_policy is not None)}",
                f"policy_generated_at_utc={'' if not spread_policy else spread_policy.get('generated_at_utc', '')}",
                f"policy_mc_mode={mc_mode}",
                f"policy_variant={(active_variant.iloc[0] if len(active_variant) else '')}",
                f"threshold_scope_counts={scope_note}",
                f"avg_edge_min_used={float(_num(preds['spread_edge_threshold_used']).mean()):.2f}",
                f"avg_confidence_min_used={float(_num(preds['spread_confidence_threshold_used']).mean()):.2f}",
                f"avg_prob_edge_min_used={float(_num(preds['spread_prob_edge_threshold_used']).mean()):.4f}",
            ]
            if spread_policy is None:
                policy_notes.append("Spread calibration policy missing; using static defaults.")
            elif not policy_branch:
                policy_notes.append("Spread calibration policy did not include current mc_mode; using static defaults.")
            stages.append(
                StageRecord(
                    "Backtest Calibration",
                    policy_status,
                    policy_notes,
                    {
                        "rows": int(len(preds)),
                        "policy_loaded": int(spread_policy is not None),
                        "avg_edge_min_used": float(_num(preds["spread_edge_threshold_used"]).mean()),
                        "avg_confidence_min_used": float(_num(preds["spread_confidence_threshold_used"]).mean()),
                        "avg_prob_edge_min_used": float(_num(preds["spread_prob_edge_threshold_used"]).mean()),
                    },
                )
            )

            total_pick_under = preds["total_pick"].astype(str).str.lower().eq("under")
            total_edge_for_pick = np.where(
                preds["total_pick"].astype(str).str.lower().eq("over"),
                _num(preds["total_edge_over"]),
                _num(preds["total_edge_under"]),
            )
            total_edge_for_pick_abs = _num(pd.Series(total_edge_for_pick, index=preds.index)).abs()
            total_line_ok = _num(preds["market_total"]).notna() & _num(preds["model_total"]).notna()
            total_auto_low_market = _num(preds["market_total"]) <= 130.0
            total_under_150_edge12 = total_pick_under & (_num(preds["market_total"]) <= 150.0) & (total_edge_for_pick_abs >= 12.0)
            total_edge12_any = total_edge_for_pick_abs >= 12.0
            total_recommend_mask = total_line_ok & (total_auto_low_market | total_under_150_edge12 | total_edge12_any)
            total_mc_blocked = pd.Series(False, index=preds.index)
            if mc_mode == "confidence_filter" and "mc_filter_pass" in preds.columns:
                total_mc_blocked = ~preds["mc_filter_pass"].astype(bool)
                total_recommend_mask = total_recommend_mask & preds["mc_filter_pass"].astype(bool)
            preds["total_bet_flag"] = total_recommend_mask.astype(bool)
            preds["total_bet_recommended"] = preds["total_bet_flag"].astype(bool)
            preds["total_bet_reason"] = np.select(
                [
                    preds["total_bet_flag"] & preds["total_pick_flipped"].astype(bool),
                    preds["total_bet_flag"] & total_auto_low_market,
                    preds["total_bet_flag"] & total_under_150_edge12,
                    preds["total_bet_flag"],
                    ~total_line_ok,
                    total_mc_blocked,
                    total_pick_under & (_num(preds["market_total"]) <= 150.0),
                ],
                [
                    "recommended:totals_flip_chaos_inverse",
                    "recommended:market_total_le_130_any_side",
                    "recommended:under_le_150_edge_ge_12",
                    "recommended:edge_ge_12",
                    "pass:missing_market_or_model_total",
                    "pass:mc_filter_blocked",
                    "pass:under_le_150_edge_lt_12",
                ],
                default="pass:edge_lt_12",
            )

            # Moneyline value layer: compare model win probability to market implied probability.
            implied_a_from_odds = _american_to_implied_prob(preds.get("market_moneyline_team_a"))
            implied_b_from_odds = _american_to_implied_prob(preds.get("market_moneyline_team_b"))
            implied_a_direct = _num(upcoming.get("market_home_win_prob"))
            implied_b_direct = _num(upcoming.get("market_away_win_prob"))
            preds["market_implied_prob_team_a"] = implied_a_from_odds.where(implied_a_from_odds.notna(), implied_a_direct)
            preds["market_implied_prob_team_b"] = implied_b_from_odds.where(implied_b_from_odds.notna(), implied_b_direct)
            preds["moneyline_edge_team_a"] = _num(preds["team_a_win_probability"]) - _num(preds["market_implied_prob_team_a"])
            preds["moneyline_edge_team_b"] = _num(preds["team_b_win_probability"]) - _num(preds["market_implied_prob_team_b"])
            preds["moneyline_pick"] = np.where(_num(preds["moneyline_edge_team_a"]) >= _num(preds["moneyline_edge_team_b"]), "team_a", "team_b")
            preds["moneyline_edge"] = np.where(
                preds["moneyline_pick"].eq("team_a"),
                _num(preds["moneyline_edge_team_a"]),
                _num(preds["moneyline_edge_team_b"]),
            )
            preds["moneyline_pick_prob"] = np.where(
                preds["moneyline_pick"].eq("team_a"),
                _num(preds["team_a_win_probability"]),
                _num(preds["team_b_win_probability"]),
            )
            preds["moneyline_market_line"] = np.where(
                preds["moneyline_pick"].eq("team_a"),
                _num(preds["market_moneyline_team_a"]),
                _num(preds["market_moneyline_team_b"]),
            )
            preds["moneyline_market_line"] = _num(preds["moneyline_market_line"]).where(
                _num(preds["moneyline_market_line"]).notna(),
                np.where(
                    preds["moneyline_pick"].eq("team_a"),
                    _prob_to_fair_american(preds["market_implied_prob_team_a"]),
                    _prob_to_fair_american(preds["market_implied_prob_team_b"]),
                ),
            )
            preds["moneyline_model_line"] = _prob_to_fair_american(preds["moneyline_pick_prob"])
            preds["moneyline_confidence"] = (
                35.0
                + (np.clip((_num(preds["moneyline_pick_prob"]) - 0.5).abs() / 0.5, 0.0, 1.0) * 35.0)
                + (np.clip(_num(preds["moneyline_edge"]), 0.0, 0.25) * 400.0)
            ).clip(0.0, 100.0)
            moneyline_recommend_mask = (
                (_num(preds["moneyline_edge"]) >= 0.03)
                & _num(preds["moneyline_market_line"]).notna()
                & _num(preds["moneyline_pick_prob"]).notna()
                & (_num(preds["moneyline_confidence"]) >= 58.0)
            )
            if mc_mode == "confidence_filter" and "mc_filter_pass" in preds.columns:
                moneyline_recommend_mask = moneyline_recommend_mask & preds["mc_filter_pass"].astype(bool)
            preds["moneyline_bet_flag"] = moneyline_recommend_mask.astype(bool)

            preds["final_bet_flag"] = preds["spread_bet_flag"] | preds["total_bet_flag"]
            preds["final_bet_flag"] = preds["final_bet_flag"] | preds["moneyline_bet_flag"]

            spread_bets = preds[preds["spread_bet_flag"]].copy()
            spread_bets["bet_type"] = "spread"
            spread_bets["bet_side"] = np.where(spread_bets["model_pick"] == spread_bets["team_a"], "team_a", "team_b")
            spread_bets["model_pick"] = spread_bets["model_pick"]
            spread_bets["model_pick_raw"] = spread_bets["model_pick_raw"]
            spread_bets["model_pick_flipped"] = spread_bets["model_pick_flipped"]
            spread_bets["model_pick_flip_reason"] = spread_bets["model_pick_flip_reason"]
            spread_bets["total_pick"] = spread_bets["total_pick"]
            spread_bets["total_pick_raw"] = spread_bets["total_pick_raw"]
            spread_bets["total_pick_flipped"] = spread_bets["total_pick_flipped"]
            spread_bets["total_pick_flip_reason"] = spread_bets["total_pick_flip_reason"]
            spread_bets["blowout_score_total"] = _num(spread_bets["blowout_score_total"])
            spread_bets["market_line"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["market_spread_team_a"], spread_bets["market_spread_team_b"])
            spread_bets["model_line"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["model_spread_team_a"], spread_bets["model_spread_team_b"])
            spread_bets["edge"] = _num(spread_bets["spread_edge_for_pick"])
            spread_bets["win_probability"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["team_a_win_probability"], spread_bets["team_b_win_probability"])
            spread_bets["cover_probability"] = np.where(spread_bets["bet_side"].eq("team_a"), spread_bets["team_a_cover_probability"], spread_bets["team_b_cover_probability"])
            spread_bets["mc_win_probability"] = np.where(
                spread_bets["bet_side"].eq("team_a"),
                spread_bets.get("team_a_win_probability_mc"),
                spread_bets.get("team_b_win_probability_mc"),
            )
            spread_bets["mc_cover_probability"] = np.where(
                spread_bets["bet_side"].eq("team_a"),
                spread_bets.get("team_a_cover_probability_mc"),
                spread_bets.get("team_b_cover_probability_mc"),
            )
            spread_bets["confidence"] = spread_bets["spread_confidence"]
            spread_bets["bet_reason_short"] = spread_bets["spread_bet_reason"].fillna("recommended:spread")

            total_bets = preds[preds["total_bet_flag"]].copy()
            total_bets["bet_type"] = "total"
            total_bets["bet_side"] = total_bets["total_pick"].str.lower()
            total_bets["model_pick"] = total_bets["total_pick"].str.lower()
            total_bets["model_pick_raw"] = total_bets["total_pick_raw"].str.lower()
            total_bets["model_pick_flipped"] = total_bets["total_pick_flipped"].astype(bool)
            total_bets["model_pick_flip_reason"] = total_bets["total_pick_flip_reason"]
            total_bets["total_pick_raw"] = total_bets["total_pick_raw"].str.lower()
            total_bets["total_pick_flipped"] = total_bets["total_pick_flipped"].astype(bool)
            total_bets["total_pick_flip_reason"] = total_bets["total_pick_flip_reason"]
            total_bets["blowout_score_total"] = _num(total_bets["blowout_score_total"])
            total_bets["market_line"] = _num(total_bets["market_total"])
            total_bets["model_line"] = _num(total_bets["model_total"])
            total_edge_pick = pd.Series(
                np.where(total_bets["bet_side"].eq("over"), _num(total_bets["total_edge_over"]), _num(total_bets["total_edge_under"])),
                index=total_bets.index,
            )
            total_bets["edge"] = _num(total_edge_pick).abs()
            total_bets["win_probability"] = np.where(total_bets["bet_side"].eq("over"), _num(total_bets["over_probability"]), _num(total_bets["under_probability"]))
            total_bets["cover_probability"] = np.nan
            total_bets["mc_win_probability"] = np.nan
            total_bets["mc_cover_probability"] = np.nan
            total_bets["confidence"] = _num(total_bets["total_confidence"])
            total_bets["bet_reason_short"] = total_bets["total_bet_reason"].fillna("recommended:total")

            moneyline_bets = preds[preds["moneyline_bet_flag"]].copy()
            moneyline_bets["bet_type"] = "moneyline"
            moneyline_bets["bet_side"] = moneyline_bets["moneyline_pick"]
            moneyline_bets["model_pick"] = np.where(moneyline_bets["moneyline_pick"].eq("team_a"), moneyline_bets["team_a"], moneyline_bets["team_b"])
            moneyline_bets["model_pick_raw"] = moneyline_bets["model_pick"]
            moneyline_bets["model_pick_flipped"] = False
            moneyline_bets["model_pick_flip_reason"] = ""
            moneyline_bets["total_pick"] = moneyline_bets["total_pick"]
            moneyline_bets["total_pick_raw"] = moneyline_bets["total_pick_raw"]
            moneyline_bets["total_pick_flipped"] = moneyline_bets["total_pick_flipped"]
            moneyline_bets["total_pick_flip_reason"] = moneyline_bets["total_pick_flip_reason"]
            moneyline_bets["blowout_score_total"] = _num(moneyline_bets["blowout_score_total"])
            moneyline_bets["market_line"] = _num(moneyline_bets["moneyline_market_line"]).round(0)
            moneyline_bets["model_line"] = _num(moneyline_bets["moneyline_model_line"]).round(0)
            moneyline_bets["edge"] = _num(moneyline_bets["moneyline_edge"])
            moneyline_bets["win_probability"] = _num(moneyline_bets["moneyline_pick_prob"])
            moneyline_bets["cover_probability"] = np.nan
            moneyline_bets["mc_win_probability"] = np.where(
                moneyline_bets["bet_side"].eq("team_a"),
                moneyline_bets.get("team_a_win_probability_mc"),
                moneyline_bets.get("team_b_win_probability_mc"),
            )
            moneyline_bets["mc_cover_probability"] = np.where(
                moneyline_bets["bet_side"].eq("team_a"),
                moneyline_bets.get("team_a_cover_probability_mc"),
                moneyline_bets.get("team_b_cover_probability_mc"),
            )
            moneyline_bets["confidence"] = _num(moneyline_bets["moneyline_confidence"])
            moneyline_bets["bet_reason_short"] = "moneyline value edge"

            bets = pd.concat([spread_bets, total_bets, moneyline_bets], ignore_index=True, sort=False)
            bets["situational_support"] = bets["agreement_bucket"].astype(str).str.contains("Situational", na=False)
            bets["monte_carlo_support"] = bets["agreement_bucket"].astype(str).str.contains("Monte Carlo", na=False)
            bets["recommended_stake_units"] = ((_num(bets["confidence"]) - 50.0) / 20.0).clip(0.0, 3.0).round(2)
            bets = bets.sort_values(["confidence", "edge"], ascending=[False, False], kind="mergesort")
            bets["bet_rank"] = np.arange(1, len(bets) + 1)
            bets["created_at_utc"] = now_utc

            preds = self._write_contract_csv(
                "game_predictions_master.csv",
                _apply_output_rounding("game_predictions_master.csv", preds),
                self.paths.game_predictions_master,
            )
            bets = self._write_contract_csv(
                "bet_recommendations.csv",
                _apply_output_rounding("bet_recommendations.csv", bets),
                self.paths.bet_recommendations,
            )

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
                        "notes": (
                            "ides run summary"
                            + f" | moneyline_bets={int((bets.get('bet_type', pd.Series(dtype=str)) == 'moneyline').sum()) if len(bets) else 0}"
                        ),
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
            recent_results = _build_recent_predictions_results_frame(
                hist_scored,
                as_of=as_of,
                run_id=run_id,
                now_utc=now_utc,
            )
            recent_results = self._write_contract_csv(
                "recent_predictions_results.csv",
                recent_results,
                self.paths.recent_predictions_results,
            )

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
                "recent_predictions_results": str(self.paths.recent_predictions_results),
                "agreement_analysis_results": str(self.paths.agreement_analysis_results),
            }
            if self.paths.backtest_calibration_policy.exists():
                outputs["backtest_calibration_policy"] = str(self.paths.backtest_calibration_policy)

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
                "hours_back": int(hours_back),
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
        require_wagertalk: bool = True,
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

            wagertalk_rows = int(ds.audit.get("inputs", {}).get("wagertalk_historical_odds_rows", 0))
            historical_odds_source = hist.get("historical_odds_source", pd.Series(np.nan, index=hist.index))
            if not isinstance(historical_odds_source, pd.Series):
                historical_odds_source = pd.Series(historical_odds_source, index=hist.index)
            wagertalk_mask = historical_odds_source.fillna("").astype(str).eq("wagertalk_historical_odds")
            wagertalk_matches = int(wagertalk_mask.sum())
            wagertalk_share = float(wagertalk_matches / len(hist)) if len(hist) else 0.0
            spread_coverage = float(_num(hist.get("market_spread")).notna().mean()) if len(hist) else 0.0
            totals_coverage = float(_num(hist.get("market_total")).notna().mean()) if len(hist) else 0.0

            wagertalk_notes = [
                f"wagertalk_rows_available={wagertalk_rows}",
                f"wagertalk_matches_in_training={wagertalk_matches}",
                f"wagertalk_match_share={wagertalk_share:.4f}",
                f"historical_spread_coverage={spread_coverage:.4f}",
                f"historical_total_coverage={totals_coverage:.4f}",
                f"require_wagertalk={int(bool(require_wagertalk))}",
            ]
            wagertalk_status = "PASS"
            wagertalk_error: str | None = None
            if require_wagertalk and wagertalk_rows <= 0:
                wagertalk_status = "BLOCKED"
                wagertalk_error = "wagertalk_historical_odds_missing"
                wagertalk_notes.append("WagerTalk gate failed: data/wagertalk_historical_odds.csv is missing or empty.")
            elif require_wagertalk and wagertalk_matches <= 0:
                wagertalk_status = "BLOCKED"
                wagertalk_error = "wagertalk_historical_matches_zero"
                wagertalk_notes.append("WagerTalk gate failed: no WagerTalk historical odds rows matched training games.")
            stages.append(
                StageRecord(
                    "Market Lines Steward",
                    wagertalk_status,
                    wagertalk_notes,
                    {
                        "rows": int(len(hist)),
                        "wagertalk_rows_available": wagertalk_rows,
                        "wagertalk_matches": wagertalk_matches,
                        "wagertalk_match_share": wagertalk_share,
                        "historical_spread_coverage": spread_coverage,
                        "historical_total_coverage": totals_coverage,
                        "require_wagertalk": int(bool(require_wagertalk)),
                    },
                )
            )
            if wagertalk_error is not None:
                self._append_pipeline_log(run_id, stages, wagertalk_error)
                return RunResult(ok=False, status="BLOCKED", stages=stages, outputs=outputs, error=wagertalk_error)

            artifacts = run_variant_backtest(hist)
            scorecard = artifacts.scorecard.copy()

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
            backtest["roi_spread"] = scorecard.get("roi_spread")
            backtest["roi_total"] = scorecard.get("roi_total")
            backtest["notes"] = (
                "ides variant backtest"
                + " | ml_win_pct="
                + scorecard.get("moneyline_win_pct_all", pd.Series(np.nan, index=scorecard.index)).round(4).astype(str)
                + " | ml_roi="
                + scorecard.get("roi_moneyline", pd.Series(np.nan, index=scorecard.index)).round(2).astype(str)
                + " | ml_proxy_rate="
                + scorecard.get("moneyline_proxy_usage_rate", pd.Series(np.nan, index=scorecard.index)).round(4).astype(str)
                + " | five_u_annualized="
                + scorecard.get("five_unit_annualized", pd.Series(np.nan, index=scorecard.index)).round(2).astype(str)
                + f" | wt_matches={wagertalk_matches}"
                + f" | wt_share={wagertalk_share:.4f}"
            )
            backtest["created_at_utc"] = now_utc
            backtest = self._write_contract_csv("backtest_model_summary.csv", backtest, self.paths.backtest_model_summary)
            outputs["backtest_model_summary"] = str(self.paths.backtest_model_summary)

            edge_summary = artifacts.edge_band_summary.copy()
            edge_summary["run_id"] = run_id
            edge_summary["model_version"] = MODEL_VERSION
            edge_summary["variant_name"] = edge_summary.get("variant_id")
            edge_summary["created_at_utc"] = now_utc
            edge_summary = self._write_contract_csv("backtest_edge_band_summary.csv", edge_summary, self.paths.backtest_edge_band_summary)
            outputs["backtest_edge_band_summary"] = str(self.paths.backtest_edge_band_summary)

            ledger = artifacts.bet_ledger.copy()
            dt = pd.to_datetime(ledger.get("game_datetime_utc"), utc=True, errors="coerce")
            pst = dt.dt.tz_convert("America/Los_Angeles")
            ledger["run_id"] = run_id
            ledger["model_version"] = MODEL_VERSION
            ledger["variant_name"] = ledger.get("variant_id")
            ledger["game_start_datetime_utc"] = np.where(dt.notna(), dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ"), np.nan)
            ledger["game_date_pst"] = np.where(pst.notna(), pst.dt.strftime("%Y-%m-%d"), np.nan)
            ledger["season"] = np.where(pst.notna(), np.where(pst.dt.month >= 7, pst.dt.year + 1, pst.dt.year), np.nan)
            ledger["created_at_utc"] = now_utc
            ledger = self._write_contract_csv("backtest_bet_ledger.csv", ledger, self.paths.backtest_bet_ledger)
            outputs["backtest_bet_ledger"] = str(self.paths.backtest_bet_ledger)

            kelly = artifacts.kelly_summary.copy()
            kelly["run_id"] = run_id
            kelly["model_version"] = MODEL_VERSION
            kelly["variant_name"] = kelly.get("variant_id")
            kelly["created_at_utc"] = now_utc
            kelly = self._write_contract_csv("backtest_kelly_summary.csv", kelly, self.paths.backtest_kelly_summary)
            outputs["backtest_kelly_summary"] = str(self.paths.backtest_kelly_summary)

            calibration_policy = build_backtest_calibration_policy(
                scorecard,
                artifacts.bet_ledger,
                generated_at_utc=now_utc,
                default_thresholds=self._default_spread_thresholds(),
            )
            write_json(self.paths.backtest_calibration_policy, calibration_policy)
            outputs["backtest_calibration_policy"] = str(self.paths.backtest_calibration_policy)

            schema = validate_backtest_summary(backtest)
            issues = [] if schema.ok else [f"backtest_model_summary missing columns: {schema.missing_columns}"]
            ml_proxy_count = int((ledger.get("market_type", pd.Series(dtype=str)).eq("moneyline") & ledger.get("odds_source", pd.Series(dtype=str)).eq("spread_proxy")).sum())
            ml_rows = int(ledger.get("market_type", pd.Series(dtype=str)).eq("moneyline").sum())
            ml_proxy_rate = (ml_proxy_count / ml_rows) if ml_rows > 0 else 0.0
            ats_rows = int(ledger.get("market_type", pd.Series(dtype=str)).eq("ats").sum())
            total_rows = int(ledger.get("market_type", pd.Series(dtype=str)).eq("total").sum())
            max_five_u_annualized = float(pd.to_numeric(scorecard.get("five_unit_annualized"), errors="coerce").max()) if len(scorecard) else 0.0
            graded_band_counts = (
                edge_summary.groupby(["market_type", "edge_band"], observed=False)["bets_graded"].sum().reset_index()
                if not edge_summary.empty
                else pd.DataFrame(columns=["market_type", "edge_band", "bets_graded"])
            )
            graded_by_band_note = ",".join(
                [
                    f"{str(r.market_type)}:{str(r.edge_band)}="
                    f"{int(0 if pd.isna(pd.to_numeric(r.bets_graded, errors='coerce')) else pd.to_numeric(r.bets_graded, errors='coerce'))}"
                    for r in graded_band_counts.itertuples(index=False)
                ]
            )
            five_by_variant_note = ",".join(
                [
                    f"{str(r.variant_id)}={float(pd.to_numeric(r.five_unit_annualized, errors='coerce')):.2f}"
                    for r in scorecard[["variant_id", "five_unit_annualized"]].itertuples(index=False)
                ]
            ) if {"variant_id", "five_unit_annualized"}.issubset(set(scorecard.columns)) else ""
            eval_notes = [
                f"moneyline_proxy_usage_count={ml_proxy_count}",
                f"moneyline_proxy_usage_rate={ml_proxy_rate:.4f}",
                f"graded_samples=ats:{ats_rows},moneyline:{ml_rows},totals:{total_rows}",
                f"graded_samples_by_band={graded_by_band_note}",
                f"five_unit_annualized_max={max_five_u_annualized:.2f}",
                f"five_unit_annualized_by_variant={five_by_variant_note}",
                f"wagertalk_rows_available={wagertalk_rows}",
                f"wagertalk_matches_in_training={wagertalk_matches}",
                f"wagertalk_match_share={wagertalk_share:.4f}",
                f"historical_spread_coverage={spread_coverage:.4f}",
                f"historical_total_coverage={totals_coverage:.4f}",
                f"calibration_policy_path={self.paths.backtest_calibration_policy}",
                "calibration_policy_mode=spread_thresholds",
            ]
            stages.append(
                StageRecord(
                    "Evaluation Agent",
                    "PASS" if not issues else "FAIL",
                    issues + eval_notes,
                    {
                        "rows": int(len(backtest)),
                        "edge_band_rows": int(len(edge_summary)),
                        "ledger_rows": int(len(ledger)),
                        "kelly_rows": int(len(kelly)),
                        "moneyline_proxy_usage_count": ml_proxy_count,
                        "moneyline_proxy_usage_rate": ml_proxy_rate,
                        "ats_rows": ats_rows,
                        "moneyline_rows": ml_rows,
                        "total_rows": total_rows,
                    },
                )
            )

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
