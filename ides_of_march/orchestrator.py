from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, IDES_DIR, output_paths
from .data_steward import DataStewardResult, build_data_steward_frame
from .evaluation import run_variant_backtest
from .layer1_base_strength import apply_base_strength
from .layer2_context import apply_context_adjustments
from .layer3_situational import apply_situational_layer, discover_situational_rules
from .layer4_monte_carlo import apply_monte_carlo_layer
from .layer5_agreement import apply_agreement_layer, summarize_agreement_buckets
from .layer6_decision import build_bet_recs_frame, apply_decision_layer, fit_direct_win_model
from .safety import SafetyStatus, run_data_integrity_audit, run_model_safety_audit
from .schemas import (
    validate_agreement,
    validate_bet_recs,
    validate_predictions,
    validate_rulebook,
    validate_variant_scorecard,
)
from .utils import ensure_dir, pretty_exception, utc_now_iso, write_json


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


class IDESOrchestrator:
    def __init__(self, *, data_dir: Path = DATA_DIR, output_dir: Path = IDES_DIR) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.paths = output_paths(output_dir)

    def audit(self, *, as_of: pd.Timestamp, hours_ahead: int = 48) -> RunResult:
        stages: list[StageRecord] = []
        try:
            ds = build_data_steward_frame(data_dir=self.data_dir, as_of=as_of, hours_ahead=hours_ahead)
            integrity = run_data_integrity_audit(ds.upcoming_games, ds.historical_games)
            stages.append(
                StageRecord(
                    agent="Data Integrity Auditor",
                    status=integrity.status,
                    notes=integrity.issues,
                    metrics=integrity.metrics,
                )
            )
            ok = integrity.status in {"PASS", "WARN"}
            return RunResult(ok=ok, status=integrity.status, stages=stages, outputs={}, error=None)
        except Exception as exc:
            return RunResult(ok=False, status="FAIL", stages=stages, outputs={}, error=pretty_exception(exc))

    def _score_history_for_reports(
        self,
        historical: pd.DataFrame,
        *,
        mc_mode: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if historical.empty:
            empty_rules = discover_situational_rules(historical)
            return historical, empty_rules

        hist = apply_base_strength(historical)
        hist = apply_context_adjustments(hist)
        rulebook = discover_situational_rules(hist)
        hist = apply_situational_layer(hist, rulebook)
        hist = apply_monte_carlo_layer(hist, mode=mc_mode, n_sims=250)
        direct_model = fit_direct_win_model(hist)
        hist = apply_decision_layer(hist, direct_win_model=direct_model, mc_mode=mc_mode)
        hist = apply_agreement_layer(hist)
        return hist, rulebook

    def predict(
        self,
        *,
        as_of: pd.Timestamp,
        mc_mode: str = "confidence_only",
        hours_ahead: int = 48,
    ) -> RunResult:
        stages: list[StageRecord] = []
        outputs: dict[str, str] = {}

        try:
            ds = build_data_steward_frame(data_dir=self.data_dir, as_of=as_of, hours_ahead=hours_ahead)
            integrity = run_data_integrity_audit(ds.upcoming_games, ds.historical_games)
            stages.append(StageRecord("Data Integrity Auditor", integrity.status, integrity.issues, integrity.metrics))
            if integrity.status in {"FAIL", "BLOCKED"}:
                return RunResult(ok=False, status=integrity.status, stages=stages, outputs=outputs, error="integrity_gate_failed")

            upcoming = apply_base_strength(ds.upcoming_games)
            upcoming = apply_context_adjustments(upcoming)
            stages.append(StageRecord("Base Model Agent", "PASS", [], {"rows": int(len(upcoming))}))

            historical_scored, rulebook = self._score_history_for_reports(ds.historical_games, mc_mode=mc_mode)
            upcoming = apply_situational_layer(upcoming, rulebook)
            stages.append(
                StageRecord(
                    "Situational Research Agent",
                    "PASS",
                    [],
                    {
                        "rules_total": int(len(rulebook)),
                        "rules_accepted": int(rulebook.get("accepted", pd.Series(dtype=bool)).sum()),
                    },
                )
            )

            upcoming = apply_monte_carlo_layer(upcoming, mode=mc_mode, n_sims=500)
            stages.append(StageRecord("Monte Carlo Agent", "PASS", [], {"mc_mode": mc_mode, "rows": int(len(upcoming))}))

            direct_model = fit_direct_win_model(historical_scored)
            upcoming = apply_decision_layer(upcoming, direct_win_model=direct_model, mc_mode=mc_mode)
            upcoming = apply_agreement_layer(upcoming)
            stages.append(StageRecord("Agreement Analysis Agent", "PASS", [], {"rows": int(len(upcoming))}))

            safety = run_model_safety_audit(upcoming)
            stages.append(StageRecord("Model Safety Auditor", safety.status, safety.issues, safety.metrics))

            prediction_cols = [
                "game_id",
                "event_id",
                "game_datetime_utc",
                "home_team",
                "away_team",
                "home_team_id",
                "away_team_id",
                "projected_spread",
                "market_spread",
                "projected_margin_home",
                "projected_total_ctx",
                "market_total",
                "edge_home",
                "win_prob_home",
                "ats_cover_prob_home",
                "confidence_score",
                "agreement_bucket",
                "bet_recommendation",
                "line_source_used",
                "context_summary",
                "situational_active_rules",
                "mc_volatility",
                "mc_home_win_prob",
                "mc_home_cover_prob",
                "model_prob_source",
                "mc_mode",
                "active_as_of_utc",
            ]
            predictions = upcoming.reindex(columns=prediction_cols).copy()

            bet_recs = build_bet_recs_frame(predictions)
            agreement_report = summarize_agreement_buckets(historical_scored)

            ensure_dir(self.paths.predictions_latest)
            predictions.to_csv(self.paths.predictions_latest, index=False)
            bet_recs.to_csv(self.paths.bet_recs, index=False)
            agreement_report.to_csv(self.paths.agreement_bucket_report, index=False)
            rulebook.to_csv(self.paths.situational_rulebook, index=False)

            outputs = {
                "predictions_latest": str(self.paths.predictions_latest),
                "bet_recs": str(self.paths.bet_recs),
                "agreement_bucket_report": str(self.paths.agreement_bucket_report),
                "situational_rulebook": str(self.paths.situational_rulebook),
            }

            pred_schema = validate_predictions(predictions)
            bet_schema = validate_bet_recs(bet_recs)
            agr_schema = validate_agreement(agreement_report)
            rule_schema = validate_rulebook(rulebook)

            schema_issues = []
            if not pred_schema.ok:
                schema_issues.append(f"predictions missing columns: {pred_schema.missing_columns}")
            if not bet_schema.ok:
                schema_issues.append(f"bet_recs missing columns: {bet_schema.missing_columns}")
            if not agr_schema.ok:
                schema_issues.append(f"agreement report missing columns: {agr_schema.missing_columns}")
            if not rule_schema.ok:
                schema_issues.append(f"rulebook missing columns: {rule_schema.missing_columns}")

            final_status = "PASS"
            if safety.status == "FAIL" or schema_issues:
                final_status = "FAIL"
            elif safety.status == "WARN" or integrity.status == "WARN":
                final_status = "WARN"

            if schema_issues:
                stages.append(StageRecord("Output Schema Contract", "FAIL", schema_issues, {}))
            else:
                stages.append(StageRecord("Output Schema Contract", "PASS", [], {}))

            manifest = {
                "run_type": "predict",
                "run_at_utc": utc_now_iso(),
                "as_of_utc": as_of.isoformat(),
                "mc_mode": mc_mode,
                "status": final_status,
                "stages": [asdict(s) for s in stages],
                "outputs": outputs,
            }
            write_json(self.paths.run_manifest, manifest)
            outputs["run_manifest"] = str(self.paths.run_manifest)

            return RunResult(
                ok=final_status in {"PASS", "WARN"},
                status=final_status,
                stages=stages,
                outputs=outputs,
                error=None if final_status in {"PASS", "WARN"} else "predict_failed",
            )
        except Exception as exc:
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
                return RunResult(ok=False, status="BLOCKED", stages=stages, outputs=outputs, error="no_historical_games")

            scorecard = run_variant_backtest(hist)
            rulebook = discover_situational_rules(hist)

            ensure_dir(self.paths.backtest_variant_scorecard)
            scorecard.to_csv(self.paths.backtest_variant_scorecard, index=False)
            rulebook.to_csv(self.paths.situational_rulebook, index=False)

            outputs = {
                "backtest_variant_scorecard": str(self.paths.backtest_variant_scorecard),
                "situational_rulebook": str(self.paths.situational_rulebook),
            }

            variant_schema = validate_variant_scorecard(scorecard)
            rule_schema = validate_rulebook(rulebook)
            issues = []
            if not variant_schema.ok:
                issues.append(f"variant scorecard missing columns: {variant_schema.missing_columns}")
            if not rule_schema.ok:
                issues.append(f"rulebook missing columns: {rule_schema.missing_columns}")

            stages.append(
                StageRecord(
                    "Evaluation Agent",
                    "PASS" if not issues else "FAIL",
                    issues,
                    {
                        "variants": int(len(scorecard)),
                        "rows_historical": int(len(hist)),
                    },
                )
            )

            final_status = "PASS" if not issues else "FAIL"
            manifest = {
                "run_type": "backtest",
                "run_at_utc": utc_now_iso(),
                "as_of_utc": as_of.isoformat(),
                "start_date": start_date,
                "end_date": end_date,
                "status": final_status,
                "stages": [asdict(s) for s in stages],
                "outputs": outputs,
            }
            write_json(self.paths.run_manifest, manifest)
            outputs["run_manifest"] = str(self.paths.run_manifest)

            return RunResult(
                ok=final_status == "PASS",
                status=final_status,
                stages=stages,
                outputs=outputs,
                error=None if final_status == "PASS" else "backtest_failed",
            )
        except Exception as exc:
            return RunResult(ok=False, status="FAIL", stages=stages, outputs=outputs, error=pretty_exception(exc))
