from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_MODEL_NAMES, ModelLabConfig, canonicalize_game_id, derive_season_id


STANDARD_COLUMNS = [
    "season_id",
    "game_id",
    "event_id",
    "game_datetime_utc",
    "game_date",
    "home_team_id",
    "away_team_id",
    "neutral_site",
    "actual_margin",
    "actual_total",
    "home_won",
    "spread_line",
    "total_line",
    "home_ml",
    "away_ml",
    "pred_spread",
    "pred_total",
    "pred_conf",
    "model_name",
    "source_file",
]


def _empty_standard() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[STANDARD_COLUMNS]


def _coalesce(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.Series:
    if target in df.columns:
        base = df[target]
    else:
        base = pd.Series(pd.NA, index=df.index)

    for col in candidates:
        if col in df.columns:
            base = base.where(base.notna(), df[col])
    return base


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_standard()

    out = df.copy()
    out["event_id"] = _coalesce(out, "event_id", ["game_id"]).map(canonicalize_game_id)
    out["game_id"] = _coalesce(out, "game_id", ["event_id"]).map(canonicalize_game_id)
    out["game_id"] = out["game_id"].where(out["game_id"].astype(str) != "", out["event_id"])

    out["game_datetime_utc"] = pd.to_datetime(
        _coalesce(out, "game_datetime_utc", ["game_datetime"]),
        utc=True,
        errors="coerce",
    )
    out["season_id"] = derive_season_id(out["game_datetime_utc"], out.get("season_id"))
    out["game_date"] = _coalesce(out, "game_date", ["date"])
    out["neutral_site"] = pd.to_numeric(_coalesce(out, "neutral_site", []), errors="coerce").fillna(0).astype(int)

    out["actual_margin"] = pd.to_numeric(_coalesce(out, "actual_margin", []), errors="coerce")
    out["actual_total"] = pd.to_numeric(_coalesce(out, "actual_total", []), errors="coerce")
    out["home_won"] = pd.to_numeric(_coalesce(out, "home_won", []), errors="coerce")

    if out["home_won"].isna().all() and out["actual_margin"].notna().any():
        out["home_won"] = (out["actual_margin"] > 0).astype("Int64")

    out["spread_line"] = pd.to_numeric(
        _coalesce(out, "spread_line", ["market_spread", "home_market_spread"]),
        errors="coerce",
    )
    out["total_line"] = pd.to_numeric(
        _coalesce(out, "total_line", ["market_total", "over_under"]),
        errors="coerce",
    )
    out["home_ml"] = pd.to_numeric(_coalesce(out, "home_ml", []), errors="coerce")
    out["away_ml"] = pd.to_numeric(_coalesce(out, "away_ml", []), errors="coerce")

    out["pred_spread"] = pd.to_numeric(_coalesce(out, "pred_spread", []), errors="coerce")
    out["pred_total"] = pd.to_numeric(_coalesce(out, "pred_total", []), errors="coerce")
    out["pred_conf"] = pd.to_numeric(_coalesce(out, "pred_conf", []), errors="coerce")

    return _ensure_standard_columns(out)


@dataclass
class ModelPredictionAdapter:
    model_name: str

    def load_predictions(self) -> pd.DataFrame:
        raise NotImplementedError


class BacktestAdapter(ModelPredictionAdapter):
    COLUMN_MAP = {
        "FourFactors": ("fourfactors_spread", "fourfactors_total", "fourfactors_conf"),
        "AdjEfficiency": ("adjefficiency_spread", "adjefficiency_total", "adjefficiency_conf"),
        "Pythagorean": ("pythagorean_spread", "pythagorean_total", "pythagorean_conf"),
        "Situational": ("situational_spread", "situational_total", "situational_conf"),
        "CAGERankings": ("cagerankings_spread", "cagerankings_total", "cagerankings_conf"),
        "LuckRegression": ("luckregression_spread", "luckregression_total", "luckregression_conf"),
        "Variance": ("variance_spread", "variance_total", "variance_conf"),
        "HomeAwayForm": ("homeawayform_spread", "homeawayform_total", "homeawayform_conf"),
    }

    def __init__(self, model_name: str, path: Path):
        super().__init__(model_name)
        self.path = path

    def load_predictions(self) -> pd.DataFrame:
        if self.model_name not in self.COLUMN_MAP:
            return _empty_standard()
        if not self.path.exists() or self.path.stat().st_size == 0:
            return _empty_standard()

        df = pd.read_csv(self.path, low_memory=False)
        spread_col, total_col, conf_col = self.COLUMN_MAP[self.model_name]
        if spread_col not in df.columns:
            return _empty_standard()

        out = pd.DataFrame()
        out["event_id"] = _coalesce(df, "event_id", ["game_id"])
        out["game_id"] = _coalesce(df, "game_id", ["event_id"])
        out["game_datetime_utc"] = _coalesce(df, "game_datetime_utc", ["game_datetime"])
        out["game_date"] = _coalesce(df, "game_date", ["date"])
        out["home_team_id"] = _coalesce(df, "home_team_id", [])
        out["away_team_id"] = _coalesce(df, "away_team_id", [])
        out["neutral_site"] = _coalesce(df, "neutral_site", [])

        out["actual_margin"] = _coalesce(df, "actual_margin", [])
        out["actual_total"] = _coalesce(df, "actual_total", [])
        out["home_won"] = _coalesce(df, "home_won", [])

        out["spread_line"] = _coalesce(df, "spread_line", ["market_spread", "home_market_spread"])
        out["total_line"] = _coalesce(df, "total_line", ["market_total"])
        out["home_ml"] = _coalesce(df, "home_ml", [])
        out["away_ml"] = _coalesce(df, "away_ml", [])

        out["pred_spread"] = df[spread_col]
        out["pred_total"] = df[total_col] if total_col in df.columns else pd.NA
        out["pred_conf"] = df[conf_col] if conf_col in df.columns else pd.NA
        out["model_name"] = self.model_name
        out["source_file"] = str(self.path)
        return _normalize_frame(out)


class CombinedLiveAdapter(ModelPredictionAdapter):
    COLUMN_MAP = {
        "FourFactors": (["fourfactors_spread"], ["fourfactors_total"], ["fourfactors_conf"]),
        "AdjEfficiency": (["adjefficiency_spread"], ["adjefficiency_total"], ["adjefficiency_conf"]),
        "Pythagorean": (["pythagorean_spread"], ["pythagorean_total"], ["pythagorean_conf"]),
        "Situational": (["situational_spread"], ["situational_total"], ["situational_conf"]),
        "CAGERankings": (["cagerankings_spread"], ["cagerankings_total"], ["cagerankings_conf"]),
        "LuckRegression": (["luckregression_spread", "momentum_spread"], ["luckregression_total", "momentum_total"], ["luckregression_conf", "momentum_conf"]),
        "Variance": (["variance_spread", "atsintelligence_spread"], ["variance_total", "atsintelligence_total"], ["variance_conf", "atsintelligence_conf"]),
        "HomeAwayForm": (["homeawayform_spread", "regressedeff_spread"], ["homeawayform_total", "regressedeff_total"], ["homeawayform_conf", "regressedeff_conf"]),
    }

    def __init__(self, model_name: str, path: Path):
        super().__init__(model_name)
        self.path = path

    def load_predictions(self) -> pd.DataFrame:
        if self.model_name not in self.COLUMN_MAP:
            return _empty_standard()
        if not self.path.exists() or self.path.stat().st_size == 0:
            return _empty_standard()

        df = pd.read_csv(self.path, low_memory=False)
        spread_cols, total_cols, conf_cols = self.COLUMN_MAP[self.model_name]

        spread = _coalesce(df, "pred_spread", spread_cols)
        if pd.to_numeric(spread, errors="coerce").notna().sum() == 0:
            return _empty_standard()

        out = pd.DataFrame()
        out["event_id"] = _coalesce(df, "event_id", ["game_id"])
        out["game_id"] = _coalesce(df, "game_id", ["event_id"])
        out["game_datetime_utc"] = _coalesce(df, "game_datetime_utc", ["game_datetime"])
        out["game_date"] = _coalesce(df, "game_date", ["date"])
        out["home_team_id"] = _coalesce(df, "home_team_id", [])
        out["away_team_id"] = _coalesce(df, "away_team_id", [])
        out["neutral_site"] = _coalesce(df, "neutral_site", [])

        out["actual_margin"] = _coalesce(df, "actual_margin", [])
        out["actual_total"] = _coalesce(df, "actual_total", [])
        out["home_won"] = _coalesce(df, "home_won", [])

        out["spread_line"] = _coalesce(df, "spread_line", ["market_spread"])
        out["total_line"] = _coalesce(df, "total_line", ["market_total", "over_under"])
        out["home_ml"] = _coalesce(df, "home_ml", [])
        out["away_ml"] = _coalesce(df, "away_ml", [])

        out["pred_spread"] = spread
        out["pred_total"] = _coalesce(df, "pred_total", total_cols)
        out["pred_conf"] = _coalesce(df, "pred_conf", conf_cols)
        out["model_name"] = self.model_name
        out["source_file"] = str(self.path)
        return _normalize_frame(out)


def load_predictions(config: ModelLabConfig, model_name: str) -> pd.DataFrame:
    backtest_path = config.repo_root / config.backtest_results_path
    combined_path = config.repo_root / config.predictions_combined_path

    adapters = [
        BacktestAdapter(model_name=model_name, path=backtest_path),
        CombinedLiveAdapter(model_name=model_name, path=combined_path),
    ]

    for adapter in adapters:
        df = adapter.load_predictions()
        if not df.empty:
            return df
    return _empty_standard()


def load_all_available_predictions(
    config: ModelLabConfig,
    model_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    names = model_names or DEFAULT_MODEL_NAMES
    output: dict[str, pd.DataFrame] = {}
    for model_name in names:
        df = load_predictions(config, model_name)
        if not df.empty:
            output[model_name] = df
    return output
