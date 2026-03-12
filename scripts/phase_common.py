from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RepoPaths:
    rd2_root: Path
    super_root: Path
    rd2_data: Path
    super_data: Path
    warehouse_parquet: Path
    backtest_csv: Path
    layer_registry_csv: Path
    predictions_master_csv: Path


def resolve_paths() -> RepoPaths:
    rd2_root = Path(__file__).resolve().parents[1]
    super_root = rd2_root.parent
    rd2_data = rd2_root / "data"
    super_data = super_root / "data"

    warehouse_candidates = [
        super_data / "historical_warehouse.parquet",
        rd2_data / "historical_warehouse.parquet",
        super_root / "historical_warehouse.parquet",
    ]
    warehouse = next((p for p in warehouse_candidates if p.exists()), warehouse_candidates[0])

    return RepoPaths(
        rd2_root=rd2_root,
        super_root=super_root,
        rd2_data=rd2_data,
        super_data=super_data,
        warehouse_parquet=warehouse,
        backtest_csv=rd2_data / "backtest_results_latest.csv",
        layer_registry_csv=rd2_data / "layer_registry.csv",
        predictions_master_csv=rd2_data / "reports" / "game_predictions_master.csv",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def normalize_game_id(series: pd.Series) -> pd.Series:
    out = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    )
    return out


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float) != 0
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def derive_phase(game_datetime: pd.Series) -> pd.Series:
    dt = pd.to_datetime(game_datetime, errors="coerce", utc=True)
    month = dt.dt.month
    day = dt.dt.day
    out = pd.Series("regular_season", index=game_datetime.index, dtype="object")
    conf_tourney = month.eq(3) & day.le(14)
    ncaa = month.eq(3) & day.ge(15)
    out.loc[conf_tourney] = "conference_tournament"
    out.loc[ncaa] = "ncaa_tournament"
    return out


def _pick_first(frame: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = [c for c in cols if c in frame.columns]
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in cols:
        cand = to_num(frame[col])
        out = out.where(out.notna(), cand)
    return out


def build_side_rows_from_warehouse(warehouse: pd.DataFrame) -> pd.DataFrame:
    if warehouse.empty:
        return pd.DataFrame()

    work = warehouse.copy()
    if "game_id" not in work.columns:
        return pd.DataFrame()
    work["game_id"] = normalize_game_id(work["game_id"])
    work = work[work["game_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    base_phase = (
        work["phase"]
        if "phase" in work.columns
        else derive_phase(work.get("game_start_datetime_utc", pd.Series(index=work.index)))
    )
    is_tourney = (
        to_bool(work["is_tournament"])
        if "is_tournament" in work.columns
        else base_phase.astype(str).str.contains("tournament", case=False, na=False)
    )

    def _one_side(side: str) -> pd.DataFrame:
        team = "a" if side == "a" else "b"
        opp = "b" if team == "a" else "a"
        out = pd.DataFrame(index=work.index)
        out["game_id"] = work["game_id"]
        out["season"] = to_num(work.get("season", pd.Series(np.nan, index=work.index)))
        out["game_datetime_utc"] = work.get("game_start_datetime_utc", pd.Series(np.nan, index=work.index))
        out["phase"] = base_phase
        out["is_tournament"] = is_tourney
        out["is_ncaa_tournament"] = to_bool(work.get("is_ncaa_tournament", pd.Series(False, index=work.index)))
        out["is_conference_tournament"] = to_bool(work.get("is_conference_tournament", pd.Series(False, index=work.index)))
        out["tourney_round"] = work.get("tourney_round", work.get("round_name", pd.Series(np.nan, index=work.index)))
        out["team_name"] = (
            work.get(f"team_{team}", pd.Series(np.nan, index=work.index))
            .astype(str)
            .replace({"nan": np.nan, "None": np.nan})
        )
        out["opponent_name"] = (
            work.get(f"team_{opp}", pd.Series(np.nan, index=work.index))
            .astype(str)
            .replace({"nan": np.nan, "None": np.nan})
        )
        out["conference_team"] = work.get(f"team_{team}_conference", pd.Series(np.nan, index=work.index))
        out["conference_opponent"] = work.get(f"team_{opp}_conference", pd.Series(np.nan, index=work.index))
        out["seed_team"] = to_num(work.get(f"team_{team}_seed", pd.Series(np.nan, index=work.index)))
        out["seed_opponent"] = to_num(work.get(f"team_{opp}_seed", pd.Series(np.nan, index=work.index)))

        team_final = to_num(work.get(f"team_{team}_final_score", pd.Series(np.nan, index=work.index)))
        opp_final = to_num(work.get(f"team_{opp}_final_score", pd.Series(np.nan, index=work.index)))
        margin = team_final - opp_final
        if "final_margin" in work.columns and team == "a":
            margin = to_num(work["final_margin"])
        elif "final_margin" in work.columns and team == "b":
            margin = -to_num(work["final_margin"])
        out["final_margin"] = margin
        out["ml_won"] = (margin > 0).astype(float)

        covered_col = f"covered_team_{team}"
        out["covered"] = to_num(work[covered_col]) if covered_col in work.columns else np.nan

        over_hit = (
            to_num(work["over_hit"])
            if "over_hit" in work.columns
            else np.where(
                to_num(work.get("market_total", pd.Series(np.nan, index=work.index))).notna(),
                (
                    to_num(work.get(f"team_{team}_final_score", pd.Series(np.nan, index=work.index)))
                    + to_num(work.get(f"team_{opp}_final_score", pd.Series(np.nan, index=work.index)))
                    > to_num(work.get("market_total", pd.Series(np.nan, index=work.index)))
                ).astype(float),
                np.nan,
            )
        )
        out["total_covered"] = over_hit

        market_spread = to_num(work.get(f"market_spread_team_{team}", pd.Series(np.nan, index=work.index)))
        model_spread = to_num(work.get(f"model_spread_team_{team}", pd.Series(np.nan, index=work.index)))
        edge = (market_spread - model_spread).abs()
        fallback_prob = to_num(work.get(f"team_{team}_win_probability", pd.Series(np.nan, index=work.index)))
        out["model_edge"] = edge.where(edge.notna(), (fallback_prob - 0.5).abs() * 20.0)
        out["is_underdog"] = np.where(
            market_spread.notna(),
            (market_spread > 0).astype(float),
            np.where(
                out["seed_team"].notna() & out["seed_opponent"].notna(),
                (out["seed_team"] > out["seed_opponent"]).astype(float),
                np.nan,
            ),
        )

        market_total = to_num(work.get("market_total", pd.Series(np.nan, index=work.index)))
        model_total = to_num(work.get("model_total", pd.Series(np.nan, index=work.index)))
        over_prob = to_num(work.get("over_probability", pd.Series(np.nan, index=work.index)))
        out["model_projects_over"] = np.where(
            model_total.notna() & market_total.notna(),
            (model_total > market_total),
            np.where(over_prob.notna(), over_prob > 0.5, False),
        ).astype(float)
        out["line_open"] = to_num(work.get("line_open_spread", pd.Series(np.nan, index=work.index)))
        out["line_close"] = to_num(work.get("line_close_spread", market_spread))

        def _metric(name: str) -> pd.Series:
            candidates = [
                f"{name}_team_{team}",
                f"{name}_{team}",
                f"{name}_{'a' if team == 'a' else 'b'}",
            ]
            for col in candidates:
                if col in work.columns:
                    return to_num(work[col])
            return pd.Series(np.nan, index=work.index, dtype="float64")

        out["pace_team"] = _metric("pace_rank")
        # Opponent metrics by explicit side
        out["pace_opponent"] = to_num(work.get(f"pace_rank_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["efg_pct_team"] = to_num(work.get(f"efg_pct_team_{team}", pd.Series(np.nan, index=work.index)))
        out["efg_pct_opponent"] = to_num(work.get(f"efg_pct_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["efg_allowed_team"] = to_num(work.get(f"opp_efg_pct_team_{team}", pd.Series(np.nan, index=work.index)))
        out["efg_allowed_opponent"] = to_num(work.get(f"opp_efg_pct_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["oreb_pct_team"] = to_num(work.get(f"oreb_pct_team_{team}", pd.Series(np.nan, index=work.index)))
        out["dreb_pct_opp"] = 1.0 - to_num(work.get(f"oreb_pct_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["three_pa_rate_team"] = _pick_first(work, [f"3PA_rate_team_{team}", f"three_pa_rate_team_{team}"])
        out["three_pa_rate_opponent"] = _pick_first(work, [f"3PA_rate_team_{opp}", f"three_pa_rate_team_{opp}"])
        out["fta_rate_team"] = to_num(work.get(f"fta_rate_team_{team}", pd.Series(np.nan, index=work.index)))
        out["fta_rate_opponent"] = to_num(work.get(f"fta_rate_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["def_rating_team"] = to_num(work.get(f"DefEff_team_{team}", pd.Series(np.nan, index=work.index)))
        out["def_rating_opponent"] = to_num(work.get(f"DefEff_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["mti_team"] = to_num(work.get(f"MTI_team_{team}", pd.Series(np.nan, index=work.index)))
        out["mti_opponent"] = to_num(work.get(f"MTI_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["spr_team"] = to_num(work.get(f"SPR_team_{team}", pd.Series(np.nan, index=work.index)))
        out["spr_opponent"] = to_num(work.get(f"SPR_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["odi_team"] = to_num(work.get(f"ODI_team_{team}", pd.Series(np.nan, index=work.index)))
        out["odi_opponent"] = to_num(work.get(f"ODI_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["dpc_team"] = to_num(work.get(f"DPC_team_{team}", pd.Series(np.nan, index=work.index)))
        out["dpc_opponent"] = to_num(work.get(f"DPC_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["sci_team"] = to_num(work.get(f"SCI_team_{team}", pd.Series(np.nan, index=work.index)))
        out["sci_opponent"] = to_num(work.get(f"SCI_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["ane_team"] = to_num(work.get(f"ANE_team_{team}", pd.Series(np.nan, index=work.index)))
        out["ane_opponent"] = to_num(work.get(f"ANE_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["forced_tov_rate_team"] = to_num(work.get(f"forced_tov_rate_team_{team}", pd.Series(np.nan, index=work.index)))
        out["forced_tov_rate_opponent"] = to_num(work.get(f"forced_tov_rate_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["ppp_team"] = to_num(work.get(f"ppp_team_{team}", pd.Series(np.nan, index=work.index)))
        out["ppp_opponent"] = to_num(work.get(f"ppp_team_{opp}", pd.Series(np.nan, index=work.index)))
        out["opp_efg_pct_team"] = to_num(work.get(f"opp_efg_pct_team_{team}", pd.Series(np.nan, index=work.index)))
        out["opp_efg_pct_opponent"] = to_num(work.get(f"opp_efg_pct_team_{opp}", pd.Series(np.nan, index=work.index)))
        return out

    a = _one_side("a")
    b = _one_side("b")
    out = pd.concat([a, b], ignore_index=True)
    out["source_dataset"] = "historical_warehouse"
    return out


def build_side_rows_from_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    if backtest.empty:
        return pd.DataFrame()
    if "game_id" not in backtest.columns:
        return pd.DataFrame()

    work = backtest.copy()
    work["game_id"] = normalize_game_id(work["game_id"])
    work = work[work["game_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    dt_col = "game_datetime" if "game_datetime" in work.columns else "game_datetime_utc"
    phase = derive_phase(work.get(dt_col, pd.Series(np.nan, index=work.index)))
    is_tourney = phase.astype(str).str.contains("tournament", case=False, na=False)

    market_spread = to_num(work.get("market_spread", pd.Series(np.nan, index=work.index)))
    ens_spread = to_num(work.get("ens_spread", pd.Series(np.nan, index=work.index)))
    actual_margin = to_num(work.get("actual_margin", pd.Series(np.nan, index=work.index)))
    market_total = to_num(work.get("market_total", pd.Series(np.nan, index=work.index)))
    ens_total = to_num(work.get("ens_total", pd.Series(np.nan, index=work.index)))
    actual_total = to_num(work.get("actual_total", pd.Series(np.nan, index=work.index)))
    edge_home = (market_spread - ens_spread).abs().where(market_spread.notna(), np.nan)
    edge_fallback = to_num(work.get("pred_margin_ATS", pd.Series(np.nan, index=work.index))).abs()
    model_edge_home = edge_home.where(edge_home.notna(), edge_fallback)

    def _one_side(side: str) -> pd.DataFrame:
        team = "home" if side == "home" else "away"
        opp = "away" if team == "home" else "home"
        sign = 1.0 if side == "home" else -1.0

        out = pd.DataFrame(index=work.index)
        out["game_id"] = work["game_id"]
        out["season"] = pd.to_datetime(work.get(dt_col), errors="coerce", utc=True).dt.year
        out["game_datetime_utc"] = work.get(dt_col, pd.Series(np.nan, index=work.index))
        out["phase"] = phase
        out["is_tournament"] = is_tourney
        out["is_ncaa_tournament"] = (phase == "ncaa_tournament")
        out["is_conference_tournament"] = (phase == "conference_tournament")
        out["tourney_round"] = np.nan
        out["team_name"] = (
            work.get(f"{team}_team", pd.Series(np.nan, index=work.index))
            .astype(str)
            .replace({"nan": np.nan, "None": np.nan})
        )
        out["opponent_name"] = (
            work.get(f"{opp}_team", pd.Series(np.nan, index=work.index))
            .astype(str)
            .replace({"nan": np.nan, "None": np.nan})
        )
        out["conference_team"] = np.nan
        out["conference_opponent"] = np.nan
        out["seed_team"] = np.nan
        out["seed_opponent"] = np.nan
        out["final_margin"] = actual_margin * sign
        out["ml_won"] = (out["final_margin"] > 0).astype(float)
        out["covered"] = np.where(
            market_spread.notna() & actual_margin.notna(),
            ((actual_margin * sign) + (market_spread * sign) > 0).astype(float),
            np.nan,
        )
        out["total_covered"] = np.where(
            market_total.notna() & actual_total.notna(),
            (actual_total > market_total).astype(float),
            np.nan,
        )
        out["model_edge"] = model_edge_home
        out["is_underdog"] = np.where(market_spread.notna(), ((market_spread * sign) > 0).astype(float), np.nan)
        out["model_projects_over"] = np.where(
            ens_total.notna() & market_total.notna(),
            (ens_total > market_total).astype(float),
            np.nan,
        )
        out["line_open"] = to_num(work.get("opening_spread", pd.Series(np.nan, index=work.index))) * sign
        out["line_close"] = to_num(work.get("closing_spread", market_spread)) * sign
        out["source_dataset"] = "backtest_results_latest"
        return out

    home = _one_side("home")
    away = _one_side("away")
    out = pd.concat([home, away], ignore_index=True)
    return out


def merge_and_dedupe_side_rows(*frames: pd.DataFrame) -> pd.DataFrame:
    available = [f.copy() for f in frames if f is not None and not f.empty]
    if not available:
        return pd.DataFrame()

    combined = pd.concat(available, ignore_index=True, sort=False)
    for col in ["game_id", "team_name", "opponent_name"]:
        if col not in combined.columns:
            combined[col] = np.nan

    key = ["game_id", "team_name", "opponent_name"]
    quality_cols = [
        "model_edge",
        "covered",
        "ml_won",
        "total_covered",
        "is_underdog",
        "seed_team",
        "seed_opponent",
        "phase",
    ]
    combined["_quality"] = combined[quality_cols].notna().sum(axis=1)
    combined = combined.sort_values(["_quality"], ascending=False)
    deduped = combined.drop_duplicates(subset=key, keep="first").drop(columns=["_quality"])
    return deduped.reset_index(drop=True)
