import argparse
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbb_prediction_model import ModelConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

REQUIRED_DELTA_COLUMNS = [
    "model_efg_delta",
    "model_tov_delta",
    "model_orb_delta",
    "model_drb_delta",
    "model_ftr_delta",
    "model_tpar_delta",
]

DEFAULT_SCORE_COL = "clv_vs_consensus"

SEARCH_PARAM_KEYS = [
    "efg_weight",
    "tov_weight",
    "orb_weight",
    "drb_weight",
    "ftr_weight",
    "three_par_weight",
    "vs_exp_weight",
    "eff_composite_weight",
    "schedule_adjustment_factor",
    "adj_pace_weight",
    "pace_regression_factor",
    "cage_prior_weight",
    "decay_floor",
    "decay_cliff",
]


def _safe_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _resolve_prior_value(prior_blob: dict, key: str, default: float) -> float:
    if key in prior_blob:
        return _safe_float(prior_blob[key], default)

    if key == "raw_weight" and "vs_exp_weight" in prior_blob:
        return max(0.0, min(1.0, 1.0 - _safe_float(prior_blob["vs_exp_weight"], 1.0 - default)))

    return default


def _load_prior_weights(path: Path) -> tuple[dict, str]:
    cfg = ModelConfig()
    defaults = {
        "efg_weight": cfg.efg_weight,
        "tov_weight": cfg.tov_weight,
        "orb_weight": cfg.orb_weight,
        "drb_weight": cfg.drb_weight,
        "ftr_weight": cfg.ftr_weight,
        "three_par_weight": 0.0,
        "vs_exp_weight": cfg.vs_exp_weight,
        "raw_weight": cfg.raw_weight,
        "eff_composite_weight": cfg.eff_composite_weight,
        "schedule_adjustment_factor": cfg.schedule_adjustment_factor,
        "adj_pace_weight": 0.0,
        "pace_regression_factor": 0.0,
        "cage_prior_weight": 0.0,
        "decay_floor": cfg.decay_floor,
        "decay_cliff": cfg.decay_cliff,
    }

    if not path.exists():
        return defaults, "ModelConfig defaults"

    with path.open() as f:
        loaded = json.load(f)

    prior = {}
    for key, default in defaults.items():
        prior[key] = _resolve_prior_value(loaded, key, default)

    return prior, "active_weights.json"


def _frange(start: float, end: float, step: float) -> list[float]:
    if start > end:
        return []

    values = []
    idx = 0
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 4))
        idx += 1
        current = start + idx * step
    return values


def _range_around_prior(
    prior: float,
    step: float,
    min_valid: float,
    max_valid: float,
    explicit_values: Iterable[float] | None = None,
) -> list[float]:
    lower = max(min_valid, round(prior - 0.10, 8))
    upper = min(max_valid, round(prior + 0.10, 8))

    if explicit_values is not None:
        return [
            round(v, 4)
            for v in explicit_values
            if lower - 1e-9 <= float(v) <= upper + 1e-9
        ]

    base_start = round(min_valid + round((lower - min_valid) / step) * step, 8)
    while base_start < lower - 1e-9:
        base_start = round(base_start + step, 8)

    values = _frange(base_start, upper, step)
    return [v for v in values if lower - 1e-9 <= v <= upper + 1e-9]


def _load_scoring_data(primary_path: Path, fallback_path: Path) -> tuple[pd.DataFrame, Path]:
    source_path = primary_path if primary_path.exists() else fallback_path
    if not source_path.exists():
        raise FileNotFoundError(f"No scoring file found at {primary_path} or {fallback_path}")

    df = pd.read_csv(source_path, low_memory=False)
    if df.empty:
        raise ValueError(f"{source_path} has no rows")

    return df, source_path


def _extract_graded_rows(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if "graded" not in df.columns:
        raise ValueError("Scoring file missing required 'graded' column")
    if score_col not in df.columns:
        raise ValueError(f"Scoring file missing requested score column: {score_col}")

    graded_series = df["graded"]
    if graded_series.dtype == bool:
        graded_mask = graded_series
    else:
        graded_mask = graded_series.astype(str).str.lower().isin(["true", "1", "yes"])

    working = df[graded_mask].copy()
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
    working = working[working[score_col].notna()].copy()
    return working


def _resolve_column(df: pd.DataFrame, *options: str) -> str | None:
    for col in options:
        if col in df.columns:
            return col
    return None


def _prepare_fast_scoring_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    missing_delta_cols = [col for col in REQUIRED_DELTA_COLUMNS if col not in df.columns]
    if missing_delta_cols:
        log.warning(
            "Missing fast-score delta columns (%s). Falling back to MAE scoring.",
            ", ".join(missing_delta_cols),
        )
        return df, False

    closing_col = _resolve_column(df, "closing_line", "closing_spread_line", "spread_line", "market_spread")
    eff_edge_col = _resolve_column(df, "eff_edge")

    if closing_col is None or eff_edge_col is None:
        log.warning(
            "Missing required columns for CLV recompute (closing_line/eff_edge). Falling back to MAE scoring."
        )
        return df, False

    prepared = df.copy()
    for col in REQUIRED_DELTA_COLUMNS + [closing_col, eff_edge_col]:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    prepared = prepared.dropna(subset=REQUIRED_DELTA_COLUMNS + [closing_col, eff_edge_col]).copy()
    prepared = prepared.rename(columns={closing_col: "_closing_line", eff_edge_col: "_eff_edge"})
    return prepared, True


def _build_search_grid(prior: dict) -> dict[str, list[float]]:
    grid = {
        "efg_weight": _range_around_prior(prior["efg_weight"], step=0.04, min_valid=0.0, max_valid=1.0),
        "tov_weight": _range_around_prior(prior["tov_weight"], step=0.04, min_valid=0.0, max_valid=1.0),
        "orb_weight": _range_around_prior(prior["orb_weight"], step=0.04, min_valid=0.0, max_valid=1.0),
        "drb_weight": _range_around_prior(prior["drb_weight"], step=0.04, min_valid=0.0, max_valid=1.0),
        "ftr_weight": _range_around_prior(prior["ftr_weight"], step=0.04, min_valid=0.0, max_valid=1.0),
        "three_par_weight": _range_around_prior(
            prior["three_par_weight"],
            step=0.03,
            min_valid=0.0,
            max_valid=0.12,
            explicit_values=[0.00, 0.03, 0.06, 0.09, 0.12],
        ),
        "vs_exp_weight": _range_around_prior(prior["vs_exp_weight"], step=0.05, min_valid=0.50, max_valid=0.90),
        "eff_composite_weight": _range_around_prior(
            prior["eff_composite_weight"],
            step=0.05,
            min_valid=0.50,
            max_valid=0.70,
            explicit_values=[0.50, 0.55, 0.60, 0.65, 0.70],
        ),
        "schedule_adjustment_factor": _range_around_prior(
            prior["schedule_adjustment_factor"],
            step=0.10,
            min_valid=0.30,
            max_valid=0.70,
            explicit_values=[0.30, 0.40, 0.50, 0.60, 0.70],
        ),
        "adj_pace_weight": _range_around_prior(
            prior["adj_pace_weight"],
            step=0.20,
            min_valid=0.0,
            max_valid=0.60,
            explicit_values=[0.00, 0.20, 0.40, 0.60],
        ),
        "pace_regression_factor": _range_around_prior(
            prior["pace_regression_factor"],
            step=0.25,
            min_valid=0.0,
            max_valid=0.50,
            explicit_values=[0.00, 0.25, 0.50],
        ),
        "cage_prior_weight": _range_around_prior(
            prior["cage_prior_weight"],
            step=0.10,
            min_valid=0.0,
            max_valid=0.30,
            explicit_values=[0.00, 0.10, 0.20, 0.30],
        ),
        "decay_floor": _range_around_prior(
            prior["decay_floor"],
            step=0.10,
            min_valid=0.40,
            max_valid=0.60,
            explicit_values=[0.40, 0.50, 0.60],
        ),
        "decay_cliff": _range_around_prior(
            prior["decay_cliff"],
            step=0.10,
            min_valid=0.65,
            max_valid=0.85,
            explicit_values=[0.65, 0.75, 0.85],
        ),
    }

    for key, values in grid.items():
        if not values:
            clipped = max(min(prior[key], 1.0), 0.0)
            grid[key] = [round(clipped, 4)]

    return grid


def _candidate_composite(df: pd.DataFrame, combo: dict) -> pd.Series:
    return (
        combo["efg_weight"] * df["model_efg_delta"]
        + combo["tov_weight"] * df["model_tov_delta"]
        + combo["orb_weight"] * df["model_orb_delta"]
        + combo["drb_weight"] * df["model_drb_delta"]
        + combo["ftr_weight"] * df["model_ftr_delta"]
        + combo["three_par_weight"] * df["model_tpar_delta"]
    )


def _score_combo_fast(df: pd.DataFrame, combo: dict) -> tuple[float, int]:
    reweighted_composite = _candidate_composite(df, combo)
    implied_spread = -(
        combo["eff_composite_weight"] * df["_eff_edge"]
        + (1.0 - combo["eff_composite_weight"]) * reweighted_composite
    )
    clv = implied_spread - df["_closing_line"]

    clv = clv.dropna()
    if clv.empty:
        return float("-inf"), 0

    return float(clv.mean()), int(clv.shape[0])


def _score_combo_mae(df: pd.DataFrame, combo: dict) -> tuple[float, int]:
    pred_col = _resolve_column(df, "predicted_spread", "pred_spread")
    close_col = _resolve_column(df, "closing_line", "closing_spread_line", "spread_line", "market_spread")
    err_col = _resolve_column(df, "spread_error")

    scoped = df.copy()
    if err_col is not None:
        scoped["_spread_error"] = pd.to_numeric(scoped[err_col], errors="coerce")
    elif pred_col is not None and close_col is not None:
        scoped["_pred"] = pd.to_numeric(scoped[pred_col], errors="coerce")
        scoped["_close"] = pd.to_numeric(scoped[close_col], errors="coerce")
        scoped["_spread_error"] = scoped["_pred"] - scoped["_close"]
    else:
        raise ValueError("Cannot compute MAE fallback: missing spread_error or predicted/closing spread columns")

    scoped = scoped.dropna(subset=["_spread_error"]).copy()
    if scoped.empty:
        return float("-inf"), 0

    score = -float(scoped["_spread_error"].abs().mean())
    return score, int(scoped.shape[0])


def _score_combo_walk_forward(
    df: pd.DataFrame,
    combo: dict,
    use_fast_clv: bool,
    min_train_weeks: int,
    test_window_weeks: int,
) -> tuple[float, int, int]:
    if "game_datetime_utc" not in df.columns:
        return float("-inf"), 0, 0

    scoped = df.copy()
    scoped["game_datetime_utc"] = pd.to_datetime(scoped["game_datetime_utc"], errors="coerce", utc=True)
    scoped = scoped.dropna(subset=["game_datetime_utc"]).sort_values("game_datetime_utc")
    if scoped.empty:
        return float("-inf"), 0, 0

    scoped["week"] = scoped["game_datetime_utc"].dt.to_period("W")
    weeks = sorted(scoped["week"].dropna().unique())
    if len(weeks) < (min_train_weeks + test_window_weeks):
        return float("-inf"), 0, 0

    fold_scores: list[float] = []
    sample_size = 0
    for i in range(min_train_weeks, len(weeks) - test_window_weeks + 1):
        test_weeks = weeks[i : i + test_window_weeks]
        test = scoped[scoped["week"].isin(test_weeks)]
        if test.empty:
            continue

        if use_fast_clv:
            score, n_games = _score_combo_fast(test, combo)
        else:
            score, n_games = _score_combo_mae(test, combo)

        if n_games <= 0 or score == float("-inf"):
            continue

        fold_scores.append(score)
        sample_size += n_games

    if not fold_scores:
        return float("-inf"), 0, 0

    mean_score = float(pd.Series(fold_scores).mean())
    stability_penalty = float(pd.Series(fold_scores).std(ddof=0) or 0.0)
    robust_score = mean_score - (0.15 * stability_penalty)
    return robust_score, sample_size, len(fold_scores)


def run_search(
    df: pd.DataFrame,
    prior: dict,
    use_fast_clv: bool,
    robust_cv: bool,
    min_train_weeks: int,
    test_window_weeks: int,
    min_folds: int,
) -> tuple[dict, int, list[tuple[float, int, dict]]]:
    grid = _build_search_grid(prior)

    keys = SEARCH_PARAM_KEYS
    iterables = [grid[k] for k in keys]

    best_combo = None
    best_score = float("-inf")
    best_n = 0
    search_space_size = 0
    top_results: list[tuple[float, int, dict]] = []

    for values in itertools.product(*iterables):
        combo = dict(zip(keys, values))

        factor_sum = (
            combo["efg_weight"]
            + combo["tov_weight"]
            + combo["orb_weight"]
            + combo["drb_weight"]
            + combo["ftr_weight"]
            + combo["three_par_weight"]
        )
        if not (0.90 <= factor_sum <= 1.10):
            continue

        combo["raw_weight"] = round(1.0 - combo["vs_exp_weight"], 4)
        search_space_size += 1

        fold_count = 0
        if robust_cv:
            score, n_games, fold_count = _score_combo_walk_forward(
                df,
                combo,
                use_fast_clv=use_fast_clv,
                min_train_weeks=min_train_weeks,
                test_window_weeks=test_window_weeks,
            )
            if fold_count < min_folds:
                continue
        elif use_fast_clv:
            score, n_games = _score_combo_fast(df, combo)
        else:
            score, n_games = _score_combo_mae(df, combo)

        tracked_combo = {**combo, "fold_count": fold_count} if robust_cv else combo.copy()
        entry = (score, n_games, tracked_combo)
        top_results.append(entry)
        top_results = sorted(top_results, key=lambda x: (x[0], x[1]), reverse=True)[:5]

        if score > best_score:
            best_combo = tracked_combo
            best_score = score
            best_n = n_games

    if best_combo is None:
        raise ValueError("No valid candidate combinations were generated from current prior constraints")

    return {
        **best_combo,
        "n_games_used": int(best_n),
        "score": float(best_score),
        "fold_count": int(best_combo.get("fold_count", 0)),
    }, search_space_size, top_results


def _print_top_results(top_results: list[tuple[float, int, dict]], score_col: str, use_fast_clv: bool) -> None:
    metric_name = f"mean_{score_col}" if use_fast_clv else "-mae_spread_error"
    log.info("Top %s candidate sets by %s:", len(top_results), metric_name)
    for idx, (score, n_games, combo) in enumerate(top_results, start=1):
        log.info("%s) score=%.6f n_games=%s params=%s", idx, score, n_games, combo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-log", default="data/results_log.csv")
    parser.add_argument("--graded", default="data/predictions_graded.csv")
    parser.add_argument("--prior", default="data/active_weights.json")
    parser.add_argument("--output", default="data/candidate_weights.json")
    parser.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-games", type=int, default=200)
    parser.add_argument("--full-season", action="store_true")
    parser.add_argument("--disable-robust-cv", action="store_true")
    parser.add_argument("--min-train-weeks", type=int, default=4)
    parser.add_argument("--test-window-weeks", type=int, default=1)
    parser.add_argument("--min-folds", type=int, default=4)
    args = parser.parse_args()

    results_log_path = Path(args.results_log)
    graded_path = Path(args.graded)
    prior_path = Path(args.prior)
    output_path = Path(args.output)

    try:
        raw_df, source_path = _load_scoring_data(graded_path, results_log_path)
        scored_df = _extract_graded_rows(raw_df, args.score_col)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("Skipping optimization: %s", exc)
        return

    if not args.full_season and len(scored_df) < args.min_games:
        log.warning(
            "Skipping optimization: graded rows (%s) below --min-games threshold (%s). Use --full-season to bypass.",
            len(scored_df),
            args.min_games,
        )
        return

    prior, prior_source = _load_prior_weights(prior_path)
    prepared_df, use_fast_clv = _prepare_fast_scoring_columns(scored_df)

    robust_cv = not args.disable_robust_cv and "game_datetime_utc" in prepared_df.columns
    if robust_cv:
        log.info(
            "Robust CV enabled (min_train_weeks=%s, test_window_weeks=%s, min_folds=%s)",
            args.min_train_weeks,
            args.test_window_weeks,
            args.min_folds,
        )
    else:
        log.info("Robust CV disabled; using single-sample scoring.")

    best, search_space_size, top_results = run_search(
        prepared_df,
        prior,
        use_fast_clv=use_fast_clv,
        robust_cv=robust_cv,
        min_train_weeks=args.min_train_weeks,
        test_window_weeks=args.test_window_weeks,
        min_folds=args.min_folds,
    )

    score_key = "best_clv_mean" if use_fast_clv else "best_mae_score"
    output_payload = {
        **{k: best[k] for k in SEARCH_PARAM_KEYS},
        "raw_weight": best["raw_weight"],
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_games_used": best["n_games_used"],
        score_key: round(best["score"], 4),
        "search_space_size": search_space_size,
        "prior_source": prior_source,
        "score_col": args.score_col,
        "robust_cv": robust_cv,
        "fold_count": best.get("fold_count", 0),
    }

    _print_top_results(top_results, args.score_col, use_fast_clv)
    log.info("Scoring source: %s", source_path)
    log.info("Graded rows used for scoring: %s", len(scored_df))
    log.info("Search space evaluated: %s", search_space_size)

    if args.dry_run:
        log.info("Dry run enabled; skipping write to %s", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2))
    log.info("Wrote candidate weights to %s", output_path)


if __name__ == "__main__":
    main()
