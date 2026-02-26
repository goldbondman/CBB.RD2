import argparse
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cbb_prediction_model import ModelConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

CLV_COL = "clv_vs_consensus"
MARKET_SPREAD_COL = "market_spread"


def _load_scoring_data(primary_path: Path, fallback_path: Path) -> tuple[pd.DataFrame, Path]:
    source_path = primary_path if primary_path.exists() else fallback_path
    if not source_path.exists():
        raise FileNotFoundError(f"No scoring file found at {primary_path} or {fallback_path}")

    df = pd.read_csv(source_path, low_memory=False)
    if CLV_COL not in df.columns:
        raise ValueError(f"{source_path} is missing required column: {CLV_COL}")
    if MARKET_SPREAD_COL not in df.columns:
        raise ValueError(f"{source_path} is missing required column: {MARKET_SPREAD_COL}")

    df[CLV_COL] = pd.to_numeric(df[CLV_COL], errors="coerce")
    df[MARKET_SPREAD_COL] = pd.to_numeric(df[MARKET_SPREAD_COL], errors="coerce")

    usable = df[df[CLV_COL].notna() & df[MARKET_SPREAD_COL].notna()].copy()
    return usable, source_path


def _load_prior_weights(path: Path) -> dict:
    cfg = ModelConfig()
    defaults = {
        "efg_weight": cfg.efg_weight,
        "tov_weight": cfg.tov_weight,
        "orb_weight": cfg.orb_weight,
        "drb_weight": cfg.drb_weight,
        "ftr_weight": cfg.ftr_weight,
        "vs_exp_weight": cfg.vs_exp_weight,
        "raw_weight": cfg.raw_weight,
        "eff_composite_weight": cfg.eff_composite_weight,
        "schedule_adjustment_factor": cfg.schedule_adjustment_factor,
        "decay_floor": cfg.decay_floor,
    }

    if not path.exists():
        return defaults

    with path.open() as f:
        loaded = json.load(f)

    return {
        key: float(loaded.get(key, defaults[key]))
        for key in defaults
    }


def _within_prior_bounds(candidate: dict, prior: dict, max_delta: float = 0.10) -> bool:
    for key, value in candidate.items():
        if key in prior and abs(float(value) - float(prior[key])) > max_delta:
            return False
    return True


def _score_combo(df: pd.DataFrame, combo: dict) -> tuple[float, int]:
    scoped = df
    combo_cols = [col for col in combo if col in df.columns]

    for col in combo_cols:
        scoped = scoped[scoped[col].round(6) == round(float(combo[col]), 6)]

    if scoped.empty:
        return float("-inf"), 0

    return float(scoped[CLV_COL].mean()), int(len(scoped))


def _iter_four_factor_weights(step: float = 0.05):
    values = [round(v * step, 2) for v in range(int(1 / step) + 1)]
    for efg, tov, orb, drb, ftr in itertools.product(values, repeat=5):
        if abs((efg + tov + orb + drb + ftr) - 1.0) < 1e-9:
            yield efg, tov, orb, drb, ftr


def run_search(df: pd.DataFrame, prior: dict) -> tuple[dict, int]:
    eff_composite_grid = [0.50, 0.55, 0.60, 0.65, 0.70]
    vs_exp_grid = [0.60, 0.65, 0.70, 0.75, 0.80]
    schedule_grid = [0.30, 0.40, 0.50, 0.60, 0.70]
    decay_floor_grid = [0.40, 0.50, 0.60]

    best_combo = None
    best_score = float("-inf")
    best_n = 0
    search_space_size = 0

    for efg, tov, orb, drb, ftr in _iter_four_factor_weights(step=0.05):
        for eff_composite_weight, vs_exp_weight, schedule_adjustment_factor, decay_floor in itertools.product(
            eff_composite_grid, vs_exp_grid, schedule_grid, decay_floor_grid
        ):
            combo = {
                "efg_weight": efg,
                "tov_weight": tov,
                "orb_weight": orb,
                "drb_weight": drb,
                "ftr_weight": ftr,
                "eff_composite_weight": eff_composite_weight,
                "vs_exp_weight": vs_exp_weight,
                "raw_weight": round(1.0 - vs_exp_weight, 2),
                "schedule_adjustment_factor": schedule_adjustment_factor,
                "decay_floor": decay_floor,
            }

            if not _within_prior_bounds(combo, prior, max_delta=0.10):
                continue

            search_space_size += 1
            score, n_games = _score_combo(df, combo)
            if score > best_score:
                best_combo = combo
                best_score = score
                best_n = n_games

    if best_combo is None:
        fallback = {
            "efg_weight": prior["efg_weight"],
            "tov_weight": prior["tov_weight"],
            "orb_weight": prior["orb_weight"],
            "drb_weight": prior["drb_weight"],
            "ftr_weight": prior["ftr_weight"],
            "eff_composite_weight": prior["eff_composite_weight"],
            "vs_exp_weight": prior["vs_exp_weight"],
            "raw_weight": prior["raw_weight"],
            "schedule_adjustment_factor": prior["schedule_adjustment_factor"],
            "decay_floor": prior["decay_floor"],
        }
        best_combo = fallback
        best_score, best_n = _score_combo(df, fallback)

    result = {
        **best_combo,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_games_used": best_n,
        "best_clv_mean": None if best_score == float("-inf") else round(best_score, 6),
        "search_space_size": search_space_size,
    }
    return result, search_space_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-log", default="data/results_log.csv")
    parser.add_argument("--graded", default="data/predictions_graded.csv")
    parser.add_argument("--prior", default="data/active_weights.json")
    parser.add_argument("--output", default="data/candidate_weights.json")
    args = parser.parse_args()

    results_log_path = Path(args.results_log)
    graded_path = Path(args.graded)
    prior_path = Path(args.prior)
    output_path = Path(args.output)

    try:
        scoring_df, source_path = _load_scoring_data(results_log_path, graded_path)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("Skipping optimization: %s", exc)
        return

    prior = _load_prior_weights(prior_path)
    best, search_space_size = run_search(scoring_df, prior)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best, indent=2))

    log.info("Scoring source: %s", source_path)
    log.info("Rows usable for CLV scoring: %s", len(scoring_df))
    log.info("Search space evaluated: %s", search_space_size)
    log.info("Wrote candidate weights to %s", output_path)


if __name__ == "__main__":
    main()
