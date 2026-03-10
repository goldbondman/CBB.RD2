from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
IDES_DIR = DATA_DIR / "ides_of_march"

DEFAULT_HOURS_AHEAD = 48
MIN_HISTORY_GAMES = 120
MIN_RULE_SAMPLE = 50
RULE_SHRINK_K = 75.0

MODEL_B_WEIGHTS = {
    "adj_em_margin": 0.45,
    "efg_margin": 0.22,
    "to_margin": 0.14,
    "oreb_margin": 0.11,
    "ft_scoring_pressure_margin": 0.08,
}

MODEL_A_WEIGHTS = {
    "adj_em_margin": 0.58,
    "efg_margin": 0.18,
    "to_margin": 0.12,
    "oreb_margin": 0.07,
    "ftr_margin": 0.05,
}


@dataclass(frozen=True)
class OutputPaths:
    predictions_latest: Path
    bet_recs: Path
    agreement_bucket_report: Path
    situational_rulebook: Path
    backtest_variant_scorecard: Path
    run_manifest: Path


def output_paths(base_dir: Path = IDES_DIR) -> OutputPaths:
    return OutputPaths(
        predictions_latest=base_dir / "predictions_latest.csv",
        bet_recs=base_dir / "bet_recs.csv",
        agreement_bucket_report=base_dir / "agreement_bucket_report.csv",
        situational_rulebook=base_dir / "situational_rulebook.csv",
        backtest_variant_scorecard=base_dir / "backtest_variant_scorecard.csv",
        run_manifest=base_dir / "run_manifest.json",
    )
