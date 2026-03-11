from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ACTIONABLE_DIR = DATA_DIR / "actionable"
REPORTS_DIR = DATA_DIR / "reports"
PLUMBING_DIR = DATA_DIR / "plumbing"
CONTRACTS_DIR = DATA_DIR / "contracts"
LOGS_DIR = DATA_DIR / "logs"

DEFAULT_HOURS_AHEAD = 48
MIN_HISTORY_GAMES = 120
MIN_RULE_SAMPLE = 50
RULE_SHRINK_K = 75.0
SPREAD_BET_EDGE_MIN = 1.0
SPREAD_BET_CONFIDENCE_MIN = 52.0
SPREAD_BET_ATS_PROB_EDGE_MIN = 0.02

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
    games_schedule_master: Path
    team_game_boxscores: Path
    player_game_boxscores: Path
    team_rolling_features: Path
    game_matchup_features: Path
    game_totals_features: Path
    game_context_adjustments: Path
    situational_signals_game_level: Path
    game_monte_carlo_outputs: Path
    game_predictions_master: Path
    bet_recommendations: Path
    watchlist_games: Path
    no_bet_explanations: Path
    daily_card_summary: Path
    agreement_analysis_results: Path
    backtest_model_summary: Path
    csv_contract_registry: Path
    schema_rules: Path
    pipeline_run_log: Path
    run_manifest: Path


def output_paths(
    *,
    plumbing_dir: Path = PLUMBING_DIR,
    reports_dir: Path = REPORTS_DIR,
    actionable_dir: Path = ACTIONABLE_DIR,
    contracts_dir: Path = CONTRACTS_DIR,
    logs_dir: Path = LOGS_DIR,
) -> OutputPaths:
    return OutputPaths(
        games_schedule_master=plumbing_dir / "games_schedule_master.csv",
        team_game_boxscores=plumbing_dir / "team_game_boxscores.csv",
        player_game_boxscores=plumbing_dir / "player_game_boxscores.csv",
        team_rolling_features=plumbing_dir / "team_rolling_features.csv",
        game_matchup_features=plumbing_dir / "game_matchup_features.csv",
        game_totals_features=plumbing_dir / "game_totals_features.csv",
        game_context_adjustments=plumbing_dir / "game_context_adjustments.csv",
        situational_signals_game_level=plumbing_dir / "situational_signals_game_level.csv",
        game_monte_carlo_outputs=plumbing_dir / "game_monte_carlo_outputs.csv",
        game_predictions_master=reports_dir / "game_predictions_master.csv",
        bet_recommendations=actionable_dir / "bet_recommendations.csv",
        watchlist_games=actionable_dir / "watchlist_games.csv",
        no_bet_explanations=actionable_dir / "no_bet_explanations.csv",
        daily_card_summary=actionable_dir / "daily_card_summary.csv",
        agreement_analysis_results=reports_dir / "agreement_analysis_results.csv",
        backtest_model_summary=reports_dir / "backtest_model_summary.csv",
        csv_contract_registry=contracts_dir / "csv_contract_registry.csv",
        schema_rules=contracts_dir / "schema_rules.json",
        pipeline_run_log=logs_dir / "pipeline_run_log.csv",
        run_manifest=logs_dir / "run_manifest.json",
    )
