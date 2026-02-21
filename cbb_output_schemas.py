"""
CBB Output File Schemas — Centralized schema contract.

Single source of truth for required columns across all pipeline CSV outputs.
Addresses Root Cause #3 (no centralized schema contract) and Root Cause #5
(no pre-write integrity gate) from docs_pipeline_field_trace_report.md.

Usage:
    from cbb_output_schemas import validate_output, OUTPUT_FILE_SCHEMAS

    validate_output(df, "team_game_logs")  # raises ValueError on missing cols
"""

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

log = logging.getLogger(__name__)

# ── Required column sets per output file ─────────────────────────────────────
# Each key is a logical output name; value is the set of columns that MUST
# exist in the DataFrame before it is written to CSV.

OUTPUT_FILE_SCHEMAS: Dict[str, List[str]] = {
    "games": [
        "game_id", "home_team", "away_team",
        "home_team_id", "away_team_id",
        "completed",
    ],
    "team_game_logs": [
        "event_id", "team_id", "conference", "wins", "losses",
        "fgm", "fga", "ftm", "fta", "tpm", "tpa",
        "orb", "drb", "reb", "tov", "ast",
        "opp_fgm", "opp_fga", "opp_ftm", "opp_fta",
        "opp_tpm", "opp_tpa", "opp_orb", "opp_drb", "opp_tov",
        "FGA", "FGM", "FTA", "FTM", "TPA", "TPM",
        "ORB", "DRB", "RB", "TO", "AST",
    ],
    "player_game_logs": [
        "event_id", "team_id", "athlete_id",
        "fgm", "fga", "ftm", "fta", "tpm", "tpa",
        "orb", "drb", "reb", "tov", "ast",
        "FGA", "FGM", "FTA", "FTM", "TPA", "TPM",
        "ORB", "DRB", "RB", "TO", "AST",
    ],
    "player_game_metrics": [
        "event_id", "team_id", "athlete_id",
        "fgm", "fga", "ftm", "fta", "tpm", "tpa",
        "orb", "drb", "reb", "tov", "ast",
        "FGA", "FGM", "FTA", "FTM", "TPA", "TPM",
        "ORB", "DRB", "RB", "TO", "AST",
    ],
    "team_game_metrics": [
        "event_id", "team_id",
    ],
    "team_game_sos": [
        "event_id", "team_id",
    ],
    "team_game_weighted": [
        "event_id", "team_id",
    ],
    "predictions": [
        "game_id", "home_team_id", "away_team_id",
        "home_team", "away_team",
        "pred_spread", "pred_total", "model_confidence",
        "home_conference", "home_wins", "home_losses",
        "away_conference", "away_wins", "away_losses",
        "home_FGA", "home_FGM", "home_FTA", "home_FTM",
        "home_TPA", "home_TPM", "home_ORB", "home_DRB",
        "home_RB", "home_TO", "home_AST",
        "away_FGA", "away_FGM", "away_FTA", "away_FTM",
        "away_TPA", "away_TPM", "away_ORB", "away_DRB",
        "away_RB", "away_TO", "away_AST",
    ],
}


def validate_output(
    df: pd.DataFrame,
    schema_name: str,
    *,
    strict: bool = False,
) -> List[str]:
    """Validate a DataFrame against a named output schema.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    schema_name : str
        Key into ``OUTPUT_FILE_SCHEMAS``.
    strict : bool
        If *True*, raise ``ValueError`` on any missing columns.
        If *False* (default), log warnings and return the list of missing columns.

    Returns
    -------
    list[str]
        Sorted list of missing required columns (empty if all present).

    Raises
    ------
    KeyError
        If *schema_name* is not defined in ``OUTPUT_FILE_SCHEMAS``.
    ValueError
        If *strict* is True and required columns are missing.
    """
    required = OUTPUT_FILE_SCHEMAS.get(schema_name)
    if required is None:
        raise KeyError(f"Unknown output schema: {schema_name!r}")

    missing = sorted(set(required) - set(df.columns))

    if missing:
        msg = f"Output '{schema_name}' missing required columns: {missing}"
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    return missing


def completeness_report(
    dataframes: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Generate a data-quality completeness report for multiple outputs.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Mapping of schema name → DataFrame (only schemas defined in
        ``OUTPUT_FILE_SCHEMAS`` are checked).

    Returns
    -------
    pd.DataFrame
        One row per output with columns:
        ``output, rows, required_cols, present_cols, missing_cols, missing_list,
        null_pct`` (average null percentage across required columns).
    """
    rows = []
    for name, df in dataframes.items():
        schema = OUTPUT_FILE_SCHEMAS.get(name)
        if schema is None:
            continue

        required_set = set(schema)
        present = sorted(required_set & set(df.columns))
        missing = sorted(required_set - set(df.columns))

        # Average null percentage across present required columns
        if present and len(df) > 0:
            null_pct = round(
                df[present].isnull().mean().mean() * 100, 2
            )
        else:
            null_pct = 0.0 if len(df) == 0 else 100.0

        rows.append({
            "output": name,
            "rows": len(df),
            "required_cols": len(schema),
            "present_cols": len(present),
            "missing_cols": len(missing),
            "missing_list": ", ".join(missing) if missing else "",
            "null_pct": null_pct,
        })

    return pd.DataFrame(rows)
