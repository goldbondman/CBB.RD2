#!/usr/bin/env python3
"""
fix_conferences.py
==================
Retroactively patches conference names in all existing CSVs to ensure
accuracy, using the corrected ESPN_CONFERENCE_MAP from espn_config.py.

Strategy:
  1. Build a canonical team_id -> conf_id mapping from games.csv (which
     stores the raw ESPN conference IDs).
  2. Use the corrected ESPN_CONFERENCE_MAP to derive the correct
     conference name for each conf_id.
  3. Apply the correct conference name to every CSV that contains a
     conference column, keyed by team_id (where available) or conf_id.

Run this script once after updating ESPN_CONFERENCE_MAP, then the live
pipeline will produce correct data going forward.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

from espn_config import ESPN_CONFERENCE_MAP

DATA = pathlib.Path("data")


# ---------------------------------------------------------------------------
# Step 1: build canonical team_id -> conf_id and conf_id -> correct name
# ---------------------------------------------------------------------------

def _build_team_conf_maps(games_path: pathlib.Path) -> tuple[dict, dict]:
    """Return (team_id_to_conf_id, conf_id_to_name) dicts from games.csv."""
    df = pd.read_csv(games_path, dtype=str)

    team_to_conf: dict[str, str] = {}

    for prefix in ("home_", "away_"):
        tid_col = f"{prefix}team_id"
        cid_col = f"{prefix}conf_id"
        if tid_col not in df.columns or cid_col not in df.columns:
            continue
        subset = df[[tid_col, cid_col]].dropna().drop_duplicates()
        for _, row in subset.iterrows():
            tid = str(row[tid_col]).strip()
            cid = str(row[cid_col]).strip().split(".")[0]  # strip ".0"
            if tid and cid and cid != "nan":
                team_to_conf[tid] = cid

    conf_id_to_name: dict[str, str] = {
        cid: ESPN_CONFERENCE_MAP.get(cid, cid)
        for cid in set(team_to_conf.values())
    }

    return team_to_conf, conf_id_to_name


def _correct_name(raw_conf_id: str | float, conf_id_to_name: dict) -> str:
    """Map a raw conf_id value (possibly float string) to the correct name."""
    cid = str(raw_conf_id).strip().split(".")[0]
    return conf_id_to_name.get(cid, ESPN_CONFERENCE_MAP.get(cid, ""))


# ---------------------------------------------------------------------------
# Step 2: patch helpers
# ---------------------------------------------------------------------------

def _patch_by_conf_id(
    df: pd.DataFrame,
    conf_id_col: str,
    conf_name_cols: list[str],
    conf_id_to_name: dict,
) -> bool:
    """
    Overwrite *conf_name_cols* using the corrected mapping from *conf_id_col*.
    Returns True if any change was made.
    """
    if conf_id_col not in df.columns:
        return False
    changed = False
    corrected = (
        df[conf_id_col]
        .astype(str)
        .apply(lambda x: x.split(".")[0] if pd.notna(x) and x not in ("nan", "None", "") else "")
        .map(conf_id_to_name)
    )
    for col in conf_name_cols:
        if col not in df.columns:
            continue
        mask = corrected.notna() & (corrected != "") & (corrected != df[col])
        if mask.any():
            df.loc[mask, col] = corrected[mask]
            changed = True
    return changed


def _patch_by_team_id(
    df: pd.DataFrame,
    team_id_col: str,
    conf_name_cols: list[str],
    team_to_conf: dict,
    conf_id_to_name: dict,
) -> bool:
    """
    Overwrite *conf_name_cols* using team_id -> conf_id -> correct name.
    Returns True if any change was made.
    """
    if team_id_col not in df.columns:
        return False
    changed = False
    corrected = (
        df[team_id_col]
        .astype(str)
        .str.split(".")
        .str[0]
        .map(team_to_conf)
        .map(lambda cid: conf_id_to_name.get(cid, "") if pd.notna(cid) else "")
    )
    for col in conf_name_cols:
        if col not in df.columns:
            continue
        mask = corrected.notna() & (corrected != "") & (corrected != df[col])
        if mask.any():
            df.loc[mask, col] = corrected[mask]
            changed = True
    return changed


# ---------------------------------------------------------------------------
# Step 3: per-file patching logic
# ---------------------------------------------------------------------------

def _fix_games(
    path: pathlib.Path,
    team_to_conf: dict,
    conf_id_to_name: dict,
) -> bool:
    """Patch conference name columns in games.csv using the conf_id columns.

    Parameters
    ----------
    path:           Path to games.csv.
    team_to_conf:   Mapping of team_id (str) -> ESPN conf_id (str).
    conf_id_to_name: Mapping of ESPN conf_id (str) -> correct conference name.

    Returns True if any value was changed and the file was rewritten.
    """
    df = pd.read_csv(path, dtype=str)
    changed = False

    for prefix in ("home_", "away_"):
        conf_id_col = f"{prefix}conf_id"
        conf_name_col = f"{prefix}conference"
        if conf_id_col in df.columns and conf_name_col in df.columns:
            changed |= _patch_by_conf_id(
                df, conf_id_col, [conf_name_col], conf_id_to_name
            )

    if "conference_name" in df.columns:
        # conference_name mirrors home_conference for the game's primary team
        if "home_conference" in df.columns:
            mask = df["home_conference"].notna() & (
                df["home_conference"] != df.get("conference_name", "")
            )
            if mask.any():
                df.loc[mask, "conference_name"] = df.loc[mask, "home_conference"]
                changed = True

    if changed:
        df.to_csv(path, index=False)
    return changed


def _fix_game_logs_style(
    path: pathlib.Path,
    conf_id_to_name: dict,
    team_to_conf: dict | None = None,
) -> bool:
    """Patch conference columns in team-game-log style files.

    These files have a direct ``conf_id`` column as well as
    ``conference``, ``conference_name``, and ``home_/away_conference``
    columns.  When ``conf_id`` is NaN for a row, the optional
    ``team_to_conf`` mapping is used as a fallback via ``team_id``.

    Parameters
    ----------
    path:            Path to the CSV file.
    conf_id_to_name: Mapping of ESPN conf_id (str) -> correct conference name.
    team_to_conf:    Optional mapping of team_id (str) -> ESPN conf_id (str),
                     used when conf_id is missing.

    Returns True if any value was changed and the file was rewritten.
    """
    df = pd.read_csv(path, dtype=str)
    changed = False

    # Direct conf_id column
    changed |= _patch_by_conf_id(
        df, "conf_id",
        ["conference", "conference_name"],
        conf_id_to_name,
    )

    # For rows where conf_id is missing, fall back to team_id lookup
    if team_to_conf and "team_id" in df.columns:
        conf_cols_to_fix = [c for c in ["conference", "conference_name"] if c in df.columns]
        if conf_cols_to_fix:
            changed |= _patch_by_team_id(
                df, "team_id", conf_cols_to_fix, team_to_conf, conf_id_to_name
            )

    # home/away conference via their conf_id columns
    for prefix in ("home_", "away_"):
        cid_col = f"{prefix}conf_id"
        tid_col = f"{prefix}team_id"
        conf_col = f"{prefix}conference"
        if cid_col in df.columns:
            changed |= _patch_by_conf_id(
                df, cid_col, [conf_col], conf_id_to_name
            )
        # Fall back to team_id if conf_id is missing
        if team_to_conf and tid_col in df.columns and conf_col in df.columns:
            changed |= _patch_by_team_id(
                df, tid_col, [conf_col], team_to_conf, conf_id_to_name
            )

    # Patch opp_conference using opponent_id -> team_id -> conf lookup.
    # This is the primary cause of conf_wins/conf_losses always being 0:
    # espn_metrics.py requires opp_conference == conference for conf_game=True.
    if team_to_conf and "opponent_id" in df.columns and "opp_conference" in df.columns:
        changed |= _patch_by_team_id(
            df, "opponent_id", ["opp_conference"], team_to_conf, conf_id_to_name
        )

    if changed:
        df.to_csv(path, index=False)
    return changed


def _fix_team_id_only(
    path: pathlib.Path,
    team_to_conf: dict,
    conf_id_to_name: dict,
    team_id_col: str = "team_id",
    conf_cols: list[str] | None = None,
) -> bool:
    """Patch conference columns in files that have team_id but no conf_id.

    Examples: ``cbb_rankings.csv``, ``team_season_summary.csv``.

    Parameters
    ----------
    path:            Path to the CSV file.
    team_to_conf:    Mapping of team_id (str) -> ESPN conf_id (str).
    conf_id_to_name: Mapping of ESPN conf_id (str) -> correct conference name.
    team_id_col:     Name of the team identifier column (default: ``team_id``).
    conf_cols:       Conference columns to patch; auto-detected from headers
                     when *None*.

    Returns True if any value was changed and the file was rewritten.
    """
    df = pd.read_csv(path, dtype=str)
    if conf_cols is None:
        conf_cols = [c for c in df.columns if "conference" in c.lower()]
    changed = _patch_by_team_id(
        df, team_id_col, conf_cols, team_to_conf, conf_id_to_name
    )
    if changed:
        df.to_csv(path, index=False)
    return changed


def _fix_predictions_style(
    path: pathlib.Path,
    team_to_conf: dict,
    conf_id_to_name: dict,
) -> bool:
    """Patch conference columns in predictions-style CSV files.

    These files use ``home_team_id`` / ``away_team_id`` to identify teams
    and store conference data in ``home_conference`` / ``away_conference``
    (and optionally ``conference_name``).

    Parameters
    ----------
    path:            Path to the CSV file.
    team_to_conf:    Mapping of team_id (str) -> ESPN conf_id (str).
    conf_id_to_name: Mapping of ESPN conf_id (str) -> correct conference name.

    Returns True if any value was changed and the file was rewritten.
    """
    df = pd.read_csv(path, dtype=str)
    changed = False

    for prefix in ("home_", "away_"):
        tid_col = f"{prefix}team_id"
        conf_col = f"{prefix}conference"
        if tid_col in df.columns and conf_col in df.columns:
            changed |= _patch_by_team_id(
                df, tid_col, [conf_col], team_to_conf, conf_id_to_name
            )

    if "conference_name" in df.columns and "home_conference" in df.columns:
        mask = df["home_conference"].notna() & (
            df["home_conference"] != df["conference_name"]
        )
        if mask.any():
            df.loc[mask, "conference_name"] = df.loc[mask, "home_conference"]
            changed = True

    if changed:
        df.to_csv(path, index=False)
    return changed


# ---------------------------------------------------------------------------
# Step 4: main dispatcher
# ---------------------------------------------------------------------------

GAME_LOG_STYLE_FILES = [
    "team_game_logs.csv",
    "team_game_metrics.csv",
    "team_game_sos.csv",
    "team_game_weighted.csv",
    "team_pretournament_snapshot.csv",
    "team_tournament_metrics.csv",
]

TEAM_ID_ONLY_FILES = [
    "cbb_rankings.csv",
    "team_season_summary.csv",
    "team_season_summary_latest.csv",
    "team_ats_profile.csv",
    "team_luck_regression.csv",
    "team_resume.csv",
    "conference_summary.csv",
    "conference_daily_summary.csv",
    "team_rolling_l5.csv",
    "team_rolling_l10.csv",
    "team_weighted_rolling.csv",
    "team_form_snapshot.csv",
    "team_situational.csv",
    "stat_rankings.csv",
    "upset_watch.csv",
]

PREDICTIONS_STYLE_FILES = [
    "predictions_latest.csv",
    "predictions_primary.csv",
    "predictions_combined_latest.csv",
    "predictions_graded.csv",
    "predictions_with_context.csv",
    "predictions_mc_latest.csv",
    "predictions_history.csv",
    "backtest_training_data.csv",
    "model_accuracy_report.csv",
    "backtest_predictions_with_context.csv",
]


def _rebuild_rankings_by_conference(
    rankings_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """Rebuild cbb_rankings_by_conference.csv from the fixed cbb_rankings.csv."""
    df = pd.read_csv(rankings_path, dtype=str)
    if "conference" not in df.columns:
        return

    numeric_cols = [
        "adj_net_rtg", "adj_ortg", "adj_drtg", "barthag",
        "cage_power_index", "resume_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "team_id" in df.columns:
        df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")

    agg_cols = {k: ["mean", "median", "max", "min"] if k == "adj_net_rtg"
                else ["mean"] for k in ["adj_net_rtg", "adj_ortg", "adj_drtg",
                                         "barthag", "cage_power_index", "resume_score"]
                if k in df.columns}
    if "barthag" in agg_cols:
        agg_cols["barthag"] = ["mean", "max"]
    if "team_id" in df.columns:
        agg_cols["team_id"] = "count"

    if not agg_cols:
        return

    conf_df = df.groupby("conference").agg(agg_cols)
    conf_df.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                       for c in conf_df.columns]
    if "team_id_count" in conf_df.columns:
        conf_df = conf_df.rename(columns={"team_id_count": "team_count"})
    sort_col = "adj_net_rtg_mean" if "adj_net_rtg_mean" in conf_df.columns else conf_df.columns[0]
    conf_df = conf_df.sort_values(sort_col, ascending=False)
    conf_df.index.name = "conference"
    result = conf_df.reset_index()
    result["conference_name"] = result["conference"]
    result.to_csv(out_path, index=False)


def fix_all(data_dir: pathlib.Path = DATA) -> None:
    games_path = data_dir / "games.csv"
    if not games_path.exists():
        print(f"ERROR: {games_path} not found — cannot build team->conf_id mapping", file=sys.stderr)
        sys.exit(1)

    team_to_conf, conf_id_to_name = _build_team_conf_maps(games_path)
    print(f"Built mapping for {len(team_to_conf)} teams across {len(conf_id_to_name)} conferences")

    fixed_count = 0

    # --- games.csv (root) ---
    changed = _fix_games(games_path, team_to_conf, conf_id_to_name)
    if changed:
        print(f"  Fixed: {games_path}")
        fixed_count += 1

    # --- game-log style (have conf_id column) ---
    for fname in GAME_LOG_STYLE_FILES:
        for candidate in [data_dir / fname, data_dir / "csv" / fname]:
            if candidate.exists():
                changed = _fix_game_logs_style(candidate, conf_id_to_name, team_to_conf)
                if changed:
                    print(f"  Fixed: {candidate}")
                    fixed_count += 1

    # --- team_id-only style ---
    for fname in TEAM_ID_ONLY_FILES:
        for candidate in [data_dir / fname, data_dir / "csv" / fname]:
            if candidate.exists():
                changed = _fix_team_id_only(candidate, team_to_conf, conf_id_to_name)
                if changed:
                    print(f"  Fixed: {candidate}")
                    fixed_count += 1

    # --- cbb_rankings snapshots (data/ root, timestamped) ---
    for snapshot in data_dir.glob("cbb_rankings_*.csv"):
        changed = _fix_team_id_only(snapshot, team_to_conf, conf_id_to_name)
        if changed:
            print(f"  Fixed: {snapshot}")
            fixed_count += 1

    # --- predictions style ---
    for fname in PREDICTIONS_STYLE_FILES:
        for candidate in [data_dir / fname, data_dir / "csv" / fname]:
            if candidate.exists():
                changed = _fix_predictions_style(candidate, team_to_conf, conf_id_to_name)
                if changed:
                    print(f"  Fixed: {candidate}")
                    fixed_count += 1

    # --- dated predictions (data/ root) ---
    for pred_file in data_dir.glob("predictions_2*.csv"):
        if pred_file.exists():
            changed = _fix_predictions_style(pred_file, team_to_conf, conf_id_to_name)
            if changed:
                print(f"  Fixed: {pred_file}")
                fixed_count += 1

    # --- cbb_rankings_by_conference (aggregated — rebuild from fixed cbb_rankings.csv) ---
    for rankings_src in [data_dir / "cbb_rankings.csv", data_dir / "csv" / "cbb_rankings.csv"]:
        if rankings_src.exists():
            conf_out = rankings_src.parent / "cbb_rankings_by_conference.csv"
            _rebuild_rankings_by_conference(rankings_src, conf_out)
            print(f"  Rebuilt: {conf_out}")
            fixed_count += 1

    print(f"\nDone. Patched {fixed_count} file(s).")


if __name__ == "__main__":
    fix_all()
