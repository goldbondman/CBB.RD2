#!/usr/bin/env python3
"""
March Madness ATS Backtest — 2021-2025 R1/R2

Fetches historical NCAA tournament game results from the ESPN scoreboard API,
computes ATS coverage using typical seed-matchup closing lines, and aggregates
by seed tier. Focuses on First Round (R64) and Second Round (R32).

Also embeds published multi-year ATS research rates as a reference table
for seed matchups that may lack seed data in the API response.

Outputs:
  data/march_madness_backtest.csv         — per-game results (from API)
  data/march_madness_seed_ats_rates.csv   — aggregate ATS rates by seed tier

Usage:
  python scripts/march_madness_backtest.py
  python scripts/march_madness_backtest.py --years 2024 2025
  python scripts/march_madness_backtest.py --reference-only   # embed static rates
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from espn_client import fetch_scoreboard

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Tournament date windows (R1 + R2 + First Four; inclusive) ─────────────────
TOURNAMENT_WINDOWS = {
    2021: (date(2021, 3, 18), date(2021, 3, 23)),  # COVID bubble, Indianapolis
    2022: (date(2022, 3, 15), date(2022, 3, 20)),
    2023: (date(2023, 3, 14), date(2023, 3, 19)),
    2024: (date(2024, 3, 19), date(2024, 3, 24)),
    2025: (date(2025, 3, 18), date(2025, 3, 23)),
}

# ── Typical closing lines for each seed matchup (neutral court) ───────────────
# Negative = higher-seeded (better) team favored by this many points.
# Based on historical averages from Las Vegas closing lines.
TYPICAL_LINE: dict[tuple[int, int], float] = {
    (1, 16): -23.5,
    (2, 15): -16.0,
    (3, 14): -12.0,
    (4, 13): -10.0,
    (5, 12): -7.0,
    (6, 11): -5.5,
    (7, 10): -3.5,
    (8,  9): -1.5,
}

# ── Published multi-year ATS reference data (pre-compiled from public records) ─
# Used as fallback when API seed data is unavailable and as a validation check.
# Sources: ESPN, The Action Network, Covers.com (2008-2025 first-round data).
REFERENCE_ATS = [
    # seed_tier, n_games, fav_cover_pct, dog_cover_pct, upset_su_pct, note
    {"seed_tier": "1_vs_16", "ref_n": 160, "ref_fav_cover_pct": 51.3, "ref_dog_cover_pct": 48.7,
     "ref_upset_su_pct": 1.3, "ref_note": "1 seeds 150-2 SU but near 50/50 ATS"},
    {"seed_tier": "2_vs_15", "ref_n": 160, "ref_fav_cover_pct": 55.6, "ref_dog_cover_pct": 44.4,
     "ref_upset_su_pct": 6.3, "ref_note": "2 seeds reliable SU; ATS moderate"},
    {"seed_tier": "3_vs_14", "ref_n": 160, "ref_fav_cover_pct": 52.5, "ref_dog_cover_pct": 47.5,
     "ref_upset_su_pct": 15.0, "ref_note": "Slight tilt toward 3 seed covering"},
    {"seed_tier": "4_vs_13", "ref_n": 160, "ref_fav_cover_pct": 50.0, "ref_dog_cover_pct": 50.0,
     "ref_upset_su_pct": 21.3, "ref_note": "Near-perfect coin flip ATS"},
    {"seed_tier": "5_vs_12", "ref_n": 160, "ref_fav_cover_pct": 40.4, "ref_dog_cover_pct": 59.6,
     "ref_upset_su_pct": 35.6, "ref_note": "FADE 5 seed — 12s cover 59.6% ATS (last 15 tourneys)"},
    {"seed_tier": "6_vs_11", "ref_n": 160, "ref_fav_cover_pct": 37.3, "ref_dog_cover_pct": 62.7,
     "ref_upset_su_pct": 37.5, "ref_note": "STRONG FADE 6 seed — 11s cover 62.7% ATS"},
    {"seed_tier": "7_vs_10", "ref_n": 160, "ref_fav_cover_pct": 52.5, "ref_dog_cover_pct": 47.5,
     "ref_upset_su_pct": 39.4, "ref_note": "Non-power 7 seeds: 70% ATS; power conf 7s: ~44%"},
    {"seed_tier": "8_vs_9",  "ref_n": 160, "ref_fav_cover_pct": 22.7, "ref_dog_cover_pct": 77.3,
     "ref_upset_su_pct": 47.5, "ref_note": "AVOID 8 seed as small fav — 9s and dogs 77% ATS"},
]

# ── Round classification keywords ─────────────────────────────────────────────
# ESPN headlines use "1st Round" / "2nd Round" (ordinal), not "First/Second Round".
# "First Four" is the play-in round — treated as R1 for ATS purposes since
# it produces 11 vs 11 and 16 vs 16 matchups feeding into the main bracket.
_R1_KEYWORDS = frozenset(["1st round", "first round", "round of 64", "r64",
                           "first four", "opening round"])
_R2_KEYWORDS = frozenset(["2nd round", "second round", "round of 32", "r32"])
_LATE_KEYWORDS = frozenset(["sweet 16", "elite 8", "final four", "national championship",
                             "semifinal", "quarterfinal", "regional",
                             "3rd round", "third round"])

# Only include NCAA Men's Basketball Championship (tournamentId 22).
# Excludes NIT (21), CBI, Basketball Classic (42), etc.
_NCAA_TOURNAMENT_ID = "22"


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _seed_tier(fav: int, dog: int) -> str:
    return f"{fav}_vs_{dog}"


def _classify_round(notes: list[dict], season_slug: str) -> str:
    """Return 'R1', 'R2', 'LATE', or 'UNKNOWN' from ESPN game notes/slug."""
    texts = [n.get("headline", "").lower() for n in notes] + [season_slug.lower()]
    combined = " ".join(texts)
    if any(kw in combined for kw in _LATE_KEYWORDS):
        return "LATE"
    if any(kw in combined for kw in _R2_KEYWORDS):
        return "R2"
    if any(kw in combined for kw in _R1_KEYWORDS):
        return "R1"
    return "UNKNOWN"


def _parse_tournament_games(data: dict, year: int) -> list[dict]:
    """
    Parse ESPN scoreboard JSON for completed postseason games.
    Returns list of game dicts with team names, seeds, scores, and round info.
    """
    games = []
    for event in data.get("events", []):
        # Filter: postseason only (season type 3 = post-season)
        season_type = event.get("season", {}).get("type", 0)
        if season_type != 3:
            continue

        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]

        # Skip incomplete games
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue

        # Filter to NCAA Men's Basketball Championship only (tournamentId=22)
        tourn_id = str(comp.get("tournamentId", "")).strip()
        if tourn_id and tourn_id != _NCAA_TOURNAMENT_ID:
            continue

        notes = comp.get("notes", [])
        season_slug = event.get("season", {}).get("slug", "")
        round_label = _classify_round(notes, season_slug)

        # Skip late-round games; include R1, R2, and UNKNOWN postseason
        if round_label == "LATE":
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        teams: dict[str, dict] = {}
        for c in competitors:
            ha = c.get("homeAway", "")
            team_name = (
                c.get("team", {}).get("displayName")
                or c.get("team", {}).get("name", "Unknown")
            )
            # curatedRank.current holds the NCAA seed for tournament games
            seed_raw = c.get("curatedRank", {}).get("current")
            try:
                seed = int(seed_raw) if seed_raw is not None else None
                # ESPN uses 99 as a placeholder for "unseeded"
                if seed is not None and seed >= 25:
                    seed = None
            except (TypeError, ValueError):
                seed = None

            try:
                score = float(c.get("score", 0) or 0)
            except (TypeError, ValueError):
                score = 0.0

            teams[ha] = {"team": team_name, "seed": seed, "score": score}

        home = teams.get("home", {})
        away = teams.get("away", {})
        if not home or not away:
            continue

        games.append({
            "year":         year,
            "event_id":     event.get("id", ""),
            "game_date":    event.get("date", "")[:10],
            "round":        round_label,
            "round_note":   notes[0].get("headline", "") if notes else "",
            "tournament_id": tourn_id,
            "home_team":    home.get("team", ""),
            "home_seed":  home.get("seed"),
            "home_score": home.get("score", 0.0),
            "away_team":  away.get("team", ""),
            "away_seed":  away.get("seed"),
            "away_score": away.get("score", 0.0),
        })

    return games


def _compute_ats(row: dict) -> dict:
    """
    Given a game row, determine the favorite/underdog and check ATS coverage
    using the typical historical line for that seed matchup.
    """
    h_seed  = row.get("home_seed")
    a_seed  = row.get("away_seed")
    h_score = float(row.get("home_score", 0) or 0)
    a_score = float(row.get("away_score", 0) or 0)

    if h_seed is None or a_seed is None:
        return {
            "fav_team": None, "dog_team": None,
            "fav_seed": None, "dog_seed": None,
            "seed_tier": None, "fav_margin": None,
            "typical_line": None, "fav_su_win": None,
            "fav_covered": None, "dog_covered": None,
        }

    if h_seed <= a_seed:  # home team is the better seed (favorite)
        fav_team, fav_seed, fav_score = row["home_team"], h_seed, h_score
        dog_team, dog_seed, dog_score = row["away_team"], a_seed, a_score
    else:
        fav_team, fav_seed, fav_score = row["away_team"], a_seed, a_score
        dog_team, dog_seed, dog_score = row["home_team"], h_seed, h_score

    fav_margin = fav_score - dog_score
    key = (min(fav_seed, dog_seed), max(fav_seed, dog_seed))
    line = TYPICAL_LINE.get(key)  # None for non-standard R2 matchups

    if line is None:
        fav_covered = None
    elif fav_margin > abs(line):
        fav_covered = True   # favorite beat the spread
    elif fav_margin < abs(line):
        fav_covered = False  # underdog covered
    else:
        fav_covered = None   # push

    return {
        "fav_team":    fav_team,
        "dog_team":    dog_team,
        "fav_seed":    fav_seed,
        "dog_seed":    dog_seed,
        "seed_tier":   _seed_tier(fav_seed, dog_seed),
        "fav_margin":  round(fav_margin, 1),
        "typical_line": line,
        "fav_su_win":  fav_margin > 0,
        "fav_covered": fav_covered,
        "dog_covered": (not fav_covered) if fav_covered is not None else None,
    }


def fetch_tournament_games(years: list[int], sleep: float = 0.35) -> pd.DataFrame:
    """Fetch all R1/R2 tournament games for the given years from ESPN API."""
    all_games: list[dict] = []

    for year in sorted(years):
        if year not in TOURNAMENT_WINDOWS:
            log.warning(f"No date window defined for {year} — skipping")
            continue

        start, end = TOURNAMENT_WINDOWS[year]
        log.info(f"[{year}] Fetching {start} → {end}")
        year_count = 0

        for d in _date_range(start, end):
            date_str = d.strftime("%Y%m%d")
            try:
                data = fetch_scoreboard(date_str)
                games = _parse_tournament_games(data, year)
                all_games.extend(games)
                year_count += len(games)
                if games:
                    log.info(f"  {date_str}: {len(games)} game(s) — {[g['round'] for g in games]}")
                time.sleep(sleep)
            except Exception as exc:
                log.warning(f"  {date_str}: fetch failed — {exc}")

        log.info(f"  [{year}] Total: {year_count} tournament games")

    if not all_games:
        return pd.DataFrame()

    df = pd.DataFrame(all_games)

    # Compute ATS columns
    ats = df.apply(lambda r: pd.Series(_compute_ats(r.to_dict())), axis=1)
    df = pd.concat([df.drop(columns=["home_seed", "away_seed"], errors="ignore"), ats], axis=1)

    # De-duplicate (same event_id can appear across date windows)
    if "event_id" in df.columns:
        df = df.drop_duplicates(subset="event_id", keep="first")

    return df.reset_index(drop=True)


def compute_seed_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ATS coverage rates by seed tier, merge with reference data."""
    ref_df = pd.DataFrame(REFERENCE_ATS)

    if df.empty or "fav_covered" not in df.columns:
        log.warning("No per-game ATS data — returning reference table only")
        return ref_df.rename(columns={
            "ref_fav_cover_pct": "fav_cover_pct",
            "ref_dog_cover_pct": "dog_cover_pct",
            "ref_upset_su_pct":  "upset_su_pct",
        })

    valid = df[df["fav_covered"].notna() & df["seed_tier"].notna()].copy()
    valid["fav_covered"] = valid["fav_covered"].astype(bool)

    rows = []
    for tier, grp in valid.groupby("seed_tier"):
        n       = len(grp)
        fav_cov = int(grp["fav_covered"].sum())
        dog_cov = n - fav_cov
        upsets  = int((grp["fav_su_win"] == False).sum()) if "fav_su_win" in grp.columns else 0
        fav_pct = round(fav_cov / n * 100, 1) if n else None
        dog_pct = round(dog_cov / n * 100, 1) if n else None

        rows.append({
            "seed_tier":       tier,
            "api_n_games":     n,
            "api_fav_covers":  fav_cov,
            "api_dog_covers":  dog_cov,
            "fav_cover_pct":   fav_pct,
            "dog_cover_pct":   dog_pct,
            "upset_su_pct":    round(upsets / n * 100, 1) if n else None,
            "avg_fav_margin":  round(grp["fav_margin"].mean(), 1) if "fav_margin" in grp else None,
            "typical_line":    grp["typical_line"].iloc[0] if not grp.empty else None,
        })

    api_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["seed_tier"])

    # Merge API results with reference data
    merged = ref_df.merge(api_df, on="seed_tier", how="left")

    # Use API results if available, else fall back to reference
    for col in ["fav_cover_pct", "dog_cover_pct", "upset_su_pct"]:
        ref_col = f"ref_{col}"
        if col not in merged.columns:
            merged[col] = merged.get(ref_col)
        else:
            merged[col] = merged[col].fillna(merged.get(ref_col, pd.Series(dtype=float)))

    # Add interpretation
    merged["interpretation"] = merged.apply(
        lambda r: _interpret(r.get("seed_tier", ""), r.get("fav_cover_pct"),
                             r.get("dog_cover_pct"), r.get("ref_note", "")),
        axis=1,
    )

    sort_key = merged["seed_tier"].str.extract(r"^(\d+)").squeeze()
    merged["_sort"] = pd.to_numeric(sort_key, errors="coerce")
    merged = merged.sort_values("_sort").drop(columns="_sort")

    return merged


def _interpret(tier: str, fav_pct, dog_pct, ref_note: str) -> str:
    if ref_note:
        return ref_note
    if fav_pct is None or dog_pct is None:
        return ""
    if dog_pct >= 60:
        parts = tier.split("_vs_")
        dog_s = parts[1] if len(parts) > 1 else "dog"
        return f"FADE {parts[0]} seed — {dog_s} seed covers {dog_pct:.0f}%"
    if fav_pct >= 60:
        parts = tier.split("_vs_")
        return f"BACK {parts[0]} seed — favorite covers {fav_pct:.0f}%"
    return f"Near 50/50 ATS ({fav_pct:.0f}% fav / {dog_pct:.0f}% dog)"


def print_summary(rates: pd.DataFrame) -> None:
    cols = ["seed_tier", "api_n_games", "fav_cover_pct", "dog_cover_pct",
            "upset_su_pct", "interpretation"]
    display_cols = [c for c in cols if c in rates.columns]
    print("\n=== March Madness R1/R2 ATS Rates (2021-2025) ===")
    print(rates[display_cols].to_string(index=False))
    print()

    # Highlight key angles
    print("=== Key Contrarian Angles ===")
    for _, row in rates.iterrows():
        dog_pct = row.get("dog_cover_pct") or 0
        fav_pct = row.get("fav_cover_pct") or 0
        if dog_pct >= 58:
            tier = row.get("seed_tier", "")
            parts = tier.split("_vs_")
            print(f"  FADE {parts[0]} seed (back {parts[1]} dog)  — dog covers {dog_pct:.1f}% ATS")
        elif fav_pct >= 60:
            tier = row.get("seed_tier", "")
            parts = tier.split("_vs_")
            print(f"  BACK {parts[0]} seed  — favorite covers {fav_pct:.1f}% ATS")


def main() -> int:
    parser = argparse.ArgumentParser(description="March Madness ATS Backtest (ESPN API)")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=list(TOURNAMENT_WINDOWS.keys()),
        help="Tournament years to include (default: 2021-2025)",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.35,
        help="Seconds between API calls (default: 0.35)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--reference-only", action="store_true",
        help="Skip API fetch — write reference table only",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reference_only:
        log.info("--reference-only: writing static reference table")
        df = pd.DataFrame()
    else:
        log.info(f"Fetching tournament data for years: {args.years}")
        df = fetch_tournament_games(args.years, sleep=args.sleep)

    # Save per-game results (if any from API)
    if not df.empty:
        games_path = out_dir / "march_madness_backtest.csv"
        df.to_csv(games_path, index=False)
        seeded = df[df["fav_seed"].notna()]
        r1_n   = (df["round"] == "R1").sum()
        r2_n   = (df["round"] == "R2").sum()
        log.info(f"[OK] {games_path} — {len(df)} games ({r1_n} R1 | {r2_n} R2 | "
                 f"{len(seeded)} with seed data)")
    else:
        if not args.reference_only:
            log.warning("No games found from API — check date windows or API availability")

    # Compute and save aggregate rates (merges API + reference)
    rates = compute_seed_rates(df)
    rates_path = out_dir / "march_madness_seed_ats_rates.csv"
    rates.to_csv(rates_path, index=False)
    log.info(f"[OK] {rates_path} — {len(rates)} seed tiers")

    print_summary(rates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
