"""
espn_pipeline.py — Tournament Metrics Integration Patch
========================================================
Add the following changes to espn_pipeline.py to wire in espn_tournament.py.

1. ADD to imports section (top of file):
──────────────────────────────────────────────────────────────────────────────
from espn_tournament import compute_tournament_metrics, build_pretournament_snapshot

2. ADD to espn_config.py (or wherever your OUT_* paths are defined):
──────────────────────────────────────────────────────────────────────────────
OUT_TOURNAMENT_METRICS  = CSV_DIR / "team_tournament_metrics.csv"
OUT_TOURNAMENT_SNAPSHOT = CSV_DIR / "team_pretournament_snapshot.csv"

3. REPLACE the block in build_team_and_player_logs() that ends with:
   log.info(f"team_game_weighted.csv: {len(df_weighted_out)} total rows")

   WITH the version below (adds tournament metrics after weighted metrics):
──────────────────────────────────────────────────────────────────────────────
"""

# ── Paste this block AFTER df_weighted_out is written ────────────────────────
# Runs compute_tournament_metrics on the full weighted history, then extracts
# the pre-tournament snapshot (one row per team = their most recent game stats).

TOURNAMENT_INTEGRATION_CODE = '''
        # ── Compute tournament composite metrics ──
        # Runs on the SOS-enriched weighted output so adj_ortg, adj_drtg,
        # opp_avg_net_rtg, efg_vs_opp, etc. are all available.
        #
        # player_metrics_for_tournament is optional — if player logs exist,
        # star reliance uses actual usage/scoring distribution.
        # Otherwise falls back to box-score proxies.
        player_metrics_for_tournament = (
            pd.read_csv(OUT_PLAYER_METRICS)
            if OUT_PLAYER_METRICS.exists() and OUT_PLAYER_METRICS.stat().st_size > 0
            else pd.DataFrame()
        )

        df_tournament = compute_tournament_metrics(
            df_weighted_out,
            player_df=player_metrics_for_tournament if not player_metrics_for_tournament.empty else None,
        )
        df_tournament_out = _append_dedupe_write(
            OUT_TOURNAMENT_METRICS,
            df_tournament,
            unique_keys=["event_id", "team_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id"],
        )
        log.info(f"team_tournament_metrics.csv: {len(df_tournament_out)} total rows")

        # ── Build pre-tournament snapshot (latest row per team) ──
        # This is the primary input for matchup projections (game totals, UWS).
        # One row per team. Regenerated each run so it's always current.
        df_snapshot = build_pretournament_snapshot(df_tournament_out)
        if not DRY_RUN:
            OUT_TOURNAMENT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            df_snapshot.to_csv(OUT_TOURNAMENT_SNAPSHOT, index=False)
        log.info(f"team_pretournament_snapshot.csv: {len(df_snapshot)} teams")
'''

# ── How to use matchup projections (ad-hoc or scheduled) ─────────────────────
MATCHUP_PROJECTION_EXAMPLE = '''
"""
Example: Generate game total + UWS for a specific matchup.
Run this standalone after the pipeline has populated team_pretournament_snapshot.csv.

Usage:
    python espn_matchup_example.py
"""
import pandas as pd
from espn_tournament import build_matchup_row, compute_matchup_projections

# Load the pre-tournament snapshot built by the pipeline
snapshot = pd.read_csv("data/team_pretournament_snapshot.csv")
snapshot = snapshot.set_index("team_id")

# ── Single matchup (e.g. 5-seed vs 12-seed) ──
fav_id  = "52"      # replace with ESPN team_id for the favorite
dog_id  = "277"     # replace with ESPN team_id for the underdog

fav = snapshot.loc[fav_id]
dog = snapshot.loc[dog_id]

result = build_matchup_row(
    fav_snapshot=fav,
    dog_snapshot=dog,
    fav_seed=5,
    dog_seed=12,
    game_type="ncaa_r1",
)

print("=" * 60)
print(f"MATCHUP: {fav['team']} (#{result['fav_seed']}) vs "
      f"{dog['team']} (#{result['dog_seed']})")
print("=" * 60)
print(f"Game Total Projection : {result['game_total_projection']} pts")
print(f"Confidence Band       : ±{result['total_confidence_band']} pts")
print(f"Direction Signal      : {result['total_direction']}")
print(f"Active Flags          : {result['total_flags']}")
print()
print(f"Underdog Winner Score : {result['uws_total']} / 70")
print(f"Alert Level           : {result['uws_alert_level']}")
print(f"Upset Probability     : {result['uws_upset_probability']:.1%}")
print(f"Primary Narrative     : {result['uws_primary_narrative']}")
print()
print("UWS Component Breakdown:")
for k, v in result.items():
    if k.startswith("c") and "_" in k and isinstance(v, float):
        print(f"  {k:35s} {v:4.1f} / 10")
print()
print("Favorite Vulnerabilities:", result.get("fav_vulnerability_flags", "NONE"))


# ── Batch: all first-round matchups ──
# Build a matchup DataFrame with home_/away_ prefixed columns
# then run compute_matchup_projections() in one pass.

matchups = pd.DataFrame([
    # (fav_team_id, dog_team_id, fav_seed, dog_seed)
    ("52",  "277", 1, 16),
    ("150", "328", 5, 12),
    ("2",   "99",  4, 13),
])
matchups.columns = ["fav_team_id", "dog_team_id", "fav_seed", "dog_seed"]

# Join stats with prefixes
fav_stats = snapshot.add_prefix("fav_").reset_index().rename(
    columns={"team_id": "fav_team_id"}
)
dog_stats = snapshot.add_prefix("dog_").reset_index().rename(
    columns={"team_id": "dog_team_id"}
)
matchup_full = (
    matchups
    .merge(fav_stats, on="fav_team_id", how="left")
    .merge(dog_stats, on="dog_team_id", how="left")
)

results_df = compute_matchup_projections(
    matchup_full,
    game_type="ncaa_r1",
    favorite_col_prefix="fav_",
    underdog_col_prefix="dog_",
)

# Print summary table
summary_cols = [
    "fav_team", "dog_team", "fav_seed", "dog_seed",
    "game_total_projection", "total_direction",
    "uws_total", "uws_alert_level", "uws_upset_probability",
]
available = [c for c in summary_cols if c in results_df.columns]
print(results_df[available].to_string(index=False))
'''

if __name__ == "__main__":
    # Demonstrate what the integration looks like without running anything
    print("espn_tournament_integration.py — reference file only.")
    print("Copy TOURNAMENT_INTEGRATION_CODE into espn_pipeline.py as documented above.")
    print("Run MATCHUP_PROJECTION_EXAMPLE as espn_matchup_example.py after pipeline runs.")
