import requests
import re
import json
import time
import random
import logging
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

# Ensure repo-root imports work when invoked as "python ingestion/historical_backfill.py".
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.id_utils import canonicalize_espn_game_id
from pipeline.team_utils import TeamCanonicalizer, normalize_team_name, slugify

try:
    from ingestion.wagertalk_scraper import wagertalk_candidates_for_date
except Exception:  # pragma: no cover - optional source adapter
    wagertalk_candidates_for_date = None

log = logging.getLogger(__name__)
# Standard log format for CI/CD readability
logging.basicConfig(level=logging.INFO, format="[DIAG] HistoricalBackfill | %(levelname)-8s | %(message)s")

@dataclass
class CandidateLine:
    provider_name: str
    provider_game_date: str # YYYY-MM-DD
    provider_home_team_raw: str
    provider_away_team_raw: str
    candidate_home_spread: float
    candidate_total: float
    provider_tip_time: Optional[str] = None # HH:MM (UTC if possible)
    provider_home_slug_norm: str = ""
    provider_away_slug_norm: str = ""
    candidate_is_closing: bool = False
    url: str = ""
    retrieval_timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    espn_game_id: Optional[str] = None # For live_capture

class Throttler:
    """Shared throttler for all source adapters with jitter and exponential backoff."""
    def __init__(self, rps_limit: float = 0.5):
        self.rps_limit = rps_limit
        self.last_request: Dict[str, float] = {}

    def wait(self, domain: str):
        now = time.time()
        if domain in self.last_request:
            elapsed = now - self.last_request[domain]
            # RPS limit 0.5 = 1 request per 2 seconds
            wait_time = (1.0 / self.rps_limit) - elapsed
            if wait_time > 0:
                time.sleep(wait_time + (random.random() * 0.5))  # add jitter
        self.last_request[domain] = time.time()

class HistoricalBackfill:
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.cache_path = data_dir / "historical_market_lines_cache.csv"
        self.closing_path = data_dir / "market_lines_closing.csv"
        self.unresolved_path = data_dir / "unresolved_games.csv"
        self.audit_path = data_dir / "market_line_backfill_audit.csv"
        self.wagertalk_path = data_dir / "wagertalk_historical_odds.csv"
        self.throttler = Throttler(rps_limit=0.5) # 1 req every 2 seconds
        self.canonicalizer = TeamCanonicalizer()
        self.url_cache: Dict[str, str] = {}
        self.request_count = 0
        self.max_requests_per_run = 5000

        # Load cache
        if self.cache_path.exists():
            self.cache_df = pd.read_csv(self.cache_path, dtype={"espn_game_id": str})
        else:
            self.cache_df = pd.DataFrame(columns=["espn_game_id"])

        # Load market_lines.csv (live_capture source)
        self.live_capture_df = pd.DataFrame()
        ml_path = data_dir / "market_lines.csv"
        if ml_path.exists():
            try:
                self.live_capture_df = pd.read_csv(ml_path, dtype={"event_id": str})
                self.live_capture_df["event_id"] = self.live_capture_df["event_id"].apply(canonicalize_espn_game_id)
            except Exception as e:
                log.warning("Could not load market_lines.csv: %s", e)

    def fetch_url(self, url: str, domain: str) -> Optional[str]:
        if url in self.url_cache:
            return self.url_cache[url]

        if self.request_count >= self.max_requests_per_run:
            log.warning("Max requests reached for this run. Skipping %s", url)
            return None

        self.throttler.wait(domain)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        }

        backoff = 10.0
        for attempt in range(3):
            try:
                self.request_count += 1
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    self.url_cache[url] = resp.text
                    return resp.text
                elif resp.status_code in {429, 500, 502, 503, 504}:
                    log.warning("Received status %s for %s, attempt %s. Backing off %ss...",
                                resp.status_code, url, attempt+1, backoff)
                    time.sleep(backoff + (random.random() * 5))
                    backoff *= 2
                else:
                    log.error("Failed to fetch %s, status %s", url, resp.status_code)
                    break
            except Exception as e:
                log.error("Error fetching %s: %s", url, e)
                time.sleep(backoff)
                backoff *= 2
        return None

    def get_team_rankings_candidates(self, game_date: date) -> List[CandidateLine]:
        """Layer 1: TeamRankings odds history parsing."""
        date_str = game_date.strftime("%Y-%m-%d")
        url = f"https://www.teamrankings.com/ncaa-basketball/odds-history/results/{date_str}"
        domain = "teamrankings.com"

        html = self.fetch_url(url, domain)
        if not html: return []

        candidates = []
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="datatable")
        if not table: return []

        rows = table.find_all("tr")[1:]  # skip header
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 5: continue

            # col 0: Away @ Home
            # col 2: Line (e.g. -4.5 or 4.5)
            # col 3: Total (e.g. 145.5)

            matchup_text = cols[0].text.strip()
            if "@" not in matchup_text: continue
            away_raw, home_raw = [t.strip() for t in matchup_text.split("@", 1)]

            try:
                line_cell = cols[2].text.strip()
                total_cell = cols[3].text.strip()

                # Simple parsing for spread. TeamRankings is favorite-based usually.
                # If Home is favorite, spread is negative.
                spread = 0.0
                if line_cell and line_cell != "PK":
                    spread = float(line_cell)

                total = float(total_cell) if total_cell else 0.0

                candidates.append(CandidateLine(
                    provider_name="teamrankings",
                    provider_game_date=date_str,
                    provider_home_team_raw=home_raw,
                    provider_away_team_raw=away_raw,
                    candidate_home_spread=spread,
                    candidate_total=total,
                    candidate_is_closing=True,
                    url=url,
                    provider_home_slug_norm=slugify(self.canonicalizer.get_canonical_name(home_raw, "teamrankings")),
                    provider_away_slug_norm=slugify(self.canonicalizer.get_canonical_name(away_raw, "teamrankings"))
                ))
            except (ValueError, IndexError):
                continue

        return candidates

    def get_vegas_insider_candidates(self, game_date: date) -> List[CandidateLine]:
        """Layer 1: VegasInsider matchups parsing."""
        date_str = game_date.strftime("%m-%d-%Y") # VegasInsider often uses MM-DD-YYYY
        url = f"https://www.vegasinsider.com/college-basketball/matchups/?date={date_str}"
        domain = "vegasinsider.com"

        html = self.fetch_url(url, domain)
        if not html: return []

        candidates = []
        soup = BeautifulSoup(html, "html.parser")
        # VegasInsider structure: each game is in a matchup-container or similar
        containers = soup.find_all("div", class_="matchup-container")
        for ct in containers:
            try:
                teams = ct.find_all("a", class_="team-name")
                if len(teams) < 2: continue
                away_raw = teams[0].text.strip()
                home_raw = teams[1].text.strip()

                # Odds are usually in a table inside the container
                # Need specific parsing for closing spread/total
                odds_table = ct.find("table")
                if not odds_table: continue

                # VegasInsider layout changes often, this is a placeholder for specific logic
                # Assume we find spread and total
                spread = 0.0
                total = 0.0

                candidates.append(CandidateLine(
                    provider_name="vegasinsider",
                    provider_game_date=game_date.strftime("%Y-%m-%d"),
                    provider_home_team_raw=home_raw,
                    provider_away_team_raw=away_raw,
                    candidate_home_spread=spread,
                    candidate_total=total,
                    candidate_is_closing=True,
                    url=url,
                    provider_home_slug_norm=slugify(self.canonicalizer.get_canonical_name(home_raw, "vegasinsider")),
                    provider_away_slug_norm=slugify(self.canonicalizer.get_canonical_name(away_raw, "vegasinsider"))
                ))
            except Exception:
                continue

        return candidates

    def get_wagertalk_candidates(self, game_date: date) -> List[CandidateLine]:
        """Layer 1: WagerTalk historical lines from local cache CSV."""
        if wagertalk_candidates_for_date is None:
            return []
        if not self.wagertalk_path.exists():
            return []

        try:
            raw_candidates = wagertalk_candidates_for_date(
                game_date=game_date,
                data_dir=self.data_dir,
                cache_path=self.wagertalk_path,
            )
        except Exception as e:
            log.warning("WagerTalk adapter failed for %s: %s", game_date, e)
            return []

        candidates: List[CandidateLine] = []
        for raw in raw_candidates:
            home_raw = str(raw.get("provider_home_team_raw", "")).strip()
            away_raw = str(raw.get("provider_away_team_raw", "")).strip()
            if not home_raw or not away_raw:
                continue
            try:
                spread = float(raw.get("candidate_home_spread"))
                total = float(raw.get("candidate_total"))
            except (TypeError, ValueError):
                continue

            candidates.append(CandidateLine(
                provider_name="wagertalk",
                provider_game_date=raw.get("provider_game_date", game_date.strftime("%Y-%m-%d")),
                provider_home_team_raw=home_raw,
                provider_away_team_raw=away_raw,
                candidate_home_spread=spread,
                candidate_total=total,
                candidate_is_closing=bool(raw.get("candidate_is_closing", True)),
                url=str(raw.get("url", "")),
                provider_home_slug_norm=slugify(self.canonicalizer.get_canonical_name(home_raw, "wagertalk")),
                provider_away_slug_norm=slugify(self.canonicalizer.get_canonical_name(away_raw, "wagertalk")),
            ))

        return candidates

    def get_live_capture_candidates(self, espn_game_id: str) -> List[CandidateLine]:
        """Treats existing market_lines.csv as a candidate provider."""
        if self.live_capture_df.empty: return []

        rows = self.live_capture_df[self.live_capture_df["event_id"] == espn_game_id]
        if rows.empty: return []

        # Prefer 'closing' then 'pregame' then 'opening'
        for capture_type in ["closing", "pregame", "opening"]:
            type_rows = rows[rows["capture_type"] == capture_type]
            if not type_rows.empty:
                latest = type_rows.sort_values("captured_at_utc").iloc[-1]

                # Check for spread and total
                spread = latest.get("home_spread_current")
                if pd.isna(spread): spread = latest.get("home_spread_open")

                total = latest.get("total_current")
                if pd.isna(total): total = latest.get("total_open")

                if pd.notna(spread) and pd.notna(total):
                    return [CandidateLine(
                        provider_name="live_capture",
                        provider_game_date=str(latest.get("captured_at_utc"))[:10],
                        provider_home_team_raw=str(latest.get("home_team_name", "")),
                        provider_away_team_raw=str(latest.get("away_team_name", "")),
                        candidate_home_spread=float(spread),
                        candidate_total=float(total),
                        candidate_is_closing=(capture_type == "closing"),
                        espn_game_id=espn_game_id,
                        provider_home_slug_norm=slugify(self.canonicalizer.get_canonical_name(str(latest.get("home_team_name", "")), "live_capture")),
                        provider_away_slug_norm=slugify(self.canonicalizer.get_canonical_name(str(latest.get("away_team_name", "")), "live_capture"))
                    )]
        return []

    def compute_confidence(self, espn_game: Dict, candidate: CandidateLine) -> float:
        """Weighted matching engine."""
        # Hard check for live_capture with matching ID
        if candidate.provider_name == "live_capture" and candidate.espn_game_id == espn_game["espn_game_id"]:
            return 1.0

        score = 0.0
        espn_home_slug = slugify(self.canonicalizer.get_canonical_name(espn_game["home_team"], "espn"))
        espn_away_slug = slugify(self.canonicalizer.get_canonical_name(espn_game["away_team"], "espn"))
        espn_date = espn_game["game_date"]

        # 1. Home slug match (exact)
        if espn_home_slug == candidate.provider_home_slug_norm:
            score += 0.30
        # 2. Away slug match (exact)
        if espn_away_slug == candidate.provider_away_slug_norm:
            score += 0.30
        # 3. Date match
        if espn_date == candidate.provider_game_date:
            score += 0.15
        # 4. Neutral/Venue match (if known, stubbed)
        # score += 0.10

        # 5. Partial matches / Inversion check
        if (espn_home_slug == candidate.provider_away_slug_norm and
            espn_away_slug == candidate.provider_home_slug_norm):
            return -1.0 # Inversion signal

        return score

    def resolve_candidates_for_game(self, espn_game: Dict, all_candidates: List[CandidateLine]) -> Dict:
        """Consensus and tie-breaking logic."""
        matched = []
        audit_rows = []

        # Include live capture specifically for this game
        live_cands = self.get_live_capture_candidates(espn_game["espn_game_id"])

        for cand in all_candidates + live_cands:
            conf = self.compute_confidence(espn_game, cand)
            inversion = False

            if conf == -1.0:
                # Test inverted
                inverted_cand = CandidateLine(**asdict(cand))
                inverted_cand.provider_home_slug_norm, inverted_cand.provider_away_slug_norm = cand.provider_away_slug_norm, cand.provider_home_slug_norm
                inverted_cand.candidate_home_spread = -cand.candidate_home_spread
                conf = self.compute_confidence(espn_game, inverted_cand)
                if conf >= 0.70:
                    cand = inverted_cand
                    inversion = True

            audit_rows.append({
                "espn_game_id": espn_game["espn_game_id"],
                "provider": cand.provider_name,
                "confidence": conf,
                "inversion_flag": inversion,
                "spread": cand.candidate_home_spread,
                "total": cand.candidate_total
            })

            if conf >= 0.70:
                if not (80 <= cand.candidate_total <= 200): continue
                if abs(cand.candidate_home_spread) > 60: continue
                matched.append(cand)

        if not matched:
            return {"audit": audit_rows, "resolved": None}

        # Consensus: median
        res_spread = np.median([c.candidate_home_spread for c in matched])
        res_total = np.median([c.candidate_total for c in matched])

        # Filter outliers
        final_matched = [c for c in matched if abs(c.candidate_home_spread - res_spread) <= 2.5 and abs(c.candidate_total - res_total) <= 3.5]

        if not final_matched:
            return {"audit": audit_rows, "resolved": None}

        res_spread = np.median([c.candidate_home_spread for c in final_matched])
        res_total = np.median([c.candidate_total for c in final_matched])

        # Choice: Tie-break by (highest confidence, then live_capture preference, then closest to median)
        def sort_key(c):
            # Highest confidence first
            conf = next(a["confidence"] for a in audit_rows if a["provider"] == c.provider_name and abs(a["spread"] - c.candidate_home_spread) < 0.01)
            is_live = 1 if c.provider_name == "live_capture" else 0
            dist = -abs(c.candidate_home_spread - res_spread)
            return (conf, is_live, dist)

        best_cand = max(final_matched, key=sort_key)

        # Get inversion flag from the best candidate's audit entry
        best_audit = next((a for a in audit_rows if a["provider"] == best_cand.provider_name and abs(a["spread"] - best_cand.candidate_home_spread) < 0.01), {})

        resolved = {
            "espn_game_id": espn_game["espn_game_id"],
            "game_date": espn_game["game_date"],
            "tip_time_utc": espn_game.get("tip_time_utc"),
            "home_team_id": espn_game.get("home_team_id"),
            "away_team_id": espn_game.get("away_team_id"),
            "normalized_home_slug": slugify(self.canonicalizer.get_canonical_name(espn_game["home_team"], "espn")),
            "normalized_away_slug": slugify(self.canonicalizer.get_canonical_name(espn_game["away_team"], "espn")),
            "close_home_spread": res_spread,
            "close_total": res_total,
            "close_source_primary": best_cand.provider_name,
            "close_source_list": "|".join(set(c.provider_name for c in final_matched)),
            "close_candidate_count": len(final_matched),
            "line_match_confidence_score": max(r["confidence"] for r in audit_rows),
            "match_method": "consensus_median",
            "line_inversion_flag": best_audit.get("inversion_flag", False),
            "validation_flags": "none",
            "created_at_utc": datetime.now(timezone.utc).isoformat()
        }

        return {"audit": audit_rows, "resolved": resolved}

    def process_backfill(self, espn_games: List[Dict]):
        """Main execution flow with caching."""
        all_resolved = []
        all_audit = []
        unresolved = []

        games_by_date = {}
        for g in espn_games:
            dt = g["game_date"]
            if dt not in games_by_date: games_by_date[dt] = []
            games_by_date[dt].append(g)

        for g_date_str, games in sorted(games_by_date.items()):
            g_date = datetime.strptime(g_date_str, "%Y-%m-%d").date()
            log.info("Processing %d games for %s", len(games), g_date_str)

            # Fetch candidates once per date
            candidates = []
            candidates.extend(self.get_wagertalk_candidates(g_date))
            candidates.extend(self.get_team_rankings_candidates(g_date))
            # candidates.extend(self.get_vegas_insider_candidates(g_date))

            for g in games:
                # Check cache
                cached = self.cache_df[self.cache_df["espn_game_id"] == g["espn_game_id"]]
                if not cached.empty:
                    all_resolved.append(cached.iloc[0].to_dict())
                    continue

                res_package = self.resolve_candidates_for_game(g, candidates)
                all_audit.extend(res_package["audit"])

                if res_package["resolved"]:
                    all_resolved.append(res_package["resolved"])
                    # Append to cache
                    new_cache_row = pd.DataFrame([res_package["resolved"]])
                    self.cache_df = pd.concat([self.cache_df, new_cache_row], ignore_index=True)
                    new_cache_row.to_csv(self.cache_path, mode='a', header=not self.cache_path.exists(), index=False)
                else:
                    unresolved.append(g)

        # Write resolved lines
        if all_resolved:
            pd.DataFrame(all_resolved).to_csv(self.closing_path, index=False)
            log.info("✓ Wrote %d resolved lines to %s", len(all_resolved), self.closing_path)

        if all_audit:
            pd.DataFrame(all_audit).to_csv(self.audit_path, index=False)

        if unresolved:
            pd.DataFrame(unresolved).to_csv(self.unresolved_path, index=False)
            log.warning("! Failed to resolve %d games", len(unresolved))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    # Load games from games.csv
    try:
        from espn_config import OUT_GAMES
        games_df = pd.read_csv(OUT_GAMES, dtype={"game_id": str})
        games_df["espn_game_id"] = games_df["game_id"].apply(canonicalize_espn_game_id)

        # Filter games.csv to requested range
        if args.start_date:
            games_df = games_df[games_df["date"].astype(str) >= args.start_date]
        if args.end_date:
            games_df = games_df[games_df["date"].astype(str) <= args.end_date]
        elif args.days_back:
            cutoff = (date.today() - timedelta(days=args.days_back)).strftime("%Y%m%d")
            games_df = games_df[games_df["date"].astype(str) >= cutoff]

        # Convert to list of dicts for processor
        espn_games = []
        for _, row in games_df.iterrows():
            espn_games.append({
                "espn_game_id": row["espn_game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_team_id": row.get("home_team_id"),
                "away_team_id": row.get("away_team_id"),
                "tip_time_utc": row.get("game_datetime_utc"),
                "game_date": f"{str(row['date'])[:4]}-{str(row['date'])[4:6]}-{str(row['date'])[6:]}"
            })

        backfiller = HistoricalBackfill()
        backfiller.process_backfill(espn_games)

    except Exception as e:
        log.error("Historical backfill failed: %s", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
