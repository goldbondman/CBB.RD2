"""
WagerTalk historical odds scraper for CBB.

Fetches schedule, lines, and scores from WagerTalk's internal API:
  /odds?action=getData&date=YYYY-MM-DD&data={L4.txt | lines-L4.txt | scores-L4.txt}

Sportsbook IDs (from sportsbooks.txt):
  25=Tickets  26=Money(ML)  116=Open  109=DraftKings  112=Fanduel
  108=Circa    114=SuperBook  107=Caesars  106=BetMGM    117=SouthPoint
  113=HardRock 110=ESPNBet   111=Fanatics  115=Consensus

Value encoding:
  Total row  -> bare number, e.g. "147.5" or "145.5u-15" (side + ML juice)
  Spread row -> sign+number+ML, e.g. "-3.5-17" ("-3.5 at -117")
  ML shorthand: drop leading "1" from absolute value (e.g. "-17" = -117).
"""

from __future__ import annotations

import argparse
import csv
import html
import logging
import random
import re
import time
from dataclasses import dataclass, field, fields, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[WagerTalk] %(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.wagertalk.com"
ODDS_URL = f"{BASE_URL}/odds"
SPORT = "L4"          # College Basketball
PERIOD = 0            # Full-game lines
CONSENSUS_BOOK = 115
OPEN_BOOK = 116
BOOKS: Dict[int, str] = {
    25: "tickets", 26: "money",
    116: "open", 109: "draftkings", 112: "fanduel",
    108: "circa", 114: "superbook", 107: "caesars", 106: "betmgm",
    117: "southpoint", 113: "hardrock", 110: "espnbet",
    111: "fanatics", 115: "consensus",
}
SEASON_START = date(2025, 11, 1)
OFFSEASON_START = (4, 15)
OFFSEASON_END = (10, 31)
DEFAULT_DELAY = 2.5      # seconds between requests (be polite)
MAX_RETRIES = 3
BACKOFF_BASE = 8.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
@dataclass
class GameOdds:
    game_date: str          # YYYY-MM-DD
    game_id: int
    away_team: str
    away_abbr: str
    home_team: str
    home_abbr: str
    tip_time_et: str        # HH:MM
    consensus_spread: Optional[float] = None   # home spread (negative = home fav)
    consensus_total: Optional[float] = None
    consensus_ml_away: Optional[int] = None
    consensus_ml_home: Optional[int] = None
    open_spread: Optional[float] = None
    open_total: Optional[float] = None
    dk_spread: Optional[float] = None
    dk_total: Optional[float] = None
    fd_spread: Optional[float] = None
    fd_total: Optional[float] = None
    caesars_spread: Optional[float] = None
    betmgm_spread: Optional[float] = None
    away_score: Optional[int] = None
    home_score: Optional[int] = None
    game_status: str = ""   # "Final", "In Progress", etc.
    scraped_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# ---------------------------------------------------------------------------
# Value parser
# ---------------------------------------------------------------------------
_FRAC_MAP = {
    "\u00bd": ".5",
    "\u00bc": ".25",
    "\u00be": ".75",
    "\u00c2\u00bd": ".5",
    "\u00c2\u00bc": ".25",
    "\u00c2\u00be": ".75",
}
_FRAC_ENT = {"&frac12;": ".5", "&frac14;": ".25", "&frac34;": ".75"}


def _decode_fracs(s: str) -> str:
    """Replace fraction chars/entities with decimals."""
    for k, v in _FRAC_ENT.items():
        s = s.replace(k, v)
    for k, v in _FRAC_MAP.items():
        s = s.replace(k, v)
    return s


def _decode_ml(raw: str) -> Optional[int]:
    """
    Decode WagerTalk shorthand moneyline.
    Drop-leading-1 convention for 3-digit lines:
      '-17'  -> -117    '+08' -> +108    'EVEN' -> +100
      '-100' -> -100    '+110' -> +110
    """
    raw = raw.strip()
    if not raw or raw in {"", "pk", "PK"}:
        return None
    if raw.upper() == "EVEN":
        return 100
    try:
        v = int(raw)
    except ValueError:
        return None
    # 2-digit abs value means 3-digit ML (drop leading 1 convention)
    if abs(v) <= 99:
        if v < 0:
            return -(100 + abs(v))
        else:
            return 100 + v
    return v


def parse_cell_value(raw: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Parse a WagerTalk cell value. Returns (spread, total, ml_juice).
    Exactly one of spread or total will be set for standard cells.

    Examples:
      '147.5'      -> (None, 147.5, None)
      '145.5u-15'  -> (None, 145.5, -115)    [under at -115]
      '-3.5-17'    -> (-3.5, None, -117)      [spread -3.5 at -117]
      '-11-10'     -> (-11.0, None, -110)
      'PK-10'      -> (0.0, None, -110)
      'EVEN'       -> (None, None, 100)       [pure ML]
    """
    if not raw:
        return None, None, None

    s = _decode_fracs(html.unescape(raw.strip()))

    # "PK" or "pk" spread
    s = re.sub(r'^pk', '0', s, flags=re.IGNORECASE)

    # Pattern: optional_sign + number + optional(o|u) + optional_ml_part
    #          e.g.  "-3.5-17"  "147.5u-15"  "147.5"  "-11-10"
    m = re.match(
        r'^([+-]?\d+(?:\.\d+)?)'   # group 1: main number
        r'([ou]?)'                  # group 2: o/u indicator (optional)
        r'([+-]\d+)?$',             # group 3: ML portion (optional)
        s, re.IGNORECASE
    )
    if not m:
        return None, None, None

    num = float(m.group(1))
    side = m.group(2).lower()  # 'o', 'u', or ''
    ml_raw = m.group(3) or ""

    ml = _decode_ml(ml_raw) if ml_raw else None

    # Classification: total if num >= 100 AND no negative sign leading, or side indicator present
    if side in ("o", "u"):
        return None, num, ml
    if num >= 100.0:
        return None, num, ml
    # Otherwise it's a spread
    return num, None, ml


def is_probable_offseason_day(target_day: date) -> bool:
    """Skip clear CBB offseason dates during historical backfills."""
    month_day = (target_day.month, target_day.day)
    return OFFSEASON_START <= month_day <= OFFSEASON_END


def iter_scrape_dates(start: date, end: date) -> List[date]:
    """Yield only the dates worth querying for a CBB odds backfill."""
    days: List[date] = []
    current = start
    while current <= end:
        if not is_probable_offseason_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Session manager with warm-up & retry
# ---------------------------------------------------------------------------
class WagerTalkSession:
    def __init__(self, delay: float = DEFAULT_DELAY):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self._last_req: float = 0.0
        self._warmed = False

    def _throttle(self):
        elapsed = time.time() - self._last_req
        wait = self.delay - elapsed + random.uniform(0.2, 0.8)
        if wait > 0:
            time.sleep(wait)
        self._last_req = time.time()

    def warm_up(self, date_str: str):
        """Load main page to establish session cookies."""
        url = f"{ODDS_URL}?sport={SPORT}&date={date_str}&cb={random.random()}"
        self._throttle()
        try:
            r = self.session.get(url, timeout=20)
            r.raise_for_status()
            self._warmed = True
            log.debug("Session warmed for date %s", date_str)
        except Exception as e:
            log.warning("Warm-up failed: %s", e)

    def get_data(self, date_str: str, data_file: str) -> Optional[str]:
        """Fetch a data file from the internal API with retries."""
        if not self._warmed:
            self.warm_up(date_str)

        url = f"{ODDS_URL}?action=getData&date={date_str}&data={data_file}"
        backoff = BACKOFF_BASE
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                r = self.session.get(url, timeout=25)
                if r.status_code == 200:
                    return r.text
                elif r.status_code in {429, 503, 502}:
                    log.warning("Rate limited (%s) for %s, attempt %d. Backing off %ds",
                                r.status_code, data_file, attempt + 1, int(backoff))
                    time.sleep(backoff + random.uniform(0, 5))
                    backoff *= 2
                else:
                    log.error("HTTP %s for %s", r.status_code, url)
                    return None
            except requests.RequestException as e:
                log.warning("Request error attempt %d: %s", attempt + 1, e)
                time.sleep(backoff)
                backoff *= 2
        return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_schedule(text: str) -> List[Dict]:
    """
    Parse L4.txt schedule into list of game dicts.
    Game row: {game_id,,,date,...,{away_name,away_abbr,,,,home_name,home_abbr,,,}}
    """
    games = []
    # Match individual game blocks: starts with a number, has nested team block
    pattern = re.compile(
        r'\{(\d+),'         # game_id
        r',,'               # skip two empty
        r'(\d{4}-\d{2}-\d{2}),'   # game_date
        r'[^,]*,'           # repeat date (skip)
        r'[^,]*,'           # date_short (skip)
        r'([^,]*),'         # time_away (HH:MM)
        r'[^,]*,'           # time_home (skip dup)
        r'[^,]*'            # full datetime (skip)
        r'(?:,[^,]*){5},'   # 5 more comma-separated fields
        r'\{'               # open team block
        r'([^,}]*),'        # away_name
        r'([^,}]*),'        # away_abbr
        r'[^,]*,[^,]*,[^,]*,'  # 3 skip
        r'([^,}]*),'        # home_name
        r'([^,}]*),'        # home_abbr
    )
    for m in pattern.finditer(text):
        game_id = int(m.group(1))
        game_date = m.group(2)
        tip_time = m.group(3).strip()
        away_name = html.unescape(m.group(4).strip())
        away_abbr = m.group(5).strip()
        home_name = html.unescape(m.group(6).strip())
        home_abbr = m.group(7).strip()

        # Skip write-in placeholder rows and garbled names
        if "&nbsp;" in away_name or not away_name.strip() or away_name.strip() in (" ", "null"):
            continue
        if any(ord(c) > 127 for c in away_name):   # non-ASCII garbage
            continue

        games.append({
            "game_id": game_id,
            "game_date": game_date,
            "tip_time_et": tip_time,
            "away_team": away_name,
            "away_abbr": away_abbr,
            "home_team": home_name,
            "home_abbr": home_abbr,
        })

    return games


def parse_scores(text: str) -> Dict[int, Dict]:
    """
    Parse scores-L4.txt.
    Format: {t{game_id},{away_score},,{home_score},{status}}
    Returns {game_id: {away_score, home_score, status}}
    """
    scores: Dict[int, Dict] = {}
    for m in re.finditer(r'\{t(\d+),(\d*),([^,]*),(\d*),([^}]*)\}', text):
        game_id = int(m.group(1))
        try:
            away = int(m.group(2)) if m.group(2) else None
        except ValueError:
            away = None
        try:
            home = int(m.group(4)) if m.group(4) else None
        except ValueError:
            home = None
        status = m.group(5).strip()
        scores[game_id] = {"away_score": away, "home_score": home, "status": status}
    return scores


def parse_lines(text: str) -> Dict[Tuple[int, int, int], str]:
    """
    Parse lines-L4.txt.
    Format: {t{game_id}p{period}b{book_id}r{row},,{line_ts},{value}}
    Returns {(game_id, book_id, row): raw_value_string}
    """
    lines: Dict[Tuple[int, int, int], str] = {}
    pattern = re.compile(
        r'\{t(\d+)p(\d+)b(\d+)r(\d+),'   # game_id, period, book_id, row
        r'[^,]*,'                          # blank
        r'[^,]*,'                          # line timestamp
        r'([^}]*)\}'                       # value
    )
    for m in pattern.finditer(text):
        game_id = int(m.group(1))
        period = int(m.group(2))
        book_id = int(m.group(3))
        row = int(m.group(4))
        value = m.group(5).strip()
        if period == PERIOD:
            lines[(game_id, book_id, row)] = value
    return lines


def _extract_book_lines(
    game_id: int,
    book_id: int,
    lines: Dict[Tuple[int, int, int], str],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Given game_id and book_id, return (spread, total) for that book.
    Row 1 and row 2 each carry either a spread value or a total - identify by shape.
    Spread = |value| < 100.   Total = value >= 80 (totals are typically 120-200 in CBB).
    Returns home-team spread convention (negative = home fav).
    """
    r1 = lines.get((game_id, book_id, 1), "")
    r2 = lines.get((game_id, book_id, 2), "")

    spread: Optional[float] = None
    total: Optional[float] = None

    for raw in (r1, r2):
        sp, tot, _ = parse_cell_value(raw)
        if tot is not None and total is None:
            total = tot
        if sp is not None and spread is None:
            spread = sp

    # WagerTalk displays spread from the perspective of the home-team row (row 2 typically).
    # If spread came from r1 (away row), it is the away team's spread; flip sign for home convention.
    if spread is not None:
        sp1, _, _ = parse_cell_value(r1)
        sp2, _, _ = parse_cell_value(r2)
        if sp1 is not None and sp2 is None:
            # Spread was on the away row; convert to home-team convention
            spread = -spread
        # else: spread was on home row (already home convention) or both present (use r2)

    return spread, total


def _extract_ml(game_id: int, book_id: int, row: int,
                lines: Dict[Tuple[int, int, int], str]) -> Optional[int]:
    raw = lines.get((game_id, book_id, row), "")
    _, _, ml = parse_cell_value(raw)
    return ml


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------
def build_game_odds(
    games: List[Dict],
    scores: Dict[int, Dict],
    lines: Dict[Tuple[int, int, int], str],
) -> List[GameOdds]:
    results: List[GameOdds] = []

    for g in games:
        gid = g["game_id"]

        # Consensus spread & total
        cons_spread, cons_total = _extract_book_lines(gid, CONSENSUS_BOOK, lines)
        # Opening line
        open_spread, open_total = _extract_book_lines(gid, OPEN_BOOK, lines)
        # Per-book
        dk_spread, dk_total = _extract_book_lines(gid, 109, lines)
        fd_spread, fd_total = _extract_book_lines(gid, 112, lines)
        caes_spread, _ = _extract_book_lines(gid, 107, lines)
        mgm_spread, _ = _extract_book_lines(gid, 106, lines)

        # Moneylines from consensus (book 115), rows 1 & 2
        ml_r1 = _extract_ml(gid, CONSENSUS_BOOK, 1, lines)
        ml_r2 = _extract_ml(gid, CONSENSUS_BOOK, 2, lines)

        # Determine which row is away vs home based on spread shape
        # Row with total -> that's one side; row with spread is the other.
        # ML assignment: if consensus r1 has spread, it's away ML; if r2 has spread, it's home ML.
        cons_r1_raw = lines.get((gid, CONSENSUS_BOOK, 1), "")
        cons_r1_sp, cons_r1_tot, _ = parse_cell_value(cons_r1_raw)
        ml_away: Optional[int] = None
        ml_home: Optional[int] = None
        if cons_r1_sp is not None:
            # r1 is away team spread row -> ml_r1 = away ML
            ml_away = ml_r1
            ml_home = ml_r2
        else:
            # r1 is total row -> r2 is spread+ML row (home row)
            ml_home = ml_r2
            ml_away = ml_r1

        score_data = scores.get(gid, {})

        rec = GameOdds(
            game_date=g["game_date"],
            game_id=gid,
            away_team=g["away_team"],
            away_abbr=g["away_abbr"],
            home_team=g["home_team"],
            home_abbr=g["home_abbr"],
            tip_time_et=g["tip_time_et"],
            consensus_spread=cons_spread,
            consensus_total=cons_total,
            consensus_ml_away=ml_away,
            consensus_ml_home=ml_home,
            open_spread=open_spread,
            open_total=open_total,
            dk_spread=dk_spread,
            dk_total=dk_total,
            fd_spread=fd_spread,
            fd_total=fd_total,
            caesars_spread=caes_spread,
            betmgm_spread=mgm_spread,
            away_score=score_data.get("away_score"),
            home_score=score_data.get("home_score"),
            game_status=score_data.get("status", ""),
        )
        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Scraper orchestration
# ---------------------------------------------------------------------------
class WagerTalkScraper:
    def __init__(
        self,
        output_path: Path,
        delay: float = DEFAULT_DELAY,
        skip_existing: bool = True,
    ):
        self.output_path = output_path
        self.skip_existing = skip_existing
        self.session = WagerTalkSession(delay=delay)
        self._scraped_dates: set = set()

        # Load already-scraped dates to allow resume
        if skip_existing and output_path.exists():
            try:
                import pandas as pd
                existing = pd.read_csv(output_path, usecols=["game_date"])
                self._scraped_dates = set(existing["game_date"].unique())
                log.info("Loaded %d already-scraped dates from %s",
                         len(self._scraped_dates), output_path)
            except Exception as e:
                log.warning("Could not load existing output: %s", e)

    def scrape_date(self, date_str: str) -> List[GameOdds]:
        """Fetch and parse all CBB games for a single date."""
        log.info("Scraping %s ...", date_str)

        sched_raw = self.session.get_data(date_str, f"{SPORT}.txt")
        if not sched_raw:
            log.warning("No schedule data for %s", date_str)
            return []

        games = parse_schedule(sched_raw)
        if not games:
            log.info("No games on %s", date_str)
            return []
        log.info("  %d games found", len(games))

        lines_raw = self.session.get_data(date_str, f"lines-{SPORT}.txt")
        lines: Dict = parse_lines(lines_raw) if lines_raw else {}

        scores_raw = self.session.get_data(date_str, f"scores-{SPORT}.txt")
        scores: Dict = parse_scores(scores_raw) if scores_raw else {}

        return build_game_odds(games, scores, lines)

    def scrape_range(self, start: date, end: date):
        """Scrape all dates from start to end (inclusive)."""
        all_rows: List[GameOdds] = []
        total_calendar_days = (end - start).days + 1
        target_days = iter_scrape_dates(start, end)
        skipped_offseason_days = total_calendar_days - len(target_days)
        if skipped_offseason_days:
            log.info(
                "Skipping %d probable offseason dates in requested window",
                skipped_offseason_days,
            )
        if not target_days:
            log.info("No probable in-season dates found in %s -> %s", start, end)
            return all_rows

        total_days = len(target_days)

        for done, current in enumerate(target_days, start=1):
            date_str = current.strftime("%Y-%m-%d")
            if date_str in self._scraped_dates:
                log.info("Skipping %s (already scraped)", date_str)
                continue

            try:
                day_results = self.scrape_date(date_str)
                if day_results:
                    all_rows.extend(day_results)
                    self._write_rows(day_results, append=self.output_path.exists())
                    log.info("  -> Saved %d games for %s", len(day_results), date_str)
            except Exception as e:
                log.error("Error scraping %s: %s", date_str, e)

            if done % 10 == 0:
                log.info("Progress: %d/%d days", done, total_days)

        log.info("Scrape complete. %d total game records written.", len(all_rows))
        return all_rows

    def _write_rows(self, rows: List[GameOdds], append: bool = False):
        if not rows:
            return
        mode = "a" if append else "w"
        write_header = not append or not self.output_path.exists()
        field_names = [f.name for f in fields(GameOdds)]
        with open(self.output_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))


# ---------------------------------------------------------------------------
# CandidateLine adapter for HistoricalBackfill integration
# ---------------------------------------------------------------------------
_WAGERTALK_FRAME_CACHE: Dict[str, "pd.DataFrame"] = {}


def _first_notna(*values):
    for value in values:
        if value is None:
            continue
        try:
            import pandas as pd  # local import keeps scraper runtime lightweight

            if pd.isna(value):
                continue
        except Exception:
            pass
        return value
    return None


def _load_wagertalk_frame(csv_path: Path):
    import pandas as pd

    cache_key = str(csv_path.resolve())
    cached = _WAGERTALK_FRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    df = pd.read_csv(csv_path, low_memory=False)
    required = {"game_date", "game_id", "away_team", "home_team"}
    missing = sorted(required.difference(df.columns))
    if missing:
        log.warning("WagerTalk CSV missing required columns %s: %s", missing, csv_path)
        empty = pd.DataFrame(columns=list(required))
        _WAGERTALK_FRAME_CACHE[cache_key] = empty
        return empty

    numeric_cols = [
        "consensus_spread",
        "consensus_total",
        "dk_spread",
        "dk_total",
        "fd_spread",
        "fd_total",
        "open_spread",
        "open_total",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "scraped_at_utc" in df.columns:
        df["_scraped_ts"] = pd.to_datetime(df["scraped_at_utc"], utc=True, errors="coerce")
        df = df.sort_values(["game_date", "game_id", "_scraped_ts"], kind="stable")
    else:
        df = df.sort_values(["game_date", "game_id"], kind="stable")

    deduped = df.drop_duplicates(subset=["game_date", "game_id"], keep="last").copy()
    if "_scraped_ts" in deduped.columns:
        deduped = deduped.drop(columns=["_scraped_ts"])
    deduped["game_date"] = deduped["game_date"].astype(str)

    _WAGERTALK_FRAME_CACHE[cache_key] = deduped
    return deduped


def wagertalk_candidates_for_date(
    game_date: date,
    data_dir: Path = Path("data"),
    cache_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Load WagerTalk closing lines as CandidateLine-compatible dicts for a given date.
    Reads from the pre-scraped CSV (wagertalk_historical_odds.csv).
    Returns list of dicts with keys matching CandidateLine fields.
    """
    csv_path = cache_path or (data_dir / "wagertalk_historical_odds.csv")
    if not csv_path.exists():
        return []

    import pandas as pd

    df = _load_wagertalk_frame(csv_path)
    if df.empty:
        return []

    date_str = game_date.strftime("%Y-%m-%d")
    day = df[df["game_date"] == date_str]

    candidates = []
    for _, row in day.iterrows():
        away_team = str(row.get("away_team", "")).strip()
        home_team = str(row.get("home_team", "")).strip()
        if not away_team or away_team.lower() == "nan":
            continue
        if not home_team or home_team.lower() == "nan":
            continue

        spread = _first_notna(
            row.get("consensus_spread"),
            row.get("dk_spread"),
            row.get("fd_spread"),
            row.get("open_spread"),
        )
        total = _first_notna(
            row.get("consensus_total"),
            row.get("dk_total"),
            row.get("fd_total"),
            row.get("open_total"),
        )
        if pd.isna(spread) or pd.isna(total):
            continue

        candidates.append({
            "provider_name": "wagertalk",
            "provider_game_date": date_str,
            "provider_home_team_raw": home_team,
            "provider_away_team_raw": away_team,
            "candidate_home_spread": float(spread),
            "candidate_total": float(total),
            "candidate_is_closing": True,
            "url": f"https://www.wagertalk.com/odds?sport=L4&date={date_str}",
        })

    return candidates


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scrape WagerTalk historical CBB odds."
    )
    parser.add_argument(
        "--start-date", default=SEASON_START.strftime("%Y-%m-%d"),
        help=f"Start date YYYY-MM-DD (default: {SEASON_START})"
    )
    parser.add_argument(
        "--end-date", default=date.today().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--output", default="data/wagertalk_historical_odds.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between requests (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-scrape dates that already exist in output"
    )
    parser.add_argument(
        "--date", default=None,
        help="Scrape a single date (overrides --start-date/--end-date)"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scraper = WagerTalkScraper(
        output_path=output_path,
        delay=args.delay,
        skip_existing=not args.no_skip,
    )

    if args.date:
        if scraper.skip_existing and args.date in scraper._scraped_dates:
            log.info("Skipping %s (already scraped)", args.date)
            return
        rows = scraper.scrape_date(args.date)
        if rows:
            scraper._write_rows(rows, append=output_path.exists())
            log.info("Wrote %d games for %s", len(rows), args.date)
    else:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        log.info("Scraping %s -> %s (%d days)",
                 start, end, (end - start).days + 1)
        scraper.scrape_range(start, end)


if __name__ == "__main__":
    main()
