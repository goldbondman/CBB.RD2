import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from parser import HandicapperParser


@pytest.fixture()
def parser():
    return HandicapperParser()


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_removes_urls(parser):
    result = parser.normalize_text("Check https://example.com for picks")
    assert "http" not in result


def test_normalize_removes_mentions(parser):
    result = parser.normalize_text("@CBB_Edge Illinois -3.5 (2u)")
    assert "@" not in result


def test_normalize_removes_hashtags(parser):
    result = parser.normalize_text("Illinois -3.5 (2u) #CBB")
    assert "#" not in result


def test_normalize_lowercases(parser):
    result = parser.normalize_text("ILLINOIS -3.5 (2u)")
    assert result == result.lower()


# ---------------------------------------------------------------------------
# handicapper 1 – spread picks
# ---------------------------------------------------------------------------

def test_spread_pick_parsed(parser):
    picks = parser.extract_picks("Illinois -3.5 (2u) vs Wisconsin", handicapper_id=1)
    spread = [p for p in picks if p['market'] == 'spread']
    assert len(spread) >= 1
    assert spread[0]['line'] == 3.5
    assert spread[0]['units'] == 2.0
    assert spread[0]['parse_status'] == 'success'


def test_spread_pick_team_name(parser):
    picks = parser.extract_picks("Illinois -3.5 (2u) vs Wisconsin", handicapper_id=1)
    spread = [p for p in picks if p['market'] == 'spread']
    assert 'illinois' in spread[0]['team_raw'].lower()


def test_gonzaga_spread(parser):
    picks = parser.extract_picks("Gonzaga +2.5 (1u) - Zags undervalued on road @CBB_Edge", handicapper_id=1)
    spread = [p for p in picks if p['market'] == 'spread']
    assert len(spread) >= 1
    assert spread[0]['line'] == 2.5
    assert spread[0]['units'] == 1.0


# ---------------------------------------------------------------------------
# handicapper 2 – total picks
# ---------------------------------------------------------------------------

def test_total_pick_parsed(parser):
    picks = parser.extract_picks("UConn/Kentucky OVER 142.5 (3u) - both shooting lights out LFG", handicapper_id=2)
    totals = [p for p in picks if p['market'] == 'total']
    assert len(totals) >= 1
    assert totals[0]['line'] == 142.5
    assert totals[0]['units'] == 3.0
    assert totals[0]['parse_status'] == 'success'


def test_total_pick_team_raw(parser):
    picks = parser.extract_picks("UConn/Kentucky OVER 142.5 (3u)", handicapper_id=2)
    totals = [p for p in picks if p['market'] == 'total']
    assert 'uconn' in totals[0]['team_raw'].lower()


# ---------------------------------------------------------------------------
# handicapper 3 – fade picks
# ---------------------------------------------------------------------------

def test_fade_pick_detected(parser):
    picks = parser.extract_picks("FADE Duke ML +120 vs UNC. Blue devils overrated this year", handicapper_id=3)
    fades = [p for p in picks if p['market'] == 'fade']
    assert len(fades) >= 1
    assert 'duke' in fades[0]['team_raw'].lower()
    assert fades[0]['parse_status'] == 'fade_detected'


# ---------------------------------------------------------------------------
# parse_tweet_to_raw_picks – metadata enrichment
# ---------------------------------------------------------------------------

def test_parse_tweet_adds_tweet_id(parser):
    picks = parser.parse_tweet_to_raw_picks(
        "Illinois -3.5 (2u)", handicapper_id=1,
        tweet_id="99999", created_at="2025-01-01T10:00:00Z"
    )
    assert all(p['tweet_id'] == "99999" for p in picks)


def test_parse_tweet_adds_handicapper_id(parser):
    picks = parser.parse_tweet_to_raw_picks(
        "Illinois -3.5 (2u)", handicapper_id=1,
        tweet_id="99999", created_at="2025-01-01T10:00:00Z"
    )
    assert all(p['handicapper_id'] == 1 for p in picks)


def test_parse_tweet_adds_parsed_at(parser):
    picks = parser.parse_tweet_to_raw_picks(
        "Illinois -3.5 (2u)", handicapper_id=1,
        tweet_id="99999", created_at="2025-01-01T10:00:00Z"
    )
    assert all('parsed_at' in p for p in picks)


# ---------------------------------------------------------------------------
# Failure fallback
# ---------------------------------------------------------------------------

def test_unparseable_tweet_returns_failed_status(parser):
    picks = parser.extract_picks("Totally unrecognizable text xyz", handicapper_id=1)
    assert len(picks) >= 1
    assert any(p.get('parse_status') == 'failed' for p in picks)


# ---------------------------------------------------------------------------
# save_raw_picks – CSV integration
# ---------------------------------------------------------------------------

def test_save_raw_picks_appends_to_csv(tmp_path):
    """save_raw_picks should append records to raw_picks.csv via CSVDataManager."""
    # Create minimal CSV fixtures
    (tmp_path / "handicappers.csv").write_text(
        "handicapper_id,name,tier,status\n1,Alice,A,active\n"
    )
    (tmp_path / "raw_tweets.csv").write_text(
        "tweet_id,handicapper_id,text,created_at,ingested_at\n"
        "101,1,Duke -5,2025-01-01T10:00:00Z,2025-01-01T10:05:00Z\n"
    )
    (tmp_path / "raw_picks.csv").write_text(
        "raw_pick_id,handicapper_id,tweet_id,market,team_raw,line,units,odds,parse_status,parsed_at\n"
        "1,1,101,spread,duke,5.0,1.0,,success,2025-01-01T10:05:00\n"
    )
    (tmp_path / "picks.csv").write_text(
        "pick_id,raw_pick_id,handicapper_id,game_id,team,line,units,mapping_status\n"
        "301,1,1,1001,Duke,-5.0,1.0,mapped\n"
    )
    (tmp_path / "games.csv").write_text(
        "game_id,date,home_team,away_team,closing_spread,total_line\n"
        "1001,2025-01-01,Duke,UNC,-5.0,145.5\n"
    )

    p = HandicapperParser()
    raw_picks = p.parse_tweet_to_raw_picks(
        "Illinois -3.5 (2u)", handicapper_id=1,
        tweet_id="999", created_at="2025-01-02T10:00:00Z"
    )
    p.save_raw_picks(raw_picks, data_dir=str(tmp_path))

    import pandas as pd
    saved = pd.read_csv(tmp_path / "raw_picks.csv")
    assert len(saved) > 1
    assert 'raw_pick_id' in saved.columns
