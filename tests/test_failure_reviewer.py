"""Tests for failure_reviewer.py, team_normalizer.py, game_mapper.py, and CSVDataManager."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from data_loader import CSVDataManager, load_app_data, save_app_data
from team_normalizer import CBBNormalizer
from game_mapper import GameMapper
from failure_reviewer import FailureReviewer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def handicapper_data_dir(tmp_path):
    """Minimal set of CSV files matching the data/handicapper schema."""
    (tmp_path / "handicappers.csv").write_text(
        "handicapper_id,handle,tier,status,lifetime_roi,win_pct,total_picks\n"
        "1,@CBB_Edge,sharp,active,0.032,0.562,245\n"
        "3,@MidMajorGuru,medium,active,-0.012,0.521,156\n"
    )
    (tmp_path / "raw_tweets.csv").write_text(
        "tweet_id,handicapper_id,created_at,text,tweet_url,ingested_at\n"
        "1001,1,2026-02-28 14:23:00,Illinois -3.5 (2u),https://t.co/a,2026-02-28 14:25:00\n"
        "1002,3,2026-02-28 16:45:00,FADE Duke ML,https://t.co/b,2026-02-28 16:47:00\n"
        "1003,1,2026-02-28 18:00:00,RandomTeam spread,https://t.co/c,2026-02-28 18:02:00\n"
    )
    (tmp_path / "raw_picks.csv").write_text(
        "raw_pick_id,tweet_id,handicapper_id,market,team_raw,line,units,odds,parse_status,parsed_at\n"
        "1,1001,1,spread,Illinois,-3.5,2.0,,success,2026-02-28 14:25:30\n"
        "3,1002,3,moneyline,Duke,,1.0,+120,fade_detected,2026-02-28 16:47:20\n"
        "7,1003,1,spread,St John Red,,1.0,,poor_team_match,2026-02-28 18:02:10\n"
        "8,1003,1,spread,Florida Atlanti,,1.0,,failed,2026-02-28 18:02:11\n"
    )
    (tmp_path / "picks.csv").write_text(
        "pick_id,raw_pick_id,handicapper_id,game_id,market,side,line,units,mapping_status,created_at\n"
        "1,1,1,123,spread,home,-3.5,2.0,ok,2026-02-28 14:25:30\n"
    )
    (tmp_path / "games.csv").write_text(
        "game_id,date,home_team,away_team,home_score,away_score,closing_spread,total_line\n"
        "123,2026-02-28 19:00:00,Illinois,Wisconsin,78,72,-4.5,138.5\n"
        "124,2026-02-28 20:30:00,Duke,UNC,82,79,,142.0\n"
        "125,2026-02-28 21:00:00,Florida Atlantic,Saint Joseph's,71,69,-1.5,135.0\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# CSVDataManager tests
# ---------------------------------------------------------------------------

class TestCSVDataManager:
    def test_load_app_data(self, handicapper_data_dir):
        dm = CSVDataManager(handicapper_data_dir)
        data = dm.load_app_data()
        assert set(data.keys()) == {'handicappers', 'raw_tweets', 'raw_picks', 'picks', 'games'}

    def test_append_record_assigns_id(self, handicapper_data_dir):
        dm = CSVDataManager(handicapper_data_dir)
        new_id = dm.append_record('picks', {
            'raw_pick_id': 7,
            'handicapper_id': 1,
            'game_id': 124,
            'market': 'spread',
            'side': 'home',
            'line': -3.5,
            'units': 1.0,
            'mapping_status': 'manual_override',
            'created_at': '2026-03-01 10:00:00',
        })
        assert new_id == 2  # max(1) + 1

    def test_append_record_persists(self, handicapper_data_dir):
        dm = CSVDataManager(handicapper_data_dir)
        dm.append_record('picks', {
            'raw_pick_id': 7,
            'handicapper_id': 1,
            'game_id': 124,
            'market': 'spread',
            'side': 'home',
            'line': -3.5,
            'units': 1.0,
            'mapping_status': 'manual_override',
            'created_at': '2026-03-01 10:00:00',
        })
        saved = pd.read_csv(handicapper_data_dir / 'picks.csv')
        assert len(saved) == 2


# ---------------------------------------------------------------------------
# CBBNormalizer tests
# ---------------------------------------------------------------------------

class TestCBBNormalizer:
    def test_normalize_known_alias(self):
        n = CBBNormalizer()
        assert n.normalize_team("uconn") == "UConn"

    def test_normalize_strips_whitespace(self):
        n = CBBNormalizer()
        result = n.normalize_team("  Duke  ")
        assert result.strip() == "Duke"

    def test_find_best_match_exact(self):
        n = CBBNormalizer()
        teams = ["Illinois", "Wisconsin", "Duke"]
        match, score = n.find_best_match("Illinois", teams)
        assert match == "Illinois"
        assert score > 0.9

    def test_find_best_match_fuzzy(self):
        n = CBBNormalizer()
        teams = ["Florida Atlantic", "Duke", "Illinois"]
        match, score = n.find_best_match("Florida Atlanti", teams)
        assert match == "Florida Atlantic"

    def test_find_best_match_no_match(self):
        n = CBBNormalizer()
        teams = ["Illinois", "Wisconsin"]
        match, score = n.find_best_match("XYZXYZXYZ", teams, threshold=0.99)
        assert match is None


# ---------------------------------------------------------------------------
# GameMapper tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_games():
    return pd.DataFrame([
        {"game_id": 123, "date": "2026-02-28", "home_team": "Illinois", "away_team": "Wisconsin",
         "closing_spread": -4.5, "total_line": 138.5},
        {"game_id": 124, "date": "2026-02-28", "home_team": "Duke", "away_team": "UNC",
         "closing_spread": -3.0, "total_line": 142.0},
        {"game_id": 125, "date": "2026-02-28", "home_team": "Florida Atlantic", "away_team": "Saint Joseph's",
         "closing_spread": -1.5, "total_line": 135.0},
    ])


class TestGameMapper:
    def test_map_exact_match(self, sample_games):
        mapper = GameMapper()
        pick = pd.Series({"team_raw": "Illinois", "market": "spread"})
        result = mapper.map_raw_pick_to_game(pick, sample_games)
        assert result["game_id"] == 123
        assert result["mapping_status"] == "ok"

    def test_map_fuzzy_match(self, sample_games):
        mapper = GameMapper()
        pick = pd.Series({"team_raw": "Florida Atlanti", "market": "spread"})
        result = mapper.map_raw_pick_to_game(pick, sample_games)
        assert result["game_id"] == 125

    def test_map_no_match(self, sample_games):
        mapper = GameMapper()
        pick = pd.Series({"team_raw": "ZZZUNKNOWNTEAM", "market": "spread"})
        result = mapper.map_raw_pick_to_game(pick, sample_games)
        assert result["game_id"] is None
        assert result["mapping_status"] == "no_match"

    def test_map_empty_games(self):
        mapper = GameMapper()
        pick = pd.Series({"team_raw": "Duke", "market": "spread"})
        result = mapper.map_raw_pick_to_game(pick, pd.DataFrame())
        assert result["game_id"] is None

    def test_map_empty_team_raw(self, sample_games):
        mapper = GameMapper()
        pick = pd.Series({"team_raw": "", "market": "spread"})
        result = mapper.map_raw_pick_to_game(pick, sample_games)
        assert result["game_id"] is None


# ---------------------------------------------------------------------------
# FailureReviewer tests
# ---------------------------------------------------------------------------

class TestFailureReviewer:
    def test_get_failed_picks_returns_failures(self, handicapper_data_dir):
        reviewer = FailureReviewer(handicapper_data_dir)
        failed = reviewer.get_failed_picks()
        # raw_pick_id 1 is 'success' AND in picks → should not appear
        # raw_pick_id 3 is 'fade_detected' → should appear
        # raw_pick_id 7 is 'poor_team_match' → should appear
        # raw_pick_id 8 is 'failed' → should appear
        assert 1 not in failed['raw_pick_id'].tolist()
        assert 3 in failed['raw_pick_id'].tolist()
        assert 7 in failed['raw_pick_id'].tolist()
        assert 8 in failed['raw_pick_id'].tolist()

    def test_get_failed_picks_empty_when_all_mapped(self, tmp_path):
        """If every 'success' pick is in picks and no failures exist, return empty."""
        (tmp_path / "handicappers.csv").write_text(
            "handicapper_id,handle,tier,status\n1,@Test,sharp,active\n"
        )
        (tmp_path / "raw_tweets.csv").write_text(
            "tweet_id,handicapper_id,created_at,text,tweet_url,ingested_at\n"
            "1001,1,2026-02-28,text,http://t.co,2026-02-28\n"
        )
        (tmp_path / "raw_picks.csv").write_text(
            "raw_pick_id,tweet_id,handicapper_id,market,team_raw,line,units,odds,parse_status,parsed_at\n"
            "1,1001,1,spread,Duke,-5.0,1.0,,success,2026-02-28\n"
        )
        (tmp_path / "picks.csv").write_text(
            "pick_id,raw_pick_id,handicapper_id,game_id,market,side,line,units,mapping_status,created_at\n"
            "1,1,1,123,spread,home,-5.0,1.0,ok,2026-02-28\n"
        )
        (tmp_path / "games.csv").write_text(
            "game_id,date,home_team,away_team,closing_spread,total_line\n"
            "123,2026-02-28,Duke,UNC,-5.0,142.0\n"
        )
        reviewer = FailureReviewer(tmp_path)
        failed = reviewer.get_failed_picks()
        assert failed.empty

    def test_bulk_review_summary_no_crash(self, handicapper_data_dir, capsys):
        reviewer = FailureReviewer(handicapper_data_dir)
        reviewer.bulk_review_summary()
        captured = capsys.readouterr()
        assert "FAILURE SUMMARY" in captured.out

    def test_review_single_failure_not_found(self, handicapper_data_dir, capsys):
        reviewer = FailureReviewer(handicapper_data_dir)
        result = reviewer.review_single_failure(9999)
        assert result == {}
        captured = capsys.readouterr()
        assert "not found" in captured.out
