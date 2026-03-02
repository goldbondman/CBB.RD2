"""
tests/test_twitter_scraper.py
─────────────────────────────
Unit tests for twitter_scraper.py – all network calls are mocked.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import twitter_scraper as ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handicappers_csv(tmp_path: Path) -> Path:
    p = tmp_path / "handicappers.csv"
    p.write_text(
        "handicapper_id,handle,tier,status,lifetime_roi,win_pct,total_picks,"
        "parse_template_id,notes,created_at,last_ingested_tweet_id,"
        "last_scrape_time,scrape_success_rate\n"
        '1,@TestHandle,sharp,active,0.05,0.55,100,1,"test note",2026-01-01,,,1.0\n'
        '2,@InactiveHandle,sharp,inactive,0.01,0.51,50,1,"inactive",2026-01-02,,,1.0\n'
    )
    return p


# ---------------------------------------------------------------------------
# _ensure_tracking_columns
# ---------------------------------------------------------------------------

class TestEnsureTrackingColumns:
    def test_adds_missing_columns(self):
        df = pd.DataFrame({"handicapper_id": [1], "handle": ["@X"]})
        result = ts._ensure_tracking_columns(df)
        for col in ts.HANDICAPPERS_TRACKING_COLS:
            assert col in result.columns

    def test_does_not_overwrite_existing_columns(self):
        df = pd.DataFrame({
            "handicapper_id": [1],
            "last_ingested_tweet_id": ["999"],
            "last_scrape_time": ["2026-01-01T00:00:00+00:00"],
            "scrape_success_rate": [0.8],
        })
        result = ts._ensure_tracking_columns(df)
        assert result.at[0, "last_ingested_tweet_id"] == "999"
        assert result.at[0, "scrape_success_rate"] == 0.8


# ---------------------------------------------------------------------------
# load_handicappers
# ---------------------------------------------------------------------------

class TestLoadHandicappers:
    def test_loads_only_active(self, tmp_path, monkeypatch):
        csv_path = _make_handicappers_csv(tmp_path)
        monkeypatch.setattr(ts, "HANDICAPPERS_CSV", csv_path)
        df = ts.load_handicappers()
        assert len(df) == 1
        assert df.at[0, "handle"] == "@TestHandle"

    def test_filter_by_handle(self, tmp_path, monkeypatch):
        csv_path = _make_handicappers_csv(tmp_path)
        monkeypatch.setattr(ts, "HANDICAPPERS_CSV", csv_path)
        df = ts.load_handicappers(handles=["@TestHandle"])
        assert len(df) == 1

    def test_unknown_handle_returns_empty(self, tmp_path, monkeypatch):
        csv_path = _make_handicappers_csv(tmp_path)
        monkeypatch.setattr(ts, "HANDICAPPERS_CSV", csv_path)
        df = ts.load_handicappers(handles=["@NoSuchHandle"])
        assert df.empty


# ---------------------------------------------------------------------------
# append_tweets
# ---------------------------------------------------------------------------

class TestAppendTweets:
    def test_writes_new_rows(self, tmp_path, monkeypatch):
        out = tmp_path / "raw_tweets.csv"
        monkeypatch.setattr(ts, "RAW_TWEETS_CSV", out)
        rows = [
            {
                "tweet_id": "111",
                "handicapper_id": "1",
                "created_at": "2026-01-01T10:00:00+00:00",
                "text": "Duke -5.5",
                "tweet_url": "https://twitter.com/x/status/111",
                "ingested_at": "2026-01-01T10:01:00+00:00",
            }
        ]
        written = ts.append_tweets(rows)
        assert written == 1
        content = out.read_text()
        assert "111" in content
        assert "Duke -5.5" in content

    def test_deduplicates_on_second_call(self, tmp_path, monkeypatch):
        out = tmp_path / "raw_tweets.csv"
        monkeypatch.setattr(ts, "RAW_TWEETS_CSV", out)
        rows = [
            {
                "tweet_id": "222",
                "handicapper_id": "1",
                "created_at": "2026-01-01T10:00:00+00:00",
                "text": "Kansas -7",
                "tweet_url": "https://twitter.com/x/status/222",
                "ingested_at": "2026-01-01T10:01:00+00:00",
            }
        ]
        ts.append_tweets(rows)
        written2 = ts.append_tweets(rows)  # same rows → should be deduped
        assert written2 == 0

    def test_empty_list_returns_zero(self, tmp_path, monkeypatch):
        out = tmp_path / "raw_tweets.csv"
        monkeypatch.setattr(ts, "RAW_TWEETS_CSV", out)
        assert ts.append_tweets([]) == 0


# ---------------------------------------------------------------------------
# _update_tracking
# ---------------------------------------------------------------------------

class TestUpdateTracking:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "handicapper_id": ["1"],
            "last_ingested_tweet_id": [""],
            "last_scrape_time": [""],
            "scrape_success_rate": ["1.0"],
        })

    def test_updates_last_scrape_time(self):
        df = self._make_df()
        ts._update_tracking(df, 0, [], success=True)
        assert df.at[0, "last_scrape_time"] != ""

    def test_updates_last_tweet_id(self):
        df = self._make_df()
        tweets = [{"tweet_id": "500"}, {"tweet_id": "300"}]
        ts._update_tracking(df, 0, tweets, success=True)
        assert df.at[0, "last_ingested_tweet_id"] == "500"

    def test_success_rate_decreases_on_failure(self):
        df = self._make_df()
        df.at[0, "scrape_success_rate"] = "1.0"
        ts._update_tracking(df, 0, [], success=False)
        assert float(df.at[0, "scrape_success_rate"]) < 1.0


# ---------------------------------------------------------------------------
# TwitterAPIv2Scraper (mocked)
# ---------------------------------------------------------------------------

class TestTwitterAPIv2Scraper:
    def _make_scraper(self) -> ts.TwitterAPIv2Scraper:
        return ts.TwitterAPIv2Scraper(bearer_token="fake_token")

    def _mock_resp(self, json_data: dict, status: int = 200) -> MagicMock:
        m = MagicMock()
        m.status_code = status
        m.json.return_value = json_data
        m.headers = {}
        return m

    def test_fetch_returns_tweets(self):
        scraper = self._make_scraper()
        user_resp = self._mock_resp({"data": {"id": "12345"}})
        tweets_resp = self._mock_resp({
            "data": [
                {"id": "9001", "text": "Duke -5", "created_at": "2026-01-01T10:00:00Z"}
            ]
        })
        with patch("twitter_scraper.requests.get", side_effect=[user_resp, tweets_resp]):
            results = scraper.fetch("@CBB_Edge")
        assert len(results) == 1
        assert results[0]["tweet_id"] == "9001"
        assert results[0]["text"] == "Duke -5"
        assert "CBB_Edge" in results[0]["tweet_url"]

    def test_fetch_returns_empty_on_user_lookup_failure(self):
        scraper = self._make_scraper()
        bad_resp = self._mock_resp({}, status=404)
        with patch("twitter_scraper.requests.get", return_value=bad_resp):
            results = scraper.fetch("@NoSuchUser")
        assert results == []

    def test_fetch_passes_since_id_param(self):
        scraper = self._make_scraper()
        user_resp = self._mock_resp({"data": {"id": "12345"}})
        tweets_resp = self._mock_resp({"data": []})
        with patch("twitter_scraper.requests.get", side_effect=[user_resp, tweets_resp]) as mock_get:
            scraper.fetch("@CBB_Edge", since_id="8000")
        # second call (tweets endpoint) should include since_id in params
        _, kwargs = mock_get.call_args_list[1]
        assert kwargs["params"]["since_id"] == "8000"


# ---------------------------------------------------------------------------
# HTMLFallbackScraper (mocked)
# ---------------------------------------------------------------------------

NITTER_HTML_SAMPLE = """
<html><body>
  <div class="timeline-item">
    <a class="tweet-link" href="/CBB_Edge/status/777333111222"></a>
    <div class="tweet-content">Houston -3 tonight (2u) #CBB</div>
    <div class="tweet-date"><a title="2026-01-02T12:00:00Z">Jan 2</a></div>
  </div>
  <div class="timeline-item">
    <a class="tweet-link" href="/CBB_Edge/status/600000000000"></a>
    <div class="tweet-content">Old tweet</div>
    <div class="tweet-date"><a title="2026-01-01T10:00:00Z">Jan 1</a></div>
  </div>
</body></html>
"""


class TestHTMLFallbackScraper:
    def _make_scraper(self) -> ts.HTMLFallbackScraper:
        return ts.HTMLFallbackScraper()

    def _mock_resp(self, text: str, status: int = 200) -> MagicMock:
        m = MagicMock()
        m.status_code = status
        m.text = text
        m.headers = {}
        return m

    def test_parse_returns_tweets(self):
        scraper = self._make_scraper()
        resp = self._mock_resp(NITTER_HTML_SAMPLE)
        with patch("twitter_scraper.requests.get", return_value=resp):
            results = scraper.fetch("@CBB_Edge")
        assert len(results) == 2
        assert results[0]["tweet_id"] == "777333111222"
        assert "Houston" in results[0]["text"]

    def test_since_id_filters_older_tweets(self):
        scraper = self._make_scraper()
        resp = self._mock_resp(NITTER_HTML_SAMPLE)
        with patch("twitter_scraper.requests.get", return_value=resp):
            # Only tweets with id > 700000000000 should appear
            results = scraper.fetch("@CBB_Edge", since_id="700000000000")
        assert len(results) == 1
        assert results[0]["tweet_id"] == "777333111222"

    def test_returns_empty_on_bad_status(self):
        scraper = self._make_scraper()
        resp = self._mock_resp("", status=404)
        with patch("twitter_scraper.requests.get", return_value=resp):
            results = scraper.fetch("@CBB_Edge")
        assert results == []


# ---------------------------------------------------------------------------
# run() integration (fully mocked)
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_no_bearer_token(self, tmp_path, monkeypatch):
        """Without a bearer token, fallback is used."""
        csv_path = _make_handicappers_csv(tmp_path)
        out = tmp_path / "raw_tweets.csv"
        monkeypatch.setattr(ts, "HANDICAPPERS_CSV", csv_path)
        monkeypatch.setattr(ts, "RAW_TWEETS_CSV", out)
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)

        mock_tweets = [{"tweet_id": "333", "handle": "@TestHandle",
                        "created_at": "2026-01-01T10:00:00+00:00",
                        "text": "test", "tweet_url": "http://x",
                        "ingested_at": "2026-01-01T10:01:00+00:00"}]

        with patch.object(ts.HTMLFallbackScraper, "fetch", return_value=mock_tweets):
            count = ts.run()

        assert count == 1
        assert out.exists()

    def test_run_with_bearer_token(self, tmp_path, monkeypatch):
        """With a bearer token, API v2 is tried first."""
        csv_path = _make_handicappers_csv(tmp_path)
        out = tmp_path / "raw_tweets.csv"
        monkeypatch.setattr(ts, "HANDICAPPERS_CSV", csv_path)
        monkeypatch.setattr(ts, "RAW_TWEETS_CSV", out)
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "fake_token")

        mock_tweets = [{"tweet_id": "444", "handle": "@TestHandle",
                        "created_at": "2026-01-01T10:00:00+00:00",
                        "text": "test2", "tweet_url": "http://x",
                        "ingested_at": "2026-01-01T10:01:00+00:00"}]

        with patch.object(ts.TwitterAPIv2Scraper, "fetch", return_value=mock_tweets):
            count = ts.run()

        assert count == 1

    def test_run_empty_handicappers(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "handicappers.csv"
        csv_path.write_text(
            "handicapper_id,handle,tier,status\n"
            "1,@X,sharp,inactive\n"
        )
        monkeypatch.setattr(ts, "HANDICAPPERS_CSV", csv_path)
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
        count = ts.run()
        assert count == 0
