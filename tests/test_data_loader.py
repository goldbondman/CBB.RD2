import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from data_loader import load_app_data, save_app_data


@pytest.fixture()
def sample_data_dir(tmp_path):
    """Create a minimal set of the 5 CSV files for testing."""
    (tmp_path / "handicappers.csv").write_text(
        "handicapper_id,name,tier,status\n"
        "1,Alice,A,active\n"
        "2,Bob,B,inactive\n"
    )
    (tmp_path / "raw_tweets.csv").write_text(
        "tweet_id,handicapper_id,text,created_at,ingested_at\n"
        "101,1,Duke -5,2025-01-01T10:00:00Z,2025-01-01T10:05:00Z\n"
    )
    (tmp_path / "raw_picks.csv").write_text(
        "raw_pick_id,handicapper_id,tweet_id,raw_text,line,units,parse_status\n"
        "201,1,101,Duke -5,-5.0,1.0,parsed\n"
    )
    (tmp_path / "picks.csv").write_text(
        "pick_id,raw_pick_id,handicapper_id,game_id,team,line,units,mapping_status\n"
        "301,201,1,1001,Duke,-5.0,1.0,mapped\n"
    )
    (tmp_path / "games.csv").write_text(
        "game_id,date,home_team,away_team,closing_spread,total_line\n"
        "1001,2025-01-01,Duke,UNC,-5.0,145.5\n"
    )
    return tmp_path


def test_load_app_data_returns_all_keys(sample_data_dir):
    data = load_app_data(sample_data_dir)
    assert set(data.keys()) == {'handicappers', 'raw_tweets', 'raw_picks', 'picks', 'games'}


def test_load_app_data_handicappers_dtypes(sample_data_dir):
    data = load_app_data(sample_data_dir)
    df = data['handicappers']
    assert df['handicapper_id'].dtype == 'int32'
    assert str(df['tier'].dtype) == 'category'
    assert str(df['status'].dtype) == 'category'


def test_load_app_data_raw_tweets_timestamps(sample_data_dir):
    data = load_app_data(sample_data_dir)
    df = data['raw_tweets']
    assert pd.api.types.is_datetime64_any_dtype(df['created_at'])
    assert pd.api.types.is_datetime64_any_dtype(df['ingested_at'])
    assert df['handicapper_id'].dtype == 'int32'


def test_load_app_data_raw_picks_dtypes(sample_data_dir):
    data = load_app_data(sample_data_dir)
    df = data['raw_picks']
    assert df['raw_pick_id'].dtype == 'int32'
    assert df['handicapper_id'].dtype == 'int32'
    assert df['line'].dtype == 'float64'
    assert df['units'].dtype == 'float64'
    assert str(df['parse_status'].dtype) == 'category'


def test_load_app_data_picks_dtypes(sample_data_dir):
    data = load_app_data(sample_data_dir)
    df = data['picks']
    assert df['pick_id'].dtype == 'int32'
    assert df['raw_pick_id'].dtype == 'int32'
    assert df['handicapper_id'].dtype == 'int32'
    assert df['game_id'].dtype == 'int32'
    assert df['line'].dtype == 'float64'
    assert df['units'].dtype == 'float64'
    assert str(df['mapping_status'].dtype) == 'category'


def test_load_app_data_games_dtypes(sample_data_dir):
    data = load_app_data(sample_data_dir)
    df = data['games']
    assert df['game_id'].dtype == 'int32'
    assert df['closing_spread'].dtype == 'float64'
    assert df['total_line'].dtype == 'float64'
    assert pd.api.types.is_datetime64_any_dtype(df['date'])


def test_load_app_data_pk_validation_fails_on_duplicates(sample_data_dir):
    """Duplicate primary keys must raise AssertionError."""
    dup = (
        "handicapper_id,name,tier,status\n"
        "1,Alice,A,active\n"
        "1,Alice Dup,A,active\n"
    )
    (sample_data_dir / "handicappers.csv").write_text(dup)
    with pytest.raises(AssertionError, match="Duplicate handicapper_id in handicappers"):
        load_app_data(sample_data_dir)


def test_save_app_data_round_trips(sample_data_dir, tmp_path):
    out_dir = tmp_path / "out"
    data = load_app_data(sample_data_dir)
    save_app_data(data, out_dir)
    reloaded = load_app_data(out_dir)
    for name in data:
        assert len(reloaded[name]) == len(data[name]), f"Row count mismatch for {name}"


def test_save_app_data_creates_directory(tmp_path):
    new_dir = tmp_path / "brand_new"
    assert not new_dir.exists()
    save_app_data({}, new_dir)
    assert new_dir.exists()
