import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from team_normalizer import CBBNormalizer


@pytest.fixture
def normalizer():
    return CBBNormalizer()


# ---------------------------------------------------------------------------
# normalize_team_name – exact matches (confidence 1.0)
# ---------------------------------------------------------------------------

def test_exact_full_name_illinois(normalizer):
    canonical, conf = normalizer.normalize_team_name("illinois")
    assert canonical == "Illinois"
    assert conf == 1.0


def test_exact_full_name_uconn(normalizer):
    canonical, conf = normalizer.normalize_team_name("uconn")
    assert canonical == "UConn"
    assert conf == 1.0


def test_exact_key_fau_lowercase(normalizer):
    canonical, conf = normalizer.normalize_team_name("fau")
    assert canonical == "Florida Atlantic"
    assert conf == 1.0


# ---------------------------------------------------------------------------
# normalize_team_name – multi-word pattern matches (confidence 0.98)
# ---------------------------------------------------------------------------

def test_illinois_fighting_illini(normalizer):
    canonical, conf = normalizer.normalize_team_name("Illinois Fighting Illini")
    assert canonical == "Illinois"
    assert conf >= 0.98


def test_uconn_huskies(normalizer):
    canonical, conf = normalizer.normalize_team_name("UConn Huskies")
    assert canonical == "UConn"
    assert conf >= 0.98


def test_kansas_jayhawks(normalizer):
    canonical, conf = normalizer.normalize_team_name("Kansas Jayhawks")
    assert canonical == "Kansas"
    assert conf >= 0.98


def test_kentucky_wildcats(normalizer):
    canonical, conf = normalizer.normalize_team_name("Kentucky Wildcats")
    assert canonical == "Kentucky"
    assert conf >= 0.98


def test_duke_blue_devils(normalizer):
    canonical, conf = normalizer.normalize_team_name("Duke Blue Devils")
    assert canonical == "Duke"
    assert conf >= 0.98


def test_north_carolina_tar_heels(normalizer):
    canonical, conf = normalizer.normalize_team_name("North Carolina Tar Heels")
    assert canonical == "North Carolina"
    assert conf >= 0.98


def test_gonzaga_bulldogs(normalizer):
    canonical, conf = normalizer.normalize_team_name("Gonzaga Bulldogs")
    assert canonical == "Gonzaga"
    assert conf >= 0.98


def test_st_johns_with_apostrophe(normalizer):
    canonical, conf = normalizer.normalize_team_name("St. John's Red Storm")
    assert canonical == "St. John's"
    assert conf >= 0.98


def test_st_johns_no_apostrophe(normalizer):
    canonical, conf = normalizer.normalize_team_name("St Johns Red Storm")
    assert canonical == "St. John's"
    assert conf >= 0.95


def test_florida_atlantic_owls(normalizer):
    canonical, conf = normalizer.normalize_team_name("Florida Atlantic Owls")
    assert canonical == "Florida Atlantic"
    assert conf >= 0.98


def test_fau_abbreviation(normalizer):
    canonical, conf = normalizer.normalize_team_name("FAU")
    assert canonical == "Florida Atlantic"
    assert conf >= 0.95


def test_university_of_miami_hurricanes(normalizer):
    canonical, conf = normalizer.normalize_team_name("University of Miami Hurricanes")
    assert canonical == "Miami (FL)"
    assert conf >= 0.95


# ---------------------------------------------------------------------------
# normalize_team_name – unrecognized / None cases
# ---------------------------------------------------------------------------

def test_random_junk_returns_none(normalizer):
    canonical, conf = normalizer.normalize_team_name("Random Junk Team")
    assert canonical is None
    assert conf == 0.0


def test_empty_string_returns_none(normalizer):
    canonical, conf = normalizer.normalize_team_name("")
    assert canonical is None
    assert conf == 0.0


def test_none_input_returns_none(normalizer):
    canonical, conf = normalizer.normalize_team_name(None)
    assert canonical is None
    assert conf == 0.0


# ---------------------------------------------------------------------------
# normalize_team_name – fuzzy matching
# ---------------------------------------------------------------------------

def test_fuzzy_st_john_redstorm(normalizer):
    canonical, conf = normalizer.normalize_team_name("St John Redstorm")
    assert canonical == "St. John's"
    assert conf >= 0.75


# ---------------------------------------------------------------------------
# batch_normalize
# ---------------------------------------------------------------------------

def test_batch_normalize_returns_dataframe(normalizer):
    import pandas as pd

    teams = ["Illinois Fighting Illini", "FAU", "Random Junk Team"]
    df = normalizer.batch_normalize(teams)

    assert list(df.columns) == ['team_raw', 'team_canonical', 'confidence']
    assert len(df) == 3


def test_batch_normalize_illinois(normalizer):
    df = normalizer.batch_normalize(["Illinois Fighting Illini"])
    assert df.iloc[0]['team_canonical'] == "Illinois"
    assert df.iloc[0]['confidence'] >= 0.98


def test_batch_normalize_unknown_team(normalizer):
    df = normalizer.batch_normalize(["Random Junk Team"])
    assert df.iloc[0]['team_canonical'] is None
    assert df.iloc[0]['confidence'] == 0.0


# ---------------------------------------------------------------------------
# team_mapping sanity checks
# ---------------------------------------------------------------------------

def test_kansas_key_not_typo(normalizer):
    """'kansass' typo should be fixed to 'kansas'."""
    assert 'kansass' not in normalizer.team_mapping
    assert 'kansas' in normalizer.team_mapping


def test_fau_key_lowercase(normalizer):
    """'faU' mixed-case key should be normalised to 'fau'."""
    assert 'faU' not in normalizer.team_mapping
    assert 'fau' in normalizer.team_mapping
