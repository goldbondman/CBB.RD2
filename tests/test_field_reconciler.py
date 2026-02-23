"""Tests for field_reconciler.py — ESPN API field drift detection and correction."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.field_reconciler import (
    bigram_similarity,
    load_column_map,
    get_all_known_labels,
    reconcile,
    generate_patch_suggestions,
    write_drift_report,
    check_pipeline_output,
    fix_uppercase_duplicate_columns,
    _deep_merge,
    FieldDrift,
)


# ── bigram_similarity ────────────────────────────────────────────────────────

class TestBigramSimilarity:
    def test_identical_strings(self):
        assert bigram_similarity("abc", "abc") == 1.0

    def test_completely_different(self):
        assert bigram_similarity("abc", "xyz") == 0.0

    def test_case_insensitive(self):
        assert bigram_similarity("FGM", "fgm") == 1.0

    def test_single_char_returns_zero(self):
        assert bigram_similarity("a", "a") == 0.0

    def test_empty_returns_zero(self):
        assert bigram_similarity("", "abc") == 0.0

    def test_similar_strings_high_score(self):
        score = bigram_similarity("fieldgoals", "fieldgoalsmade")
        assert score > 0.5

    def test_partial_overlap(self):
        score = bigram_similarity("rebounds", "totalrebounds")
        assert 0.3 < score < 1.0


# ── load_column_map ──────────────────────────────────────────────────────────

class TestLoadColumnMap:
    def test_loads_existing_map(self):
        col_map = load_column_map(
            map_path="column_map.json",
            override_path="column_map_overrides.json",
        )
        assert "player_stats" in col_map
        assert "team_stats" in col_map
        assert "pipeline_required_cols" in col_map

    def test_missing_map_returns_empty(self, tmp_path):
        col_map = load_column_map(
            map_path=str(tmp_path / "nonexistent.json"),
            override_path=str(tmp_path / "nonexistent_overrides.json"),
        )
        assert col_map == {}

    def test_overrides_merge(self, tmp_path):
        base = {"player_stats": {"espn_labels": {"min": {"pipeline_col": "min"}}}}
        overrides = {"player_stats": {"espn_labels": {"usage": {"pipeline_col": "usg"}}}}

        base_path = tmp_path / "base.json"
        override_path = tmp_path / "overrides.json"
        base_path.write_text(json.dumps(base))
        override_path.write_text(json.dumps(overrides))

        merged = load_column_map(str(base_path), str(override_path))
        labels = merged["player_stats"]["espn_labels"]
        assert "min" in labels
        assert "usage" in labels


# ── get_all_known_labels ─────────────────────────────────────────────────────

class TestGetAllKnownLabels:
    def test_flattens_player_labels(self):
        col_map = load_column_map()
        labels = get_all_known_labels(col_map, "player_stats")
        assert labels.get("min") == "min"
        assert labels.get("minutes") == "min"
        assert labels.get("pts") == "pts"
        assert labels.get("+/-") == "plus_minus"

    def test_includes_aliases(self):
        col_map = load_column_map()
        labels = get_all_known_labels(col_map, "player_stats")
        assert labels.get("reb") == "reb"
        assert labels.get("rebounds") == "reb"

    def test_split_fields_included(self):
        col_map = load_column_map()
        labels = get_all_known_labels(col_map, "player_stats")
        # Split fields should be present (mapped to None)
        assert "fg" in labels
        assert "3pt" in labels
        assert "ft" in labels


# ── reconcile ────────────────────────────────────────────────────────────────

class TestReconcile:
    def test_no_drift_when_all_mapped(self, tmp_path):
        audit = {
            "source_file": "test.json",
            "player_stat_labels": [
                {"raw": "MIN", "lower": "min"},
                {"raw": "PTS", "lower": "pts"},
                {"raw": "FG", "lower": "fg"},
                {"raw": "3PT", "lower": "3pt"},
                {"raw": "FT", "lower": "ft"},
                {"raw": "OREB", "lower": "oreb"},
                {"raw": "DREB", "lower": "dreb"},
                {"raw": "REB", "lower": "reb"},
                {"raw": "AST", "lower": "ast"},
                {"raw": "STL", "lower": "stl"},
                {"raw": "BLK", "lower": "blk"},
                {"raw": "TO", "lower": "to"},
                {"raw": "PF", "lower": "pf"},
                {"raw": "+/-", "lower": "+/-"},
            ],
            "team_stat_labels": [],
            "unmapped_team_labels": [],
            "unmapped_player_labels": [],
        }
        audit_path = tmp_path / "audit.json"
        audit_path.write_text(json.dumps(audit))

        drifts = reconcile(str(audit_path), verbose=False)
        critical = [d for d in drifts if d.severity == "CRITICAL"]
        assert len(critical) == 0

    def test_detects_new_field(self, tmp_path):
        audit = {
            "source_file": "test.json",
            "player_stat_labels": [
                {"raw": "MIN", "lower": "min"},
                {"raw": "PTS", "lower": "pts"},
                {"raw": "FG", "lower": "fg"},
                {"raw": "3PT", "lower": "3pt"},
                {"raw": "FT", "lower": "ft"},
                {"raw": "OREB", "lower": "oreb"},
                {"raw": "DREB", "lower": "dreb"},
                {"raw": "REB", "lower": "reb"},
                {"raw": "AST", "lower": "ast"},
                {"raw": "STL", "lower": "stl"},
                {"raw": "BLK", "lower": "blk"},
                {"raw": "TO", "lower": "to"},
                {"raw": "PF", "lower": "pf"},
                {"raw": "+/-", "lower": "+/-"},
                {"raw": "USAGE", "lower": "usage"},
            ],
            "team_stat_labels": [],
            "unmapped_team_labels": [],
            "unmapped_player_labels": [],
        }
        audit_path = tmp_path / "audit.json"
        audit_path.write_text(json.dumps(audit))

        drifts = reconcile(str(audit_path), verbose=False)
        new_fields = [d for d in drifts if d.drift_type == "NEW_FIELD"]
        assert any(d.espn_label == "usage" for d in new_fields)

    def test_detects_dropped_column(self, tmp_path):
        # Audit that maps nothing → all required cols are DROPPED
        audit = {
            "source_file": "test.json",
            "player_stat_labels": [],
            "team_stat_labels": [],
            "unmapped_team_labels": [],
            "unmapped_player_labels": [],
        }
        audit_path = tmp_path / "audit.json"
        audit_path.write_text(json.dumps(audit))

        drifts = reconcile(str(audit_path), verbose=False)
        dropped = [d for d in drifts if d.drift_type == "DROPPED"]
        assert len(dropped) > 0
        assert all(d.severity == "CRITICAL" for d in dropped)

    def test_missing_audit_file_returns_empty(self):
        drifts = reconcile("nonexistent_path.json", verbose=False)
        assert drifts == []


# ── generate_patch_suggestions ───────────────────────────────────────────────

class TestGeneratePatchSuggestions:
    def test_generates_for_auto_fixable(self):
        drifts = [
            FieldDrift(
                drift_type="RENAMED", severity="WARNING",
                espn_label="fieldgoal", espn_raw="FieldGoal",
                current_col=None, suggested_col="fgm",
                fuzzy_score=0.90, pipeline_col_affected="fgm",
                auto_fixable=True, notes="test",
            ),
        ]
        suggestions = generate_patch_suggestions(drifts)
        assert "fieldgoal" in suggestions["player_stats"]["espn_labels"]

    def test_skips_non_auto_fixable(self):
        drifts = [
            FieldDrift(
                drift_type="NEW_FIELD", severity="INFO",
                espn_label="usage", espn_raw="USAGE",
                current_col=None, suggested_col=None,
                fuzzy_score=0.3, pipeline_col_affected=None,
                auto_fixable=False, notes="test",
            ),
        ]
        suggestions = generate_patch_suggestions(drifts)
        assert len(suggestions["player_stats"]["espn_labels"]) == 0


# ── write_drift_report ───────────────────────────────────────────────────────

class TestWriteDriftReport:
    def test_writes_json_and_md(self, tmp_path):
        drifts = [
            FieldDrift(
                drift_type="NEW_FIELD", severity="INFO",
                espn_label="usage", espn_raw="USAGE",
                current_col=None, suggested_col=None,
                fuzzy_score=None, pipeline_col_affected=None,
                auto_fixable=False, notes="test new field",
            ),
        ]
        json_path = str(tmp_path / "report.json")
        write_drift_report(drifts, json_path)

        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.md").exists()

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["total_drifts"] == 1
        assert report["info"] == 1

    def test_no_drifts_report(self, tmp_path):
        json_path = str(tmp_path / "report.json")
        write_drift_report([], json_path)

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["total_drifts"] == 0


# ── check_pipeline_output ────────────────────────────────────────────────────

class TestCheckPipelineOutput:
    def test_detects_empty_columns(self, tmp_path):
        df = pd.DataFrame({
            "min": [30, 28, 25],
            "pts": [20, 15, 10],
            "fgm": [None, None, None],
            "fga": [None, None, None],
            "tpm": [2, 3, 1],
            "tpa": [5, 6, 4],
            "ftm": [3, 2, 1],
            "fta": [4, 3, 2],
            "orb": [1, 2, 3],
            "drb": [4, 5, 6],
            "reb": [5, 7, 9],
            "ast": [3, 4, 5],
            "stl": [1, 2, 1],
            "blk": [0, 1, 2],
            "tov": [2, 3, 1],
            "pf": [2, 3, 4],
            "plus_minus": [5, -3, 2],
        })
        csv_path = str(tmp_path / "player_game_metrics.csv")
        df.to_csv(csv_path, index=False)

        result = check_pipeline_output(
            player_metrics_path=csv_path,
            team_metrics_path=str(tmp_path / "nonexistent.csv"),
        )
        assert "fgm" in result["critical_failures"]
        assert "fga" in result["critical_failures"]

    def test_detects_uppercase_twin(self, tmp_path):
        df = pd.DataFrame({
            "fgm": [None, None, None],
            "FGM": [7, 8, 9],
            "fga": [None, None, None],
            "FGA": [13, 14, 15],
            "min": [30, 28, 25],
            "pts": [20, 15, 10],
            "tpm": [2, 3, 1],
            "tpa": [5, 6, 4],
            "ftm": [3, 2, 1],
            "fta": [4, 3, 2],
            "orb": [1, 2, 3],
            "drb": [4, 5, 6],
            "reb": [5, 7, 9],
            "ast": [3, 4, 5],
            "stl": [1, 2, 1],
            "blk": [0, 1, 2],
            "tov": [2, 3, 1],
            "pf": [2, 3, 4],
            "plus_minus": [5, -3, 2],
        })
        csv_path = str(tmp_path / "player_game_metrics.csv")
        df.to_csv(csv_path, index=False)

        result = check_pipeline_output(
            player_metrics_path=csv_path,
            team_metrics_path=str(tmp_path / "nonexistent.csv"),
        )
        assert result["player_metrics"]["fgm"].get("has_uppercase_twin") is True
        assert "fgm" in result["critical_failures"]


# ── fix_uppercase_duplicate_columns ──────────────────────────────────────────

class TestFixUppercaseDuplicateColumns:
    def test_copies_uppercase_to_lowercase(self, tmp_path):
        df = pd.DataFrame({
            "fgm": [None, None, None],
            "FGM": [7, 8, 9],
            "pts": [20, 15, 10],
        })
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        summary = fix_uppercase_duplicate_columns(csv_path)
        assert len(summary["fixed"]) == 1

        fixed_df = pd.read_csv(csv_path)
        assert "FGM" not in fixed_df.columns
        assert list(fixed_df["fgm"]) == [7, 8, 9]

    def test_keeps_lowercase_when_both_have_data(self, tmp_path):
        df = pd.DataFrame({
            "fgm": [7, 8, 9],
            "FGM": [10, 11, 12],
            "pts": [20, 15, 10],
        })
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        summary = fix_uppercase_duplicate_columns(csv_path)
        assert len(summary["kept_lowercase"]) == 1

        fixed_df = pd.read_csv(csv_path)
        assert "FGM" not in fixed_df.columns
        assert list(fixed_df["fgm"]) == [7, 8, 9]

    def test_dry_run_no_changes(self, tmp_path):
        df = pd.DataFrame({
            "fgm": [None, None, None],
            "FGM": [7, 8, 9],
            "pts": [20, 15, 10],
        })
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        fix_uppercase_duplicate_columns(csv_path, dry_run=True)

        df_after = pd.read_csv(csv_path)
        assert "FGM" in df_after.columns  # unchanged
        assert df_after["fgm"].isna().all()  # unchanged

    def test_no_duplicates_found(self, tmp_path):
        df = pd.DataFrame({
            "fgm": [7, 8, 9],
            "pts": [20, 15, 10],
        })
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)

        summary = fix_uppercase_duplicate_columns(csv_path)
        assert summary["status"] == "clean"

    def test_missing_file(self, tmp_path):
        summary = fix_uppercase_duplicate_columns(str(tmp_path / "missing.csv"))
        assert summary["status"] == "skipped"


# ── _deep_merge ──────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_nested_merge(self):
        base = {"a": {"b": 1, "c": 2}}
        overrides = {"a": {"c": 3, "d": 4}}
        result = _deep_merge(base, overrides)
        assert result == {"a": {"b": 1, "c": 3, "d": 4}}

    def test_ignores_underscore_keys(self):
        base = {"a": 1}
        overrides = {"_comment": "ignored", "b": 2}
        result = _deep_merge(base, overrides)
        assert "_comment" not in result
        assert result["b"] == 2

    def test_does_not_mutate_base(self):
        base = {"a": {"b": 1}}
        overrides = {"a": {"b": 2}}
        _deep_merge(base, overrides)
        assert base["a"]["b"] == 1
