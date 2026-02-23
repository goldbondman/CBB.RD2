"""
Field Reconciler — ESPN API field-name drift detection and auto-correction.

Reads the ESPN API audit JSON, compares against the canonical column_map.json,
detects drift (renames, new fields, dropped fields), and optionally patches
CSVs to fix the uppercase/lowercase duplicate column bug.

Usage:
    python field_reconciler.py                              # full reconcile + report
    python field_reconciler.py --check-outputs              # check CSVs for empty columns
    python field_reconciler.py --fix-duplicates             # fix FGM/fgm style duplication
    python field_reconciler.py --fix-duplicates --dry-run
    python field_reconciler.py --apply-suggestions          # apply auto_fixable patches
    python field_reconciler.py --audit-path <path>          # specify audit file
    python field_reconciler.py --update-map                 # interactive review
"""

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Section A — Load and merge maps ──────────────────────────────────────────


def load_column_map(
    map_path: str = "column_map.json",
    override_path: str = "column_map_overrides.json",
) -> dict:
    """
    Load column_map.json, then merge column_map_overrides.json on top.
    Overrides win on any key conflict.
    Return merged dict.
    """
    base: dict = {}
    map_file = Path(map_path)
    if map_file.exists():
        base = json.loads(map_file.read_text())
    else:
        print(f"[WARN] {map_path} not found — using empty base map")

    override_file = Path(override_path)
    if override_file.exists():
        overrides = json.loads(override_file.read_text())
        base = _deep_merge(base, overrides)

    return base


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict. Overrides win on conflict."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        elif value:  # only override with non-empty values
            merged[key] = value
    return merged


def get_all_known_labels(col_map: dict, section: str) -> Dict[str, Optional[str]]:
    """
    Flatten the column map for a section ("player_stats" or "team_stats")
    into a simple dict: lowercase_label -> pipeline_col.
    Includes all aliases. Used for O(1) lookup during reconciliation.
    Returns: {"min": "min", "minutes": "min", "MIN": "min", "fg": None, ...}
    """
    result: Dict[str, Optional[str]] = {}
    section_data = col_map.get(section, {})
    espn_labels = section_data.get("espn_labels", {})

    for label, info in espn_labels.items():
        pipeline_col = info.get("pipeline_col")
        result[label.lower()] = pipeline_col
        for alias in info.get("aliases", []):
            result[alias.lower()] = pipeline_col

    # Also include split_fields labels as known (mapped to None)
    split_fields = section_data.get("split_fields", {})
    for label in split_fields:
        result.setdefault(label.lower(), None)

    return result


# ── Section B — Reconcile audit against map ──────────────────────────────────


def bigram_similarity(a: str, b: str) -> float:
    """Character bigram overlap similarity (Jaccard on bigrams)."""
    def bigrams(s: str) -> set:
        return set(s[i : i + 2] for i in range(len(s) - 1))

    ba, bb = bigrams(a.lower()), bigrams(b.lower())
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


@dataclass
class FieldDrift:
    drift_type: str                    # RENAMED / NEW_FIELD / DROPPED / UNMAPPED / SPLIT_NEEDED
    severity: str                      # CRITICAL / WARNING / INFO
    espn_label: str                    # what ESPN sent (lowered)
    espn_raw: str                      # original casing from API
    current_col: Optional[str]         # what pipeline currently maps it to
    suggested_col: Optional[str]       # what reconciler thinks it should be
    fuzzy_score: Optional[float]       # 0-1 similarity if fuzzy matched
    pipeline_col_affected: Optional[str]  # which pipeline column will be empty if unresolved
    auto_fixable: bool                 # True if reconciler can fix without human review
    notes: str


def _normalize_label(label: str) -> str:
    """Normalize label for matching: lowercase, strip spaces and special chars."""
    return re.sub(r"[^a-z0-9/+-]", "", str(label).strip().lower())


def _looks_like_split(value: str) -> bool:
    """Check if a value looks like a combined made-attempted string (e.g. '28-52')."""
    return bool(re.match(r"^\d+-\d+$", str(value).strip()))


def reconcile(
    audit_path: str = "data/espn_api_field_audit.json",
    col_map: Optional[dict] = None,
    verbose: bool = True,
) -> List[FieldDrift]:
    """
    Compare audit labels to column map.
    Returns list of FieldDrift objects, sorted by severity.
    """
    if col_map is None:
        col_map = load_column_map()

    audit_file = Path(audit_path)
    if not audit_file.exists():
        if verbose:
            print(f"[WARN] Audit file not found: {audit_path}")
        return []

    audit = json.loads(audit_file.read_text())
    drifts: List[FieldDrift] = []

    # -- Player stat labels --
    player_labels = audit.get("player_stat_labels", [])
    known_player = get_all_known_labels(col_map, "player_stats")
    split_fields = col_map.get("player_stats", {}).get("split_fields", {})
    required_player = set(
        col_map.get("pipeline_required_cols", {}).get("player", [])
    )

    if verbose:
        team_labels = audit.get("team_stat_labels", [])
        print(
            f"[RECONCILE] Loaded {len(player_labels)} player labels from audit, "
            f"{len(team_labels)} team labels"
        )

    # Track which pipeline columns are produced by any audit label
    produced_cols: set = set()

    for entry in player_labels:
        label_lower = entry.get("lower", "").strip()
        label_raw = entry.get("raw", label_lower)
        normalized = _normalize_label(label_lower)

        # Look up in known labels
        matched_col = None
        found = False
        for candidate in (label_lower, normalized):
            if candidate in known_player:
                matched_col = known_player[candidate]
                found = True
                break

        if found:
            if matched_col is not None:
                produced_cols.add(matched_col)
            # Check if this is a split field
            if label_lower in split_fields or normalized in split_fields:
                sf = split_fields.get(label_lower) or split_fields.get(normalized)
                if sf:
                    for oc in sf.get("output_cols", []):
                        produced_cols.add(oc)
            continue

        # Not found — run fuzzy match
        best_score = 0.0
        best_label = ""
        best_col: Optional[str] = None
        for known_lbl, known_col in known_player.items():
            score = bigram_similarity(label_lower, known_lbl)
            if score > best_score:
                best_score = score
                best_label = known_lbl
                best_col = known_col

        if best_score >= 0.85:
            drifts.append(FieldDrift(
                drift_type="RENAMED",
                severity="WARNING",
                espn_label=label_lower,
                espn_raw=label_raw,
                current_col=None,
                suggested_col=best_col,
                fuzzy_score=round(best_score, 3),
                pipeline_col_affected=best_col,
                auto_fixable=True,
                notes=f"Fuzzy match to '{best_label}' (score={best_score:.3f})",
            ))
            if best_col:
                produced_cols.add(best_col)
        elif best_score >= 0.60:
            drifts.append(FieldDrift(
                drift_type="RENAMED",
                severity="WARNING",
                espn_label=label_lower,
                espn_raw=label_raw,
                current_col=None,
                suggested_col=best_col,
                fuzzy_score=round(best_score, 3),
                pipeline_col_affected=best_col,
                auto_fixable=False,
                notes=f"Possible match to '{best_label}' (score={best_score:.3f}) — needs review",
            ))
        else:
            drifts.append(FieldDrift(
                drift_type="NEW_FIELD",
                severity="INFO",
                espn_label=label_lower,
                espn_raw=label_raw,
                current_col=None,
                suggested_col=None,
                fuzzy_score=round(best_score, 3) if best_score > 0 else None,
                pipeline_col_affected=None,
                auto_fixable=False,
                notes=f"New ESPN field — no pipeline mapping exists",
            ))

    # -- Team stat labels --
    team_labels = audit.get("team_stat_labels", [])
    known_team = get_all_known_labels(col_map, "team_stats")
    required_team = set(
        col_map.get("pipeline_required_cols", {}).get("team", [])
    )

    for entry in team_labels:
        abbr = entry.get("abbreviation", "")
        name = entry.get("name", "")
        label_lower = abbr.lower().replace(" ", "_") if abbr else name.lower().replace(" ", "_")
        normalized = _normalize_label(label_lower)

        # Check for shooting split
        dv = entry.get("displayValue", "")
        name_lower = name.lower()
        is_shooting = any(x in name_lower for x in ["goal", "throw", "three"])
        if is_shooting and _looks_like_split(dv):
            # Shooting splits produce fgm/fga, tpm/tpa, ftm/fta
            if "three" in name_lower or "3" in name_lower:
                produced_cols.update(["tpm", "tpa"])
            elif "free" in name_lower:
                produced_cols.update(["ftm", "fta"])
            else:
                produced_cols.update(["fgm", "fga"])
            continue

        # Look up in known labels
        found = False
        for candidate in (label_lower, normalized, name_lower.replace(" ", "")):
            if candidate in known_team:
                col = known_team[candidate]
                if col:
                    produced_cols.add(col)
                found = True
                break

        if not found and not is_shooting:
            best_score = 0.0
            best_label = ""
            for known_lbl in known_team:
                score = bigram_similarity(label_lower, known_lbl)
                if score > best_score:
                    best_score = score
                    best_label = known_lbl

            if best_score < 0.60:
                drifts.append(FieldDrift(
                    drift_type="NEW_FIELD",
                    severity="INFO",
                    espn_label=label_lower,
                    espn_raw=name or abbr,
                    current_col=None,
                    suggested_col=None,
                    fuzzy_score=round(best_score, 3) if best_score > 0 else None,
                    pipeline_col_affected=None,
                    auto_fixable=False,
                    notes=f"New ESPN team field — no pipeline mapping",
                ))

    # -- Check for dropped required columns --
    for req_col in sorted(required_player):
        if req_col not in produced_cols:
            drifts.append(FieldDrift(
                drift_type="DROPPED",
                severity="CRITICAL",
                espn_label="",
                espn_raw="",
                current_col=req_col,
                suggested_col=None,
                fuzzy_score=None,
                pipeline_col_affected=req_col,
                auto_fixable=False,
                notes=f"Required column '{req_col}' not produced by any audit label",
            ))

    # Sort by severity: CRITICAL first, then WARNING, then INFO
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    drifts.sort(key=lambda d: severity_order.get(d.severity, 3))

    if verbose:
        critical_count = sum(1 for d in drifts if d.severity == "CRITICAL")
        warn_count = sum(1 for d in drifts if d.severity == "WARNING")
        info_count = sum(1 for d in drifts if d.severity == "INFO")

        if not drifts:
            print(f"[RECONCILE] All {len(required_player)} required player columns mapped ✓")
        else:
            for d in drifts:
                if d.severity == "CRITICAL":
                    print(
                        f"[CRITICAL]  Required column \"{d.pipeline_col_affected}\" "
                        f"unmapped — will be empty in player_game_metrics.csv"
                    )
                elif d.severity == "WARNING":
                    print(
                        f"[WARN]      {d.drift_type}: \"{d.espn_raw}\" "
                        f"(lower: \"{d.espn_label}\") — {d.notes}"
                    )
                else:
                    print(
                        f"[INFO]      New ESPN field detected: \"{d.espn_raw}\" "
                        f"(lower: \"{d.espn_label}\") — no pipeline mapping"
                    )
            if not critical_count:
                print(f"[RECONCILE] All {len(required_player)} required player columns mapped ✓")
            print(
                f"[RECONCILE] Drift summary: {critical_count} critical, "
                f"{warn_count} warnings, {info_count} info"
            )

    return drifts


# ── Section C — Auto-patch suggestions ───────────────────────────────────────


def generate_patch_suggestions(drifts: List[FieldDrift]) -> dict:
    """
    For auto_fixable drifts, generate a suggested patch to column_map.json.
    Returns a dict of suggested additions — never writes directly.
    Human must review and apply via --apply-suggestions flag.
    """
    suggestions: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "player_stats": {"espn_labels": {}},
        "team_stats": {"espn_labels": {}},
    }
    for d in drifts:
        if not d.auto_fixable:
            continue
        entry = {
            "pipeline_col": d.suggested_col,
            "aliases": [d.espn_raw],
            "note": f"Auto-suggested: {d.notes}",
        }
        suggestions["player_stats"]["espn_labels"][d.espn_label] = entry

    return suggestions


def apply_suggestions(
    suggestions: dict,
    map_path: str = "column_map.json",
) -> None:
    """Apply auto_fixable suggestions to column_map.json."""
    assert "overrides" not in map_path, "Cannot write to overrides file"

    map_file = Path(map_path)
    if not map_file.exists():
        print(f"[ERROR] {map_path} not found")
        return

    col_map = json.loads(map_file.read_text())

    for section in ("player_stats", "team_stats"):
        new_labels = suggestions.get(section, {}).get("espn_labels", {})
        if not new_labels:
            continue
        existing = col_map.setdefault(section, {}).setdefault("espn_labels", {})
        for label, entry in new_labels.items():
            if label not in existing:
                existing[label] = entry
                print(f"[PATCH] Added '{label}' -> '{entry.get('pipeline_col')}' to {section}")

    col_map["updated_at"] = datetime.now(timezone.utc).isoformat()
    map_file.write_text(json.dumps(col_map, indent=2) + "\n")
    print(f"[OK] {map_path} updated")


def write_drift_report(
    drifts: List[FieldDrift],
    output_path: str = "data/field_drift_report.json",
) -> None:
    """
    Write full drift report to JSON.
    Also write a human-readable summary to data/field_drift_report.md.
    """
    assert "overrides" not in output_path, "Cannot write to overrides file"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_drifts": len(drifts),
        "critical": sum(1 for d in drifts if d.severity == "CRITICAL"),
        "warnings": sum(1 for d in drifts if d.severity == "WARNING"),
        "info": sum(1 for d in drifts if d.severity == "INFO"),
        "drifts": [asdict(d) for d in drifts],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[OK]        {output_path} written")

    # Markdown summary
    md_path = out.with_suffix(".md")
    lines = [
        "# Field Drift Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"| Severity | Count |",
        f"|----------|-------|",
        f"| CRITICAL | {report['critical']} |",
        f"| WARNING  | {report['warnings']} |",
        f"| INFO     | {report['info']} |",
        "",
    ]
    if drifts:
        lines.append("## Details\n")
        for d in drifts:
            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(d.severity, "⚪")
            lines.append(
                f"- {icon} **{d.severity}** [{d.drift_type}] "
                f"`{d.espn_raw}` → `{d.suggested_col or '?'}` — {d.notes}"
            )
    else:
        lines.append("✅ No drift detected — all fields mapped correctly.\n")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"[OK]        {md_path} written")


# ── Section D — Pipeline integration check ───────────────────────────────────


def check_pipeline_output(
    player_metrics_path: str = "data/player_game_metrics.csv",
    team_metrics_path: str = "data/team_game_metrics.csv",
    drifts: Optional[List[FieldDrift]] = None,
) -> dict:
    """
    After pipeline runs, verify the output CSVs actually have data
    in the columns that matter. Catches the silent-empty-column failure.
    """
    col_map = load_column_map()
    required_player = col_map.get("pipeline_required_cols", {}).get("player", [])
    required_team = col_map.get("pipeline_required_cols", {}).get("team", [])

    result: Dict[str, Any] = {
        "player_metrics": {},
        "team_metrics": {},
        "critical_failures": [],
        "recommended_actions": [],
    }

    # Check player metrics
    player_path = Path(player_metrics_path)
    if player_path.exists() and player_path.stat().st_size > 10:
        try:
            import pandas as pd
            df = pd.read_csv(player_path, low_memory=False)
            total_rows = len(df)

            for col in required_player:
                col_info: Dict[str, Any] = {"status": "OK"}

                if col not in df.columns:
                    col_info["status"] = "MISSING"
                    col_info["null_pct"] = 1.0
                    result["critical_failures"].append(col)
                    print(f"[CRITICAL]  Player column '{col}' missing from CSV")
                else:
                    null_count = df[col].isna().sum()
                    null_pct = null_count / max(total_rows, 1)
                    col_info["null_pct"] = round(null_pct, 4)

                    # Check for uppercase twin
                    upper_col = col.upper()
                    has_twin = upper_col in df.columns and upper_col != col
                    col_info["has_uppercase_twin"] = has_twin

                    if has_twin:
                        upper_null = df[upper_col].isna().sum()
                        upper_pct = upper_null / max(total_rows, 1)
                        if null_pct > 0.80 and upper_pct < 0.50:
                            col_info["status"] = "CRITICAL"
                            col_info["note"] = (
                                f"lowercase '{col}' is {null_pct:.0%} null but "
                                f"uppercase '{upper_col}' is {upper_pct:.0%} null"
                            )
                            result["critical_failures"].append(col)
                            result["recommended_actions"].append(
                                f"Copy {upper_col} values to {col}, drop {upper_col}"
                            )
                            print(
                                f"[CRITICAL]  Uppercase duplicate detected: {upper_col} has data, "
                                f"{col} is empty → run --fix-duplicates"
                            )
                        elif null_pct > 0.80:
                            col_info["status"] = "CRITICAL"
                            result["critical_failures"].append(col)
                            print(f"[CRITICAL]  Column '{col}' is {null_pct:.0%} null")
                    elif null_pct > 0.80:
                        col_info["status"] = "CRITICAL"
                        result["critical_failures"].append(col)
                        print(f"[CRITICAL]  Column '{col}' is {null_pct:.0%} null")
                    elif null_pct > 0.50:
                        col_info["status"] = "WARNING"
                        print(f"[WARN]      Column '{col}' is {null_pct:.0%} null")

                result["player_metrics"][col] = col_info
        except Exception as exc:
            print(f"[ERROR] Failed to check {player_metrics_path}: {exc}")
    else:
        print(f"[WARN] {player_metrics_path} not found or empty")

    # Check team metrics
    team_path = Path(team_metrics_path)
    if team_path.exists() and team_path.stat().st_size > 10:
        try:
            import pandas as pd
            df = pd.read_csv(team_path, low_memory=False)
            total_rows = len(df)

            for col in required_team:
                col_info = {"status": "OK"}

                if col not in df.columns:
                    col_info["status"] = "MISSING"
                    col_info["null_pct"] = 1.0
                    print(f"[WARN]      Team column '{col}' missing from CSV")
                else:
                    null_count = df[col].isna().sum()
                    null_pct = null_count / max(total_rows, 1)
                    col_info["null_pct"] = round(null_pct, 4)
                    if null_pct > 0.80:
                        col_info["status"] = "CRITICAL"
                        print(f"[WARN]      Team column '{col}' is {null_pct:.0%} null")

                result["team_metrics"][col] = col_info
        except Exception as exc:
            print(f"[ERROR] Failed to check {team_metrics_path}: {exc}")
    else:
        print(f"[WARN] {team_metrics_path} not found or empty")

    if not result["critical_failures"]:
        print("[OK]        All required columns have data ✓")

    return result


# ── Section E — Column deduplication fix ─────────────────────────────────────


def fix_uppercase_duplicate_columns(
    csv_path: str,
    dry_run: bool = False,
) -> dict:
    """
    Detects and fixes the specific fgm/FGM duplication bug.

    Logic:
    1. Load CSV, identify column pairs where both lowercase and uppercase exist
    2. For each pair:
       a. If lowercase is empty and uppercase has data → copy uppercase to lowercase, drop uppercase
       b. If both have data → keep lowercase, drop uppercase
       c. If both empty → keep lowercase, drop uppercase
    3. Write fixed CSV back to same path (or dry_run: print what would happen)
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.exists() or path.stat().st_size < 10:
        print(f"[WARN] {csv_path} not found or empty")
        return {"status": "skipped", "reason": "file not found"}

    df = pd.read_csv(path, low_memory=False)
    summary: Dict[str, Any] = {"fixed": [], "kept_lowercase": [], "both_empty": []}

    # Find lowercase/uppercase pairs
    lower_cols = {c for c in df.columns if c == c.lower() and c != c.upper()}
    upper_cols = {c for c in df.columns if c == c.upper() and c != c.lower()}

    pairs_found = []
    for lc in lower_cols:
        uc = lc.upper()
        if uc in upper_cols:
            pairs_found.append((lc, uc))

    if not pairs_found:
        print(f"[OK]        No uppercase duplicate columns found in {path.name}")
        return {"status": "clean", "pairs": []}

    drop_cols = []
    for lc, uc in pairs_found:
        lc_null = df[lc].isna().sum()
        uc_null = df[uc].isna().sum()
        total = len(df)
        lc_pct = lc_null / max(total, 1)
        uc_pct = uc_null / max(total, 1)

        if lc_pct > 0.80 and uc_pct < 0.50:
            # Case a: lowercase empty, uppercase has data → copy
            action = f"Copy {uc} → {lc} (lc {lc_pct:.0%} null, uc {uc_pct:.0%} null)"
            if not dry_run:
                df[lc] = df[uc]
            summary["fixed"].append({"lowercase": lc, "uppercase": uc, "action": action})
            drop_cols.append(uc)
            print(f"[FIX]       Uppercase duplicate detected: {uc} has data, {lc} is empty → copying values")
        elif lc_pct <= 0.80:
            # Case b: both have data → keep lowercase
            action = f"Keep {lc}, drop {uc} (both have data)"
            summary["kept_lowercase"].append({"lowercase": lc, "uppercase": uc, "action": action})
            drop_cols.append(uc)
            if not dry_run:
                pass  # just drop uppercase
            print(f"[FIX]       Keeping {lc} (has data), dropping duplicate {uc}")
        else:
            # Case c: both empty
            action = f"Keep {lc}, drop {uc} (both empty)"
            summary["both_empty"].append({"lowercase": lc, "uppercase": uc, "action": action})
            drop_cols.append(uc)
            print(f"[FIX]       Both {lc} and {uc} empty — keeping {lc}, dropping {uc}")

    if not dry_run and drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
        df.to_csv(path, index=False)
        print(f"[OK]        {path.name} rewritten — dropped {len(drop_cols)} uppercase columns")
    elif dry_run:
        print(f"[DRY-RUN]   Would drop {len(drop_cols)} columns from {path.name}: {drop_cols}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESPN API field reconciliation and drift detection"
    )
    parser.add_argument(
        "--audit-path",
        default="data/espn_api_field_audit.json",
        help="Path to ESPN API audit JSON",
    )
    parser.add_argument(
        "--check-outputs",
        action="store_true",
        help="Check existing CSVs for empty columns",
    )
    parser.add_argument(
        "--fix-duplicates",
        action="store_true",
        help="Fix uppercase/lowercase duplicate columns in CSVs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    parser.add_argument(
        "--apply-suggestions",
        action="store_true",
        help="Apply auto_fixable drift patches to column_map.json",
    )
    parser.add_argument(
        "--update-map",
        action="store_true",
        help="Interactively review and accept/reject drift suggestions",
    )

    args = parser.parse_args()

    # If only --check-outputs, skip reconciliation
    if args.check_outputs and not args.fix_duplicates:
        result = check_pipeline_output()
        if result["critical_failures"]:
            print(f"\n[SUMMARY] {len(result['critical_failures'])} critical column failures detected")
            sys.exit(1)
        return

    # If only --fix-duplicates
    if args.fix_duplicates and not args.check_outputs:
        for csv_path in ("data/player_game_metrics.csv", "data/player_game_logs.csv",
                         "data/team_game_metrics.csv", "data/team_game_logs.csv"):
            fix_uppercase_duplicate_columns(csv_path, dry_run=args.dry_run)
        return

    # Both --check-outputs and --fix-duplicates
    if args.check_outputs and args.fix_duplicates:
        for csv_path in ("data/player_game_metrics.csv", "data/player_game_logs.csv",
                         "data/team_game_metrics.csv", "data/team_game_logs.csv"):
            fix_uppercase_duplicate_columns(csv_path, dry_run=args.dry_run)
        result = check_pipeline_output()
        if result["critical_failures"]:
            print(f"\n[SUMMARY] {len(result['critical_failures'])} critical column failures remain after fix")
        return

    # Full reconcile
    drifts = reconcile(audit_path=args.audit_path)

    # Write drift report
    Path("data").mkdir(parents=True, exist_ok=True)
    write_drift_report(drifts)

    if args.apply_suggestions:
        suggestions = generate_patch_suggestions(drifts)
        auto_count = sum(
            len(v.get("espn_labels", {}))
            for v in suggestions.values()
            if isinstance(v, dict)
        )
        if auto_count:
            apply_suggestions(suggestions)
        else:
            print("[OK]        No auto-fixable suggestions to apply")

    if args.update_map:
        suggestions = generate_patch_suggestions(drifts)
        non_auto = [d for d in drifts if not d.auto_fixable and d.drift_type == "RENAMED"]
        if not non_auto:
            print("[OK]        No suggestions requiring manual review")
        else:
            for d in non_auto:
                print(
                    f"\n[REVIEW] '{d.espn_raw}' → suggested: '{d.suggested_col}' "
                    f"(score={d.fuzzy_score})"
                )
                resp = input("  Accept? [y/N]: ").strip().lower()
                if resp == "y":
                    suggestions.setdefault("player_stats", {}).setdefault(
                        "espn_labels", {}
                    )[d.espn_label] = {
                        "pipeline_col": d.suggested_col,
                        "aliases": [d.espn_raw],
                        "note": f"Manually accepted: {d.notes}",
                    }
            apply_suggestions(suggestions)


if __name__ == "__main__":
    main()
