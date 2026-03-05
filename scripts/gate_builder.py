#!/usr/bin/env python3
"""Evaluate additive betting gates and recommend a default gate set."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


WIN_PROFIT_UNITS = 100.0 / 110.0

GATE_RESULTS_COLUMNS = [
    "gate_family",
    "gate_name",
    "parameter",
    "sample_size_before",
    "sample_size_after",
    "win_rate_before",
    "win_rate_after",
    "roi_before",
    "roi_after",
    "roi_delta",
    "pass_rate",
    "enabled",
    "excluded",
    "excluded_reason",
    "input_source_file",
    "generated_at_utc",
]

BEST_RULES_COLUMNS = [
    "rule_order",
    "gate_family",
    "gate_name",
    "parameter",
    "enabled",
    "selection_reason",
    "combined_sample_size",
    "combined_win_rate",
    "combined_roi",
    "generated_at_utc",
]

DEFAULT_CONFIG = {
    "enabled": {
        "min_edge": True,
        "volatility_block": True,
        "low_total_block": True,
        "sharp_confirmation": True,
        "overconfidence_block": True,
    },
    "min_edge_thresholds": [1.5, 2.5, 3.5, 4.5],
    "volatility_block_tiers": ["high"],
    "low_total_thresholds": [128.0, 132.0, 136.0],
    "sharp_score_thresholds": [50.0, 60.0, 70.0],
    "min_sample_after": 25,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pick_first(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _safe_row_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except Exception:
        return 0


@dataclass
class SourceSpec:
    path: Path
    event_col: str
    game_col: str
    dt_col: str
    model_spread_col: str
    market_spread_col: str
    actual_margin_col: str
    market_total_col: str | None
    home_team_id_col: str | None
    away_team_id_col: str | None


def _discover_source(data_dir: Path) -> tuple[SourceSpec | None, list[str], dict[str, list[str]]]:
    candidates = [
        "results_log.csv",
        "results_log_graded.csv",
        "predictions_graded.csv",
    ]
    inspected: list[str] = []
    missing: dict[str, list[str]] = {}
    valid: list[tuple[tuple[int, int], SourceSpec]] = []

    for idx, name in enumerate(candidates):
        path = data_dir / name
        if not path.exists() or path.stat().st_size <= 1:
            inspected.append(f"{name}:missing_or_empty")
            continue
        try:
            cols = list(pd.read_csv(path, nrows=0, low_memory=False).columns)
        except Exception:
            inspected.append(f"{name}:unreadable")
            continue

        event_col = _pick_first(cols, ["event_id", "game_id"])
        game_col = _pick_first(cols, ["game_id", "event_id"])
        dt_col = _pick_first(cols, ["game_datetime_utc", "game_date"])
        model_col = _pick_first(cols, ["pred_spread", "predicted_spread", "model_spread", "ens_spread"])
        market_col = _pick_first(cols, ["market_spread", "spread_line"])
        actual_col = _pick_first(cols, ["actual_margin"])

        required_missing: list[str] = []
        if event_col is None:
            required_missing.append("event_id|game_id")
        if game_col is None:
            required_missing.append("game_id|event_id")
        if dt_col is None:
            required_missing.append("game_datetime_utc|game_date")
        if model_col is None:
            required_missing.append("pred_spread|predicted_spread|model_spread|ens_spread")
        if market_col is None:
            required_missing.append("market_spread|spread_line")
        if actual_col is None:
            required_missing.append("actual_margin")

        if required_missing:
            missing[str(path)] = required_missing
            inspected.append(f"{name}:missing_required")
            continue

        rows = _safe_row_count(path)
        spec = SourceSpec(
            path=path,
            event_col=event_col or "",
            game_col=game_col or "",
            dt_col=dt_col or "",
            model_spread_col=model_col or "",
            market_spread_col=market_col or "",
            actual_margin_col=actual_col or "",
            market_total_col=_pick_first(cols, ["market_total", "total_line", "over_under"]),
            home_team_id_col=_pick_first(cols, ["home_team_id"]),
            away_team_id_col=_pick_first(cols, ["away_team_id"]),
        )
        valid.append(((rows, -idx), spec))
        inspected.append(f"{name}:valid:rows={rows}")

    if not valid:
        return None, inspected, missing
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1], inspected, missing


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if not path.exists():
        return config
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return config

    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config[key])  # type: ignore[index]
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    return config


def _load_volatility(data_dir: Path) -> tuple[pd.DataFrame, str]:
    p = data_dir / "teams" / "team_volatility.csv"
    if not p.exists():
        return pd.DataFrame(columns=["event_id", "team_id", "volatility_tier"]), "missing_file"
    try:
        df = pd.read_csv(p, usecols=["event_id", "team_id", "volatility_tier"], low_memory=False)
    except ValueError:
        return pd.DataFrame(columns=["event_id", "team_id", "volatility_tier"]), "missing_columns"
    df["event_id"] = df["event_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    df = df.drop_duplicates(subset=["event_id", "team_id"], keep="last")
    return df, "ok"


def _load_sharp_score(data_dir: Path) -> tuple[pd.DataFrame, str]:
    p = data_dir / "market" / "sharp_signal_score.csv"
    if not p.exists():
        return pd.DataFrame(columns=["event_id", "team_id", "sharp_score"]), "missing_file"
    try:
        header = list(pd.read_csv(p, nrows=0, low_memory=False).columns)
    except Exception:
        return pd.DataFrame(columns=["event_id", "team_id", "sharp_score"]), "unreadable"
    event_col = _pick_first(header, ["event_id", "game_id"])
    team_col = _pick_first(header, ["team_id"])
    score_col = _pick_first(header, ["sharp_score", "score"])
    if event_col is None or score_col is None:
        return pd.DataFrame(columns=["event_id", "team_id", "sharp_score"]), "missing_columns"
    usecols = [event_col, score_col]
    if team_col:
        usecols.append(team_col)
    df = pd.read_csv(p, usecols=sorted(set(usecols)), low_memory=False)
    out = pd.DataFrame()
    out["event_id"] = df[event_col].astype(str)
    out["team_id"] = df[team_col].astype(str) if team_col else ""
    out["sharp_score"] = pd.to_numeric(df[score_col], errors="coerce")
    out = out.drop_duplicates(subset=["event_id", "team_id"], keep="last")
    return out, "ok"


def _load_overconfidence(data_dir: Path) -> tuple[pd.DataFrame, str]:
    p = data_dir / "analytics" / "overconfidence_report.csv"
    if not p.exists():
        return pd.DataFrame(columns=["event_id", "overconfidence_flag"]), "missing_file"
    try:
        header = list(pd.read_csv(p, nrows=0, low_memory=False).columns)
    except Exception:
        return pd.DataFrame(columns=["event_id", "overconfidence_flag"]), "unreadable"
    event_col = _pick_first(header, ["event_id", "game_id"])
    flag_col = _pick_first(header, ["overconfidence_flag", "is_overconfidence", "flag"])
    if event_col is None or flag_col is None:
        return pd.DataFrame(columns=["event_id", "overconfidence_flag"]), "missing_columns"
    df = pd.read_csv(p, usecols=[event_col, flag_col], low_memory=False)
    out = pd.DataFrame()
    out["event_id"] = df[event_col].astype(str)
    low = df[flag_col].astype(str).str.strip().str.lower()
    out["overconfidence_flag"] = low.isin({"1", "true", "t", "yes", "y"})
    out = out.drop_duplicates(subset=["event_id"], keep="last")
    return out, "ok"


def _metrics(df: pd.DataFrame) -> tuple[int, float, float]:
    if df.empty:
        return 0, float("nan"), float("nan")
    wins = int((df["result"] == "win").sum())
    losses = int((df["result"] == "loss").sum())
    denom = wins + losses
    win_rate = float(wins / denom) if denom > 0 else float("nan")
    roi = float(df["profit_units"].mean()) if len(df) else float("nan")
    return int(len(df)), win_rate, roi


def _build_base_bets(spec: SourceSpec, data_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    usecols = [spec.event_col, spec.game_col, spec.dt_col, spec.model_spread_col, spec.market_spread_col, spec.actual_margin_col]
    if spec.market_total_col:
        usecols.append(spec.market_total_col)
    if spec.home_team_id_col:
        usecols.append(spec.home_team_id_col)
    if spec.away_team_id_col:
        usecols.append(spec.away_team_id_col)
    raw = pd.read_csv(spec.path, usecols=sorted(set(usecols)), low_memory=False)

    out = pd.DataFrame()
    out["event_id"] = raw[spec.event_col].astype(str)
    out["game_id"] = raw[spec.game_col].astype(str)
    out["game_datetime_utc"] = pd.to_datetime(raw[spec.dt_col], utc=True, errors="coerce")
    out["model_spread"] = pd.to_numeric(raw[spec.model_spread_col], errors="coerce")
    out["market_spread"] = pd.to_numeric(raw[spec.market_spread_col], errors="coerce")
    out["actual_margin"] = pd.to_numeric(raw[spec.actual_margin_col], errors="coerce")
    out["market_total"] = pd.to_numeric(raw[spec.market_total_col], errors="coerce") if spec.market_total_col else np.nan
    out["home_team_id"] = raw[spec.home_team_id_col].astype(str) if spec.home_team_id_col else ""
    out["away_team_id"] = raw[spec.away_team_id_col].astype(str) if spec.away_team_id_col else ""

    out = out.dropna(subset=["model_spread", "market_spread", "actual_margin"])
    out = out.drop_duplicates(subset=["event_id", "game_id"], keep="last")
    out["edge"] = out["model_spread"] - out["market_spread"]
    out = out[out["edge"] != 0].copy()
    out["abs_edge"] = out["edge"].abs()
    out["cover_margin"] = out["actual_margin"] - out["market_spread"]
    prod = out["edge"] * out["cover_margin"]
    out["result"] = np.where(prod < 0, "win", np.where(prod > 0, "loss", "push"))
    out["profit_units"] = np.where(out["result"] == "win", WIN_PROFIT_UNITS, np.where(out["result"] == "loss", -1.0, 0.0))
    out["pick_team_id"] = np.where(out["edge"] < 0, out["home_team_id"], out["away_team_id"])

    statuses: dict[str, str] = {}
    vol, vol_status = _load_volatility(data_dir)
    statuses["volatility"] = vol_status
    if not vol.empty:
        out = out.merge(vol, left_on=["event_id", "pick_team_id"], right_on=["event_id", "team_id"], how="left")
    else:
        out["volatility_tier"] = np.nan

    sharp, sharp_status = _load_sharp_score(data_dir)
    statuses["sharp_score"] = sharp_status
    if not sharp.empty:
        team_merge = sharp[(sharp["team_id"] != "") & sharp["team_id"].notna()]
        if not team_merge.empty:
            out = out.merge(
                team_merge[["event_id", "team_id", "sharp_score"]],
                left_on=["event_id", "pick_team_id"],
                right_on=["event_id", "team_id"],
                how="left",
            )
        event_mean = sharp.groupby("event_id", dropna=False)["sharp_score"].mean().rename("sharp_score_event")
        out = out.merge(event_mean, on="event_id", how="left")
        out["sharp_score"] = out["sharp_score"].fillna(out["sharp_score_event"])
        out = out.drop(columns=[c for c in ["team_id", "sharp_score_event"] if c in out.columns])
    else:
        out["sharp_score"] = np.nan

    oc, oc_status = _load_overconfidence(data_dir)
    statuses["overconfidence"] = oc_status
    if not oc.empty:
        out = out.merge(oc, on="event_id", how="left")
    else:
        out["overconfidence_flag"] = False

    return out, statuses


def _evaluate_gates(base_df: pd.DataFrame, source_file: str, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    enabled = config.get("enabled", {}) if isinstance(config.get("enabled"), dict) else {}
    min_sample_after = int(config.get("min_sample_after", 25))

    before_n, before_wr, before_roi = _metrics(base_df)
    gate_rows: list[dict[str, object]] = []

    def add_gate(family: str, name: str, parameter: str, mask: pd.Series, *, feature_available: bool = True) -> None:
        if base_df.empty:
            after = base_df
        else:
            after = base_df[mask.fillna(False)] if feature_available else base_df.iloc[0:0]
        after_n, after_wr, after_roi = _metrics(after)
        excluded = (after_n < min_sample_after) or (not feature_available)
        reason = "missing_feature" if not feature_available else ("below_min_sample" if after_n < min_sample_after else "")
        gate_rows.append(
            {
                "gate_family": family,
                "gate_name": name,
                "parameter": parameter,
                "sample_size_before": before_n,
                "sample_size_after": after_n,
                "win_rate_before": before_wr,
                "win_rate_after": after_wr,
                "roi_before": before_roi,
                "roi_after": after_roi,
                "roi_delta": (after_roi - before_roi) if np.isfinite(after_roi) and np.isfinite(before_roi) else np.nan,
                "pass_rate": (after_n / before_n) if before_n else np.nan,
                "enabled": True,
                "excluded": excluded,
                "excluded_reason": reason,
                "input_source_file": source_file,
                "generated_at_utc": _utc_now(),
            }
        )

    if bool(enabled.get("min_edge", True)):
        for thr in config.get("min_edge_thresholds", [1.5, 2.5, 3.5, 4.5]):
            t = float(thr)
            add_gate("min_edge", f"min_edge_{t:g}", str(t), base_df["abs_edge"] >= t, feature_available=True)

    if bool(enabled.get("volatility_block", True)):
        blocked_tiers = [str(x).strip().lower() for x in config.get("volatility_block_tiers", ["high"])]
        has_feature = "volatility_tier" in base_df.columns and base_df["volatility_tier"].notna().any()
        mask = ~base_df["volatility_tier"].astype(str).str.lower().isin(blocked_tiers) if has_feature else pd.Series(False, index=base_df.index)
        add_gate(
            "volatility_block",
            "exclude_volatility_tiers",
            ",".join(blocked_tiers),
            mask,
            feature_available=has_feature,
        )

    if bool(enabled.get("low_total_block", True)):
        has_total = "market_total" in base_df.columns and base_df["market_total"].notna().any()
        for thr in config.get("low_total_thresholds", [128.0, 132.0, 136.0]):
            t = float(thr)
            mask = base_df["market_total"] >= t if has_total else pd.Series(False, index=base_df.index)
            add_gate("low_total_block", f"min_market_total_{t:g}", str(t), mask, feature_available=has_total)

    if bool(enabled.get("sharp_confirmation", True)):
        has_sharp = "sharp_score" in base_df.columns and base_df["sharp_score"].notna().any()
        for thr in config.get("sharp_score_thresholds", [50.0, 60.0, 70.0]):
            t = float(thr)
            mask = base_df["sharp_score"] >= t if has_sharp else pd.Series(False, index=base_df.index)
            add_gate("sharp_confirmation", f"min_sharp_score_{t:g}", str(t), mask, feature_available=has_sharp)

    if bool(enabled.get("overconfidence_block", True)):
        has_oc = "overconfidence_flag" in base_df.columns
        mask = ~base_df["overconfidence_flag"].fillna(False).astype(bool) if has_oc else pd.Series(False, index=base_df.index)
        add_gate("overconfidence_block", "exclude_overconfidence", "true", mask, feature_available=has_oc)

    results = pd.DataFrame(gate_rows, columns=GATE_RESULTS_COLUMNS)
    if results.empty:
        results = pd.DataFrame(columns=GATE_RESULTS_COLUMNS)
        return results, pd.DataFrame(columns=BEST_RULES_COLUMNS)

    chosen: list[pd.Series] = []
    for family, fam_df in results.groupby("gate_family", sort=False):
        eligible = fam_df[(~fam_df["excluded"]) & fam_df["roi_delta"].notna()]
        if eligible.empty:
            continue
        best = eligible.sort_values(["roi_delta", "sample_size_after"], ascending=[False, False]).iloc[0]
        if float(best["roi_delta"]) > 0:
            chosen.append(best)

    current = base_df.copy()
    best_rows: list[dict[str, object]] = []
    order = 1
    for row in chosen:
        fam = str(row["gate_family"])
        name = str(row["gate_name"])
        param = str(row["parameter"])
        if fam == "min_edge":
            thr = float(param)
            current = current[current["abs_edge"] >= thr].copy()
        elif fam == "volatility_block":
            blocked_tiers = [x.strip().lower() for x in param.split(",") if x.strip()]
            current = current[~current["volatility_tier"].astype(str).str.lower().isin(blocked_tiers)].copy()
        elif fam == "low_total_block":
            thr = float(param)
            current = current[current["market_total"] >= thr].copy()
        elif fam == "sharp_confirmation":
            thr = float(param)
            current = current[current["sharp_score"] >= thr].copy()
        elif fam == "overconfidence_block":
            current = current[~current["overconfidence_flag"].fillna(False).astype(bool)].copy()
        n, wr, roi = _metrics(current)
        best_rows.append(
            {
                "rule_order": order,
                "gate_family": fam,
                "gate_name": name,
                "parameter": param,
                "enabled": True,
                "selection_reason": "best_positive_roi_delta",
                "combined_sample_size": n,
                "combined_win_rate": wr,
                "combined_roi": roi,
                "generated_at_utc": _utc_now(),
            }
        )
        order += 1

    n, wr, roi = _metrics(current)
    best_rows.append(
        {
            "rule_order": 999,
            "gate_family": "combined_default",
            "gate_name": "recommended_default_set",
            "parameter": ";".join(str(r["gate_name"]) for r in chosen),
            "enabled": True,
            "selection_reason": "sequential_apply_selected_best_gates",
            "combined_sample_size": n,
            "combined_win_rate": wr,
            "combined_roi": roi,
            "generated_at_utc": _utc_now(),
        }
    )
    best = pd.DataFrame(best_rows, columns=BEST_RULES_COLUMNS)
    return results, best


def run_gate_builder(
    *,
    data_dir: Path,
    output_results_csv: Path,
    output_best_csv: Path,
    output_md: Path,
    config_path: Path,
) -> int:
    output_results_csv.parent.mkdir(parents=True, exist_ok=True)
    output_best_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    spec, inspected, missing = _discover_source(data_dir)
    if spec is None:
        pd.DataFrame(columns=GATE_RESULTS_COLUMNS).to_csv(output_results_csv, index=False)
        pd.DataFrame(columns=BEST_RULES_COLUMNS).to_csv(output_best_csv, index=False)
        lines = [
            "# Exec Summary: gate_builder",
            "",
            "- status: `BLOCKED`",
            f"- generated_at_utc: `{_utc_now()}`",
            "- reason: No source table with model/market/actual spread columns found.",
            "- inspected:",
        ]
        for item in inspected:
            lines.append(f"  - `{item}`")
        if missing:
            lines.append("- missing columns:")
            for f, cols in missing.items():
                lines.append(f"  - `{f}`: `{', '.join(cols)}`")
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1

    config = _load_config(config_path)
    base_df, statuses = _build_base_bets(spec, data_dir)
    if base_df.empty:
        pd.DataFrame(columns=GATE_RESULTS_COLUMNS).to_csv(output_results_csv, index=False)
        pd.DataFrame(columns=BEST_RULES_COLUMNS).to_csv(output_best_csv, index=False)
        lines = [
            "# Exec Summary: gate_builder",
            "",
            "- status: `BLOCKED`",
            f"- generated_at_utc: `{_utc_now()}`",
            f"- reason: `{spec.path.name}` loaded but no eligible spread bets after filtering.",
        ]
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1

    gate_results, best_rules = _evaluate_gates(base_df, str(spec.path), config)
    gate_results.to_csv(output_results_csv, index=False)
    best_rules.to_csv(output_best_csv, index=False)

    lines = [
        "# Exec Summary: gate_builder",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- input_source_file: `{spec.path}`",
        f"- base_bet_rows: `{len(base_df)}`",
        f"- gate_results_rows: `{len(gate_results)}`",
        f"- recommended_rules_rows: `{len(best_rules)}`",
        f"- config_path: `{config_path}`",
        "- optional_feature_status:",
    ]
    for key, value in statuses.items():
        lines.append(f"  - `{key}`: `{value}`")
    lines.append("- inspected_inputs:")
    for item in inspected:
        lines.append(f"  - `{item}`")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-results-csv", type=Path, default=Path("data/gates/gate_results.csv"))
    parser.add_argument("--output-best-csv", type=Path, default=Path("data/gates/gate_rules_best.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/gates/gate_exec_summary.md"))
    parser.add_argument("--config", type=Path, default=Path("config/gate_builder_context.yml"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_results_csv = args.output_results_csv if args.output_results_csv.is_absolute() else repo_root / args.output_results_csv
    output_best_csv = args.output_best_csv if args.output_best_csv.is_absolute() else repo_root / args.output_best_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md
    config_path = args.config if args.config.is_absolute() else repo_root / args.config

    rc = run_gate_builder(
        data_dir=data_dir,
        output_results_csv=output_results_csv,
        output_best_csv=output_best_csv,
        output_md=output_md,
        config_path=config_path,
    )
    print(
        json.dumps(
            {
                "output_results_csv": str(output_results_csv),
                "output_best_csv": str(output_best_csv),
                "output_md": str(output_md),
                "exit_code": rc,
            }
        )
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
