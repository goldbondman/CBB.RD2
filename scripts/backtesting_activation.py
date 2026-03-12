from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_registry import (
    get_tier,
    load_registry,
    save_registry,
    update_registry_from_backtest,
)
from pes_backtesting import BacktestConfig, run_pes_full_backtest
from pes_metric import PESConfig, compute_pes_matchup
from phase_common import (
    build_side_rows_from_backtest,
    build_side_rows_from_warehouse,
    ensure_parent,
    merge_and_dedupe_side_rows,
    resolve_paths,
    safe_read_csv,
    safe_read_parquet,
    to_num,
)
from situational_layer_backtesting import ColumnConfig, run_full_backtest


log = logging.getLogger("backtesting_activation")
logging.basicConfig(level=logging.INFO, format="[PHASE2] %(levelname)-8s %(message)s")


def _wilson_ci(hits: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = float(stats.norm.ppf((1 + confidence) / 2))
    phat = hits / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return float(center - half), float(center + half)


def _to_int_or_zero(value: Any) -> int:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return 0
    return int(num)


def _status_from_stats(hit_rate: float, lift: float, p_value: float, n: int, min_sample: int) -> str:
    if n < min_sample:
        return "PROVISIONAL"
    if pd.isna(hit_rate) or pd.isna(lift):
        return "INCONCLUSIVE"
    if lift > 0 and pd.notna(p_value) and p_value <= 0.05:
        return "VALIDATED"
    if lift > 0:
        return "INCONCLUSIVE"
    return "FAILED"


def _extract_situational_rows(result: dict[str, Any], min_sample: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    over = result.get("over", pd.DataFrame())
    for _, r in over.iterrows():
        n = _to_int_or_zero(r.get("layered_n"))
        hit = pd.to_numeric(r.get("layered_hit_rate"), errors="coerce")
        lift = pd.to_numeric(r.get("lift"), errors="coerce")
        pval = pd.to_numeric(r.get("p_value"), errors="coerce")
        hits = int(round(float(hit) * n)) if pd.notna(hit) else 0
        ci_low, ci_high = _wilson_ci(hits, n)
        rows.append(
            {
                "layer_name": str(r.get("label")),
                "scenario": "situational_over",
                "market": "total",
                "status": _status_from_stats(hit, lift, pval, n, min_sample),
                "hit_rate": hit,
                "lift": lift,
                "p_value": pval,
                "n": n,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "season_consistent": pd.to_numeric(r.get("season_positive_ratio"), errors="coerce"),
                "notes": str(r.get("verdict", "")),
            }
        )

    dog = result.get("underdog", pd.DataFrame())
    for _, r in dog.iterrows():
        for market, n_col, hr_col, lift_col, p_col, season_col, verdict_col in [
            ("ats", "ats_n", "ats_hit_rate", "ats_lift", "ats_p_value", "season_positive_ratio_ats", "ats_verdict"),
            ("ml", "ml_n", "ml_hit_rate", "ml_lift", "ml_p_value", "season_positive_ratio_ml", "ml_verdict"),
        ]:
            n = _to_int_or_zero(r.get(n_col))
            hit = pd.to_numeric(r.get(hr_col), errors="coerce")
            lift = pd.to_numeric(r.get(lift_col), errors="coerce")
            pval = pd.to_numeric(r.get(p_col), errors="coerce")
            hits = int(round(float(hit) * n)) if pd.notna(hit) else 0
            ci_low, ci_high = _wilson_ci(hits, n)
            rows.append(
                {
                    "layer_name": str(r.get("layer")),
                    "scenario": "situational_underdog",
                    "market": market,
                    "status": _status_from_stats(hit, lift, pval, n, min_sample),
                    "hit_rate": hit,
                    "lift": lift,
                    "p_value": pval,
                    "n": n,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "season_consistent": pd.to_numeric(r.get(season_col), errors="coerce"),
                    "notes": str(r.get(verdict_col, "")),
                }
            )

    def _rows_from_block(key: str, scenario: str, market_name: str) -> None:
        block = result.get(key, {})
        frame = block.get("results", pd.DataFrame()) if isinstance(block, dict) else pd.DataFrame()
        if frame.empty:
            return
        for _, r in frame.iterrows():
            n = _to_int_or_zero(r.get("layered_n", r.get("n")))
            hit = pd.to_numeric(r.get("layered_hit_rate", r.get("hit_rate")), errors="coerce")
            lift = pd.to_numeric(r.get("lift"), errors="coerce")
            pval = pd.to_numeric(r.get("p_value"), errors="coerce")
            hits = int(round(float(hit) * n)) if pd.notna(hit) else 0
            ci_low, ci_high = _wilson_ci(hits, n)
            rows.append(
                {
                    "layer_name": str(r.get("label", r.get("layer"))),
                    "scenario": scenario,
                    "market": market_name,
                    "status": _status_from_stats(hit, lift, pval, n, min_sample),
                    "hit_rate": hit,
                    "lift": lift,
                    "p_value": pval,
                    "n": n,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "season_consistent": np.nan,
                    "notes": str(r.get("verdict", "")),
                }
            )

    _rows_from_block("blowout_win", "situational_blowout_win", "ats")
    _rows_from_block("underdog_ml_win", "situational_underdog_ml", "ml")
    _rows_from_block("blowout_over", "situational_blowout_over", "total")
    _rows_from_block("blowout_under", "situational_blowout_under", "total")

    march = result.get("march_madness_rd1_rd2", {})
    march_df = march.get("results", pd.DataFrame()) if isinstance(march, dict) else pd.DataFrame()
    if not march_df.empty:
        for _, r in march_df.iterrows():
            for market, n_col, hr_col, lift_col, p_col, season_col, verdict_col in [
                ("ats", "ats_n", "ats_hit_rate", "ats_lift", "ats_p_value", "season_positive_ratio_ats", "ats_verdict"),
                ("ml", "ml_n", "ml_hit_rate", "ml_lift", "ml_p_value", "season_positive_ratio_ml", "ml_verdict"),
                ("total", "total_n", "total_hit_rate", "total_lift", "total_p_value", "season_positive_ratio_total", "total_verdict"),
            ]:
                n = _to_int_or_zero(r.get(n_col))
                hit = pd.to_numeric(r.get(hr_col), errors="coerce")
                lift = pd.to_numeric(r.get(lift_col), errors="coerce")
                pval = pd.to_numeric(r.get(p_col), errors="coerce")
                hits = int(round(float(hit) * n)) if pd.notna(hit) else 0
                ci_low, ci_high = _wilson_ci(hits, n)
                rows.append(
                    {
                        "layer_name": str(r.get("layer")),
                        "scenario": "march_madness_rd1_rd2",
                        "market": market,
                        "status": _status_from_stats(hit, lift, pval, n, min_sample),
                        "hit_rate": hit,
                        "lift": lift,
                        "p_value": pval,
                        "n": n,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "season_consistent": pd.to_numeric(r.get(season_col), errors="coerce"),
                        "notes": str(r.get(verdict_col, "")),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["source"] = "situational_layer_backtesting"
    return out


def _extract_pes_rows(result: dict[str, Any], min_sample: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    validation = result.get("validation", {})
    for block_name in ("full", "tournament"):
        frame = validation.get(block_name, pd.DataFrame()) if isinstance(validation, dict) else pd.DataFrame()
        if frame.empty:
            continue
        for _, r in frame.iterrows():
            n = _to_int_or_zero(r.get("n"))
            hit = pd.to_numeric(r.get("hit_rate"), errors="coerce")
            lift = pd.to_numeric(r.get("lift_over_base"), errors="coerce")
            pval = pd.to_numeric(r.get("p_value"), errors="coerce")
            hits = int(round(float(hit) * n)) if pd.notna(hit) else 0
            ci_low, ci_high = _wilson_ci(hits, n)
            rows.append(
                {
                    "layer_name": f"pes_{block_name}_{r.get('population')}",
                    "scenario": "pes_validation",
                    "market": str(r.get("market", "mixed")).lower(),
                    "status": _status_from_stats(hit, lift, pval, n, min_sample),
                    "hit_rate": hit,
                    "lift": lift,
                    "p_value": pval,
                    "n": n,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "season_consistent": np.nan,
                    "notes": str(r.get("verdict", "")),
                }
            )

    enhancement = result.get("enhancement", {})
    enh_df = enhancement.get("results", pd.DataFrame()) if isinstance(enhancement, dict) else pd.DataFrame()
    if not enh_df.empty:
        for _, r in enh_df.iterrows():
            n = _to_int_or_zero(r.get("n"))
            hit = pd.to_numeric(r.get("hit_rate"), errors="coerce")
            lift = pd.to_numeric(r.get("lift"), errors="coerce")
            pval = pd.to_numeric(r.get("p_value"), errors="coerce")
            hits = int(round(float(hit) * n)) if pd.notna(hit) else 0
            ci_low, ci_high = _wilson_ci(hits, n)
            rows.append(
                {
                    "layer_name": f"pes_enhancement_cutoff_{r.get('cutoff')}",
                    "scenario": "pes_enhancement",
                    "market": str(r.get("market", "mixed")).lower(),
                    "status": _status_from_stats(hit, lift, pval, n, min_sample),
                    "hit_rate": hit,
                    "lift": lift,
                    "p_value": pval,
                    "n": n,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "season_consistent": np.nan,
                    "notes": str(r.get("verdict", "")),
                }
            )

    flip = result.get("flip", {})
    flip_df = flip.get("results", pd.DataFrame()) if isinstance(flip, dict) else pd.DataFrame()
    if not flip_df.empty:
        for _, r in flip_df.iterrows():
            n = _to_int_or_zero(r.get("n"))
            hit = pd.to_numeric(r.get("flip_hit_rate"), errors="coerce")
            base = pd.to_numeric(r.get("base_hit_rate"), errors="coerce")
            lift = hit - base if pd.notna(hit) and pd.notna(base) else np.nan
            pval = pd.to_numeric(r.get("p_value"), errors="coerce")
            hits = int(round(float(hit) * n)) if pd.notna(hit) else 0
            ci_low, ci_high = _wilson_ci(hits, n)
            rows.append(
                {
                    "layer_name": f"pes_flip_cutoff_{r.get('flip_threshold')}",
                    "scenario": "pes_flip",
                    "market": str(r.get("market", "mixed")).lower(),
                    "status": _status_from_stats(hit, lift, pval, n, min_sample),
                    "hit_rate": hit,
                    "lift": lift,
                    "p_value": pval,
                    "n": n,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "season_consistent": np.nan,
                    "notes": str(r.get("verdict", "")),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["source"] = "pes_backtesting"
    return out


def evaluate_upset_candidates(frame: pd.DataFrame, min_sample: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["seed_team"] = to_num(work.get("seed_team", pd.Series(np.nan, index=work.index)))
    work["seed_opponent"] = to_num(work.get("seed_opponent", pd.Series(np.nan, index=work.index)))
    work["ml_won"] = to_num(work.get("ml_won", pd.Series(np.nan, index=work.index)))
    work["is_tournament"] = to_num(work.get("is_tournament", pd.Series(np.nan, index=work.index))).fillna(0) != 0

    base_scope = (
        work["is_tournament"]
        & work["seed_team"].notna()
        & work["seed_opponent"].notna()
        & work["ml_won"].isin([0, 1])
        & (work["seed_team"] > work["seed_opponent"])
    )
    if int(base_scope.sum()) == 0:
        return pd.DataFrame(
            [
                {
                    "layer_name": "upset_candidates",
                    "scenario": "upset_validation",
                    "market": "ml",
                    "status": "BLOCKED",
                    "hit_rate": np.nan,
                    "lift": np.nan,
                    "p_value": np.nan,
                    "n": 0,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "season_consistent": np.nan,
                    "notes": "No tournament underdog seed sample available.",
                    "source": "upset_patterns",
                }
            ]
        )

    base_rate = float(work.loc[base_scope, "ml_won"].mean())
    candidates = {
        "double_digit_seed_dog": base_scope & (work["seed_team"] >= 10),
        "eight_nine_seed_dog": base_scope & work["seed_team"].isin([8, 9]) & work["seed_opponent"].isin([8, 9]),
        "twelve_over_five_profile": base_scope & (work["seed_team"] == 12) & (work["seed_opponent"] == 5),
        "one_possession_seed_gap": base_scope & ((work["seed_team"] - work["seed_opponent"]).between(1, 3)),
    }

    rows: list[dict[str, Any]] = []
    for name, mask in candidates.items():
        subset = work.loc[mask]
        n = int(len(subset))
        if n == 0:
            continue
        hit = float(subset["ml_won"].mean())
        hits = int(subset["ml_won"].sum())
        ci_low, ci_high = _wilson_ci(hits, n)
        pval = float(stats.binomtest(hits, n, base_rate, alternative="greater").pvalue)
        lift = hit - base_rate
        rows.append(
            {
                "layer_name": name,
                "scenario": "upset_validation",
                "market": "ml",
                "status": _status_from_stats(hit, lift, pval, n, min_sample),
                "hit_rate": hit,
                "lift": lift,
                "p_value": pval,
                "n": n,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "season_consistent": np.nan,
                "notes": f"baseline_upset_rate={base_rate:.3f}",
                "source": "upset_patterns",
            }
        )
    return pd.DataFrame(rows)


def _prepare_for_registry(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if df.empty:
        return out
    for scenario, part in df.groupby("scenario", dropna=False):
        frame = part.copy()
        frame["layer"] = frame["layer_name"]
        frame["season_positive_ratio"] = pd.to_numeric(frame["season_consistent"], errors="coerce")
        frame["verdict"] = frame["status"]
        out[str(scenario)] = frame[
            [
                "layer",
                "market",
                "hit_rate",
                "lift",
                "p_value",
                "n",
                "season_positive_ratio",
                "verdict",
            ]
        ].copy()
    return out


def _write_markdown_report(path: Path, validation: pd.DataFrame, blocked_notes: list[str]) -> None:
    ensure_parent(path)
    if validation.empty:
        path.write_text("# Layer Validation Report\n\nNo validation rows were generated.\n", encoding="utf-8")
        return

    def _section(title: str, statuses: set[str]) -> str:
        sub = validation[validation["status"].isin(statuses)].copy()
        if sub.empty:
            return f"## {title}\n\nNone.\n"
        cols = ["layer_name", "scenario", "market", "n", "hit_rate", "lift", "p_value", "notes"]
        try:
            body = sub[cols].to_markdown(index=False)
        except ImportError:
            body = "```text\n" + sub[cols].to_string(index=False) + "\n```"
        return f"## {title}\n\n{body}\n"

    lines = [
        "# Layer Validation Report",
        "",
        f"Generated at UTC: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Total rows: {len(validation)}",
        "",
        _section("Statistically Supported", {"VALIDATED"}),
        _section("Weak / Inconclusive", {"INCONCLUSIVE", "PROVISIONAL"}),
        _section("Failed", {"FAILED"}),
        _section("Blocked / Theoretical Only", {"BLOCKED"}),
        "## Recommended Actions",
    ]

    for status, action in [
        ("VALIDATED", "promote"),
        ("INCONCLUSIVE", "monitor"),
        ("PROVISIONAL", "revisit"),
        ("FAILED", "retire"),
        ("BLOCKED", "revisit after prerequisites"),
    ]:
        n = int((validation["status"] == status).sum())
        lines.append(f"- {status}: {n} layer(s) -> {action}")

    if blocked_notes:
        lines.extend(["", "## Blocking Notes"])
        lines.extend([f"- {note}" for note in blocked_notes])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 backtesting activation and layer validation")
    parser.add_argument("--output-dir", default="data/layer_backtests")
    parser.add_argument("--min-sample", type=int, default=50)
    parser.add_argument("--edge-threshold", type=float, default=4.0)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    args = parser.parse_args()

    paths = resolve_paths()
    output_dir = Path(args.output_dir)
    output_dir = output_dir if output_dir.is_absolute() else paths.rd2_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    warehouse = safe_read_parquet(paths.warehouse_parquet)
    backtest = safe_read_csv(paths.backtest_csv)

    blocked_notes: list[str] = []
    if warehouse.empty:
        blocked_notes.append(f"Warehouse missing or unreadable: {paths.warehouse_parquet}")
    if backtest.empty:
        blocked_notes.append(f"Backtest results missing or unreadable: {paths.backtest_csv}")

    wh_side = build_side_rows_from_warehouse(warehouse)
    bt_side = build_side_rows_from_backtest(backtest)
    layer_input = merge_and_dedupe_side_rows(wh_side, bt_side)
    if layer_input.empty:
        log.error("No rows available for layer validation.")
        return 2

    required_defaults = {
        "covered": np.nan,
        "ml_won": np.nan,
        "total_covered": np.nan,
        "model_edge": np.nan,
        "is_underdog": np.nan,
        "model_projects_over": np.nan,
        "season": np.nan,
        "is_tournament": np.nan,
        "tourney_round": np.nan,
        "seed_team": np.nan,
        "seed_opponent": np.nan,
        "oreb_pct_team": np.nan,
        "oreb_pct_opponent": np.nan,
        "tov_pct_team": np.nan,
        "tov_pct_opponent": np.nan,
        "pace_rank_team": np.nan,
        "pace_rank_opponent": np.nan,
        "forced_tov_rate_team": np.nan,
        "forced_tov_rate_opponent": np.nan,
        "efg_pct_team": np.nan,
        "efg_pct_opponent": np.nan,
        "ppp_team": np.nan,
        "ppp_opponent": np.nan,
        "fta_rate_team": np.nan,
        "fta_rate_opponent": np.nan,
        "opp_efg_pct_team": np.nan,
        "opp_efg_pct_opponent": np.nan,
    }
    for col, default in required_defaults.items():
        if col not in layer_input.columns:
            layer_input[col] = default

    pes_cfg = PESConfig()
    back_cfg = BacktestConfig()
    layer_input = compute_pes_matchup(layer_input, cfg=pes_cfg, tournament=True)
    layer_input_path = output_dir / "layer_backtest_input.csv"
    layer_input.to_csv(layer_input_path, index=False)
    log.info("Prepared layer input rows=%s cols=%s -> %s", len(layer_input), len(layer_input.columns), layer_input_path)

    situational_cfg = ColumnConfig()
    situational_result = run_full_backtest(
        df=layer_input,
        cfg=situational_cfg,
        edge_threshold=float(args.edge_threshold),
        min_sample=int(args.min_sample),
        p_threshold=float(args.p_threshold),
        output_dir=output_dir,
    )
    situational_rows = _extract_situational_rows(situational_result, min_sample=int(args.min_sample))

    pes_result = run_pes_full_backtest(
        df=layer_input,
        pes_cfg=pes_cfg,
        backtest_cfg=back_cfg,
        edge_threshold=float(args.edge_threshold),
        min_sample=int(max(30, args.min_sample)),
        result_cols=["covered", "ml_won", "total_covered"],
        export_csv=False,
    )
    pes_rows = _extract_pes_rows(pes_result, min_sample=int(args.min_sample))

    upset_rows = evaluate_upset_candidates(layer_input, min_sample=int(args.min_sample))

    validation = pd.concat([situational_rows, pes_rows, upset_rows], ignore_index=True, sort=False)
    if validation.empty:
        blocked_notes.append("No rows produced by situational/PES/upset analyzers.")
    validation["last_backtest_date"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    validation["provisional_flag"] = np.where(validation["n"].fillna(0) < int(args.min_sample), True, False)

    validation_out = paths.rd2_data / "layer_validation_results.csv"
    validation.to_csv(validation_out, index=False)

    registry = load_registry(str(paths.layer_registry_csv))
    registry_updates = _prepare_for_registry(validation)
    latest_registry = update_registry_from_backtest(
        registry=registry,
        backtest_results=registry_updates,
        p_threshold=float(args.p_threshold),
        lift_threshold=0.03,
        min_sample=int(args.min_sample),
        season_consistency_threshold=0.60,
    )
    if not latest_registry.empty:
        latest_registry["tier"] = latest_registry.apply(get_tier, axis=1)
    save_registry(latest_registry, str(paths.layer_registry_csv))
    latest_registry_out = paths.rd2_data / "layer_registry_latest.csv"
    latest_registry.to_csv(latest_registry_out, index=False)

    report_path = paths.rd2_root / "docs" / "reports" / "layer_validation_report.md"
    _write_markdown_report(report_path, validation, blocked_notes)

    log.info("Validation rows=%s -> %s", len(validation), validation_out)
    log.info("Registry latest rows=%s -> %s", len(latest_registry), latest_registry_out)
    log.info("Report -> %s", report_path)
    if blocked_notes:
        for note in blocked_notes:
            log.warning("BLOCKED/WARN: %s", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
