import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from espn_config import get_conference_tier, get_game_tier

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)

MIN_SAMPLE = 15
ROLLING_DAYS = 60
SIG_LEVEL = 0.10
MAX_CORRECTION = 4.0


def _ats_metrics(grp: pd.DataFrame) -> dict:
    metrics = {
        "home_cover_rate": round(grp["home_covered_pred"].mean() * 100, 1),
    }
    if "model_picked_home" in grp.columns:
        picked_home = grp["model_picked_home"].astype(str).str.lower().isin(["true", "1", "yes"])
        model_correct = np.where(picked_home, grp["home_covered_pred"], ~grp["home_covered_pred"].astype(bool))
        metrics["ats_pct"] = round(pd.Series(model_correct, index=grp.index).mean() * 100, 1)
    else:
        # model_picked_home/model_picked_side missing, so true model ATS% cannot be derived.
        pass
    return metrics


def load_graded(path: Path, rolling_days: int) -> pd.DataFrame:
    """Load graded predictions and attach conference tier labels."""
    if not path.exists():
        fallback = Path("data/results_log_graded.csv")
        if fallback.exists():
            log.warning("%s missing, falling back to %s", path, fallback)
            path = fallback
        else:
            raise FileNotFoundError(f"No graded file found at {path} or {fallback}")

    df = pd.read_csv(path, dtype={"event_id": str}, parse_dates=["game_datetime_utc"])

    if "graded" in df.columns:
        graded = df[df["graded"] == True].copy()
    else:
        graded = df.copy()

    if len(graded) == 0:
        raise ValueError("No graded predictions found")

    if rolling_days and "game_datetime_utc" in graded.columns:
        cutoff = (pd.Timestamp.now('UTC') - pd.Timedelta(days=rolling_days)).tz_localize(None)
        graded["game_datetime_utc"] = pd.to_datetime(graded["game_datetime_utc"], utc=True, errors="coerce").dt.tz_convert(None)
        graded = graded[graded["game_datetime_utc"] >= cutoff]
        log.info("Rolling %sd window: %s graded rows", rolling_days, len(graded))

    if "home_conference" in graded.columns and "away_conference" in graded.columns:
        graded["game_tier"] = graded.apply(
            lambda r: get_game_tier(r.get("home_conference", ""), r.get("away_conference", "")),
            axis=1,
        )
        graded["home_tier"] = graded["home_conference"].apply(get_conference_tier)
        graded["away_tier"] = graded["away_conference"].apply(get_conference_tier)
    elif "conference" in graded.columns:
        graded["game_tier"] = graded["conference"].apply(get_conference_tier)
    else:
        graded["game_tier"] = "UNKNOWN"
        log.warning("No conference columns found — tier analysis will be limited")

    for col in [
        "spread_error",
        "abs_spread_error",
        "home_covered_pred",
        "predicted_spread",
        "model_confidence",
        "actual_margin",
        "closing_line",
        "rest_days",
        "home_rest_days",
    ]:
        if col in graded.columns:
            graded[col] = pd.to_numeric(graded[col], errors="coerce")

    if "abs_spread_error" not in graded.columns and "spread_error" in graded.columns:
        graded["abs_spread_error"] = graded["spread_error"].abs()

    if "home_covered_pred" not in graded.columns:
        graded["home_covered_pred"] = np.nan

    log.info(
        "Loaded %s graded rows. Tier distribution: %s",
        len(graded),
        graded["game_tier"].value_counts().to_dict(),
    )
    return graded


def _mean_bias_stats(errors: pd.Series) -> tuple[float, float, float, float]:
    """Return mean / p-value / confidence interval for spread error series."""
    clean = errors.dropna()
    if len(clean) < 3:
        return 0.0, 1.0, 0.0, 0.0

    mean = float(clean.mean())
    sem = float(stats.sem(clean))
    _, p_val = stats.ttest_1samp(clean, 0.0)
    ci_lo, ci_hi = stats.t.interval(0.90, df=len(clean) - 1, loc=mean, scale=sem)
    return mean, float(p_val), float(ci_lo), float(ci_hi)


def analyze_conference_bias(df: pd.DataFrame, min_sample: int, sig_level: float) -> list[dict]:
    results = []

    if "conference" in df.columns or "home_conference" in df.columns:
        conf_col = "conference" if "conference" in df.columns else "home_conference"
        for conf, grp in df.groupby(conf_col):
            if len(grp) < min_sample:
                continue
            bias, p_val, ci_lo, ci_hi = _mean_bias_stats(grp["spread_error"])
            results.append(
                {
                    "dimension": "conference",
                    "group": str(conf),
                    "sample_n": len(grp),
                    "mean_error": round(bias, 3),
                    "ci_lo": round(ci_lo, 3),
                    "ci_hi": round(ci_hi, 3),
                    "p_value": round(p_val, 4),
                    "actionable": p_val < sig_level and abs(bias) > 0.5,
                    "correction": round(np.clip(-bias, -MAX_CORRECTION, MAX_CORRECTION), 3),
                    **_ats_metrics(grp),
                    "mae": round(grp["abs_spread_error"].mean(), 2),
                }
            )

    for tier, grp in df.groupby("game_tier"):
        if len(grp) < min_sample:
            continue
        bias, p_val, ci_lo, ci_hi = _mean_bias_stats(grp["spread_error"])
        results.append(
            {
                "dimension": "conference_tier",
                "group": str(tier),
                "sample_n": len(grp),
                "mean_error": round(bias, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "p_value": round(p_val, 4),
                "actionable": p_val < sig_level and abs(bias) > 0.5,
                "correction": round(np.clip(-bias, -MAX_CORRECTION, MAX_CORRECTION), 3),
                **_ats_metrics(grp),
                "mae": round(grp["abs_spread_error"].mean(), 2),
            }
        )

    return results


def analyze_variance_tier_bias(df: pd.DataFrame, min_sample: int, sig_level: float) -> list[dict]:
    results = []
    var_col = None
    for c in ["net_rtg_std_l10_home", "net_rtg_std_l10_away", "home_net_rtg_std_l10", "away_net_rtg_std_l10", "net_rtg_std_l10"]:
        if c in df.columns:
            var_col = c
            break

    if var_col is None:
        log.info("No variance column found — skipping variance tier analysis")
        return results

    d = df.copy()
    d["avg_variance"] = d[var_col]
    valid = d["avg_variance"].dropna()
    if valid.nunique() < 4:
        log.info("Insufficient variance spread for quartiles")
        return results

    d.loc[valid.index, "variance_quartile"] = pd.qcut(valid, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")

    for qtile, grp in d.groupby("variance_quartile"):
        if len(grp) < min_sample:
            continue
        bias, p_val, ci_lo, ci_hi = _mean_bias_stats(grp["spread_error"])
        results.append(
            {
                "dimension": "variance_quartile",
                "group": str(qtile),
                "sample_n": len(grp),
                "mean_error": round(bias, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "p_value": round(p_val, 4),
                "actionable": p_val < sig_level and abs(bias) > 0.5,
                "correction": round(np.clip(-bias, -MAX_CORRECTION, MAX_CORRECTION), 3),
                **_ats_metrics(grp),
                "mae": round(grp["abs_spread_error"].mean(), 2),
            }
        )
    return results


def analyze_rest_bias(df: pd.DataFrame, min_sample: int, sig_level: float) -> list[dict]:
    results = []
    if "rest_days" not in df.columns and "home_rest_days" not in df.columns:
        return results

    rest_col = "home_rest_days" if "home_rest_days" in df.columns else "rest_days"
    d = df.copy()
    d["rest_bucket"] = pd.cut(
        pd.to_numeric(d[rest_col], errors="coerce"),
        bins=[-1, 1, 3, 6, 30],
        labels=["back_to_back", "normal_2_3", "extra_4_6", "long_7plus"],
    )

    for bucket, grp in d.groupby("rest_bucket"):
        if len(grp) < min_sample:
            continue
        bias, p_val, ci_lo, ci_hi = _mean_bias_stats(grp["spread_error"])
        results.append(
            {
                "dimension": "rest_days",
                "group": str(bucket),
                "sample_n": len(grp),
                "mean_error": round(bias, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "p_value": round(p_val, 4),
                "actionable": p_val < sig_level and abs(bias) > 0.5,
                "correction": round(np.clip(-bias, -MAX_CORRECTION, MAX_CORRECTION), 3),
                **_ats_metrics(grp),
                "mae": round(grp["abs_spread_error"].mean(), 2),
            }
        )
    return results


def analyze_luck_bias(df: pd.DataFrame, min_sample: int, sig_level: float) -> list[dict]:
    results = []
    luck_col = None
    for c in ["home_luck_score", "luck_score_a", "luck_score"]:
        if c in df.columns:
            luck_col = c
            break

    if luck_col is None:
        return results

    d = df.copy()
    luck = pd.to_numeric(d[luck_col], errors="coerce").dropna()
    if luck.nunique() < 4:
        return results

    d.loc[luck.index, "luck_quartile"] = pd.qcut(
        luck,
        q=4,
        labels=["Q1_unlucky", "Q2", "Q3", "Q4_lucky"],
        duplicates="drop",
    )

    quartile_biases = {}
    for qtile, grp in d.groupby("luck_quartile"):
        if len(grp) < min_sample:
            continue
        bias, p_val, ci_lo, ci_hi = _mean_bias_stats(grp["spread_error"])
        quartile_biases[str(qtile)] = bias
        results.append(
            {
                "dimension": "luck_quartile",
                "group": str(qtile),
                "sample_n": len(grp),
                "mean_error": round(bias, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "p_value": round(p_val, 4),
                "actionable": p_val < sig_level and abs(bias) > 0.3,
                "correction": round(np.clip(-bias, -MAX_CORRECTION, MAX_CORRECTION), 3),
                **_ats_metrics(grp),
                "mae": round(grp["abs_spread_error"].mean(), 2),
            }
        )

    if "Q4_lucky" in quartile_biases and quartile_biases["Q4_lucky"] > 1.0:
        recommended_coeff = round(0.4 * (1 + quartile_biases["Q4_lucky"] / 3.0), 2)
        log.info(
            "Luck regression under-applied: Q4 bias = +%.2f. Recommended coefficient: %s (current: 0.4)",
            quartile_biases["Q4_lucky"],
            recommended_coeff,
        )

    return results


def analyze_line_size_bias(df: pd.DataFrame, min_sample: int, sig_level: float) -> list[dict]:
    results = []
    if "closing_line" not in df.columns and "predicted_spread" not in df.columns:
        return results

    line_col = "closing_line" if "closing_line" in df.columns else "predicted_spread"
    d = df.copy()
    d["abs_line"] = pd.to_numeric(d[line_col], errors="coerce").abs()
    d["line_bucket"] = pd.cut(
        d["abs_line"],
        bins=[0, 3, 7, 14, 100],
        labels=["pickem_0_3", "small_3_7", "medium_7_14", "large_14plus"],
    )

    for bucket, grp in d.groupby("line_bucket"):
        if len(grp) < min_sample:
            continue
        bias, p_val, ci_lo, ci_hi = _mean_bias_stats(grp["spread_error"])
        hcr = grp["home_covered_pred"].mean() * 100
        results.append(
            {
                "dimension": "line_size",
                "group": str(bucket),
                "sample_n": len(grp),
                "mean_error": round(bias, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "p_value": round(p_val, 4),
                "actionable": p_val < sig_level and abs(bias) > 0.5,
                "correction": round(np.clip(-bias, -MAX_CORRECTION, MAX_CORRECTION), 3),
                **_ats_metrics(grp),
                "mae": round(grp["abs_spread_error"].mean(), 2),
                "note": "Large favorites home cover rate caution" if (bucket == "large_14plus" and hcr < 48) else "",
            }
        )
    return results


def analyze_momentum_bias(df: pd.DataFrame, min_sample: int, sig_level: float) -> list[dict]:
    results = []
    momentum_col = None
    for c in ["home_momentum_score", "momentum_score_a", "momentum"]:
        if c in df.columns:
            momentum_col = c
            break

    if momentum_col is None:
        return results

    d = df.copy()
    d["momentum_tier"] = pd.cut(
        pd.to_numeric(d[momentum_col], errors="coerce"),
        bins=[-999, -10, -4, 4, 10, 999],
        labels=["COLD", "COOLING", "STEADY", "RISING", "HOT"],
    )

    for tier, grp in d.groupby("momentum_tier"):
        if len(grp) < min_sample:
            continue
        bias, p_val, ci_lo, ci_hi = _mean_bias_stats(grp["spread_error"])
        results.append(
            {
                "dimension": "momentum_tier",
                "group": str(tier),
                "sample_n": len(grp),
                "mean_error": round(bias, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "p_value": round(p_val, 4),
                "actionable": p_val < sig_level and abs(bias) > 0.5,
                "correction": round(np.clip(-bias, -MAX_CORRECTION, MAX_CORRECTION), 3),
                **_ats_metrics(grp),
                "mae": round(grp["abs_spread_error"].mean(), 2),
            }
        )
    return results


def analyze_cross_tier_bias(df: pd.DataFrame, min_sample: int, sig_level: float) -> list[dict]:
    results = []
    for tier in ["CROSS_HIGH_MID", "CROSS_HIGH_LOW", "CROSS_MID_LOW"]:
        grp = df[df["game_tier"] == tier]
        if len(grp) < min_sample:
            continue

        bias, p_val, ci_lo, ci_hi = _mean_bias_stats(grp["spread_error"])
        hcr = grp["home_covered_pred"].mean() * 100
        direction = "home_underdog_outperforms" if bias > 0 else "home_favorite_covers_more"
        results.append(
            {
                "dimension": "cross_tier_matchup",
                "group": tier,
                "sample_n": len(grp),
                "mean_error": round(bias, 3),
                "ci_lo": round(ci_lo, 3),
                "ci_hi": round(ci_hi, 3),
                "p_value": round(p_val, 4),
                "actionable": p_val < sig_level and abs(bias) > 0.5,
                "correction": round(np.clip(-bias, -MAX_CORRECTION, MAX_CORRECTION), 3),
                **_ats_metrics(grp),
                "mae": round(grp["abs_spread_error"].mean(), 2),
                "error_direction": direction,
                "note": f"⚠️ Systematic {abs(bias):.1f}pt error on {tier} games — {direction}" if p_val < sig_level else "",
            }
        )
    return results


def write_bias_table(all_biases: list[dict], output_path: Path) -> None:
    if not all_biases:
        log.warning("No bias results to write")
        return

    df = pd.DataFrame(all_biases)
    df["computed_at_utc"] = datetime.now(timezone.utc).isoformat()
    df.sort_values(["actionable", "p_value"], ascending=[False, True], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    actionable = df[df["actionable"] == True]
    log.info("Bias table: %s entries, %s actionable corrections", len(df), len(actionable))


def write_bias_report(all_biases: list[dict], df_graded: pd.DataFrame, report_path: Path) -> None:
    actionable = [b for b in all_biases if b.get("actionable")]

    tier_ats = {}
    if "game_tier" in df_graded.columns:
        for tier, grp in df_graded.groupby("game_tier"):
            if len(grp) >= 5:
                tier_ats[str(tier)] = {
                    "n": len(grp),
                    **_ats_metrics(grp),
                    "mae": round(grp["abs_spread_error"].mean(), 2),
                }
                if "model_picked_home" in grp.columns:
                    picked_home = grp["model_picked_home"].astype(str).str.lower().isin(["true", "1", "yes"])
                    model_correct = np.where(picked_home, grp["home_covered_pred"], ~grp["home_covered_pred"].astype(bool))
                    tier_ats[str(tier)]["ats_pct"] = round(pd.Series(model_correct, index=grp.index).mean() * 100, 1)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "graded_total": len(df_graded),
        "overall_home_cover_rate": round(df_graded["home_covered_pred"].mean() * 100, 1),
        "overall_mae": round(df_graded["abs_spread_error"].mean(), 2),
        "actionable_biases": len(actionable),
        "bias_by_tier": tier_ats,
        "top_corrections": sorted(actionable, key=lambda x: abs(x.get("correction", 0)), reverse=True)[:10],
        "dimensions_checked": list({b["dimension"] for b in all_biases}),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Bias report → %s", report_path)


def append_bias_history(all_biases: list[dict], history_path: Path) -> None:
    if not all_biases:
        return

    df_new = pd.DataFrame(all_biases)
    df_new["week_of"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if history_path.exists():
        df_existing = pd.read_csv(history_path)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(history_path, index=False)
    log.info("Bias history → %s (%s total rows)", history_path, len(df_out))


def main() -> None:
    try:
        from espn_config import (
            OUT_PREDICTIONS_GRADED as graded_path,
            OUT_BIAS_TABLE as bias_table_path,
            OUT_BIAS_REPORT as bias_report_path,
            OUT_BIAS_HISTORY as bias_history_path,
        )
    except ImportError:
        graded_path = Path("data/predictions_graded.csv")
        bias_table_path = Path("data/model_bias_table.csv")
        bias_report_path = Path("data/bias_report.json")
        bias_history_path = Path("data/bias_history.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-sample", type=int, default=MIN_SAMPLE)
    parser.add_argument("--window", type=int, default=ROLLING_DAYS)
    parser.add_argument("--sig-level", type=float, default=SIG_LEVEL)
    args = parser.parse_args()

    try:
        df = load_graded(Path(graded_path), args.window)
    except Exception as exc:
        log.warning("Bias detector skipped: %s", exc)
        Path(bias_table_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "dimension",
                "group",
                "sample_n",
                "mean_error",
                "ci_lo",
                "ci_hi",
                "p_value",
                "actionable",
                "correction",
                "home_cover_rate",
                "mae",
                "computed_at_utc",
            ]
        ).to_csv(bias_table_path, index=False)
        with Path(bias_report_path).open("w", encoding="utf-8") as f:
            json.dump({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "skipped", "reason": str(exc)}, f, indent=2)
        pd.DataFrame(columns=["dimension", "group", "sample_n", "mean_error", "week_of"]).to_csv(bias_history_path, index=False)
        return

    all_biases = []
    all_biases += analyze_conference_bias(df, args.min_sample, args.sig_level)
    all_biases += analyze_variance_tier_bias(df, args.min_sample, args.sig_level)
    all_biases += analyze_rest_bias(df, args.min_sample, args.sig_level)
    all_biases += analyze_luck_bias(df, args.min_sample, args.sig_level)
    all_biases += analyze_line_size_bias(df, args.min_sample, args.sig_level)
    all_biases += analyze_momentum_bias(df, args.min_sample, args.sig_level)
    all_biases += analyze_cross_tier_bias(df, args.min_sample, args.sig_level)

    write_bias_table(all_biases, Path(bias_table_path))
    write_bias_report(all_biases, df, Path(bias_report_path))
    append_bias_history(all_biases, Path(bias_history_path))


if __name__ == "__main__":
    main()
