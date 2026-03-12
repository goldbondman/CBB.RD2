"""Layer registry for active/watchlist/pruned situational layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LayerRecord:
    layer_name: str
    scenario: str
    market: str
    status: str
    hit_rate: float
    lift: float
    p_value: float
    n: int
    season_consistent: bool
    last_backtest_date: str
    prune_reason: str | None = None
    notes: str | None = None


REGISTRY_COLUMNS = list(asdict(LayerRecord("", "", "", "watchlist", np.nan, np.nan, np.nan, 0, False, "")).keys()) + [
    "registry_version_utc"
]


def _now_utc_iso() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def load_registry(path: str = "data/layer_registry.csv") -> pd.DataFrame:
    """Load existing layer registry or initialize empty."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(p, index=False)
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    df = pd.read_csv(p, low_memory=False)
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[REGISTRY_COLUMNS]


def save_registry(df: pd.DataFrame, path: str = "data/layer_registry.csv") -> None:
    """Append a versioned registry snapshot (never destructive overwrite)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    clean["registry_version_utc"] = _now_utc_iso()
    clean = clean.reindex(columns=REGISTRY_COLUMNS)
    if p.exists() and p.stat().st_size > 0:
        prior = pd.read_csv(p, low_memory=False)
        out = pd.concat([prior, clean], ignore_index=True, sort=False)
    else:
        out = clean
    out.to_csv(p, index=False)


def get_active_layers(registry: pd.DataFrame) -> pd.DataFrame:
    """Return active rows only."""
    if registry.empty:
        return registry.copy()
    return registry[registry["status"].astype(str).str.lower().eq("active")].copy()


def _pick_status(
    *,
    p_value: float,
    lift: float,
    n: int,
    season_consistent: bool,
    p_threshold: float,
    lift_threshold: float,
    min_sample: int,
) -> tuple[str, str | None]:
    if n < min_sample:
        return "pruned", "insufficient_sample"
    if pd.isna(lift) or lift <= 0:
        return "pruned", "negative_or_zero_lift"
    if pd.notna(p_value) and p_value <= p_threshold and lift >= lift_threshold and season_consistent:
        return "active", None
    if lift > 0 and (pd.isna(p_value) or p_value > p_threshold):
        return "watchlist", "promising_not_significant"
    return "pruned", "failed_thresholds"


def update_registry_from_backtest(
    registry: pd.DataFrame,
    backtest_results: dict[str, pd.DataFrame],
    p_threshold: float = 0.05,
    lift_threshold: float = 0.03,
    min_sample: int = 150,
    season_consistency_threshold: float = 0.60,
) -> pd.DataFrame:
    """Apply promotion/demotion rules from fresh backtest outputs."""
    records: list[dict[str, Any]] = []
    date = _now_utc_iso()
    for scenario, frame in backtest_results.items():
        if frame is None or frame.empty:
            continue
        f = frame.copy()
        layer_col = "layer" if "layer" in f.columns else ("label" if "label" in f.columns else None)
        if layer_col is None:
            continue
        if "market" not in f.columns:
            f["market"] = scenario
        if "hit_rate" not in f.columns:
            f["hit_rate"] = pd.to_numeric(f.get("layered_hit_rate", f.get("ats_hit_rate", np.nan)), errors="coerce")
        if "lift" not in f.columns:
            f["lift"] = pd.to_numeric(f.get("ats_lift", f.get("layered_lift", np.nan)), errors="coerce")
        if "p_value" not in f.columns:
            f["p_value"] = pd.to_numeric(f.get("ats_p_value", f.get("layered_p_value", np.nan)), errors="coerce")
        if "n" not in f.columns:
            f["n"] = pd.to_numeric(f.get("ats_n", f.get("layered_n", np.nan)), errors="coerce")
        if "season_positive_ratio" not in f.columns:
            f["season_positive_ratio"] = pd.to_numeric(
                f.get("season_positive_ratio", f.get("season_positive_ratio_ats", np.nan)),
                errors="coerce",
            )
        for _, row in f.iterrows():
            season_ok = bool(pd.notna(row.get("season_positive_ratio")) and float(row.get("season_positive_ratio")) >= season_consistency_threshold)
            status, reason = _pick_status(
                p_value=float(row.get("p_value")) if pd.notna(row.get("p_value")) else np.nan,
                lift=float(row.get("lift")) if pd.notna(row.get("lift")) else np.nan,
                n=int(float(row.get("n"))) if pd.notna(row.get("n")) else 0,
                season_consistent=season_ok,
                p_threshold=p_threshold,
                lift_threshold=lift_threshold,
                min_sample=min_sample,
            )
            records.append(
                {
                    "layer_name": str(row.get(layer_col)),
                    "scenario": scenario,
                    "market": str(row.get("market")),
                    "status": status,
                    "hit_rate": float(row.get("hit_rate")) if pd.notna(row.get("hit_rate")) else np.nan,
                    "lift": float(row.get("lift")) if pd.notna(row.get("lift")) else np.nan,
                    "p_value": float(row.get("p_value")) if pd.notna(row.get("p_value")) else np.nan,
                    "n": int(float(row.get("n"))) if pd.notna(row.get("n")) else 0,
                    "season_consistent": season_ok,
                    "last_backtest_date": date,
                    "prune_reason": reason if status == "pruned" else None,
                    "notes": str(row.get("verdict")) if pd.notna(row.get("verdict")) else None,
                }
            )

    updates = pd.DataFrame(records)
    if updates.empty:
        return registry.copy()

    if registry.empty:
        merged = updates.copy()
    else:
        key_cols = ["layer_name", "scenario", "market"]
        prev = registry.sort_values("last_backtest_date").drop_duplicates(key_cols, keep="last")
        merged = prev.merge(updates, on=key_cols, how="outer", suffixes=("_old", ""))
        for col in ["status", "hit_rate", "lift", "p_value", "n", "season_consistent", "last_backtest_date", "prune_reason", "notes"]:
            merged[col] = merged[col].where(merged[col].notna(), merged.get(f"{col}_old"))
        drop_cols = [c for c in merged.columns if c.endswith("_old")]
        merged = merged.drop(columns=drop_cols, errors="ignore")

    promoted = int((merged["status"] == "active").sum())
    pruned = int((merged["status"] == "pruned").sum())
    watch = int((merged["status"] == "watchlist").sum())
    print(f"Registry update summary: active={promoted} watchlist={watch} pruned={pruned}")
    return merged.reindex(columns=[c for c in REGISTRY_COLUMNS if c != "registry_version_utc"])


def get_tier(layer_row: pd.Series) -> str:
    """Return tier label for one layer row."""
    status = str(layer_row.get("status", "")).lower()
    hr = pd.to_numeric(layer_row.get("hit_rate"), errors="coerce")
    consistent = bool(layer_row.get("season_consistent", False))
    if status == "active" and pd.notna(hr) and hr >= 0.55 and consistent:
        return "Tier 1"
    if status in {"active", "watchlist"}:
        return "Tier 2"
    return "Tier 3"


def main() -> int:
    """Minimal demo."""
    reg = load_registry()
    if reg.empty:
        seed = pd.DataFrame(
            [
                LayerRecord(
                    layer_name="home_rest_form_stack",
                    scenario="legacy_situational",
                    market="ATS",
                    status="watchlist",
                    hit_rate=np.nan,
                    lift=np.nan,
                    p_value=np.nan,
                    n=0,
                    season_consistent=False,
                    last_backtest_date=_now_utc_iso(),
                    notes="initialized",
                ).__dict__
            ]
        )
        save_registry(seed)
        print("Initialized registry with seed row.")
    else:
        print(f"Loaded registry rows={len(reg)} active={len(get_active_layers(reg))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

