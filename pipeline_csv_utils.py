from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)

NULLABLE_OK_DEFAULT = {
    "spread", "over_under", "home_ml", "away_ml",
    "home_rank", "away_rank", "venue", "neutral_site",
    "odds_provider", "odds_details", "home_h1", "home_h2",
    "away_h1", "away_h2", "vegas_line", "total_line",
    "home_ot1", "away_ot1", "home_ot2", "away_ot2",
}


def fix_case_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize duplicate columns that differ only by case (e.g., FGM vs fgm)."""
    out = df.copy()
    grouped: dict[str, list[str]] = {}
    for col in out.columns:
        grouped.setdefault(col.lower(), []).append(col)

    for _, cols in grouped.items():
        if len(cols) < 2:
            continue
        preferred = next((c for c in cols if c == c.lower()), cols[0])
        pref = out[preferred]
        for col in cols:
            if col == preferred:
                continue
            src = out[col]
            pref = pref.where(pref.notna(), src)
        out[preferred] = pref
        drop_cols = [c for c in cols if c != preferred]
        out = out.drop(columns=drop_cols)
    return out


def safe_write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    label: Optional[str] = None,
    allow_empty: bool = False,
    nullable_ok: Optional[Iterable[str]] = None,
) -> bool:
    """Write CSV with guardrails and logging. Returns True when written."""
    target = Path(path)
    name = label or str(target)
    nullable = {c.lower() for c in (nullable_ok or NULLABLE_OK_DEFAULT)}

    checked = fix_case_duplicates(df)

    if checked.empty and not allow_empty:
        log.error(f"[BLOCKED] {name}: refusing to write 0-row DataFrame")
        return False

    null_cols = [c for c in checked.columns if checked[c].isna().all() and c.lower() not in nullable]
    if null_cols:
        log.warning(f"[WARN] {name}: entirely-null columns ({len(null_cols)}): {null_cols[:10]}")

    zero_cols = []
    for c in checked.select_dtypes(include="number").columns:
        s = checked[c].dropna()
        if len(s) > 0 and (s == 0).all():
            zero_cols.append(c)
    if zero_cols:
        log.warning(f"[WARN] {name}: all-zero numeric columns ({len(zero_cols)}): {zero_cols[:10]}")

    target.parent.mkdir(parents=True, exist_ok=True)
    checked.to_csv(target, index=False)
    log.info(f"[OK] {name}: {len(checked)} rows, {len(checked.columns)} cols written")
    return True
