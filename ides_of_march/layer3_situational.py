from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import (
    ENABLE_UPSET_LAYER,
    MIN_RULE_SAMPLE,
    REPO_ROOT,
    RULE_SHRINK_K,
    UPSET_LAYER_CANDIDATES_PATH,
    UPSET_LAYER_EFFECT_CAP,
    UPSET_LAYER_MIN_SAMPLE,
)


@dataclass(frozen=True)
class RuleDef:
    rule_id: str
    description: str
    direction: int  # +1 home side, -1 away side


_COND_PATTERN = re.compile(
    r"""df\[['"](?P<col>[^'"]+)['"]\]\s*(?P<op>>=|<=|==|>|<)\s*(?P<val>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"""
)


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    if UPSET_LAYER_CANDIDATES_PATH:
        paths.append(Path(UPSET_LAYER_CANDIDATES_PATH))
    paths.extend(
        [
            REPO_ROOT / "data" / "upset_layer_candidates.csv",
            REPO_ROOT.parent / "data" / "upset_layer_candidates.csv",
        ]
    )
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    return dedup


def _load_upset_candidates() -> pd.DataFrame:
    if not ENABLE_UPSET_LAYER:
        return pd.DataFrame(columns=["rule_id", "description", "python_condition", "effect", "n"])

    path = next((p for p in _candidate_paths() if p.exists()), None)
    if path is None:
        return pd.DataFrame(columns=["rule_id", "description", "python_condition", "effect", "n"])

    try:
        raw = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=["rule_id", "description", "python_condition", "effect", "n"])

    if raw.empty or "layer_name" not in raw.columns:
        return pd.DataFrame(columns=["rule_id", "description", "python_condition", "effect", "n"])

    work = raw.copy()
    if "is_redundant_with_existing" in work.columns:
        non_redundant = ~work["is_redundant_with_existing"].fillna(False).astype(bool)
        work = work.loc[non_redundant].copy()

    n = pd.to_numeric(work.get("n"), errors="coerce")
    lift = pd.to_numeric(work.get("lift"), errors="coerce")
    work = work.loc[(n >= float(UPSET_LAYER_MIN_SAMPLE)) & (lift > 0)].copy()
    if work.empty:
        return pd.DataFrame(columns=["rule_id", "description", "python_condition", "effect", "n"])

    work["rule_id"] = "upset_" + work["layer_name"].astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9_]+", "_", regex=True)
    work["description"] = work.get("description", work["layer_name"]).astype(str)
    work["python_condition"] = work.get("python_condition", "").astype(str)
    work["effect"] = (pd.to_numeric(work.get("lift"), errors="coerce") * 0.08).clip(lower=0.0, upper=float(UPSET_LAYER_EFFECT_CAP))
    work["n"] = pd.to_numeric(work.get("n"), errors="coerce")
    return work[["rule_id", "description", "python_condition", "effect", "n"]].dropna(subset=["rule_id", "effect"])


def _evaluate_external_condition(df: pd.DataFrame, condition: str) -> pd.Series:
    matches = list(_COND_PATTERN.finditer(str(condition)))
    if not matches:
        return pd.Series(False, index=df.index, dtype=bool)

    out = pd.Series(True, index=df.index, dtype=bool)
    for m in matches:
        col = m.group("col")
        op = m.group("op")
        val = float(m.group("val"))
        if col not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        series = pd.to_numeric(df[col], errors="coerce")
        if op == ">=":
            mask = series >= val
        elif op == "<=":
            mask = series <= val
        elif op == ">":
            mask = series > val
        elif op == "<":
            mask = series < val
        else:
            mask = series == val
        out = out & mask.fillna(False)
    return out


def _prepare_upset_candidate_masks(df: pd.DataFrame) -> list[dict[str, object]]:
    candidates = _load_upset_candidates()
    if candidates.empty:
        return []
    out: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        cond = str(row.get("python_condition", "")).strip()
        if not cond:
            continue
        mask = _evaluate_external_condition(df, cond)
        if not bool(mask.any()):
            continue
        out.append(
            {
                "rule_id": str(row["rule_id"]),
                "effect": float(pd.to_numeric(row.get("effect"), errors="coerce")),
                "mask": mask.fillna(False).astype(bool),
            }
        )
    return out


def _underdog_direction(series: pd.Series) -> pd.Series:
    spread = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.where(spread > 0, 1.0, np.where(spread < 0, -1.0, 0.0)),
        index=spread.index,
        dtype=float,
    )


def _rule_definitions() -> list[RuleDef]:
    return [
        RuleDef("home_rest_form_stack", "Home rest + form stack", +1),
        RuleDef("away_rest_form_stack", "Away rest + form stack", -1),
        RuleDef("home_efg_edge_4pct", "Home eFG edge >= 4% with model support", +1),
        RuleDef("away_efg_edge_4pct", "Away eFG edge >= 4% with model support", -1),
        RuleDef("home_oreb_edge_3pct", "Home OREB edge >= 3% with model support", +1),
        RuleDef("away_oreb_edge_3pct", "Away OREB edge >= 3% with model support", -1),
        RuleDef("home_pace_mismatch_fast_edge", "Home pace mismatch (fast imposer)", +1),
        RuleDef("away_pace_mismatch_fast_edge", "Away pace mismatch (fast imposer)", -1),
        RuleDef("fade_away_threept_vs_slow", "Fade away 3PT-heavy team into slow game", +1),
        RuleDef("fade_home_threept_vs_slow", "Fade home 3PT-heavy team into slow game", -1),
        RuleDef("fade_home_weak_sos_form_pop", "Fade home weak-SOS form pop", -1),
        RuleDef("fade_away_weak_sos_form_pop", "Fade away weak-SOS form pop", +1),
        RuleDef("conference_bye_edge_home", "Conference tournament bye/rest edge home", +1),
        RuleDef("conference_bye_edge_away", "Conference tournament bye/rest edge away", -1),
        RuleDef("conference_fatigue_fade_home", "Conference tournament fatigue fade home", -1),
        RuleDef("conference_fatigue_fade_away", "Conference tournament fatigue fade away", +1),
        RuleDef("fade_blueblood_home", "Fade away blue-blood when model backs home", +1),
        RuleDef("fade_blueblood_away", "Fade home blue-blood when model backs away", -1),
    ]


def _rule_mask(df: pd.DataFrame, rule_id: str) -> pd.Series:
    def _num(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    rest_diff = _num("rest_diff")
    form_diff = _num("form_delta_diff")
    market_spread = _num("market_spread")
    margin_ctx = _num("margin_ctx_blend")
    model_edge_pre = market_spread + margin_ctx
    efg_margin = _num("efg_margin_l5")
    oreb_margin = _num("oreb_margin_l5")
    away_rest = _num("away_days_rest")
    home_rest = _num("home_days_rest")
    home_oreb = _num("home_orb_pct_l5")
    away_oreb = _num("away_orb_pct_l5")
    home_dreb = _num("home_drb_pct_l5")
    away_dreb = _num("away_drb_pct_l5")
    away_three = _num("away_three_par_l5")
    home_three = _num("home_three_par_l5")
    home_pace = _num("home_pace_l5")
    away_pace = _num("away_pace_l5")
    home_form = _num("home_Form_Delta")
    away_form = _num("away_Form_Delta")
    home_sos = _num("home_sos_pre")
    away_sos = _num("away_sos_pre")

    is_conf_tourney = df.get("is_conference_tournament", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    home_team = df.get("home_team", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    away_team = df.get("away_team", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    blueblood_tags = ["duke", "kentucky", "kansas", "north carolina", "michigan st", "michigan state"]
    away_is_blueblood = pd.Series(False, index=df.index)
    home_is_blueblood = pd.Series(False, index=df.index)
    for tag in blueblood_tags:
        away_is_blueblood = away_is_blueblood | away_team.str.contains(tag, regex=False)
        home_is_blueblood = home_is_blueblood | home_team.str.contains(tag, regex=False)

    if rule_id == "home_rest_form_stack":
        return (rest_diff >= 2.0) & (form_diff >= 0.6)
    if rule_id == "away_rest_form_stack":
        return (rest_diff <= -2.0) & (form_diff <= -0.6)
    if rule_id == "home_efg_edge_4pct":
        return (efg_margin >= 0.04) & (model_edge_pre >= 3.0)
    if rule_id == "away_efg_edge_4pct":
        return (efg_margin <= -0.04) & (model_edge_pre <= -3.0)
    if rule_id == "home_oreb_edge_3pct":
        return ((oreb_margin >= 0.03) | ((home_oreb >= 0.31) & (away_dreb <= 0.68))) & (model_edge_pre >= 2.0)
    if rule_id == "away_oreb_edge_3pct":
        return ((oreb_margin <= -0.03) | ((away_oreb >= 0.31) & (home_dreb <= 0.68))) & (model_edge_pre <= -2.0)
    if rule_id == "home_pace_mismatch_fast_edge":
        return ((home_pace - away_pace) >= 5.0) & (model_edge_pre >= 2.0)
    if rule_id == "away_pace_mismatch_fast_edge":
        return ((away_pace - home_pace) >= 5.0) & (model_edge_pre <= -2.0)
    if rule_id == "fade_away_threept_vs_slow":
        return (away_three >= 0.39) & (home_pace <= 66.5)
    if rule_id == "fade_home_threept_vs_slow":
        return (home_three >= 0.39) & (away_pace <= 66.5)
    if rule_id == "fade_home_weak_sos_form_pop":
        return (home_form >= 1.0) & (home_sos <= 0.0) & (model_edge_pre >= 2.0)
    if rule_id == "fade_away_weak_sos_form_pop":
        return (away_form >= 1.0) & (away_sos <= 0.0) & (model_edge_pre <= -2.0)
    if rule_id == "conference_bye_edge_home":
        return is_conf_tourney & (rest_diff >= 1.0)
    if rule_id == "conference_bye_edge_away":
        return is_conf_tourney & (rest_diff <= -1.0)
    if rule_id == "conference_fatigue_fade_home":
        return is_conf_tourney & (home_rest <= 0.0) & (away_rest >= 1.0)
    if rule_id == "conference_fatigue_fade_away":
        return is_conf_tourney & (away_rest <= 0.0) & (home_rest >= 1.0)
    if rule_id == "fade_blueblood_home":
        return away_is_blueblood & (model_edge_pre >= 3.0)
    if rule_id == "fade_blueblood_away":
        return home_is_blueblood & (model_edge_pre <= -3.0)
    return pd.Series(False, index=df.index)


def discover_situational_rules(
    historical_games: pd.DataFrame,
    *,
    min_sample: int = MIN_RULE_SAMPLE,
    shrink_k: float = RULE_SHRINK_K,
) -> pd.DataFrame:
    df = historical_games.copy()
    if "actual_home_covered" not in df.columns:
        margin = pd.to_numeric(df.get("actual_margin"), errors="coerce")
        spread = pd.to_numeric(df.get("market_spread"), errors="coerce")
        df["actual_home_covered"] = (margin + spread) > 0

    rows: list[dict[str, object]] = []
    for rule in _rule_definitions():
        mask = _rule_mask(df, rule.rule_id)
        subset = df[mask].copy()
        n = int(len(subset))

        if n == 0:
            rows.append(
                {
                    "rule_id": rule.rule_id,
                    "description": rule.description,
                    "direction": int(rule.direction),
                    "sample_size": 0,
                    "raw_ats_rate": np.nan,
                    "shrunk_ats_rate": 0.5,
                    "effect": 0.0,
                    "accepted": False,
                }
            )
            continue

        home_cover = subset["actual_home_covered"].astype(float)
        side_cover_rate = home_cover.mean() if rule.direction > 0 else (1.0 - home_cover).mean()
        shrunk_rate = ((n * side_cover_rate) + (shrink_k * 0.5)) / (n + shrink_k)
        effect = (shrunk_rate - 0.5) * float(rule.direction)

        accepted = bool((n >= min_sample) and (abs(effect) >= 0.012))

        rows.append(
            {
                "rule_id": rule.rule_id,
                "description": rule.description,
                "direction": int(rule.direction),
                "sample_size": n,
                "raw_ats_rate": float(side_cover_rate),
                "shrunk_ats_rate": float(shrunk_rate),
                "effect": float(effect),
                "accepted": accepted,
            }
        )

    return pd.DataFrame(rows)


def apply_situational_layer(game_frame: pd.DataFrame, rulebook: pd.DataFrame) -> pd.DataFrame:
    out = game_frame.copy()

    rule_map = {
        str(r["rule_id"]): r
        for _, r in rulebook.iterrows()
        if bool(r.get("accepted", False))
    }

    active_rules: list[str] = []
    scores: list[float] = []
    adjustments: list[float] = []
    conf_adj: list[float] = []
    upset_candidates = _prepare_upset_candidate_masks(out)
    underdog_dir = _underdog_direction(out.get("market_spread", pd.Series(np.nan, index=out.index)))

    for idx, _row in out.iterrows():
        active: list[str] = []
        score = 0.0
        one_row = out.loc[[idx]]

        for rule_id, rule in rule_map.items():
            if bool(_rule_mask(one_row, rule_id).iloc[0]):
                eff = float(rule.get("effect", 0.0))
                active.append(rule_id)
                score += eff

        if upset_candidates:
            direction = float(pd.to_numeric(underdog_dir.loc[idx], errors="coerce"))
            if direction != 0.0:
                for candidate in upset_candidates:
                    cand_mask = candidate["mask"]
                    if bool(cand_mask.loc[idx]):
                        active.append(str(candidate["rule_id"]))
                        score += float(candidate["effect"]) * direction

        spread_adj = float(np.clip(score * 3.0, -1.5, 1.5))
        confidence_boost = float(np.clip(abs(score) * 20.0, 0.0, 12.0))

        active_rules.append("|".join(active) if active else "")
        scores.append(score)
        adjustments.append(spread_adj)
        conf_adj.append(confidence_boost)

    out["situational_active_rules"] = active_rules
    out["situational_score"] = scores
    out["situational_spread_adjustment"] = adjustments
    out["situational_confidence_boost"] = conf_adj
    out["situational_signal"] = np.sign(pd.to_numeric(out["situational_score"], errors="coerce").fillna(0.0)).astype(int)
    return out
