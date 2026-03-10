from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MIN_RULE_SAMPLE, RULE_SHRINK_K


@dataclass(frozen=True)
class RuleDef:
    rule_id: str
    description: str
    direction: int  # +1 home side, -1 away side


def _rule_definitions() -> list[RuleDef]:
    return [
        RuleDef("home_rest_form_edge", "Home + rest edge + positive form delta", +1),
        RuleDef("road_favorite_short_rest", "Road favorite on short rest", -1),
        RuleDef("home_oreb_vs_weak_dreb", "Home OREB edge vs weak away DREB", +1),
        RuleDef("away_threept_vs_slow", "Away high 3PT dependence vs slow home pace", -1),
        RuleDef("home_threept_vs_slow", "Home high 3PT dependence vs slow away pace", +1),
        RuleDef("home_improving_weak_sos", "Home recent improvement vs weak SOS", +1),
        RuleDef("away_improving_weak_sos", "Away recent improvement vs weak SOS", -1),
    ]


def _rule_mask(df: pd.DataFrame, rule_id: str) -> pd.Series:
    def _num(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    rest_diff = _num("rest_diff")
    form_diff = _num("form_delta_diff")
    market_spread = _num("market_spread")
    away_rest = _num("away_days_rest")
    home_oreb = _num("home_orb_pct_l5")
    away_dreb = _num("away_drb_pct_l5")
    away_three = _num("away_three_par_l5")
    home_three = _num("home_three_par_l5")
    home_pace = _num("home_pace_l5")
    away_pace = _num("away_pace_l5")
    home_form = _num("home_Form_Delta")
    away_form = _num("away_Form_Delta")
    home_sos = _num("home_sos_pre")
    away_sos = _num("away_sos_pre")

    if rule_id == "home_rest_form_edge":
        return (rest_diff >= 2.0) & (form_diff >= 0.8)
    if rule_id == "road_favorite_short_rest":
        return (market_spread > 0) & (away_rest <= 1.0)
    if rule_id == "home_oreb_vs_weak_dreb":
        return (home_oreb >= 0.31) & (away_dreb <= 0.68)
    if rule_id == "away_threept_vs_slow":
        return (away_three >= 0.38) & (home_pace <= 67.0)
    if rule_id == "home_threept_vs_slow":
        return (home_three >= 0.38) & (away_pace <= 67.0)
    if rule_id == "home_improving_weak_sos":
        return (home_form >= 1.0) & (home_sos <= 0.0)
    if rule_id == "away_improving_weak_sos":
        return (away_form >= 1.0) & (away_sos <= 0.0)
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

        accepted = bool((n >= min_sample) and (abs(effect) >= 0.015))

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

    for idx, row in out.iterrows():
        active: list[str] = []
        score = 0.0
        one_row = out.loc[[idx]]

        for rule_id, rule in rule_map.items():
            if bool(_rule_mask(one_row, rule_id).iloc[0]):
                eff = float(rule.get("effect", 0.0))
                active.append(rule_id)
                score += eff

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
