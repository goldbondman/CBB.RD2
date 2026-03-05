#!/usr/bin/env python3
"""
Advanced Metrics Codex for CBB.

Sequential joint architecture:
1) Totals model predicts scoring environment (pred_total).
2) Spread model allocates that environment into team shares (allocation_pct).
3) Team scores are reconciled exactly:
   home + away == total, home - away == margin.

This keeps environment (sum) features separated from relative (diff) features
to avoid double-counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import erf, sqrt, tanh
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_numeric(
    mapping: Optional[Mapping[str, object]],
    keys: Sequence[str],
    default: float = 0.0,
) -> float:
    if not mapping:
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return _safe_float(mapping[key], default)
    return default


def _std(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr, ddof=0))


def _mean_abs(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def _z(value: float, key: str, z_context: Optional[Mapping[str, Mapping[str, float]]]) -> float:
    if not z_context or key not in z_context:
        return 0.0
    mean = _safe_float(z_context[key].get("mean"), 0.0)
    std = max(_safe_float(z_context[key].get("std"), 0.0), 1e-9)
    return (value - mean) / std


def american_to_implied_prob(odds: float) -> Optional[float]:
    if odds is None:
        return None
    o = _safe_float(odds, 0.0)
    if o == 0.0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


@dataclass
class GameInputs:
    home_team: str
    away_team: str
    game_date: date
    closing_spread: float
    closing_total: float
    closing_ml_home: Optional[float] = None
    closing_ml_away: Optional[float] = None
    home_team_stats: Dict[str, object] = field(default_factory=dict)
    away_team_stats: Dict[str, object] = field(default_factory=dict)
    home_rotation: List[Dict[str, object]] = field(default_factory=list)
    away_rotation: List[Dict[str, object]] = field(default_factory=list)
    days_rest_home: float = 0.0
    days_rest_away: float = 0.0
    back2back_home: bool = False
    back2back_away: bool = False
    elevation_home: float = 0.0
    elevation_away: float = 0.0
    cross_country_miles: float = 0.0


@dataclass
class ModelConfig:
    spread_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "odi_star_diff": 0.25,
            "sme_diff": 0.20,
            "posw_diff": 0.15,
            "pxp_diff": 0.15,
            "lns_diff": 0.15,
            "vol_diff": 0.10,
        }
    )
    totals_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "posw_sum": 0.25,
            "sch": 0.20,
            "wl_sum": 0.20,
            "svi_avg": 0.20,
            "tc_diff": 0.15,
        }
    )
    # Optional extended model weights from your "ultimate" spec.
    spread_weights_ultimate: Dict[str, float] = field(
        default_factory=lambda: {
            "odi_star_diff": 0.22,
            "sme_diff": 0.18,
            "posw_diff": 0.13,
            "pxp_diff": 0.12,
            "lns_diff": 0.12,
            "away_efg_diff": 0.10,
            "rfd": 0.07,
            "vol_diff": 0.06,
        }
    )
    totals_weights_ultimate: Dict[str, float] = field(
        default_factory=lambda: {
            "posw_sum": 0.22,
            "sch": 0.17,
            "wl_sum": 0.17,
            "svi_avg": 0.17,
            "tc_diff": 0.12,
            "alt_diff": 0.08,
            "rfd_sum": 0.07,
        }
    )
    use_ultimate: bool = True
    allocation_cap: float = 0.20
    allocation_scale: float = 0.08
    totals_adjustment_scale: float = 6.0
    spread_edge_threshold: float = 1.5
    total_edge_threshold: float = 2.0
    ml_edge_threshold: float = 0.05
    margin_sigma: float = 11.0
    spread_feature_scales: Dict[str, float] = field(
        default_factory=lambda: {
            "odi_star_diff": 1.0,
            "sme_diff": 1.0,
            "posw_diff": 2.0,
            "pxp_diff": 1.0,
            "lns_diff": 1.0,
            "vol_diff": 8.0,
            "away_efg_diff": 0.05,
            "rfd": 3.0,
            "pei_diff": 0.2,
        }
    )
    totals_feature_scales: Dict[str, float] = field(
        default_factory=lambda: {
            "posw_sum": 2.0,
            "sch": 8.0,
            "wl_sum": 2.0,
            "svi_avg": 2.0,
            "tc_diff": 2.0,
            "alt_diff": 1000.0,
            "rfd_sum": 3.0,
        }
    )

    def __post_init__(self) -> None:
        for weights in (
            self.spread_weights,
            self.totals_weights,
            self.spread_weights_ultimate,
            self.totals_weights_ultimate,
        ):
            self._normalize_weight_dict(weights)

    @staticmethod
    def _normalize_weight_dict(weights: Dict[str, float]) -> None:
        total = sum(max(v, 0.0) for v in weights.values())
        if total <= 0:
            return
        for k, v in list(weights.items()):
            weights[k] = max(v, 0.0) / total


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def compute_odi_star(
    off_adj: Mapping[str, object],
    opp_def_adj: Mapping[str, object],
    league_avg: Mapping[str, object],
) -> float:
    efg_odi = (
        _first_numeric(off_adj, ("efg", "efg_adj"), 0.0) -
        _first_numeric(league_avg, ("efg",), 0.0)
    ) - (
        _first_numeric(opp_def_adj, ("efg", "efg_adj"), 0.0) -
        _first_numeric(league_avg, ("efg",), 0.0)
    )
    tov_odi = (
        _first_numeric(opp_def_adj, ("tov", "tov_adj"), 0.0) -
        _first_numeric(league_avg, ("tov",), 0.0)
    ) - (
        _first_numeric(off_adj, ("tov", "tov_adj"), 0.0) -
        _first_numeric(league_avg, ("tov",), 0.0)
    )
    orb_odi = (
        _first_numeric(off_adj, ("orb", "orb_adj"), 0.0) -
        _first_numeric(league_avg, ("orb",), 0.0)
    ) - (
        _first_numeric(opp_def_adj, ("orb", "orb_adj"), 0.0) -
        _first_numeric(league_avg, ("orb",), 0.0)
    )
    ftr_odi = (
        _first_numeric(off_adj, ("ftr", "ftr_adj"), 0.0) -
        _first_numeric(league_avg, ("ftr",), 0.0)
    ) - (
        _first_numeric(opp_def_adj, ("ftr", "ftr_adj"), 0.0) -
        _first_numeric(league_avg, ("ftr",), 0.0)
    )
    return 0.4 * efg_odi + 0.25 * tov_odi + 0.2 * orb_odi + 0.15 * ftr_odi


def compute_pei(orb_off_adj: float, tov_off_adj: float, orb_def_adj: float, tov_def_adj: float) -> float:
    return (orb_off_adj - orb_def_adj) + (tov_def_adj - tov_off_adj)


def compute_posw(pei: float, pace_team: float, pace_opp: float) -> float:
    projected_pace = (pace_team + pace_opp) / 2.0
    return pei * projected_pace / 100.0


def compute_svi(
    efg: float,
    three_pa_rate: float,
    ft_pts_per_fga: float,
    z_context: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> float:
    return (
        _z(efg, "efg", z_context)
        + 0.5 * _z(three_pa_rate, "three_pa_rate", z_context)
        + 0.5 * _z(ft_pts_per_fga, "ft_pts_per_fga", z_context)
    )


def compute_ics(returning_minutes: float, rotation_stability: float, star_continuity: float) -> float:
    return 0.4 * returning_minutes + 0.3 * rotation_stability + 0.3 * star_continuity


def compute_bsi(bench_minutes: float, bench_ts_rel: float, bench_reb_rel: float) -> float:
    return bench_minutes * (0.6 * bench_ts_rel + 0.4 * bench_reb_rel)


def compute_pxp(ics: float, bsi: float, clutch_pct: float) -> float:
    return 0.4 * ics + 0.3 * bsi + 0.3 * clutch_pct


def compute_lns(rotation: Sequence[Mapping[str, object]]) -> float:
    return sum(
        _first_numeric(p, ("minutes_share",), 0.0) * _first_numeric(p, ("tois", "player_tois"), 0.0)
        for p in rotation
    )


def compute_usef(
    star_ts: float,
    team_ts: float,
    star_usage: float,
    rotation: Sequence[Mapping[str, object]],
) -> float:
    sie = (star_ts - team_ts) * star_usage
    top_usage = sorted(
        rotation,
        key=lambda r: _first_numeric(r, ("usage", "usage_rate"), 0.0),
        reverse=True,
    )[:3]
    usage_sum = sum(_first_numeric(p, ("usage", "usage_rate"), 0.0) for p in top_usage)
    if usage_sum <= 0:
        weighted_ts_top_cluster = np.mean(
            [_first_numeric(p, ("ts", "ts_pct"), 0.0) for p in top_usage]
        ) if top_usage else 0.0
    else:
        weighted_ts_top_cluster = sum(
            _first_numeric(p, ("ts", "ts_pct"), 0.0) * _first_numeric(p, ("usage", "usage_rate"), 0.0)
            for p in top_usage
        ) / usage_sum
    return sie + weighted_ts_top_cluster


def compute_dpc(bsi: float, best_rotation_netrtg: float, worst_rotation_netrtg: float) -> float:
    return bsi + (best_rotation_netrtg - worst_rotation_netrtg)


def compute_fii(dpc: float, backup_quality_key_positions: float, ffc_inverted: float) -> float:
    return dpc * backup_quality_key_positions + ffc_inverted


def compute_sme(star_profile: Mapping[str, object], opp_weakness_profile: Mapping[str, object]) -> float:
    star_vec = np.array(
        [
            _first_numeric(star_profile, ("3pa_rate", "three_pa_rate"), 0.0),
            _first_numeric(star_profile, ("rim_rate",), 0.0),
            _first_numeric(star_profile, ("midrange_rate", "mid_rate"), 0.0),
            _first_numeric(star_profile, ("post_ups", "post_up_rate"), 0.0),
        ],
        dtype=float,
    )
    opp_vec = np.array(
        [
            _first_numeric(opp_weakness_profile, ("opp3p_allowed", "opp_3p_allowed"), 0.0),
            _first_numeric(opp_weakness_profile, ("opp_rim_fg_allowed",), 0.0),
            _first_numeric(opp_weakness_profile, ("opp_mid_fg_allowed",), 0.0),
            _first_numeric(opp_weakness_profile, ("opp_post_ups_allowed", "opp_post_def_weakness"), 0.0),
        ],
        dtype=float,
    )
    return float(np.dot(star_vec, opp_vec))


def compute_sch(
    pace_a: float,
    pace_b: float,
    three_pa_rate_a: float,
    three_pa_rate_b: float,
    ftr_a: float,
    ftr_b: float,
    size_a: float,
    size_b: float,
) -> float:
    return (
        abs(pace_a - pace_b)
        + abs(three_pa_rate_a - three_pa_rate_b)
        + abs(ftr_a - ftr_b)
        + abs(size_a - size_b)
    )


def compute_vol(netrtg_adj_l8: Sequence[float]) -> float:
    return _std(netrtg_adj_l8)


def compute_tc(
    game_pace_l12: Sequence[float],
    team_season_pace_avg: float,
    z_context: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> float:
    mad = _mean_abs([_safe_float(p, 0.0) - team_season_pace_avg for p in game_pace_l12])
    return -_z(mad, "pace_deviation", z_context)


def compute_wl(
    ft_ppp: float,
    opp_ft_ppp: float,
    ftr: float,
    opp_ftr: float,
    z_context: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> float:
    return _z(ft_ppp - opp_ft_ppp, "ft_ppp_diff", z_context) + 0.5 * _z(ftr - opp_ftr, "ftr_diff", z_context)


def compute_rfd(days_rest_home: float, days_rest_away: float, back2back_home: bool, back2back_away: bool) -> float:
    back2back_penalty = float(back2back_away) - float(back2back_home)
    return (days_rest_home - days_rest_away) + 0.5 * back2back_penalty


def compute_gsr(tournament_stage: float, elimination_risk: float, spread_magnitude: float) -> float:
    return 0.4 * tournament_stage + 0.3 * elimination_risk + 0.3 * spread_magnitude


def compute_alt(elevation_home: float, elevation_away: float, cross_country_miles: float) -> float:
    return (elevation_home - elevation_away) + (cross_country_miles / 1000.0)


# ---------------------------------------------------------------------------
# Sequential joint model
# ---------------------------------------------------------------------------

class AdvancedMetricsCodex:
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        league_averages: Optional[Mapping[str, float]] = None,
        z_context: Optional[Mapping[str, Mapping[str, float]]] = None,
    ):
        self.config = config or ModelConfig()
        self.league_averages = league_averages or {
            "efg": 0.50,
            "tov": 0.18,
            "orb": 0.30,
            "ftr": 0.30,
        }
        self.z_context = z_context or {}

    def compute_team_metrics(
        self,
        team_stats: Mapping[str, object],
        opp_stats: Mapping[str, object],
        rotation: Sequence[Mapping[str, object]],
        opp_rotation: Sequence[Mapping[str, object]],
        *,
        side: str,
    ) -> Dict[str, float]:
        is_home = side == "home"

        efg = _first_numeric(team_stats, ("efg_home_adj", "efg_away_adj", "efg_adj", "efg"), 0.50)
        tov = _first_numeric(team_stats, ("tov_home_adj", "tov_away_adj", "tov_adj", "tov_pct_adj"), 0.18)
        orb = _first_numeric(team_stats, ("orb_home_adj", "orb_away_adj", "orb_adj", "orb_pct_adj"), 0.30)
        ftr = _first_numeric(team_stats, ("ftr_home_adj", "ftr_away_adj", "ftr_adj", "ftr"), 0.30)
        pace = _first_numeric(team_stats, ("pace_home", "pace_away", "pace_adj", "pace"), 67.0)
        drb = _first_numeric(team_stats, ("drb_home_adj", "drb_away_adj", "drb_adj", "drb_pct_adj"), 0.70)

        opp_def_efg = _first_numeric(opp_stats, ("efg_def_home_adj", "efg_def_away_adj", "efg_def_adj", "opp_efg_allowed"), 0.50)
        opp_def_tov = _first_numeric(opp_stats, ("tov_def_home_adj", "tov_def_away_adj", "tov_def_adj"), 0.18)
        opp_def_orb = _first_numeric(opp_stats, ("orb_def_home_adj", "orb_def_away_adj", "orb_def_adj"), 0.30)
        opp_def_ftr = _first_numeric(opp_stats, ("ftr_def_home_adj", "ftr_def_away_adj", "ftr_def_adj", "opp_ftr"), 0.30)
        opp_pace = _first_numeric(opp_stats, ("pace_home", "pace_away", "pace_adj", "pace"), 67.0)

        ftm = _first_numeric(team_stats, ("ftm_home", "ftm_away", "ftm"), 0.0)
        fga = max(_first_numeric(team_stats, ("fga_home", "fga_away", "fga"), 1.0), 1.0)
        three_pa_rate = _first_numeric(team_stats, ("three_pa_rate_home", "three_pa_rate_away", "three_pa_rate", "three_pa_fga"), 0.35)
        ft_pts_per_fga = ftm / fga

        odi = compute_odi_star(
            {"efg": efg, "tov": tov, "orb": orb, "ftr": ftr},
            {"efg": opp_def_efg, "tov": opp_def_tov, "orb": opp_def_orb, "ftr": opp_def_ftr},
            self.league_averages,
        )
        pei = compute_pei(orb, tov, opp_def_orb, opp_def_tov)
        posw = compute_posw(pei, pace, opp_pace)
        svi = compute_svi(efg, three_pa_rate, ft_pts_per_fga, self.z_context)

        returning_minutes = _first_numeric(team_stats, ("returning_minutes",), 0.0)
        rotation_stability = _first_numeric(team_stats, ("rotation_stability",), 0.0)
        star_continuity = _first_numeric(team_stats, ("star_continuity",), 0.0)
        clutch_pct = _first_numeric(team_stats, ("clutch_pct",), 0.0)
        bench_minutes = _first_numeric(team_stats, ("bench_minutes",), 0.0)
        bench_ts_rel = _first_numeric(team_stats, ("bench_ts_rel",), 0.0)
        bench_reb_rel = _first_numeric(team_stats, ("bench_reb_rel",), 0.0)
        ics = compute_ics(returning_minutes, rotation_stability, star_continuity)
        bsi = compute_bsi(bench_minutes, bench_ts_rel, bench_reb_rel)
        pxp = compute_pxp(ics, bsi, clutch_pct)

        lns = compute_lns(rotation)
        star_ts = _first_numeric(team_stats, ("star_ts", "star_ts_pct"), 0.0)
        team_ts = _first_numeric(team_stats, ("team_ts", "team_ts_pct"), 0.0)
        star_usage = _first_numeric(team_stats, ("star_usage", "star_usage_rate"), 0.0)
        usef = compute_usef(star_ts, team_ts, star_usage, rotation)

        best_rot = _first_numeric(team_stats, ("best_rotation_netrtg",), 0.0)
        worst_rot = _first_numeric(team_stats, ("worst_rotation_netrtg",), 0.0)
        dpc = compute_dpc(bsi, best_rot, worst_rot)
        backup_quality = _first_numeric(team_stats, ("backup_quality_key_positions",), 0.0)
        ffc_inverted = _first_numeric(team_stats, ("ffc_inverted",), 0.0)
        fii = compute_fii(dpc, backup_quality, ffc_inverted)

        star_profile = {
            "three_pa_rate": _first_numeric(team_stats, ("star_three_pa_rate",), 0.0),
            "rim_rate": _first_numeric(team_stats, ("star_rim_rate",), 0.0),
            "midrange_rate": _first_numeric(team_stats, ("star_midrange_rate",), 0.0),
            "post_ups": _first_numeric(team_stats, ("star_post_ups",), 0.0),
        }
        opp_weakness = {
            "opp3p_allowed": _first_numeric(opp_stats, ("opp3p_allowed", "opp_3p_allowed"), 0.0),
            "opp_rim_fg_allowed": _first_numeric(opp_stats, ("opp_rim_fg_allowed",), 0.0),
            "opp_mid_fg_allowed": _first_numeric(opp_stats, ("opp_mid_fg_allowed",), 0.0),
            "opp_post_ups_allowed": _first_numeric(opp_stats, ("opp_post_ups_allowed", "opp_post_def_weakness"), 0.0),
        }
        sme = compute_sme(star_profile, opp_weakness)

        size = _first_numeric(team_stats, ("lineup_height_avg", "size"), 78.5)
        opp_size = _first_numeric(opp_stats, ("lineup_height_avg", "size"), 78.5)
        sch = compute_sch(
            pace, opp_pace,
            three_pa_rate,
            _first_numeric(opp_stats, ("three_pa_rate_home", "three_pa_rate_away", "three_pa_rate"), 0.35),
            ftr,
            _first_numeric(opp_stats, ("ftr_home_adj", "ftr_away_adj", "ftr_adj", "ftr"), 0.30),
            size, opp_size,
        )

        netrtg_adj_l8 = team_stats.get("netrtg_adj_l8", [])
        if not isinstance(netrtg_adj_l8, (list, tuple, np.ndarray)):
            netrtg_adj_l8 = []
        vol = compute_vol([_safe_float(v, 0.0) for v in netrtg_adj_l8])
        game_pace_l12 = team_stats.get("game_pace_l12", [])
        if not isinstance(game_pace_l12, (list, tuple, np.ndarray)):
            game_pace_l12 = []
        tc = compute_tc(
            [_safe_float(v, 0.0) for v in game_pace_l12],
            _first_numeric(team_stats, ("team_season_pace_avg", "pace_season_avg", "pace"), pace),
            self.z_context,
        )

        ft_ppp = _first_numeric(team_stats, ("ft_ppp_home", "ft_ppp_away", "ft_ppp"), 0.20)
        opp_ft_ppp = _first_numeric(opp_stats, ("ft_ppp_allowed_home", "ft_ppp_allowed_away", "opp_ft_ppp"), 0.20)
        wl = compute_wl(
            ft_ppp,
            opp_ft_ppp,
            ftr,
            _first_numeric(opp_stats, ("ftr_home_adj", "ftr_away_adj", "ftr_adj", "ftr"), 0.30),
            self.z_context,
        )

        if is_home:
            efg_split = _first_numeric(team_stats, ("efg_home_adj", "efg_adj", "efg"), efg)
        else:
            efg_split = _first_numeric(team_stats, ("efg_away_adj", "efg_adj", "efg"), efg)

        return {
            "odi_star": odi,
            "pei": pei,
            "posw": posw,
            "svi": svi,
            "ics": ics,
            "bsi": bsi,
            "pxp": pxp,
            "lns": lns,
            "usef": usef,
            "dpc": dpc,
            "fii": fii,
            "sme": sme,
            "sch": sch,
            "vol": vol,
            "tc": tc,
            "wl": wl,
            "pace": pace,
            "efg_split": efg_split,
            "rotation_count": float(len(rotation)),
            "opp_rotation_count": float(len(opp_rotation)),
            "raw_efg": efg,
            "raw_tov": tov,
            "raw_orb": orb,
            "raw_drb": drb,
            "raw_ftr": ftr,
        }

    def compute_totals_features(self, inputs: GameInputs) -> Dict[str, float]:
        home = self.compute_team_metrics(
            inputs.home_team_stats,
            inputs.away_team_stats,
            inputs.home_rotation,
            inputs.away_rotation,
            side="home",
        )
        away = self.compute_team_metrics(
            inputs.away_team_stats,
            inputs.home_team_stats,
            inputs.away_rotation,
            inputs.home_rotation,
            side="away",
        )
        alt = compute_alt(inputs.elevation_home, inputs.elevation_away, inputs.cross_country_miles)
        rfd = compute_rfd(
            inputs.days_rest_home,
            inputs.days_rest_away,
            inputs.back2back_home,
            inputs.back2back_away,
        )
        return {
            "posw_sum": home["posw"] + away["posw"],
            "sch": compute_sch(
                home["pace"], away["pace"],
                _first_numeric(inputs.home_team_stats, ("three_pa_rate_home", "three_pa_rate"), 0.35),
                _first_numeric(inputs.away_team_stats, ("three_pa_rate_away", "three_pa_rate"), 0.35),
                _first_numeric(inputs.home_team_stats, ("ftr_home_adj", "ftr_adj", "ftr"), home["raw_ftr"]),
                _first_numeric(inputs.away_team_stats, ("ftr_away_adj", "ftr_adj", "ftr"), away["raw_ftr"]),
                _first_numeric(inputs.home_team_stats, ("lineup_height_avg", "size"), 78.5),
                _first_numeric(inputs.away_team_stats, ("lineup_height_avg", "size"), 78.5),
            ),
            "wl_sum": home["wl"] + away["wl"],
            "svi_avg": (home["svi"] + away["svi"]) / 2.0,
            "tc_diff": home["tc"] - away["tc"],
            "alt_diff": alt,
            "rfd_sum": rfd,
            "_home": home,
            "_away": away,
        }

    def compute_spread_features(self, inputs: GameInputs) -> Dict[str, float]:
        totals_features = self.compute_totals_features(inputs)
        home = totals_features["_home"]
        away = totals_features["_away"]
        rfd = compute_rfd(
            inputs.days_rest_home,
            inputs.days_rest_away,
            inputs.back2back_home,
            inputs.back2back_away,
        )
        return {
            "odi_star_diff": home["odi_star"] - away["odi_star"],
            "sme_diff": home["sme"] - away["sme"],
            "pxp_diff": home["pxp"] - away["pxp"],
            "posw_diff": home["posw"] - away["posw"],
            "lns_diff": home["lns"] - away["lns"],
            "vol_diff": home["vol"] - away["vol"],
            "away_efg_diff": home["efg_split"] - away["efg_split"],
            "pei_diff": home["pei"] - away["pei"],
            "rfd": rfd,
            "_totals_features": totals_features,
        }

    def _weighted_signal(
        self,
        features: Mapping[str, float],
        weights: Mapping[str, float],
        scales: Optional[Mapping[str, float]] = None,
    ) -> float:
        out = 0.0
        for k, w in weights.items():
            scale = max(_safe_float((scales or {}).get(k), 1.0), 1e-9)
            out += (_safe_float(features.get(k), 0.0) / scale) * w
        return out

    def _base_total(self, inputs: GameInputs) -> float:
        pace_home = _first_numeric(inputs.home_team_stats, ("pace_home", "pace", "pace_adj"), 67.0)
        pace_away = _first_numeric(inputs.away_team_stats, ("pace_away", "pace", "pace_adj"), 67.0)
        projected_pace = (pace_home + pace_away) / 2.0

        home_ortg = _first_numeric(inputs.home_team_stats, ("ortg_home_adj", "ortg_adj", "ortg"), 104.0)
        away_ortg = _first_numeric(inputs.away_team_stats, ("ortg_away_adj", "ortg_adj", "ortg"), 104.0)
        home_ppp = home_ortg / 100.0
        away_ppp = away_ortg / 100.0
        return projected_pace * (home_ppp + away_ppp)

    def predict_game(self, inputs: GameInputs) -> Dict[str, object]:
        totals_features = self.compute_totals_features(inputs)
        spread_features = self.compute_spread_features(inputs)
        totals_weights = self.config.totals_weights_ultimate if self.config.use_ultimate else self.config.totals_weights
        spread_weights = self.config.spread_weights_ultimate if self.config.use_ultimate else self.config.spread_weights

        base_total = self._base_total(inputs)
        totals_signal = self._weighted_signal(
            totals_features,
            totals_weights,
            scales=self.config.totals_feature_scales,
        )
        pred_total = base_total + self.config.totals_adjustment_scale * totals_signal

        spread_signal = self._weighted_signal(
            spread_features,
            spread_weights,
            scales=self.config.spread_feature_scales,
        )
        allocation_pct = tanh(spread_signal) * self.config.allocation_scale
        allocation_pct = float(np.clip(allocation_pct, -self.config.allocation_cap, self.config.allocation_cap))

        pred_home_score = pred_total * (0.5 + allocation_pct)
        pred_away_score = pred_total * (0.5 - allocation_pct)
        pred_margin = pred_home_score - pred_away_score

        spread_edge = pred_margin - _safe_float(inputs.closing_spread, 0.0)
        total_edge = pred_total - _safe_float(inputs.closing_total, 0.0)

        sigma = max(self.config.margin_sigma, 1e-6)
        win_prob_home = 0.5 * (1.0 + erf(pred_margin / (sigma * sqrt(2.0))))
        ml_implied_home = american_to_implied_prob(inputs.closing_ml_home) if inputs.closing_ml_home is not None else None
        ml_edge_home = (win_prob_home - ml_implied_home) if ml_implied_home is not None else None

        recommendation = self.generate_bet_recommendations(
            spread_edge=spread_edge,
            total_edge=total_edge,
            ml_edge_home=ml_edge_home,
            closing_spread=inputs.closing_spread,
            closing_total=inputs.closing_total,
        )

        pred_home_r = round(pred_home_score, 1)
        pred_away_r = round(pred_away_score, 1)
        pred_total_r = round(pred_home_r + pred_away_r, 1)
        pred_margin_r = round(pred_home_r - pred_away_r, 1)

        return {
            "home_team": inputs.home_team,
            "away_team": inputs.away_team,
            "game_date": str(inputs.game_date),
            "pred_home_score": pred_home_r,
            "pred_away_score": pred_away_r,
            "pred_total": pred_total_r,
            "pred_margin": pred_margin_r,
            "allocation_pct": round(allocation_pct, 4),
            "spread_edge": round(spread_edge, 2),
            "total_edge": round(total_edge, 2),
            "win_prob_home": round(win_prob_home, 4),
            "ml_implied_prob_home": round(ml_implied_home, 4) if ml_implied_home is not None else None,
            "ml_edge_home": round(ml_edge_home, 4) if ml_edge_home is not None else None,
            "recommendations": recommendation,
            "totals_features": {k: v for k, v in totals_features.items() if not k.startswith("_")},
            "spread_features": {k: v for k, v in spread_features.items() if not k.startswith("_")},
        }

    def generate_bet_recommendations(
        self,
        *,
        spread_edge: float,
        total_edge: float,
        ml_edge_home: Optional[float],
        closing_spread: float,
        closing_total: float,
    ) -> List[str]:
        recs: List[str] = []
        if abs(spread_edge) >= self.config.spread_edge_threshold:
            side = "HOME" if spread_edge > 0 else "AWAY"
            recs.append(f"SPREAD: {side} (edge {spread_edge:+.2f} vs {closing_spread:+.1f})")
        if abs(total_edge) >= self.config.total_edge_threshold:
            side = "OVER" if total_edge > 0 else "UNDER"
            recs.append(f"TOTAL: {side} (edge {total_edge:+.2f} vs {closing_total:.1f})")
        if ml_edge_home is not None and abs(ml_edge_home) >= self.config.ml_edge_threshold:
            side = "HOME ML" if ml_edge_home > 0 else "AWAY ML"
            recs.append(f"ML: {side} (edge {ml_edge_home:+.2%})")
        return recs if recs else ["PASS - no threshold edge"]


def predict_cbb_game(
    inputs: GameInputs,
    *,
    config: Optional[ModelConfig] = None,
    league_averages: Optional[Mapping[str, float]] = None,
    z_context: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, object]:
    model = AdvancedMetricsCodex(config=config, league_averages=league_averages, z_context=z_context)
    return model.predict_game(inputs)
