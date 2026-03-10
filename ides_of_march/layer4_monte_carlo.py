from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from cbb_monte_carlo import GameSimInput, simulate_game  # type: ignore
except Exception:  # pragma: no cover - fallback path for local testability
    GameSimInput = None
    simulate_game = None


def _fallback_mc_row(row: pd.Series) -> dict[str, float]:
    margin = float(pd.to_numeric(pd.Series([row.get("projected_margin_pre_mc")]), errors="coerce").iloc[0] or 0.0)
    market_spread = float(pd.to_numeric(pd.Series([row.get("market_spread")]), errors="coerce").iloc[0] or 0.0)
    spread_std = 11.0

    home_win_prob = float(1.0 / (1.0 + np.exp(-margin / 6.0)))
    home_cover_prob = float(1.0 / (1.0 + np.exp(-(margin + market_spread) / 4.5)))

    return {
        "mc_home_win_prob": home_win_prob,
        "mc_away_win_prob": 1.0 - home_win_prob,
        "mc_home_cover_prob": home_cover_prob,
        "mc_away_cover_prob": 1.0 - home_cover_prob,
        "mc_ats_cover_prob": max(home_cover_prob, 1.0 - home_cover_prob),
        "mc_total_p10": float(pd.to_numeric(pd.Series([row.get("projected_total_ctx")]), errors="coerce").iloc[0] - 14.0),
        "mc_total_p90": float(pd.to_numeric(pd.Series([row.get("projected_total_ctx")]), errors="coerce").iloc[0] + 14.0),
        "mc_margin_p10": margin - 1.28 * spread_std,
        "mc_margin_p90": margin + 1.28 * spread_std,
        "mc_volatility": spread_std,
    }


def apply_monte_carlo_layer(
    game_frame: pd.DataFrame,
    *,
    mode: str = "confidence_only",
    n_sims: int = 3000,
    fast_approx: bool = False,
) -> pd.DataFrame:
    out = game_frame.copy()
    out["projected_margin_pre_mc"] = pd.to_numeric(out.get("margin_ctx_blend"), errors="coerce").fillna(0.0) + pd.to_numeric(
        out.get("situational_spread_adjustment"), errors="coerce"
    ).fillna(0.0)

    if fast_approx:
        margin = pd.to_numeric(out["projected_margin_pre_mc"], errors="coerce").fillna(0.0)
        market_spread = pd.to_numeric(out.get("market_spread"), errors="coerce").fillna(0.0)
        total = pd.to_numeric(out.get("projected_total_ctx"), errors="coerce").fillna(138.0)
        spread_std = 11.0
        home_win_prob = 1.0 / (1.0 + np.exp(-margin / 6.0))
        home_cover_prob = 1.0 / (1.0 + np.exp(-(margin + market_spread) / 4.5))

        out["mc_home_win_prob"] = home_win_prob
        out["mc_away_win_prob"] = 1.0 - home_win_prob
        out["mc_home_cover_prob"] = home_cover_prob
        out["mc_away_cover_prob"] = 1.0 - home_cover_prob
        out["mc_ats_cover_prob"] = np.maximum(home_cover_prob, 1.0 - home_cover_prob)
        out["mc_total_p10"] = total - 14.0
        out["mc_total_p90"] = total + 14.0
        out["mc_margin_p10"] = margin - (1.28 * spread_std)
        out["mc_margin_p90"] = margin + (1.28 * spread_std)
        out["mc_volatility"] = spread_std
        out["mc_signal"] = np.sign(pd.to_numeric(out.get("mc_home_cover_prob"), errors="coerce").fillna(0.5) - 0.5).astype(int)
        out["mc_filter_pass"] = pd.to_numeric(out.get("mc_volatility"), errors="coerce").fillna(99.0) <= 12.5
        out["mc_mode"] = mode
        out["mc_blend_weight"] = 0.4 if mode == "blended" else 0.0
        return out

    rows: list[dict[str, float]] = []
    for _, row in out.iterrows():
        if simulate_game is None or GameSimInput is None:
            rows.append(_fallback_mc_row(row))
            continue

        try:
            spread_home_line = pd.to_numeric(pd.Series([row.get("market_spread")]), errors="coerce").iloc[0]
            total_line = pd.to_numeric(pd.Series([row.get("market_total")]), errors="coerce").iloc[0]
            projected_spread_home_line = -float(row.get("projected_margin_pre_mc", 0.0))
            projected_total = float(pd.to_numeric(pd.Series([row.get("projected_total_ctx")]), errors="coerce").iloc[0] or 138.0)

            sim_in = GameSimInput(
                game_id=str(row.get("game_id", "")),
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
                home_team_id=str(row.get("home_team_id", "")),
                away_team_id=str(row.get("away_team_id", "")),
                ensemble_spread=projected_spread_home_line,
                ensemble_total=projected_total,
                spread_line=float(spread_home_line) if pd.notna(spread_home_line) else None,
                total_line=float(total_line) if pd.notna(total_line) else None,
                home_cage_t=float(pd.to_numeric(pd.Series([row.get("home_pace_l5")]), errors="coerce").iloc[0] or 68.0),
                away_cage_t=float(pd.to_numeric(pd.Series([row.get("away_pace_l5")]), errors="coerce").iloc[0] or 68.0),
                home_consistency=60.0,
                away_consistency=60.0,
                edge_flag=0,
            )
            result = simulate_game(sim_in, n_sims=n_sims)
            rows.append(
                {
                    "mc_home_win_prob": float(result.home_win_pct),
                    "mc_away_win_prob": float(result.away_win_pct),
                    "mc_home_cover_prob": float(result.home_covers_pct) if result.home_covers_pct is not None else np.nan,
                    "mc_away_cover_prob": float(result.away_covers_pct) if result.away_covers_pct is not None else np.nan,
                    "mc_ats_cover_prob": float(result.cover_probability) if result.cover_probability is not None else np.nan,
                    "mc_total_p10": float(result.total_p10),
                    "mc_total_p90": float(result.total_p90),
                    "mc_margin_p10": float(result.spread_p10),
                    "mc_margin_p90": float(result.spread_p90),
                    "mc_volatility": float(result.spread_std_realized),
                }
            )
        except Exception:
            rows.append(_fallback_mc_row(row))

    mc = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, mc], axis=1)

    out["mc_signal"] = np.sign(pd.to_numeric(out.get("mc_home_cover_prob"), errors="coerce").fillna(0.5) - 0.5).astype(int)
    out["mc_filter_pass"] = (
        pd.to_numeric(out.get("mc_volatility"), errors="coerce").fillna(99.0) <= 12.5
    )
    out["mc_mode"] = mode
    out["mc_blend_weight"] = 0.4 if mode == "blended" else 0.0
    return out
