"""
CBB analytics/model configuration shared across cbb_* and espn_* modules.
"""

import json
from pathlib import Path
from typing import Dict

LEAGUE_AVG_ORTG = 103.0
LEAGUE_AVG_DRTG = 103.0
LEAGUE_AVG_PACE = 70.0
LEAGUE_AVG_EFG  = 50.5
LEAGUE_AVG_TOV  = 18.0
LEAGUE_AVG_FTR  = 28.0
LEAGUE_AVG_ORB  = 30.0
LEAGUE_AVG_DRB  = 70.0

HCA            = 3.2
PYTH_EXP       = 11.5
SIGMA          = 11.0
EFF_TO_PTS     = LEAGUE_AVG_PACE / 100.0
VIG_BREAK_EVEN = 52.38

QUAD_1_MIN_NET = 8.0
QUAD_2_MIN_NET = 0.0
QUAD_3_MIN_NET = -8.0

ENSEMBLE_MODEL_NAMES = [
    "FourFactors",
    "AdjEfficiency",
    "Pythagorean",
    "Momentum",
    "Situational",
    "CAGERankings",
    "RegressedEff",
]

DEFAULT_SPREAD_WEIGHTS = {
    "FourFactors":   0.12,
    "AdjEfficiency": 0.22,
    "Pythagorean":   0.14,
    "Momentum":      0.16,
    "Situational":   0.10,
    "CAGERankings":  0.18,
    "RegressedEff":  0.08,
}

DEFAULT_TOTAL_WEIGHTS = {
    "FourFactors":   0.15,
    "AdjEfficiency": 0.24,
    "Pythagorean":   0.10,
    "Momentum":      0.14,
    "Situational":   0.12,
    "CAGERankings":  0.17,
    "RegressedEff":  0.08,
}

WEIGHTS_PATH = Path("data") / "backtest_optimized_weights.json"


def load_ensemble_weights() -> Dict[str, Dict[str, float]]:
    spread = dict(DEFAULT_SPREAD_WEIGHTS)
    total = dict(DEFAULT_TOTAL_WEIGHTS)
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 10:
        try:
            payload = json.loads(WEIGHTS_PATH.read_text())
            if isinstance(payload.get("weights"), dict):
                spread.update(payload["weights"])
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return {"spread": spread, "total": total}
