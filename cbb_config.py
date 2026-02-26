"""
CBB analytics/model configuration shared across cbb_* and espn_* modules.
"""

import json
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

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
VIG_BREAK_EVEN = 52.38

QUAD_1_MIN_NET = 8.0
QUAD_2_MIN_NET = 0.0
QUAD_3_MIN_NET = -8.0

ENSEMBLE_MODEL_NAMES = [
    "FourFactors",
    "AdjEfficiency",
    "Pythagorean",
    "Situational",
    "CAGERankings",
    "LuckRegression",
    "Variance",
    "HomeAwayForm",
]

DEFAULT_SPREAD_WEIGHTS = {
    "FourFactors":   0.125,
    "AdjEfficiency": 0.125,
    "Pythagorean":   0.125,
    "Situational":   0.125,
    "CAGERankings":  0.125,
    "LuckRegression": 0.125,
    "Variance":      0.125,
    "HomeAwayForm":  0.125,
}

DEFAULT_TOTAL_WEIGHTS = {
    "FourFactors":   0.125,
    "AdjEfficiency": 0.125,
    "Pythagorean":   0.125,
    "Situational":   0.125,
    "CAGERankings":  0.125,
    "LuckRegression": 0.125,
    "Variance":      0.125,
    "HomeAwayForm":  0.125,
}


WEIGHT_SOURCES = [
    Path("data") / "active_weights.json",
    Path("data") / "backtest_optimized_weights.json",
]
WEIGHTS_PATH = WEIGHT_SOURCES[0]


def load_ensemble_weights() -> Dict[str, Dict[str, float]]:
    spread = dict(DEFAULT_SPREAD_WEIGHTS)
    total = dict(DEFAULT_TOTAL_WEIGHTS)
    resolved = next((p for p in WEIGHT_SOURCES if p.exists() and p.stat().st_size > 10), None)
    if resolved is not None:
        try:
            payload = json.loads(resolved.read_text())
            if isinstance(payload.get("weights"), dict):
                spread.update(payload["weights"])
            if isinstance(payload.get("total_weights"), dict):
                for k, v in payload["total_weights"].items():
                    if k in total:
                        total[k] = float(v)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    _zero = [name for name, w in spread.items() if float(w) == 0.0]
    if _zero:
        log.warning(
            "[CONFIG] Zero-weight spread models (effectively disabled): %s. "
            "These models still execute each prediction cycle. "
            "Set weight to None or remove from MODELS list to skip execution.",
            ", ".join(_zero),
        )

    return {"spread": spread, "total": total}
