from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def _add_game(
    rows: list[dict[str, object]],
    *,
    event_id: str,
    dt_utc: str,
    team_a: str,
    team_b: str,
    style_a: dict[str, float],
    style_b: dict[str, float],
    completed: bool = True,
) -> None:
    rows.append(
        {
            "event_id": event_id,
            "game_datetime_utc": dt_utc,
            "team_id": team_a,
            "team": team_a,
            "opponent_id": team_b,
            "opponent": team_b,
            "completed": completed,
            "pace": style_a["pace"],
            "efg_pct": style_a["efg_pct"],
            "three_par": style_a["three_par"],
            "orb_pct": style_a["orb_pct"],
            "tov_pct": style_a["tov_pct"],
            "ftr": style_a["ftr"],
        }
    )
    rows.append(
        {
            "event_id": event_id,
            "game_datetime_utc": dt_utc,
            "team_id": team_b,
            "team": team_b,
            "opponent_id": team_a,
            "opponent": team_a,
            "completed": completed,
            "pace": style_b["pace"],
            "efg_pct": style_b["efg_pct"],
            "three_par": style_b["three_par"],
            "orb_pct": style_b["orb_pct"],
            "tov_pct": style_b["tov_pct"],
            "ftr": style_b["ftr"],
        }
    )


def test_opponent_style_similarity_top_prior_match(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    style_t = {"pace": 67.0, "efg_pct": 0.50, "three_par": 0.33, "orb_pct": 0.28, "tov_pct": 0.17, "ftr": 0.25}
    style_o1 = {"pace": 71.0, "efg_pct": 0.54, "three_par": 0.36, "orb_pct": 0.31, "tov_pct": 0.15, "ftr": 0.29}
    style_o2 = {"pace": 61.0, "efg_pct": 0.45, "three_par": 0.21, "orb_pct": 0.22, "tov_pct": 0.23, "ftr": 0.18}
    style_o3 = {"pace": 70.5, "efg_pct": 0.53, "three_par": 0.35, "orb_pct": 0.30, "tov_pct": 0.16, "ftr": 0.28}
    style_x = {"pace": 65.0, "efg_pct": 0.48, "three_par": 0.30, "orb_pct": 0.27, "tov_pct": 0.19, "ftr": 0.22}

    rows: list[dict[str, object]] = []
    # Warm-up games so opponents have pregame style windows.
    _add_game(rows, event_id="w1", dt_utc="2026-01-01T00:00:00Z", team_a="O1", team_b="X", style_a=style_o1, style_b=style_x)
    _add_game(rows, event_id="w2", dt_utc="2026-01-02T00:00:00Z", team_a="O1", team_b="X", style_a=style_o1, style_b=style_x)
    _add_game(rows, event_id="w3", dt_utc="2026-01-03T00:00:00Z", team_a="O1", team_b="X", style_a=style_o1, style_b=style_x)
    _add_game(rows, event_id="w4", dt_utc="2026-01-01T00:00:00Z", team_a="O2", team_b="X", style_a=style_o2, style_b=style_x)
    _add_game(rows, event_id="w5", dt_utc="2026-01-02T00:00:00Z", team_a="O2", team_b="X", style_a=style_o2, style_b=style_x)
    _add_game(rows, event_id="w6", dt_utc="2026-01-03T00:00:00Z", team_a="O2", team_b="X", style_a=style_o2, style_b=style_x)
    _add_game(rows, event_id="w7", dt_utc="2026-01-01T00:00:00Z", team_a="O3", team_b="X", style_a=style_o3, style_b=style_x)
    _add_game(rows, event_id="w8", dt_utc="2026-01-02T00:00:00Z", team_a="O3", team_b="X", style_a=style_o3, style_b=style_x)
    _add_game(rows, event_id="w9", dt_utc="2026-01-03T00:00:00Z", team_a="O3", team_b="X", style_a=style_o3, style_b=style_x)
    _add_game(rows, event_id="w10", dt_utc="2026-01-01T00:00:00Z", team_a="T", team_b="X", style_a=style_t, style_b=style_x)
    _add_game(rows, event_id="w11", dt_utc="2026-01-02T00:00:00Z", team_a="T", team_b="X", style_a=style_t, style_b=style_x)
    _add_game(rows, event_id="w12", dt_utc="2026-01-03T00:00:00Z", team_a="T", team_b="X", style_a=style_t, style_b=style_x)

    # Target sequence for T: O1 then O2 then O3 (O3 is stylistically closest to O1).
    _add_game(rows, event_id="g1", dt_utc="2026-01-05T00:00:00Z", team_a="T", team_b="O1", style_a=style_t, style_b=style_o1)
    _add_game(rows, event_id="g2", dt_utc="2026-01-06T00:00:00Z", team_a="T", team_b="O2", style_a=style_t, style_b=style_o2)
    _add_game(rows, event_id="g3", dt_utc="2026-01-07T00:00:00Z", team_a="T", team_b="O3", style_a=style_t, style_b=style_o3)

    pd.DataFrame(rows).to_csv(data_dir / "team_game_weighted.csv", index=False)

    out_csv = data_dir / "matchups" / "opponent_style_similarity.csv"
    out_md = data_dir / "matchups" / "opponent_style_similarity_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "scripts/opponent_style_similarity.py",
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(out_csv),
            "--output-md",
            str(out_md),
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_csv.exists()
    assert out_md.exists()

    out = pd.read_csv(out_csv)
    target = out[(out["team_id"] == "T") & (out["event_id"] == "g3")].iloc[0]
    assert pd.notna(target["similarity_score"])
    assert str(target["top1_opponent_id"]) == "O1"
    assert "status: `OK`" in out_md.read_text(encoding="utf-8")
