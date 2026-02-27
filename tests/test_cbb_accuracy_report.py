import pandas as pd

import cbb_accuracy_report


def test_hydrate_model_spreads_backfills_from_predictions_graded(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    results = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "actual_margin": 8,
                "spread_line": -2.5,
                "m1_spread": None,
                "m2_spread": None,
                "m3_spread": None,
                "m4_spread": None,
                "m5_spread": None,
                "m6_spread": None,
                "m7_spread": None,
            }
        ]
    )
    results.to_csv(data_dir / "results_log.csv", index=False)

    graded = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "ens_fourfactors_spread": -1.0,
                "ens_adjefficiency_spread": -1.5,
                "ens_pythagorean_spread": -2.0,
                "ens_momentum_spread": -2.5,
                "ens_situational_spread": -3.0,
                "ens_cagerankings_spread": -3.5,
                "ens_regressedeff_spread": -4.0,
            }
        ]
    )
    graded.to_csv(data_dir / "predictions_graded.csv", index=False)

    monkeypatch.setattr(cbb_accuracy_report, "DATA_DIR", data_dir)
    monkeypatch.setattr(cbb_accuracy_report, "RESULTS_LOG", data_dir / "results_log.csv")
    monkeypatch.setattr(cbb_accuracy_report, "GRADED_PREDICTIONS", data_dir / "predictions_graded.csv")
    monkeypatch.setattr(cbb_accuracy_report, "OUT_PATH", data_dir / "cbb_accuracy_report.csv")

    cbb_accuracy_report.main()

    out = pd.read_csv(data_dir / "cbb_accuracy_report.csv")
    assert len(out) == 7
    assert (out["graded_games"] == 1).all()
    assert (out["verification_status"] == "verified").all()
    assert out["l14_ats_pct"].notna().all()
