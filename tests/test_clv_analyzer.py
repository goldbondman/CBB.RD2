import pandas as pd

from models.clv_analyzer import build_clv_reports


def test_build_clv_reports_uses_close_line_when_open_missing(tmp_path):
    acc_path = tmp_path / "model_accuracy_report.csv"
    pred_context_path = tmp_path / "predictions_with_context.csv"
    pred_fallback_path = tmp_path / "predictions_combined_latest.csv"
    out_report = tmp_path / "clv_report.csv"
    out_submodel = tmp_path / "clv_by_submodel.csv"

    # Empty accuracy/context inputs should not block close-line CLV reporting.
    pd.DataFrame(columns=["game_id", "actual_margin"]).to_csv(acc_path, index=False)
    pd.DataFrame(columns=["game_id"]).to_csv(pred_context_path, index=False)

    pd.DataFrame(
        {
            "game_id": ["1", "2", "3"],
            "spread_line": [-3.5, -1.0, None],
            "ens_ens_spread": [-2.5, -0.5, -4.0],
            "ens_fourfactors_spread": [-2.0, -0.2, -3.5],
        }
    ).to_csv(pred_fallback_path, index=False)

    # Use explicit fallback path by passing it as the primary context source.
    game_report, submodel_df = build_clv_reports(
        accuracy_path=acc_path,
        pred_context_path=pred_fallback_path,
        out_report=out_report,
        out_submodel=out_submodel,
    )

    assert len(game_report) == 2
    assert set(submodel_df["model_name"]) == {"ens_ens_spread", "ens_fourfactors_spread"}
    assert (submodel_df["n_games_with_clv"] == 2).all()
    # Fixed bug in clv_analyzer now uses spread_line as fallback for open, so it's no longer NaN
    assert (submodel_df["mean_clv_vs_open"].notna()).any()
