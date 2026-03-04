# CSV Quality Audit v2

- Files audited: 111
- Violations: 23

## Top Violations

file_path,read_ok,read_error,row_count,col_count,all_null_cols,all_null_col_pct,rolling_col_count,rolling_all_null_col_count,rolling_broken_flag,expected_min_rows,below_expected_min_rows,market_games_last_7d
data/cbb_rankings_20260222.csv,True,,349,58,12,20.69,2,0,False,,False,0
data/cbb_rankings_20260223.csv,True,,338,58,12,20.69,2,0,False,,False,0
data/clv_by_submodel.csv,True,,8,7,1,14.29,0,0,False,,False,0
data/clv_report.csv,True,,60,14,2,14.29,0,0,False,,False,0
data/edge_history.csv,True,,7,13,5,38.46,0,0,False,,False,0
data/market_lines.csv,True,,30,49,19,38.78,0,0,False,,False,1
data/market_lines_closing.csv,True,,11,52,22,42.31,0,0,False,,False,1
data/market_lines_latest.csv,True,,11,49,21,42.86,0,0,False,,False,1
data/market_lines_snapshots.csv,True,,11,49,24,48.98,0,0,False,,False,1
data/matchup_preview.csv,True,,143,16,2,12.5,0,0,False,,False,0
data/predictions_20260223.csv,True,,9,106,26,24.53,4,4,True,,False,0
data/predictions_20260224.csv,True,,28,106,25,23.58,4,4,True,,False,0
data/predictions_20260225.csv,True,,8,106,29,27.36,4,4,True,,False,0
data/predictions_combined_latest.csv,True,,0,164,164,100.0,4,4,True,,False,0
data/predictions_graded.csv,True,,36,275,80,29.09,10,0,False,,False,0
data/predictions_mc_latest.csv,True,,221,186,50,26.88,4,0,False,,False,0
data/predictions_with_context.csv,True,,36,255,73,28.63,10,0,False,,False,0
data/results_alerts.csv,True,,0,1,1,100.0,0,0,False,,False,0
data/results_log.csv,True,,286,82,18,21.95,0,0,False,,False,0
data/results_log_by_team.csv,True,,652,87,18,20.69,0,0,False,,False,0
