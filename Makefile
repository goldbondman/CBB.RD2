.PHONY: all ingest features predict grade tune health

# Full daily pipeline (mirrors GitHub Actions order)
all: ingest features predict

ingest:
	python -m ingestion.espn_pipeline
	python -m ingestion.cbb_results_tracker

features:
	python -m features.espn_metrics
	python -m features.espn_weighted_metrics
	python -m features.espn_rankings
	python -m features.team_form_snapshot

predict:
	python -m models.espn_prediction_runner
	python -m enrichment.predictions_with_context
	python -m enrichment.edge_history --mode append

grade:
	python -m evaluation.predictions_graded
	python -m enrichment.edge_history --mode grade
	python -m evaluation.model_accuracy_weekly

tune:
	python -m evaluation.bias_detector
	python -m evaluation.optimize_weights
	python -m evaluation.calibrate_confidence

health:
	python -m tests.test_pipeline_health

# Usage: make predict, make grade, make tune
