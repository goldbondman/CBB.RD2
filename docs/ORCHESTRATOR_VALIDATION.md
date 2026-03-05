# Orchestrator Validation Notes

## Local Smoke Command

```bash
py -3 -m model_lab.orchestrator run --run-id orch_ci_smoke --profile ci --limit 200
```

## Expected Behavior

- Writes `data/model_lab_runs/<run_id>/orchestrator/orchestrator_manifest.json`.
- Writes `data/model_lab_runs/<run_id>/orchestrator/ORCHESTRATOR_EXEC_SUMMARY.md`.
- Writes per-module manifests under `data/model_lab_runs/<run_id>/orchestrator/modules/`.
- Marks modules with missing dependency artifacts as `SKIPPED` and records exact missing requirements.
- Returns non-zero in `ci` profile only when a selected `FAST` module fails.
