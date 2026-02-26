# FORENSIC ANSWERS (from repo audit; do not skip)
# WORKFLOW → SCRIPT ORDER (from /tmp/audit_results.txt):
# - market_lines.yml runs ingestion.market_lines then enrichment.predictions_with_context.
# - cbb_analytics.yml runs espn_pipeline.py, cbb_results_tracker.py, cbb_backtester.py, build_backtest_csvs.py.
# - update_espn_cbb.yml runs espn_pipeline.py, field_reconciler.py, espn_rankings.py.
# - cbb_predictions_rolling.yml runs espn_pipeline.py fallback, cbb_prediction_model.py flow, cbb_ensemble.py, cbb_monte_carlo.py.
#
# SCRIPT WRITE/READ DEPENDENCIES (from /tmp/audit_results.txt + repo grep):
# - predictions_latest.csv is written in espn_prediction_runner.py and read by tracker/market workflows.
# - market_lines.csv is produced by ingestion.market_lines and used by enrichment/predictions_with_context.py.
# - Schema-managed outputs include predictions_latest.csv, predictions_combined_latest.csv, results_log*.csv, games.csv.
#
# ARTIFACT CHAIN FINDINGS:
# - Current named artifacts in audit are linked: espn-cbb-csvs, cbb-results-log, cbb-predictions-rolling-latest.
# - Workflows still have brittle cross-run artifact assumptions; enforce same-run guarantees + fallback generation.
#
# COLUMN CONTRACT FINDINGS:
# - Both event_id and game_id are used across workflows/scripts.
# - Both pred_spread and predicted_spread appear; many prediction columns are fully-null in data/*.csv snapshots.
# - conference/conf_id exists; translation to conference_name is inconsistently guaranteed.
#
# DTYPE/NULL FINDINGS:
# - Multiple prediction outputs show 25+ fully-null columns.
# - Risk of object dtypes in numeric operations and merges causing null propagation.
#
# GIT RACE FINDINGS:
# - All 4 workflows use rebase/push patterns; market_lines.yml has no concurrency group.
# - rebase stash patterns appear and must be removed.
#
# SILENT FAILURE FINDINGS:
# - Several scripts have except->continue/pass behavior (espn_prediction_runner.py, cbb_ensemble.py, cbb_prediction_model.py, etc.).
#
# ROOT CAUSE CATEGORIES FOUND:
# CLASS A — Artifact contract hardening needed (cross-run download assumptions, missing explicit pre-check gates).
# CLASS B — Data write path failures (allow_empty=True + weak post-write assertions).
# CLASS C — Column name contract mismatches (event_id/game_id, pred_spread/predicted_spread).
# CLASS D — Dtype/null propagation failures (object numerics, null-heavy joins, fragile aggregation).
# CLASS E — Git commit race conditions (missing concurrency in market_lines, aggressive rebase flows).
# CLASS F — Missing source bootstrap risk (market_lines.csv placeholder/absence before enrichment).
# CLASS G — Conference ID translation inconsistency (numeric conference identifiers leak through).
# CLASS H — Schema validation misalignment risk (required column drift vs runtime outputs).

## Context
The CBB pipeline intermittently succeeds but produces unstable outputs (null-heavy predictions, fragile artifact handoffs, and race-prone workflow commits). Forensics show mixed column contracts (`event_id` vs `game_id`, `pred_spread` vs `predicted_spread`), weak write assertions, and dtype/merge drift that can silently degrade model outputs. Fixes must be applied in dependency order with explicit verification gates after every atomic change.

## Pre-flight: capture baseline
1) Read every file before editing:
```bash
cat .github/workflows/update_espn_cbb.yml
cat .github/workflows/cbb_predictions_rolling.yml
cat .github/workflows/cbb_analytics.yml
cat .github/workflows/market_lines.yml
cat pipeline_csv_utils.py
cat config/schemas.py
cat cbb_output_schemas.py
cat espn_prediction_runner.py
cat cbb_prediction_model.py
cat cbb_ensemble.py
cat cbb_results_tracker.py
cat ingestion/market_lines.py
cat enrichment/predictions_with_context.py
cat espn_config.py
```
2) Capture baseline null rates and dtype snapshots for all existing output CSVs:
```bash
python - <<'PY'
import pandas as pd, pathlib, json, sys
out = {}
for p in sorted(pathlib.Path('data').glob('*.csv')):
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        out[p.name] = {'error': str(e)}
        continue
    null_rate = (df.isna().mean().sort_values(ascending=False).head(100)).to_dict() if len(df.columns) else {}
    dtypes = {c:str(t) for c,t in df.dtypes.items()}
    out[p.name] = {'rows': int(len(df)), 'cols': int(len(df.columns)), 'null_rate': null_rate, 'dtypes': dtypes}
pathlib.Path('data/audit_baseline.json').write_text(json.dumps(out, indent=2))
print('wrote data/audit_baseline.json')
PY
```
3) Preserve forensic snapshot:
```bash
python /tmp/dependency_audit.py > data/dependency_audit_baseline.txt
```

## Fix 1: [Class A] Artifact uploads
Add/normalize artifact upload steps in each producer workflow so every consumed artifact has a deterministic producer in the same run chain.
- File: `.github/workflows/update_espn_cbb.yml` (job `update`) ensure `espn-cbb-csvs` upload includes all required downstream CSVs.
- File: `.github/workflows/cbb_predictions_rolling.yml` (job `predict`) ensure `cbb-predictions-rolling-latest` upload includes `predictions_latest.csv`, `predictions_combined_latest.csv`, and `ensemble_predictions_latest.csv`.
- File: `.github/workflows/cbb_analytics.yml` ensure `cbb-results-log` upload is unconditional with `if: always()` and explicit path checks.
Verify:
```bash
python - <<'PY'
import pathlib, re, sys
wfs = list(pathlib.Path('.github/workflows').glob('*.yml'))
t = '\n'.join([p.read_text() for p in wfs])
required = ['name: espn-cbb-csvs','name: cbb-predictions-rolling-latest','name: cbb-results-log']
missing=[r for r in required if r not in t]
if missing:
    print('Missing uploads:', missing)
    sys.exit(1)
print('Artifact upload names present.')
PY
```

## Fix 2: [Class A] Artifact download consolidation
Remove or gate download steps that assume unavailable run contexts; add fallback generation immediately in same job if download fails.
- Exact files: `.github/workflows/market_lines.yml`, `.github/workflows/cbb_analytics.yml`, `.github/workflows/cbb_predictions_rolling.yml`, `.github/workflows/update_espn_cbb.yml`.
- Rule: each `download-artifact` must be followed by a check+fallback generation path.
Verify:
```bash
python /tmp/dependency_audit.py | tee /tmp/audit_after_artifacts.txt
python - <<'PY'
import pathlib, sys
txt = pathlib.Path('/tmp/audit_after_artifacts.txt').read_text()
if 'BROKEN — NEVER UPLOADED' in txt:
    print(txt)
    sys.exit(1)
print('No broken artifact download names detected.')
PY
```

## Fix 3: [Class B] Write path — predictions_latest
Harden `espn_prediction_runner.py` write path:
- Require non-empty rows for scheduled/live prediction windows unless explicitly `allow_empty` with reason logging.
- After `safe_write_csv(...predictions_latest...)`, re-read file and assert expected key columns (`event_id`, `game_id`, `pred_spread`) exist.
- On failure, raise and `sys.exit(1)` in caller workflow.
Verify:
```bash
python espn_prediction_runner.py || exit 1
python - <<'PY'
import pandas as pd, sys
p='data/predictions_latest.csv'
df=pd.read_csv(p)
req=['event_id','game_id','pred_spread']
missing=[c for c in req if c not in df.columns]
if len(df)==0 or missing:
    print('predictions_latest invalid', len(df), missing)
    sys.exit(1)
print('predictions_latest valid:', len(df))
PY
```

## Fix 4: [Class B] Write path — market_lines
In `ingestion/market_lines.py` and `enrichment/predictions_with_context.py`, fail loudly if `market_lines.csv` is absent/empty when mode requires market enrichment.
- Add explicit `Path.exists`, min-size, min-row assertions.
- Log pulled/inserted/rejected counts and write DQ issue rows to CSV audit output.
Verify:
```bash
python -m ingestion.market_lines --mode rolling || exit 1
python -m enrichment.predictions_with_context || exit 1
python - <<'PY'
import pandas as pd, sys
ml=pd.read_csv('data/market_lines.csv')
if ml.empty:
    print('market_lines empty')
    sys.exit(1)
print('market_lines rows',len(ml))
PY
```

## Fix 5: [Class C] Column name normalization
Create one canonical column normalizer (shared util) and apply at every pipeline boundary.
- Canonical map must include: `event_id <- {event_id, game_id}`, `pred_spread <- {pred_spread, predicted_spread}`, and any audit-found aliases.
- Update schema checks in `config/schemas.py` and `cbb_output_schemas.py` to accept aliases only via canonicalization step (not ad hoc per script).
Verify:
```bash
python - <<'PY'
import pandas as pd, sys
for p in ['data/predictions_latest.csv','data/predictions_combined_latest.csv','data/results_log.csv']:
    df=pd.read_csv(p)
    for c in ['event_id','pred_spread']:
        if c not in df.columns and 'predictions' in p:
            print(p,'missing',c); sys.exit(1)
print('Column normalization checks passed.')
PY
```

## Fix 6: [Class D] Dtype normalization
Add `normalize_numeric_dtypes(df, numeric_cols)` helper and call it at each entry point doing math/joins in:
- `cbb_prediction_model.py`
- `cbb_ensemble.py`
- `espn_prediction_runner.py`
- `cbb_results_tracker.py`
Use `pd.to_numeric(errors='coerce')` and explicit nullable numeric dtypes.
Verify:
```bash
python - <<'PY'
import pandas as pd, sys
for p in ['data/predictions_latest.csv','data/predictions_combined_latest.csv']:
    df=pd.read_csv(p)
    for c in ['pred_spread','pred_total']:
        if c in df.columns and str(df[c].dtype)=='object':
            print(p,c,'still object'); sys.exit(1)
print('Dtype normalization checks passed.')
PY
```

## Fix 7: [Class D] Wins/losses aggregation
In `cbb_results_tracker.py` and any season-summary builders, ensure wins/losses derive from finalized game outcomes and not null/placeholder columns.
- Add deterministic aggregation source selection and explicit fallback hierarchy.
- Assert `wins + losses > 0` for active teams in-season windows.
Verify:
```bash
python cbb_results_tracker.py --dry-run || exit 1
python - <<'PY'
import pandas as pd, sys
p='data/team_season_summary.csv'
df=pd.read_csv(p)
if 'wins' not in df.columns or 'losses' not in df.columns:
    print('wins/losses missing'); sys.exit(1)
ok=((pd.to_numeric(df['wins'],errors='coerce').fillna(0)+pd.to_numeric(df['losses'],errors='coerce').fillna(0))>0).mean()
if ok < 0.80:
    print('wins/losses coverage too low',ok); sys.exit(1)
print('wins/losses coverage',ok)
PY
```

## Fix 8: [Class E] Git race condition
Workflow hardening in all 4 workflow YAML files:
- Add/standardize `concurrency:` blocks (including `market_lines.yml`).
- Remove `rebase stash` and fragile merge conflict loops.
- Replace with deterministic retry push loop:
```bash
for i in $(seq 1 5); do
  git pull --rebase origin main || true
  git push origin main && break
  sleep $((i*3))
done
```
Verify:
```bash
python - <<'PY'
import pathlib, sys
for f in pathlib.Path('.github/workflows').glob('*.yml'):
    t=f.read_text()
    if 'concurrency:' not in t:
        print('missing concurrency',f); sys.exit(1)
print('Workflow concurrency present and stash-rebase-pattern removed.')
PY
```

## Fix 9: [Class F] market_lines.csv bootstrap
Before enrichment, guarantee bootstrap creation from available odds source with provenance columns.
- Update `ingestion/market_lines.py` to output required schema even on partial data.
- Ensure provenance columns: `source`, `pulled_at_utc`, `verification_status`, `verification_notes`.
Verify:
```bash
python -m ingestion.market_lines --mode rolling || exit 1
python - <<'PY'
import pandas as pd, sys
m=pd.read_csv('data/market_lines.csv')
req=['event_id','pulled_at_utc','verification_status']
miss=[c for c in req if c not in m.columns]
if miss or len(m)==0:
    print('market_lines bootstrap failed',miss,len(m)); sys.exit(1)
if (m['verification_status'].astype(str)=='verified').sum()==0:
    print('no verified lines'); sys.exit(1)
print('market_lines bootstrap ok')
PY
```

## Fix 10: [Class G] Conference name translation
Ensure conference IDs are translated to human-readable `conference_name` everywhere outputs expose conference fields.
- Apply in `espn_prediction_runner.py`, `espn_rankings.py`, and any output assembly in `cbb_prediction_model.py`.
Verify:
```bash
python - <<'PY'
import pandas as pd, sys, re
for p in ['data/predictions_latest.csv','data/cbb_rankings.csv']:
    df=pd.read_csv(p)
    if 'conference_name' in df.columns:
        bad=df['conference_name'].astype(str).str.fullmatch(r'\d+').fillna(False).mean()
        if bad>0:
            print(p,'numeric conference_name ratio',bad); sys.exit(1)
print('conference_name translation checks passed.')
PY
```

## Fix 11: [Class H] Schema validation alignment
Align required schema contracts with canonical runner outputs post-normalization.
- Update `config/schemas.py` and `cbb_output_schemas.py` to enforce required canonical fields and reject drift.
- Remove permissive behavior that masks data loss without recording DQ reason.
Verify:
```bash
python -m py_compile config/schemas.py cbb_output_schemas.py pipeline_csv_utils.py || exit 1
python check_schema_drift.py || exit 1
```

## Post-fix verification suite
Create `scripts/verify_pipeline_integrity.py` that exits 0/1 and checks:
- artifact contract integrity
- required CSV presence and non-empty row counts
- canonical column presence
- null-rate non-regression vs `data/audit_baseline.json`
- dtype sanity for numeric model fields
- conference_name non-numeric
- market enrichment coverage (`market_evaluated > 0` if column exists)
- no bare `except:` in touched scripts unless explicitly logged+re-raised
```bash
python scripts/verify_pipeline_integrity.py
```

## Commit
```bash
git add .github/workflows/update_espn_cbb.yml \
        .github/workflows/cbb_predictions_rolling.yml \
        .github/workflows/cbb_analytics.yml \
        .github/workflows/market_lines.yml \
        pipeline_csv_utils.py config/schemas.py cbb_output_schemas.py \
        espn_prediction_runner.py cbb_prediction_model.py cbb_ensemble.py \
        cbb_results_tracker.py ingestion/market_lines.py \
        enrichment/predictions_with_context.py espn_rankings.py \
        scripts/verify_pipeline_integrity.py

git commit -m "pipeline: fix classes A-H (artifacts, write paths, column contracts, dtype/null, git race, source bootstrap, conference translation, schema alignment)"

for i in $(seq 1 5); do
  git push origin HEAD && break
  sleep $((i*3))
done
```

## Success criteria
```bash
python - <<'PY'
import subprocess, sys
checks = [
    "python /tmp/dependency_audit.py > /tmp/final_audit.txt",
    "python scripts/verify_pipeline_integrity.py",
    "python check_schema_drift.py",
]
for cmd in checks:
    p = subprocess.run(cmd, shell=True)
            if p.returncode != 0:
            print('FAIL', cmd); sys.exit(1)
print('PASS all binary checks')
PY
```

# marker: uses actions/upload-artifact@v4
# marker: for i in seq 1 5; do
