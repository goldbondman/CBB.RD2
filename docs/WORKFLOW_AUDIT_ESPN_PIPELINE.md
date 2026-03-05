# Workflow Audit: Update ESPN CBB Data

## Workflow Path
- `.github/workflows/update_espn_cbb.yml`

## Scripts and Expected Outputs
- `espn_pipeline.py` -> `data/games.csv`, `data/team_game_weighted.csv`, `data/team_pretournament_snapshot.csv` and related core artifacts.
- `cbb_rotation_features.py` -> `data/rotation_features.csv`.
- `cbb_situational_features.py` -> `data/situational_features.csv`.
- `cbb_luck_regression_features.py` -> `data/luck_regression_features.csv`.
- `player_availability_features.py` -> `data/player_availability_features.csv`.
- `cbb_season_summaries.py` -> `data/player_season_summary.csv`, `data/team_season_summary.csv`, and others.
- `espn_rankings.py` -> `data/cbb_rankings.csv`, `data/cbb_rankings_by_conference.csv`.

## Existing Policy/Tooling Discovery
- Integrity/freshness gates exist but are not wired here:
  - `pipeline/integrity.py`
  - `scripts/verify_pipeline_integrity.py`
  - `scripts/ci_data_quality_gate.py`
  - `.github/scripts/validate_workflows.py`
- `data/*.csv` is intentionally tracked by git (`.gitignore` allows `data/*.csv`), so commit policy must be controlled to prevent bloat.

## Issues and Risks

### Correctness
1. **Input default ambiguity on schedule runs**
- File: `.github/workflows/update_espn_cbb.yml:80,83`
- `github.event.inputs.days_back` is not present for scheduled events; expression fallback exists but is not validated as numeric/range-safe.

2. **Pipeline output flag can become false-green**
- File: `.github/workflows/update_espn_cbb.yml:77-83`
- `success=true` is always written after command with no explicit contract guard besides shell default behavior; no robust stage manifest confirms required outputs were actually produced.

3. **Only one output is validated after multiple producers**
- File: `.github/workflows/update_espn_cbb.yml:95-109`
- Only `team_game_weighted.csv` row count is checked, while critical outputs (`games.csv`, `team_pretournament_snapshot.csv`) are not validated here.

### Dependency/Order
4. **Feature builders execute unguarded**
- File: `.github/workflows/update_espn_cbb.yml:86-93`
- Scripts run even if upstream files or required columns are missing/stale; dependencies are implied, not enforced.

5. **Rankings dependency is weakly coupled to upstream job artifact**
- File: `.github/workflows/update_espn_cbb.yml:168-175`
- Artifact download allows failure via `continue-on-error: true`; rankings may run against stale checkout files.

6. **Rankings gate lacks freshness/provenance checks**
- File: `.github/workflows/update_espn_cbb.yml:177-189`
- Existence/row checks do not ensure files are from the current run, risking stale inputs.

7. **No dynamic DAG/contract execution path**
- File: `.github/workflows/update_espn_cbb.yml` (overall)
- Step order is hardcoded rather than derived from explicit produced/required artifact contracts.

### Reliability
8. **Global concurrency group without cancellation can backlog updates**
- File: `.github/workflows/update_espn_cbb.yml:30-33`
- `cancel-in-progress: false` may queue stale runs and delay fresh data publication.

9. **Commits run even after failures**
- File: `.github/workflows/update_espn_cbb.yml:121-145,231-255`
- `if: always()` on commit steps can persist partial or invalid outputs.

10. **Whole `data/` artifact upload is over-broad**
- File: `.github/workflows/update_espn_cbb.yml:112-118`
- Can upload unnecessary large files and historical archives, increasing run time/cost and failure risk.

### Determinism
11. **Determinism controls are missing**
- File: `.github/workflows/update_espn_cbb.yml` (job/global env)
- No explicit `TZ=UTC`, `PYTHONHASHSEED=0`, locale normalization.

12. **Dependency resolver not locked**
- File: `.github/workflows/update_espn_cbb.yml:70-74,192-196`
- `pip install -r requirements.txt` only; no constraints lock path, increasing reproducibility drift.

### Security/ToS
13. **Permissions are broader than necessary**
- File: `.github/workflows/update_espn_cbb.yml:34-37,44-47,152-155`
- `actions: write` and broad write permissions increase blast radius.

14. **Unsafe automated conflict resolution in git push loop**
- File: `.github/workflows/update_espn_cbb.yml:134-139,244-249`
- Uses `merge -X ours` and `git reset --hard` in CI, which can silently discard upstream changes.

### Repo Hygiene
15. **Unbounded `git add data/` risks repository bloat**
- File: `.github/workflows/update_espn_cbb.yml:126,236`
- Stages all files under `data/`, including potentially large snapshots/archives.

16. **No staged-size guard before commit**
- File: `.github/workflows/update_espn_cbb.yml:121-145,231-255`
- Large accidental updates can be pushed without volume checks.

17. **No explicit run manifest contract between jobs**
- File: `.github/workflows/update_espn_cbb.yml` (overall)
- No artifact manifest with row counts/timestamps, making dependency freshness and traceability weak.

18. **Workflow health validator is unconditional and late**
- File: `.github/workflows/update_espn_cbb.yml:257-260`
- Should run conditionally if present and ideally earlier as a fast-fail sanity check.

## Audit Summary
The workflow currently works in best-case runs but lacks explicit dependency contracts, strict freshness validation, deterministic runtime guarantees, and safe commit boundaries. It should be refactored into staged, dependency-aware execution with manifest handoff, strict artifact gating, and manual-only bounded data commits.
