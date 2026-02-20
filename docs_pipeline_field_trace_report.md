# Pipeline Field Trace Report

## Scope
Trace for required fields through:
INGESTION → NORMALIZATION → MERGE → SCHEMA ENFORCEMENT → FINAL WRITE.

### Required player fields
FGA, FGM, FTA, FTM, TPA, TPM, ORB, DRB, RB, TO, AST

### Required team fields
wins, losses, conference

## A) FIELD TRACE TABLE

| Field | First Seen In | Last Seen In | Drop Location | Root Cause |
|---|---|---|---|---|
| FGA | `espn_parsers.py::parse_summary` player row `fga` / team row `fga` | `team_game_weighted.csv` load path supports `fga` (`espn_prediction_runner.py::load_team_game_data`) | `espn_prediction_runner.py::generate_predictions` row dict omits `fga` | Final prediction writer uses explicit column allowlist (`row` dict), so raw box columns are excluded. |
| FGM | `parse_summary` → `fgm` | same as above | same | same |
| FTA | `parse_summary` → `fta` | same as above | same | same |
| FTM | `parse_summary` → `ftm` | same as above | same | same |
| TPA | `parse_summary` → `tpa` | same as above | same | same |
| TPM | `parse_summary` → `tpm` | same as above | same | same |
| ORB | `parse_summary` → `orb` | same as above | same | same |
| DRB | `parse_summary` → `drb` | same as above | same | same |
| AST | `parse_summary` → `ast` | `team_game_weighted` path keeps `ast` metrics; player paths keep `ast` too | `espn_prediction_runner.py::generate_predictions` row dict omits `ast` | Explicit prediction output schema excludes raw/player counting stats. |
| RB | Not created as `rb`; normalized as `reb` in parser maps | `reb` persists in player/team logs | `espn_parsers.py::PLAYER_STAT_MAP` + `summary_to_team_rows` use `reb`, never `rb` | Naming mismatch (`RB` expected downstream, but canonical name is `reb`). |
| TO | Not created as `to`; normalized as `tov` in parser maps | `tov` persists in team/player outputs | `espn_parsers.py::PLAYER_STAT_MAP` + team stat col_map map `to`→`tov` | Naming mismatch (`TO` expected downstream, but canonical name is `tov`). |
| wins | `espn_parsers.py::_team_meta` + `summary_to_team_rows` | Team files (`team_game_logs/metrics/sos/weighted`) | `espn_prediction_runner.py::generate_predictions` row dict omits | Final prediction CSV schema excludes team record fields. |
| losses | `espn_parsers.py::_team_meta` + `summary_to_team_rows` | Team files (`team_game_logs/metrics/sos/weighted`) | `espn_prediction_runner.py::generate_predictions` row dict omits | Final prediction CSV schema excludes team record fields. |
| conference | `espn_parsers.py::_team_meta` + `summary_to_team_rows` | Team files loaded with `conference` as non-numeric | `espn_prediction_runner.py::generate_predictions` row dict omits | Final prediction CSV schema excludes conference/team metadata fields. |

## B) SCHEMA EXCLUSION

| CSV | Missing Fields | Schema Issue | Fix Needed |
|---|---|---|---|
| `predictions_*.csv` / `predictions_latest.csv` | All required player fields; wins/losses/conference | Manual output schema via `row = {...}` has no passthrough for those fields | Add explicit columns in `row` (team and/or home_/away_ prefixed versions) or write a separate enriched downstream CSV that joins matchup + latest team snapshot + selected box stats. |
| `player_game_logs.csv` / `player_game_metrics.csv` | `RB`, `TO` (uppercase names) | Empty-schema fallback and parser canonical names use `reb` and `tov`; no alias columns (`RB`, `TO`) | Decide canonical contract. If downstream requires uppercase legacy names, add alias columns (`RB=reb`, `TO=tov`) before write. |

## C) MERGE ISSUES

| File | Function | Operation | Impact |
|---|---|---|---|
| `espn_player_metrics.py` | `add_player_per_game_metrics` | `df.merge(team_poss, on=["event_id","team_id"], how="left")` then drop `_team_poss` | No required fields dropped; only helper column removed. |
| `espn_pipeline.py` | `_append_dedupe_write` | `pd.concat` + `drop_duplicates(subset=unique_keys)` | De-duplicates rows, but does not column-filter required fields. |
| `espn_tournament.py` | `build_pretournament_snapshot` | `groupby("team_id").last()` | Keeps one latest row per team; no explicit required-column filter, but not a player-level sink. |

## D) WRITE PATH ISSUES

| File | Write Call | Missing Fields | Fix |
|---|---|---|---|
| `espn_prediction_runner.py` | `write_predictions(): df.to_csv(...)` | Missing all requested fields because upstream `df` is built from restricted `row` dict | Expand `row` schema to include required fields (likely prefixed home/away team context), or enrich with latest `team_game_weighted` snapshot before writing. |
| `espn_pipeline.py` | Empty fallback writes for player artifacts | `RB`/`TO` aliases absent (only `reb`/`tov`) | Add alias columns in fallback schema and normal run path if downstream contract expects RB/TO exactly. |

## E) TOP 5 ROOT CAUSES (Ranked)

1. **Primary:** Final predictions output uses a hard-coded column schema (`row` dict) and never carries forward raw player/team metadata columns.  
2. **Naming mismatch:** Pipeline canonicalizes to lowercase `tov`/`reb`, while downstream expectation is uppercase `TO`/`RB`.  
3. **No centralized schema contract:** There is no `OUTPUT_FILE_SCHEMAS`-style single source of truth to enforce/validate required fields before write.  
4. **Semantic overwrite of record fields:** `derive_records()` recomputes wins/losses from results, replacing parser-provided records; values remain present but may differ from ingestion source semantics (entering-game vs source snapshot timing).  
5. **No pre-write required-field integrity gate:** Write paths do not assert required downstream columns, so omissions propagate silently.
