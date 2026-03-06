# Git `--ignore-missing` Inventory and Fix

## Root Cause

`git add --ignore-missing` is only valid when paired with `--dry-run`.  
Without `--dry-run`, git exits with code 128:

```
fatal: the option '--ignore-missing' requires '--dry-run'
```

The intended behavior in every occurrence was: **stage each file if it exists; silently skip it if it is absent.**  
The correct fix is a per-file shell existence check (`[[ -e "$_f" ]]`) before calling `git add -- "$_f"`.

---

## Occurrences Found and Fixed

| File | Original line | Invalid command | Fix applied |
|---|---|---|---|
| `.github/workflows/cbb_perplexity_models.yml` | 378 | `git add --ignore-missing -- data/...` | Per-file loop with `[[ -e ]]` guard + diagnostics |
| `.github/workflows/update_espn_cbb.yml` | 243 | `git add --ignore-missing -- data/...` | Per-file loop with `[[ -e ]]` guard |
| `.github/workflows/update_espn_cbb.yml` | 398 | `git add --ignore-missing -- data/...` | Per-file loop with `[[ -e ]]` guard |
| `.github/workflows/cbb_analytics.yml` | 765 | `git add --ignore-missing data/...` | Per-file loop with `[[ -e ]]` guard |
| `.github/workflows/cbb_predictions_rolling.yml` | 533 | `git add --ignore-missing data/...` | Per-file loop with `[[ -e ]]` guard |
| `.github/workflows/cbb_pipeline.yml` | 267 | `git add --ignore-missing data/...` | Per-file loop with `[[ -e ]]` guard |
| `.github/workflows/cbb_pipeline.yml` | 368 | `git add --ignore-missing data/...` | Per-file loop with `[[ -e ]]` guard |

---

## Fix Pattern Applied

```bash
for _f in \
  path/to/file1 \
  path/to/file2; do
  if [[ -e "$_f" ]]; then
    echo "[INFO] staging: $_f"
    git add -- "$_f"
  else
    echo "[SKIP] missing, not staging: $_f"
  fi
done
```

For `cbb_perplexity_models.yml` (primary failing workflow), additional diagnostics were added immediately before the loop:

```bash
echo "[INFO] staging diagnostics: pwd=$(pwd)"
echo "[INFO] git status --short:"
git status --short || true
echo "[INFO] target files for staging:"
# ... then the per-file loop with git ls-files --error-unmatch check ...
```

---

## Intended Behavior Preserved

- Files that exist and are tracked → staged with `git add -- <file>`
- Files that exist but are new (untracked) → staged with `git add -- <file>`
- Files that are absent → logged and skipped (no error)
- The `git diff --staged --quiet` check after the loop still short-circuits if nothing changed
- All subsequent commit/push logic is unchanged

---

## Verification

To verify locally that the fix is correct, run:

```bash
# Confirm the flag is gone
grep -rn "ignore-missing" .github/workflows/

# Confirm syntax is valid YAML
python -c "import yaml; yaml.safe_load(open('.github/workflows/cbb_perplexity_models.yml'))" && echo OK

# Simulate the staging loop (dry run)
for _f in data/predictions_joint_latest.csv data/predictions_joint_snapshots.csv; do
  if [[ -e "$_f" ]]; then echo "[INFO] would stage: $_f"; else echo "[SKIP] missing: $_f"; fi
done
```
