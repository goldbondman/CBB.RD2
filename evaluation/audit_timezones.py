import pandas as pd, pathlib, re

issues = []

py_files = list(pathlib.Path('.').glob('*.py'))
for py_file in py_files:
    source = py_file.read_text(errors='ignore')
    lines  = source.split('\n')

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()

        # Naive datetime comparison (no timezone)
        if re.search(r'datetime\.now\(\)', line_stripped) and 'utc' not in line_stripped.lower():
            issues.append(f"{py_file}:{i} — datetime.now() without timezone: {line_stripped[:80]}")

        # Mixing aware and naive datetimes
        if 'pd.Timestamp.now()' in line_stripped and 'tz' not in line_stripped:
            issues.append(f"{py_file}:{i} — pd.Timestamp.now() without tz: {line_stripped[:80]}")

        # days_back computation without UTC
        if 'days_back' in line_stripped and 'timedelta' in line_stripped and 'utc' not in line_stripped.lower():
            issues.append(f"{py_file}:{i} — days_back without UTC anchor: {line_stripped[:80]}")

        # to_datetime without utc=True
        if 'to_datetime' in line_stripped and 'utc' not in line_stripped and 'datetime' in line_stripped.lower():
            issues.append(f"{py_file}:{i} — to_datetime without utc=True: {line_stripped[:80]}")

if issues:
    print(f"[WARN] {len(issues)} potential timezone issues:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("[OK] No obvious timezone inconsistencies found")

# Check CSV datetime columns for mixed tz-aware/naive values
for path in pathlib.Path('data').rglob('*.csv'):
    try:
        df = pd.read_csv(path, nrows=100, low_memory=False)
        for col in df.columns:
            if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                sample = df[col].dropna().head(5).tolist()
                has_z   = any('Z' in str(v) or '+00' in str(v) for v in sample)
                has_pst = any('PST' in str(v) or 'PDT' in str(v) for v in sample)
                if has_z and has_pst:
                    print(f"[WARN] {path}:{col} — mixed UTC and PST values in same column")
    except:
        pass
