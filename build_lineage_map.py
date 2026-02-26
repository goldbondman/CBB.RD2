import ast
import json
import pathlib
import re
from collections import defaultdict

ROOT = pathlib.Path('.')
DATA_DIR = ROOT / 'data'


def unparse(node):
    try:
        return ast.unparse(node)
    except Exception:
        return ''


def build_constant_map(py_files):
    const_map = {}
    # Catch string literals and Path-based constructions like DATA_DIR / 'file.csv'
    pattern = re.compile(r'([A-Z][A-Z0-9_]*)\s*=\s*.*?[\"\']([^\"\']+\.csv)[\"\']')
    path_pattern = re.compile(r'([A-Z][A-Z0-9_]*)\s*=\s*.*?\/\s*[\"\']([^\"\']+\.csv)[\"\']')
    dir_pattern = re.compile(r'(?:DATA_DIR|data_dir|output_dir)\s*/\s*[\'"]([^\'\"]+\.csv)[\'"]')
    path_pattern = re.compile(r'([A-Z][A-Z0-9_]*)\s*=\s*(?:DATA_DIR|data_dir|output_dir)\s*/\s*[\"\']([^\"\']+\.csv)[\"\']')

    for py in py_files:
        text = py.read_text(errors='ignore')
        for name, csv in pattern.findall(text):
            if '/' not in csv:
                csv = f"data/{csv}"
            const_map[name] = csv
        for name, csv in path_pattern.findall(text):
            if '/' not in csv:
                csv = f"data/{csv}"
            const_map[name] = csv
        for m in dir_pattern.finditer(text):
            const_map[m.group(0)] = f"data/{m.group(1)}"
            const_map[m.group(1)] = f"data/{m.group(1)}"
    return const_map


def resolve_csv_arg(arg_node, const_map):
    if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
        return const_map.get(arg_node.value, arg_node.value)
    if isinstance(arg_node, ast.Name):
        return const_map.get(arg_node.id, arg_node.id)
    res = unparse(arg_node)
    return const_map.get(res, res)


def extract_df_columns_from_expr(expr):
    cols = set()
    for n in ast.walk(expr):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            cols.add(n.slice.value)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'get':
            if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                cols.add(n.args[0].value)
    return cols


def collect_script_lineage(path, const_map):
    text = path.read_text(errors='ignore')
    tree = ast.parse(text)

    assignments_by_df = defaultdict(lambda: defaultdict(list))
    read_csv_calls = []
    to_csv_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'read_csv' and node.args:
                read_csv_calls.append(resolve_csv_arg(node.args[0], const_map))

            if node.func.attr == 'to_csv' and node.args:
                df_var = unparse(node.func.value)
                to_csv_calls.append((df_var, resolve_csv_arg(node.args[0], const_map), getattr(node, 'lineno', None)))

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                        df_var = target.value.id
                        col = target.slice.value
                        assignments_by_df[df_var][col].append({
                            'line': getattr(node, 'lineno', None),
                            'formula': unparse(node.value)[:240],
                            'source_columns': sorted(extract_df_columns_from_expr(node.value)),
                        })

    return {
        'assignments_by_df': assignments_by_df,
        'read_csv_calls': read_csv_calls,
        'to_csv_calls': to_csv_calls,
    }


def load_existing_csv_headers():
    csv_headers = {}
    if not DATA_DIR.exists():
        return csv_headers
    csv_paths = list(DATA_DIR.rglob('*.csv'))
    csv_paths = list({p.resolve(): p for p in csv_paths}.values())

    for csv in csv_paths:

    csv_paths = list(DATA_DIR.rglob('*.csv'))
    csv_paths = list({p.resolve(): p for p in csv_paths}.values())

    for csv in csv_paths:
        try:
            with csv.open('r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip()
            if header:
                csv_headers[str(csv.relative_to(ROOT))] = [c.strip() for c in header.split(',') if c.strip()]
        except Exception:
            pass
    return csv_headers


def main():
    py_files = sorted(ROOT.rglob('*.py'))
    const_map = build_constant_map(py_files)

    lineage = {}
    global_columns = defaultdict(list)
    script_data = {str(f): collect_script_lineage(f, const_map) for f in py_files}

    for script, info in script_data.items():
        for src in info['read_csv_calls']:
            lineage.setdefault(src, {}).setdefault('readers', []).append(script)

        for df_var, out_csv, line_no in info['to_csv_calls']:
            entry = lineage.setdefault(out_csv, {})
            entry.setdefault('writers', []).append({'script': script, 'dataframe_var': df_var, 'line': line_no})
            df_assigns = info['assignments_by_df'].get(df_var, {})
            if df_assigns:
                entry.setdefault('column_lineage', {}).update(df_assigns)

        for df_var, cmap in info['assignments_by_df'].items():
            for col, traces in cmap.items():
                for t in traces:
                    global_columns[col].append({'script': script, 'dataframe_var': df_var, **t})

    lineage['__columns__'] = global_columns

    csv_headers = load_existing_csv_headers()
    per_csv_orphans = {}
    all_orphans = set()

    for csv_path, cols in csv_headers.items():
        traced = set()
        # Use absolute/canonical path matching where possible
        if csv_path in lineage:
            traced |= set(lineage[csv_path].get('column_lineage', {}).keys())

        # Fallback check: require path identity if no direct string match
        for key, val in lineage.items():
            if key.startswith('__'):
                continue
            try:
                if pathlib.Path(key).resolve() == pathlib.Path(csv_path).resolve() and isinstance(val, dict):
                    traced |= set(val.get('column_lineage', {}).keys())
            except Exception:
                continue
        # Fallback to stem-match only if no direct path match, but restrict to same directory if known
        found_path = ROOT / csv_path
        for key, val in lineage.items():
            if key.startswith('__'):
                continue
            if isinstance(val, dict):
                key_path = pathlib.Path(str(key))
                try:
                    key_is_match = key_path.resolve() == pathlib.Path(found_path).resolve()
                except Exception:
                    key_is_match = False
                if not key_is_match:
                    continue
                traced |= set(val.get('column_lineage', {}).keys())

        missing = sorted(set(cols) - traced)
        if missing:
            per_csv_orphans[csv_path] = missing
            all_orphans.update(missing)

    lineage['__summary__'] = {
        'total_python_files_scanned': len(py_files),
        'total_csv_files_tracked': len([k for k in lineage if not k.startswith('__')]),
        'columns_with_any_lineage': len(global_columns),
        'csvs_detected_in_data_dir': len(csv_headers),
        'csvs_with_untraceable_columns': len(per_csv_orphans),
        'untraceable_columns': sorted(all_orphans),
        'untraceable_columns_by_csv': per_csv_orphans,
    }

    out = DATA_DIR / 'lineage_map.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lineage, indent=2, default=list))

    print('Lineage map built:')
    print(f"  Python files scanned: {lineage['__summary__']['total_python_files_scanned']}")
    print(f"  CSV files tracked: {lineage['__summary__']['total_csv_files_tracked']}")
    print(f"  Columns with traced lineage: {lineage['__summary__']['columns_with_any_lineage']}")
    print(f"  CSVs with untraceable columns: {lineage['__summary__']['csvs_with_untraceable_columns']}")
    print(f"  Total untraceable columns: {len(lineage['__summary__']['untraceable_columns'])}")
    if lineage['__summary__']['untraceable_columns']:
        print(f"  Sample untraceables: {lineage['__summary__']['untraceable_columns'][:20]}")


if __name__ == '__main__':
    main()
