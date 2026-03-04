"""Feature cache layer."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache" / "features"


def _safe_token(value: object) -> str:
    text = str(value).strip()
    return text.replace("\\", "_").replace("/", "_").replace(":", "_")


def schema_hash(df: pd.DataFrame) -> str:
    payload = {
        "columns": list(df.columns),
        "dtypes": [str(dtype) for dtype in df.dtypes],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_paths(
    *,
    grain: str,
    feature_name: str,
    season_id: object,
    window_id: str = "raw",
    cache_root: Path = CACHE_ROOT,
) -> tuple[Path, Path]:
    season_token = _safe_token(season_id)
    window_token = _safe_token(window_id)
    dir_path = cache_root / grain / feature_name
    data_path = dir_path / f"season_{season_token}__window_{window_token}.csv"
    manifest_path = dir_path / f"season_{season_token}__window_{window_token}.manifest.json"
    return data_path, manifest_path


def load_cached_feature(
    *,
    grain: str,
    feature_name: str,
    season_id: object,
    version_hash: str,
    window_id: str = "raw",
    cache_root: Path = CACHE_ROOT,
) -> pd.DataFrame | None:
    data_path, manifest_path = cache_paths(
        grain=grain,
        feature_name=feature_name,
        season_id=season_id,
        window_id=window_id,
        cache_root=cache_root,
    )
    if not data_path.exists() or not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if manifest.get("version_hash") != version_hash:
        return None

    try:
        return pd.read_csv(data_path, low_memory=False)
    except Exception:
        return None


def save_cached_feature(
    *,
    df: pd.DataFrame,
    grain: str,
    feature_name: str,
    season_id: object,
    version_hash: str,
    key_fields: tuple[str, ...],
    window_id: str = "raw",
    cache_root: Path = CACHE_ROOT,
) -> None:
    data_path, manifest_path = cache_paths(
        grain=grain,
        feature_name=feature_name,
        season_id=season_id,
        window_id=window_id,
        cache_root=cache_root,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    manifest = {
        "feature_name": feature_name,
        "grain": grain,
        "season_id": season_id,
        "window_id": window_id,
        "version_hash": version_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "schema_hash": schema_hash(df),
        "key_fields": list(key_fields),
        "data_path": str(data_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
