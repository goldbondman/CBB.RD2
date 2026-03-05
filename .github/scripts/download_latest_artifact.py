#!/usr/bin/env python3
"""Download the latest successful workflow artifact from a specific branch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

API_BASE = "https://api.github.com"


class ArtifactDownloadError(RuntimeError):
    """Raised when artifact resolution or download cannot be completed safely."""


def _headers(token: str, accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "cbb-download-latest-artifact",
    }


def _api_json(path: str, token: str) -> dict[str, Any]:
    req = urllib.request.Request(f"{API_BASE}{path}", headers=_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ArtifactDownloadError(f"GitHub API error for {path}: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise ArtifactDownloadError(f"Network error calling GitHub API {path}: {exc}") from exc


def _download_binary(url: str, token: str) -> bytes:
    req = urllib.request.Request(url, headers=_headers(token, accept="application/octet-stream"))
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ArtifactDownloadError(f"Artifact download failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise ArtifactDownloadError(f"Network error downloading artifact: {exc}") from exc


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_dest = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = (destination / member.filename).resolve()
            if not str(member_path).startswith(str(resolved_dest)):
                raise ArtifactDownloadError(f"Unsafe path in artifact archive: {member.filename}")
        zf.extractall(destination)


def _find_artifact(
    repo: str,
    workflow_file: str,
    artifact_name: str,
    branch: str,
    max_runs: int,
    token: str,
) -> tuple[int, dict[str, Any], int]:
    workflow_ref = urllib.parse.quote(workflow_file, safe="")
    runs_path = (
        f"/repos/{repo}/actions/workflows/{workflow_ref}/runs"
        f"?status=success&branch={urllib.parse.quote(branch, safe='')}&per_page={max_runs}"
    )
    runs_payload = _api_json(runs_path, token)
    runs = runs_payload.get("workflow_runs", [])
    if not runs:
        raise ArtifactDownloadError(
            f"No successful runs found for workflow '{workflow_file}' on branch '{branch}'."
        )

    for run in runs:
        run_id = int(run["id"])
        artifacts_payload = _api_json(f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", token)
        for artifact in artifacts_payload.get("artifacts", []):
            if artifact.get("name") != artifact_name:
                continue
            if artifact.get("expired"):
                continue
            if int(artifact.get("size_in_bytes", 0)) <= 0:
                continue
            return run_id, artifact, len(runs)

    raise ArtifactDownloadError(
        "No matching non-expired artifact found. "
        f"workflow='{workflow_file}', artifact='{artifact_name}', branch='{branch}', "
        f"searched_runs={len(runs)}."
    )


def _write_github_output(run_id: int, artifact_id: int) -> None:
    out_path = os.getenv("GITHUB_OUTPUT")
    if not out_path:
        return
    with Path(out_path).open("a", encoding="utf-8") as fh:
        fh.write(f"run_id={run_id}\n")
        fh.write(f"artifact_id={artifact_id}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--workflow-file", required=True, help="Workflow file name, e.g. update_espn_cbb.yml")
    parser.add_argument("--artifact-name", required=True, help="Exact artifact name")
    parser.add_argument("--branch", default="main", help="Branch to search (default: main)")
    parser.add_argument("--dest", required=True, type=Path, help="Destination folder for extracted artifact")
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum successful runs to inspect")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_runs < 1 or args.max_runs > 100:
        raise ArtifactDownloadError("--max-runs must be between 1 and 100")

    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token:
        raise ArtifactDownloadError("Missing GH_TOKEN or GITHUB_TOKEN for GitHub API access")

    run_id, artifact, searched_runs = _find_artifact(
        repo=args.repo,
        workflow_file=args.workflow_file,
        artifact_name=args.artifact_name,
        branch=args.branch,
        max_runs=args.max_runs,
        token=token,
    )
    artifact_id = int(artifact["id"])
    size = int(artifact.get("size_in_bytes", 0))
    created_at = artifact.get("created_at", "unknown")
    print(
        "[INFO] Found artifact "
        f"name={args.artifact_name} run_id={run_id} artifact_id={artifact_id} "
        f"size_bytes={size} created_at={created_at} searched_runs={searched_runs}"
    )

    archive_url = artifact.get("archive_download_url")
    if not archive_url:
        raise ArtifactDownloadError(
            f"Artifact '{args.artifact_name}' on run {run_id} has no archive download URL."
        )

    data = _download_binary(str(archive_url), token)
    with tempfile.NamedTemporaryFile(prefix="artifact-", suffix=".zip", delete=False) as tmp:
        tmp.write(data)
        zip_path = Path(tmp.name)

    try:
        _safe_extract(zip_path, args.dest)
    finally:
        zip_path.unlink(missing_ok=True)

    _write_github_output(run_id=run_id, artifact_id=artifact_id)
    print(f"[OK] Extracted artifact to {args.dest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArtifactDownloadError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
