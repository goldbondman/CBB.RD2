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

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        endpoint: str | None = None,
        status_code: int | None = None,
        accept: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.endpoint = endpoint
        self.status_code = status_code
        self.accept = accept
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": str(self),
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "accept_header": self.accept,
            "detail": self.detail,
        }


def _trim_detail(detail: str, limit: int = 2000) -> str:
    compact = " ".join(detail.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "...(truncated)"


class _SafeArtifactRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow archive redirects without leaking GitHub auth headers cross-host."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if code not in (301, 302, 303, 307, 308):
            return None
        return urllib.request.Request(
            newurl,
            headers={
                "Accept": "*/*",
                "User-Agent": "cbb-download-latest-artifact",
            },
            method="GET",
        )


def _headers(token: str, accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Accept": accept,
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "cbb-download-latest-artifact",
    }


def _api_json(path: str, token: str, stage: str) -> dict[str, Any]:
    req_headers = _headers(token)
    endpoint = f"{API_BASE}{path}"
    req = urllib.request.Request(endpoint, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _trim_detail(exc.read().decode("utf-8", errors="replace"))
        raise ArtifactDownloadError(
            stage,
            f"GitHub API error for {path}: {exc.code} {detail}",
            endpoint=endpoint,
            status_code=exc.code,
            accept=req_headers.get("Accept"),
            detail=detail,
        ) from exc
    except urllib.error.URLError as exc:
        raise ArtifactDownloadError(
            stage,
            f"Network error calling GitHub API {path}: {exc}",
            endpoint=endpoint,
            accept=req_headers.get("Accept"),
            detail=str(exc),
        ) from exc


def _download_binary(url: str, token: str) -> bytes:
    req_headers = _headers(token)
    req = urllib.request.Request(url, headers=req_headers)
    opener = urllib.request.build_opener(_SafeArtifactRedirectHandler())
    try:
        with opener.open(req, timeout=300) as resp:
            data = resp.read()
            if not data:
                raise ArtifactDownloadError(
                    "VALIDATION_FAILED",
                    "Downloaded artifact archive is empty.",
                    endpoint=resp.geturl(),
                    status_code=getattr(resp, "status", None),
                    accept=req_headers.get("Accept"),
                )
            return data
    except urllib.error.HTTPError as exc:
        detail = _trim_detail(exc.read().decode("utf-8", errors="replace"))
        raise ArtifactDownloadError(
            "DOWNLOAD_FAILED",
            f"Artifact download failed: {exc.code} {detail}",
            endpoint=url,
            status_code=exc.code,
            accept=req_headers.get("Accept"),
            detail=detail,
        ) from exc
    except urllib.error.URLError as exc:
        raise ArtifactDownloadError(
            "DOWNLOAD_FAILED",
            f"Network error downloading artifact: {exc}",
            endpoint=url,
            accept=req_headers.get("Accept"),
            detail=str(exc),
        ) from exc


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_dest = destination.resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                member_path = (destination / member.filename).resolve()
                if not str(member_path).startswith(str(resolved_dest)):
                    raise ArtifactDownloadError(
                        "EXTRACT_FAILED",
                        f"Unsafe path in artifact archive: {member.filename}",
                    )
            zf.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise ArtifactDownloadError(
            "EXTRACT_FAILED",
            f"Invalid zip archive: {exc}",
            endpoint=str(zip_path),
            detail=str(exc),
        ) from exc


def _find_artifact(
    repo: str,
    workflow_file: str,
    artifact_name: str,
    branch: str,
    max_runs: int,
    token: str,
    allow_failed_runs: bool = False,
) -> tuple[int, dict[str, Any], int]:
    workflow_ref = urllib.parse.quote(workflow_file, safe="")
    branch_enc = urllib.parse.quote(branch, safe="")

    def _search_runs(status: str) -> tuple[list[Any], int | None, dict[str, Any] | None]:
        runs_path = (
            f"/repos/{repo}/actions/workflows/{workflow_ref}/runs"
            f"?status={status}&branch={branch_enc}&per_page={max_runs}"
        )
        payload = _api_json(runs_path, token, stage="LOOKUP_FAILED")
        candidate_runs = payload.get("workflow_runs", [])
        for run in candidate_runs:
            run_id = int(run["id"])
            artifacts_payload = _api_json(
                f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100",
                token,
                stage="LOOKUP_FAILED",
            )
            for artifact in artifacts_payload.get("artifacts", []):
                if artifact.get("name") != artifact_name:
                    continue
                if artifact.get("expired"):
                    continue
                if int(artifact.get("size_in_bytes", 0)) <= 0:
                    continue
                return candidate_runs, run_id, artifact
        return candidate_runs, None, None

    # Always try successful runs first.
    success_runs, run_id, artifact = _search_runs("success")
    if run_id is not None and artifact is not None:
        return run_id, artifact, len(success_runs)

    if not success_runs:
        no_runs_msg = (
            f"No successful runs found for workflow '{workflow_file}' on branch '{branch}'."
        )
        if not allow_failed_runs:
            raise ArtifactDownloadError("LOOKUP_FAILED", no_runs_msg)

    # Optionally fall back to completed (potentially failed) runs.  Workflows
    # that upload artifacts via ``if: always()`` will have valid data even when
    # the overall job conclusion is failure.
    if allow_failed_runs:
        failed_runs, run_id, artifact = _search_runs("failure")
        total_searched = len(success_runs) + len(failed_runs)
        if run_id is not None and artifact is not None:
            return run_id, artifact, total_searched
        raise ArtifactDownloadError(
            "LOOKUP_FAILED",
            "No matching non-expired artifact found (searched successful and failed runs). "
            f"workflow='{workflow_file}', artifact='{artifact_name}', branch='{branch}', "
            f"searched_runs={total_searched}.",
        )

    raise ArtifactDownloadError(
        "LOOKUP_FAILED",
        "No matching non-expired artifact found. "
        f"workflow='{workflow_file}', artifact='{artifact_name}', branch='{branch}', "
        f"searched_runs={len(success_runs)}.",
    )


def _write_github_output(run_id: int, artifact_id: int) -> None:
    out_path = os.getenv("GITHUB_OUTPUT")
    if not out_path:
        return
    with Path(out_path).open("a", encoding="utf-8") as fh:
        fh.write(f"run_id={run_id}\n")
        fh.write(f"artifact_id={artifact_id}\n")


def _write_debug_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fetch_artifact_metadata(repo: str, artifact_id: int, token: str) -> dict[str, Any]:
    return _api_json(
        f"/repos/{repo}/actions/artifacts/{artifact_id}",
        token,
        stage="METADATA_FAILED",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--workflow-file", required=True, help="Workflow file name, e.g. update_espn_cbb.yml")
    parser.add_argument("--artifact-name", required=True, help="Exact artifact name")
    parser.add_argument("--branch", default="main", help="Branch to search (default: main)")
    parser.add_argument("--dest", required=True, type=Path, help="Destination folder for extracted artifact")
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum successful runs to inspect")
    parser.add_argument(
        "--allow-failed-runs",
        action="store_true",
        default=False,
        help=(
            "When no artifact is found in successful runs, also search the most recent "
            "failed runs. Useful for workflows that upload artifacts via 'if: always()'."
        ),
    )
    parser.add_argument(
        "--debug-json",
        type=Path,
        help="Optional path for structured diagnostics (success or failure).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    diag: dict[str, Any] = {
        "repo": args.repo,
        "workflow_file": args.workflow_file,
        "artifact_name": args.artifact_name,
        "branch": args.branch,
        "max_runs": args.max_runs,
        "status": "STARTED",
    }

    try:
        if args.max_runs < 1 or args.max_runs > 100:
            raise ArtifactDownloadError("VALIDATION_FAILED", "--max-runs must be between 1 and 100")

        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        if not token:
            raise ArtifactDownloadError(
                "VALIDATION_FAILED",
                "Missing GH_TOKEN or GITHUB_TOKEN for GitHub API access",
            )

        run_id, artifact, searched_runs = _find_artifact(
            repo=args.repo,
            workflow_file=args.workflow_file,
            artifact_name=args.artifact_name,
            branch=args.branch,
            max_runs=args.max_runs,
            token=token,
            allow_failed_runs=args.allow_failed_runs,
        )
        artifact_id = int(artifact["id"])
        size = int(artifact.get("size_in_bytes", 0))
        created_at = artifact.get("created_at", "unknown")
        diag.update(
            {
                "status": "LOOKUP_OK",
                "run_id": run_id,
                "artifact_id": artifact_id,
                "searched_runs": searched_runs,
                "artifact_size_bytes": size,
                "artifact_created_at": created_at,
            }
        )
        print(
            "[INFO] Found artifact "
            f"name={args.artifact_name} run_id={run_id} artifact_id={artifact_id} "
            f"size_bytes={size} created_at={created_at} searched_runs={searched_runs}"
        )

        metadata = _fetch_artifact_metadata(args.repo, artifact_id, token)
        archive_url = metadata.get("archive_download_url")
        if not archive_url:
            raise ArtifactDownloadError(
                "METADATA_FAILED",
                f"Artifact '{args.artifact_name}' on run {run_id} has no archive download URL.",
                endpoint=f"{API_BASE}/repos/{args.repo}/actions/artifacts/{artifact_id}",
                accept="application/vnd.github+json",
            )
        diag["archive_download_url"] = archive_url
        diag["download_accept_header"] = "application/vnd.github+json"

        data = _download_binary(str(archive_url), token)
        with tempfile.NamedTemporaryFile(prefix="artifact-", suffix=".zip", delete=False) as tmp:
            tmp.write(data)
            zip_path = Path(tmp.name)

        try:
            if not zipfile.is_zipfile(zip_path):
                raise ArtifactDownloadError(
                    "VALIDATION_FAILED",
                    f"Downloaded payload is not a valid zip archive: {zip_path}",
                    endpoint=str(archive_url),
                    accept="application/vnd.github+json",
                )
            _safe_extract(zip_path, args.dest)
        finally:
            zip_path.unlink(missing_ok=True)

        diag.update(
            {
                "status": "SUCCESS",
                "extracted_to": str(args.dest),
                "downloaded_bytes": len(data),
            }
        )
        _write_github_output(run_id=run_id, artifact_id=artifact_id)
        print(f"[OK] Extracted artifact to {args.dest}")
        if args.debug_json:
            _write_debug_json(args.debug_json, diag)
        return 0
    except ArtifactDownloadError as exc:
        diag.update({"status": "FAILED", "error": exc.to_dict()})
        if args.debug_json:
            _write_debug_json(args.debug_json, diag)
        print(f"[ERROR][STAGE={exc.stage}] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
