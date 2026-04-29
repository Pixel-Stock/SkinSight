from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from app.schemas import AnalysisResult, DetailedReportResponse, ProgressReport

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _supabase_base_url() -> str | None:
    project_id = os.getenv("SUPABASE_PROJECT_ID", "").strip()
    explicit_url = os.getenv("SUPABASE_URL", "").strip()
    if explicit_url:
        return explicit_url.rstrip("/")
    if project_id:
        return f"https://{project_id}.supabase.co"
    return None


def _service_role_key() -> str | None:
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or None


def _headers() -> dict[str, str] | None:
    base_url = _supabase_base_url()
    key = _service_role_key()
    if not base_url or not key:
        return None
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    json_body: Any = None,
    timeout: float = 10.0,
) -> tuple[int, str]:
    base_url = _supabase_base_url()
    headers = _headers()
    if not base_url or headers is None:
        return 0, "Supabase configuration missing"

    url = f"{base_url}/rest/v1/{path.lstrip('/')}"
    try:
        with httpx.Client(timeout=timeout) as client:
            res = client.request(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params,
                json=json_body,
            )
            return res.status_code, res.text
    except Exception as exc:
        return 0, str(exc)


def _insert_row(table: str, row: dict[str, Any]) -> None:
    status_code, text = _request("POST", table, json_body=row)
    if status_code >= 300:
        logger.warning(
            "Supabase insert failed for %s: %s %s",
            table,
            status_code,
            text[:200],
        )


def store_analysis(analysis: AnalysisResult, client_id: str | None = None) -> None:
    row = {
        "client_id": client_id,
        "acne_severity": analysis.acne_severity,
        "acne_score": analysis.acne_score,
        "lesion_count": len(analysis.lesions),
        "zone_counts": analysis.zone_counts,
        "hyperpigmentation": analysis.hyperpigmentation.model_dump(),
        "summary": analysis.summary,
        "annotated_image_base64": analysis.annotated_image_base64,
        "heatmap_image_base64": analysis.heatmap_image_base64,
    }
    _insert_row("analysis_runs", row)


def store_report(
    analysis: AnalysisResult,
    report: DetailedReportResponse,
    client_id: str | None = None,
) -> None:
    row = {
        "client_id": client_id,
        "analysis_summary": analysis.summary,
        "acne_severity": analysis.acne_severity,
        "model": report.model,
        "generated_by": report.generated_by,
        "report_text": report.report,
        "disclaimer": report.disclaimer,
    }
    _insert_row("detailed_reports", row)


def store_progress(report: ProgressReport, client_id: str | None = None) -> None:
    row = {
        "client_id": client_id,
        "similarity": report.similarity,
        "baseline_lesions": report.baseline_lesions,
        "followup_lesions": report.followup_lesions,
        "improvement_percent": report.improvement_percent,
        "timeline": report.timeline,
        "summary": report.summary,
        "stages": json.dumps([stage.model_dump() for stage in report.stages]),
    }
    _insert_row("progress_runs", row)


def db_health() -> dict[str, Any]:
    base_url = _supabase_base_url()
    if not base_url:
        return {"ok": False, "detail": "SUPABASE_URL / SUPABASE_PROJECT_ID missing"}
    if not _service_role_key():
        return {"ok": False, "detail": "SUPABASE_SERVICE_ROLE_KEY missing"}

    status_code, text = _request(
        "GET",
        "analysis_runs",
        params={"select": "id,created_at", "order": "created_at.desc", "limit": "1"},
        timeout=8.0,
    )
    if status_code in (200, 206):
        return {"ok": True, "detail": "Supabase reachable and analysis_runs readable"}
    if status_code == 404:
        return {"ok": False, "detail": "Table analysis_runs not found. Run supabase_schema.sql"}
    if status_code == 401:
        return {"ok": False, "detail": "Unauthorized. Check service role key"}
    return {"ok": False, "detail": f"Supabase error {status_code}: {text[:200]}"}


def get_analysis_history(
    client_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 100))
    status_code, text = _request(
        "GET",
        "analysis_runs",
        params={
            "select": "id,created_at,acne_severity,acne_score,lesion_count,hyperpigmentation,summary",
            "client_id": f"eq.{client_id}",
            "order": "created_at.desc",
            "limit": str(safe_limit),
        },
    )
    if status_code not in (200, 206):
        logger.warning("History fetch failed: %s %s", status_code, text[:200])
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []
