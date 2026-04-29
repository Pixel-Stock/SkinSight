from __future__ import annotations

from datetime import datetime, timezone
import os

import httpx

from app.schemas import AnalysisResult, DetailedReportResponse

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma3:1b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0


def _build_prompt(analysis: AnalysisResult) -> str:
    zone_lines = "\n".join(
        f"- {zone}: {count}" for zone, count in analysis.zone_counts.items()
    )

    return f"""
You are an evidence-oriented skincare assistant.
Provide a detailed, practical skin report using the metrics below.
Do not claim a medical diagnosis. Use "possible" language and clear safety notes.

Input metrics:
- Acne severity: {analysis.acne_severity}
- Acne score (0 to 1): {analysis.acne_score}
- Lesion count: {len(analysis.lesions)}
- Hyperpigmentation coverage %: {analysis.hyperpigmentation.coverage_percent}
- Hyperpigmentation severity: {analysis.hyperpigmentation.severity}
- Lesion zone counts:
{zone_lines}

Output format requirements:
1) Observed Skin Pattern (4-6 bullets)
2) Possible Conditions (ranked, with confidence Low/Medium/High)
3) Possible Triggers (lifestyle, skincare, hormones, environment)
4) Remedy Plan:
   - Morning routine
   - Evening routine
   - Weekly routine
   - Habit changes
5) Ingredient Guide:
   - Helpful ingredients and why
   - Ingredients to avoid right now
6) When to Consult a Dermatologist (red flags)
7) Four-Week Tracking Checklist

Keep it concise but detailed. Use plain language for a non-medical user.
""".strip()


def _fallback_report(analysis: AnalysisResult, reason: str) -> DetailedReportResponse:
    lesion_count = len(analysis.lesions)
    sorted_zones = sorted(analysis.zone_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_zone = sorted_zones[0][0] if sorted_zones else "nose"

    report = f"""
1) Observed Skin Pattern
- Acne severity appears {analysis.acne_severity} with an acne score of {analysis.acne_score:.2f}.
- Approximate lesion count: {lesion_count}.
- Most affected area: {top_zone}.
- Hyperpigmentation coverage is {analysis.hyperpigmentation.coverage_percent:.1f}% ({analysis.hyperpigmentation.severity}).

2) Possible Conditions (non-diagnostic)
- Acne vulgaris pattern: High confidence.
- Post-inflammatory hyperpigmentation tendency: Medium confidence.
- Barrier irritation overlap: Low to Medium confidence.

3) Possible Triggers
- Friction/sweat and occlusion (helmets, masks, pillowcases).
- Comedogenic or overly heavy products.
- Hormonal variation and stress-related flare cycles.
- Inconsistent sunscreen use leading to dark-mark persistence.

4) Remedy Plan
- Morning: gentle cleanser, non-comedogenic moisturizer, broad-spectrum SPF 30+.
- Evening: gentle cleanser, adapalene or salicylic acid on alternate nights, moisturizer.
- Weekly: 1-2 recovery nights with only cleanser + moisturizer.
- Habits: avoid picking, clean phone/pillowcase, keep hair products off face.

5) Ingredient Guide
- Helpful now: niacinamide, azelaic acid, salicylic acid (as tolerated), adapalene at night.
- Use caution: strong scrubs, high-fragrance products, over-layering multiple strong actives.

6) When to Consult a Dermatologist
- Painful nodules/cysts, visible scarring, worsening despite 8-12 weeks of routine,
  or irritation that does not settle after stopping actives.

7) Four-Week Tracking Checklist
- Capture photos in same lighting weekly.
- Track new lesion count and dark-mark intensity.
- Track irritation signs (burning, peeling, redness duration).
- Review trend at week 4 and adjust routine if needed.

Report generation note: local model response unavailable ({reason}).
""".strip()

    return DetailedReportResponse(
        generated_by="fallback",
        model=DEFAULT_MODEL,
        report=report,
        disclaimer=(
            "This report is informational and not a medical diagnosis. "
            "Consult a qualified dermatologist for clinical decisions."
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def generate_detailed_report(analysis: AnalysisResult) -> DetailedReportResponse:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    timeout_seconds = float(
        os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))
    )

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful skincare assistant. "
                    "Never provide definitive diagnosis."
                ),
            },
            {"role": "user", "content": _build_prompt(analysis)},
        ],
        "options": {
            "temperature": 0.2,
            "num_predict": 700,
        },
    }

    try:
        with httpx.Client(
            timeout=httpx.Timeout(
                timeout=timeout_seconds,
                connect=5.0,
                read=timeout_seconds,
                write=20.0,
            )
        ) as client:
            response = client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        content = (
            data.get("message", {}).get("content")
            if isinstance(data, dict)
            else None
        )
        if not content:
            return _fallback_report(analysis, "empty response from ollama")

        return DetailedReportResponse(
            generated_by="ollama",
            model=model,
            report=content.strip(),
            disclaimer=(
                "This report is informational and not a medical diagnosis. "
                "Consult a qualified dermatologist for clinical decisions."
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        return _fallback_report(analysis, str(exc))
