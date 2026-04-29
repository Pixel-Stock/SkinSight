from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    label: str
    confidence: float
    zone: str


class HyperpigmentationReport(BaseModel):
    coverage_percent: float
    severity: str


class AnalysisResult(BaseModel):
    acne_severity: str
    acne_score: float
    lesions: list[BoundingBox]
    zone_counts: dict[str, int]
    hyperpigmentation: HyperpigmentationReport
    summary: str
    annotated_image_base64: str
    heatmap_image_base64: str


class ProgressStage(BaseModel):
    key: str
    title: str
    bullets: list[str]


class ProgressReport(BaseModel):
    similarity: float
    baseline_lesions: int
    followup_lesions: int
    improvement_percent: float
    timeline: str  # "short_term" or "long_term"
    stages: list[ProgressStage]
    summary: str


class DetailedReportRequest(BaseModel):
    analysis: AnalysisResult


class DetailedReportResponse(BaseModel):
    generated_by: str
    model: str
    report: str
    disclaimer: str
    created_at: str
