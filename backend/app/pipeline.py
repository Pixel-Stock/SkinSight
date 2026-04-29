import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.schemas import (
    AnalysisResult,
    BoundingBox,
    HyperpigmentationReport,
    ProgressStage,
    ProgressReport,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model caches
# ---------------------------------------------------------------------------
FACE_MESH = None
_YOLO_CLS_MODEL = None
_YOLO_DET_MODEL = None
_YOLO_LOAD_ATTEMPTED = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEVERITY_LEVELS = ["Clear", "Mild", "Moderate", "Severe"]
ZONE_ORDER = ["forehead", "left_cheek", "right_cheek", "nose", "chin_jawline"]
ZONE_COLORS = {
    "forehead": (99, 102, 241),
    "left_cheek": (16, 185, 129),
    "right_cheek": (16, 185, 129),
    "nose": (245, 158, 11),
    "chin_jawline": (236, 72, 153),
}

# MediaPipe FaceMesh landmark indices per facial zone
FACE_ZONE_LANDMARKS: dict[str, list[int]] = {
    "forehead": [10, 67, 69, 104, 108, 109, 151, 337, 338, 371, 397, 398],
    "left_cheek": [50, 100, 101, 102, 103, 116, 117, 118, 119, 120, 121, 123],
    "right_cheek": [280, 330, 331, 332, 333, 346, 347, 348, 349, 350, 351, 353],
    "nose": [1, 2, 3, 4, 5, 6, 48, 115, 122, 131, 134, 135, 138, 139, 168],
    "chin_jawline": [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163],
}

# HSV bounds keyed by Fitzpatrick phototype for hyperpigmentation detection
_FITZPATRICK_LOWER: dict[int, tuple[int, int, int]] = {
    1: (0, 0, 0),
    2: (0, 0, 50),
    3: (0, 0, 80),
    4: (0, 10, 100),
    5: (0, 20, 110),
    6: (5, 30, 120),
}
_FITZPATRICK_UPPER: dict[int, tuple[int, int, int]] = {
    1: (180, 50, 255),
    2: (180, 60, 240),
    3: (180, 70, 230),
    4: (180, 80, 220),
    5: (180, 100, 200),
    6: (180, 120, 180),
}


# ---------------------------------------------------------------------------
# YOLO model loading (lazy, cached, fail-safe)
# ---------------------------------------------------------------------------


def _load_yolo_models() -> None:
    """Load YOLO classification and segmentation models once.

    Detection uses YOLO11 segmentation (yolo11s-seg) when available, with a
    LAB+contour fallback to preserve behavior if model loading/inference fails.
    """
    global _YOLO_CLS_MODEL, _YOLO_DET_MODEL, _YOLO_LOAD_ATTEMPTED
    if _YOLO_LOAD_ATTEMPTED:
        return
    _YOLO_LOAD_ATTEMPTED = True
    _YOLO_DET_MODEL = None
    try:
        from ultralytics import YOLO  # type: ignore[import]

        seg_candidates = [
            "yolo11s-seg.pt",  # Ultralytics canonical filename
            "yolov11s-seg.pt",  # local variant users often keep manually
        ]
        seg_loaded = False
        for candidate in seg_candidates:
            try:
                if Path(candidate).exists() or candidate == "yolo11s-seg.pt":
                    _YOLO_DET_MODEL = YOLO(candidate)
                    logger.info("Segmentation model loaded: %s", candidate)
                    seg_loaded = True
                    break
            except Exception as exc:
                logger.warning("Failed to load segmentation model %s: %s", candidate, exc)

        if not seg_loaded:
            logger.warning(
                "YOLO11 segmentation unavailable - lesion detection will use LAB fallback"
            )

        try:
            _YOLO_CLS_MODEL = YOLO("yolov8s-cls.pt")
            logger.info("YOLOv8s-cls loaded successfully")
        except Exception as exc:
            logger.warning(
                "YOLOv8s-cls load failed – using heuristic fallback: %s", exc
            )
            _YOLO_CLS_MODEL = None

    except ImportError:
        logger.warning("ultralytics not installed – severity heuristic will be used")
        _YOLO_CLS_MODEL = None


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh (lazy, cached)
# ---------------------------------------------------------------------------


def _get_face_mesh():
    global FACE_MESH
    if FACE_MESH is not None:
        return FACE_MESH
    try:
        import mediapipe as mp
        FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return FACE_MESH
    except Exception as exc:
        logger.warning("MediaPipe FaceMesh unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    return image


def _encode_image_base64(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Failed to encode annotated image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Face bounding box
# ---------------------------------------------------------------------------


def _extract_face_landmarks(image_bgr: np.ndarray):
    """Extract 468-point facial landmarks with MediaPipe FaceMesh."""
    face_mesh = _get_face_mesh()
    if face_mesh is None:
        return None

    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark
    except Exception as exc:
        logger.warning("MediaPipe landmark extraction failed: %s", exc)
        return None


def _face_bbox_from_landmarks(
    landmarks, width: int, height: int
) -> tuple[int, int, int, int]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = max(0, int((min(xs) - 0.03) * width))
    y1 = max(0, int((min(ys) - 0.08) * height))
    x2 = min(width - 1, int((max(xs) + 0.03) * width))
    y2 = min(height - 1, int((max(ys) + 0.04) * height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Derived face bounds are invalid")
    return x1, y1, x2, y2


def _face_bbox_fallback(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Detect face using Haar Cascade with cascading parameter strategies."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    # Try progressively looser parameters until a face is detected
    detect_strategies = [
        {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": (120, 120)},
        {"scaleFactor": 1.1, "minNeighbors": 4, "minSize": (100, 100)},
        {"scaleFactor": 1.15, "minNeighbors": 3, "minSize": (100, 100)},
        {"scaleFactor": 1.05, "minNeighbors": 5, "minSize": (80, 80)},
        {"scaleFactor": 1.2, "minNeighbors": 3, "minSize": (90, 90)},
    ]

    detections = []
    for strategy in detect_strategies:
        detections = detector.detectMultiScale(gray, **strategy)
        if len(detections) > 0:
            logger.debug(f"Face detected with strategy: {strategy}")
            break

    if len(detections) == 0:
        raise ValueError("No face detected. Please upload a clear front-facing selfie")

    # Use largest detected face
    x, y, bw, bh = sorted(detections, key=lambda it: it[2] * it[3], reverse=True)[0]
    x1 = max(0, int(x - 0.05 * bw))
    y1 = max(0, int(y - 0.10 * bh))
    x2 = min(image_bgr.shape[1] - 1, int(x + 1.05 * bw))
    y2 = min(image_bgr.shape[0] - 1, int(y + 1.08 * bh))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Facial zone segmentation
# ---------------------------------------------------------------------------


def _build_landmark_zone_hulls(
    landmarks, width: int, height: int
) -> dict[str, np.ndarray]:
    """Return per-zone convex hulls derived from MediaPipe landmark indices."""
    hulls: dict[str, np.ndarray] = {}
    num_landmarks = len(landmarks)
    for zone, indices in FACE_ZONE_LANDMARKS.items():
        pts = np.array(
            [
                (int(landmarks[i].x * width), int(landmarks[i].y * height))
                for i in indices
                if i < num_landmarks
            ],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            hulls[zone] = cv2.convexHull(pts)
    return hulls


def _zone_for_point_hull(px: int, py: int, zone_hulls: dict[str, np.ndarray]) -> str:
    """Return the zone whose convex hull contains (px, py) with the largest margin."""
    best_zone = "nose"
    best_dist = -float("inf")
    for zone, hull in zone_hulls.items():
        dist = cv2.pointPolygonTest(hull, (float(px), float(py)), measureDist=True)
        if dist > best_dist:
            best_dist = dist
            best_zone = zone
    return best_zone


def _build_bbox_zones(
    face_bbox: tuple[int, int, int, int],
) -> dict[str, tuple[int, int, int, int]]:
    """Rectangular zone approximation used when landmarks are unavailable."""
    x1, y1, x2, y2 = face_bbox
    fw = x2 - x1
    fh = y2 - y1
    return {
        "forehead": (x1 + int(0.20 * fw), y1, x1 + int(0.80 * fw), y1 + int(0.25 * fh)),
        "left_cheek": (
            x1,
            y1 + int(0.28 * fh),
            x1 + int(0.45 * fw),
            y1 + int(0.68 * fh),
        ),
        "right_cheek": (
            x1 + int(0.55 * fw),
            y1 + int(0.28 * fh),
            x2,
            y1 + int(0.68 * fh),
        ),
        "nose": (
            x1 + int(0.40 * fw),
            y1 + int(0.24 * fh),
            x1 + int(0.60 * fw),
            y1 + int(0.72 * fh),
        ),
        "chin_jawline": (
            x1 + int(0.20 * fw),
            y1 + int(0.70 * fh),
            x1 + int(0.80 * fw),
            y2,
        ),
    }


def _zone_for_point_bbox(
    px: int, py: int, zones: dict[str, tuple[int, int, int, int]]
) -> str:
    for name, (zx1, zy1, zx2, zy2) in zones.items():
        if zx1 <= px <= zx2 and zy1 <= py <= zy2:
            return name
    return "nose"


def _resolve_zone(
    px: int,
    py: int,
    zone_hulls: Optional[dict[str, np.ndarray]],
    bbox_zones: Optional[dict[str, tuple[int, int, int, int]]],
) -> str:
    if zone_hulls:
        return _zone_for_point_hull(px, py, zone_hulls)
    if bbox_zones:
        return _zone_for_point_bbox(px, py, bbox_zones)
    return "nose"


# ---------------------------------------------------------------------------
# Lesion detection — YOLOv8m-det primary, LAB+contour fallback
# ---------------------------------------------------------------------------


def _detect_lesions_yolo(
    image_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    zone_hulls: Optional[dict[str, np.ndarray]],
    bbox_zones: Optional[dict[str, tuple[int, int, int, int]]],
) -> tuple[list[BoundingBox], dict[str, int]]:
    """Run YOLO segmentation on the face crop and convert masks to lesion boxes."""
    _load_yolo_models()
    if _YOLO_DET_MODEL is None:
        return [], {}

    x1, y1, x2, y2 = face_bbox
    face_bgr = image_bgr[y1:y2, x1:x2]
    if face_bgr.size == 0:
        return [], {}

    try:
        pil_face = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        results = _YOLO_DET_MODEL.predict(
            pil_face,
            conf=0.2,
            imgsz=640,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], {}

        masks = results[0].masks
        face_area = max(1, face_bgr.shape[0] * face_bgr.shape[1])

        lesions: list[BoundingBox] = []
        zone_counts: dict[str, int] = {z: 0 for z in ZONE_ORDER}

        for idx, box in enumerate(boxes):
            bx1, by1, bx2, by2 = (int(v) for v in box.xyxy[0].tolist())
            rw, rh = bx2 - bx1, by2 - by1
            if rw < 3 or rh < 3:
                continue

            if masks is not None and idx < len(masks.data):
                seg_mask = masks.data[idx].cpu().numpy() > 0.5
                seg_area = int(np.count_nonzero(seg_mask))
            else:
                seg_area = rw * rh

            # Keep compact lesions; reject tiny noise and very large non-lesion regions.
            if seg_area < 18 or seg_area > int(face_area * 0.03):
                continue

            cx = x1 + bx1 + rw // 2
            cy = y1 + by1 + rh // 2
            conf = float(box.conf[0])
            zone = _resolve_zone(cx, cy, zone_hulls, bbox_zones)
            lesions.append(
                BoundingBox(
                    x=x1 + bx1,
                    y=y1 + by1,
                    width=rw,
                    height=rh,
                    label="lesion",
                    confidence=round(conf, 3),
                    zone=zone,
                )
            )

        lesions = sorted(lesions, key=lambda b: b.confidence, reverse=True)[:60]
        zone_counts = {z: sum(1 for b in lesions if b.zone == z) for z in ZONE_ORDER}
        return lesions, zone_counts

    except Exception as exc:
        logger.warning("YOLO detection inference error: %s", exc)
        return [], {}


def _detect_lesions_lab(
    image_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    zone_hulls: Optional[dict[str, np.ndarray]],
    bbox_zones: Optional[dict[str, tuple[int, int, int, int]]],
) -> tuple[list[BoundingBox], dict[str, int]]:
    """LAB colour-space + contour heuristic — always available."""
    x1, y1, x2, y2 = face_bbox
    face = image_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return [], {z: 0 for z in ZONE_ORDER}

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    _, a_channel, _ = cv2.split(lab)
    blur = cv2.GaussianBlur(a_channel, (5, 5), 0)
    thresh_val = int(np.percentile(blur, 84))
    mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lesions: list[BoundingBox] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 18 or area > 1600:
            continue
        rx, ry, rw, rh = cv2.boundingRect(contour)
        cx = x1 + rx + rw // 2
        cy = y1 + ry + rh // 2
        zone = _resolve_zone(cx, cy, zone_hulls, bbox_zones)
        confidence = float(min(0.99, 0.45 + area / 1600.0))
        lesions.append(
            BoundingBox(
                x=int(x1 + rx),
                y=int(y1 + ry),
                width=int(rw),
                height=int(rh),
                label="lesion",
                confidence=round(confidence, 3),
                zone=zone,
            )
        )

    lesions = sorted(lesions, key=lambda b: b.confidence, reverse=True)[:60]
    zone_counts = {z: sum(1 for b in lesions if b.zone == z) for z in ZONE_ORDER}
    return lesions, zone_counts


def _detect_lesions(
    image_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    zone_hulls: Optional[dict[str, np.ndarray]],
    bbox_zones: Optional[dict[str, tuple[int, int, int, int]]],
) -> tuple[list[BoundingBox], dict[str, int]]:
    """Use YOLO11 segmentation first, then fall back to LAB contours."""
    yolo_lesions, yolo_zone_counts = _detect_lesions_yolo(
        image_bgr, face_bbox, zone_hulls, bbox_zones
    )
    if yolo_lesions:
        return yolo_lesions, yolo_zone_counts
    return _detect_lesions_lab(image_bgr, face_bbox, zone_hulls, bbox_zones)


# ---------------------------------------------------------------------------
# Acne severity — YOLOv8s-cls primary, lesion-count heuristic fallback
# ---------------------------------------------------------------------------


def _severity_from_lesion_count(lesion_count: int) -> tuple[str, float]:
    """Map raw lesion count to severity level with a score in [0, 1]."""
    if lesion_count < 5:
        idx = 0
    elif lesion_count < 15:
        idx = 1
    elif lesion_count < 30:
        idx = 2
    else:
        idx = 3
    return SEVERITY_LEVELS[idx], round(idx / 3.0, 3)


def _predict_acne_severity(
    image_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    lesion_count: int,
) -> tuple[str, float]:
    """
    Grade acne severity.  YOLOv8s-cls provides a raw class index; because the
    model is not fine-tuned on acne data we blend it 40 / 60 with the
    lesion-count heuristic to avoid unreliable standalone predictions.
    Falls back fully to the heuristic if YOLO is unavailable or fails.
    """
    _load_yolo_models()

    heuristic_severity, heuristic_score = _severity_from_lesion_count(lesion_count)
    heuristic_idx = SEVERITY_LEVELS.index(heuristic_severity)

    if _YOLO_CLS_MODEL is not None:
        x1, y1, x2, y2 = face_bbox
        face_bgr = image_bgr[y1:y2, x1:x2]
        if face_bgr.size > 0:
            try:
                pil_face = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
                results = _YOLO_CLS_MODEL.predict(pil_face, imgsz=224, verbose=False)
                yolo_idx = int(results[0].probs.top1) % 4  # guard against >4 classes
                blended_idx = int(
                    np.clip(round(heuristic_idx * 0.6 + yolo_idx * 0.4), 0, 3)
                )
                return SEVERITY_LEVELS[blended_idx], round(blended_idx / 3.0, 3)
            except Exception as exc:
                logger.warning("YOLOv8s-cls inference error: %s", exc)

    return heuristic_severity, heuristic_score


# ---------------------------------------------------------------------------
# Fitzpatrick phototype detection
# ---------------------------------------------------------------------------


def _detect_fitzpatrick_type(l_channel: np.ndarray) -> int:
    """
    Estimate Fitzpatrick skin phototype from the median LAB L* value of the
    face crop.  Thresholds calibrated for OpenCV's 0-255 L* range.
    """
    l_mean = float(np.median(l_channel))
    if l_mean > 200:
        return 1
    elif l_mean > 170:
        return 2
    elif l_mean > 140:
        return 3
    elif l_mean > 110:
        return 4
    elif l_mean > 80:
        return 5
    else:
        return 6


# ---------------------------------------------------------------------------
# Hyperpigmentation — HSV thresholding with Fitzpatrick calibration
# ---------------------------------------------------------------------------


def _hyperpigmentation_report(
    image_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
) -> tuple[HyperpigmentationReport, np.ndarray]:
    x1, y1, x2, y2 = face_bbox
    face = image_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return (
            HyperpigmentationReport(coverage_percent=0.0, severity="Low"),
            np.zeros((1, 1), dtype=np.uint8),
        )

    # Fitzpatrick phototype from LAB L*
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    fitz_type = _detect_fitzpatrick_type(lab[:, :, 0])

    # HSV-based dark-spot mask calibrated per phototype
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    lower = np.array(_FITZPATRICK_LOWER[fitz_type], dtype=np.uint8)
    upper = np.array(_FITZPATRICK_UPPER[fitz_type], dtype=np.uint8)
    dark_mask = cv2.inRange(hsv, lower, upper)

    # Morphological clean-up to remove noise
    kernel = np.ones((5, 5), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    dark_mask = cv2.GaussianBlur(dark_mask, (5, 5), 0)
    _, dark_mask = cv2.threshold(dark_mask, 80, 255, cv2.THRESH_BINARY)

    coverage = float(np.count_nonzero(dark_mask) / dark_mask.size * 100.0)
    if coverage < 5:
        severity = "Low"
    elif coverage < 15:
        severity = "Moderate"
    else:
        severity = "High"

    return (
        HyperpigmentationReport(coverage_percent=round(coverage, 2), severity=severity),
        dark_mask,
    )


# ---------------------------------------------------------------------------
# Annotated overlay
# ---------------------------------------------------------------------------


def _draw_overlay(
    image_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    zone_hulls: Optional[dict[str, np.ndarray]],
    bbox_zones: Optional[dict[str, tuple[int, int, int, int]]],
    lesions: list[BoundingBox],
    hyperpig_mask: np.ndarray,
) -> np.ndarray:
    output = image_bgr.copy()
    x1, y1, x2, y2 = face_bbox

    # Zone outlines
    if zone_hulls:
        for zone_name, hull in zone_hulls.items():
            color = ZONE_COLORS[zone_name]
            cv2.polylines(output, [hull], isClosed=True, color=color, thickness=2)
            M = cv2.moments(hull)
            if M["m00"] != 0:
                lx = int(M["m10"] / M["m00"])
                ly = int(M["m01"] / M["m00"])
                cv2.putText(
                    output,
                    zone_name,
                    (lx - 28, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    elif bbox_zones:
        for zone_name, (zx1, zy1, zx2, zy2) in bbox_zones.items():
            color = ZONE_COLORS[zone_name]
            cv2.rectangle(output, (zx1, zy1), (zx2, zy2), color, 2)
            cv2.putText(
                output,
                zone_name,
                (zx1, max(18, zy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    # Lesion bounding boxes — confidence-based colour coding
    # high (≥0.75): red (inflammatory)  mid (≥0.55): amber  low: green (comedonal)
    for idx, lesion in enumerate(lesions, start=1):
        if lesion.confidence >= 0.75:
            box_color = (0, 0, 220)  # red
        elif lesion.confidence >= 0.55:
            box_color = (0, 165, 255)  # amber
        else:
            box_color = (60, 180, 60)  # green
        cv2.rectangle(
            output,
            (lesion.x, lesion.y),
            (lesion.x + lesion.width, lesion.y + lesion.height),
            box_color,
            2,
        )
        cv2.putText(
            output,
            f"L{idx}",
            (lesion.x, max(18, lesion.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )

    # Hyperpigmentation tint overlay
    if hyperpig_mask.size > 1:
        region = output[y1:y2, x1:x2]
        if region.shape[:2] == hyperpig_mask.shape[:2]:
            tint = np.zeros_like(region)
            tint[:, :, 0] = 180  # blue channel  → purple in BGR
            tint[:, :, 2] = 130  # red channel   → purple in BGR
            alpha = (hyperpig_mask.astype(np.float32) / 255.0)[:, :, None] * 0.35
            output[y1:y2, x1:x2] = (region * (1 - alpha) + tint * alpha).astype(
                np.uint8
            )

    # Face bounding box
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return output


def _build_lesion_heatmap(
    image_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    lesions: list[BoundingBox],
) -> np.ndarray:
    """Create a lesion-density heatmap overlay for visual hotspot inspection."""
    output = image_bgr.copy()
    x1, y1, x2, y2 = face_bbox
    face_h = max(1, y2 - y1)
    face_w = max(1, x2 - x1)

    density = np.zeros((face_h, face_w), dtype=np.float32)
    if not lesions:
        return output

    radius = max(8, int(min(face_h, face_w) * 0.035))
    for lesion in lesions:
        cx = int(np.clip(lesion.x + lesion.width // 2 - x1, 0, face_w - 1))
        cy = int(np.clip(lesion.y + lesion.height // 2 - y1, 0, face_h - 1))
        strength = 0.6 + float(lesion.confidence) * 0.4
        cv2.circle(density, (cx, cy), radius, strength, -1)

    density = cv2.GaussianBlur(density, (0, 0), sigmaX=13, sigmaY=13)
    max_val = float(np.max(density))
    if max_val <= 1e-6:
        return output

    norm = np.uint8(np.clip((density / max_val) * 255.0, 0, 255))
    color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    alpha = (norm.astype(np.float32) / 255.0)[:, :, None] * 0.55

    region = output[y1:y2, x1:x2]
    blended = (region.astype(np.float32) * (1.0 - alpha)) + (
        color_map.astype(np.float32) * alpha
    )
    output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return output


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def _summary_text(
    acne_severity: str,
    lesions: list[BoundingBox],
    hyperpigmentation: HyperpigmentationReport,
    zone_counts: dict[str, int],
) -> str:
    most_affected = (
        max(zone_counts.items(), key=lambda kv: kv[1])[0] if zone_counts else "nose"
    )
    count = len(lesions)
    return (
        f"Detected acne severity is {acne_severity}. "
        f"A total of {count} probable lesion(s) were found, "
        f"with the highest concentration in the {most_affected} region. "
        f"Estimated hyperpigmentation coverage is "
        f"{hyperpigmentation.coverage_percent}% ({hyperpigmentation.severity}). "
        "Use this report as a visual tracking baseline and consult a dermatologist for diagnosis."
    )


# ---------------------------------------------------------------------------
# Primary analysis entry point
# ---------------------------------------------------------------------------


def analyze_image(image_bytes: bytes) -> AnalysisResult:
    image_bgr = _decode_image(image_bytes)
    h, w = image_bgr.shape[:2]
    if min(h, w) < 220:
        raise ValueError("Image is too small. Please upload a higher resolution selfie")

    # Attempt MediaPipe landmark-based segmentation; fall back to Haar cascade
    landmarks = _extract_face_landmarks(image_bgr)
    if landmarks is not None:
        face_bbox = _face_bbox_from_landmarks(landmarks, w, h)
        zone_hulls: Optional[dict[str, np.ndarray]] = _build_landmark_zone_hulls(
            landmarks, w, h
        )
        bbox_zones: Optional[dict[str, tuple[int, int, int, int]]] = None
    else:
        face_bbox = _face_bbox_fallback(image_bgr)
        zone_hulls = None
        bbox_zones = _build_bbox_zones(face_bbox)

    lesions, zone_counts = _detect_lesions(image_bgr, face_bbox, zone_hulls, bbox_zones)
    hyperpigmentation, hyperpig_mask = _hyperpigmentation_report(image_bgr, face_bbox)
    acne_severity, acne_score = _predict_acne_severity(
        image_bgr, face_bbox, len(lesions)
    )
    annotated = _draw_overlay(
        image_bgr, face_bbox, zone_hulls, bbox_zones, lesions, hyperpig_mask
    )
    heatmap = _build_lesion_heatmap(image_bgr, face_bbox, lesions)
    encoded = _encode_image_base64(annotated)
    heatmap_encoded = _encode_image_base64(heatmap)
    summary = _summary_text(acne_severity, lesions, hyperpigmentation, zone_counts)

    return AnalysisResult(
        acne_severity=acne_severity,
        acne_score=acne_score,
        lesions=lesions,
        zone_counts=zone_counts,
        hyperpigmentation=hyperpigmentation,
        summary=summary,
        annotated_image_base64=encoded,
        heatmap_image_base64=heatmap_encoded,
    )


# ---------------------------------------------------------------------------
# Progress tracking — SSIM + ORB registration + lesion delta
# ---------------------------------------------------------------------------


def _register_images(
    img_a: np.ndarray, img_b: np.ndarray, target_size: tuple[int, int] = (512, 512)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align img_b to img_a using ORB keypoints + RANSAC homography.
    Returns both images at target_size; returns raw resize if alignment fails.
    """
    a_resized = cv2.resize(img_a, target_size)
    b_resized = cv2.resize(img_b, target_size)

    gray_a = cv2.cvtColor(a_resized, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b_resized, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return a_resized, b_resized

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(desc_a, desc_b), key=lambda m: m.distance)
    good = matches[: max(10, len(matches) // 3)]

    if len(good) < 4:
        return a_resized, b_resized

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)  # type: ignore[call-overload]
    if H is None:
        return a_resized, b_resized

    b_aligned = cv2.warpPerspective(b_resized, H, target_size)
    return a_resized, b_aligned


def _count_lesions_quick(image_bgr: np.ndarray) -> int:
    """Lightweight lesion count used only for progress delta calculation."""
    h, w = image_bgr.shape[:2]
    landmarks = _extract_face_landmarks(image_bgr)

    if landmarks is not None:
        face_bbox = _face_bbox_from_landmarks(landmarks, w, h)
        zone_hulls: Optional[dict[str, np.ndarray]] = _build_landmark_zone_hulls(
            landmarks, w, h
        )
        bbox_zones: Optional[dict[str, tuple[int, int, int, int]]] = None
    else:
        try:
            face_bbox = _face_bbox_fallback(image_bgr)
        except ValueError:
            face_bbox = (0, 0, w - 1, h - 1)
        zone_hulls = None
        bbox_zones = _build_bbox_zones(face_bbox)

    lesions, _ = _detect_lesions(image_bgr, face_bbox, zone_hulls, bbox_zones)
    return len(lesions)


def compare_progress(baseline_bytes: bytes, followup_bytes: bytes) -> ProgressReport:
    """
    Accept a baseline and a follow-up skin image, register them, compute SSIM,
    count lesions in each, and return a structured ProgressReport.
    """
    try:
        from skimage.metrics import (
            structural_similarity as ssim,  # type: ignore[import]
        )
    except ImportError as exc:
        raise RuntimeError(
            "scikit-image is required for progress tracking. "
            "Install it with: pip install scikit-image"
        ) from exc

    baseline_bgr = _decode_image(baseline_bytes)
    followup_bgr = _decode_image(followup_bytes)

    # Image registration
    try:
        baseline_aligned, followup_aligned = _register_images(
            baseline_bgr, followup_bgr
        )
    except Exception as exc:
        logger.warning("Image registration failed; using plain resize: %s", exc)
        target = (512, 512)
        baseline_aligned = cv2.resize(baseline_bgr, target)
        followup_aligned = cv2.resize(followup_bgr, target)

    # SSIM on grayscale aligned images
    gray_base = cv2.cvtColor(baseline_aligned, cv2.COLOR_BGR2GRAY)
    gray_follow = cv2.cvtColor(followup_aligned, cv2.COLOR_BGR2GRAY)
    similarity = float(ssim(gray_base, gray_follow, data_range=255))

    # Lesion counts on original (unresized) images for accuracy
    baseline_count = _count_lesions_quick(baseline_bgr)
    followup_count = _count_lesions_quick(followup_bgr)

    # Improvement: positive = fewer lesions (improvement), negative = more (worsening)
    if baseline_count > 0:
        improvement = (baseline_count - followup_count) / baseline_count * 100.0
    elif followup_count == 0:
        improvement = 100.0
    else:
        improvement = -100.0
    improvement = float(np.clip(improvement, -100.0, 100.0))

    # Timeline: images that are structurally very similar were taken close together
    timeline = "short_term" if similarity >= 0.75 else "long_term"

    # Build human-readable summary
    lesion_delta = followup_count - baseline_count
    delta_sign = "+" if lesion_delta >= 0 else ""

    if improvement > 10:
        trend = f"improvement of {improvement:.1f}%"
    elif improvement < -10:
        trend = f"worsening of {abs(improvement):.1f}%"
    else:
        trend = "stable condition with minimal change"

    timeline_desc = (
        "short-term comparison (days to a few weeks)"
        if timeline == "short_term"
        else "long-term comparison (several weeks to months)"
    )

    summary = (
        f"Lesion count changed from {baseline_count} to {followup_count} "
        f"({delta_sign}{lesion_delta} lesions). "
        f"Structural image similarity: {similarity:.2%}. "
        f"Overall skin trend: {trend}. "
        f"This appears to be a {timeline_desc}. "
        "Consult a dermatologist to validate these findings and adjust your treatment plan."
    )

    now_stage = ProgressStage(
        key="now",
        title="Now",
        bullets=[
            f"Current follow-up lesion count: {followup_count}.",
            f"Observed change from baseline: {delta_sign}{lesion_delta} lesions.",
            f"Immediate trend status: {trend}.",
        ],
    )

    short_term_bullet = (
        "Maintain current routine for 2-4 weeks and monitor irritation signs weekly."
        if improvement >= 0
        else "Simplify routine for 1-2 weeks, prioritize barrier repair, then reintroduce actives slowly."
    )
    short_term_stage = ProgressStage(
        key="short_term",
        title="Short Term (2-4 weeks)",
        bullets=[
            short_term_bullet,
            "Track weekly photos in consistent lighting and angle.",
            "Adjust active ingredient frequency based on tolerance.",
        ],
    )

    long_term_stage = ProgressStage(
        key="long_term",
        title="Long Term (8-12 weeks)",
        bullets=[
            "Target sustained reduction in new lesion formation and post-inflammatory marks.",
            "Reassess progress using lesion count trend and image similarity together.",
            "Consult dermatology if worsening persists or scarring risk increases.",
        ],
    )

    return ProgressReport(
        similarity=round(similarity, 4),
        baseline_lesions=baseline_count,
        followup_lesions=followup_count,
        improvement_percent=round(improvement, 2),
        timeline=timeline,
        stages=[now_stage, short_term_stage, long_term_stage],
        summary=summary,
    )
