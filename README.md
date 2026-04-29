# SkinSight AI - AI Skin Health Analyzer

**Upload a selfie → Get a full dermatological-grade visual skin report in under 5 seconds.**

SkinSight AI is an intelligent web application that analyzes facial skin conditions using computer vision and AI, delivering instant, visual, and actionable insights - making professional-level skin screening accessible to everyone.

---

## **Quick Start Guide**

Follow these steps to run the project locally:

### 1. Backend Setup
```bash
cd backend
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Notes for first run
- First `/analyze` request may take longer because YOLO weights are downloaded and cached automatically.
- `/report` uses a local Ollama endpoint when available; if not, it safely falls back to an internal templated report.
- Bun is optional. If Bun is installed you can use `bun install` and `bun run dev`.

### 4. One-click startup (Windows)
From project root, run:
```bat
Run-SkinSight.bat
```
This installs missing dependencies, starts backend + frontend in separate terminals, and opens the browser automatically.

### 5. Supabase setup
- Backend persistence is configured through `backend/.env`.
- Run `backend/supabase_schema.sql` in the Supabase SQL editor once to create required tables.
- Verify integration:
  - `GET /db/health` -> should return `ok: true`
  - `GET /history` with header `X-Client-Id` -> returns stored analysis history

---

## **Problem Statement**

- **3 billion people** worldwide lack access to a dermatologist.
- Dermatology consultations cost **$150–$300+** with **3–6 week** waiting times.
- **85% of skin conditions** are visually diagnosable, yet remain undetected due to access barriers.
- Existing apps lack **visual depth**, **medical-grade accuracy**, and **skin-tone inclusivity**.

**Result:** Millions suffer in silence with preventable or treatable skin issues.

---

## **Our Solution**

**SkinSight AI** bridges the gap between self-diagnosis and professional care.

**How it works:**

1. User uploads a clear facial photo
2. AI processes the image in real-time
3. Delivers a **structured, visual-first dermatological report**

### Core Outputs

- **Acne Severity Grading** (Clear → Mild → Moderate → Severe)
- **Lesion Detection** with color-coded bounding boxes
- **Facial Zone Segmentation** (Forehead, Cheeks, Nose, Chin/Jawline)
- **Hyperpigmentation Coverage** estimation with traced overlays
- **Progress Tracking** (Now / Short-term / Long-term)

---

## **New Features**
- **History Tracking**: Keep a record of your skin analysis results.
- **Marketplace**: Browse recommended products based on your skin report.
> These features are currently available on the **`feat1`** branch.

---

## **Unique Selling Points (USPs)**

- **Visual-first interface** - All findings overlaid directly on the user's photo
- **Real-time inference** (< 5 seconds)
- **Non-diagnostic framing** - Medically responsible language
- **Skin-tone inclusive** - Trained across **Fitzpatrick Scale I–VI**
- **Mobile-friendly** web app

---

## **Tech Stack**

| Layer                 | Technology                                       |
| --------------------- | ------------------------------------------------ |
| **Frontend**          | React.js + TailwindCSS                           |
| **Backend**           | FastAPI (Python)                                 |
| **CV Models**         | YOLOv8/v11 (lesion detection), MediaPipe (face mesh) |
| **Segmentation**      | SAM (Segment Anything) / DeepLabv3               |
| **Hyperpigmentation** | OpenCV HSV + Custom Skin Tone Calibration        |
| **LLM Layer**         | Claude (Anthropic) API                           |
| **Deployment**        | Docker + Render / Hugging Face Spaces            |

---

## Core Logic & AI Models

The system employs two distinct computer vision models to perform a granular analysis:

### Facial Region Model (`facial-region-detection-dt8b3`)

- Identifies specific parts of the face:
  - Forehead  
  - Nose  
  - Chin  
  - Cheeks  
- Enables mapping of detected acne to specific clinical regions, which is essential for accurate scoring.

### Acne Detection Model (`acne-vulgaris`)

- Detects and classifies individual acne lesions into four categories:
  - **Comedones** (Blackheads / Whiteheads)
  - **Papules**
  - **Pustules**
  - **Nodules**
  - 
**Pipeline Flow:**

`Image Upload → Preprocessing → Face Mesh → Zone Segmentation → Lesion Detection → 

The system employs advanced computer vision models for granular analysis:

### 1. YOLO Models Integration
SkinSight AI utilizes three specialized YOLO models:
- **`yolo11s-seg.pt`**: Primarily used for high-precision lesion segmentation to identify the exact boundaries of skin issues.
- **`yolov8s-cls.pt`**: Employed for acne severity classification, grading the overall condition into clinical levels (Clear, Mild, Moderate, Severe).
- **`yolov8m.pt`**: Serves as a robust object detection backbone for identifying various skin markers.

### 2. Roboflow & Acne Vulgaris Model
We leverage **Roboflow** for dataset management and model hosting. The **acne-vulgaris** model, trained on thousands of dermatological images, is used to detect and classify individual lesions (Comedones, Papules, Pustules, Nodules). This model is integrated into our pipeline to provide medical-grade detection accuracy.

### 3. Facial Region Detection
- **MediaPipe / Facial-region-detection**: Identifies specific parts of the face (Forehead, Nose, Chin, Cheeks) to map detected lesions to clinical regions for accurate scoring.

**Pipeline Flow:**
`Image Upload → Preprocessing → Face Mesh → Zone Segmentation → Lesion Detection (YOLO/Roboflow) → Hyperpigmentation Analysis → Severity Scoring → Visual Overlays → LLM Summary`

---

## System Architecture

### End-to-End Inference Pipeline

```mermaid
flowchart TD
    A[User Photo Upload]
    --> B[Image Preprocessing<br/>Resize • Normalize • Lighting Correction]

    B --> C[MediaPipe Face Mesh<br/>468 Landmark Detection]

    C --> D[Facial Zone Segmentation<br/>Forehead • Cheeks • Nose • Chin+Jawline]

    D --> E[YOLO/Roboflow Lesion Detection<br/>Bounding Boxes + Classification]
    D --> F[Hyperpigmentation Analysis<br/>OpenCV HSV + Skin Tone Calibration]

    E & F --> G[Severity Scoring Engine<br/>Rule-based + ML Classifier]

    G --> H[Visual Overlay Composer<br/>Annotated Image Generation]

    H --> I[Claude LLM Layer<br/>Report + Recommendations]

    I --> J[Final Structured Report<br/>+ Annotated Photo]

    classDef input fill:#2563EB,stroke:#1E3A8A,color:#fff,rx:20,ry:20
    classDef process fill:#10B981,stroke:#166534,color:#fff,rx:20,ry:20
    classDef core fill:#8B5CF6,stroke:#5B21B6,color:#fff,rx:20,ry:20
    classDef output fill:#F59E0B,stroke:#B45309,color:#fff,rx:20,ry:20

    class A input
    class B,C process
    class D,E,F,G,H core
    class I,J output
```
