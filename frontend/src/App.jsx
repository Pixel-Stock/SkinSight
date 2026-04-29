import { useEffect, useState, useMemo, useRef } from "react";
import { jsPDF } from "jspdf";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";
const CLIENT_ID_KEY = "skinsight_client_id";

function getClientId() {
  const existing = localStorage.getItem(CLIENT_ID_KEY);
  if (existing) return existing;
  const id =
    (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.floor(Math.random() * 1e9)}`);
  localStorage.setItem(CLIENT_ID_KEY, id);
  return id;
}

const ZONE_LABELS = {
  forehead: "Forehead",
  left_cheek: "Left Cheek",
  right_cheek: "Right Cheek",
  nose: "Nose",
  chin_jawline: "Chin / Jawline",
};

const SEVERITY_CONFIG = {
  Clear: { color: "#d4d4d4", bg: "rgba(212,212,212,0.16)", label: "Clear" },
  Mild: { color: "#a3a3a3", bg: "rgba(163,163,163,0.18)", label: "Mild" },
  Moderate: {
    color: "#737373",
    bg: "rgba(115,115,115,0.2)",
    label: "Moderate",
  },
  Severe: { color: "#525252", bg: "rgba(82,82,82,0.24)", label: "Severe" },
};

function getSeverityConfig(s) {
  const key = s && SEVERITY_CONFIG[s] ? s : "Clear";
  return SEVERITY_CONFIG[key];
}

function Spinner() {
  return (
    <svg className="spinner" viewBox="0 0 24 24" fill="none">
      <circle className="spinner-ring" cx="12" cy="12" r="10" strokeWidth="3" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      className="upload-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
      />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg
      className="camera-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15a2.25 2.25 0 0 0 2.25-2.25V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0ZM18.75 10.5h.008v.008h-.008V10.5Z"
      />
    </svg>
  );
}

function CircularProgress({ percent, color }) {
  const r = 36;
  const circ = 2 * Math.PI * r;
  const offset = circ - (Math.min(percent, 100) / 100) * circ;
  return (
    <svg className="circ-progress" viewBox="0 0 88 88">
      <circle className="circ-track" cx="44" cy="44" r={r} />
      <circle
        className="circ-fill"
        cx="44"
        cy="44"
        r={r}
        stroke={color}
        strokeDasharray={circ}
        strokeDashoffset={offset}
      />
      <text
        className="circ-label"
        x="44"
        y="44"
        textAnchor="middle"
        dominantBaseline="middle"
      >
        {percent.toFixed(1)}%
      </text>
    </svg>
  );
}

function ZoneBar({ label, count, max }) {
  const pct = max > 0 ? (count / max) * 100 : 0;
  return (
    <div className="zone-row">
      <span className="zone-name">{label}</span>
      <div className="zone-track">
        <div className="zone-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="zone-count">{count}</span>
    </div>
  );
}

function DropZone({ file, previewUrl, onFile }) {
  const [dragging, setDragging] = useState(false);
  const ref = useRef(null);
  return (
    <div
      className={`drop-zone ${dragging ? "drag-over" : ""} ${file ? "has-file" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer?.files?.[0];
        if (f && f.type.startsWith("image/")) onFile(f);
      }}
      onClick={() => ref.current?.click()}
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && ref.current?.click()}
    >
      <input
        ref={ref}
        type="file"
        accept="image/*"
        className="hidden-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      {file && previewUrl ? (
        <div className="drop-preview">
          <img src={previewUrl} alt="preview" className="preview-img" />
          <div className="drop-overlay">
            <UploadIcon />
            <span>Change photo</span>
          </div>
        </div>
      ) : (
        <div className="drop-placeholder">
          <UploadIcon />
          <p className="drop-title">
            Drag &amp; drop or <span className="drop-link">browse</span>
          </p>
          <p className="drop-hint">
            JPG, PNG, WebP &mdash; front-facing selfie
          </p>
        </div>
      )}
    </div>
  );
}

function CameraCapture({ onCapture, onError }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [cameraError, setCameraError] = useState("");

  useEffect(() => {
    async function startCamera() {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        setStream(s);
        if (videoRef.current) videoRef.current.srcObject = s;
      } catch (err) {
        console.error("Camera error:", err);
        setCameraError("Unable to access camera. Please ensure permissions are granted.");
        if (onError) onError(err);
      }
    }
    startCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const capture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas) {
      const context = canvas.getContext("2d");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Mirror if needed, matching the video scaleX(-1)
      context.translate(canvas.width, 0);
      context.scale(-1, 1);
      
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
          onCapture(file);
        }
      }, "image/jpeg", 0.95);
    }
  };

  if (cameraError) {
      return (
          <div className="camera-capture">
              <div className="camera-error">
                  <CameraIcon />
                  <p>{cameraError}</p>
              </div>
          </div>
      );
  }

  return (
    <div className="camera-capture">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="camera-video"
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <div className="camera-overlay">
        <button type="button" className="capture-btn" onClick={capture} title="Capture Photo">
          <div className="capture-inner" />
        </button>
      </div>
    </div>
  );
}

function MiniDrop({ label, file, previewUrl, onFile }) {
  const ref = useRef(null);
  const [dragging, setDragging] = useState(false);
  return (
    <div
      className={`mini-drop ${dragging ? "drag-over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer?.files?.[0];
        if (f && f.type.startsWith("image/")) onFile(f);
      }}
      onClick={() => ref.current?.click()}
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && ref.current?.click()}
    >
      <input
        ref={ref}
        type="file"
        accept="image/*"
        className="hidden-input"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      {file && previewUrl ? (
        <div className="mini-preview">
          <img src={previewUrl} alt={label} />
          <span className="mini-change">Tap to change</span>
        </div>
      ) : (
        <div className="mini-placeholder">
          <UploadIcon />
          <span>{label}</span>
        </div>
      )}
    </div>
  );
}

function StatCard({ icon, label, value, sub, accent }) {
  return (
    <div className="stat-card" style={{ "--accent": accent }}>
      <span className="stat-icon">{icon}</span>
      <div className="stat-body">
        <span className="stat-label">{label}</span>
        <span className="stat-value">{value}</span>
        {sub && <span className="stat-sub">{sub}</span>}
      </div>
    </div>
  );
}

const REPORT_QUOTES = [
  "Healthy skin is a reflection of overall wellness.",
  "Consistency beats intensity in skincare routines.",
  "Gentle care today prevents irritation tomorrow.",
  "Progress photos tell the story better than memory.",
];

function AuthGate({ onSuccess }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");

  const onSubmit = (e) => {
    e.preventDefault();
    setError("");
    if (!email || !password || (mode === "register" && !name)) {
      setError("Please fill all required fields.");
      return;
    }
    if (mode === "register" && password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    onSuccess({ name, email, mode });
  };

  return (
    <div className="auth-page">
      <div className="auth-wave auth-wave-a" />
      <div className="auth-wave auth-wave-b" />
      <div className="auth-panel">
        <p className="auth-kicker">SkinSight Access</p>
        <h1 className="auth-title">{mode === "login" ? "Welcome Back" : "Create Account"}</h1>
        <p className="auth-sub">
          {mode === "login"
            ? "Sign in to continue to your skin analysis dashboard."
            : "Register to save your tracking timeline and reports."}
        </p>

        <form onSubmit={onSubmit} className="auth-form">
          {mode === "register" && (
            <label className="auth-field">
              <span>Full Name</span>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
              />
            </label>
          )}
          <label className="auth-field">
            <span>Email</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="name@example.com"
            />
          </label>
          <label className="auth-field">
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </label>
          {mode === "register" && (
            <label className="auth-field">
              <span>Confirm Password</span>
              <input
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                placeholder="••••••••"
              />
            </label>
          )}

          {error && <div className="auth-error">{error}</div>}

          <button type="submit" className="landing-btn auth-submit">
            {mode === "login" ? "Login" : "Register"}
          </button>
        </form>

        <button
          className="auth-switch"
          onClick={() => {
            setMode(mode === "login" ? "register" : "login");
            setError("");
          }}
        >
          {mode === "login"
            ? "New here? Create an account"
            : "Already have an account? Login"}
        </button>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════
   ANALYZE TAB
══════════════════════════════════════════════ */
function AnalyzeTab({ forcedView = "annotated" }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [detailedReport, setDetailedReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [reportQuoteIndex, setReportQuoteIndex] = useState(0);
  const [imageSubView, setImageSubView] = useState("annotated");
  const [showResults, setShowResults] = useState(false);
  const [heatmapOpen, setHeatmapOpen] = useState(false);
  const [heatmapZoom, setHeatmapZoom] = useState(1);
  const [sourceMode, setSourceMode] = useState("upload"); // "upload" or "camera"

  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  );

  const normalizedReportText = useMemo(() => {
    const raw = detailedReport?.report ?? "";
    return raw
      .replace(/^\s*\*\*([^*]+)\*\*\s*$/gm, "$1")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/^\s*\*\s+/gm, "- ")
      .replace(/\r\n/g, "\n")
      .trim();
  }, [detailedReport]);

  const summaryText = useMemo(() => {
    const raw = String(result?.summary || "").trim().replace(/^['"]+|['"]+$/g, "");
    if (!raw) return "Analysis complete.";
    const sentence = raw.split(/(?<=[.!?])\s+/)[0] || raw;
    return sentence.length > 180 ? `${sentence.slice(0, 177)}...` : sentence;
  }, [result]);

  const lesionCount = result?.lesions?.length ?? 0;
  const topZone = useMemo(() => {
    const entries = Object.entries(result?.zone_counts ?? {});
    if (!entries.length) return "N/A";
    return entries.sort((a, b) => b[1] - a[1])[0][0];
  }, [result]);

  const closeHeatmapModal = () => {
    setHeatmapOpen(false);
    setHeatmapZoom(1);
  };

  const adjustHeatmapZoom = (delta) => {
    setHeatmapZoom((z) => Math.min(3, Math.max(0.7, +(z + delta).toFixed(2))));
  };

  const resetHeatmapZoom = () => setHeatmapZoom(1);

  const exportReportPdf = () => {
    if (!detailedReport?.report) return;
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 42;
    const usableWidth = pageWidth - margin * 2;
    let y = margin;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.text("SkinSight AI Report", margin, y);
    y += 22;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.text(`Model: ${detailedReport.model}`, margin, y);
    doc.text(`Source: ${detailedReport.generated_by}`, pageWidth - margin, y, {
      align: "right",
    });
    y += 18;

    doc.setDrawColor(190);
    doc.line(margin, y, pageWidth - margin, y);
    y += 16;

    doc.setFontSize(11);
    const lines = doc.splitTextToSize(normalizedReportText, usableWidth);
    for (const line of lines) {
      if (y > pageHeight - margin) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += 16;
    }

    const safeStamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
    doc.save(`skinsight-report-${safeStamp}.pdf`);
    setActionMessage("Report exported as PDF.");
  };

  const copyReportText = async () => {
    if (!normalizedReportText) return;
    await navigator.clipboard.writeText(normalizedReportText);
    setActionMessage("Report copied to clipboard.");
  };

  const copySummaryText = async () => {
    if (!result) return;
    const lines = [
      `Severity: ${result.acne_severity}`,
      `Lesions: ${lesionCount}`,
      `Top zone: ${topZone}`,
      `Hyperpigmentation: ${Number(result.hyperpigmentation?.coverage_percent ?? 0).toFixed(1)}% (${result.hyperpigmentation?.severity ?? "N/A"})`,
      `Summary: ${summaryText}`,
    ];
    await navigator.clipboard.writeText(lines.join("\n"));
    setActionMessage("Summary copied to clipboard.");
  };

  const downloadAnalysisJson = () => {
    if (!result) return;
    const payload = {
      analysis: result,
      report: detailedReport ?? null,
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "skinsight-analysis.json";
    a.click();
    URL.revokeObjectURL(url);
    setActionMessage("Analysis JSON downloaded.");
  };

  const downloadHeatmapImage = () => {
    if (!result?.heatmap_image_base64) return;
    const a = document.createElement("a");
    a.href = `data:image/jpeg;base64,${result.heatmap_image_base64}`;
    a.download = `skinsight-heatmap-${Date.now()}.jpg`;
    a.click();
    setActionMessage("Heatmap image downloaded.");
  };

  useEffect(() => {
    setImageSubView(forcedView);
  }, [forcedView]);

  useEffect(() => {
    if (!reportLoading) return undefined;
    const quoteTimer = setInterval(() => {
      setReportQuoteIndex((i) => (i + 1) % REPORT_QUOTES.length);
    }, 2400);
    return () => clearInterval(quoteTimer);
  }, [reportLoading]);

  useEffect(() => {
    if (!heatmapOpen) return undefined;
    const onKeyDown = (evt) => {
      if (evt.key === "Escape") closeHeatmapModal();
    };
    const oldOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = oldOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [heatmapOpen]);

  const onAnalyze = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please upload a selfie first.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    setDetailedReport(null);
    setReportError("");
    setImageSubView(forcedView || "annotated");
    setReportQuoteIndex(0);
    setShowResults(false);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "X-Client-Id": getClientId() },
        body: fd,
      });
      if (!res.ok) {
        const p = await res.json().catch(() => ({}));
        throw new Error(p.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      setResult(data);

      setReportLoading(true);
      try {
        const reportRes = await fetch(`${API_BASE}/report`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Client-Id": getClientId(),
          },
          body: JSON.stringify({ analysis: data }),
        });
        if (!reportRes.ok) {
          const p = await reportRes.json().catch(() => ({}));
          throw new Error(p.detail || `Server error ${reportRes.status}`);
        }
        setDetailedReport(await reportRes.json());
      } catch (reportErr) {
        setReportError(reportErr.message || "Detailed report generation failed.");
      } finally {
        setReportLoading(false);
      }

      setTimeout(() => setShowResults(true), 50);
    } catch (err) {
      const isNetworkError =
        err instanceof TypeError &&
        String(err.message || "").toLowerCase().includes("fetch");
      setError(
        isNetworkError
          ? "Cannot connect to backend API on port 8000. Start backend and try again."
          : err.message || "Unexpected error — please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  const sevConfig = result ? getSeverityConfig(result.acne_severity) : null;
  const zoneCounts = result?.zone_counts ?? {};
  const maxZone = Math.max(0, ...Object.values(zoneCounts));
  const hyper = result?.hyperpigmentation ?? {};
  const hyperPct = parseFloat(hyper.coverage_percent ?? 0);
  // acne_score is [0..1]; display as percentage
  const acneScorePct = ((result?.acne_score ?? 0) * 100).toFixed(1);

  return (
    <div className="tab-content">
      {/* ── Upload + Annotated side-by-side ── */}
      <div className="analyze-top-row">
        <section className="card upload-card">
          <div className="card-header">
            <h2 className="card-title">Add Your Photo</h2>
            <p className="card-sub">
              Well-lit, front-facing selfie for best results
            </p>
          </div>
          
          <div className="upload-methods">
            <button 
              type="button" 
              className={`method-btn ${sourceMode === "upload" ? "active" : ""}`}
              onClick={() => setSourceMode("upload")}
            >
              <UploadIcon />
              <span>File Upload</span>
            </button>
            <button 
              type="button" 
              className={`method-btn ${sourceMode === "camera" ? "active" : ""}`}
              onClick={() => setSourceMode("camera")}
            >
              <CameraIcon />
              <span>Take Photo</span>
            </button>
          </div>

          <form onSubmit={onAnalyze}>
            {sourceMode === "upload" ? (
              <DropZone file={file} previewUrl={previewUrl} onFile={setFile} />
            ) : (
              <CameraCapture onCapture={(f) => { setFile(f); setSourceMode("upload"); }} />
            )}
            {error && (
              <div className="alert alert-error" role="alert">
                <svg
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="alert-icon"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-8-5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 10a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
                    clipRule="evenodd"
                  />
                </svg>
                {error}
              </div>
            )}
            <button
              type="submit"
              className="btn-primary btn-large"
              disabled={loading || !file}
            >
              {loading ? (
                <>
                  <Spinner /> Analyzing&hellip;
                </>
              ) : (
                <>
                  <svg
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    className="btn-icon"
                  >
                    <path d="M10 12.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
                    <path
                      fillRule="evenodd"
                      d="M.664 10.59a1.651 1.651 0 0 1 0-1.186A10.004 10.004 0 0 1 10 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0 1 10 17c-4.257 0-7.893-2.66-9.336-6.41ZM14 10a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z"
                      clipRule="evenodd"
                    />
                  </svg>
                  Analyze Skin
                </>
              )}
            </button>
          </form>
        </section>

        {result && (
          <div
            className={`card annotated-side-card ${showResults ? "results-visible" : ""}`}
          >
            <div className="card-header">
              <h2 className="card-title">AI Annotated</h2>
              <div className="image-subtabs">
                <button
                  className={`image-subtab ${imageSubView === "annotated" ? "image-subtab-active" : ""}`}
                  onClick={() => setImageSubView("annotated")}
                  type="button"
                >
                  Annotated
                </button>
                <button
                  className={`image-subtab ${imageSubView === "heatmap" ? "image-subtab-active" : ""}`}
                  onClick={() => setImageSubView("heatmap")}
                  type="button"
                >
                  Heatmap
                </button>
                <button
                  className={`image-subtab ${imageSubView === "original" ? "image-subtab-active" : ""}`}
                  onClick={() => setImageSubView("original")}
                  type="button"
                >
                  Original
                </button>
              </div>
            </div>
            <div className="image-frame annotated-frame annotated-side-frame">
              {(imageSubView === "annotated" && result.annotated_image_base64) ||
              (imageSubView === "heatmap" && result.heatmap_image_base64) ||
              (imageSubView === "original" && previewUrl) ? (
                <>
                  {imageSubView === "heatmap" && result.heatmap_image_base64 && (
                    <button
                      className="image-popout-btn"
                      type="button"
                      onClick={() => setHeatmapOpen(true)}
                    >
                      Expand
                    </button>
                  )}
                  <img
                    src={
                      imageSubView === "heatmap"
                        ? `data:image/jpeg;base64,${result.heatmap_image_base64}`
                        : imageSubView === "original"
                          ? previewUrl
                          : `data:image/jpeg;base64,${result.annotated_image_base64}`
                    }
                    alt={
                      imageSubView === "heatmap"
                        ? "Lesion heatmap"
                        : imageSubView === "original"
                          ? "Original upload"
                          : "Annotated"
                    }
                  />
                </>
              ) : (
                <div className="img-placeholder">Processing&hellip;</div>
              )}
            </div>
          </div>
        )}

        {heatmapOpen && result?.heatmap_image_base64 && (
          <div className="heatmap-modal" role="dialog" aria-modal="true" aria-label="Lesion heatmap preview">
            <button className="heatmap-modal-backdrop" type="button" onClick={closeHeatmapModal} />
            <div className="heatmap-modal-panel">
              <div className="heatmap-modal-header">
                <div>
                  <p className="report-doc-kicker">Lesion Heatmap</p>
                  <h4>Full-size focus view</h4>
                </div>
                <div className="heatmap-toolbar">
                  <button className="report-action-btn" type="button" onClick={() => adjustHeatmapZoom(-0.15)}>
                    -
                  </button>
                  <span className="heatmap-zoom-label">{Math.round(heatmapZoom * 100)}%</span>
                  <button className="report-action-btn" type="button" onClick={() => adjustHeatmapZoom(0.15)}>
                    +
                  </button>
                  <button className="report-action-btn" type="button" onClick={resetHeatmapZoom}>
                    Reset
                  </button>
                  <button className="report-action-btn" type="button" onClick={downloadHeatmapImage}>
                    Download
                  </button>
                  <button className="report-action-btn" type="button" onClick={closeHeatmapModal}>
                    Close
                  </button>
                </div>
              </div>
              <div className="heatmap-modal-frame">
                <img
                  className="heatmap-modal-image"
                  style={{ transform: `scale(${heatmapZoom})` }}
                  src={`data:image/jpeg;base64,${result.heatmap_image_base64}`}
                  alt="Enlarged lesion heatmap"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Results ── */}
      {result && (
        <div
          className={`results-wrapper ${showResults ? "results-visible" : ""}`}
        >
          {/* Severity Banner */}
          <div
            className="severity-banner"
            style={{ "--sev-color": sevConfig.color, "--sev-bg": sevConfig.bg }}
          >
            <div className="sev-left">
              <span
                className="sev-badge"
                style={{
                  color: sevConfig.color,
                  background: sevConfig.bg,
                  borderColor: sevConfig.color,
                }}
              >
                <span
                  className="sev-dot"
                  style={{ background: sevConfig.color }}
                />
                {sevConfig.label}
              </span>
              <div>
                <p className="sev-title">Analysis Complete</p>
                <p className="sev-sub">
                  Acne score: <strong>{acneScorePct}%</strong>
                </p>
              </div>
            </div>
            {/* ring uses acne_score * 360 — score is [0,1] */}
            <div
              className="sev-score-ring"
              style={{
                background: `conic-gradient(${sevConfig.color} ${(result.acne_score ?? 0) * 360}deg, #1e1e2e 0deg)`,
              }}
            >
              <span>{acneScorePct}%</span>
            </div>
          </div>

          {/* 4-col Stat Cards */}
          <div className="stats-row">
            <StatCard
              icon="AS"
              label="Acne Severity"
              value={result.acne_severity ?? "—"}
              accent={sevConfig.color}
            />
            <StatCard
              icon="LC"
              label="Lesion Count"
              value={result.lesions?.length ?? 0}
              sub="detected lesions"
              accent="#e5e5e5"
            />
            <StatCard
              icon="HP"
              label="Hyperpigmentation"
              value={`${hyperPct.toFixed(1)}%`}
              sub={hyper.severity ?? ""}
              accent="#a3a3a3"
            />
            <StatCard
              icon="MZ"
              label="Most Affected Zone"
              accent="#737373"
              value={
                Object.entries(zoneCounts).sort((a, b) => b[1] - a[1])[0]
                  ? ZONE_LABELS[
                      Object.entries(zoneCounts).sort(
                        (a, b) => b[1] - a[1],
                      )[0][0]
                    ] ||
                    Object.entries(zoneCounts).sort((a, b) => b[1] - a[1])[0][0]
                  : "—"
              }
            />
          </div>

          {/* Zone breakdown + Hyperpigmentation */}
          <div className="detail-row">
            <div className="card detail-card">
              <h3 className="detail-title">
                <svg
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="detail-icon"
                >
                  <path d="M15.5 2A1.5 1.5 0 0 0 14 3.5v13a1.5 1.5 0 0 0 3 0v-13A1.5 1.5 0 0 0 15.5 2ZM9.5 6A1.5 1.5 0 0 0 8 7.5v9a1.5 1.5 0 0 0 3 0v-9A1.5 1.5 0 0 0 9.5 6ZM3.5 10A1.5 1.5 0 0 0 2 11.5v5a1.5 1.5 0 0 0 3 0v-5A1.5 1.5 0 0 0 3.5 10Z" />
                </svg>
                Zone Breakdown
              </h3>
              <div className="zones">
                {Object.entries(ZONE_LABELS).map(([key, label]) => (
                  <ZoneBar
                    key={key}
                    label={label}
                    count={zoneCounts[key] ?? 0}
                    max={maxZone}
                  />
                ))}
              </div>
            </div>

            <div className="card detail-card hyper-card">
              <h3 className="detail-title">
                <svg
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="detail-icon"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm.75-11.25a.75.75 0 0 0-1.5 0v2.5h-2.5a.75.75 0 0 0 0 1.5h2.5v2.5a.75.75 0 0 0 1.5 0v-2.5h2.5a.75.75 0 0 0 0-1.5h-2.5v-2.5Z"
                    clipRule="evenodd"
                  />
                </svg>
                Hyperpigmentation
              </h3>
              <div className="hyper-body">
                <CircularProgress
                  percent={hyperPct}
                  color={
                    hyperPct > 20
                      ? "#525252"
                      : hyperPct > 10
                        ? "#737373"
                        : "#d4d4d4"
                  }
                />
                <div className="hyper-info">
                  <div className="hyper-stat">
                    <span className="hyper-stat-label">Coverage</span>
                    <span className="hyper-stat-val">
                      {hyperPct.toFixed(1)}%
                    </span>
                  </div>
                  <div className="hyper-stat">
                    <span className="hyper-stat-label">Severity</span>
                    <span className="hyper-stat-val">
                      {hyper.severity ?? "—"}
                    </span>
                  </div>
                  <div className="hyper-gauge-row">
                    {["Low", "Moderate", "High"].map((l, i) => (
                      <div
                        key={l}
                        className={`hyper-gauge-seg ${i === 0 ? "seg-green" : i === 1 ? "seg-amber" : "seg-red"}`}
                      >
                        <span>{l}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Summary */}
          {result.summary && (
            <div className="card summary-card">
              <h3 className="detail-title">
                <svg
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="detail-icon"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-7-4a1 1 0 1 1-2 0 1 1 0 0 1 2 0ZM9 9a.75.75 0 0 0 0 1.5h.253a.25.25 0 0 1 .244.304l-.459 2.066A1.75 1.75 0 0 0 10.747 15H11a.75.75 0 0 0 0-1.5h-.253a.25.25 0 0 1-.244-.304l.459-2.066A1.75 1.75 0 0 0 9.253 9H9Z"
                    clipRule="evenodd"
                  />
                </svg>
                AI Summary
              </h3>
              <p className="summary-lead">{summaryText}</p>
              <div className="summary-facts">
                <span className="summary-fact">Severity: {result.acne_severity}</span>
                <span className="summary-fact">Lesions: {lesionCount}</span>
                <span className="summary-fact">Top zone: {topZone}</span>
                <span className="summary-fact">
                  Heatmap: {Number(result.hyperpigmentation?.coverage_percent ?? 0).toFixed(1)}% / {result.hyperpigmentation?.severity || "N/A"}
                </span>
              </div>
              <div className="summary-actions">
                <button className="report-action-btn" type="button" onClick={copySummaryText}>
                  Copy Summary
                </button>
                <button
                  className="report-action-btn"
                  type="button"
                  onClick={() => setHeatmapOpen(true)}
                  disabled={!result.heatmap_image_base64}
                >
                  Maximize Heatmap
                </button>
              </div>
              <div className="disclaimer">
                <svg
                  viewBox="0 0 16 16"
                  fill="currentColor"
                  className="disclaimer-icon"
                >
                  <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1Zm0 11a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5Zm.75-3.75a.75.75 0 0 1-1.5 0V5.75a.75.75 0 0 1 1.5 0v2.5Z" />
                </svg>
                This analysis is for informational purposes only and does not
                constitute medical advice. Consult a dermatologist for clinical
                diagnosis.
              </div>
            </div>
          )}

          <div className="card summary-card">
            <h3 className="detail-title">
              <svg
                viewBox="0 0 20 20"
                fill="currentColor"
                className="detail-icon"
              >
                <path
                  fillRule="evenodd"
                  d="M4.25 3A2.25 2.25 0 0 0 2 5.25v9.5A2.25 2.25 0 0 0 4.25 17h11.5A2.25 2.25 0 0 0 18 14.75v-9.5A2.25 2.25 0 0 0 15.75 3H4.25ZM5.5 6.75a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5a.75.75 0 0 1-.75-.75Zm0 3a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5a.75.75 0 0 1-.75-.75Zm0 3a.75.75 0 0 1 .75-.75h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1-.75-.75Z"
                  clipRule="evenodd"
                />
              </svg>
              Detailed AI Report
            </h3>
            {reportLoading && (
              <div className="report-skeleton">
                <div className="report-loading-row">
                  <Spinner />
                  <span>Preparing your dermatology-style report document...</span>
                </div>
                <blockquote className="report-quote">
                  "{REPORT_QUOTES[reportQuoteIndex]}"
                </blockquote>
                <div className="report-shimmer-line" />
                <div className="report-shimmer-line" />
                <div className="report-shimmer-line short" />
              </div>
            )}
            {!reportLoading && reportError && (
              <div className="alert alert-error" role="alert">
                <svg
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="alert-icon"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-8-5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 10a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
                    clipRule="evenodd"
                  />
                </svg>
                {reportError}
              </div>
            )}
            {!reportLoading && detailedReport?.report && (
              <article className="report-document">
                <header className="report-doc-header">
                  <div>
                    <p className="report-doc-kicker">SkinSight Report Document</p>
                    <h4>AI Skin Condition Guidance</h4>
                  </div>
                  <div className="report-doc-meta">
                    <span>Model: {detailedReport.model}</span>
                    <span>Source: {detailedReport.generated_by}</span>
                  </div>
                </header>
                <div className="report-actions">
                  <button className="report-action-btn" type="button" onClick={exportReportPdf}>
                    Export PDF
                  </button>
                  <button className="report-action-btn" type="button" onClick={copyReportText}>
                    Copy Text
                  </button>
                  <button className="report-action-btn" type="button" onClick={downloadAnalysisJson}>
                    Download JSON
                  </button>
                </div>
                {!!actionMessage && <p className="report-action-note">{actionMessage}</p>}
                <pre className="detailed-report-text">{normalizedReportText}</pre>
                <div className="disclaimer">{detailedReport.disclaimer}</div>
              </article>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════
   PROGRESS TRACKING TAB
══════════════════════════════════════════════ */
function TrackTab() {
  const [beforeFile, setBeforeFile] = useState(null);
  const [afterFile, setAfterFile] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [compareResult, setCompareResult] = useState(null);
  const [compareError, setCompareError] = useState("");

  const beforeUrl = useMemo(
    () => (beforeFile ? URL.createObjectURL(beforeFile) : null),
    [beforeFile],
  );
  const afterUrl = useMemo(
    () => (afterFile ? URL.createObjectURL(afterFile) : null),
    [afterFile],
  );

  const onCompare = async () => {
    if (!beforeFile || !afterFile) {
      setCompareError("Upload both scans first.");
      return;
    }
    setComparing(true);
    setCompareError("");
    setCompareResult(null);
    try {
      const fd = new FormData();
      fd.append("baseline", beforeFile); // BEFORE = acne skin
      fd.append("followup", afterFile); // AFTER  = clearer skin
      const res = await fetch(`${API_BASE}/track`, {
        method: "POST",
        headers: { "X-Client-Id": getClientId() },
        body: fd,
      });
      if (!res.ok) {
        const p = await res.json().catch(() => ({}));
        throw new Error(p.detail || `Server error ${res.status}`);
      }
      setCompareResult(await res.json());
    } catch (err) {
      setCompareError(err.message || "Comparison failed.");
    } finally {
      setComparing(false);
    }
  };

  const improvement = compareResult?.improvement_percent ?? 0;
  const improved = improvement > 0;

  return (
    <div className="tab-content">
      <div className="card track-card-full">
        <div className="card-header">
          <h2 className="card-title">Progress Tracking</h2>
          <p className="card-sub">
            Upload a <strong>before</strong> photo (with acne) on the left and
            an <strong>after</strong> photo (clearer skin) on the right to
            measure your improvement
          </p>
        </div>

        {/* Before ──→ After upload row */}
        <div className="track-uploads">
          {/* BEFORE */}
          <div className="track-upload-slot">
            <p className="slot-label">
              <span className="slot-num before-num">BEFORE</span>
              Skin with Acne
            </p>
            <MiniDrop
              label="Upload Before Photo"
              file={beforeFile}
              previewUrl={beforeUrl}
              onFile={setBeforeFile}
            />
            {compareResult && (
              <div className="slot-count">
                <span className="slot-count-num bad">
                  {compareResult.baseline_lesions}
                </span>
                <span className="slot-count-label">lesions</span>
              </div>
            )}
          </div>

          {/* Arrow */}
          <div className="track-arrow">
            <svg viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M3 10a.75.75 0 0 1 .75-.75h10.638L10.23 5.29a.75.75 0 1 1 1.04-1.08l5.5 5.25a.75.75 0 0 1 0 1.08l-5.5 5.25a.75.75 0 1 1-1.04-1.08l4.158-3.96H3.75A.75.75 0 0 1 3 10Z"
                clipRule="evenodd"
              />
            </svg>
            <span className="track-arrow-label">Progress</span>
          </div>

          {/* AFTER */}
          <div className="track-upload-slot">
            <p className="slot-label">
              <span className="slot-num after-num">AFTER</span>
              Clearer Skin
            </p>
            <MiniDrop
              label="Upload After Photo"
              file={afterFile}
              previewUrl={afterUrl}
              onFile={setAfterFile}
            />
            {compareResult && (
              <div className="slot-count">
                <span className={`slot-count-num ${improved ? "good" : "bad"}`}>
                  {compareResult.followup_lesions}
                </span>
                <span className="slot-count-label">lesions</span>
              </div>
            )}
          </div>
        </div>

        {compareError && (
          <div className="alert alert-error">
            <svg viewBox="0 0 20 20" fill="currentColor" className="alert-icon">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-8-5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0v-4.5A.75.75 0 0 1 10 5Zm0 10a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
                clipRule="evenodd"
              />
            </svg>
            {compareError}
          </div>
        )}

        <button
          className="btn-primary"
          onClick={onCompare}
          disabled={comparing || !beforeFile || !afterFile}
        >
          {comparing ? (
            <>
              <Spinner /> Comparing&hellip;
            </>
          ) : (
            "Compare Progress"
          )}
        </button>

        {compareResult && (
          <div className="compare-results">
            {/* 4-col metrics */}
            <div className="compare-metrics">
              <div className="compare-metric">
                <span className="cm-label">Before Lesions</span>
                <span className="cm-value accent-red">
                  {compareResult.baseline_lesions}
                </span>
              </div>
              <div className="compare-metric">
                <span className="cm-label">After Lesions</span>
                <span
                  className={`cm-value ${improved ? "accent-green" : "accent-red"}`}
                >
                  {compareResult.followup_lesions}
                </span>
              </div>
              <div className="compare-metric">
                <span className="cm-label">Improvement</span>
                <span
                  className={`cm-value ${improved ? "accent-green" : "accent-red"}`}
                >
                  {improvement > 0 ? "+" : ""}
                  {improvement.toFixed(1)}%
                </span>
              </div>
              <div className="compare-metric">
                <span className="cm-label">Image Similarity</span>
                <span className="cm-value accent-purple">
                  {(compareResult.similarity * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Trend bar */}
            <div className="trend-bar-wrap">
              <div className="trend-bar-track">
                <div
                  className={`trend-bar-fill ${improved ? "trend-good" : "trend-bad"}`}
                  style={{ width: `${Math.min(Math.abs(improvement), 100)}%` }}
                />
              </div>
              <span
                className={`trend-label ${improved ? "accent-green" : "accent-red"}`}
              >
                {improvement > 10
                  ? `${improvement.toFixed(1)}% improvement`
                  : improvement < -10
                    ? `${Math.abs(improvement).toFixed(1)}% worsening`
                    : "Stable — minimal change"}
              </span>
            </div>

            {compareResult.summary && (
              <div className="compare-summary">{compareResult.summary}</div>
            )}

            {Array.isArray(compareResult.stages) && compareResult.stages.length > 0 && (
              <div className="progress-stages">
                {compareResult.stages.map((stage) => (
                  <div key={stage.key} className="progress-stage-card">
                    <h4>{stage.title}</h4>
                    <ul>
                      {stage.bullets.map((item, idx) => (
                        <li key={`${stage.key}-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════
   ROOT APP
══════════════════════════════════════════════ */
export default function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [analyzeViewPreset, setAnalyzeViewPreset] = useState("annotated");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showProduct, setShowProduct] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      setShowProduct(false);
      return undefined;
    }
    const timer = setTimeout(() => setShowProduct(true), 1800);
    return () => clearTimeout(timer);
  }, [isAuthenticated]);

  if (!isAuthenticated) {
    return <AuthGate onSuccess={() => setIsAuthenticated(true)} />;
  }

  if (!showProduct) {
    return (
      <div className="landing-page">
        <div className="landing-panel">
          <p className="landing-kicker">AI Facial Screening</p>
          <h1 className="landing-title">SkinSight</h1>
          <p className="landing-sub">
            Clinical-style visual analysis with acne grading, lesion mapping,
            and detailed AI guidance.
          </p>
          <button className="landing-btn" onClick={() => setShowProduct(true)}>
            Enter Product
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />

      <div className="container">
        <nav className="top-nav">
          <div className="top-nav-brand">SkinSight</div>
          <div className="top-nav-actions">
            <button
              className={`top-nav-btn ${activeTab === "analyze" && analyzeViewPreset !== "heatmap" ? "top-nav-btn-active" : ""}`}
              onClick={() => {
                setActiveTab("analyze");
                setAnalyzeViewPreset("annotated");
              }}
            >
              Skin Analysis
            </button>
            <button
              className={`top-nav-btn ${activeTab === "track" ? "top-nav-btn-active" : ""}`}
              onClick={() => setActiveTab("track")}
            >
              Progress Tracking
            </button>
            <button
              className={`top-nav-btn ${activeTab === "analyze" && analyzeViewPreset === "heatmap" ? "top-nav-btn-active" : ""}`}
              onClick={() => {
                setActiveTab("analyze");
                setAnalyzeViewPreset("heatmap");
              }}
            >
              Lesion Heatmap
            </button>
          </div>
        </nav>

        {/* Hero */}
        <header className="hero">
          <div className="hero-layout">
            <div className="hero-main">
              <div className="hero-badge">
                <span className="badge-dot" />
                AI-Powered Analysis
              </div>
              <h1 className="hero-title">
                <span className="gradient-text">SkinSight</span>
                <span className="hero-title-white"> AI</span>
              </h1>
              <p className="hero-sub">
                Dermatological-grade skin analysis in seconds
              </p>
              <div className="hero-features">
                {[
                  "Acne Detection",
                  "Zone Mapping",
                  "Hyperpigmentation",
                  "Progress Tracking",
                ].map((f) => (
                  <span key={f} className="feature-pill">
                    {f}
                  </span>
                ))}
              </div>
            </div>

            <aside className="hero-side-cards">
              <div className="hero-side-card">
                <p>AI Report</p>
                <strong>Structured + Exportable</strong>
              </div>
              <div className="hero-side-card">
                <p>Image Views</p>
                <strong>Annotated / Heatmap / Original</strong>
              </div>
              <div className="hero-side-card">
                <p>Tracking</p>
                <strong>Now, Short Term, Long Term</strong>
              </div>
            </aside>
          </div>
        </header>

        {/* Active tab */}
        {activeTab === "analyze" ? (
          <AnalyzeTab forcedView={analyzeViewPreset} />
        ) : (
          <TrackTab />
        )}

        <footer className="footer">
          <p>
            SkinSight AI &mdash; Hackathon Build &mdash; Not for clinical use
          </p>
        </footer>
      </div>
    </div>
  );
}
