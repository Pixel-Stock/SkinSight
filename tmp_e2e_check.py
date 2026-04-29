import mimetypes
import os
import requests

base = "http://127.0.0.1:8000"
health = requests.get(f"{base}/health", timeout=10)
print("HEALTH", health.status_code)

analyze_path = r"C:/Users/Rajesh/downloads/SkinSight/backend/static/Acne-Pimples.jpg"
ctype = mimetypes.guess_type(analyze_path)[0] or "image/jpeg"
with open(analyze_path, "rb") as f:
    files = {"file": (os.path.basename(analyze_path), f, ctype)}
    analysis = requests.post(f"{base}/analyze", files=files, timeout=180).json()

report = requests.post(f"{base}/report", json={"analysis": analysis}, timeout=240).json()
print("REPORT_MODEL", report.get("model"))
print("REPORT_SOURCE", report.get("generated_by"))
