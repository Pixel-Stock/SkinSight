import mimetypes
import os
import requests

base = "http://127.0.0.1:8000"
candidates = [
    r"C:/Users/Rajesh/downloads/SkinSight/backend/test_face.jpg",
    r"C:/Users/Rajesh/downloads/SkinSight/backend/static/Acne-Pimples.jpg",
    r"C:/Users/Rajesh/downloads/SkinSight/backend/static/acne.webp",
    r"C:/Users/Rajesh/downloads/SkinSight/backend/static/teenage-girl-before-after-acne-260nw-1623905551.jpg",
    r"C:/Users/Rajesh/downloads/SkinSight/backend/static/teenage-girl-before-after-acne-260nw-162390551.jpg",
]
for path in candidates:
    ctype = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, ctype)}
        r = requests.post(f"{base}/analyze", files=files, timeout=180)
    payload = r.json()
    detail = payload.get("detail") if isinstance(payload, dict) else None
    print(path, r.status_code, detail if r.status_code != 200 else payload.get("acne_severity"))
