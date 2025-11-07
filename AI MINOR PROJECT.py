import os
import base64
import csv
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify, render_template, send_file
from deepface import DeepFace
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

cascade_path = "haarcascade_frontalface_default.xml"
if os.path.exists(cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/emotions.csv"

def ensure_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_id", "role", "emotion", "confidence", "engagement"])

def map_emotion_to_engagement(emotion):
    m = {
        "happy": 0.92, "excited": 0.95, "surprise": 0.82,
        "neutral": 0.6, "sad": 0.32, "angry": 0.18,
        "fear": 0.25, "disgust": 0.2
    }
    return float(m.get(emotion.lower(), 0.5))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"status": "error", "message": "no image"}), 400
    image_data = data["image"]
    user_id = data.get("user_id", "anonymous")
    role = data.get("role", "student")

    if "," in image_data:
        header, b64 = image_data.split(",", 1)
    else:
        b64 = image_data
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        return jsonify({"status": "error", "message": "invalid base64"}), 400

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"status": "error", "message": "could not decode image"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        pad = int(0.2 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        face_img = img[y1:y2, x1:x2]
    else:
        face_img = img

    try:
        result = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False, detector_backend="opencv")
    except Exception as e:
        result = {"dominant_emotion": "unknown", "emotion": {}}

    res = result[0] if isinstance(result, list) else result
    if "dominant_emotion" in res:
        dominant = res["dominant_emotion"]
    elif "emotion" in res and isinstance(res["emotion"], dict) and res["emotion"]:
        dominant = max(res["emotion"], key=res["emotion"].get)
    else:
        dominant = "unknown"

    confidence = None
    if "emotion" in res and isinstance(res["emotion"], dict):
        confidence = float(res["emotion"].get(dominant, 0.0))
    else:
        confidence = float(res.get("confidence", 0.0) or 0.0)

    engagement = map_emotion_to_engagement(dominant)
    timestamp = datetime.utcnow().isoformat()

    ensure_log()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_id, role, dominant, confidence, engagement])

    return jsonify({"status": "ok", "emotion": dominant, "confidence": confidence, "engagement": engagement})

@app.route("/data")
def data():
    if not os.path.exists(LOG_FILE):
        return jsonify({"emotion_counts": {}, "avg_engagement_by_user": [], "recent": []})
    df = pd.read_csv(LOG_FILE)
    last = df.tail(1000)
    emotion_counts = last["emotion"].value_counts().to_dict()
    avg_engagement_by_user = last.groupby("user_id")["engagement"].mean().reset_index().to_dict(orient="records")
    recent = last.tail(200).to_dict(orient="records")
    return jsonify({"emotion_counts": emotion_counts, "avg_engagement_by_user": avg_engagement_by_user, "recent": recent})

@app.route("/download_logs")
def download_logs():
    if not os.path.exists(LOG_FILE):
        return jsonify({"status": "error", "message": "no logs"}), 404
    return send_file(LOG_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
