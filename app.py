#!/usr/bin/env python3
"""
Task 4 & Task 5: VisionGuard Flask Web Application & Inference Service
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

Routes:
  - GET  /         : Image upload portal & user interface
  - POST /predict  : Form submission with strict multi-layer validation & inference
  - GET  /history  : Database audit trail showing past predictions (Task 5)
  - GET  /health   : Container readiness / liveness healthcheck (Tasks 4.4 & 6.3)
  - POST /api/predict : REST API endpoint for Cloud/Vercel integration (Tasks 7 & 8)

Validation & Security Policies (Task 4.2):
  1. File type verification (Strict whitelist: png, jpg, jpeg, webp)
  2. Maximum payload size restriction (10 MB via MAX_CONTENT_LENGTH)
  3. Image byte decoding integrity check (Rejects corrupted/tampered bytes)
  4. Filename sanitization via werkzeug.utils.secure_filename
  5. Zero-disk retention (Direct stream processing via io.BytesIO)
  6. Debug mode OFF by default for security and thread-safe execution
"""

import os
import io

from flask import (
    Flask, render_template, request, jsonify, flash
)
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv

from src.inference import predict_image, get_model, MODEL_VERSION, DEFAULT_THRESHOLD
from src.models_db import init_db, log_prediction, get_all_logs, get_db_connection

load_dotenv()

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "10"))
MAX_CONTENT_LENGTH = MAX_FILE_SIZE_MB * 1024 * 1024

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "visionguard-secret-key-23L0913")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Initialize database schema at startup
init_db()

# Pre-load champion model into RAM at startup (Task 4.1)
try:
    get_model()
except Exception as exc:
    print(f"[WARN] Startup model loading deferred: {exc}")


def is_allowed_extension(filename: str) -> bool:
    """Check if file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_stream(file_stream: io.BytesIO) -> None:
    """Verifies that the byte stream can be genuinely decoded as an image."""
    try:
        file_stream.seek(0)
        with Image.open(file_stream) as img:
            img.verify()
        file_stream.seek(0)
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable image stream: {str(e)}")


@app.route("/", methods=["GET"])
def index():
    """Upload interface page."""
    return render_template(
        "index.html",
        model_version=MODEL_VERSION,
        threshold=DEFAULT_THRESHOLD,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Form upload handler with full validation and zero-disk retention."""
    if "image" not in request.files:
        flash("Error: No image file was submitted in the request.", "danger")
        return render_template("error.html", error_title="Missing File", error_message="Please select an image file to upload."), 400

    file = request.files["image"]
    if file.filename == "":
        flash("Error: No file selected.", "danger")
        return render_template("error.html", error_title="No File Selected", error_message="You submitted an empty file input. Please choose a valid image."), 400

    raw_filename = file.filename
    clean_filename = secure_filename(raw_filename)

    # 1. Reject invalid file extensions
    if not is_allowed_extension(clean_filename):
        err_msg = f"Rejected file '{clean_filename}'. Only PNG, JPG, JPEG, and WEBP formats are accepted."
        return render_template("error.html", error_title="Unsupported File Type", error_message=err_msg), 400

    # 2. Read into RAM buffer without writing to disk
    file_bytes = file.read()
    if len(file_bytes) == 0:
        return render_template("error.html", error_title="Empty File", error_message="The uploaded file contains 0 bytes."), 400

    # 3. Check file size limit
    if len(file_bytes) > MAX_CONTENT_LENGTH:
        err_msg = f"File size ({len(file_bytes) / (1024*1024):.2f} MB) exceeds maximum limit of {MAX_FILE_SIZE_MB} MB."
        return render_template("error.html", error_title="File Size Exceeded", error_message=err_msg), 413

    # 4. Byte decode verification (detect pseudo-images / corrupt bytes)
    stream = io.BytesIO(file_bytes)
    try:
        validate_image_stream(stream)
    except ValueError:
        err_msg = f"Image validation failed: The file '{clean_filename}' appears corrupted or cannot be decoded."
        return render_template("error.html", error_title="Corrupt Image", error_message=err_msg), 400

    # 5. Extract optional user threshold
    threshold = DEFAULT_THRESHOLD
    if "threshold" in request.form and request.form["threshold"]:
        try:
            val = float(request.form["threshold"])
            if 0.05 <= val <= 0.95:
                threshold = val
        except ValueError:
            pass

    # 6. Execute inference
    res = predict_image(file_bytes, threshold=threshold)

    # 7. Log to database trail (Task 5)
    client_ip = request.remote_addr or "127.0.0.1"
    log_id = log_prediction(
        filename=clean_filename,
        prediction_label=res["label"],
        confidence=res["confidence"],
        decision_threshold=res["decision_threshold"],
        raw_score=res["raw_score"],
        model_version=res["model_version"],
        latency_ms=res["latency_ms"],
        client_ip=client_ip,
    )

    # Convert image to base64 for preview in browser without disk save
    import base64
    preview_base64 = base64.b64encode(file_bytes).decode("utf-8")
    ext = clean_filename.rsplit(".", 1)[1].lower()
    mime_type = "image/png" if ext == "png" else "image/jpeg"

    return render_template(
        "result.html",
        result=res,
        filename=clean_filename,
        log_id=log_id,
        preview_src=f"data:{mime_type};base64,{preview_base64}",
    )


@app.route("/history", methods=["GET"])
def history():
    """Task 5: Database trail history view."""
    logs = get_all_logs(limit=50)
    return render_template("history.html", logs=logs, total_count=len(logs))


@app.route("/health", methods=["GET"])
def health():
    """Task 4.4 & 6.3: Container and Service Health Check."""
    db_status = "ok"
    try:
        conn = get_db_connection()
        conn.close()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    model_loaded = False
    try:
        model_loaded = get_model() is not None
    except Exception:
        model_loaded = False
    healthy = (db_status == "ok")

    payload = {
        "status": "healthy" if healthy else "degraded",
        "service": "VisionGuard Web App",
        "model_version": MODEL_VERSION,
        "model_loaded": model_loaded,
        "database": db_status,
        "decision_threshold": DEFAULT_THRESHOLD,
        "debug_mode": app.debug,
    }
    return jsonify(payload), (200 if healthy else 503)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Tasks 7 & 8: Cloud/Vercel REST API endpoint.
    Accepts multipart image upload or binary data, returns JSON.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field found in multipart/form-data request."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file provided."}), 400

    fname = secure_filename(file.filename)
    if not is_allowed_extension(fname):
        return jsonify({"error": f"Invalid extension. Allowed: {list(ALLOWED_EXTENSIONS)}"}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_CONTENT_LENGTH:
        return jsonify({"error": f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB."}), 413

    try:
        validate_image_stream(io.BytesIO(file_bytes))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    thresh = DEFAULT_THRESHOLD
    if "threshold" in request.form:
        try:
            thresh = float(request.form["threshold"])
        except ValueError:
            pass

    res = predict_image(file_bytes, threshold=thresh)
    log_prediction(
        filename=fname,
        prediction_label=res["label"],
        confidence=res["confidence"],
        decision_threshold=res["decision_threshold"],
        raw_score=res["raw_score"],
        model_version=res["model_version"],
        latency_ms=res["latency_ms"],
        client_ip=request.remote_addr or "remote-api",
    )
    return jsonify({"success": True, "filename": fname, **res}), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template(
        "error.html",
        error_title="Payload Too Large (413)",
        error_message=f"Uploaded file exceeds server limit of {MAX_FILE_SIZE_MB} MB."
    ), 413


@app.errorhandler(404)
def not_found(error):
    return render_template(
        "error.html",
        error_title="Page Not Found (404)",
        error_message="The requested endpoint does not exist."
    ), 404


@app.errorhandler(500)
def server_error(error):
    return render_template(
        "error.html",
        error_title="Internal Server Error (500)",
        error_message="An unexpected server error occurred during request processing."
    ), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    debug_flag = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true")
    print(f"[INFO] Starting VisionGuard Flask server on http://0.0.0.0:{port} (Debug: {debug_flag})")
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
