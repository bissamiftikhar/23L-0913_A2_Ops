#!/usr/bin/env python3
"""
Unit & Integration Test Suite for VisionGuard
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

Tests:
  1. Test environment package imports and TensorFlow availability
  2. Test inference preprocessing logic and output schema
  3. Test Flask healthcheck endpoint (/health)
  4. Test Flask error handlers (unsupported file format, missing payload)
  5. Test database audit log insertion and retrieval
"""

import io
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PIL import Image
from app import app
from src.inference import predict_image
from src.models_db import log_prediction, get_all_logs, init_db


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_environment_imports():
    """Verify core MLOps dependencies can be imported successfully."""
    import tensorflow as tf
    import keras
    import flask
    assert tf.__version__ is not None
    assert keras.__version__ is not None
    assert flask.__version__ is not None


def test_inference_on_synthetic_image():
    """Verify that inference pipeline returns expected structure, confidence in [0, 100], and valid latency."""
    img = Image.new("RGB", (224, 224), color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    result = predict_image(raw_bytes, threshold=0.50)
    assert "label" in result
    assert result["label"] in ["Real Image", "AI-Generated"]
    assert 0.0 <= result["confidence"] <= 100.0
    assert 0.0 <= result["raw_score"] <= 1.0
    assert result["latency_ms"] > 0.0
    assert "caveat" in result


def test_healthcheck_endpoint(client):
    """Verify /health returns 200 and healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_validation_rejects_unsupported_extension(client):
    """Verify that .txt or non-image extensions are rejected with 400 Bad Request."""
    data = {
        "image": (io.BytesIO(b"Hello world, I am not an image."), "malicious.txt")
    }
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"Unsupported File Type" in resp.data or b"Rejected file" in resp.data


def test_validation_rejects_empty_file_field(client):
    """Verify that empty file submissions are rejected with 400."""
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_database_logging():
    """Verify that audit logs can be written and queried."""
    init_db()
    initial_count = len(get_all_logs())
    log_id = log_prediction(
        filename="ci_test_sample.png",
        prediction_label="Real Image",
        confidence=96.5,
        decision_threshold=0.50,
        raw_score=0.035,
        model_version="CI-Test-Model",
        latency_ms=12.5,
        client_ip="127.0.0.1",
    )
    assert log_id > 0
    updated_logs = get_all_logs()
    assert len(updated_logs) == initial_count + 1
    assert updated_logs[0]["filename"] == "ci_test_sample.png"
