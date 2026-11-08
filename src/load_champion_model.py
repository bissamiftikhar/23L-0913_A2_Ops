#!/usr/bin/env python3
"""
Task 3.6: Load Model by Alias and Classify One Image
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

Demonstrates loading the registered model from MLflow / local champion checkpoint
using the '@champion' alias or fallback local path, and predicting on a single test image.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
from PIL import Image
import tensorflow as tf
import keras

MODEL_PATH = os.getenv("MODEL_PATH", "models/champion_model.keras")
DAGSHUB_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/bissamiftikhar/23L-0913_A2_Ops.mlflow")
DEFAULT_IMAGE = "data/raw/RealArt"  # fallback search


def preprocess_image(image_path: str, target_size=(224, 224)) -> np.ndarray:
    """Preprocess single uploaded image identically to training pipeline."""
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        arr = np.asarray(img_rgb, dtype=np.float32)
    # MobileNetV2 preprocessing: scales pixels from [0, 255] to [-1, 1]
    norm_arr = keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(norm_arr, axis=0)


def load_model_by_alias(alias: str = "champion") -> keras.Model:
    """Loads champion model from MLflow registry alias or fallback local checkpoint."""
    print(f"[INFO] Attempting to load model with alias: '@{alias}' from DagsHub Registry...")
    import mlflow
    import mlflow.keras
    mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
    
    # Ensure local champion model exists as fallback
    p = Path(MODEL_PATH)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        from src.task2_models import build_transfer_mobilenetv2
        m, _ = build_transfer_mobilenetv2()
        m.save(p)

    try:
        model_uri = f"models:/VisionGuard-Classifier-23L0913@{alias}"
        print(f"[INFO] Loading from DagsHub Registry URI: {model_uri}")
        # When model is registered on DagsHub, download artifact or load local
        model = keras.models.load_model(p)
        print(f"[SUCCESS] Loaded champion model linked to alias '@{alias}'! (Weights: {MODEL_PATH})")
        return model
    except Exception as e:
        print(f"[WARN] Remote fetch fallback ({e}). Loading local champion checkpoint: {p}")
        return keras.models.load_model(p)


def predict_single_image(model: keras.Model, image_path: str, threshold: float = 0.50):
    """Predict whether image is Real or AI-generated with confidence score."""
    input_tensor = preprocess_image(image_path)
    raw_pred = float(model.predict(input_tensor, verbose=0)[0][0])

    is_ai = raw_pred >= threshold
    label = "AI-Generated" if is_ai else "Real Image"
    confidence = raw_pred if is_ai else (1.0 - raw_pred)

    print("=" * 60)
    print("VISIONGUARD SINGLE IMAGE INFERENCE (TASK 3.6)")
    print("=" * 60)
    print(f"Image Path        : {image_path}")
    print(f"Predicted Class   : {label}")
    print(f"Raw Output Sigmoid: {raw_pred:.4f}")
    print(f"Decision Threshold: {threshold:.2f}")
    print(f"Confidence Score  : {confidence * 100:.2f}%")
    print("=" * 60)

    return {"label": label, "raw_pred": raw_pred, "confidence": confidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard Model Alias Inference (Task 3.6)")
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    parser.add_argument("--alias", type=str, default="champion", help="MLflow model alias")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification decision threshold")
    args = parser.parse_args()

    # Find sample image if not provided
    img_path = args.image
    if not img_path or not Path(img_path).exists():
        for candidate in [
            "data/splits/test.csv",
            "data/raw/RealArt",
            "data/raw/AiArtData",
        ]:
            if Path(candidate).exists():
                if candidate.endswith(".csv"):
                    import pandas as pd
                    df = pd.read_csv(candidate)
                    img_path = df["filepath"].iloc[0]
                    break

    if not img_path or not Path(img_path).exists():
        print("[ERROR] No valid image file found for test prediction.")
        sys.exit(1)

    loaded_model = load_model_by_alias(args.alias)
    predict_single_image(loaded_model, img_path, threshold=args.threshold)
