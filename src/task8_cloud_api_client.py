#!/usr/bin/env python3
"""
Task 7 & Task 8: Cloud Inference API Deployment & Local Flask Integration
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

Demonstrates:
  1. Testing the public cloud model inference endpoint (/api/predict)
  2. Sending multiple test images (both Real and AI-generated)
  3. Recording status codes, predicted labels, confidence percentages, and server latencies
  4. Logging results into the local database audit trail (Task 5 & 8)
  5. Generating evidence table and terminal outputs for the final PDF report
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

# Cloud API URL: configure via environment variable or default to local/tunnel endpoint
CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://localhost:5001/api/predict")
ARTIFACTS_DIR = Path("artifacts/task8")


def test_cloud_api_with_images(
    api_url: str = CLOUD_API_URL,
    test_images: List[str] = None
) -> List[Dict[str, Any]]:
    """Sends multiple images to the cloud inference API and logs structured responses."""
    print("=" * 80)
    print("TASK 7 & 8: CLOUD INFERENCE API INTEGRATION & VERIFICATION")
    print(f"Target Cloud API URL : {api_url}")
    print(f"Student Roll Number  : 23L-0913")
    print(f"Timestamp            : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 80)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not test_images:
        # Pick 4 representative images from dataset splits or raw data
        test_images = []
        for p in [
            "data/splits/test.csv",
            "data/raw/RealArt",
            "data/raw/AiArtData",
        ]:
            path_obj = Path(p)
            if path_obj.exists():
                if p.endswith(".csv"):
                    import pandas as pd
                    df = pd.read_csv(p)
                    real_imgs = df[df["class"] == "real"]["filepath"].head(2).tolist()
                    ai_imgs = df[df["class"] == "ai_generated"]["filepath"].head(2).tolist()
                    test_images = real_imgs + ai_imgs
                    break

    results = []
    print(f"\n[INFO] Sending {len(test_images)} test requests to Cloud API endpoint...\n")

    for idx, img_path in enumerate(test_images, 1):
        if not Path(img_path).exists():
            continue

        fname = Path(img_path).name
        print(f"--- [Request {idx}/{len(test_images)}] Uploading: {fname} ---")

        start_req = time.perf_counter()
        with open(img_path, "rb") as f:
            files = {"image": (fname, f, "image/jpeg")}
            data = {"threshold": 0.50}

            try:
                resp = requests.post(api_url, files=files, data=data, timeout=30)
                roundtrip_ms = round((time.perf_counter() - start_req) * 1000.0, 2)

                status_code = resp.status_code
                if status_code == 200:
                    body = resp.json()
                    verdict = body.get("label", "Unknown")
                    confidence = body.get("confidence", 0.0)
                    raw_score = body.get("raw_score", 0.0)
                    model_ver = body.get("model_version", "N/A")
                    print(f"  Status       : {status_code} OK")
                    print(f"  Verdict      : {verdict}")
                    print(f"  Confidence   : {confidence}% (Raw: {raw_score})")
                    print(f"  Roundtrip    : {roundtrip_ms} ms")
                    print(f"  Model Serving: {model_ver}")
                else:
                    print(f"  Status       : {status_code} FAILED -> {resp.text[:100]}")
                    verdict = "ERROR"
                    confidence = 0.0
                    raw_score = 0.0
                    model_ver = "ERROR"

                results.append({
                    "Request #": idx,
                    "File Name": fname,
                    "Status": status_code,
                    "Verdict": verdict,
                    "Confidence (%)": confidence,
                    "Raw Score": raw_score,
                    "Roundtrip Latency (ms)": roundtrip_ms,
                    "Serving Model": model_ver,
                })

            except Exception as exc:
                print(f"  Connection Error: {exc}")
                results.append({
                    "Request #": idx,
                    "File Name": fname,
                    "Status": "ConnectionError",
                    "Verdict": "Error",
                    "Confidence (%)": 0.0,
                    "Raw Score": 0.0,
                    "Roundtrip Latency (ms)": 0.0,
                    "Serving Model": "N/A",
                })

    if results:
        import pandas as pd
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(ARTIFACTS_DIR / "task8_api_integration_results.csv", index=False)
        print("\n" + "=" * 80)
        print("TASK 8 API INTEGRATION RESULTS TABLE")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)

    return results


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else CLOUD_API_URL
    test_cloud_api_with_images(api_url=url)
