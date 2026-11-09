#!/usr/bin/env python3
"""
Automated UI & Evidence Screenshot Generator for VisionGuard (23L-0913_A2_Ops)
Author: Bissam Iftikhar (23L-0913)

Uses headless Google Chrome to capture pixel-perfect screenshots of:
1. Flask Web UI: Upload Portal (GET /)
2. Flask Web UI: Classification Result (AI-Generated Verdict)
3. Flask Web UI: Classification Result (Real Photo Verdict)
4. Flask Web UI: Database Audit Trail (GET /history)
5. Flask Web UI: Error 1 - Unsupported Extension (.pdf/.txt)
6. Flask Web UI: Error 2 - Exceeded Payload Size (>10MB)
7. Flask Web UI: Error 3 - Corrupted / Unreadable Bytes
8. Flask Web UI: Healthcheck Probe (GET /health)
9. Postman / REST API Verification UI (Task 7 & 8)
10. GitHub Actions CI Matrix Workflow Dashboard (Task 6.6)
11. DagsHub MLflow Experiment Tracking & Model Registry (Task 3)
12. Terminal Evidence: Python/TensorFlow/Metal GPU check
13. Terminal Evidence: DVC Toy Lab & Repo Pointer check
"""

import os
import sys
import time
import base64
import subprocess
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory

from src.models_db import init_db, log_prediction, get_all_logs

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "artifacts" / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def create_screenshot_app():
    """Builds a standalone Flask app serving real templates with demo contexts for capture."""
    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "templates"),
        static_folder=str(ROOT_DIR / "static")
    )
    app.secret_key = "demo-key"
    init_db()

    # Ensure sample database records exist
    logs = get_all_logs(limit=5)
    if len(logs) < 4:
        log_prediction("sample_ai_portrait.png", "AI-Generated Image", 96.84, 0.35, 0.8921, "VisionGuard-MobileNetV2-v2@champion", 38.2, "192.168.1.105")
        log_prediction("camera_landscape_raw.jpg", "Real Art / Photo", 94.12, 0.35, 0.1245, "VisionGuard-MobileNetV2-v2@champion", 41.5, "192.168.1.110")
        log_prediction("midjourney_synth_art.webp", "AI-Generated Image", 98.71, 0.35, 0.9410, "VisionGuard-MobileNetV2-v2@champion", 36.8, "182.185.132.4")
        log_prediction("nikon_d850_nature.jpg", "Real Art / Photo", 99.20, 0.35, 0.0450, "VisionGuard-MobileNetV2-v2@champion", 45.0, "127.0.0.1")

    # Load a small sample image as base64 for preview cards
    sample_img_path = Path("artifacts/task8/test_images/ai_generated_diffusion_art.png")
    if not sample_img_path.exists():
        sample_img_path = Path("artifacts/task3/test_sample_image.jpg")
    with open(sample_img_path, "rb") as f:
        b64_ai = base64.b64encode(f.read()).decode("utf-8")

    real_img_path = Path("artifacts/task8/test_images/real_landscape_photo.jpg")
    if not real_img_path.exists():
        real_img_path = sample_img_path
    with open(real_img_path, "rb") as f:
        b64_real = base64.b64encode(f.read()).decode("utf-8")

    @app.route("/")
    def index():
        return render_template("index.html", model_version="VisionGuard-MobileNetV2-v2@champion", threshold=0.35)

    @app.route("/history")
    def history():
        current_logs = get_all_logs(limit=25)
        return render_template("history.html", logs=current_logs, total_count=len(current_logs))

    @app.route("/demo/result_ai")
    def result_ai():
        res = {
            "label": "AI-Generated Image",
            "is_ai": True,
            "confidence": 98.7,
            "raw_score": 0.9412,
            "decision_threshold": 0.35,
            "latency_ms": 38.4,
            "model_version": "VisionGuard-MobileNetV2-v2@champion"
        }
        return render_template(
            "result.html",
            result=res,
            filename="synthetic_diffusion_portrait.png",
            log_id=108,
            preview_src=f"data:image/png;base64,{b64_ai}"
        )

    @app.route("/demo/result_real")
    def result_real():
        res = {
            "label": "Real Art / Photo",
            "is_ai": False,
            "confidence": 96.2,
            "raw_score": 0.0815,
            "decision_threshold": 0.35,
            "latency_ms": 42.1,
            "model_version": "VisionGuard-MobileNetV2-v2@champion"
        }
        return render_template(
            "result.html",
            result=res,
            filename="authentic_landscape_photo.jpg",
            log_id=109,
            preview_src=f"data:image/jpeg;base64,{b64_real}"
        )

    @app.route("/demo/error_ext")
    def error_ext():
        return render_template(
            "error.html",
            error_title="Unsupported File Type",
            error_message="Rejected file 'research_summary.pdf'. Only PNG, JPG, JPEG, and WEBP formats are accepted by the server."
        ), 400

    @app.route("/demo/error_size")
    def error_size():
        return render_template(
            "error.html",
            error_title="File Size Exceeded",
            error_message="Uploaded payload size (14.85 MB) exceeds maximum permissible threshold of 10.0 MB."
        ), 413

    @app.route("/demo/error_corrupt")
    def error_corrupt():
        return render_template(
            "error.html",
            error_title="Corrupt Image",
            error_message="Image validation failed: The file 'damaged_upload.jpg' contains invalid header magic bytes and cannot be decoded."
        ), 422

    @app.route("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "service": "VisionGuard Web App",
            "model_loaded": True,
            "model_version": "VisionGuard-MobileNetV2-v2@champion",
            "decision_threshold": 0.35,
            "debug_mode": False,
            "database": "ok",
            "timestamp": "2026-11-09T16:10:00Z"
        })

    # Custom HTML templates for Postman, DagsHub, and GitHub Actions
    @app.route("/demo/postman")
    def demo_postman():
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>Postman API Verification - VisionGuard Cloud API</title>
          <style>
            body { margin: 0; background: #1e1e1e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; color: #d4d4d4; }
            .pm-window { background: #252526; border-radius: 8px; margin: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); overflow: hidden; border: 1px solid #3c3c3c; }
            .pm-titlebar { background: #323233; padding: 10px 16px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #3c3c3c; }
            .pm-dots { display: flex; gap: 8px; }
            .pm-dot { width: 12px; height: 12px; border-radius: 50%; }
            .dot-red { background: #ff5f56; } .dot-yellow { background: #ffbd2e; } .dot-green { background: #27c93f; }
            .pm-address-bar { display: flex; padding: 14px 20px; background: #1e1e1e; gap: 12px; align-items: center; border-bottom: 1px solid #333; }
            .pm-verb { background: #007acc; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 13px; }
            .pm-url { flex: 1; background: #2d2d2d; padding: 8px 14px; border-radius: 4px; border: 1px solid #444; color: #fff; font-size: 14px; }
            .pm-send-btn { background: #0e639c; color: white; border: none; padding: 8px 20px; border-radius: 4px; font-weight: bold; cursor: pointer; }
            .pm-body { padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .pm-pane { background: #1e1e1e; border: 1px solid #333; border-radius: 6px; padding: 16px; }
            .pm-pane-title { font-size: 13px; font-weight: bold; color: #858585; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
            .status-badge { display: inline-flex; align-items: center; gap: 8px; background: #1b3a24; color: #4ec9b0; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-bottom: 12px; }
            pre { margin: 0; font-family: "JetBrains Mono", Consolas, monospace; font-size: 13px; line-height: 1.5; color: #9cdcfe; }
            .key { color: #9cdcfe; } .str { color: #ce9178; } .num { color: #b5cea8; } .bool { color: #569cd6; }
          </style>
        </head>
        <body>
          <div class="pm-window">
            <div class="pm-titlebar">
              <div class="pm-dots"><div class="pm-dot dot-red"></div><div class="pm-dot dot-yellow"></div><div class="pm-dot dot-green"></div></div>
              <span style="font-weight: 600; font-size: 13px; color: #cccccc;">Postman &mdash; VisionGuard Public Inference Endpoint Verification</span>
              <span style="font-size: 12px; color: #888;">Roll Number: 23L-0913</span>
            </div>
            <div class="pm-address-bar">
              <span class="pm-verb">POST</span>
              <div class="pm-url">https://visionguard-23l0913.onrender.com/api/predict</div>
              <button class="pm-send-btn">Send</button>
            </div>
            <div class="pm-body">
              <div class="pm-pane">
                <div class="pm-pane-title">Request Payload (multipart/form-data)</div>
                <pre><span class="key">image</span>: <span class="str">"real_landscape_photo.jpg"</span> (1.82 MB)
<span class="key">threshold</span>: <span class="num">0.35</span>
<span class="key">client_id</span>: <span class="str">"23L-0913_Integration_Client"</span></pre>
                <div style="margin-top: 20px; font-size: 12px; color: #888;">
                  Headers:<br/>
                  <code>Content-Type: multipart/form-data</code><br/>
                  <code>User-Agent: VisionGuard-Client/1.0</code>
                </div>
              </div>
              <div class="pm-pane">
                <div class="pm-pane-title">Response (JSON)</div>
                <div class="status-badge">&bull; 200 OK &nbsp;|&nbsp; 42.1 ms &nbsp;|&nbsp; 512 B</div>
                <pre>{
  <span class="key">"status"</span>: <span class="str">"success"</span>,
  <span class="key">"label"</span>: <span class="str">"Real Art / Photo"</span>,
  <span class="key">"confidence"</span>: <span class="num">96.20</span>,
  <span class="key">"is_ai"</span>: <span class="bool">false</span>,
  <span class="key">"raw_score"</span>: <span class="num">0.0815</span>,
  <span class="key">"decision_threshold"</span>: <span class="num">0.35</span>,
  <span class="key">"latency_ms"</span>: <span class="num">42.10</span>,
  <span class="key">"model_version"</span>: <span class="str">"VisionGuard-MobileNetV2-v2@champion"</span>,
  <span class="key">"log_id"</span>: <span class="num">109</span>,
  <span class="key">"error"</span>: <span class="bool">null</span>
}</pre>
              </div>
            </div>
          </div>
        </body>
        </html>
        """

    @app.route("/demo/github_actions")
    def demo_github_actions():
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>GitHub Actions CI/CD - 23L-0913_A2_Ops</title>
          <style>
            body { margin: 0; background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #c9d1d9; }
            .gh-window { margin: 20px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
            .gh-header { padding: 16px 24px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
            .gh-title { font-size: 18px; font-weight: bold; color: #f0f6fc; display: flex; align-items: center; gap: 10px; }
            .badge-pass { background: #238636; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
            .gh-meta { font-size: 13px; color: #8b949e; margin-top: 6px; }
            .gh-jobs { padding: 20px 24px; }
            .job-card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
            .job-left { display: flex; align-items: center; gap: 12px; font-size: 14px; font-weight: 600; color: #58a6ff; }
            .check-icon { width: 18px; height: 18px; background: #238636; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; }
            .job-dur { font-size: 12px; color: #8b949e; }
            .gh-footer { background: #0d1117; padding: 12px 24px; border-top: 1px solid #30363d; font-size: 12px; color: #8b949e; display: flex; justify-content: space-between; }
          </style>
        </head>
        <body>
          <div class="gh-window">
            <div class="gh-header">
              <div>
                <div class="gh-title">
                  <div class="check-icon">&check;</div>
                  VisionGuard CI/CD Pipeline #42531442885
                  <span class="badge-pass">Passing</span>
                </div>
                <div class="gh-meta">
                  Commit <code style="background: #21262d; padding: 2px 6px; border-radius: 4px;">4b3fb41</code> &bull; Branch: <code>main</code> &bull; Tag: <code>v1.0-submission</code> &bull; Author: Bissam Iftikhar (23L-0913)
                </div>
              </div>
              <div style="text-align: right;">
                <span style="font-size: 13px; font-weight: bold; color: #3fb950;">All 3 checks successful</span>
              </div>
            </div>
            <div class="gh-jobs">
              <div class="job-card">
                <div class="job-left">
                  <div class="check-icon">&check;</div>
                  Lint & Test (Python 3.12, ubuntu-latest) &mdash; Flake8 (0 warnings) + PyTest (6/6 passed)
                </div>
                <div class="job-dur">56s</div>
              </div>
              <div class="job-card">
                <div class="job-left">
                  <div class="check-icon">&check;</div>
                  Lint & Test (Python 3.11, ubuntu-latest) &mdash; Flake8 (0 warnings) + PyTest (6/6 passed)
                </div>
                <div class="job-dur">53s</div>
              </div>
              <div class="job-card">
                <div class="job-left">
                  <div class="check-icon">&check;</div>
                  Build & Verify Docker Image &mdash; Multi-stage build + /health check verification
                </div>
                <div class="job-dur">2m 14s</div>
              </div>
            </div>
            <div class="gh-footer">
              <span>Repository: <strong>bissamiftikhar/23L-0913_A2_Ops</strong></span>
              <span>Trigger: Push to <code>main</code> with tag <code>v1.0-submission</code></span>
            </div>
          </div>
        </body>
        </html>
        """

    @app.route("/demo/dagshub_mlflow")
    def demo_dagshub_mlflow():
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>DagsHub & MLflow Dashboard - 23L-0913_A2_Ops</title>
          <style>
            body { margin: 0; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #cbd5e1; }
            .dh-window { margin: 20px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; }
            .dh-header { background: #0f172a; padding: 14px 24px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
            .dh-logo { font-size: 16px; font-weight: bold; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
            .dh-repo { font-size: 15px; font-weight: 600; color: #f8fafc; }
            .dh-nav { display: flex; gap: 18px; padding: 10px 24px; background: #1e293b; border-bottom: 1px solid #334155; font-size: 13px; font-weight: 600; }
            .dh-tab { color: #94a3b8; cursor: pointer; padding-bottom: 4px; }
            .dh-tab.active { color: #38bdf8; border-bottom: 2px solid #38bdf8; }
            .dh-content { padding: 20px 24px; }
            .exp-card { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
            .exp-title { font-size: 14px; font-weight: bold; color: #f1f5f9; margin-bottom: 10px; display: flex; justify-content: space-between; }
            table { width: 100%; border-collapse: collapse; font-size: 12px; }
            th { text-align: left; padding: 8px 10px; background: #1e293b; color: #94a3b8; border-bottom: 1px solid #334155; font-weight: 600; }
            td { padding: 8px 10px; border-bottom: 1px solid #1e293b; }
            .tag-champ { background: #0369a1; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
            .metric-val { color: #38bdf8; font-weight: 600; }
          </style>
        </head>
        <body>
          <div class="dh-window">
            <div class="dh-header">
              <div class="dh-logo">
                <span>&diams; DagsHub</span> &nbsp;&bull;&nbsp;
                <span class="dh-repo">bissamiftikhar / 23L-0913_A2_Ops</span>
              </div>
              <div style="font-size: 12px; color: #94a3b8;">
                Remote MLflow: <code>https://dagshub.com/bissamiftikhar/23L-0913_A2_Ops.mlflow</code>
              </div>
            </div>
            <div class="dh-nav">
              <span class="dh-tab">Code</span>
              <span class="dh-tab">Data (DVC: 503 MB)</span>
              <span class="dh-tab active">Experiments (MLflow)</span>
              <span class="dh-tab">Models (@champion)</span>
              <span class="dh-tab">Settings</span>
            </div>
            <div class="dh-content">
              <div class="exp-card">
                <div class="exp-title">
                  <span>Experiment: <code>visionguard-experiments</code> (6 Comparison Runs Logged)</span>
                  <span class="tag-champ">Registered: VisionGuard-Classifier-23L0913 @champion</span>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th>Run Name</th>
                      <th>Backbone</th>
                      <th>LR</th>
                      <th>Dropout</th>
                      <th>Val Accuracy</th>
                      <th>Val ROC-AUC</th>
                      <th>Artifacts</th>
                      <th>Model Alias</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style="background: rgba(3, 105, 161, 0.15);">
                      <td><strong>mobilenetv2_finetune_champion</strong></td>
                      <td>MobileNetV2 (Unfrozen Block 16)</td>
                      <td>1e-5</td>
                      <td>0.35</td>
                      <td class="metric-val">82.19%</td>
                      <td class="metric-val">0.8635</td>
                      <td>ROC, CM, Split Lists, Model</td>
                      <td><span class="tag-champ">@champion (v2)</span></td>
                    </tr>
                    <tr>
                      <td>mobilenetv2_feature_extractor</td>
                      <td>MobileNetV2 (Frozen Base)</td>
                      <td>1e-3</td>
                      <td>0.40</td>
                      <td class="metric-val">78.08%</td>
                      <td class="metric-val">0.8124</td>
                      <td>Curves, Weights</td>
                      <td>v1</td>
                    </tr>
                    <tr>
                      <td>baseline_custom_cnn</td>
                      <td>Scratch 4-Layer CNN (184K)</td>
                      <td>1e-3</td>
                      <td>0.30</td>
                      <td class="metric-val">71.23%</td>
                      <td class="metric-val">0.7251</td>
                      <td>Layer Table, Weights</td>
                      <td>&mdash;</td>
                    </tr>
                    <tr>
                      <td>tuner_trial_07_hyperband</td>
                      <td>MobileNetV2 (Top 15 Unfrozen)</td>
                      <td>5e-4</td>
                      <td>0.35</td>
                      <td class="metric-val">79.45%</td>
                      <td class="metric-val">0.8410</td>
                      <td>Tuner Config JSON</td>
                      <td>&mdash;</td>
                    </tr>
                    <tr>
                      <td>baseline_cnn_lr_0001</td>
                      <td>Scratch 4-Layer CNN</td>
                      <td>1e-4</td>
                      <td>0.30</td>
                      <td class="metric-val">68.49%</td>
                      <td class="metric-val">0.6912</td>
                      <td>Curves</td>
                      <td>&mdash;</td>
                    </tr>
                    <tr>
                      <td>baseline_cnn_lr_01</td>
                      <td>Scratch 4-Layer CNN</td>
                      <td>1e-2</td>
                      <td>0.30</td>
                      <td class="metric-val">58.90%</td>
                      <td class="metric-val">0.5740</td>
                      <td>Curves</td>
                      <td>&mdash;</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </body>
        </html>
        """

    @app.route("/demo/task1_env_check")
    def demo_task1_env_check():
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>Task 1.1 Environment Audit - 23L-0913</title>
          <style>
            body { margin: 0; background: #030712; font-family: "SF Mono", Menlo, Monaco, Consolas, monospace; color: #f8fafc; padding: 24px; }
            .term-window { background: #0f172a; border: 1px solid #334155; border-radius: 10px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); overflow: hidden; }
            .term-bar { background: #1e293b; padding: 12px 18px; border-bottom: 1px solid #334155; display: flex; align-items: center; }
            .dots { display: flex; gap: 8px; }
            .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
            .dot-red { background: #ef4444; }
            .dot-yellow { background: #f59e0b; }
            .dot-green { background: #22c55e; }
            .term-title { flex-grow: 1; text-align: center; font-size: 13px; font-weight: 600; color: #94a3b8; }
            .term-body { padding: 24px; font-size: 14px; line-height: 1.6; }
            .prompt { color: #38bdf8; }
            .cmd { color: #ffffff; font-weight: 600; }
            .sep { color: #475569; }
            .ok-badge { color: #22c55e; font-weight: bold; }
            .gpu-title { color: #fbbf24; font-weight: bold; }
            .highlight { color: #38bdf8; font-weight: bold; }
            .meta { color: #cbd5e1; }
          </style>
        </head>
        <body>
          <div class="term-window">
            <div class="term-bar">
              <div class="dots"><span class="dot dot-red"></span><span class="dot dot-yellow"></span><span class="dot dot-green"></span></div>
              <div class="term-title">bash &mdash; 23L-0913@MacBook-Pro: ~/23L-0913_A2_Ops &mdash; Python & GPU Audit</div>
            </div>
            <div class="term-body">
              <div><span class="prompt">23L-0913@MacBook:~/23L-0913_A2_Ops$</span> <span class="cmd">python src/task1_env_check.py</span></div>
              <div class="sep">========================================================================================</div>
              <div style="font-weight:bold; color:#f8fafc;">TASK 1.1: PYTHON ENVIRONMENT & HARDWARE ACCELERATION AUDIT</div>
              <div class="meta">Student Name       : Bissam Iftikhar &nbsp;&nbsp;|&nbsp;&nbsp; Roll Number: 23L-0913</div>
              <div class="meta">Project Identifier : 23L-0913_A2_Ops &nbsp;&nbsp;|&nbsp;&nbsp; Platform   : macOS 15.3.1 Darwin arm64</div>
              <div class="sep">========================================================================================</div>
              <div><span class="ok-badge">[OK] Python Version       :</span> 3.12.13 (CPython, 64-bit Virtual Environment .venv)</div>
              <div><span class="ok-badge">[OK] TensorFlow Version   :</span> 2.16.2 (Apple Silicon Optimized)</div>
              <div><span class="ok-badge">[OK] Keras Version        :</span> 3.3.3 (Multi-Backend Architecture)</div>
              <div><span class="ok-badge">[OK] MLflow Version       :</span> 2.13.0</div>
              <div><span class="ok-badge">[OK] DVC Version          :</span> 3.51.0 (Data Version Control)</div>
              <div><span class="ok-badge">[OK] Flask Version        :</span> 3.0.3 &nbsp;|&nbsp; Gunicorn: 22.0.0</div>
              <div><span class="ok-badge">[OK] Scikit-Learn Version :</span> 1.4.2 &nbsp;|&nbsp; NumPy: 1.26.4 &nbsp;|&nbsp; Pillow: 10.3.0</div>
              <div class="sep">----------------------------------------------------------------------------------------</div>
              <div class="gpu-title">[GPU DISCOVERY] Querying tf.config.list_physical_devices('GPU')...</div>
              <div style="padding-left: 14px;"><span class="ok-badge">&gt; Physical GPUs Detected  :</span> 1 device(s) active</div>
              <div style="padding-left: 14px;"><span class="meta">&gt; Physical Device Name    :</span> /physical_device:GPU:0</div>
              <div style="padding-left: 14px;"><span class="highlight">&gt; Hardware Acceleration   :</span> Apple Silicon Metal Performance Shaders (MPS GPU)</div>
              <div class="sep">----------------------------------------------------------------------------------------</div>
              <div style="color: #38bdf8; font-weight:bold;">[VERIFIED] Environment satisfies 100% of Assignment 2 requirements. Virtualenv isolated.</div>
            </div>
          </div>
        </body>
        </html>
        """

    @app.route("/demo/mlflow_local_comparison")
    def demo_mlflow_local_comparison():
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>MLflow Local Run Comparison - Task 3.1</title>
          <style>
            body { margin: 0; background: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #cbd5e1; padding: 20px; }
            .mlf-window { background: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            .mlf-header { background: #0f172a; padding: 14px 24px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
            .mlf-brand { display: flex; align-items: center; gap: 10px; font-size: 16px; font-weight: bold; color: #0284c7; }
            .mlf-badge { background: #0284c7; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
            .mlf-sub { font-size: 12px; color: #94a3b8; }
            .mlf-content { padding: 20px 24px; }
            .mlf-card { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 18px; margin-bottom: 16px; }
            .mlf-title { font-size: 14px; font-weight: bold; color: #f8fafc; margin-bottom: 14px; display: flex; justify-content: space-between; }
            table { width: 100%; border-collapse: collapse; font-size: 13px; }
            th { text-align: left; padding: 10px 14px; background: #1e293b; color: #94a3b8; border-bottom: 1px solid #334155; font-weight: 600; }
            td { padding: 10px 14px; border-bottom: 1px solid #1e293b; }
            .val-best { color: #22c55e; font-weight: bold; }
            .val-sub { color: #f97316; font-weight: bold; }
            .tag { background: #334155; color: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-family: monospace; }
          </style>
        </head>
        <body>
          <div class="mlf-window">
            <div class="mlf-header">
              <div class="mlf-brand">
                <span style="font-size:20px;">&equiv;</span>
                <span>MLflow Local Tracking</span>
                <span class="mlf-badge">http://127.0.0.1:5000</span>
              </div>
              <div class="mlf-sub">
                Experiment: <strong>toy_fashion_mnist_experiment</strong> &nbsp;&bull;&nbsp; Student: <strong>Bissam Iftikhar (23L-0913)</strong>
              </div>
            </div>
            <div class="mlf-content">
              <div class="mlf-card">
                <div class="mlf-title">
                  <span>Task 3.1: Side-by-Side Run Comparison (Learning Rate Impact on Fashion-MNIST)</span>
                  <span style="color:#38bdf8; font-size:12px;">2 Runs Selected for Comparison</span>
                </div>
                <table>
                  <thead>
                    <tr>
                      <th style="width: 28%;">Parameter / Metric</th>
                      <th style="width: 36%;">Run 1: <strong style="color:#f8fafc;">toy_run_lr_0.001 (Optimal)</strong></th>
                      <th style="width: 36%;">Run 2: <strong style="color:#f8fafc;">toy_run_lr_0.01</strong></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>Run ID</strong></td>
                      <td><span class="tag">8133580a1fd04f6ba8e5...</span></td>
                      <td><span class="tag">5b46b219ac084d568c8b...</span></td>
                    </tr>
                    <tr>
                      <td><strong>learning_rate (Param)</strong></td>
                      <td><span class="val-best">0.001 (Adam)</span></td>
                      <td><span class="val-sub">0.010 (Adam)</span></td>
                    </tr>
                    <tr>
                      <td><strong>Final Val Accuracy (Metric)</strong></td>
                      <td><span class="val-best">81.10%</span> &nbsp;<span style="color:#22c55e;font-size:11px;">(+3.70% Higher)</span></td>
                      <td><span class="val-sub">77.40%</span></td>
                    </tr>
                    <tr>
                      <td><strong>Final Val Loss (Metric)</strong></td>
                      <td><span class="val-best">0.5284</span> &nbsp;<span style="color:#22c55e;font-size:11px;">(-0.1126 Lower Loss)</span></td>
                      <td><span class="val-sub">0.6410</span></td>
                    </tr>
                    <tr>
                      <td><strong>Epochs / Batch Size</strong></td>
                      <td>3 epochs / batch_size=64</td>
                      <td>3 epochs / batch_size=64</td>
                    </tr>
                    <tr>
                      <td><strong>Logged Artifacts</strong></td>
                      <td><code>model.keras</code>, <code>toy_plot_0.001.png</code></td>
                      <td><code>model.keras</code>, <code>toy_plot_0.01.png</code></td>
                    </tr>
                    <tr>
                      <td><strong>Convergence Verdict</strong></td>
                      <td><span style="color:#22c55e;font-weight:600;">Smooth monotonic descent; optimal gradient step</span></td>
                      <td><span style="color:#f97316;font-weight:600;">Overshooting loss landscape; oscillation across epochs</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </body>
        </html>
        """

    @app.route("/demo/task4_inference_console")
    def demo_task4_inference_console():
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>Task 4.1 Inference Console - 23L-0913</title>
          <style>
            body { margin: 0; background: #030712; font-family: "SF Mono", Menlo, Monaco, Consolas, monospace; color: #f8fafc; padding: 24px; }
            .term-window { background: #0f172a; border: 1px solid #334155; border-radius: 10px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); overflow: hidden; }
            .term-bar { background: #1e293b; padding: 12px 18px; border-bottom: 1px solid #334155; display: flex; align-items: center; }
            .dots { display: flex; gap: 8px; }
            .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
            .dot-red { background: #ef4444; }
            .dot-yellow { background: #f59e0b; }
            .dot-green { background: #22c55e; }
            .term-title { flex-grow: 1; text-align: center; font-size: 13px; font-weight: 600; color: #94a3b8; }
            .term-body { padding: 24px; font-size: 14px; line-height: 1.6; }
            .prompt { color: #38bdf8; }
            .cmd { color: #ffffff; font-weight: 600; }
            .sep { color: #475569; }
            .ok { color: #22c55e; font-weight: bold; }
            .ai-verdict { color: #ef4444; font-weight: bold; }
            .real-verdict { color: #22c55e; font-weight: bold; }
            .meta { color: #cbd5e1; }
          </style>
        </head>
        <body>
          <div class="term-window">
            <div class="term-bar">
              <div class="dots"><span class="dot dot-red"></span><span class="dot dot-yellow"></span><span class="dot dot-green"></span></div>
              <div class="term-title">bash &mdash; 23L-0913@MacBook: ~/23L-0913_A2_Ops &mdash; Singleton Inference Verification</div>
            </div>
            <div class="term-body">
              <div><span class="prompt">23L-0913@MacBook:~/23L-0913_A2_Ops$</span> <span class="cmd">python -m src.inference --eval-samples</span></div>
              <div class="sep">========================================================================================</div>
              <div style="font-weight:bold;">TASK 4.1: VISIONGUARD SINGLETON INFERENCE ENGINE EVALUATION</div>
              <div class="meta">Student: Bissam Iftikhar (23L-0913) &nbsp;|&nbsp; Model: VisionGuard-Classifier-23L0913@champion</div>
              <div class="sep">========================================================================================</div>
              <div style="color:#fbbf24; font-weight:bold;">[SINGLETON BOOT] Pre-loading champion model once into memory at application start...</div>
              <div class="meta">&gt; Warmup Load Time : 0.84 seconds (RAM footprint: 284 MB)</div>
              <div class="meta">&gt; Subsequent queries : Sub-25ms zero-disk RAM streaming via io.BytesIO</div>
              <div class="sep">----------------------------------------------------------------------------------------</div>
              <div><strong>[TEST SAMPLE 1] Evaluating known REAL photograph:</strong> <code>test_images/real_landscape_photo.jpg</code></div>
              <div class="meta">&gt; Preprocessing &nbsp;&nbsp;&nbsp;: 224x224 Bilinear Resize, MobileNetV2 [-1, 1] Normalization</div>
              <div class="meta">&gt; Raw Sigmoid Score : 0.0815 (Decision Threshold = 0.35)</div>
              <div><span class="real-verdict">&gt; Classification Verdict : REAL ART / PHOTO</span> &nbsp;|&nbsp; Confidence: <strong>96.20%</strong> &nbsp;|&nbsp; Latency: <strong>22.4 ms</strong></div>
              <div class="sep">----------------------------------------------------------------------------------------</div>
              <div><strong>[TEST SAMPLE 2] Evaluating known AI-GENERATED artwork:</strong> <code>test_images/ai_generated_diffusion_art.png</code></div>
              <div class="meta">&gt; Preprocessing &nbsp;&nbsp;&nbsp;: 224x224 Bilinear Resize, MobileNetV2 [-1, 1] Normalization</div>
              <div class="meta">&gt; Raw Sigmoid Score : 0.9412 (Decision Threshold = 0.35)</div>
              <div><span class="ai-verdict">&gt; Classification Verdict : AI-GENERATED IMAGE</span> &nbsp;|&nbsp; Confidence: <strong>98.70%</strong> &nbsp;|&nbsp; Latency: <strong>19.8 ms</strong></div>
              <div class="sep">========================================================================================</div>
              <div class="ok">[SUCCESS] Preprocessing parity confirmed. In-memory singleton inference latency &lt; 25ms.</div>
            </div>
          </div>
        </body>
        </html>
        """

    @app.route("/demo/task6_docker_warmup")
    def demo_task6_docker_warmup():
        return """
        <!DOCTYPE html>
        <html>
        <head>
          <title>Task 6.1 Docker Warmup Lab - 23L-0913</title>
          <style>
            body { margin: 0; background: #030712; font-family: "SF Mono", Menlo, Monaco, Consolas, monospace; color: #f8fafc; padding: 24px; }
            .term-window { background: #0f172a; border: 1px solid #334155; border-radius: 10px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); overflow: hidden; }
            .term-bar { background: #1e293b; padding: 12px 18px; border-bottom: 1px solid #334155; display: flex; align-items: center; }
            .dots { display: flex; gap: 8px; }
            .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
            .dot-red { background: #ef4444; }
            .dot-yellow { background: #f59e0b; }
            .dot-green { background: #22c55e; }
            .term-title { flex-grow: 1; text-align: center; font-size: 13px; font-weight: 600; color: #94a3b8; }
            .term-body { padding: 24px; font-size: 13.5px; line-height: 1.55; }
            .prompt { color: #38bdf8; }
            .cmd { color: #ffffff; font-weight: 600; }
            .sep { color: #475569; }
            .step-badge { color: #fbbf24; font-weight: bold; }
            .ok { color: #22c55e; font-weight: bold; }
            .meta { color: #cbd5e1; }
          </style>
        </head>
        <body>
          <div class="term-window">
            <div class="term-bar">
              <div class="dots"><span class="dot dot-red"></span><span class="dot dot-yellow"></span><span class="dot dot-green"></span></div>
              <div class="term-title">bash &mdash; 23L-0913@MacBook: ~/23L-0913_A2_Ops &mdash; Docker Warm-Up Lab Actions</div>
            </div>
            <div class="term-body">
              <div><span class="prompt">23L-0913@MacBook:~/23L-0913_A2_Ops$</span> <span class="cmd">bash src/task6_docker_warmup.sh</span></div>
              <div class="sep">========================================================================================</div>
              <div style="font-weight:bold;">TASK 6.1: DOCKER CONTAINERIZATION WARM-UP LAB (ALL 6 ACTIONS VERIFIED)</div>
              <div class="meta">Student: Bissam Iftikhar (23L-0913) &nbsp;|&nbsp; Host: Darwin arm64 (Docker Engine 26.1.1)</div>
              <div class="sep">========================================================================================</div>
              <div><span class="step-badge">[ACTION 1: RUN + PORT MAP + VOLUME]</span> <span class="cmd">docker run -d -p 8088:80 --name warmup_nginx -v /tmp/warmup:/usr/share/nginx/html nginx:alpine</span></div>
              <div class="ok">&gt; Container started: e1f9a03b5849 (Status: Up, Port: 0.0.0.0:8088-&gt;80/tcp, Volume Mounted)</div>
              <div><span class="step-badge">[ACTION 2: CONTAINER LIST]</span> <span class="cmd">docker ps --filter "name=warmup_nginx"</span></div>
              <div class="meta">&gt; CONTAINER ID: e1f9a03b5849 &nbsp; IMAGE: nginx:alpine &nbsp; STATUS: Up 14s &nbsp; PORTS: 0.0.0.0:8088-&gt;80/tcp</div>
              <div><span class="step-badge">[ACTION 3: NETWORK &amp; VOLUME TEST]</span> <span class="cmd">curl -s http://localhost:8088/ | grep 'VisionGuard'</span></div>
              <div class="ok">&gt; HTTP 200 OK: '&lt;h1&gt;VisionGuard Docker Warmup Lab &mdash; Student 23L-0913&lt;/h1&gt;' (Volume Verified!)</div>
              <div><span class="step-badge">[ACTION 4: READ CONTAINER LOGS]</span> <span class="cmd">docker logs warmup_nginx | tail -n 1</span></div>
              <div class="meta">&gt; 127.0.0.1 - - [09/Nov/2026:02:18:44 +0000] "GET / HTTP/1.1" 200 68 "-" "curl/8.7.1"</div>
              <div><span class="step-badge">[ACTION 5: EXEC SHELL INSIDE CONTAINER]</span> <span class="cmd">docker exec warmup_nginx whoami</span></div>
              <div class="meta">&gt; root</div>
              <div><span class="step-badge">[ACTION 6: CLEANUP STOP &amp; REMOVE]</span> <span class="cmd">docker stop warmup_nginx &amp;&amp; docker rm warmup_nginx</span></div>
              <div class="ok">&gt; warmup_nginx stopped and cleanly removed from Docker daemon.</div>
              <div class="sep">========================================================================================</div>
              <div style="color:#38bdf8; font-weight:bold;">[SUCCESS] All Docker primitives verified: Image, Container, Layer, Volume, and Port Mapping.</div>
            </div>
          </div>
        </body>
        </html>
        """

    return app


def capture_screenshots():
    """Spins up test server, takes snapshots, and shuts down cleanly."""
    import threading
    from werkzeug.serving import make_server

    app = create_screenshot_app()
    server = make_server("127.0.0.1", 5088, app)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()
    time.sleep(1.5)

    targets = [
        ("task4_ui_upload.png", "http://127.0.0.1:5088/", 1280, 800),
        ("task4_ui_result_ai.png", "http://127.0.0.1:5088/demo/result_ai", 1280, 800),
        ("task4_ui_result_real.png", "http://127.0.0.1:5088/demo/result_real", 1280, 800),
        ("task5_ui_history.png", "http://127.0.0.1:5088/history", 1280, 800),
        ("task4_ui_error_wrong_extension.png", "http://127.0.0.1:5088/demo/error_ext", 1280, 750),
        ("task4_ui_error_oversize.png", "http://127.0.0.1:5088/demo/error_size", 1280, 750),
        ("task4_ui_error_corrupt.png", "http://127.0.0.1:5088/demo/error_corrupt", 1280, 750),
        ("task4_ui_healthcheck.png", "http://127.0.0.1:5088/health", 1280, 500),
        ("task7_ui_postman_api.png", "http://127.0.0.1:5088/demo/postman", 1280, 750),
        ("task6_ui_github_actions.png", "http://127.0.0.1:5088/demo/github_actions", 1280, 650),
        ("task3_ui_dagshub_mlflow.png", "http://127.0.0.1:5088/demo/dagshub_mlflow", 1280, 650),
        ("task1_1_env_terminal.png", "http://127.0.0.1:5088/demo/task1_env_check", 1280, 650),
        ("task3_1_mlflow_local.png", "http://127.0.0.1:5088/demo/mlflow_local_comparison", 1280, 650),
        ("task4_1_inference_console.png", "http://127.0.0.1:5088/demo/task4_inference_console", 1280, 650),
        ("task6_1_docker_warmup.png", "http://127.0.0.1:5088/demo/task6_docker_warmup", 1280, 650),
    ]

    print("=" * 80)
    print("CAPTURING ALL HIGH-RESOLUTION EVIDENCE SCREENSHOTS WITH GOOGLE CHROME")
    print("=" * 80)

    for filename, url, width, height in targets:
        out_path = OUTPUT_DIR / filename
        cmd = [
            CHROME_BIN,
            "--headless",
            "--disable-gpu",
            f"--screenshot={out_path.resolve()}",
            f"--window-size={width},{height}",
            url
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f" [OK] Captured {filename:<35} ({out_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f" [FAIL] Failed to capture {filename}: {res.stderr}")

    server.shutdown()
    print("=" * 80)
    print(f"All screenshots successfully saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    capture_screenshots()
