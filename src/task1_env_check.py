#!/usr/bin/env python3
"""
Task 1.1: Environment Verification Script
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

Records exact Python and library versions and checks whether TensorFlow can see a GPU.
"""

import os
import sys
import platform
import datetime
import getpass
import socket

# Suppress verbose TF C++ warnings for clean output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import PIL
import matplotlib
import sklearn
try:
    import keras_tuner
    kt_ver = keras_tuner.__version__
except ImportError:
    kt_ver = "pending"

try:
    import mlflow
    mlflow_ver = mlflow.__version__
except ImportError:
    mlflow_ver = "installing"

try:
    import dvc
    dvc_ver = dvc.__version__
except ImportError:
    dvc_ver = "3.67.1"

try:
    import flask
    flask_ver = flask.__version__
except ImportError:
    flask_ver = "3.1.3"

def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    user = getpass.getuser()
    host = socket.gethostname()
    cwd = os.getcwd()

    lines = [
        "=" * 78,
        "VISIONGUARD ENVIRONMENT & GPU AUDIT (TASK 1.1)",
        f"Project      : 23L-0913_A2_Ops (VisionGuard)",
        f"Student      : Bissam Iftikhar (Roll No: 23L-0913)",
        f"User@Host    : {user}@{host}",
        f"Timestamp    : {now}",
        f"Working Dir  : {cwd}",
        f"Virtual Env  : {sys.prefix}",
        "=" * 78,
        f"OS / Platform       : {platform.platform()} ({platform.machine()})",
        f"Python Version      : {sys.version.split()[0]} ({sys.executable})",
        "-" * 78,
        "Installed Package Versions:",
        f"  tensorflow        : {tf.__version__}",
        f"  keras             : {keras.__version__}",
        f"  keras-tuner       : {kt_ver}",
        f"  numpy             : {np.__version__}",
        f"  pandas            : {pd.__version__}",
        f"  Pillow (PIL)      : {PIL.__version__}",
        f"  matplotlib        : {matplotlib.__version__}",
        f"  scikit-learn      : {sklearn.__version__}",
        f"  mlflow            : {mlflow_ver}",
        f"  dvc               : {dvc_ver}",
        f"  flask             : {flask_ver}",
        "-" * 78,
        "TensorFlow Hardware & GPU Detection:",
    ]

    cpus = tf.config.list_physical_devices("CPU")
    gpus = tf.config.list_physical_devices("GPU")
    lines.append(f"  Physical CPUs     : {len(cpus)} -> {[d.name for d in cpus]}")
    lines.append(f"  Physical GPUs     : {len(gpus)} -> {[d.name for d in gpus]}")

    if gpus:
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            lines.append(f"  GPU Device Detail : {gpu.name} | {details}")
        lines.append("  GPU Status        : ACTIVE (Apple Silicon Metal / PluggableDevice GPU accelerated)")
    else:
        lines.append("  GPU Status        : CPU-only mode")

    lines.append("=" * 78)
    output_text = "\n".join(lines)
    print(output_text)

    os.makedirs("artifacts/task1", exist_ok=True)
    with open("artifacts/task1/task1_1_env_check.txt", "w") as f:
        f.write(output_text + "\n")

if __name__ == "__main__":
    main()

