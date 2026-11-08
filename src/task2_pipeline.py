#!/usr/bin/env python3
"""
Task 2.1: tf.data Input Pipeline & Augmentation Visualization
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

Builds a deterministic, high-throughput tf.data pipeline that:
1. Reads stratified split CSV lists (train.csv, val.csv, test.csv)
2. Decodes image bytes into 3-channel RGB tensors and resizes to target size (default 224x224)
3. Applies random augmentation ONLY to the training dataset (RandomFlip, RandomRotation,
   RandomZoom, RandomContrast)
4. Applies model-specific preprocessing (either [0, 1] rescaling for custom CNN or
   keras.applications.mobilenet_v2.preprocess_input [-1, 1] for MobileNetV2)
5. Batches and prefetches with tf.data.AUTOTUNE
6. Saves a 4x4 visual grid of one augmented training batch for the report (Task 2.1).
"""

import os
from pathlib import Path
from typing import Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import keras
from keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
ARTIFACTS_DIR = Path("artifacts/task2")


def set_global_seeds(seed: int = SEED):
    """Fix Python, NumPy, and TensorFlow random seeds for reproducibility (Task 2.4)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def get_augmentation_layer(seed: int = SEED) -> keras.Sequential:
    """Create training-only random augmentation block."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.08, seed=seed),
            layers.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10), seed=seed),
            layers.RandomContrast(0.10, seed=seed),
        ],
        name="data_augmentation",
    )


def _load_and_decode_pil(filepath_bytes, img_height: int, img_width: int):
    """Fallback-safe PIL loader wrapped via tf.py_function to handle any image format/profile cleanly."""
    path_str = filepath_bytes.numpy().decode("utf-8")
    with Image.open(path_str) as img:
        img_rgb = img.convert("RGB").resize((img_width, img_height), Image.Resampling.BILINEAR)
        arr = np.asarray(img_rgb, dtype=np.float32)
    return arr


def parse_image(filepath: tf.Tensor, label: tf.Tensor, img_size: Tuple[int, int]):
    """Decode and resize an image tensor to (H, W, 3) float32 in [0, 255]."""
    h, w = img_size
    img = tf.py_function(
        func=_load_and_decode_pil,
        inp=[filepath, h, w],
        Tout=tf.float32,
    )
    img.set_shape((h, w, 3))
    label = tf.cast(label, tf.float32)
    return img, label


def build_tf_dataset(
    csv_path: str | Path,
    img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    is_training: bool = False,
    augment: bool = False,
    preprocess_mode: str = "rescale",  # "rescale" -> [0,1], "mobilenet_v2" -> [-1,1], "raw" -> [0,255]
    seed: int = SEED,
) -> Tuple[tf.data.Dataset, pd.DataFrame]:
    """
    Build a tf.data.Dataset from a split CSV file.
    Stages:
      1. Read filepaths & binary labels from CSV
      2. Shuffle filepaths (if training) with fixed seed
      3. Decode & resize images in parallel (num_parallel_calls=AUTOTUNE)
      4. Batch into mini-batches of size B
      5. Apply random augmentation (training only, when augment=True)
      6. Apply pixel normalization / backbone preprocessing
      7. Prefetch next batch asynchronously (prefetch(AUTOTUNE))
    """
    df = pd.read_csv(csv_path)
    filepaths = df["filepath"].astype(str).values
    labels = df["label"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda fp, lbl: parse_image(fp, lbl, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size)

    if is_training and augment:
        aug_layer = get_augmentation_layer(seed=seed)
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if preprocess_mode == "rescale":
        ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    elif preprocess_mode == "mobilenet_v2":
        ds = ds.map(
            lambda x, y: (keras.applications.mobilenet_v2.preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    elif preprocess_mode == "efficientnet":
        # EfficientNet in Keras expects [0, 255] float32 directly
        pass

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, df


def save_augmented_batch_figure():
    """Generate and save a 4x4 figure of one augmented training batch for Task 2.1."""
    set_global_seeds(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, _ = build_tf_dataset(
        "data/splits/train.csv",
        img_size=DEFAULT_IMG_SIZE,
        batch_size=16,
        is_training=True,
        augment=True,
        preprocess_mode="rescale",
        seed=SEED,
    )

    images, labels = next(iter(train_ds))
    images_np = np.clip(images.numpy(), 0.0, 1.0)
    labels_np = labels.numpy()

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(
        "VisionGuard (23L-0913_A2_Ops) — One Augmented Training Batch (224x224, Flip/Rot/Zoom/Contrast)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for i in range(16):
        r, c = divmod(i, 4)
        ax = axes[r, c]
        ax.imshow(images_np[i])
        lbl_str = "AI-Generated (1)" if labels_np[i] == 1.0 else "Real Photo/Art (0)"
        color = "#b2182b" if labels_np[i] == 1.0 else "#1b7837"
        ax.set_title(lbl_str, fontsize=9.5, fontweight="bold", color=color)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = ARTIFACTS_DIR / "task2_1_augmented_batch.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[INFO] Saved augmented training batch figure to {out_path}")


if __name__ == "__main__":
    save_augmented_batch_figure()

