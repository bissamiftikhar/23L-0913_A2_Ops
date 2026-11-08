#!/usr/bin/env python3
"""
Task 1.2 & 1.3: Dataset Download, Audit, Split Generation, and Visual EDA
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

1. Downloads 'cashbowman/ai-generated-images-vs-real-images' from Kaggle (via kagglehub)
   and organizes/symlinks into data/raw/{AiArtData,RealArt}.
2. Walks the dataset and computes for every file:
   class, file_size_bytes, file_size_kb, width, height, color_mode, format, sha256_hash, is_corrupt.
3. Detects unreadable/corrupt files and exact content duplicates (via SHA-256).
4. Produces summary audit tables (counts per class; min, median, max of width, height, file size;
   corrupt files; exact duplicates).
5. Generates Visual EDA figures:
   - Grid of sample images per class (4x4 grid: 8 Real, 8 AI-Generated)
   - Histograms of width, height, and file size (KB) per class
6. Creates deterministic stratified Train / Val / Test split lists (70% / 15% / 15%) on clean,
   deduplicated images with fixed seed (42) for Task 2.
"""

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import kagglehub

SEED = 42
RAW_DATA_DIR = Path("data/raw")
SPLITS_DIR = Path("data/splits")
ARTIFACTS_DIR = Path("artifacts/task1")

CLASS_MAP = {
    "RealArt": "real",
    "Real": "real",
    "real": "real",
    "AiArtData": "ai_generated",
    "AI": "ai_generated",
    "ai_generated": "ai_generated",
}

def ensure_dataset() -> Path:
    """Download dataset via kagglehub if not already in data/raw."""
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        print(f"[INFO] Found existing dataset in {RAW_DATA_DIR}")
        return RAW_DATA_DIR

    print("[INFO] Downloading 'cashbowman/ai-generated-images-vs-real-images' via kagglehub...")
    cache_path = Path(kagglehub.dataset_download("cashbowman/ai-generated-images-vs-real-images"))
    print(f"[INFO] Downloaded to cache: {cache_path}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Copy class folders into data/raw
    for item in cache_path.iterdir():
        if item.is_dir():
            dest = RAW_DATA_DIR / item.name
            if not dest.exists():
                shutil.copytree(item, dest)
    print(f"[INFO] Copied dataset into {RAW_DATA_DIR}")
    return RAW_DATA_DIR

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def audit_dataset(data_dir: Path) -> pd.DataFrame:
    """Walk dataset directory and audit every file."""
    records = []
    for root, _, files in os.walk(data_dir):
        root_path = Path(root)
        # Skip hidden directories
        if any(part.startswith(".") for part in root_path.parts):
            continue
        # Determine class from immediate subfolder under data/raw
        rel_parts = root_path.relative_to(data_dir).parts
        if not rel_parts:
            continue
        raw_folder = rel_parts[0]
        label_class = CLASS_MAP.get(raw_folder, raw_folder)

        for fname in sorted(files):
            if fname.startswith("."):
                continue
            fpath = root_path / fname
            rel_path = str(fpath)
            file_size_bytes = fpath.stat().st_size
            sha256_hash = compute_sha256(fpath)

            width, height, color_mode, img_format = None, None, None, None
            is_corrupt = False
            error_msg = ""

            try:
                with Image.open(fpath) as img:
                    img_format = img.format
                    color_mode = img.mode
                    width, height = img.size
                    # Force full decode to catch truncated/corrupted byte streams
                    img.load()
            except Exception as exc:
                is_corrupt = True
                error_msg = str(exc)

            records.append({
                "filepath": rel_path,
                "filename": fname,
                "folder": raw_folder,
                "class": label_class,
                "file_size_bytes": file_size_bytes,
                "file_size_kb": round(file_size_bytes / 1024.0, 2),
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4) if (width and height) else None,
                "color_mode": color_mode,
                "format": img_format,
                "sha256": sha256_hash,
                "is_corrupt": is_corrupt,
                "error_msg": error_msg,
            })

    df = pd.DataFrame(records)
    # Mark duplicates based on exact SHA-256 content hash
    df["is_duplicate"] = df.duplicated(subset=["sha256"], keep="first")
    return df

def generate_summary_tables(df: pd.DataFrame):
    """Build and save summary statistics per class and overall."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ARTIFACTS_DIR / "dataset_audit_full.csv", index=False)

    valid_df = df[~df["is_corrupt"]].copy()

    summary_rows = []
    for cls_name, group in df.groupby("class"):
        valid_grp = group[~group["is_corrupt"]]
        summary_rows.append({
            "Class": cls_name,
            "Total Files": int(len(group)),
            "Valid Files": int(len(valid_grp)),
            "Corrupt Files": int(group["is_corrupt"].sum()),
            "Exact Duplicates (SHA-256)": int(group["is_duplicate"].sum()),
            "Width Min": int(valid_grp["width"].min()),
            "Width Median": float(valid_grp["width"].median()),
            "Width Max": int(valid_grp["width"].max()),
            "Height Min": int(valid_grp["height"].min()),
            "Height Median": float(valid_grp["height"].median()),
            "Height Max": int(valid_grp["height"].max()),
            "Size (KB) Min": round(float(valid_grp["file_size_kb"].min()), 2),
            "Size (KB) Median": round(float(valid_grp["file_size_kb"].median()), 2),
            "Size (KB) Max": round(float(valid_grp["file_size_kb"].max()), 2),
        })

    # Overall row
    summary_rows.append({
        "Class": "ALL (Total)",
        "Total Files": int(len(df)),
        "Valid Files": int(len(valid_df)),
        "Corrupt Files": int(df["is_corrupt"].sum()),
        "Exact Duplicates (SHA-256)": int(df["is_duplicate"].sum()),
        "Width Min": int(valid_df["width"].min()),
        "Width Median": float(valid_df["width"].median()),
        "Width Max": int(valid_df["width"].max()),
        "Height Min": int(valid_df["height"].min()),
        "Height Median": float(valid_df["height"].median()),
        "Height Max": int(valid_df["height"].max()),
        "Size (KB) Min": round(float(valid_df["file_size_kb"].min()), 2),
        "Size (KB) Median": round(float(valid_df["file_size_kb"].median()), 2),
        "Size (KB) Max": round(float(valid_df["file_size_kb"].max()), 2),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ARTIFACTS_DIR / "task1_2_audit_summary.csv", index=False)

    # Color mode and format distributions
    mode_counts = valid_df.groupby(["class", "color_mode"]).size().unstack(fill_value=0)
    fmt_counts = valid_df.groupby(["class", "format"]).size().unstack(fill_value=0)

    # Duplicate details
    dup_df = df[df["duplicated_group"] if "duplicated_group" in df else df.duplicated(subset=["sha256"], keep=False)].sort_values("sha256")
    dup_df.to_csv(ARTIFACTS_DIR / "task1_2_duplicates.csv", index=False)

    report_info = {
        "total_files": int(len(df)),
        "valid_files": int(len(valid_df)),
        "corrupt_files": int(df["is_corrupt"].sum()),
        "corrupt_file_list": df[df["is_corrupt"]][["filepath", "class", "error_msg"]].to_dict(orient="records"),
        "exact_duplicates": int(df["is_duplicate"].sum()),
        "duplicate_pairs_count": int(len(dup_df)),
        "color_modes": mode_counts.to_dict(),
        "formats": fmt_counts.to_dict(),
        "summary_table": summary_rows,
    }
    with open(ARTIFACTS_DIR / "task1_2_audit_report.json", "w") as f:
        json.dump(report_info, f, indent=2)

    print("\n" + "=" * 88)
    print("TASK 1.2 DATASET AUDIT SUMMARY TABLE (23L-0913_A2_Ops)")
    print("=" * 88)
    print(summary_df.to_string(index=False))
    print("-" * 88)
    print("Colour Mode Breakdown:\n", mode_counts)
    print("-" * 88)
    print("Image Format Breakdown:\n", fmt_counts)
    print("-" * 88)
    if df["is_corrupt"].sum() > 0:
        print("Corrupt / Unreadable Files Detected:")
        for r in report_info["corrupt_file_list"]:
            print(f"  - {r['filepath']} ({r['class']}): {r['error_msg']}")
    if df["is_duplicate"].sum() > 0:
        print(f"Exact SHA-256 Duplicates Detected: {df['is_duplicate'].sum()} redundant files")
        for sha, grp in dup_df.groupby("sha256"):
            paths = list(grp["filepath"])
            print(f"  - SHA256 {sha[:12]}... -> {paths}")
    print("=" * 88 + "\n")
    return summary_df

def generate_visual_eda(df: pd.DataFrame):
    """Task 1.3: Generate sample image grid per class and histograms of width, height, file size."""
    valid_df = df[(~df["is_corrupt"]) & (~df["is_duplicate"])].copy()
    np.random.seed(SEED)

    # 1. Sample Grid (2 rows of Real, 2 rows of AI-Generated -> 4x4 grid)
    fig, axes = plt.subplots(4, 4, figsize=(13, 12))
    fig.suptitle(
        "VisionGuard (23L-0913_A2_Ops) — Sample Images per Class (Top 2 Rows: Real | Bottom 2 Rows: AI-Generated)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    real_samples = valid_df[valid_df["class"] == "real"].sample(8, random_state=SEED)
    ai_samples = valid_df[valid_df["class"] == "ai_generated"].sample(8, random_state=SEED)
    combined_samples = pd.concat([real_samples, ai_samples], ignore_index=True)

    for idx, row in combined_samples.iterrows():
        r, c = divmod(idx, 4)
        ax = axes[r, c]
        with Image.open(row["filepath"]) as img:
            img_rgb = img.convert("RGB")
            ax.imshow(img_rgb)
        cls_label = "Real Photo/Art" if row["class"] == "real" else "AI-Generated"
        color = "#1b7837" if row["class"] == "real" else "#b2182b"
        ax.set_title(
            f"{cls_label}\n{int(row['width'])}x{int(row['height'])} | {row['file_size_kb']:.0f} KB | {row['color_mode']}",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = ARTIFACTS_DIR / "task1_3_sample_grid.png"
    plt.savefig(grid_path, dpi=180)
    plt.close()
    print(f"[INFO] Saved sample image grid to {grid_path}")

    # 2. Histograms of Width, Height, and File Size per class
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "VisionGuard (23L-0913_A2_Ops) — Distributions of Width, Height, and File Size per Class",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    palette = {"real": "#2b83ba", "ai_generated": "#d7191c"}

    sns.histplot(
        data=valid_df, x="width", hue="class", palette=palette,
        bins=30, kde=True, ax=axes[0], alpha=0.55
    )
    axes[0].set_title("Image Width Distribution (px)", fontweight="bold")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Image Count")

    sns.histplot(
        data=valid_df, x="height", hue="class", palette=palette,
        bins=30, kde=True, ax=axes[1], alpha=0.55
    )
    axes[1].set_title("Image Height Distribution (px)", fontweight="bold")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Image Count")

    sns.histplot(
        data=valid_df, x="file_size_kb", hue="class", palette=palette,
        bins=30, kde=True, ax=axes[2], alpha=0.55, log_scale=(True, False)
    )
    axes[2].set_title("File Size Distribution (KB, Log Scale)", fontweight="bold")
    axes[2].set_xlabel("File Size (KB, log scale)")
    axes[2].set_ylabel("Image Count")

    plt.tight_layout()
    hist_path = ARTIFACTS_DIR / "task1_3_histograms.png"
    plt.savefig(hist_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved histograms to {hist_path}")

def create_splits(df: pd.DataFrame):
    """Create clean, deduplicated stratified Train (70%), Val (15%), Test (15%) splits."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    clean_df = df[(~df["is_corrupt"]) & (~df["is_duplicate"])].copy()

    # Check if any image fails TensorFlow's strict JPEG/PNG decoder or conversion to RGB
    # We pre-convert or ensure clean RGB readable images for tf.data
    clean_df["label"] = (clean_df["class"] == "ai_generated").astype(int)

    train_df, temp_df = train_test_split(
        clean_df, test_size=0.30, stratify=clean_df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    cols = ["filepath", "class", "label", "width", "height", "file_size_kb", "sha256"]
    train_df[cols].to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df[cols].to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df[cols].to_csv(SPLITS_DIR / "test.csv", index=False)

    print(
        f"[INFO] Created stratified splits (seed={SEED}): "
        f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

if __name__ == "__main__":
    data_dir = ensure_dataset()
    df_audit = audit_dataset(data_dir)
    generate_summary_tables(df_audit)
    generate_visual_eda(df_audit)
    create_splits(df_audit)

