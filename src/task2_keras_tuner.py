#!/usr/bin/env python3
"""
Task 2.7: Keras Tuner Toy Lab (Fashion-MNIST) & VisionGuard Hyperparameter Tuning
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

1. Toy Part:
   - Built-in dataset: Fashion-MNIST (10 clothing classes)
   - Search space: Dense units (32, 64, 128) and Learning Rate (1e-2, 1e-3, 1e-4)
   - Tuner: Hyperband / RandomSearch (max_epochs=4, factor=2, seed=42)
   - Outputs: Toy tuning results table (trial ID, units, lr, val_accuracy, val_loss)

2. Apply to VisionGuard (at least 8 trials):
   - Search space:
     * Learning rate: [1e-3, 5e-4, 1e-4, 5e-5]
     * Head Dropout: [0.20, 0.35, 0.50]
     * Dense Head Units: [64, 128, 256]
     * Unfrozen layers: [top 15 layers vs top 25 layers vs frozen]
   - Tuner: RandomSearch (max_trials=8, seed=42)
   - Outputs: VisionGuard tuning summary table and best hyperparameter configuration
   - Detailed commentary on why tuning on the validation set can overfit it and which setting mattered most.
"""

import os
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models, optimizers
import keras_tuner as kt

SEED = 42
ARTIFACTS_DIR = Path("artifacts/task2")


def run_toy_fashion_mnist_tuning():
    """Toy part: tune a tiny network on Fashion-MNIST with Keras Tuner."""
    print("=" * 78)
    print("TASK 2.7: KERAS TUNER TOY PART (FASHION-MNIST)")
    print("=" * 78)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    toy_tuner_dir = Path("toy_tuner_dir")

    # Load Fashion-MNIST
    (x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
    # Use subset for fast, clean demonstration
    x_train = x_train[:4000].astype("float32") / 255.0
    y_train = y_train[:4000]
    x_val = x_val[:1000].astype("float32") / 255.0
    y_val = y_val[:1000]

    def build_toy_model(hp: kt.HyperParameters) -> keras.Model:
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(
                units=hp.Choice("units", values=[32, 64, 128]),
                activation="relu",
            ),
            layers.Dropout(rate=hp.Float("dropout", 0.1, 0.4, step=0.1)),
            layers.Dense(10, activation="softmax"),
        ])
        lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = kt.Hyperband(
        build_toy_model,
        objective="val_accuracy",
        max_epochs=4,
        factor=2,
        seed=SEED,
        directory=str(toy_tuner_dir),
        project_name="fashion_mnist_toy",
        overwrite=True,
    )

    tuner.search(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=3,
        batch_size=64,
        verbose=1,
    )

    # Extract all trials
    trials = tuner.oracle.get_best_trials(num_trials=10)
    toy_results = []
    for i, t in enumerate(trials):
        hp_vals = t.hyperparameters.values
        score = t.score
        metrics = t.metrics.metrics
        val_loss = metrics.get("val_loss").get_best_value() if "val_loss" in metrics else None
        toy_results.append({
            "Rank": i + 1,
            "Trial ID": t.trial_id,
            "Units": hp_vals.get("units"),
            "Dropout": hp_vals.get("dropout"),
            "Learning Rate": hp_vals.get("learning_rate"),
            "Val Accuracy": round(float(score), 4) if score is not None else None,
            "Val Loss": round(float(val_loss), 4) if val_loss is not None else None,
        })

    toy_df = pd.DataFrame(toy_results)
    toy_df.to_csv(ARTIFACTS_DIR / "task2_7_toy_results.csv", index=False)
    print("\n--- Fashion-MNIST Toy Tuner Results Table ---")
    print(toy_df.to_string(index=False))

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"\nBest Toy Settings: Units={best_hp.get('units')}, LR={best_hp.get('learning_rate')}, Dropout={best_hp.get('dropout')}")
    return toy_df


def run_visionguard_tuning():
    """Apply part: Tune VisionGuard MobileNetV2 with at least 8 trials."""
    print("\n" + "=" * 78)
    print("TASK 2.7: VISIONGUARD HYPERPARAMETER TUNING (8+ TRIALS)")
    print("=" * 78)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vg_tuner_dir = Path("keras_tuner_dir")

    from src.task2_pipeline import build_tf_dataset

    train_ds, _ = build_tf_dataset("data/splits/train.csv", is_training=True, augment=True, batch_size=32, preprocess_mode="mobilenet_v2")
    val_ds, _ = build_tf_dataset("data/splits/val.csv", is_training=False, augment=False, batch_size=32, preprocess_mode="mobilenet_v2")

    # Take subset batches for tuning search efficiency
    tune_train_ds = train_ds.take(25)
    tune_val_ds = val_ds.take(10)

    base_mobilenet = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    def build_vg_model(hp: kt.HyperParameters) -> keras.Model:
        base_mobilenet.trainable = False

        # Tunable choice: unfreeze top blocks
        unfreeze_choice = hp.Choice("unfreeze_layers", values=["top_0", "top_15", "top_30"])
        if unfreeze_choice == "top_15":
            base_mobilenet.trainable = True
            for layer in base_mobilenet.layers[:-15]:
                layer.trainable = False
        elif unfreeze_choice == "top_30":
            base_mobilenet.trainable = True
            for layer in base_mobilenet.layers[:-30]:
                layer.trainable = False

        dense_units = hp.Choice("dense_units", values=[64, 128, 256])
        dropout_rate = hp.Choice("dropout_rate", values=[0.20, 0.35, 0.50])
        lr = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4, 5e-5])

        inputs = layers.Input(shape=(224, 224, 3), name="image_input")
        x = base_mobilenet(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="VisionGuard_Tuner")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        return model

    tuner = kt.RandomSearch(
        build_vg_model,
        objective="val_accuracy",
        max_trials=8,
        seed=SEED,
        directory=str(vg_tuner_dir),
        project_name="visionguard_tuning",
        overwrite=True,
    )

    tuner.search(
        tune_train_ds,
        validation_data=tune_val_ds,
        epochs=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
        verbose=1,
    )

    trials = tuner.oracle.get_best_trials(num_trials=8)
    vg_results = []
    for i, t in enumerate(trials):
        hp_vals = t.hyperparameters.values
        score = t.score
        metrics = t.metrics.metrics
        val_auc = metrics.get("val_auc").get_best_value() if "val_auc" in metrics else None
        val_loss = metrics.get("val_loss").get_best_value() if "val_loss" in metrics else None
        vg_results.append({
            "Trial": i + 1,
            "Trial ID": t.trial_id,
            "Unfrozen Layers": hp_vals.get("unfreeze_layers"),
            "Dense Units": hp_vals.get("dense_units"),
            "Dropout": hp_vals.get("dropout_rate"),
            "Learning Rate": hp_vals.get("learning_rate"),
            "Val Accuracy": round(float(score), 4) if score is not None else None,
            "Val AUC": round(float(val_auc), 4) if val_auc is not None else None,
            "Val Loss": round(float(val_loss), 4) if val_loss is not None else None,
        })

    vg_df = pd.DataFrame(vg_results)
    vg_df.to_csv(ARTIFACTS_DIR / "task2_7_visionguard_tuner_results.csv", index=False)
    print("\n--- VisionGuard 8-Trial Hyperparameter Results Table ---")
    print(vg_df.to_string(index=False))

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_config = best_hp.values
    with open(ARTIFACTS_DIR / "task2_7_best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\n[INFO] Best VisionGuard Hyperparameters: {best_config}")
    return vg_df


if __name__ == "__main__":
    run_toy_fashion_mnist_tuning()
    if Path("data/splits/train.csv").exists():
        run_visionguard_tuning()
