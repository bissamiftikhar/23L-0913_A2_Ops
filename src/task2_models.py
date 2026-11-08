#!/usr/bin/env python3
"""
Task 2.2 & 2.3: Model Architectures for VisionGuard
Project: 23L-0913_A2_Ops (VisionGuard)
Author: Bissam Iftikhar (23L-0913)

Defines:
1. Baseline CNN (Task 2.2):
   - Convolution (Conv2D with ReLU)
   - Batch Normalization (stabilizes and accelerates convergence)
   - Max Pooling (downsamples spatial dimensions)
   - Dropout (regularization to combat overfitting)
   - GlobalAveragePooling2D + Dense head
   - Single Sigmoid output neuron (binary classification: 0=Real, 1=AI-Generated)
2. Transfer Learning with MobileNetV2 (Task 2.3):
   - Pretrained ImageNet weights, include_top=False
   - Frozen base during Phase 1 (feature extraction)
   - Custom classification head: GlobalAveragePooling2D -> BatchNormalization -> Dense(128, relu) -> Dropout(0.4) -> Dense(1, sigmoid)
   - Fine-tuning Phase 2: Unfreezes top block (layers >= 140) with low learning rate (1e-5).
"""

import os
from typing import Tuple, List, Dict, Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras
from keras import layers, models, regularizers


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    l2_reg: float = 1e-4,
    dropout_rate: float = 0.35,
) -> keras.Model:
    """
    Task 2.2: Build a compact, effective baseline CNN from scratch.

    Architecture rationale:
    - Block 1 (Conv2D 32, 3x3 + BatchNorm + ReLU + MaxPool 2x2): captures low-level textures, edges, and color gradients.
    - Block 2 (Conv2D 64, 3x3 + BatchNorm + ReLU + MaxPool 2x2): captures mid-level image patterns and shapes.
    - Block 3 (Conv2D 128, 3x3 + BatchNorm + ReLU + MaxPool 2x2 + Spatial Dropout): captures high-level structural semantics.
    - Block 4 (Conv2D 128, 3x3 + BatchNorm + ReLU + GlobalAveragePooling2D): aggregates global feature maps into a 128-dim vector,
      dramatically reducing parameter count compared to Flatten while preventing spatial overfitting.
    - Dense Head: Dense(64, ReLU) + Dropout(dropout_rate) + Dense(1, Sigmoid).
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.20, name="drop_spatial")(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(l2_reg), name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classification Head
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN_VisionGuard")
    return model


def build_transfer_mobilenetv2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.40,
    dense_units: int = 128,
) -> Tuple[keras.Model, keras.Model]:
    """
    Task 2.3: MobileNetV2 Transfer Learning Model.

    Phase 1: Base model loaded with pretrained ImageNet weights, base.trainable = False.
    Custom classification head attached on top.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="image_input")
    # Base expects normalized [-1, 1] inputs (handled in tf.data pipeline)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_VisionGuard")
    return model, base_model


def unfreeze_mobilenetv2_top_block(
    model: keras.Model,
    base_model: keras.Model,
    unfreeze_from_layer_idx: int = 140,
) -> List[str]:
    """
    Task 2.3 Phase 2 (Fine-Tuning):
    Unfreezes the top inverted residual blocks (from unfreeze_from_layer_idx upwards),
    keeping earlier feature extractors and BatchNormalization running stats frozen.
    """
    base_model.trainable = True

    unfrozen_layer_names = []
    # Freeze all layers before unfreeze_from_layer_idx
    for idx, layer in enumerate(base_model.layers):
        if idx < unfreeze_from_layer_idx or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen_layer_names.append(f"{idx}: {layer.name}")

    return unfrozen_layer_names


def get_layer_table_and_summary(model: keras.Model) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract layer table with layer name, output shape, and parameter count."""
    rows = []
    total_params = 0
    trainable_params = 0

    for l in model.layers:
        p_count = l.count_params()
        t_count = sum(tf.keras.backend.count_params(w) for w in l.trainable_weights)
        total_params += p_count
        trainable_params += t_count
        rows.append({
            "Layer Name": l.name,
            "Type": l.__class__.__name__,
            "Output Shape": str(getattr(l, "output_shape", "Multiple")),
            "Param #": p_count,
            "Trainable": getattr(l, "trainable", True),
        })

    summary_lines = []
    model.summary(print_fn=lambda s: summary_lines.append(s))
    summary_str = "\n".join(summary_lines)
    return summary_str, rows
