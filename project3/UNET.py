# =====================================================
# 2) U-Net ConvLSTM model (with padding + cropping)
# =====================================================
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, models

from pathlib import Path
# .parent gets the directory containing the file
script_dir = Path(__file__).resolve().parent


# =====================================================
# 1) Metrics and loss
# =====================================================

def psnr_metric(y_true, y_pred):
    # Robust PSNR with dynamic range from batch
    max_val = tf.reduce_max(tf.abs(y_true)) + 1e-6
    return tf.image.psnr(y_true, y_pred, max_val=max_val)

def ssim_metric(y_true, y_pred):
    max_val = tf.reduce_max(tf.abs(y_true)) + 1e-6
    ssim = tf.image.ssim(y_true, y_pred, max_val=max_val)
    return tf.reduce_mean(ssim)

def combined_loss(y_true, y_pred):
    # Mix of MSE, MAE, and (1 - SSIM)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    max_val = tf.reduce_max(tf.abs(y_true)) + 1e-6
    ssim = tf.image.ssim(y_true, y_pred, max_val=max_val)
    ssim_loss = 1.0 - tf.reduce_mean(ssim)

    return 0.4 * mse + 0.4 * mae + 0.2 * ssim_loss

def build_unet_convlstm(seq_len, H, W, C, base=32, dropout=0.1):
    """
    U-Net + ConvLSTM model for gridded precipitation forecasting.

    Input:  (batch, seq_len, H, W, C)
    Output: (batch, H, W, 1)
    """

    # ---- Compute padded spatial size (multiple of 4 for 2 poolings) ----
    H_padded = int(np.ceil(H / 4) * 4)
    W_padded = int(np.ceil(W / 4) * 4)
    pad_bottom = H_padded - H
    pad_right  = W_padded - W

    inp = layers.Input(shape=(seq_len, H, W, C))

    # -------------------------
    # 1) Temporal encoder: ConvLSTM
    # -------------------------
    x = layers.ConvLSTM2D(
        filters=base,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False,        # -> (B, H, W, base)
        activation="tanh",
        recurrent_activation="sigmoid",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # ---- Pad to (H_padded, W_padded) for clean pooling/upsampling ----
    if pad_bottom > 0 or pad_right > 0:
        x = layers.ZeroPadding2D(
            padding=((0, pad_bottom), (0, pad_right))
        )(x)   # -> (B, H_padded, W_padded, base)

    # -------------------------
    # 2) U-Net encoder
    # -------------------------

    # Encoder block 1
    c1 = layers.Conv2D(base, (3, 3), activation="relu", padding="same")(x)
    c1 = layers.Conv2D(base, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)    # -> (H_padded/2, W_padded/2)
    p1 = layers.Dropout(dropout)(p1)

    # Encoder block 2
    c2 = layers.Conv2D(base * 2, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(base * 2, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)    # -> (H_padded/4, W_padded/4)
    p2 = layers.Dropout(dropout)(p2)

    # Bottleneck
    bn = layers.Conv2D(base * 4, (3, 3), activation="relu", padding="same")(p2)
    bn = layers.Conv2D(base * 4, (3, 3), activation="relu", padding="same")(bn)
    bn = layers.Dropout(dropout)(bn)

    # -------------------------
    # 3) U-Net decoder
    # -------------------------

    # Decoder block 2
    u2 = layers.Conv2DTranspose(base * 2, (2, 2), strides=(2, 2), padding="same")(bn)
    u2 = layers.Concatenate()([u2, c2])
    c3 = layers.Conv2D(base * 2, (3, 3), activation="relu", padding="same")(u2)
    c3 = layers.Conv2D(base * 2, (3, 3), activation="relu", padding="same")(c3)
    c3 = layers.Dropout(dropout)(c3)

    # Decoder block 1
    u1 = layers.Conv2DTranspose(base, (2, 2), strides=(2, 2), padding="same")(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = layers.Conv2D(base, (3, 3), activation="relu", padding="same")(u1)
    c4 = layers.Conv2D(base, (3, 3), activation="relu", padding="same")(c4)
    c4 = layers.Dropout(dropout)(c4)

    # -------------------------
    # 4) Output layer (linear, no [0,1] squashing)
    # -------------------------
    out = layers.Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding="same",
        activation="linear",
        name="precip_out",
    )(c4)   # shape: (B, H_padded, W_padded, 1)

    # ---- Crop back to original (H, W) if padded ----
    if pad_bottom > 0 or pad_right > 0:
        out = layers.Cropping2D(
            cropping=((0, pad_bottom), (0, pad_right))
        )(out)  # -> (B, H, W, 1)

    model = models.Model(inputs=inp, outputs=out, name="UNet_ConvLSTM_precip")

    return model

