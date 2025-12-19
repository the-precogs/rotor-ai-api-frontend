from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import tensorflow as tf

# from tensorflow.keras.models import load_model
import keras


# ===========================
#          CONFIG
# ===========================

# Single model path
MODEL_PATH = Path("app/models/best_inceptiontime_model.keras")

# Normalization params directory
NORM_DIR = Path("app/normalization_params")

# Map from class index to label – MUST match your training order
IDX_TO_LABEL: Dict[int, str] = {
    0: "normal",
    1: "bearing_fault",
    2: "unbalance_fault",
    3: "misalignment_fault",
    4: "mechanical_looseness_fault",
}


# ===========================
#   Load normalization params
# ===========================

try:
    TIME_MEAN = np.load(NORM_DIR / "time_mean.npy").astype(np.float32)  # (1, 1, 3)
    TIME_STD = np.load(NORM_DIR / "time_std.npy").astype(np.float32)  # (1, 1, 3)
    FREQ_MEAN = np.load(NORM_DIR / "freq_mean.npy").astype(np.float32)  # (1, 1, 3)
    FREQ_STD = np.load(NORM_DIR / "freq_std.npy").astype(np.float32)  # (1, 1, 3)

    # Reshape from (1, 1, 3) -> (1, 3) so it broadcasts with (T, 3)
    TIME_MEAN = TIME_MEAN.reshape(1, 3)
    TIME_STD = TIME_STD.reshape(1, 3)
    FREQ_MEAN = FREQ_MEAN.reshape(1, 3)
    FREQ_STD = FREQ_STD.reshape(1, 3)

    print("[INFO] Loaded normalization parameters from app/normalization_params/")
    print(f"  TIME_MEAN shape: {TIME_MEAN.shape}, TIME_STD shape: {TIME_STD.shape}")
    print(f"  FREQ_MEAN shape: {FREQ_MEAN.shape}, FREQ_STD shape: {FREQ_STD.shape}")

except Exception as e:
    print(f"[ERROR] Could not load normalization parameters: {e}")
    raise


# ===========================
#      Pydantic Schemas
# ===========================


class DualBranchRequest(BaseModel):
    """
    Request body for prediction.

    - time_waveform: (1024, 3)  -> raw waveform [timesteps, channels]
    - fft_magnitude: (513, 3)   -> raw FFT magnitude [freq bins, channels]
    """

    time_waveform: List[List[float]]
    fft_magnitude: List[List[float]]


class PredictionResponse(BaseModel):
    model_name: str
    predicted_class: str
    class_index: int
    probabilities: Optional[Dict[str, float]] = None


# ===========================
#        FastAPI App
# ===========================

app = FastAPI(
    title="Rotating Machine Fault Classification API",
    description=(
        "API that takes raw time-domain and FFT features, "
        "applies training-time standardization, and predicts faults."
    ),
    version="2.2.0",
)

model = None

try:
    model = keras.saving.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
    )
    print(f"[INFO] Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model from {MODEL_PATH}: {e}")


# ===========================
#        Helper Functions
# ===========================


def probs_to_dict(probs: np.ndarray) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=float).flatten()
    return {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(probs)}


def _standardize_inputs(
    time_arr: np.ndarray,
    fft_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the SAME standardization used during training.

    Assumes:
      - time_arr: (1024, 3)
      - fft_arr:  (513, 3)
      - *_MEAN / *_STD are shape (1, 1, 3) and broadcastable.
    """
    # eps = 1e-8  # to avoid division by zero

    # time_arr: (1024, 3) → broadcast with (1, 1, 3)
    time_norm = (time_arr - TIME_MEAN) / (TIME_STD)

    # fft_arr: (513, 3) → broadcast with (1, 1, 3)
    fft_norm = (fft_arr - FREQ_MEAN) / (FREQ_STD)

    return time_norm, fft_norm


def _predict_with_model(model, time_arr: np.ndarray, fft_arr: np.ndarray):
    # Validate shapes
    if time_arr.shape != (1024, 3):
        raise HTTPException(
            status_code=400,
            detail=f"time_waveform must have shape (1024, 3), got {time_arr.shape}",
        )

    if fft_arr.shape != (513, 3):
        raise HTTPException(
            status_code=400,
            detail=f"fft_magnitude must have shape (513, 3), got {fft_arr.shape}",
        )

    # -------- NORMALIZATION STEP (using training stats) --------
    time_arr_norm, fft_arr_norm = _standardize_inputs(time_arr, fft_arr)

    # Add batch dimension
    time_batch = time_arr_norm[np.newaxis, ...]  # (1, 1024, 3)
    fft_batch = fft_arr_norm[np.newaxis, ...]  # (1, 513, 3)

    try:
        preds = model.predict([time_batch, fft_batch])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    class_idx = int(np.argmax(preds, axis=1)[0])
    label = IDX_TO_LABEL.get(class_idx, f"class_{class_idx}")
    probs_dict = probs_to_dict(preds[0])

    return {
        "model_name": getattr(model, "name", "rotor_fault_classifier"),
        "predicted_class": label,
        "class_index": class_idx,
        "probabilities": probs_dict,
    }


# ===========================
#           Routes
# ===========================


@app.get("/")
def root():
    return {
        "message": "Rotating Machine Fault Classification API",
        "model": {
            "loaded": model is not None,
            "path": str(MODEL_PATH),
        },
        "normalization_params": {
            "dir": str(NORM_DIR),
            "time_mean_shape": TIME_MEAN.shape,
            "time_std_shape": TIME_STD.shape,
            "freq_mean_shape": FREQ_MEAN.shape,
            "freq_std_shape": FREQ_STD.shape,
        },
        "usage": {
            "predict": "POST JSON to /predict with time_waveform and fft_magnitude",
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(body: DualBranchRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded on server. Check MODEL_PATH and logs.",
        )

    time_arr = np.asarray(body.time_waveform, dtype=np.float32)  # (1024, 3)
    fft_arr = np.asarray(body.fft_magnitude, dtype=np.float32)  # (513, 3)

    result = _predict_with_model(model, time_arr, fft_arr)
    return PredictionResponse(**result)
