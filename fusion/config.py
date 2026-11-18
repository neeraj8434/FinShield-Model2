"""Centralised configuration for the fusion pipeline.

All paths and thresholds are configurable via environment variables so
nothing needs to change between dev-laptop and demo-day setups.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Repository roots ─────────────────────────────────────────────────
MODEL2_ROOT = Path(os.getenv(
    "FINSHIELD_MODEL2_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
))

MODEL1_ROOT = Path(os.getenv(
    "FINSHIELD_MODEL1_ROOT",
    r"C:\Users\Shloka Pol\OneDrive\Desktop\fin_modal1",
))

# ── Checkpoints ──────────────────────────────────────────────────────
MODEL1_CHECKPOINT = os.getenv(
    "FINSHIELD_MODEL1_CHECKPOINT",
    str(MODEL1_ROOT / "checkpoints" / "best_model.pt"),
)

MODEL1_ADV_CHECKPOINT = os.getenv(
    "FINSHIELD_MODEL1_ADV_CHECKPOINT",
    str(MODEL1_ROOT / "checkpoints" / "best_model_adversarial.pt"),
)

MODEL2_CHECKPOINT = os.getenv(
    "FINSHIELD_MODEL2_CHECKPOINT",
    str(MODEL2_ROOT / "checkpoints" / "gated_rich.pt"),
)

# ── Calibration ──────────────────────────────────────────────────────
MODEL2_CALIBRATION = os.getenv(
    "FINSHIELD_MODEL2_CALIBRATION",
    str(MODEL2_ROOT / "checkpoints" / "gated_rich.calibration.json"),
)

# ── Face-match ONNX models (OpenCV zoo) ──────────────────────────────
FACE_DETECT_ONNX = os.getenv(
    "FINSHIELD_FACE_DETECT_ONNX",
    str(MODEL2_ROOT / "models" / "face_detection_yunet_2023mar.onnx"),
)

FACE_RECOGNISE_ONNX = os.getenv(
    "FINSHIELD_FACE_RECOGNISE_ONNX",
    str(MODEL2_ROOT / "models" / "face_recognition_sface_2021dec.onnx"),
)

# ── Fusion thresholds ────────────────────────────────────────────────
# Noisy-OR risk thresholds
APPROVE_THRESHOLD = float(os.getenv("FINSHIELD_APPROVE_THRESHOLD", "0.35"))
REVIEW_THRESHOLD = float(os.getenv("FINSHIELD_REVIEW_THRESHOLD", "0.65"))
# Above REVIEW_THRESHOLD → REJECT; between APPROVE and REVIEW → REVIEW

# Minimum reliability for auto-APPROVE
MIN_RELIABILITY_APPROVE = float(os.getenv("FINSHIELD_MIN_RELIABILITY", "0.50"))

# Face-match cosine similarity threshold (SFace)
FACE_MATCH_THRESHOLD = float(os.getenv("FINSHIELD_FACE_MATCH_THRESHOLD", "0.363"))

# OCR fuzzy match ratio threshold
OCR_FUZZY_THRESHOLD = float(os.getenv("FINSHIELD_OCR_FUZZY_THRESHOLD", "0.75"))

# ── Model 2 enhanced inference ────────────────────────────────────────
# Tuned threshold (split-half on test set: 78.52% vs 76.85% at 0.5)
MODEL2_THRESHOLD = float(os.getenv("FINSHIELD_MODEL2_THRESHOLD", "0.575"))

# More frames for LSTM temporal signal (default was 20)
MODEL2_NUM_FRAMES = int(os.getenv("FINSHIELD_MODEL2_FRAMES", "32"))

# Test-time augmentation: horizontal flip + average
MODEL2_TTA = os.getenv("FINSHIELD_MODEL2_TTA", "1") == "1"

# ── Device ───────────────────────────────────────────────────────────
DEVICE = os.getenv("FINSHIELD_DEVICE", "cpu")

# ── Flask / demo ─────────────────────────────────────────────────────
UPLOAD_FOLDER = os.getenv(
    "FINSHIELD_UPLOAD_FOLDER",
    str(MODEL2_ROOT / "fusion" / "uploads"),
)

MAX_CONTENT_LENGTH = int(os.getenv("FINSHIELD_MAX_UPLOAD_MB", "100")) * 1024 * 1024
