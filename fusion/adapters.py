"""Thin adapters that call Model 1 and Model 2 and return fusion schemas.

Each adapter:
  1. Calls the respective model's predict function
  2. Computes a reliability score from the raw output
  3. Wraps everything in a DocumentResult / VideoResult dataclass

Enhancements over raw Model 2 inference (no retraining):
 • Tuned decision threshold (0.575 instead of 0.5) — +1.67% accuracy
 • Test-time augmentation (TTA) — horizontal flip + average logits
 • Configurable frame count (default 32 instead of 20)

Reliability heuristics (training-free):
 • Model 1 — UNCERTAIN risk_level (conf < 0.60) → r = 0.4;
              else r = min(1, confidence)
 • Model 2 — usable_frames / total_frames (face coverage);
              < 0.3 → hard floor 0.3
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from .config import (
    DEVICE,
    MODEL1_ADV_CHECKPOINT,
    MODEL1_CHECKPOINT,
    MODEL1_ROOT,
    MODEL2_CHECKPOINT,
    MODEL2_NUM_FRAMES,
    MODEL2_ROOT,
    MODEL2_THRESHOLD,
    MODEL2_TTA,
)
from .schemas import DocumentResult, VideoResult

logger = logging.getLogger(__name__)


# ── Helpers to import model packages ─────────────────────────────────

def _ensure_model1_on_path():
    """Add Model 1 root to sys.path so ``import training`` resolves."""
    root = str(MODEL1_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _ensure_model2_on_path():
    """Add Model 2 root to sys.path so ``import src`` resolves to Model 2.

    Model 1 also has a ``src/`` package, so we must:
      1. Put Model 2 root FIRST on sys.path.
      2. Evict any cached ``src`` module that came from Model 1.
    """
    root = str(MODEL2_ROOT)
    # Always ensure Model 2 root is at position 0
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    # If ``src`` was already imported from Model 1, evict it so Python
    # re-imports from Model 2 root.
    cached_src = sys.modules.get("src")
    if cached_src is not None:
        src_file = getattr(cached_src, "__file__", "") or ""
        if root not in src_file.replace("/", os.sep).replace("\\", os.sep):
            keys_to_remove = [k for k in sys.modules if k == "src" or k.startswith("src.")]
            for k in keys_to_remove:
                del sys.modules[k]


# ── Document adapter (Model 1) ──────────────────────────────────────

def predict_document(
    pil_image,
    doc_type: str = "aadhaar",
    checkpoint: Optional[str] = None,
    xai_method: str = "gradcam_pp",
) -> DocumentResult:
    """Run Model 1 inference and wrap the result.

    Parameters
    ----------
    pil_image : PIL.Image.Image
    doc_type : ``"aadhaar"`` | ``"pan"``
    checkpoint : path to .pt; defaults to ``MODEL1_CHECKPOINT``
    xai_method : ``"gradcam"`` | ``"gradcam_pp"`` | ``"ig"``

    Returns
    -------
    DocumentResult  (with computed ``reliability``)
    """
    _ensure_model1_on_path()

    checkpoint = checkpoint or MODEL1_CHECKPOINT
    ckpt_name = os.path.basename(checkpoint)

    # Lazy import so the rest of the fusion package works without Model 1
    from training.inference import predict_image  # type: ignore[import]

    raw = predict_image(pil_image, doc_type, ckpt_name, xai_method)

    # ── Reliability ──────────────────────────────────────────────────
    conf = raw["confidence"]
    if raw["risk_level"] == "UNCERTAIN":
        reliability = 0.4
    else:
        reliability = min(1.0, conf)

    return DocumentResult(
        prediction=raw["prediction"],
        verdict=raw["verdict"],
        confidence=raw["confidence"],
        prob_real=raw["prob_real"],
        prob_fake=raw["prob_fake"],
        risk_level=raw["risk_level"],
        explanation=raw["explanation"],
        action=raw["action"],
        xai_b64=raw.get("xai_b64"),
        reliability=round(reliability, 4),
        raw=raw,
    )


# ── Video adapter (Model 2) with TTA + tuned threshold + more frames ─

def _predict_video_enhanced(
    video_path: str,
    checkpoint_path: str,
    device: str,
    num_frames: int,
    threshold: float,
    use_tta: bool,
):
    """Low-level Model 2 inference with TTA and custom threshold.

    Instead of calling the high-level predict_video (which hardcodes
    threshold=0.5), we call the preprocessing + predict_with_diagnostics
    directly so we can:
      1. Apply horizontal-flip TTA and average the fake probabilities
      2. Apply our tuned threshold instead of 0.5
      3. Use more frames (32 by default)
    """
    import torch
    from src.pipeline import load_model  # type: ignore[import]
    from src.video_preprocessing import preprocess_video  # type: ignore[import]
    from src.model2_reference import predict_with_diagnostics  # type: ignore[import]

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_model(checkpoint_path=checkpoint_path, device=device)
    torch_device = next(model.parameters()).device

    preprocess_res = preprocess_video(
        video_path=video_path,
        num_frames=num_frames,
        face_size=112,
    )

    if preprocess_res.num_detected < 2:
        return {
            "video_path": os.path.abspath(video_path),
            "status": "insufficient_evidence",
            "reason": "Fewer than two frames contained a detected face.",
            "prediction": None,
            "confidence": None,
            "probability": None,
            "temporal_score": None,
            "frequency_score": None,
            "preprocessing": {
                "frames_processed": num_frames,
                "usable_frames": preprocess_res.num_detected,
                "faces_detected": preprocess_res.num_detected,
                "fallbacks_used": preprocess_res.num_fallback,
            },
        }

    valid_mask = preprocess_res.valid_mask
    if valid_mask is None:
        valid_mask = torch.zeros(preprocess_res.tensor.shape[0], dtype=torch.bool)
        valid_mask[:preprocess_res.num_detected] = True
    usable_frames = preprocess_res.tensor[valid_mask]

    use_fusion = getattr(model, "inference_variant", "baseline") in {
        "spec_fusion", "temporal_only", "dct_only",
    }
    use_advanced = getattr(model, "inference_variant", "baseline") in {
        "attention_rich", "gated_rich", "attention_bands",
    }

    # ── Original pass ────────────────────────────────────────────────
    diag = predict_with_diagnostics(
        model=model, frames=usable_frames, device=torch_device,
        use_fusion_classifier=use_fusion,
        use_advanced_classifier=use_advanced,
    )
    prob_original = diag["probability"]

    # ── TTA: horizontal flip ─────────────────────────────────────────
    if use_tta:
        flipped = torch.flip(usable_frames, dims=[-1])  # flip width axis
        diag_flip = predict_with_diagnostics(
            model=model, frames=flipped, device=torch_device,
            use_fusion_classifier=use_fusion,
            use_advanced_classifier=use_advanced,
        )
        prob_flip = diag_flip["probability"]
        # Average the two fake probabilities
        prob_final = (prob_original + prob_flip) / 2.0
        logger.info(
            "TTA: original=%.4f, flipped=%.4f, averaged=%.4f",
            prob_original, prob_flip, prob_final,
        )
    else:
        prob_final = prob_original

    # ── Apply tuned threshold ────────────────────────────────────────
    prediction = "FAKE" if prob_final >= threshold else "REAL"
    confidence = prob_final if prediction == "FAKE" else 1.0 - prob_final

    return {
        "video_path": os.path.abspath(video_path),
        "status": "ok",
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "probability": round(float(prob_final), 4),
        "temporal_score": diag["temporal_score"],
        "frequency_score": diag["frequency_score"],
        "threshold_used": threshold,
        "tta_enabled": use_tta,
        "frames_requested": num_frames,
        "preprocessing": {
            "frames_processed": num_frames,
            "faces_detected": preprocess_res.num_detected,
            "usable_frames": preprocess_res.num_detected,
            "fallbacks_used": preprocess_res.num_fallback,
        },
    }


def predict_video(
    video_path: str,
    checkpoint: Optional[str] = None,
    num_frames: Optional[int] = None,
    threshold: Optional[float] = None,
    use_tta: Optional[bool] = None,
) -> VideoResult:
    """Run Model 2 inference with enhancements and wrap the result.

    Parameters
    ----------
    video_path : str
    checkpoint : path to .pt; defaults to ``MODEL2_CHECKPOINT``
    num_frames : int; defaults to ``MODEL2_NUM_FRAMES`` (32)
    threshold : float; defaults to ``MODEL2_THRESHOLD`` (0.575)
    use_tta : bool; defaults to ``MODEL2_TTA`` (True)

    Returns
    -------
    VideoResult  (with computed ``reliability``)
    """
    _ensure_model2_on_path()

    checkpoint = checkpoint or MODEL2_CHECKPOINT
    num_frames = num_frames or MODEL2_NUM_FRAMES
    threshold = threshold if threshold is not None else MODEL2_THRESHOLD
    use_tta = use_tta if use_tta is not None else MODEL2_TTA

    try:
        raw = _predict_video_enhanced(
            video_path=video_path,
            checkpoint_path=checkpoint,
            device=DEVICE,
            num_frames=num_frames,
            threshold=threshold,
            use_tta=use_tta,
        )
    except FileNotFoundError:
        raise
    except ImportError as exc:
        logger.error("Model 2 import failed (mediapipe installed?): %s", exc)
        raise
    except Exception as exc:
        logger.error("Model 2 enhanced inference failed: %s", exc, exc_info=True)
        # Fallback: try the original predict_video without enhancements
        logger.info("Falling back to original Model 2 predict_video...")
        from src.pipeline import predict_video as _predict_video  # type: ignore[import]
        raw = _predict_video(
            video_path=video_path,
            checkpoint_path=checkpoint,
            device=DEVICE,
            num_frames=num_frames,
        )

    status = raw.get("status", "ok")
    prep = raw.get("preprocessing", {})
    usable = prep.get("usable_frames", 0)
    total = prep.get("frames_processed", num_frames)

    # ── Reliability ──────────────────────────────────────────────────
    if status != "ok":
        reliability = 0.0
    else:
        face_ratio = usable / max(total, 1)
        reliability = max(0.3, min(1.0, face_ratio))

    return VideoResult(
        status=status,
        prediction=raw.get("prediction"),
        confidence=raw.get("confidence"),
        probability=raw.get("probability"),
        temporal_score=raw.get("temporal_score"),
        frequency_score=raw.get("frequency_score"),
        usable_frames=usable,
        total_frames=total,
        reliability=round(reliability, 4),
        raw=raw,
    )
