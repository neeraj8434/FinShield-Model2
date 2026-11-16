"""End-to-End Deepfake Detection Pipeline for FinShield AI Model 2.

Integrates video frame extraction, face detection (BlazeFace), spatial-temporal
feature extraction (ResNeXt-50 + LSTM), frequency anomaly analysis (DCT), and
confidence/diagnostic scoring into a unified API.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from .model2_reference import Model, predict_with_diagnostics
from .video_preprocessing import PreprocessResult, preprocess_video

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "model_87_acc_20_frames_final_data.pt",
)


def get_default_device() -> torch.device:
    """Return the best available hardware accelerator or CPU.

    Checks for CUDA (NVIDIA GPU), Apple Silicon MPS, or falls back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_classes: int = 2,
    use_frequency_feature: bool = True,
) -> Model:
    """Instantiate Model and optionally load pretrained checkpoint weights.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to the ``.pt`` checkpoint file. If None, the model is initialized
        with default torchvision ResNeXt weights and random classifier weights.
    device : str or torch.device, optional
        Target device for model placement. If None, auto-detected via
        ``get_default_device()``.
    num_classes : int, default 2
        Number of output classes (0: REAL, 1: FAKE).
    use_frequency_feature : bool, default True
        Whether to enable parallel 2D DCT frequency feature extraction.

    Returns
    -------
    Model
        The instantiated PyTorch model in evaluation mode on the target device.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` is explicitly specified but does not exist.
    """
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    logger.info("Initializing Model 2 architecture on device: %s", device)
    model = Model(num_classes=num_classes, use_frequency_feature=use_frequency_feature)
    model.inference_variant = "untrained"

    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found at: {checkpoint_path}. "
                f"Ensure the pretrained model is downloaded into checkpoints/ or specify a valid path."
            )

        logger.info("Loading checkpoint from: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            model.inference_variant = checkpoint.get("variant", "baseline")
            model.inference_temperature = float(checkpoint.get("temperature", 1.0))
            if model.inference_variant in {"attention_rich", "gated_rich", "attention_bands"}:
                model.enable_advanced_head(model.inference_variant)
        else:
            state_dict = checkpoint
            model.inference_variant = "baseline"
            model.inference_temperature = 1.0

        # strict=False allows loading checkpoint into model containing new fusion layers
        load_result = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Checkpoint loaded successfully (missing keys: %d, unexpected keys: %d)",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    model.to(device)
    model.eval()
    return model


def predict_video(
    video_path: str,
    model: Optional[Model] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_frames: int = 20,
    face_size: int = 112,
    min_detection_confidence: float = 0.3,
    use_temporal_adjustment: bool = False,
    use_frequency_adjustment: bool = False,
    use_fusion_classifier: Optional[bool] = None,
) -> Dict[str, Any]:
    """End-to-end inference on a single video file.

    Performs video decoding, face detection & cropping via BlazeFace,
    spatial-temporal modeling, frequency analysis, and diagnostic scoring.

    Parameters
    ----------
    video_path : str
        Path to the input video file (e.g. .mp4, .avi).
    model : Model, optional
        Pre-instantiated model. If None, initialized via ``load_model()``.
    checkpoint_path : str, optional
        Path to checkpoint file used if ``model`` is None. If None and the
        default checkpoint file exists in ``checkpoints/``, it is automatically
        loaded.
    device : str or torch.device, optional
        Target device. Auto-detected if None.
    num_frames : int, default 20
        Number of frames to sample and analyze from the video.
    face_size : int, default 112
        Spatial dimensions (face_size × face_size) for face crops.
    min_detection_confidence : float, default 0.3
        Face detection threshold for BlazeFace.
    use_temporal_adjustment : bool, default True
        Whether to adjust final prediction probability using temporal consistency.
    use_frequency_adjustment : bool, default True
        Whether to adjust final prediction probability using DCT frequency score.

    Returns
    -------
    dict
        Prediction result containing:
        - ``video_path``: Absolute path to the analyzed video.
        - ``prediction``: Final predicted label ("REAL" or "FAKE").
        - ``confidence``: Model confidence in the predicted class [0.5, 1.0].
        - ``probability``: Final probability of being FAKE [0.0, 1.0].
        - ``temporal_score``: Frame-to-frame inconsistency score [0.0, 1.0].
        - ``frequency_score``: High-frequency energy anomaly score [0.0, 1.0].
        - ``preprocessing``: Face detection diagnostics (frames sampled, faces detected, fallbacks).
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")

    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Automatically load model if not passed
    if model is None:
        ckpt_to_use = checkpoint_path
        if ckpt_to_use is None and os.path.isfile(DEFAULT_CHECKPOINT_PATH):
            ckpt_to_use = DEFAULT_CHECKPOINT_PATH
        model = load_model(checkpoint_path=ckpt_to_use, device=device)

    logger.info("Preprocessing video: %s", video_path)
    preprocess_res: PreprocessResult = preprocess_video(
        video_path=video_path,
        num_frames=num_frames,
        face_size=face_size,
        min_detection_confidence=min_detection_confidence,
    )

    logger.info(
        "Running model inference (faces detected: %d/%d, fallbacks: %d/%d)...",
        preprocess_res.num_detected,
        num_frames,
        preprocess_res.num_fallback,
        num_frames,
    )
    if preprocess_res.num_detected < 2:
        return {
            "video_path": os.path.abspath(video_path),
            "status": "insufficient_evidence",
            "reason": "Fewer than two sampled frames contained a detected face.",
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
    if use_fusion_classifier is None:
        use_fusion_classifier = getattr(model, "inference_variant", "baseline") in {"spec_fusion", "temporal_only", "dct_only"}
    use_advanced_classifier = getattr(model, "inference_variant", "baseline") in {"attention_rich", "gated_rich", "attention_bands"}
    diagnostics = predict_with_diagnostics(
        model=model,
        frames=usable_frames,
        device=device,
        use_temporal_adjustment=use_temporal_adjustment,
        use_frequency_adjustment=use_frequency_adjustment,
        use_fusion_classifier=use_fusion_classifier,
        use_advanced_classifier=use_advanced_classifier,
    )

    return {
        "video_path": os.path.abspath(video_path),
        "status": "ok",
        "prediction": diagnostics["prediction"],
        "confidence": diagnostics["confidence"],
        "probability": diagnostics["probability"],
        "temporal_score": diagnostics["temporal_score"],
        "frequency_score": diagnostics["frequency_score"],
        "preprocessing": {
            "frames_processed": num_frames,
            "faces_detected": preprocess_res.num_detected,
            "usable_frames": preprocess_res.num_detected,
            "fallbacks_used": preprocess_res.num_fallback,
        },
    }


def predict_batch(
    video_paths: List[str],
    model: Optional[Model] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_frames: int = 20,
    face_size: int = 112,
    min_detection_confidence: float = 0.3,
    use_temporal_adjustment: bool = False,
    use_frequency_adjustment: bool = False,
) -> List[Dict[str, Any]]:
    """Run end-to-end prediction across multiple video files.

    Reuses the model instance across videos to optimize throughput.

    Parameters
    ----------
    video_paths : list of str
        List of paths to video files.
    Other parameters : same as ``predict_video``.

    Returns
    -------
    list of dict
        List of prediction results for each video.
    """
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    if model is None:
        ckpt_to_use = checkpoint_path
        if ckpt_to_use is None and os.path.isfile(DEFAULT_CHECKPOINT_PATH):
            ckpt_to_use = DEFAULT_CHECKPOINT_PATH
        model = load_model(checkpoint_path=ckpt_to_use, device=device)

    results = []
    for path in video_paths:
        try:
            res = predict_video(
                video_path=path,
                model=model,
                device=device,
                num_frames=num_frames,
                face_size=face_size,
                min_detection_confidence=min_detection_confidence,
                use_temporal_adjustment=use_temporal_adjustment,
                use_frequency_adjustment=use_frequency_adjustment,
            )
            results.append(res)
        except Exception as e:
            logger.error("Error processing %s: %s", path, e)
            results.append({
                "video_path": os.path.abspath(path),
                "error": str(e),
                "status": "error",
                "prediction": None,
                "confidence": None,
            })

    return results
