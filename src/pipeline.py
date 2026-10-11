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

from .model2_reference import (
    Model,
    _as_batch,
    _fake_probability_from_logits,
    adjusted_fake_probability,
    normalize_frequency_score,
    normalize_temporal_score,
    predict_with_diagnostics,
    temporal_cosine_score,
    temporal_mad_score,
)
from .video_preprocessing import PreprocessResult, preprocess_video

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "model_87_acc_20_frames_final_data.pt",
)

_EXPECTED_FUSION_KEYS = {
    "fusion_fc1.weight",
    "fusion_fc1.bias",
    "fusion_fc2.weight",
    "fusion_fc2.bias",
}


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
    fusion_checkpoint_path: Optional[str] = None,
) -> Model:
    """Instantiate Model and optionally load pretrained checkpoint weights.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to the ``.pt`` checkpoint file. If None, the model is initialized
        with random weights.
    device : str or torch.device, optional
        Target device for model placement. If None, auto-detected via
        ``get_default_device()``.
    num_classes : int, default 2
        Number of output classes (0: REAL, 1: FAKE).
    use_frequency_feature : bool, default True
        Whether to enable parallel 2D DCT frequency feature extraction.
    fusion_checkpoint_path : str, optional
        Path to a checkpoint containing trained fusion_fc1/fusion_fc2 weights.
        When provided, these weights are loaded on top of the base checkpoint.

    Returns
    -------
    Model
        The instantiated PyTorch model in evaluation mode on the target device.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` or ``fusion_checkpoint_path`` is specified but
        does not exist.
    RuntimeError
        If the checkpoint contains unexpected keys or is missing keys that
        are not the expected fusion layer keys.
    """
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    logger.info("Initializing Model 2 architecture on device: %s", device)
    model = Model(num_classes=num_classes, use_frequency_feature=use_frequency_feature)

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
        else:
            state_dict = checkpoint

        # strict=False allows loading checkpoint into model containing new fusion layers.
        load_result = model.load_state_dict(state_dict, strict=False)

        # Safety assertions: unexpected_keys must be empty (no stale/renamed
        # keys in the checkpoint) and missing_keys must be exactly the fusion
        # layer keys (which are new and not in the old checkpoint).
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint contains unexpected keys (possible layer rename or "
                f"architecture mismatch): {load_result.unexpected_keys}"
            )
        non_fusion_missing = [
            k for k in load_result.missing_keys
            if not k.startswith("fusion_")
        ]
        if non_fusion_missing:
            raise RuntimeError(
                f"Checkpoint is missing non-fusion keys (architecture mismatch): "
                f"{non_fusion_missing}"
            )

        logger.info(
            "Checkpoint loaded successfully (missing fusion keys: %d, unexpected keys: %d)",
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    # Optionally load trained fusion head weights.
    if fusion_checkpoint_path is not None:
        if not os.path.isfile(fusion_checkpoint_path):
            raise FileNotFoundError(
                f"Fusion checkpoint not found at: {fusion_checkpoint_path}"
            )
        logger.info("Loading fusion checkpoint from: %s", fusion_checkpoint_path)
        fusion_state = torch.load(fusion_checkpoint_path, map_location=device)
        if isinstance(fusion_state, dict) and "state_dict" in fusion_state:
            fusion_state = fusion_state["state_dict"]
        # Only load fusion_fc1/fusion_fc2 keys.
        fusion_keys = {k: v for k, v in fusion_state.items() if k.startswith("fusion_")}
        if not fusion_keys:
            raise RuntimeError(
                f"Fusion checkpoint contains no fusion_* keys. "
                f"Available keys: {list(fusion_state.keys())[:10]}"
            )
        model.load_state_dict(fusion_keys, strict=False)
        logger.info("Loaded %d fusion layer weights.", len(fusion_keys))

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict_fused(
    model: Model,
    frames: Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Run prediction using the trained fusion head (classify_fused_features)."""
    model.eval()
    frames = _as_batch(frames).to(device)

    _, sequence_features = model.extract_sequence_features(frames)

    ts_mad = temporal_mad_score(sequence_features)
    ts_cosine = temporal_cosine_score(sequence_features)

    raw_freq = model.freq_feature(frames)
    freq_normalized = normalize_frequency_score(raw_freq)

    logits = model.classify_fused_features(
        sequence_features,
        normalize_temporal_score(ts_mad),
        freq_normalized,
    )
    fake_probability = _fake_probability_from_logits(logits)

    fake_prob_val = float(fake_probability[0].detach().cpu().item())
    from .model2_reference import LABELS
    prediction = LABELS[int(fake_prob_val >= 0.5)]
    confidence = fake_prob_val if prediction == "FAKE" else 1.0 - fake_prob_val

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probability": round(fake_prob_val, 4),
        "temporal_score": round(float(ts_mad[0].detach().cpu().item()), 4),
        "temporal_cosine_score": round(float(ts_cosine[0].detach().cpu().item()), 4),
        "frequency_score": round(float(freq_normalized[0].detach().cpu().item()), 4),
        "mode": "fusion",
    }


def predict_video(
    video_path: str,
    model: Optional[Model] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_frames: int = 20,
    face_size: int = 112,
    min_detection_confidence: float = 0.3,
    use_temporal_adjustment: bool = True,
    use_frequency_adjustment: bool = True,
    fusion_checkpoint_path: Optional[str] = None,
    face_model_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
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
    fusion_checkpoint_path : str, optional
        Path to trained fusion head checkpoint. When provided, the prediction
        uses the learned fusion classifier instead of post-hoc blending.
    face_model_path : str, optional
        Path to a BlazeFace TFLite model file override (e.g. full-range model).
    debug_dir : str, optional
        If set, save debug overlays of face detections and crops to this directory.

    Returns
    -------
    dict
        Prediction result containing:
        - ``video_path``: Absolute path to the analyzed video.
        - ``prediction``: Final predicted label ("REAL" or "FAKE").
        - ``confidence``: Model confidence in the predicted class [0.5, 1.0].
        - ``probability``: Final probability of being FAKE [0.0, 1.0].
        - ``temporal_score``: Frame-to-frame inconsistency score (MAD).
        - ``temporal_cosine_score``: Frame-to-frame cosine distance score [0.0, 1.0].
        - ``frequency_score``: High-frequency energy anomaly score [0.0, 1.0].
        - ``mode``: "fusion" or "baseline+blend".
        - ``preprocessing``: Face detection diagnostics.
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
        model = load_model(
            checkpoint_path=ckpt_to_use,
            device=device,
            fusion_checkpoint_path=fusion_checkpoint_path,
        )

    # Build preprocessing kwargs
    preprocess_kwargs: Dict[str, Any] = dict(
        video_path=video_path,
        num_frames=num_frames,
        face_size=face_size,
        min_detection_confidence=min_detection_confidence,
    )
    if face_model_path is not None:
        preprocess_kwargs["face_model_path"] = face_model_path
    if debug_dir is not None:
        preprocess_kwargs["debug_dir"] = debug_dir

    logger.info("Preprocessing video: %s", video_path)
    preprocess_res: PreprocessResult = preprocess_video(**preprocess_kwargs)

    logger.info(
        "Running model inference (faces detected: %d/%d, fallbacks: %d/%d)...",
        preprocess_res.num_detected,
        num_frames,
        preprocess_res.num_fallback,
        num_frames,
    )

    # Choose fusion vs baseline+blend prediction path.
    if fusion_checkpoint_path is not None:
        diagnostics = _predict_fused(model, preprocess_res.tensor, device)
    else:
        diagnostics = predict_with_diagnostics(
            model=model,
            frames=preprocess_res.tensor,
            device=device,
            use_temporal_adjustment=use_temporal_adjustment,
            use_frequency_adjustment=use_frequency_adjustment,
        )
        diagnostics["mode"] = "baseline+blend"

    detected = preprocess_res.num_detected
    coverage = detected / num_frames if num_frames > 0 else 0.0
    reliable = coverage >= 0.70
    warning = (
        None if reliable
        else f"Low face-detection coverage ({detected}/{num_frames}); result unreliable"
    )

    return {
        "video_path": os.path.abspath(video_path),
        "prediction": diagnostics["prediction"],
        "confidence": diagnostics["confidence"],
        "probability": diagnostics["probability"],
        "temporal_score": diagnostics["temporal_score"],
        "temporal_cosine_score": diagnostics.get("temporal_cosine_score"),
        "frequency_score": diagnostics["frequency_score"],
        "mode": diagnostics["mode"],
        "reliable": reliable,
        "warning": warning,
        "preprocessing": {
            "frames_processed": num_frames,
            "faces_detected": detected,
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
    use_temporal_adjustment: bool = True,
    use_frequency_adjustment: bool = True,
    fusion_checkpoint_path: Optional[str] = None,
    face_model_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
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
        model = load_model(
            checkpoint_path=ckpt_to_use,
            device=device,
            fusion_checkpoint_path=fusion_checkpoint_path,
        )

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
                fusion_checkpoint_path=fusion_checkpoint_path,
                face_model_path=face_model_path,
                debug_dir=debug_dir,
            )
            results.append(res)
        except Exception as e:
            logger.error("Error processing %s: %s", path, e)
            results.append({
                "video_path": os.path.abspath(path),
                "error": str(e),
                "prediction": None,
                "confidence": None,
            })

    return results
