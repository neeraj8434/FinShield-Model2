"""Video-to-tensor preprocessing for deepfake detection.

Reads a video file, detects faces per frame using MediaPipe's BlazeFace
(Tasks API, VIDEO mode), crops/resizes each face to 112×112, and returns
an ImageNet-normalised PyTorch tensor ready for ``Model.forward()``.

Environment requirements:
    mediapipe==0.10.35  (Tasks API; newer 1.x crashes on Apple Silicon GPU/Metal)
    opencv-python==4.10.0.84
    torch, torchvision, numpy
"""

from __future__ import annotations

import logging
import os
from typing import List, NamedTuple, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import Tensor


class PreprocessResult(NamedTuple):
    """Return type for preprocess_video().

    tensor : Tensor
        Shape (num_frames, 3, face_size, face_size), ImageNet-normalised.
    num_detected : int
        Frames where a face was successfully detected.
    num_fallback : int
        Frames that used a fallback (prior-crop reuse or centre-crop).
    """
    tensor: Tensor
    num_detected: int
    num_fallback: int


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FACE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "blaze_face_short_range.tflite",
)

# ImageNet statistics used by the pretrained ResNeXt-50 backbone.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# How much to expand a detected bounding box before cropping.
# 1.2 → 20 % margin on each side.
#
# JUDGMENT CALL: 1.2× is a common default in face-recognition pipelines
# (e.g. InsightFace uses 1.25×, MTCNN papers use ~1.1–1.3).  A tighter
# crop (1.0–1.1) risks cutting off chin/forehead, which degrades
# downstream classification.  A wider crop (1.3+) adds more background,
# which is fine for detection but dilutes the spatial signal the ResNeXt
# sees.  1.2 is a safe middle ground — increase if your test videos have
# extreme head poses, decrease if you need tighter face-only features.
_BBOX_MARGIN = 1.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_bbox(
    x: int, y: int, w: int, h: int,
    img_w: int, img_h: int,
    margin: float = _BBOX_MARGIN,
) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) after expanding *w*/*h* by *margin*, clamped."""
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w, new_h = w * margin, h * margin
    x1 = max(0, int(cx - new_w / 2.0))
    y1 = max(0, int(cy - new_h / 2.0))
    x2 = min(img_w, int(cx + new_w / 2.0))
    y2 = min(img_h, int(cy + new_h / 2.0))
    return x1, y1, x2, y2


def _center_crop(frame: np.ndarray, face_size: int) -> np.ndarray:
    """Fallback: take a square centre crop when no face is detected.

    JUDGMENT CALL (fallback behaviour):
    When no face has *ever* been detected in this video yet (i.e. the very
    first frame has no detection), there is no prior crop to reuse.  Rather
    than crashing or returning a black frame, we take a centre-crop of the
    full frame.  This is the safest option because:
      • Most talking-head/selfie deepfake videos have the face near centre.
      • A black or random frame would inject garbage into the LSTM sequence.
      • Skipping the frame entirely would change the sequence length, which
        violates the model's expected input shape.
    For later frames where a previous detection exists, we reuse that
    previous crop instead of centre-cropping — this gives temporal
    consistency and is almost certainly a better crop than a generic centre
    cut.  Both fallback cases are logged as warnings so you can audit how
    often they fire.
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    crop = frame[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(crop, (face_size, face_size))


def _pick_largest_detection(
    detections: List,
) -> Optional[object]:
    """From a list of MediaPipe detections, return the one with the largest
    bounding-box area (width × height).  Returns ``None`` if the list is
    empty."""
    if not detections:
        return None
    return max(
        detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
    )


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def preprocess_video(
    video_path: str,
    num_frames: int = 20,
    face_size: int = 112,
    face_model_path: str = _FACE_MODEL_PATH,
    min_detection_confidence: float = 0.3,
) -> PreprocessResult:
    """Read a video, detect & crop faces, return an ImageNet-normalised tensor.

    Parameters
    ----------
    video_path : str
        Path to a video file readable by OpenCV (mp4, avi, …).
    num_frames : int
        Number of evenly-spaced frames to sample from the video.
    face_size : int
        Height and width of each output face crop (pixels).
    face_model_path : str
        Path to the BlazeFace TFLite model file.
    min_detection_confidence : float
        Minimum confidence threshold for face detection.  The BlazeFace
        short-range model is tuned for selfie-distance faces; for
        medium/wide shots where faces are smaller, lower this to 0.1–0.2.
        Default 0.3 balances recall vs. false positives.

    Returns
    -------
    tensor : Tensor
        Shape ``(num_frames, 3, face_size, face_size)``, float32,
        ImageNet-normalised.  Suitable for ``Model.forward()`` and the
        ``_as_batch()`` helper in ``model2_reference.py``.
    face_detected_count : int
        Number of frames where a face was successfully detected.
    fallback_count : int
        Number of frames that used a fallback (prior crop reuse or
        centre-crop).

    Raises
    ------
    ValueError
        If the video cannot be opened or contains zero frames.
    FileNotFoundError
        If *video_path* or *face_model_path* does not exist.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(face_model_path):
        raise FileNotFoundError(f"Face model not found: {face_model_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0:
        cap.release()
        raise ValueError(
            f"Video has zero or unknown frame count: {video_path} "
            f"(CAP_PROP_FRAME_COUNT={total_frames})"
        )
    if fps <= 0:
        # Some containers report 0 fps; default to 30 so timestamp math
        # still produces increasing values.
        logger.warning("Video reports fps=%.2f, defaulting to 30.0", fps)
        fps = 30.0

    # ------------------------------------------------------------------
    # Sample frame indices
    # ------------------------------------------------------------------
    sampled_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    # ------------------------------------------------------------------
    # Face detection setup  (single instance, VIDEO running mode)
    # ------------------------------------------------------------------
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=face_model_path,
            delegate=BaseOptions.Delegate.CPU,  # GPU/Metal delegate crashes on this Mac
        ),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=min_detection_confidence,
    )

    crops: List[np.ndarray] = []
    last_good_crop: Optional[np.ndarray] = None
    face_detected_count = 0
    fallback_count = 0

    try:
        detector = FaceDetector.create_from_options(options)
        try:
            for seq_idx, frame_idx in enumerate(sampled_indices):
                # --- Read the frame ---
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, bgr_frame = cap.read()
                if not ret or bgr_frame is None:
                    logger.warning(
                        "Frame %d (seq %d/%d): read failed, using fallback.",
                        frame_idx, seq_idx + 1, num_frames,
                    )
                    if last_good_crop is not None:
                        crops.append(last_good_crop.copy())
                    else:
                        # Can't even read the frame; push a black placeholder
                        crops.append(np.zeros((face_size, face_size, 3), dtype=np.uint8))
                    fallback_count += 1
                    continue

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                img_h, img_w = rgb_frame.shape[:2]

                # --- Detect face ---
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame
                )

                # VIDEO mode requires strictly increasing timestamps.
                # Use frame_index / fps * 1000 to get milliseconds, then add
                # seq_idx as a tiebreaker so duplicate frame indices (possible
                # when num_frames > total_frames) still produce increasing ts.
                frame_timestamp_ms = int(frame_idx / fps * 1000) + seq_idx

                result = detector.detect_for_video(mp_image, frame_timestamp_ms)
                detection = _pick_largest_detection(result.detections)

                if detection is not None:
                    # --- Successful detection: crop with margin ---
                    bb = detection.bounding_box
                    x1, y1, x2, y2 = _expand_bbox(
                        bb.origin_x, bb.origin_y, bb.width, bb.height,
                        img_w, img_h,
                    )
                    crop = rgb_frame[y1:y2, x1:x2]

                    # Guard against degenerate boxes (0-width/height after clamp)
                    if crop.size == 0:
                        logger.warning(
                            "Frame %d (seq %d/%d): degenerate bbox after "
                            "clamping, using fallback.",
                            frame_idx, seq_idx + 1, num_frames,
                        )
                        if last_good_crop is not None:
                            crops.append(last_good_crop.copy())
                        else:
                            crops.append(_center_crop(rgb_frame, face_size))
                        fallback_count += 1
                        continue

                    crop = cv2.resize(crop, (face_size, face_size))
                    crops.append(crop)
                    last_good_crop = crop
                    face_detected_count += 1

                else:
                    # --- No face detected: fallback ---
                    # FALLBACK BEHAVIOUR (documented per spec):
                    # • If we have a previous good crop from an earlier frame
                    #   in this same video, reuse it.  This preserves temporal
                    #   consistency and is almost always a better crop than a
                    #   generic centre-cut.
                    # • If this is the very first frame (no prior detection),
                    #   take a centre-crop of the full frame.  This avoids
                    #   injecting a black/garbage frame into the LSTM sequence.
                    # Both cases are logged as warnings.
                    fallback_count += 1
                    if last_good_crop is not None:
                        logger.warning(
                            "Frame %d (seq %d/%d): no face detected, "
                            "reusing last good crop.",
                            frame_idx, seq_idx + 1, num_frames,
                        )
                        crops.append(last_good_crop.copy())
                    else:
                        logger.warning(
                            "Frame %d (seq %d/%d): no face detected and no "
                            "prior crop exists, using centre-crop fallback.",
                            frame_idx, seq_idx + 1, num_frames,
                        )
                        crops.append(_center_crop(rgb_frame, face_size))

        finally:
            detector.close()
    finally:
        cap.release()

    # ------------------------------------------------------------------
    # Stack → Tensor → Normalise
    # ------------------------------------------------------------------
    # crops: list of num_frames arrays, each (face_size, face_size, 3) uint8 RGB
    stacked = np.stack(crops, axis=0)  # (num_frames, H, W, 3)

    tensor = torch.from_numpy(stacked).float()  # still HWC, [0, 255]
    tensor = tensor.permute(0, 3, 1, 2)  # → (num_frames, 3, H, W)
    tensor /= 255.0  # → [0, 1]

    # Per-channel ImageNet normalisation: (x - mean) / std
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return PreprocessResult(tensor, face_detected_count, fallback_count)
