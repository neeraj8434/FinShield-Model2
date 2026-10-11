"""Video-to-tensor preprocessing for deepfake detection.

Reads a video file, detects faces per frame using MediaPipe's BlazeFace
(Tasks API, VIDEO mode), crops/resizes each face to 112×112, and returns
an ImageNet-normalised PyTorch tensor ready for Model 2.

Features:
- Candidate pool sampling (e.g. 60 candidates) to find real face detections
  before taking 20 evenly spaced sequence frames.
- Sanity bounds on detected boxes (rejecting area > 60% or < 1%, aspect outside 0.6–1.6).
- Informative fallback logging ("no detection" vs "rejected: area=...% aspect=...").
- Limited crop reuse (at most 2 duplicates in a row before centre-crop fallback).
- Debug overlay visualization showing accepted (green) and rejected (red) boxes with reasons.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FACE_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "blaze_face_short_range.tflite")

# ImageNet normalization parameters (channel-wise: R, G, B)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Expand detected bounding box by 20% to capture full facial context
_BBOX_MARGIN = 1.2

# ---------------------------------------------------------------------------
# Detection sanity thresholds
# ---------------------------------------------------------------------------
_MAX_BOX_AREA_RATIO = 0.60  # box area > 60 % of frame → reject
_MIN_BOX_AREA_RATIO = 0.01  # box area < 1 % of frame → reject
_MIN_ASPECT_RATIO = 0.6     # width/height
_MAX_ASPECT_RATIO = 1.6     # width/height


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


def _check_box_sanity(
    w: int, h: int, img_w: int, img_h: int,
) -> Tuple[bool, str]:
    """Check if bounding box passes sanity thresholds and return diagnostic string.

    Returns
    -------
    is_sane : bool
        True if the box passes all area and aspect ratio checks.
    reason : str
        Formatted description: ``area=<pct>% aspect=<ratio>``.
    """
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return False, "invalid dimensions"

    frame_area = img_w * img_h
    box_area = w * h
    area_ratio = box_area / frame_area
    area_pct = area_ratio * 100.0
    aspect = w / h

    reasons: List[str] = []
    if area_ratio > _MAX_BOX_AREA_RATIO:
        reasons.append(f"area={area_pct:.1f}%>60%")
    elif area_ratio < _MIN_BOX_AREA_RATIO:
        reasons.append(f"area={area_pct:.1f}%<1%")
    if aspect < _MIN_ASPECT_RATIO or aspect > _MAX_ASPECT_RATIO:
        reasons.append(f"aspect={aspect:.2f} outside [0.6, 1.6]")

    desc = f"area={area_pct:.1f}% aspect={aspect:.2f}"
    if reasons:
        return False, desc
    return True, desc


def _is_sane_box(
    w: int, h: int, img_w: int, img_h: int,
) -> bool:
    """Return True if bounding box passes sanity checks (area ratio & aspect ratio)."""
    is_sane, _ = _check_box_sanity(w, h, img_w, img_h)
    return is_sane


def _center_crop(frame: np.ndarray, face_size: int) -> np.ndarray:
    """Fallback: take a square centre crop from the frame."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    crop = frame[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(crop, (face_size, face_size))


def _pick_largest_sane_detection(
    detections: List,
    img_w: int,
    img_h: int,
) -> Optional[object]:
    """From a list of detections, return the largest box passing sanity checks."""
    best = None
    best_area = -1
    for d in detections:
        bb = d.bounding_box
        if not _is_sane_box(bb.width, bb.height, img_w, img_h):
            continue
        area = bb.width * bb.height
        if area > best_area:
            best = d
            best_area = area
    return best


def _save_debug_frame(
    debug_dir: str,
    seq_idx: int,
    rgb_frame: np.ndarray,
    detections: List,
    accepted_bb: Optional[object],
    final_crop: Optional[np.ndarray],
    img_w: int,
    img_h: int,
    reason: str = "",
    fallback_note: Optional[str] = None,
) -> None:
    """Save a debug overlay image showing detections and the final 112×112 crop."""
    os.makedirs(debug_dir, exist_ok=True)
    vis = cv2.cvtColor(rgb_frame.copy(), cv2.COLOR_RGB2BGR)

    # Draw all detected boxes: green if accepted/sane, red if rejected
    if detections:
        for i, d in enumerate(detections):
            bb = d.bounding_box
            is_sane, r_str = _check_box_sanity(bb.width, bb.height, img_w, img_h)
            colour = (0, 255, 0) if is_sane else (0, 0, 255)
            cv2.rectangle(
                vis,
                (bb.origin_x, bb.origin_y),
                (bb.origin_x + bb.width, bb.origin_y + bb.height),
                colour, 2,
            )
            prefix = "OK" if is_sane else "REJ"
            label = f"{prefix} #{i} {r_str}"
            cv2.putText(
                vis, label,
                (bb.origin_x, max(bb.origin_y - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
            )

    # Top-left banner: status, fallback action, and reasons
    banner = f"Seq {seq_idx:02d}: "
    if accepted_bb is not None:
        banner += "ACCEPTED"
        banner_color = (0, 255, 0)
    elif fallback_note:
        banner += f"FALLBACK ({fallback_note}) — {reason}"
        banner_color = (0, 165, 255) if "reuse" in fallback_note else (0, 0, 255)
    else:
        banner += reason
        banner_color = (0, 0, 255)

    cv2.putText(vis, banner, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, banner, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 1, cv2.LINE_AA)

    # Top-right corner: final crop preview with a border
    if final_crop is not None:
        crop_bgr = cv2.cvtColor(final_crop, cv2.COLOR_RGB2BGR) if final_crop.shape[-1] == 3 else final_crop
        crop_h, crop_w = crop_bgr.shape[:2]
        if vis.shape[0] >= crop_h + 10 and vis.shape[1] >= crop_w + 10:
            x_start = vis.shape[1] - crop_w - 10
            y_start = 10
            cv2.rectangle(
                vis,
                (x_start - 2, y_start - 2),
                (x_start + crop_w + 1, y_start + crop_h + 1),
                (255, 255, 255), 2,
            )
            vis[y_start : y_start + crop_h, x_start : x_start + crop_w] = crop_bgr
            cv2.putText(
                vis, "Final Crop",
                (x_start, y_start + crop_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

    path = os.path.join(debug_dir, f"debug_frame_{seq_idx:03d}.jpg")
    cv2.imwrite(path, vis)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def preprocess_video(
    video_path: str,
    num_frames: int = 20,
    face_size: int = 112,
    face_model_path: str = _FACE_MODEL_PATH,
    min_detection_confidence: float = 0.3,
    debug_dir: Optional[str] = None,
    candidate_count: int = 60,
) -> PreprocessResult:
    """Read a video, detect & crop faces, return an ImageNet-normalised tensor.

    Sampling Strategy:
    1. Samples a candidate pool (default: 60 evenly spaced frames).
    2. Runs face detection with sanity checks on all candidate frames.
    3. If >= ``num_frames`` (20) accepted faces are found, selects 20 evenly
       spaced frames exclusively from those accepted candidates (0 fallbacks).
    4. If < ``num_frames`` accepted faces are found, incorporates the available
       accepted frames and falls back for missing frames.
    5. Fallback rule: reuses the prior face crop at most twice consecutively.
       When a 3rd duplicate would occur, falls back to a centre-crop of that
       frame and breaks the duplicate streak.

    Parameters
    ----------
    video_path : str
        Path to a video file readable by OpenCV.
    num_frames : int, default 20
        Number of output frames in the temporal sequence.
    face_size : int, default 112
        Resolution (face_size × face_size) for output face crops.
    face_model_path : str
        Path to BlazeFace TFLite model file.
    min_detection_confidence : float, default 0.3
        Face detection threshold for BlazeFace.
    debug_dir : str, optional
        Directory to save overlay debug images showing detected/rejected boxes.
    candidate_count : int, default 60
        Size of candidate pool sampled across the video before selecting the
        final 20 frames.

    Returns
    -------
    PreprocessResult
        NamedTuple of (tensor, num_detected, num_fallback).
    """
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
        logger.warning("Video reports fps=%.2f, defaulting to 30.0", fps)
        fps = 30.0

    # ------------------------------------------------------------------
    # 1. Candidate frame sampling
    # ------------------------------------------------------------------
    n_candidates = min(total_frames, max(candidate_count, num_frames))
    candidate_indices = np.linspace(0, total_frames - 1, n_candidates).astype(int)

    # ------------------------------------------------------------------
    # 2. BlazeFace face detector setup
    # ------------------------------------------------------------------
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=face_model_path,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=min_detection_confidence,
    )

    all_candidates: List[dict] = []
    accepted_candidates: List[dict] = []

    try:
        detector = FaceDetector.create_from_options(options)
        try:
            for cand_idx, frame_idx in enumerate(candidate_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, bgr_frame = cap.read()
                if not ret or bgr_frame is None:
                    all_candidates.append({
                        "cand_idx": cand_idx,
                        "frame_idx": frame_idx,
                        "has_face": False,
                        "crop": None,
                        "accepted_bb": None,
                        "detections": [],
                        "rgb_frame": np.zeros((face_size, face_size, 3), dtype=np.uint8),
                        "reason": "frame read failed",
                    })
                    continue

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                img_h, img_w = rgb_frame.shape[:2]

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_timestamp_ms = int(frame_idx / fps * 1000) + cand_idx
                result = detector.detect_for_video(mp_image, frame_timestamp_ms)

                detections = result.detections if result else []
                accepted_d = None
                rejection_reason = "no detection"

                if detections:
                    accepted_d = _pick_largest_sane_detection(detections, img_w, img_h)
                    if accepted_d is None:
                        reasons_list = []
                        for i, d in enumerate(detections):
                            bb = d.bounding_box
                            _, r_str = _check_box_sanity(bb.width, bb.height, img_w, img_h)
                            reasons_list.append(f"box {i}: {r_str}")
                        rejection_reason = f"rejected: {', '.join(reasons_list)}"
                    else:
                        rejection_reason = "accepted"

                if accepted_d is not None:
                    bb = accepted_d.bounding_box
                    x1, y1, x2, y2 = _expand_bbox(bb.origin_x, bb.origin_y, bb.width, bb.height, img_w, img_h)
                    raw_crop = rgb_frame[y1:y2, x1:x2]
                    if raw_crop.size > 0:
                        crop_resized = cv2.resize(raw_crop, (face_size, face_size))
                        cand_entry = {
                            "cand_idx": cand_idx,
                            "frame_idx": frame_idx,
                            "has_face": True,
                            "crop": crop_resized,
                            "accepted_bb": bb,
                            "detections": detections,
                            "rgb_frame": rgb_frame,
                            "reason": "accepted",
                        }
                        accepted_candidates.append(cand_entry)
                        all_candidates.append(cand_entry)
                        continue

                    rejection_reason = "rejected: degenerate bbox after clamp"

                all_candidates.append({
                    "cand_idx": cand_idx,
                    "frame_idx": frame_idx,
                    "has_face": False,
                    "crop": None,
                    "accepted_bb": None,
                    "detections": detections,
                    "rgb_frame": rgb_frame,
                    "reason": rejection_reason,
                })

        finally:
            detector.close()
    finally:
        cap.release()

    # ------------------------------------------------------------------
    # 3. Select final sequence frames
    # ------------------------------------------------------------------
    final_crops: List[np.ndarray] = []
    face_detected_count = 0
    fallback_count = 0

    if len(accepted_candidates) >= num_frames:
        # Sufficient accepted faces found: pick num_frames evenly spaced from accepted pool
        pick_indices = np.linspace(0, len(accepted_candidates) - 1, num_frames).astype(int)
        for seq_idx, p_idx in enumerate(pick_indices):
            rec = accepted_candidates[p_idx]
            final_crops.append(rec["crop"])
            face_detected_count += 1
            if debug_dir:
                img_h, img_w = rec["rgb_frame"].shape[:2]
                _save_debug_frame(
                    debug_dir=debug_dir,
                    seq_idx=seq_idx,
                    rgb_frame=rec["rgb_frame"],
                    detections=rec["detections"],
                    accepted_bb=rec["accepted_bb"],
                    final_crop=rec["crop"],
                    img_w=img_w,
                    img_h=img_h,
                    reason="accepted",
                )
    else:
        # Fewer than num_frames accepted faces found: incorporate available and use fallbacks
        n_all = len(all_candidates)
        if not accepted_candidates:
            # Zero detections across the pool: sample evenly spaced frames
            chosen_indices = list(np.linspace(0, n_all - 1, num_frames).astype(int))
        else:
            # Include all accepted candidate indices, fill remaining slots by maximizing distance
            chosen_set = set(c["cand_idx"] for c in accepted_candidates)
            avail_pool = [i for i in range(n_all) if i not in chosen_set]
            chosen_list = list(chosen_set)

            while len(chosen_list) < num_frames and avail_pool:
                best_i = max(avail_pool, key=lambda idx: min(abs(idx - s) for s in chosen_list))
                chosen_list.append(best_i)
                avail_pool.remove(best_i)

            chosen_list.sort()
            chosen_indices = chosen_list[:num_frames]

        selected_records = [all_candidates[i] for i in chosen_indices]

        last_good_crop: Optional[np.ndarray] = None
        consecutive_reuse_count = 0

        for seq_idx, rec in enumerate(selected_records):
            img_h, img_w = rec["rgb_frame"].shape[:2]

            if rec["has_face"]:
                crop = rec["crop"]
                last_good_crop = crop
                consecutive_reuse_count = 0
                face_detected_count += 1
                final_crops.append(crop)
                if debug_dir:
                    _save_debug_frame(
                        debug_dir=debug_dir,
                        seq_idx=seq_idx,
                        rgb_frame=rec["rgb_frame"],
                        detections=rec["detections"],
                        accepted_bb=rec["accepted_bb"],
                        final_crop=crop,
                        img_w=img_w,
                        img_h=img_h,
                        reason="accepted",
                    )
            else:
                fallback_count += 1
                # Duplicate limit: do NOT duplicate the same crop more than twice in a row
                if last_good_crop is not None and consecutive_reuse_count < 2:
                    crop = last_good_crop.copy()
                    consecutive_reuse_count += 1
                    fallback_note = f"reuse #{consecutive_reuse_count}"
                    logger.warning(
                        "Frame %d (seq %d/%d): no accepted face (%s), reusing last good crop (%s).",
                        rec["frame_idx"], seq_idx + 1, num_frames, rec["reason"], fallback_note,
                    )
                else:
                    crop = _center_crop(rec["rgb_frame"], face_size)
                    consecutive_reuse_count = 0
                    last_good_crop = None  # break reuse streak
                    fallback_note = "centre-crop"
                    logger.warning(
                        "Frame %d (seq %d/%d): no accepted face (%s), using centre-crop fallback.",
                        rec["frame_idx"], seq_idx + 1, num_frames, rec["reason"],
                    )

                final_crops.append(crop)
                if debug_dir:
                    _save_debug_frame(
                        debug_dir=debug_dir,
                        seq_idx=seq_idx,
                        rgb_frame=rec["rgb_frame"],
                        detections=rec["detections"],
                        accepted_bb=None,
                        final_crop=crop,
                        img_w=img_w,
                        img_h=img_h,
                        reason=rec["reason"],
                        fallback_note=fallback_note,
                    )

    # ------------------------------------------------------------------
    # 4. Normalise → Tensor
    # ------------------------------------------------------------------
    stacked = np.stack(final_crops, axis=0)  # (num_frames, H, W, 3)
    tensor = torch.from_numpy(stacked).float().permute(0, 3, 1, 2) / 255.0

    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return PreprocessResult(tensor, face_detected_count, fallback_count)
