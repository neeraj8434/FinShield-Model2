"""Face-match between a document photo and a selfie video frame.

Uses OpenCV's built-in DNN face detector (YuNet) and face recogniser
(SFace) — both ship as ONNX models from the opencv_zoo and require
**no TensorFlow, no training**.

The ONNX files must be downloaded once from the OpenCV zoo:
  face_detection_yunet_2023mar.onnx
  face_recognition_sface_2021dec.onnx

Place them in  <MODEL2_ROOT>/models/  or set env vars
FINSHIELD_FACE_DETECT_ONNX / FINSHIELD_FACE_RECOGNISE_ONNX.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import FACE_DETECT_ONNX, FACE_MATCH_THRESHOLD, FACE_RECOGNISE_ONNX
from .schemas import FaceMatchResult

logger = logging.getLogger(__name__)


def _load_detector(width: int, height: int):
    """Create YuNet face detector sized for a specific image."""
    if not os.path.isfile(FACE_DETECT_ONNX):
        raise FileNotFoundError(
            f"YuNet ONNX model not found at {FACE_DETECT_ONNX}. "
            "Download from https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
        )
    det = cv2.FaceDetectorYN.create(FACE_DETECT_ONNX, "", (width, height))
    det.setScoreThreshold(0.5)
    return det


def _load_recogniser():
    """Create SFace face recogniser."""
    if not os.path.isfile(FACE_RECOGNISE_ONNX):
        raise FileNotFoundError(
            f"SFace ONNX model not found at {FACE_RECOGNISE_ONNX}. "
            "Download from https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface"
        )
    return cv2.FaceRecognizerSF.create(FACE_RECOGNISE_ONNX, "")


def _detect_largest_face(
    img_bgr: np.ndarray,
) -> Optional[np.ndarray]:
    """Return the detection row for the largest face, or None."""
    h, w = img_bgr.shape[:2]
    det = _load_detector(w, h)
    _, faces = det.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return None
    # Pick largest by bounding-box area
    areas = faces[:, 2] * faces[:, 3]
    return faces[np.argmax(areas)]


def _get_embedding(
    img_bgr: np.ndarray,
    face: np.ndarray,
    recogniser,
) -> np.ndarray:
    """Align + extract 128-d SFace embedding."""
    aligned = recogniser.alignCrop(img_bgr, face)
    return recogniser.feature(aligned)


def extract_face_from_video(
    video_path: str,
    sample_frames: int = 5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Sample frames from a video and return (bgr_frame, face_detection) for
    the best (largest) detected face, or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    step = max(1, total // (sample_frames + 1))

    best_face = None
    best_frame = None
    best_area = 0.0

    for i in range(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(step * (i + 1), total - 1))
        ok, frame = cap.read()
        if not ok:
            continue
        face = _detect_largest_face(frame)
        if face is not None:
            area = float(face[2] * face[3])
            if area > best_area:
                best_area = area
                best_face = face
                best_frame = frame.copy()
    cap.release()

    if best_face is not None and best_frame is not None:
        return best_frame, best_face
    return None


def compare(
    doc_image,
    video_path: str,
    threshold: float = FACE_MATCH_THRESHOLD,
) -> FaceMatchResult:
    """Compare the face on a document image to a face from the selfie video.

    Parameters
    ----------
    doc_image : PIL.Image.Image | numpy.ndarray
        The uploaded ID document image.
    video_path : str
        Path to the selfie / liveness video.
    threshold : float
        Cosine-similarity threshold for match (SFace default ≈ 0.363).

    Returns
    -------
    FaceMatchResult
    """
    from PIL import Image

    try:
        # Prepare document image
        if isinstance(doc_image, Image.Image):
            doc_bgr = cv2.cvtColor(np.array(doc_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:
            doc_bgr = doc_image if doc_image.ndim == 3 else cv2.cvtColor(doc_image, cv2.COLOR_GRAY2BGR)

        # Detect face on document
        doc_face = _detect_largest_face(doc_bgr)
        if doc_face is None:
            return FaceMatchResult(
                status="unavailable",
                detail="No face detected on the document image",
            )

        # Detect face in video
        video_result = extract_face_from_video(video_path)
        if video_result is None:
            return FaceMatchResult(
                status="unavailable",
                detail="No face detected in selfie video",
            )
        vid_frame, vid_face = video_result

        # Compare embeddings
        recogniser = _load_recogniser()
        doc_emb = _get_embedding(doc_bgr, doc_face, recogniser)
        vid_emb = _get_embedding(vid_frame, vid_face, recogniser)
        score = float(recogniser.match(doc_emb, vid_emb, cv2.FaceRecognizerSF_FR_COSINE))

        matched = score >= threshold
        return FaceMatchResult(
            status="match" if matched else "mismatch",
            similarity=round(score, 4),
            threshold=threshold,
            detail=f"Cosine similarity {score:.4f} {'≥' if matched else '<'} {threshold}",
        )

    except FileNotFoundError as exc:
        return FaceMatchResult(status="unavailable", detail=str(exc))
    except Exception as exc:
        logger.error("Face match error: %s", exc, exc_info=True)
        return FaceMatchResult(status="error", detail=str(exc))
