"""Orchestration pipeline — ties adapters, OCR, semantic, face-match and
the fusion engine into a single ``run_kyc()`` call.

Supports a **demo/mock mode** for testing the frontend without loading
the actual model weights (set env ``FINSHIELD_DEMO_MODE=1``).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from PIL import Image

from . import adapters, engine, face_match, ocr, semantic
from .schemas import (
    DocumentResult,
    FaceMatchResult,
    FusionResult,
    SemanticResult,
    VideoResult,
)

logger = logging.getLogger(__name__)

DEMO_MODE = os.getenv("FINSHIELD_DEMO_MODE", "0") == "1"


# ── Mock results for demo mode ───────────────────────────────────────

def _mock_document() -> DocumentResult:
    return DocumentResult(
        prediction=0, verdict="REAL", confidence=0.92,
        prob_real=0.92, prob_fake=0.08, risk_level="LOW",
        explanation="[DEMO] Document appears authentic.",
        action="Approve — demo mode.", reliability=0.92,
    )


def _mock_video() -> VideoResult:
    return VideoResult(
        status="ok", prediction="REAL", confidence=0.85,
        probability=0.15, temporal_score=0.12, frequency_score=0.08,
        usable_frames=18, total_frames=20, reliability=0.9,
    )


def _mock_semantic() -> SemanticResult:
    from .schemas import SemanticCheck
    return SemanticResult(
        checks=[
            SemanticCheck("name", "pass", "Demo name matches"),
            SemanticCheck("dob", "pass", "Demo DOB matches"),
            SemanticCheck("id_format", "pass", "Demo ID format valid"),
            SemanticCheck("id_number", "pass", "Demo ID number matches"),
        ],
    )


def _mock_face_match() -> FaceMatchResult:
    return FaceMatchResult(
        status="match", similarity=0.72, threshold=0.363,
        detail="Demo face match",
    )


# ── Main pipeline ────────────────────────────────────────────────────

def run_kyc(
    doc_image: Optional[Image.Image] = None,
    video_path: Optional[str] = None,
    doc_type: str = "aadhaar",
    form_name: str = "",
    form_dob: str = "",
    form_id_number: str = "",
    run_face_match: bool = True,
    run_semantic: bool = True,
) -> FusionResult:
    """Full multimodal KYC check.

    Parameters
    ----------
    doc_image : PIL Image of the uploaded ID card (or None to skip).
    video_path : path to the selfie / liveness video (or None to skip).
    doc_type : ``"aadhaar"`` | ``"pan"``
    form_name, form_dob, form_id_number : KYC form values for semantic checks.
    run_face_match : whether to run face-match between doc and video.
    run_semantic : whether to run OCR + semantic validation.

    Returns
    -------
    FusionResult
    """
    doc_result: Optional[DocumentResult] = None
    vid_result: Optional[VideoResult] = None
    sem_result: Optional[SemanticResult] = None
    fm_result: Optional[FaceMatchResult] = None

    # ── 1. Document forgery detection ────────────────────────────────
    if doc_image is not None:
        if DEMO_MODE:
            doc_result = _mock_document()
        else:
            try:
                doc_result = adapters.predict_document(doc_image, doc_type)
            except Exception as exc:
                logger.error("Model 1 inference failed: %s", exc, exc_info=True)
                doc_result = DocumentResult(
                    prediction=-1, verdict="ERROR", confidence=0.0,
                    prob_real=0.0, prob_fake=0.0, risk_level="UNCERTAIN",
                    explanation=f"Document model error: {exc}",
                    action="Escalate to manual review.", reliability=0.0,
                )

    # ── 2. Video deepfake detection ──────────────────────────────────
    if video_path is not None:
        if DEMO_MODE:
            vid_result = _mock_video()
        else:
            try:
                vid_result = adapters.predict_video(video_path)
            except Exception as exc:
                logger.error("Model 2 inference failed: %s", exc, exc_info=True)
                vid_result = VideoResult(
                    status="error", prediction=None, confidence=None,
                    probability=None, temporal_score=None, frequency_score=None,
                    reliability=0.0,
                    detail=f"{type(exc).__name__}: {exc}",
                )

    # ── 3. OCR + Semantic validation ─────────────────────────────────
    if run_semantic and doc_image is not None:
        if DEMO_MODE:
            sem_result = _mock_semantic()
        else:
            try:
                ocr_text, ocr_fields = ocr.run_ocr(doc_image, doc_type)
                sem_result = semantic.validate(
                    ocr_fields,
                    form_name=form_name,
                    form_dob=form_dob,
                    form_id_number=form_id_number,
                    doc_type=doc_type,
                )
                sem_result.ocr_text = ocr_text
            except Exception as exc:
                logger.error("Semantic validation failed: %s", exc, exc_info=True)
                sem_result = SemanticResult()

    # ── 4. Face match ────────────────────────────────────────────────
    if run_face_match and doc_image is not None and video_path is not None:
        if DEMO_MODE:
            fm_result = _mock_face_match()
        else:
            try:
                fm_result = face_match.compare(doc_image, video_path)
            except Exception as exc:
                logger.error("Face match failed: %s", exc, exc_info=True)
                fm_result = FaceMatchResult(status="error", detail=str(exc))

    # ── 5. Fuse ──────────────────────────────────────────────────────
    result = engine.fuse(
        document=doc_result,
        video=vid_result,
        semantic=sem_result,
        face_match=fm_result,
    )
    return result
