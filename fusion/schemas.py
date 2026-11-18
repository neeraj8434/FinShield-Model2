"""Data classes for every inter-module contract in the fusion pipeline.

Using plain dataclasses (no pydantic dependency) so the package stays
light and the schemas are readable as documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Per-modality results ─────────────────────────────────────────────

@dataclass
class DocumentResult:
    """Output of the document-forgery adapter (Model 1)."""
    prediction: int               # 0=REAL, 1=FAKE
    verdict: str                  # "REAL" | "FAKE"
    confidence: float             # model confidence [0, 1]
    prob_real: float
    prob_fake: float
    risk_level: str               # LOW / MEDIUM / HIGH / UNCERTAIN
    explanation: str
    action: str
    xai_b64: Optional[str] = None  # Grad-CAM++ overlay (base64 PNG data URI)
    reliability: float = 1.0       # computed by adapter
    raw: Optional[Dict[str, Any]] = None   # full model output


@dataclass
class VideoResult:
    """Output of the deepfake adapter (Model 2)."""
    status: str                   # "ok" | "insufficient_evidence" | "error"
    prediction: Optional[str]     # "REAL" | "FAKE" | None
    confidence: Optional[float]
    probability: Optional[float]  # P(FAKE)
    temporal_score: Optional[float]
    frequency_score: Optional[float]
    usable_frames: int = 0
    total_frames: int = 20
    reliability: float = 1.0
    detail: str = ""              # error or status detail
    raw: Optional[Dict[str, Any]] = None


@dataclass
class SemanticCheck:
    """One semantic-validation check result."""
    field: str           # e.g. "name", "dob", "id_number", "id_format"
    status: str          # "pass" | "fail" | "warn" | "unknown"
    detail: str = ""


@dataclass
class SemanticResult:
    """Aggregate semantic-validation output."""
    checks: List[SemanticCheck] = field(default_factory=list)
    ocr_text: str = ""
    ocr_fields: Dict[str, str] = field(default_factory=dict)
    has_critical_fail: bool = False


@dataclass
class FaceMatchResult:
    """Face-match between document photo and selfie video."""
    status: str             # "match" | "mismatch" | "unavailable" | "error"
    similarity: Optional[float] = None
    threshold: float = 0.363
    detail: str = ""


# ── Fused decision ───────────────────────────────────────────────────

@dataclass
class ModalityScore:
    """Per-modality contribution to the fused decision."""
    name: str               # "document" | "video"
    fake_probability: float
    reliability: float
    calibrated_prob: float  # after reliability shrinkage
    weight: float           # attention-style weight (sums to 1)


@dataclass
class FusionResult:
    """Final fused KYC decision."""
    decision: str            # "APPROVE" | "REVIEW" | "REJECT"
    fused_risk: float        # noisy-OR combined risk [0, 1]
    reason_codes: List[str] = field(default_factory=list)

    # Per-modality detail
    modality_scores: List[ModalityScore] = field(default_factory=list)

    # Sub-results
    document: Optional[DocumentResult] = None
    video: Optional[VideoResult] = None
    semantic: Optional[SemanticResult] = None
    face_match: Optional[FaceMatchResult] = None

    # Metadata
    vetoed: bool = False
    veto_reason: str = ""
