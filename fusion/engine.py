"""Training-free reliability-weighted noisy-OR fusion engine.

Fraud = "at least one modality is manipulated", so we model the
complementary probability of everything being clean and subtract from 1:

    p'_m = 0.5 + r_m * (p_m - 0.5)        # reliability-shrunk probability
    risk = 1 - prod(1 - p'_m)              # noisy-OR

Where r_m is a [0, 1] reliability score for modality m.  When r=0 the
modality contributes nothing (shrinks to 0.5); when r=1 the raw
probability passes through.

Decision rules
--------------
APPROVE  — risk < APPROVE_THRESHOLD  AND  all modalities reliable  AND  no veto
REJECT   — risk >= REVIEW_THRESHOLD  OR   hard veto
REVIEW   — everything in between
"""

from __future__ import annotations

import math
from typing import List, Optional

from .config import (
    APPROVE_THRESHOLD,
    FACE_MATCH_THRESHOLD,
    MIN_RELIABILITY_APPROVE,
    REVIEW_THRESHOLD,
)
from .schemas import (
    DocumentResult,
    FaceMatchResult,
    FusionResult,
    ModalityScore,
    SemanticResult,
    VideoResult,
)


def _calibrate(p: float, reliability: float) -> float:
    """Shrink probability toward 0.5 by reliability factor."""
    return 0.5 + reliability * (p - 0.5)


def _noisy_or(probs: List[float]) -> float:
    """P(at least one) = 1 - prod(1 - p_i)."""
    clean = 1.0
    for p in probs:
        clean *= (1.0 - p)
    return 1.0 - clean


def _attention_weights(reliabilities: List[float]) -> List[float]:
    """Softmax-style weights derived from reliability (non-learned)."""
    if not reliabilities:
        return []
    # Temperature-scaled softmax of reliability
    temp = 2.0
    exps = [math.exp(r / temp) for r in reliabilities]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(reliabilities)] * len(reliabilities)
    return [e / total for e in exps]


def fuse(
    document: Optional[DocumentResult] = None,
    video: Optional[VideoResult] = None,
    semantic: Optional[SemanticResult] = None,
    face_match: Optional[FaceMatchResult] = None,
) -> FusionResult:
    """Run noisy-OR fusion with veto rules.

    At least one of *document* or *video* must be provided.
    Semantic validation and face-match are supplementary signals that
    can veto an APPROVE but never override a REJECT.
    """
    reasons: List[str] = []
    modality_scores: List[ModalityScore] = []
    raw_probs: List[float] = []
    reliabilities: List[float] = []

    # ── Collect modality probabilities ───────────────────────────────
    if document is not None:
        p = document.prob_fake
        r = document.reliability
        cp = _calibrate(p, r)
        raw_probs.append(cp)
        reliabilities.append(r)
        modality_scores.append(ModalityScore(
            name="document",
            fake_probability=p,
            reliability=r,
            calibrated_prob=round(cp, 4),
            weight=0.0,  # filled below
        ))
        if document.verdict == "FAKE":
            reasons.append(f"Document flagged FAKE ({document.confidence:.0%} conf)")
        if document.risk_level == "UNCERTAIN":
            reasons.append("Document model uncertain (confidence < 60%)")

    if video is not None and video.status == "ok" and video.probability is not None:
        p = video.probability
        r = video.reliability
        cp = _calibrate(p, r)
        raw_probs.append(cp)
        reliabilities.append(r)
        modality_scores.append(ModalityScore(
            name="video",
            fake_probability=p,
            reliability=r,
            calibrated_prob=round(cp, 4),
            weight=0.0,
        ))
        if video.prediction == "FAKE":
            reasons.append(f"Video flagged FAKE ({video.confidence:.0%} conf)")

    # ── Compute attention weights ────────────────────────────────────
    weights = _attention_weights(reliabilities)
    for ms, w in zip(modality_scores, weights):
        ms.weight = round(w, 4)

    # ── Noisy-OR risk ────────────────────────────────────────────────
    if raw_probs:
        risk = _noisy_or(raw_probs)
    else:
        risk = 0.5  # no usable modality → uncertain

    # ── Veto checks ──────────────────────────────────────────────────
    vetoed = False
    veto_reason = ""

    # Hard veto: insufficient evidence from video
    if video is not None and video.status == "insufficient_evidence":
        vetoed = True
        veto_reason = "Video has insufficient detected faces for analysis"
        reasons.append(veto_reason)

    # Hard veto: face mismatch
    if face_match is not None and face_match.status == "mismatch":
        vetoed = True
        veto_reason = "Document face does not match selfie video face"
        reasons.append(veto_reason)

    # Soft veto: critical semantic failure
    if semantic is not None and semantic.has_critical_fail:
        reasons.append("Critical semantic validation failure")
        # Escalate to at least REVIEW (don't veto to REJECT)
        risk = max(risk, APPROVE_THRESHOLD + 0.01)

    # Soft veto: semantic warnings
    if semantic is not None:
        for check in semantic.checks:
            if check.status == "fail":
                reasons.append(f"Semantic fail: {check.field} — {check.detail}")
            elif check.status == "warn":
                reasons.append(f"Semantic warn: {check.field} — {check.detail}")

    # ── Decision ─────────────────────────────────────────────────────
    if vetoed:
        # Hard veto → REJECT (face mismatch) or REVIEW (insufficient evidence)
        if face_match is not None and face_match.status == "mismatch":
            decision = "REJECT"
        else:
            decision = "REVIEW"
    elif risk >= REVIEW_THRESHOLD:
        decision = "REJECT"
    elif risk < APPROVE_THRESHOLD:
        # Check all reliabilities are above minimum
        all_reliable = all(r >= MIN_RELIABILITY_APPROVE for r in reliabilities)
        if all_reliable and not (semantic and semantic.has_critical_fail):
            decision = "APPROVE"
        else:
            decision = "REVIEW"
            if not all_reliable:
                reasons.append("Low modality reliability — manual review needed")
    else:
        decision = "REVIEW"

    if not reasons:
        if decision == "APPROVE":
            reasons.append("All checks passed")
        else:
            reasons.append("Moderate risk detected")

    return FusionResult(
        decision=decision,
        fused_risk=round(risk, 4),
        reason_codes=reasons,
        modality_scores=modality_scores,
        document=document,
        video=video,
        semantic=semantic,
        face_match=face_match,
        vetoed=vetoed,
        veto_reason=veto_reason,
    )
