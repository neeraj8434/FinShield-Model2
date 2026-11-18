"""Unit tests for the fusion engine — no model weights needed."""

import unittest
from fusion.schemas import (
    DocumentResult,
    FaceMatchResult,
    FusionResult,
    SemanticCheck,
    SemanticResult,
    VideoResult,
)
from fusion.engine import _calibrate, _noisy_or, fuse


class TestCalibrate(unittest.TestCase):
    def test_full_reliability_passthrough(self):
        self.assertAlmostEqual(_calibrate(0.9, 1.0), 0.9)

    def test_zero_reliability_to_half(self):
        self.assertAlmostEqual(_calibrate(0.9, 0.0), 0.5)

    def test_half_reliability(self):
        self.assertAlmostEqual(_calibrate(0.8, 0.5), 0.65)


class TestNoisyOr(unittest.TestCase):
    def test_single_prob(self):
        self.assertAlmostEqual(_noisy_or([0.7]), 0.7)

    def test_two_probs(self):
        # 1 - (1-0.6)*(1-0.4) = 1 - 0.4*0.6 = 0.76
        self.assertAlmostEqual(_noisy_or([0.6, 0.4]), 0.76)

    def test_all_zero(self):
        self.assertAlmostEqual(_noisy_or([0.0, 0.0]), 0.0)

    def test_one_certain(self):
        self.assertAlmostEqual(_noisy_or([1.0, 0.3]), 1.0)


class TestFuseBasic(unittest.TestCase):
    def _make_doc(self, prob_fake=0.1, conf=0.95, verdict="REAL", risk="LOW", reliability=0.95):
        return DocumentResult(
            prediction=0 if verdict == "REAL" else 1,
            verdict=verdict, confidence=conf,
            prob_real=1 - prob_fake, prob_fake=prob_fake,
            risk_level=risk, explanation="test", action="test",
            reliability=reliability,
        )

    def _make_vid(self, prob=0.1, conf=0.9, pred="REAL", status="ok", reliability=0.9, usable=18):
        return VideoResult(
            status=status, prediction=pred, confidence=conf,
            probability=prob, temporal_score=0.1, frequency_score=0.1,
            usable_frames=usable, total_frames=20, reliability=reliability,
        )

    def test_both_real_approve(self):
        r = fuse(document=self._make_doc(), video=self._make_vid())
        self.assertEqual(r.decision, "APPROVE")
        self.assertLess(r.fused_risk, 0.35)

    def test_doc_fake_reject(self):
        r = fuse(
            document=self._make_doc(prob_fake=0.95, verdict="FAKE", risk="HIGH", conf=0.95),
            video=self._make_vid(),
        )
        self.assertEqual(r.decision, "REJECT")

    def test_face_mismatch_veto(self):
        r = fuse(
            document=self._make_doc(),
            video=self._make_vid(),
            face_match=FaceMatchResult(status="mismatch", similarity=0.1, detail="low"),
        )
        self.assertEqual(r.decision, "REJECT")
        self.assertTrue(r.vetoed)

    def test_insufficient_video_review(self):
        r = fuse(
            document=self._make_doc(),
            video=VideoResult(
                status="insufficient_evidence", prediction=None,
                confidence=None, probability=None,
                temporal_score=None, frequency_score=None,
                usable_frames=1, total_frames=20, reliability=0.0,
            ),
        )
        self.assertEqual(r.decision, "REVIEW")
        self.assertTrue(r.vetoed)

    def test_semantic_critical_fail_no_approve(self):
        sem = SemanticResult(
            checks=[SemanticCheck("name", "fail", "mismatch")],
            has_critical_fail=True,
        )
        r = fuse(
            document=self._make_doc(),
            video=self._make_vid(),
            semantic=sem,
        )
        self.assertNotEqual(r.decision, "APPROVE")

    def test_low_reliability_no_approve(self):
        r = fuse(
            document=self._make_doc(reliability=0.3),
            video=self._make_vid(),
        )
        # Low reliability on doc should prevent auto-approve
        self.assertIn(r.decision, ("REVIEW", "APPROVE"))
        # Actually with prob_fake=0.1 and reliability=0.3,
        # calibrated = 0.5 + 0.3*(0.1-0.5) = 0.38
        # combined with video calibrated = 0.5 + 0.9*(0.1-0.5) = 0.14
        # noisy-or = 1 - (1-0.38)*(1-0.14) = 1 - 0.62*0.86 ≈ 0.467
        # 0.467 > 0.35 → REVIEW
        self.assertEqual(r.decision, "REVIEW")


class TestFuseDocOnly(unittest.TestCase):
    def test_doc_only_real(self):
        doc = DocumentResult(
            prediction=0, verdict="REAL", confidence=0.9,
            prob_real=0.9, prob_fake=0.1, risk_level="LOW",
            explanation="ok", action="ok", reliability=0.9,
        )
        r = fuse(document=doc)
        self.assertEqual(r.decision, "APPROVE")

    def test_doc_only_fake(self):
        doc = DocumentResult(
            prediction=1, verdict="FAKE", confidence=0.95,
            prob_real=0.05, prob_fake=0.95, risk_level="HIGH",
            explanation="fake", action="reject", reliability=0.95,
        )
        r = fuse(document=doc)
        self.assertEqual(r.decision, "REJECT")


if __name__ == "__main__":
    unittest.main()
