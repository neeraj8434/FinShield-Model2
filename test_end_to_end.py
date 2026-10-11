"""Comprehensive test for the end-to-end FinShield Model 2 pipeline.

Verifies:
1. Package exports in `src`.
2. Device resolution via `get_default_device()`.
3. Model instantiation and checkpoint error handling via `load_model()`.
4. End-to-end video prediction via `predict_video()` on synthetic video data.
5. Batch video prediction via `predict_batch()`.
6. Command-line interface invocation via `predict.py`.
7. Temporal MAD score correctness on known values.
8. Strict-loading assertion (unexpected keys → error).
9. Box sanity filter (huge/tiny boxes rejected).
10. Default forward path is baseline (use_fusion_classifier=False).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch


class TestPipeline(unittest.TestCase):

    def test_imports(self):
        """Verify clean import of all expected public symbols from src."""
        import src
        expected_symbols = [
            "Model",
            "predict_with_diagnostics",
            "preprocess_video",
            "PreprocessResult",
            "predict_video",
            "predict_batch",
            "load_model",
            "get_default_device",
            "LightweightFrequencyFeature",
            "VideoEvaluationDataset",
            "SyntheticEvaluationDataset",
            "create_evaluation_dataloader",
            "temporal_mad_score",
            "temporal_cosine_score",
            "normalize_temporal_score",
        ]
        for sym in expected_symbols:
            self.assertTrue(hasattr(src, sym), f"Missing symbol in src: {sym}")

    def test_get_default_device(self):
        """Verify device selection returns a torch.device."""
        import torch
        from src.pipeline import get_default_device

        device = get_default_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ["cpu", "cuda", "mps"])

    def test_load_model_missing_checkpoint(self):
        """Verify FileNotFoundError is raised when pointing to nonexistent checkpoint."""
        from src.pipeline import load_model

        with self.assertRaises(FileNotFoundError):
            load_model(checkpoint_path="nonexistent_checkpoint_file.pt")

    def test_load_model_without_checkpoint(self):
        """Verify load_model successfully initializes an unweighted Model."""
        from src.model2_reference import Model
        from src.pipeline import load_model

        model = load_model(checkpoint_path=None, device="cpu")
        self.assertIsInstance(model, Model)
        self.assertFalse(model.training)

    def test_strict_loading_unexpected_keys(self):
        """Verify load_model raises RuntimeError when checkpoint has unexpected keys."""
        from src.pipeline import load_model
        from src.model2_reference import Model

        model = Model(num_classes=2)
        # Create a state_dict with a bogus key
        state_dict = model.state_dict()
        state_dict["bogus_layer.weight"] = torch.zeros(10)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(state_dict, tmp.name)
            tmp_path = tmp.name

        try:
            with self.assertRaises(RuntimeError) as ctx:
                load_model(checkpoint_path=tmp_path, device="cpu")
            self.assertIn("unexpected", str(ctx.exception).lower())
        finally:
            os.remove(tmp_path)

    def test_strict_loading_missing_non_fusion_key(self):
        """Verify load_model raises RuntimeError when checkpoint is missing non-fusion keys."""
        from src.pipeline import load_model
        from src.model2_reference import Model

        model = Model(num_classes=2)
        state_dict = model.state_dict()
        # Remove a critical key (not fusion)
        del state_dict["linear1.weight"]
        # Also remove fusion keys (they're expected missing)
        for k in list(state_dict.keys()):
            if k.startswith("fusion_"):
                del state_dict[k]

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(state_dict, tmp.name)
            tmp_path = tmp.name

        try:
            with self.assertRaises(RuntimeError) as ctx:
                load_model(checkpoint_path=tmp_path, device="cpu")
            self.assertIn("non-fusion", str(ctx.exception).lower())
        finally:
            os.remove(tmp_path)

    @patch("src.pipeline.preprocess_video")
    def test_predict_video_with_mocked_frames(self, mock_preprocess):
        """Verify predict_video pipeline returns expected structure and diagnostics."""
        import torch
        from src.pipeline import load_model, predict_video
        from src.video_preprocessing import PreprocessResult

        mock_tensor = torch.rand(20, 3, 112, 112)
        mock_preprocess.return_value = PreprocessResult(
            tensor=mock_tensor,
            num_detected=18,
            num_fallback=2,
        )

        model = load_model(checkpoint_path=None, device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = predict_video(
                video_path=tmp_path,
                model=model,
                device="cpu",
                num_frames=20,
            )

            self.assertIn("video_path", result)
            self.assertIn("prediction", result)
            self.assertIn("confidence", result)
            self.assertIn("probability", result)
            self.assertIn("temporal_score", result)
            self.assertIn("frequency_score", result)
            self.assertIn("preprocessing", result)
            self.assertIn("mode", result)
            self.assertIn("reliable", result)
            self.assertIn("warning", result)
            self.assertTrue(result["reliable"])
            self.assertIsNone(result["warning"])

            self.assertIn(result["prediction"], ["REAL", "FAKE"])
            self.assertTrue(0.5 <= result["confidence"] <= 1.0)
            self.assertTrue(0.0 <= result["probability"] <= 1.0)

            pre = result["preprocessing"]
            self.assertEqual(pre["frames_processed"], 20)
            self.assertEqual(pre["faces_detected"], 18)
            self.assertEqual(pre["fallbacks_used"], 2)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @patch("src.pipeline.preprocess_video")
    def test_predict_video_unreliable_coverage(self, mock_preprocess):
        """Verify predict_video sets reliable=False when coverage < 70%."""
        import torch
        from src.pipeline import load_model, predict_video
        from src.video_preprocessing import PreprocessResult

        mock_tensor = torch.rand(20, 3, 112, 112)
        mock_preprocess.return_value = PreprocessResult(
            tensor=mock_tensor,
            num_detected=10,
            num_fallback=10,
        )

        model = load_model(checkpoint_path=None, device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = predict_video(
                video_path=tmp_path,
                model=model,
                device="cpu",
                num_frames=20,
            )

            self.assertFalse(result["reliable"])
            self.assertIsNotNone(result["warning"])
            self.assertIn("Low face-detection coverage (10/20)", result["warning"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @patch("src.pipeline.predict_video")
    def test_predict_batch(self, mock_predict_video):
        """Verify predict_batch iterates over list and aggregates results."""
        from src.pipeline import predict_batch

        mock_predict_video.side_effect = lambda video_path, **kwargs: {
            "video_path": video_path,
            "prediction": "REAL",
            "confidence": 0.95,
        }

        paths = ["vid1.mp4", "vid2.mp4"]
        results = predict_batch(paths, checkpoint_path=None, device="cpu")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["video_path"], "vid1.mp4")
        self.assertEqual(results[1]["video_path"], "vid2.mp4")

    def test_cli_help(self):
        """Verify predict.py CLI responds to --help."""
        cmd = [sys.executable, "predict.py", "--help"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("FinShield AI", proc.stdout)
        self.assertIn("--checkpoint", proc.stdout)
        self.assertIn("--json", proc.stdout)
        self.assertIn("--fusion-checkpoint", proc.stdout)
        self.assertIn("--face-model", proc.stdout)
        self.assertIn("--debug-dir", proc.stdout)

    def test_synthetic_dataset(self):
        """Verify SyntheticEvaluationDataset produces expected shapes and labels."""
        from src.dataset import SyntheticEvaluationDataset, create_evaluation_dataloader

        ds = SyntheticEvaluationDataset(num_samples=4, num_frames=10, face_size=64)
        self.assertEqual(len(ds), 4)
        frames, label = ds[0]
        self.assertEqual(frames.shape, (10, 3, 64, 64))
        self.assertIn(label.item(), [0, 1])

        loader = create_evaluation_dataloader(ds, batch_size=2)
        batch_frames, batch_labels = next(iter(loader))
        self.assertEqual(batch_frames.shape, (2, 10, 3, 64, 64))
        self.assertEqual(batch_labels.shape, (2,))

    def test_evaluate_cli_help(self):
        """Verify evaluate.py responds to --help."""
        cmd = [sys.executable, "evaluate.py", "--help"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("FinShield AI", proc.stdout)
        self.assertIn("--demo", proc.stdout)
        self.assertIn("--data-dir", proc.stdout)
        self.assertIn("--fusion-checkpoint", proc.stdout)

    # ------------------------------------------------------------------
    # NEW TESTS: temporal MAD, box sanity, default forward path
    # ------------------------------------------------------------------

    def test_temporal_mad_score_known_values(self):
        """Verify temporal_mad_score on a hand-made tensor with known values."""
        from src.model2_reference import temporal_mad_score

        # 1 batch, 3 frames, 4-dim features
        # Frame 0: [1, 2, 3, 4]
        # Frame 1: [2, 4, 6, 8]   → diff from frame 0: [1, 2, 3, 4] → mean = 2.5
        # Frame 2: [3, 3, 3, 3]   → diff from frame 1: [1, 1, 3, 5] → mean = 2.5
        # MAD = mean of [2.5, 2.5] = 2.5
        features = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0],
             [2.0, 4.0, 6.0, 8.0],
             [3.0, 3.0, 3.0, 3.0]],
        ])
        score = temporal_mad_score(features)
        self.assertEqual(score.shape, (1,))
        self.assertAlmostEqual(score.item(), 2.5, places=5)

    def test_temporal_mad_score_single_frame(self):
        """Verify temporal_mad_score returns 0 for single-frame input."""
        from src.model2_reference import temporal_mad_score

        features = torch.rand(2, 1, 2048)
        score = temporal_mad_score(features)
        self.assertEqual(score.shape, (2,))
        self.assertEqual(score[0].item(), 0.0)
        self.assertEqual(score[1].item(), 0.0)

    def test_normalize_temporal_score(self):
        """Verify normalize_temporal_score maps [0, +inf) to [0, 1)."""
        from src.model2_reference import normalize_temporal_score

        scores = torch.tensor([0.0, 1.0, 10.0, 100.0])
        normed = normalize_temporal_score(scores)
        self.assertAlmostEqual(normed[0].item(), 0.0, places=5)
        self.assertAlmostEqual(normed[1].item(), 0.5, places=5)
        self.assertAlmostEqual(normed[2].item(), 10.0 / 11.0, places=4)
        self.assertTrue(normed[3].item() < 1.0)

    def test_box_sanity_filter_huge_box(self):
        """Verify _is_sane_box rejects a box covering >60% of the frame."""
        from src.video_preprocessing import _is_sane_box

        # Box covers 80% of a 100x100 frame
        self.assertFalse(_is_sane_box(90, 90, 100, 100))

    def test_box_sanity_filter_tiny_box(self):
        """Verify _is_sane_box rejects a box covering <1% of the frame."""
        from src.video_preprocessing import _is_sane_box

        # Box covers 0.04% of a 1000x1000 frame
        self.assertFalse(_is_sane_box(2, 2, 1000, 1000))

    def test_box_sanity_filter_bad_aspect(self):
        """Verify _is_sane_box rejects a box with aspect ratio outside [0.6, 1.6]."""
        from src.video_preprocessing import _is_sane_box

        # Very wide box: w/h = 200/50 = 4.0
        self.assertFalse(_is_sane_box(200, 50, 1000, 1000))
        # Very tall box: w/h = 50/200 = 0.25
        self.assertFalse(_is_sane_box(50, 200, 1000, 1000))

    def test_box_sanity_filter_good_box(self):
        """Verify _is_sane_box accepts a reasonable face-sized box."""
        from src.video_preprocessing import _is_sane_box

        # Box is 200x200 in a 1000x1000 frame → 4% area, aspect=1.0
        self.assertTrue(_is_sane_box(200, 200, 1000, 1000))

    def test_default_forward_is_baseline(self):
        """Verify Model.forward() defaults to baseline path (use_fusion_classifier=False)."""
        from src.model2_reference import Model

        model = Model(num_classes=2, use_frequency_feature=True)
        model.eval()
        x = torch.rand(1, 5, 3, 112, 112)

        with torch.no_grad():
            result = model(x)

        # Default (baseline + use_freq) → 3-tuple: (fmap, logits, freq_score)
        self.assertEqual(len(result), 3)
        fmap, logits, freq_score = result
        self.assertEqual(logits.shape, (1, 2))

    def test_forward_fusion_path(self):
        """Verify Model.forward(use_fusion_classifier=True) returns 4-tuple."""
        from src.model2_reference import Model

        model = Model(num_classes=2, use_frequency_feature=True)
        model.eval()
        x = torch.rand(1, 5, 3, 112, 112)

        with torch.no_grad():
            result = model(x, use_fusion_classifier=True)

        # Fusion + use_freq → 4-tuple: (fmap, logits, freq_score, temporal_score)
        self.assertEqual(len(result), 4)
        fmap, logits, freq_score, temporal_score = result
        self.assertEqual(logits.shape, (1, 2))

    def test_box_sanity_reasons(self):
        """Verify _check_box_sanity returns expected boolean and diagnostic string."""
        from src.video_preprocessing import _check_box_sanity

        # Good box
        sane, r_str = _check_box_sanity(100, 100, 1000, 1000)
        self.assertTrue(sane)
        self.assertIn("area=1.0%", r_str)
        self.assertIn("aspect=1.00", r_str)

        # Huge box (> 60%)
        sane, r_str = _check_box_sanity(900, 900, 1000, 1000)
        self.assertFalse(sane)
        self.assertIn("area=81.0%", r_str)

        # Tiny box (< 1%)
        sane, r_str = _check_box_sanity(5, 5, 1000, 1000)
        self.assertFalse(sane)
        self.assertIn("area=0.0%", r_str)

        # Extreme aspect ratio
        sane, r_str = _check_box_sanity(250, 50, 1000, 1000)
        self.assertFalse(sane)
        self.assertIn("aspect=5.00", r_str)

    def test_fallback_duplicate_limit_logic(self):
        """Verify that fallback does not duplicate the same crop more than twice in a row."""
        crops = []
        face_a = np.ones((112, 112, 3), dtype=np.uint8) * 10
        center_1 = np.ones((112, 112, 3), dtype=np.uint8) * 20
        center_2 = np.ones((112, 112, 3), dtype=np.uint8) * 30

        # Simulate streak:
        # Frame 0: Face A detected
        last_good = face_a
        reuse_count = 0
        crops.append(face_a)

        # Frame 1: missing -> reuse #1
        if last_good is not None and reuse_count < 2:
            crops.append(last_good.copy())
            reuse_count += 1

        # Frame 2: missing -> reuse #2
        if last_good is not None and reuse_count < 2:
            crops.append(last_good.copy())
            reuse_count += 1

        # Frame 3: missing -> reuse_count is 2, cannot reuse! Must centre-crop!
        if last_good is not None and reuse_count < 2:
            crops.append(last_good.copy())
            reuse_count += 1
        else:
            crops.append(center_1)
            last_good = None
            reuse_count = 0

        self.assertEqual(len(crops), 4)
        # Verify Frame 0, 1, 2 are Face A (at most 2 duplicates of Face A)
        self.assertTrue(np.array_equal(crops[0], face_a))
        self.assertTrue(np.array_equal(crops[1], face_a))
        self.assertTrue(np.array_equal(crops[2], face_a))
        # Verify Frame 3 is centre crop (NOT face A)
        self.assertFalse(np.array_equal(crops[3], face_a))
        self.assertTrue(np.array_equal(crops[3], center_1))


if __name__ == "__main__":
    unittest.main()
