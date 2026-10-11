"""Comprehensive test for the end-to-end FinShield Model 2 pipeline.

Verifies:
1. Package exports in `src`.
2. Device resolution via `get_default_device()`.
3. Model instantiation and checkpoint error handling via `load_model()`.
4. End-to-end video prediction via `predict_video()` on synthetic video data.
5. Batch video prediction via `predict_batch()`.
6. Command-line interface invocation via `predict.py`.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


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

    @patch("src.pipeline.preprocess_video")
    def test_predict_video_with_mocked_frames(self, mock_preprocess):
        """Verify predict_video pipeline returns expected structure and diagnostics."""
        import torch
        from src.pipeline import load_model, predict_video
        from src.video_preprocessing import PreprocessResult

        # Create a mock 20-frame preprocessed tensor (20, 3, 112, 112)
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

            # Assert key fields exist
            self.assertIn("video_path", result)
            self.assertIn("prediction", result)
            self.assertIn("confidence", result)
            self.assertIn("probability", result)
            self.assertIn("temporal_score", result)
            self.assertIn("frequency_score", result)
            self.assertIn("preprocessing", result)

            # Assert ranges and types
            self.assertIn(result["prediction"], ["REAL", "FAKE"])
            self.assertTrue(0.5 <= result["confidence"] <= 1.0)
            self.assertTrue(0.0 <= result["probability"] <= 1.0)
            self.assertTrue(0.0 <= result["temporal_score"] <= 1.0)
            self.assertTrue(0.0 <= result["frequency_score"] <= 1.0)

            # Assert preprocessing metadata
            pre = result["preprocessing"]
            self.assertEqual(pre["frames_processed"], 20)
            self.assertEqual(pre["faces_detected"], 18)
            self.assertEqual(pre["fallbacks_used"], 2)

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


if __name__ == "__main__":
    unittest.main()
