"""Smoke-test for video preprocessing.

Calls ``preprocess_video()`` on the sample video in ``test_videos/``,
prints tensor diagnostics and face-detection stats.
"""

import logging
import sys
import os

# Enable WARNING-level logs so fallback messages are visible.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from src.video_preprocessing import preprocess_video

# Use whatever video exists in test_videos/
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "test_videos")
VIDEO_FILE = None
for f in sorted(os.listdir(VIDEO_DIR)):
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        VIDEO_FILE = os.path.join(VIDEO_DIR, f)
        break

if VIDEO_FILE is None:
    print("ERROR: No video file found in test_videos/")
    sys.exit(1)


def main():
    print(f"Video path: {VIDEO_FILE}")
    print(f"{'=' * 60}")

    tensor, detected, fallback = preprocess_video(
        VIDEO_FILE,
        # BlazeFace short-range needs a low threshold for this wide-shot video.
        # Selfie/close-up videos work fine with the default 0.3.
        min_detection_confidence=0.15,
    )

    print(f"\n{'=' * 60}")
    print("TENSOR DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"  Shape : {tensor.shape}")
    print(f"  Dtype : {tensor.dtype}")
    print(f"  Min   : {tensor.min().item():.4f}")
    print(f"  Max   : {tensor.max().item():.4f}")
    print(f"  Mean  : {tensor.mean().item():.4f}")
    print(f"  Std   : {tensor.std().item():.4f}")

    # Sanity check: after ImageNet normalisation, values should be
    # roughly in [-2.5, 2.5], NOT in [0, 255] or [0, 1].
    if tensor.min() >= 0.0 and tensor.max() <= 1.0:
        print("  ⚠ WARNING: values in [0,1] — normalisation may not have been applied!")
    elif tensor.min() >= 0.0 and tensor.max() > 200.0:
        print("  ⚠ WARNING: values look like raw uint8 [0,255] — no normalisation!")
    else:
        print("  ✓ Values look correctly ImageNet-normalised")

    print(f"\n{'=' * 60}")
    print("FACE DETECTION STATS")
    print(f"{'=' * 60}")
    total = detected + fallback
    print(f"  Total frames processed : {total}")
    print(f"  Face detected          : {detected}/{total}")
    print(f"  Fallback used          : {fallback}/{total}")
    if fallback > total * 0.5:
        print("  ⚠ WARNING: >50% of frames used fallback — detection may be failing a lot!")
    elif fallback == 0:
        print("  ✓ All frames had successful face detections")
    else:
        print(f"  ℹ {fallback} frame(s) used fallback — check warnings above for details")


if __name__ == "__main__":
    main()
