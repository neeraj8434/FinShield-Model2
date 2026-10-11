#!/usr/bin/env python3
"""FinShield AI — Model 2: End-to-End Deepfake Detection CLI.

Run deepfake detection directly on a video file from the command line:

    python predict.py path/to/video.mp4
    python predict.py path/to/video.mp4 --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt
    python predict.py path/to/video.mp4 --json
    python predict.py path/to/video.mp4 --fusion-checkpoint checkpoints/model_fusion_v1.pt
"""

import argparse
import json
import os
import sys

from src.pipeline import (
    DEFAULT_CHECKPOINT_PATH,
    get_default_device,
    load_model,
    predict_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinShield AI — Deepfake Video Detection Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to the input video file (.mp4, .avi, .mov, etc.)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH if os.path.isfile(DEFAULT_CHECKPOINT_PATH) else None,
        help="Path to the trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--fusion-checkpoint",
        type=str,
        default=None,
        help="Path to a trained fusion head checkpoint (.pt). When provided, "
             "the prediction uses the learned fusion classifier instead of "
             "post-hoc score blending.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Compute device for model inference (default: auto-detected)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of frames to sample and analyze from the video",
    )
    parser.add_argument(
        "--face-size",
        type=int,
        default=112,
        help="Spatial resolution (pixels) for face crops",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for BlazeFace face detection",
    )
    parser.add_argument(
        "--face-model",
        type=str,
        default=None,
        help="Path to a BlazeFace TFLite model file override (e.g. full-range model). "
             "Defaults to models/blaze_face_short_range.tflite.",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Directory to save per-frame debug overlays (boxes + crops)",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable temporal score adjustment in final probability",
    )
    parser.add_argument(
        "--no-freq",
        action="store_true",
        help="Disable frequency domain score adjustment in final probability",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output inference result in machine-readable JSON format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to save the JSON result",
    )
    return parser.parse_args()


def print_formatted_summary(res: dict, device_str: str, ckpt_str: str) -> None:
    sep = "=" * 64
    line = "-" * 64
    pred = res["prediction"]
    pred_tag = f"[{pred}]"
    conf_pct = f"{res['confidence'] * 100:.2f}%"
    mode = res.get("mode", "baseline+blend")
    reliable = res.get("reliable", True)
    warning = res.get("warning")

    print("\n" + sep)
    print("        FinShield AI — Model 2: Deepfake Detection Result")
    print(sep)
    print(f"Video File:        {res['video_path']}")
    print(f"Checkpoint:        {ckpt_str}")
    print(f"Inference Device:  {device_str}")
    print(f"Mode:              {mode}")
    if not reliable and warning:
        print(line)
        print(f"⚠️  WARNING:        {warning}")
    print(line)
    if not reliable:
        print("PREDICTION:        INCONCLUSIVE (low face coverage)")
    else:
        print(f"PREDICTION:        {pred}")
    print(f"CONFIDENCE:        {conf_pct}")
    print(f"FAKE PROBABILITY:  {res['probability']:.4f}")
    print(line)
    print("DIAGNOSTIC SCORES:")
    print(f"  • Temporal Score (MAD):    {res['temporal_score']:.4f} (frame feature jitter)")
    if res.get("temporal_cosine_score") is not None:
        print(f"  • Temporal Cosine Score:   {res['temporal_cosine_score']:.4f} (cosine distance)")
    print(f"  • Frequency Anomaly Score: {res['frequency_score']:.4f} (DCT energy ratio)")
    print(line)
    pre = res["preprocessing"]
    print("FACE PREPROCESSING DIAGNOSTICS:")
    print(f"  • Frames Sampled:          {pre['frames_processed']}")
    print(f"  • Faces Detected:          {pre['faces_detected']} / {pre['frames_processed']}")
    print(f"  • Fallback Crops Used:     {pre['fallbacks_used']} / {pre['frames_processed']}")
    print(sep + "\n")


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    device = args.device or str(get_default_device())

    try:
        model = load_model(
            checkpoint_path=args.checkpoint,
            device=device,
            fusion_checkpoint_path=args.fusion_checkpoint,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = predict_video(
            video_path=args.video,
            model=model,
            device=device,
            num_frames=args.frames,
            face_size=args.face_size,
            min_detection_confidence=args.min_conf,
            use_temporal_adjustment=not args.no_temporal,
            use_frequency_adjustment=not args.no_freq,
            fusion_checkpoint_path=args.fusion_checkpoint,
            face_model_path=args.face_model,
            debug_dir=args.debug_dir,
        )
    except Exception as e:
        print(f"Inference failed with error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        ckpt_display = args.checkpoint if args.checkpoint else "None (default unweighted model)"
        print_formatted_summary(result, str(device), ckpt_display)


if __name__ == "__main__":
    main()
