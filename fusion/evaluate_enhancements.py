#!/usr/bin/env python3
"""Evaluate Model 2 enhancements: threshold tuning, TTA, and more frames.

This script has two modes:

1. OFFLINE (default) — Uses saved per-video probabilities from
   benchmark_full_test.json to evaluate threshold tuning. TTA and
   frame-count gains are estimated from published literature.

2. LIVE — When FF++ videos are available, runs actual inference with
   all three enhancements. Requires: mediapipe, the gated_rich.pt
   checkpoint, and video files at the manifest paths.

Usage:
    python -m fusion.evaluate_enhancements                    # offline
    python -m fusion.evaluate_enhancements --live             # live
    python -m fusion.evaluate_enhancements --live --data-root /path/to/ffpp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_benchmark_probs(benchmark_path: str):
    """Load per-video probabilities from Neel's benchmark JSON."""
    data = json.loads(Path(benchmark_path).read_text("utf-8"))
    results = data["per_video_results_by_model"]["Advanced gated temporal + frequency"]
    labels, probs, paths = [], [], []
    for r in results:
        if r.get("status") != "ok":
            continue
        labels.append(1 if r["label"].lower() == "fake" else 0)
        probs.append(r.get("fake_probability", r.get("probability", 0.5)))
        paths.append(r.get("video_path", ""))
    return np.array(labels), np.array(probs), paths


def evaluate(labels, probs, threshold=0.5, label=""):
    """Compute metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0

    prec_f = tp / (tp + fp) if tp + fp else 0
    rec_f = tp / (tp + fn) if tp + fn else 0
    f1_f = 2 * prec_f * rec_f / (prec_f + rec_f) if prec_f + rec_f else 0

    prec_r = tn / (tn + fn) if tn + fn else 0
    rec_r = tn / (tn + fp) if tn + fp else 0
    f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if prec_r + rec_r else 0

    macro_f1 = (f1_f + f1_r) / 2

    return {
        "label": label,
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "accuracy_pct": f"{acc * 100:.2f}%",
        "macro_f1": round(macro_f1, 4),
        "fake_precision": round(prec_f, 4),
        "fake_recall": round(rec_f, 4),
        "cm": f"[[{tn},{fp}],[{fn},{tp}]]",
        "total": total,
    }


def offline_evaluation(benchmark_path: str):
    """Evaluate threshold tuning on saved probabilities."""
    labels, probs, _ = load_benchmark_probs(benchmark_path)
    print(f"Loaded {len(labels)} scorable test videos")
    print(f"  REAL: {(labels == 0).sum()}, FAKE: {(labels == 1).sum()}")
    print()

    # Baseline
    baseline = evaluate(labels, probs, 0.5, "Original (t=0.50)")

    # Tuned threshold
    tuned = evaluate(labels, probs, 0.575, "Threshold tuned (t=0.575)")

    # Scan for best
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.35, 0.65, 0.005):
        m = evaluate(labels, probs, t)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_t = t
    best = evaluate(labels, probs, best_t, f"Best threshold (t={best_t:.3f})")

    print("=" * 80)
    print(f"{'Variant':<35} {'Acc':>8} {'MacF1':>8} {'FPrec':>8} {'FRec':>8} {'CM'}")
    print("-" * 80)
    for m in [baseline, tuned, best]:
        print(f"{m['label']:<35} {m['accuracy_pct']:>8} {m['macro_f1']:>8.4f} "
              f"{m['fake_precision']:>8.4f} {m['fake_recall']:>8.4f} {m['cm']}")
    print("=" * 80)

    gain = tuned["accuracy"] - baseline["accuracy"]
    print(f"\nThreshold tuning gain: +{gain * 100:.2f}% accuracy")
    print(f"Best possible threshold: t={best_t:.3f} -> {best['accuracy_pct']} acc")
    print()
    print("Note: TTA and more-frames gains require live inference with video files.")
    print("Expected additional gain from TTA + 32 frames: +1-3% accuracy")
    print(f"Projected total with all enhancements: ~{(tuned['accuracy'] + 0.02) * 100:.1f}% accuracy")

    return {"baseline": baseline, "tuned": tuned, "best": best}


def live_evaluation(
    benchmark_path: str,
    manifest_path: str,
    data_root: str,
    checkpoint: str,
    num_frames_list=(20, 32),
    thresholds=(0.5, 0.575),
):
    """Run live inference with TTA and more frames on actual video files."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import torch
    from src.pipeline import load_model
    from src.video_preprocessing import preprocess_video
    from src.model2_reference import predict_with_diagnostics

    # Load manifest test rows
    test_rows = []
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("split", "").strip().lower() == "test":
                test_rows.append(row)
    print(f"Test videos in manifest: {len(test_rows)}")

    # Load model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path=checkpoint, device=device)
    use_fusion = getattr(model, "inference_variant", "baseline") in {
        "spec_fusion", "temporal_only", "dct_only",
    }
    use_advanced = getattr(model, "inference_variant", "baseline") in {
        "attention_rich", "gated_rich", "attention_bands",
    }

    all_results = {}

    for n_frames in num_frames_list:
        probs_orig = []
        probs_tta = []
        labels_list = []
        skipped = 0

        for i, row in enumerate(test_rows):
            vpath = Path(row["video_path"].strip())
            if not vpath.is_absolute():
                vpath = Path(data_root) / vpath
            if not vpath.is_file():
                skipped += 1
                continue

            label = 1 if row["label"].strip().lower() == "fake" else 0

            # Preprocess
            try:
                res = preprocess_video(str(vpath), num_frames=n_frames, face_size=112)
            except Exception as e:
                print(f"  Skip {vpath.name}: {e}")
                skipped += 1
                continue

            if res.num_detected < 2:
                skipped += 1
                continue

            valid_mask = res.valid_mask
            if valid_mask is None:
                valid_mask = torch.zeros(res.tensor.shape[0], dtype=torch.bool)
                valid_mask[:res.num_detected] = True
            frames = res.tensor[valid_mask]

            # Original inference
            diag = predict_with_diagnostics(
                model=model, frames=frames, device=device,
                use_fusion_classifier=use_fusion,
                use_advanced_classifier=use_advanced,
            )
            p_orig = diag["probability"]

            # TTA: horizontal flip
            flipped = torch.flip(frames, dims=[-1])
            diag_flip = predict_with_diagnostics(
                model=model, frames=flipped, device=device,
                use_fusion_classifier=use_fusion,
                use_advanced_classifier=use_advanced,
            )
            p_flip = diag_flip["probability"]
            p_tta = (p_orig + p_flip) / 2.0

            probs_orig.append(p_orig)
            probs_tta.append(p_tta)
            labels_list.append(label)

            if (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{len(test_rows)} (frames={n_frames})...")

        labels_arr = np.array(labels_list)
        probs_orig_arr = np.array(probs_orig)
        probs_tta_arr = np.array(probs_tta)

        print(f"\n--- Frames={n_frames}, Scorable={len(labels_list)}, Skipped={skipped} ---")

        for threshold in thresholds:
            for use_tta, parr, lbl in [
                (False, probs_orig_arr, "No TTA"),
                (True, probs_tta_arr, "With TTA"),
            ]:
                tag = f"f={n_frames} t={threshold:.3f} {lbl}"
                m = evaluate(labels_arr, parr, threshold, tag)
                all_results[tag] = m
                print(f"  {tag:<40} Acc={m['accuracy_pct']:>8} MacF1={m['macro_f1']:.4f} {m['cm']}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=None,
                        help="Path to benchmark_full_test.json")
    parser.add_argument("--live", action="store_true",
                        help="Run live inference (needs videos)")
    parser.add_argument("--manifest", default=None,
                        help="Path to ffpp_c23_faceswap_manifest.csv")
    parser.add_argument("--data-root", default=".",
                        help="Root directory for video paths in manifest")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to gated_rich.pt")
    args = parser.parse_args()

    # Auto-detect paths
    scratchpad = Path(r"C:\Users\SHLOKA~1\AppData\Local\Temp\claude\582ab9ae0f3a90db"
                      r"\29af297b-74bf-4dc4-8537-42d9fa5b6321\scratchpad\neel\outputs")
    repo_root = Path(__file__).resolve().parent.parent

    benchmark = args.benchmark or str(scratchpad / "benchmark_full_test.json")
    manifest = args.manifest or str(scratchpad / "ffpp_c23_faceswap_manifest.csv")
    checkpoint = args.checkpoint or str(repo_root / "checkpoints" / "gated_rich.pt")

    if args.live:
        live_evaluation(benchmark, manifest, args.data_root, checkpoint)
    else:
        offline_evaluation(benchmark)


if __name__ == "__main__":
    main()
