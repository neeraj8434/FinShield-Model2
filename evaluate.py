#!/usr/bin/env python3
"""FinShield AI — Model 2: Evaluation and Benchmark Runner.

Runs evaluation on labeled video data and generates the submission
comparison table comparing:
1. Original ResNeXt-50 + LSTM (Baseline)
2. Modified + Temporal Score
3. Modified + Temporal + Frequency Analysis

Usage:
    # Run evaluation on a directory with REAL/ and FAKE/ subfolders:
    python evaluate.py --data-dir path/to/dataset/

    # Run evaluation on a CSV manifest (video_path, label):
    python evaluate.py --csv manifest.csv

    # Run verification benchmark using synthetic dataset (no dataset needed):
    python evaluate.py --demo
"""

import argparse
import json
import os
import sys
from typing import Dict, Mapping

import torch

from src.dataset import (
    SyntheticEvaluationDataset,
    VideoEvaluationDataset,
    create_evaluation_dataloader,
)
from src.model2_reference import compare_model_variants, evaluate_model
from src.pipeline import (
    DEFAULT_CHECKPOINT_PATH,
    get_default_device,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinShield AI — Model 2 Variant Evaluation & Benchmark Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-dir",
        type=str,
        help="Path to directory containing REAL/ and FAKE/ video subdirectories",
    )
    group.add_argument(
        "--csv",
        type=str,
        help="Path to CSV manifest file with columns (video_path, label)",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run benchmark test using synthetic dataset for validation",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH if os.path.isfile(DEFAULT_CHECKPOINT_PATH) else None,
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Compute device for inference (default: auto-detected)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation dataloader",
    )
    parser.add_argument("--data-root", type=str, default=".", help="Root for relative video_path entries in the CSV")
    parser.add_argument("--split", type=str, default="test", help="Manifest split to evaluate (default: test)")
    parser.add_argument("--cache-dir", type=str, default="..\\work\\evaluation_face_cache",
                        help="Disk cache for preprocessed face crops shared across evaluation runs")
    parser.add_argument("--frames", type=int, default=20, help="Uniformly sampled frames per video")
    parser.add_argument("--variant", choices=["all", "baseline", "spec_fusion", "temporal_only", "dct_only", "attention_rich", "gated_rich", "attention_bands"], default="all",
                        help="Evaluate all comparison paths, or one path to avoid repeated backbone inference")
    parser.add_argument(
        "--demo-samples",
        type=int,
        default=10,
        help="Number of synthetic samples when using --demo",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write benchmark comparison JSON or Markdown output",
    )
    return parser.parse_args()


def format_markdown_table(results: Mapping[str, Dict[str, object]]) -> str:
    notes_map = {
        "Original ResNeXt-50 + LSTM": "Baseline checkpoint logits only",
        "Modified + Temporal Score": "Baseline probability blended with temporal inconsistency score",
        "Modified + Temporal + Frequency": "Baseline probability blended with temporal and frequency-domain scores",
    }

    lines = [
        "| Model | Accuracy | FAKE precision | FAKE recall | Macro F1 | Coverage | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name, metrics in results.items():
        acc = f"{metrics['accuracy']:.4f}"
        prec = f"{metrics['fake_precision']:.4f}"
        rec = f"{metrics['fake_recall']:.4f}"
        f1 = f"{metrics['macro_f1']:.4f}"
        coverage = f"{metrics.get('coverage', 1.0):.4f}"
        notes = notes_map.get(model_name, "")
        lines.append(f"| {model_name} | {acc} | {prec} | {rec} | {f1} | {coverage} | {notes} |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.demo and not args.checkpoint:
        raise SystemExit("No checkpoint is included. Pass the original checkpoint with --checkpoint.")
    if not args.demo and args.batch_size != 1:
        raise SystemExit("Evaluation uses batch size 1 because usable face-frame counts vary by video.")
    device = args.device or str(get_default_device())

    print("=" * 70)
    print("      FinShield AI — Model 2 Evaluation & Benchmark Runner")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Checkpoint:       {args.checkpoint or 'None (unweighted initialized model)'}")

    # Build dataset
    if args.demo:
        print(f"Dataset Mode:     Demo / Synthetic ({args.demo_samples} samples)")
        dataset = SyntheticEvaluationDataset(num_samples=args.demo_samples)
    elif args.data_dir:
        print(f"Dataset Mode:     Directory ({args.data_dir})")
        dataset = VideoEvaluationDataset.from_directory(args.data_dir, num_frames=args.frames)
    else:
        print(f"Dataset Mode:     CSV Manifest ({args.csv})")
        dataset = VideoEvaluationDataset.from_csv(args.csv, num_frames=args.frames,
                                                  base_dir=args.data_root, split=args.split,
                                                  disk_cache_dir=args.cache_dir)

    print(f"Total Samples:    {len(dataset)}")
    dataloader = create_evaluation_dataloader(dataset, batch_size=args.batch_size)

    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path=args.checkpoint, device=device)

    print("\nRunning comparative evaluation across 3 variants...")
    if args.variant == "all":
        results = compare_model_variants(model, dataloader, device=torch.device(device))
    elif args.variant == "baseline":
        results = {"Original ResNeXt-50 + LSTM": evaluate_model(model, dataloader, device=torch.device(device))}
    elif args.variant == "spec_fusion":
        if getattr(model, "inference_variant", "baseline") != "spec_fusion":
            raise SystemExit("--variant spec_fusion requires a checkpoint trained with the specification fusion head")
        results = {"Trained specification fusion": evaluate_model(
            model, dataloader, device=torch.device(device), use_fusion_classifier=True)}
    elif args.variant in {"attention_rich", "gated_rich", "attention_bands"}:
        if getattr(model, "inference_variant", "baseline") != args.variant:
            raise SystemExit(f"--variant {args.variant} requires a checkpoint trained with that head")
        results = {f"Trained {args.variant}": evaluate_model(
            model, dataloader, device=torch.device(device), use_advanced_classifier=True)}
    elif args.variant in {"temporal_only", "dct_only"}:
        if getattr(model, "inference_variant", "baseline") != args.variant:
            raise SystemExit(f"--variant {args.variant} requires a checkpoint trained with that head")
        results = {f"Trained {args.variant} head": evaluate_model(
            model, dataloader, device=torch.device(device), use_fusion_classifier=True)}
    for variant, result in results.items():
        if args.checkpoint:
            result["checkpoint_size_bytes"] = os.path.getsize(args.checkpoint)
            result["checkpoint_size_mb"] = round(os.path.getsize(args.checkpoint) / (1024 * 1024), 3)
        for prediction in result.get("per_video", []):
            prediction["model_variant"] = variant

    print("\n" + "=" * 70)
    print("                 SUBMISSION COMPARISON TABLE")
    print("=" * 70)
    table_md = format_markdown_table(results)
    print(table_md)
    print("=" * 70 + "\n")

    print("DETAILED CONFUSION MATRICES:")
    for name, res in results.items():
        cm = res["confusion_matrix"]
        print(f"  • {name}:")
        print(f"      TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}, TP: {cm[1][1]}")
        print(f"      Scored: {res['denominator']} / {res['total_videos']}; coverage: {res['coverage']:.4f}; rejected: {res['rejected_by_class']}")
        print(f"      AUROC: {res['auroc']}; AP: {res['average_precision']}; mean model time: {res['mean_inference_seconds']} s/video (preprocessing excluded)")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if args.output.endswith(".json"):
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(table_md + "\n")
        print(f"\nResults successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
