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
from src.model2_reference import compare_model_variants
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
        default=2,
        help="Batch size for evaluation dataloader",
    )
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
        "| Model | Accuracy | Precision | Recall | F1 | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model_name, metrics in results.items():
        acc = f"{metrics['accuracy']:.4f}"
        prec = f"{metrics['precision']:.4f}"
        rec = f"{metrics['recall']:.4f}"
        f1 = f"{metrics['f1']:.4f}"
        notes = notes_map.get(model_name, "")
        lines.append(f"| {model_name} | {acc} | {prec} | {rec} | {f1} | {notes} |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
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
        dataset = VideoEvaluationDataset.from_directory(args.data_dir)
    else:
        print(f"Dataset Mode:     CSV Manifest ({args.csv})")
        dataset = VideoEvaluationDataset.from_csv(args.csv)

    print(f"Total Samples:    {len(dataset)}")
    dataloader = create_evaluation_dataloader(dataset, batch_size=args.batch_size)

    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path=args.checkpoint, device=device)

    print("\nRunning comparative evaluation across 3 variants...")
    results = compare_model_variants(model, dataloader, device=torch.device(device))

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
