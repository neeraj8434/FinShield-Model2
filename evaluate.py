#!/usr/bin/env python3
"""FinShield AI — Model 2: Evaluation and Benchmark Runner.

Runs evaluation on labeled video data and generates the submission
comparison table comparing up to 4 variants:
1. Original ResNeXt-50 + LSTM (Baseline)
2. Modified + Temporal Score (post-hoc blend)
3. Modified + Temporal + Frequency Analysis (post-hoc blend)
4. Fusion head (trained) — only when --fusion-checkpoint is provided

Usage:
    python evaluate.py --data-dir path/to/dataset/
    python evaluate.py --csv manifest.csv
    python evaluate.py --demo
    python evaluate.py --data-dir path/to/dataset/ --fusion-checkpoint checkpoints/model_fusion_v1.pt
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, Mapping, Optional

import numpy as np
import torch

from src.dataset import (
    SyntheticEvaluationDataset,
    VideoEvaluationDataset,
    create_evaluation_dataloader,
)
from src.model2_reference import (
    Model,
    _as_batch,
    _fake_probability_from_logits,
    compare_model_variants,
    normalize_frequency_score,
    normalize_temporal_score,
    temporal_mad_score,
)
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
        "--fusion-checkpoint",
        type=str,
        default=None,
        help="Path to a trained fusion head checkpoint. Enables the 4th variant "
             "'Fusion head (trained)' in the comparison table.",
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
        help="Optional path to write benchmark comparison Markdown output. "
             "Confusion matrices are saved as JSON alongside.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_fusion_variant(
    model: Model,
    dataloader,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate using the trained fusion head (classify_fused_features)."""
    from src.model2_reference import _metrics_from_confusion_matrix

    model.eval()
    y_true = []
    y_pred = []

    for frames, labels in dataloader:
        frames = _as_batch(frames).to(device)
        labels = labels.detach().cpu().view(-1).numpy().astype(int)

        _, sequence_features = model.extract_sequence_features(frames)
        ts = normalize_temporal_score(temporal_mad_score(sequence_features))
        raw_freq = model.freq_feature(frames)
        fs = normalize_frequency_score(raw_freq)

        logits = model.classify_fused_features(sequence_features, ts, fs)
        fake_prob = _fake_probability_from_logits(logits)
        predictions = (fake_prob >= 0.5).long().detach().cpu().view(-1).numpy().astype(int)

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

    cm = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        cm[actual, predicted] += 1

    return _metrics_from_confusion_matrix(cm)


VARIANT_NOTES = OrderedDict([
    ("Original ResNeXt-50 + LSTM", "Baseline checkpoint logits only"),
    ("Modified + Temporal Score", "Baseline probability blended with temporal inconsistency score (MAD, normalised)"),
    ("Modified + Temporal + Frequency", "Baseline probability blended with temporal and frequency-domain scores"),
    ("Fusion head (trained)", "Trained fusion MLP (LSTM features + temporal + frequency)"),
])


def format_markdown_table(results: Mapping[str, Optional[Dict[str, object]]]) -> str:
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model_name in VARIANT_NOTES:
        metrics = results.get(model_name)
        notes = VARIANT_NOTES[model_name]
        if metrics is None:
            lines.append(f"| {model_name} | Not run | Not run | Not run | Not run | {notes} |")
        else:
            acc = f"{metrics['accuracy']:.4f}"
            prec = f"{metrics['precision']:.4f}"
            rec = f"{metrics['recall']:.4f}"
            f1 = f"{metrics['f1']:.4f}"
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
    if args.fusion_checkpoint:
        print(f"Fusion Ckpt:      {args.fusion_checkpoint}")

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
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        fusion_checkpoint_path=args.fusion_checkpoint,
    )

    # Run the 3 baseline variants
    print("\nRunning comparative evaluation across 3 baseline variants...")
    results: Dict[str, Optional[Dict[str, object]]] = {}
    baseline_results = compare_model_variants(model, dataloader, device=torch.device(device))
    results.update(baseline_results)

    # Run fusion variant if checkpoint was provided
    if args.fusion_checkpoint:
        print("Running evaluation for Fusion head (trained)...")
        # Re-create dataloader since the previous evaluation consumed it
        dataloader = create_evaluation_dataloader(dataset, batch_size=args.batch_size)
        fusion_result = evaluate_fusion_variant(model, dataloader, device=torch.device(device))
        results["Fusion head (trained)"] = fusion_result
    else:
        results["Fusion head (trained)"] = None

    print("\n" + "=" * 70)
    print("                 SUBMISSION COMPARISON TABLE")
    print("=" * 70)
    table_md = format_markdown_table(results)
    print(table_md)
    print("=" * 70 + "\n")

    print("DETAILED CONFUSION MATRICES:")
    for name, res in results.items():
        if res is None:
            print(f"  • {name}: Not run")
            continue
        cm = res["confusion_matrix"]
        print(f"  • {name}:")
        print(f"      TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}, TP: {cm[1][1]}")

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Write markdown table
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table_md + "\n")
        print(f"\nMarkdown table exported to: {args.output}")

        # Write confusion matrices JSON alongside
        json_path = os.path.splitext(args.output)[0] + "_confusion.json"
        json_data = {}
        for name, res in results.items():
            if res is not None:
                json_data[name] = {
                    "accuracy": res["accuracy"],
                    "precision": res["precision"],
                    "recall": res["recall"],
                    "f1": res["f1"],
                    "confusion_matrix": res["confusion_matrix"],
                }
            else:
                json_data[name] = None
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"Confusion matrices JSON exported to: {json_path}")


if __name__ == "__main__":
    main()
