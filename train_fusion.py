#!/usr/bin/env python3
"""Train the fusion head (fusion_fc1 + fusion_fc2) on cached backbone features.

Usage:
    python train_fusion.py --data-dir path/to/REAL_FAKE/ \\
        --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt \\
        --out checkpoints/model_fusion_v1.pt \\
        --epochs 30 --lr 1e-3 --batch-size 32

The script:
1. Loads the pretrained checkpoint (ResNeXt-50 + LSTM backbone).
2. Freezes everything except fusion_fc1 and fusion_fc2.
3. Runs the frozen backbone + LSTM once per video and caches
   (lstm_out, temporal_score, frequency_score, label) to --cache-dir.
4. Trains the fusion MLP on the cached features.
5. Saves only the fusion_fc1/fusion_fc2 weights plus metadata.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fusion head on cached backbone features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=str, help="Directory with REAL/ and FAKE/ subfolders of videos.")
    group.add_argument("--csv", type=str, help="CSV manifest (video_path, label).")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to base model checkpoint (.pt).")
    parser.add_argument("--out", type=str, default="checkpoints/model_fusion_v1.pt", help="Output path for fusion checkpoint.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for fusion training.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps).")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache backbone features. Defaults to a temp dir under the data directory.")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Face detection confidence threshold.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cached feature dataset
# ---------------------------------------------------------------------------

class CachedFeatureDataset(Dataset):
    """Dataset that loads pre-extracted (lstm_out, temporal_score, freq_score, label)."""

    def __init__(self, cache_files: List[str]):
        self.cache_files = cache_files

    def __len__(self) -> int:
        return len(self.cache_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        data = torch.load(self.cache_files[idx], map_location="cpu")
        return data["lstm_out"], data["temporal_score"], data["freq_score"], data["label"]


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------

def cache_features(
    video_samples: List[Tuple[str, int]],
    checkpoint_path: str,
    cache_dir: str,
    device: torch.device,
    min_detection_confidence: float = 0.3,
) -> List[str]:
    """Run the frozen backbone once per video and save features to disk."""
    from src.model2_reference import (
        Model,
        normalize_frequency_score,
        normalize_temporal_score,
        temporal_mad_score,
    )
    from src.video_preprocessing import preprocess_video

    os.makedirs(cache_dir, exist_ok=True)

    # Load model
    model = Model(num_classes=2, use_frequency_feature=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    cache_files = []
    for i, (video_path, label) in enumerate(video_samples):
        cache_file = os.path.join(cache_dir, f"features_{i:05d}.pt")
        if os.path.isfile(cache_file):
            cache_files.append(cache_file)
            continue

        try:
            res = preprocess_video(
                video_path, num_frames=20, face_size=112,
                min_detection_confidence=min_detection_confidence,
            )
            frames = res.tensor.unsqueeze(0).to(device)  # (1, 20, 3, 112, 112)

            with torch.no_grad():
                _, x_pooled = model.extract_sequence_features(frames)
                x_lstm, _ = model.lstm(x_pooled, None)
                lstm_out = model.dp(model.relu(x_lstm[:, -1, :]))  # (1, 2048)

                ts = normalize_temporal_score(temporal_mad_score(x_pooled))  # (1,)
                raw_freq = model.freq_feature(frames)
                fs = normalize_frequency_score(raw_freq)  # (1,)

            torch.save({
                "lstm_out": lstm_out.squeeze(0).cpu(),
                "temporal_score": ts.squeeze(0).cpu(),
                "freq_score": fs.squeeze(0).cpu(),
                "label": torch.tensor(label, dtype=torch.long),
                "video_path": video_path,
            }, cache_file)
            cache_files.append(cache_file)

            if (i + 1) % 10 == 0 or (i + 1) == len(video_samples):
                logger.info("Cached %d/%d videos", i + 1, len(video_samples))

        except Exception as e:
            logger.warning("Skipping %s: %s", video_path, e)

    return cache_files


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fusion(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        from src.pipeline import get_default_device
        device = get_default_device()

    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Collect video samples
    # ------------------------------------------------------------------
    video_samples: List[Tuple[str, int]] = []
    valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    if args.data_dir:
        for class_name, label in [("REAL", 0), ("real", 0), ("FAKE", 1), ("fake", 1)]:
            sub_dir = os.path.join(args.data_dir, class_name)
            if not os.path.isdir(sub_dir):
                continue
            for fname in sorted(os.listdir(sub_dir)):
                if fname.lower().endswith(valid_exts):
                    video_samples.append((os.path.join(sub_dir, fname), label))
    elif args.csv:
        import csv
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                v_path, lbl = row[0].strip(), row[1].strip().lower()
                label_map = {"0": 0, "1": 1, "real": 0, "fake": 1}
                if lbl in label_map:
                    video_samples.append((v_path, label_map[lbl]))

    if not video_samples:
        raise ValueError(
            f"No videos found. "
            f"{'Check that ' + args.data_dir + ' contains REAL/ and FAKE/ subdirectories with video files.' if args.data_dir else 'Check your CSV file.'}"
        )

    logger.info("Found %d videos total", len(video_samples))

    # ------------------------------------------------------------------
    # Split train/val by video (deterministic with seed)
    # ------------------------------------------------------------------
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(video_samples))
    val_size = max(1, int(len(video_samples) * args.val_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    logger.info("Train: %d videos, Val: %d videos", len(train_indices), len(val_indices))

    # ------------------------------------------------------------------
    # Cache backbone features
    # ------------------------------------------------------------------
    cache_dir = args.cache_dir or os.path.join(
        args.data_dir or os.path.dirname(args.csv), ".fusion_cache"
    )
    logger.info("Caching backbone features to: %s", cache_dir)

    cache_files = cache_features(
        video_samples, args.checkpoint, cache_dir, device,
        min_detection_confidence=args.min_conf,
    )

    if not cache_files:
        raise ValueError("No features were cached successfully. Check your videos and checkpoint.")

    # Build index mapping from original sample index to cache file
    full_dataset = CachedFeatureDataset(cache_files)

    # Split datasets (map original indices to cache file indices)
    # cache_files[i] corresponds to video_samples[i], so indices map directly
    train_ds = Subset(full_dataset, [i for i in train_indices if i < len(cache_files)])
    val_ds = Subset(full_dataset, [i for i in val_indices if i < len(cache_files)])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Build fusion head
    # ------------------------------------------------------------------
    from src.model2_reference import Model
    model = Model(num_classes=2, use_frequency_feature=True)
    # Load base checkpoint to get proper layer shapes
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.to(device)

    # Freeze everything except fusion layers
    for name, param in model.named_parameters():
        if name.startswith("fusion_"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Only fusion layers are unfrozen, but we still call model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for lstm_out, ts, fs, labels in train_loader:
            lstm_out = lstm_out.to(device)
            ts = ts.to(device)
            fs = fs.to(device)
            labels = labels.to(device)

            # Fused features → fusion MLP
            fused = torch.cat([lstm_out, ts.unsqueeze(-1), fs.unsqueeze(-1)], dim=-1)
            out = model.fusion_fc1(fused)
            out = model.fusion_relu(out)
            out = model.fusion_dropout(out)
            logits = model.fusion_fc2(out)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for lstm_out, ts, fs, labels in val_loader:
                lstm_out = lstm_out.to(device)
                ts = ts.to(device)
                fs = fs.to(device)
                labels = labels.to(device)

                fused = torch.cat([lstm_out, ts.unsqueeze(-1), fs.unsqueeze(-1)], dim=-1)
                out = model.fusion_fc1(fused)
                out = model.fusion_relu(out)
                out = model.fusion_dropout(out)
                logits = model.fusion_fc2(out)

                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_acc=%.4f",
            epoch, args.epochs,
            train_loss / max(train_total, 1),
            train_acc, val_acc,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
                if k.startswith("fusion_")
            }

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    logger.info("=" * 50)
    logger.info("FINAL VALIDATION METRICS")
    logger.info("=" * 50)
    logger.info("  Accuracy:  %.4f", best_val_acc)
    logger.info("  Precision: %.4f", precision)
    logger.info("  Recall:    %.4f", recall)
    logger.info("  F1:        %.4f", f1)
    logger.info("  Confusion: TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    metadata = {
        "training_args": vars(args),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "best_val_accuracy": best_val_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "date": datetime.datetime.now().isoformat(),
    }

    save_dict = {
        "state_dict": best_state,
        "metadata": metadata,
    }
    torch.save(save_dict, args.out)
    logger.info("Fusion checkpoint saved to: %s", args.out)

    # Save confusion matrix JSON alongside
    json_path = os.path.splitext(args.out)[0] + "_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metrics JSON saved to: %s", json_path)


if __name__ == "__main__":
    train_fusion(parse_args())
