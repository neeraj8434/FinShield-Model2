#!/usr/bin/env python3
"""Train richer temporal heads from the frozen-encoder feature cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.pipeline import load_model

LABELS = {"real": 0, "0": 0, "fake": 1, "1": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeatureDataset(Dataset):
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        item = torch.load(self.records[index], map_location="cpu", weights_only=True)
        frequency = item["frequency_bands"] if getattr(self, "use_bands", False) else item["frequency"]
        return item["lstm_outputs"], item["temporal_stats"], frequency, item["label"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["attention_rich", "gated_rich", "attention_bands"], required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--face-size", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest, root = Path(args.manifest), Path(args.data_root)
    initial_hash = sha256(Path(args.init_checkpoint))
    feature_base = Path(args.cache_dir) / "features" / initial_hash[:16] / f"{args.frames}frames"
    raw = {"train": [], "val": []}
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "").strip().lower()
            label = LABELS.get(row.get("label", "").strip().lower())
            if split not in raw or label is None: continue
            video = Path(row["video_path"].strip())
            if not video.is_absolute(): video = root / video
            video = video.resolve()
            if not video.is_file(): continue
            key = hashlib.sha256(f"{video}|{video.stat().st_size}|{video.stat().st_mtime_ns}|{args.frames}|{args.face_size}|{initial_hash}".encode()).hexdigest()
            feature_path = feature_base / split / f"{key}.pt"
            if feature_path.is_file(): raw[split].append(feature_path)
    train_ds, val_ds = FeatureDataset(raw["train"]), FeatureDataset(raw["val"])
    train_ds.use_bands = val_ds.use_bands = args.variant == "attention_bands"
    train_counts = Counter(int(torch.load(p, map_location="cpu", weights_only=True)["label"]) for p in train_ds.records)
    val_counts = Counter(int(torch.load(p, map_location="cpu", weights_only=True)["label"]) for p in val_ds.records)
    if not train_counts[0] or not train_counts[1] or not val_counts[0] or not val_counts[1]:
        raise RuntimeError(f"Advanced run needs cached REAL/FAKE train/val features; train={dict(train_counts)}, val={dict(val_counts)}")

    model = load_model(args.init_checkpoint, device="cpu")
    model.enable_advanced_head(args.variant)
    model.inference_variant = args.variant
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(("temporal_attention", "advanced_", "diagnostic_projection", "fusion_gate", "gated_classifier"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    class_weights = torch.tensor([sum(train_counts.values()) / (2 * train_counts[i]) for i in (0, 1)])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".jsonl")
    config = {"variant": args.variant, "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
              "initial_checkpoint": str(Path(args.init_checkpoint).resolve()), "initial_checkpoint_sha256": initial_hash,
              "trainable_layers": [n for n, p in model.named_parameters() if p.requires_grad],
              "loss": "weighted CrossEntropyLoss", "optimizer": "AdamW", "learning_rate": args.lr,
              "batch_size": 1, "frames": args.frames, "epochs": args.epochs, "seed": args.seed,
              "class_counts": {"train": dict(train_counts), "val": dict(val_counts)},
              "train_records": len(train_ds), "val_records": len(val_ds), "num_threads": args.num_threads}
    output.with_suffix(".config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_f1 = -1.0
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train(); train_losses = []
            for lstm_outputs, temporal_stats, frequency, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model.classify_advanced_lstm_outputs(lstm_outputs, temporal_stats, frequency)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss): raise RuntimeError("Non-finite advanced training loss")
                loss.backward()
                if not any(p.grad is not None for p in trainable): raise RuntimeError("Advanced head received no gradient")
                optimizer.step(); train_losses.append(float(loss.detach()))
            model.eval(); cm = torch.zeros(2, 2, dtype=torch.long); val_losses = []
            with torch.no_grad():
                for lstm_outputs, temporal_stats, frequency, labels in val_loader:
                    logits = model.classify_advanced_lstm_outputs(lstm_outputs, temporal_stats, frequency)
                    val_losses.append(float(criterion(logits, labels)))
                    for actual, predicted in zip(labels.tolist(), logits.argmax(-1).tolist()): cm[actual, predicted] += 1
            tn, fp, fn, tp = cm.flatten().tolist()
            class_f1 = []
            for support, false_pos, false_neg in ((tn, fn, fp), (tp, fp, fn)):
                precision = support / (support + false_pos) if support + false_pos else 0.0
                recall = support / (support + false_neg) if support + false_neg else 0.0
                class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            macro_f1 = sum(class_f1) / 2
            record = {"epoch": epoch, "train_loss": sum(train_losses) / len(train_losses),
                      "val_loss": sum(val_losses) / len(val_losses), "val_macro_f1": macro_f1,
                      "val_confusion_matrix": cm.tolist(), "train_count": len(train_ds), "val_count": len(val_ds),
                      "seed": args.seed, "lr": args.lr, "variant": args.variant}
            log.write(json.dumps(record) + "\n"); log.flush(); print(json.dumps(record), flush=True)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save({"state_dict": model.state_dict(), "variant": args.variant,
                            "label_mapping": {"REAL": 0, "FAKE": 1}, "frames": args.frames,
                            "face_size": args.face_size, "normalization": "ImageNet mean/std",
                            "seed": args.seed, "best_val_macro_f1": best_f1,
                            "initial_checkpoint_sha256": initial_hash}, output)
    print(f"Best validation macro F1: {best_f1:.4f}; checkpoint: {output}")


if __name__ == "__main__": main()
