#!/usr/bin/env python3
"""Train the specification fusion head from an existing baseline checkpoint.

Manifest CSV columns: video_path,label,split (relative paths use --data-root).
Labels are REAL/FAKE or 0/1; split must be train or val. The backbone and LSTM
remain frozen for this first controlled run.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import hashlib
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.model2_reference import temporal_inconsistency_score, normalize_frequency_score
from src.pipeline import get_default_device, load_model
from src.video_preprocessing import preprocess_video


LABELS = {"real": 0, "0": 0, "fake": 1, "1": 1}


class ManifestVideos(Dataset):
    def __init__(self, rows, root, split, cache_dir, frames=20, face_size=112):
        self.rows = [r for r in rows if r["split"].lower() == split]
        self.root, self.cache_dir, self.frames, self.face_size = root, Path(cache_dir), frames, face_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rejection_reasons = Counter()
        self.invalid_reasons = {}
        if not self.rows:
            raise ValueError(f"Manifest has no {split!r} samples")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = Path(row["video_path"])
        if not path.is_absolute(): path = Path(self.root) / path
        path = path.resolve()
        if not path.is_file():
            self.invalid_reasons[idx] = "missing_video"
            return torch.empty(0, 3, self.face_size, self.face_size), int(row["label"])
        cache_key = hashlib.sha256(f"{path}|{path.stat().st_size}|{path.stat().st_mtime_ns}|{self.frames}|{self.face_size}".encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.pt"
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            frames, usable = cached["frames"].float(), int(cached["usable"])
            reason = cached.get("reason", "insufficient_detected_faces" if usable < 2 else "")
        else:
            try:
                result = preprocess_video(str(path), num_frames=self.frames, face_size=self.face_size)
                mask = result.valid_mask if result.valid_mask is not None else torch.ones(result.tensor.shape[0], dtype=torch.bool)
                frames, usable = result.tensor[mask], result.num_detected
                reason = "insufficient_detected_faces" if usable < 2 else ""
            except Exception as exc:
                frames, usable, reason = torch.empty(0, 3, self.face_size, self.face_size), 0, f"preprocessing_error:{type(exc).__name__}"
            torch.save({"frames": frames.half(), "usable": usable, "reason": reason}, cache_path)
        if usable < 2:
            self.invalid_reasons[idx] = reason or "insufficient_detected_faces"
        return frames, int(row["label"])

    def audit_usable(self):
        counts = Counter()
        rejected = Counter()
        for i, row in enumerate(self.rows):
            frames, label = self[i]
            (counts if frames.shape[0] >= 2 else rejected)[int(row["label"])] += 1
        self.rejection_reasons = Counter(self.invalid_reasons.values())
        return counts, rejected


class CachedFeatureVideos(Dataset):
    """Frame embeddings and diagnostics precomputed once for a frozen encoder."""
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        item = torch.load(self.records[idx], map_location="cpu", weights_only=True)
        return item["embeddings"], item["temporal"], item["frequency"], item["label"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def precompute_features(dataset, model, device, feature_dir, checkpoint_hash):
    feature_dir.mkdir(parents=True, exist_ok=True)
    records = []
    model.eval()
    with torch.no_grad():
        for idx, row in enumerate(dataset.rows):
            video_path = Path(row["video_path"])
            if not video_path.is_absolute(): video_path = Path(dataset.root) / video_path
            video_path = video_path.resolve()
            cache_key = hashlib.sha256(
                f"{video_path}|{video_path.stat().st_size}|{video_path.stat().st_mtime_ns}|"
                f"{dataset.frames}|{dataset.face_size}|{checkpoint_hash}".encode()
            ).hexdigest()
            feature_path = feature_dir / f"{cache_key}.pt"
            if not feature_path.is_file():
                frames, label = dataset[idx]
                if frames.shape[0] < 2: continue
                batch = frames.unsqueeze(0).to(device)
                _, embeddings = model.extract_sequence_features(batch)
                temporal = temporal_inconsistency_score(embeddings)
                frequency = normalize_frequency_score(model.freq_feature(batch))
                torch.save({"embeddings": embeddings[0].cpu(), "temporal": temporal.cpu()[0],
                            "frequency": frequency.cpu()[0], "label": int(label)}, feature_path)
            cached = torch.load(feature_path, map_location="cpu", weights_only=True)
            if int(cached["label"]) == int(row["label"]): records.append(feature_path)
            else: raise ValueError(f"Label mismatch in cached feature file for {video_path}")
            if len(records) % 50 == 0: print(f"Cached frozen features: {len(records)} / {len(dataset)} ({dataset.rows[idx]['split']})", flush=True)
    return CachedFeatureVideos(records)


def read_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            label = LABELS.get(row.get("label", "").strip().lower())
            split = row.get("split", "").strip().lower()
            if label is None or split not in {"train", "val", "test"} or not row.get("video_path"):
                raise ValueError("Each manifest row needs video_path, label (REAL/FAKE or 0/1), and split (train/val/test)")
            if split == "test":
                continue
            rows.append({"video_path": row["video_path"].strip(), "label": label, "split": split})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=".")
    p.add_argument("--init-checkpoint", required=True, help="Original compatible baseline .pt checkpoint")
    p.add_argument("--output", default="checkpoints/spec_fusion.pt")
    p.add_argument("--cache-dir", default="..\\work\\preprocessed_face_cache")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.batch_size != 1:
        p.error("Variable usable-frame counts require --batch-size 1")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or get_default_device())
    rows = read_manifest(args.manifest)
    train_ds = ManifestVideos(rows, args.data_root, "train", args.cache_dir, args.frames)
    val_ds = ManifestVideos(rows, args.data_root, "val", args.cache_dir, args.frames)
    train_counts, train_rejected = train_ds.audit_usable()
    val_counts, val_rejected = val_ds.audit_usable()
    if not train_counts[0] or not train_counts[1] or not val_counts[0] or not val_counts[1]:
        raise ValueError("After face preprocessing, both REAL and FAKE need at least one scorable train and validation video")
    model = load_model(args.init_checkpoint, device=device)
    model.inference_variant = "spec_fusion"
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fusion_fc")
    checkpoint_hash = file_sha256(args.init_checkpoint)
    feature_root = Path(args.cache_dir) / "features" / checkpoint_hash[:16] / f"{args.frames}frames"
    train_features = precompute_features(train_ds, model, device, feature_root / "train", checkpoint_hash)
    val_features = precompute_features(val_ds, model, device, feature_root / "val", checkpoint_hash)
    train_counts = Counter(int(torch.load(p, map_location="cpu", weights_only=True)["label"]) for p in train_features.records)
    val_counts = Counter(int(torch.load(p, map_location="cpu", weights_only=True)["label"]) for p in val_features.records)
    if not train_counts[0] or not train_counts[1] or not val_counts[0] or not val_counts[1]:
        raise ValueError("Both REAL and FAKE need at least one usable train and validation video")
    train_loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_features, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    class_weights = torch.tensor([sum(train_counts.values()) / (2 * train_counts[i]) for i in (0, 1)],
                                 dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val = float("inf")
    best_macro_f1 = -1.0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    log_path = os.path.splitext(args.output)[0] + ".jsonl"
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    config = {"manifest": str(Path(args.manifest).resolve()), "manifest_sha256": manifest_hash,
              "initial_checkpoint": str(Path(args.init_checkpoint).resolve()), "seed": args.seed,
              "initial_checkpoint_sha256": checkpoint_hash,
              "trainable_layers": ["fusion_fc1", "fusion_fc2"], "loss": "CrossEntropyLoss",
              "optimizer": "AdamW", "learning_rate": args.lr, "schedule": "constant",
              "batch_size": args.batch_size, "frames": args.frames, "face_size": 112,
              "epochs": args.epochs, "checkpoint_selection": "highest validation macro F1; validation loss breaks ties",
              "class_counts_after_face_filter": {"train": dict(train_counts), "val": dict(val_counts)},
              "rejected_by_class": {"train": dict(train_rejected), "val": dict(val_rejected)},
              "rejection_reasons": {"train": dict(train_ds.rejection_reasons), "val": dict(val_ds.rejection_reasons)},
              "class_weights": class_weights.detach().cpu().tolist()}
    Path(os.path.splitext(args.output)[0] + ".config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with open(log_path, "w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train(); total_loss = 0.0; count = 0
            for embeddings, temporal, frequency, labels in train_loader:
                embeddings = embeddings.to(device); temporal = temporal.to(device)
                frequency = frequency.to(device); labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.classify_fused_features(embeddings, temporal, frequency)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss): raise RuntimeError("Training loss is not finite")
                loss.backward()
                if model.fusion_fc1.weight.grad is None: raise RuntimeError("Fusion head received no gradient")
                optimizer.step(); total_loss += float(loss.item()) * labels.numel(); count += labels.numel()
            model.eval(); val_loss = 0.0; val_count = 0; correct = 0
            val_cm = torch.zeros(2, 2, dtype=torch.long)
            with torch.no_grad():
                for embeddings, temporal, frequency, labels in val_loader:
                    embeddings = embeddings.to(device); temporal = temporal.to(device)
                    frequency = frequency.to(device); labels = labels.to(device)
                    logits = model.classify_fused_features(embeddings, temporal, frequency)
                    val_loss += float(criterion(logits, labels).item()) * labels.numel()
                    predictions = logits.argmax(-1)
                    correct += int((predictions == labels).sum().item()); val_count += labels.numel()
                    for actual, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                        val_cm[actual, predicted] += 1
            tn, fp, fn, tp = val_cm.flatten().tolist()
            if count == 0 or val_count == 0:
                raise RuntimeError("No videos remained scorable in this epoch")
            f1s = []
            for support, false_positive, false_negative in ((tn, fn, fp), (tp, fp, fn)):
                precision = support / (support + false_positive) if support + false_positive else 0.0
                recall = support / (support + false_negative) if support + false_negative else 0.0
                f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
            val_macro_f1 = sum(f1s) / 2
            record = {"epoch": epoch, "train_loss": total_loss / count, "val_loss": val_loss / val_count,
                      "val_accuracy": correct / val_count, "val_macro_f1": val_macro_f1,
                      "val_confusion_matrix": val_cm.tolist(), "seed": args.seed, "lr": args.lr,
                      "trainable_parameters": "fusion_fc1,fusion_fc2", "train_count": count, "val_count": val_count,
                      "train_rejected_by_class": dict(train_rejected), "val_rejected_by_class": dict(val_rejected)}
            log.write(json.dumps(record) + "\n"); log.flush(); print(json.dumps(record))
            if val_macro_f1 > best_macro_f1 or (val_macro_f1 == best_macro_f1 and record["val_loss"] < best_val):
                best_val = record["val_loss"]
                best_macro_f1 = val_macro_f1
                torch.save({"state_dict": model.state_dict(), "variant": "spec_fusion",
                            "label_mapping": {"REAL": 0, "FAKE": 1}, "frames": args.frames,
                            "face_size": 112, "normalization": "ImageNet mean/std",
                            "seed": args.seed, "best_val_loss": best_val,
                            "best_val_macro_f1": best_macro_f1}, args.output)
    print(f"Best checkpoint: {args.output}\nTraining log: {log_path}")


if __name__ == "__main__": main()
