#!/usr/bin/env python3
"""Fit and save temperature scaling from the validation feature cache only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from src.pipeline import load_model


LABELS = {"real": 0, "0": 0, "fake": 1, "1": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ece(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (probabilities >= low) & (probabilities < high if index < bins - 1 else probabilities <= high)
        if mask.any():
            result += float(mask.float().mean()) * abs(float(labels[mask].float().mean()) - float(probabilities[mask].mean()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--face-size", type=int, default=112)
    parser.add_argument("--output", default=None)
    parser.add_argument("--evaluation-json", default=None,
                        help="Optional uncalibrated held-out evaluate.py JSON to transform for before/after calibration metrics")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    data_root = Path(args.data_root)
    initial_hash = sha256(Path(args.init_checkpoint))
    feature_root = Path(args.cache_dir) / "features" / initial_hash[:16] / f"{args.frames}frames" / "val"
    videos, labels = [], []
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("split", "").strip().lower() != "val":
                continue
            label = LABELS.get(row.get("label", "").strip().lower())
            if label is None:
                raise ValueError(f"Unknown label in manifest row: {row}")
            video = Path(row["video_path"].strip())
            if not video.is_absolute():
                video = data_root / video
            video = video.resolve()
            if not video.is_file():
                continue
            key = hashlib.sha256(
                f"{video}|{video.stat().st_size}|{video.stat().st_mtime_ns}|{args.frames}|{args.face_size}|{initial_hash}".encode()
            ).hexdigest()
            feature_path = feature_root / f"{key}.pt"
            if not feature_path.is_file():
                continue  # includes videos rejected by the face-quality guard
            item = torch.load(feature_path, map_location="cpu", weights_only=True)
            videos.append((item["embeddings"], item["temporal"],
                           item.get("frequency_bands"), item["frequency"]))
            labels.append(label)
    if not labels or len(set(labels)) != 2:
        raise RuntimeError(f"Calibration needs scorable REAL and FAKE validation clips; found {len(labels)} at {feature_root}")

    model = load_model(args.checkpoint, device="cpu")
    model.eval()
    variant = getattr(model, "inference_variant", "baseline")
    logits = []
    with torch.no_grad():
        for embeddings, temporal, frequency_bands, frequency_scalar in videos:
            if variant in {"attention_rich", "gated_rich", "attention_bands"}:
                frequency_input = frequency_bands if variant == "attention_bands" else frequency_scalar
                if frequency_input is None:
                    raise RuntimeError("Frequency-band features are missing; run cache_frequency_bands.py first")
                frequency_value = frequency_input.unsqueeze(0) if frequency_input.ndim else frequency_input.reshape(1)
                value = model.classify_advanced_features(embeddings.unsqueeze(0), frequency_value)
            else:
                value = model.classify_fused_features(
                    embeddings.unsqueeze(0), temporal.reshape(1), frequency_scalar.reshape(1)
                )
            logits.append(value.squeeze(0))
    logits = torch.stack(logits).float()
    targets = torch.tensor(labels, dtype=torch.long)
    temperatures = torch.logspace(-2, 2, steps=2001)
    losses = torch.stack([F.cross_entropy(logits / temperature, targets) for temperature in temperatures])
    best_index = int(losses.argmin())
    temperature = float(temperatures[best_index])
    raw_probs = logits.softmax(dim=1)[:, 1]
    calibrated_probs = (logits / temperature).softmax(dim=1)[:, 1]
    metrics = {
        "validation_videos": len(labels), "validation_label_counts": {"REAL": labels.count(0), "FAKE": labels.count(1)},
        "temperature": temperature, "nll_before": float(F.cross_entropy(logits, targets)),
        "nll_after": float(F.cross_entropy(logits / temperature, targets)),
        "brier_before": float(((raw_probs - targets.float()) ** 2).mean()),
        "brier_after": float(((calibrated_probs - targets.float()) ** 2).mean()),
        "fake_probability_ece_before_10_bins": ece(raw_probs, targets),
        "fake_probability_ece_after_10_bins": ece(calibrated_probs, targets),
        "fit_split": "validation", "manifest_sha256": sha256(manifest), "initial_checkpoint_sha256": initial_hash,
        "frames": args.frames, "feature_cache": str(feature_root.resolve()),
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["temperature"] = temperature
    checkpoint["calibration"] = {"method": "temperature_scaling", "fit_split": "val", "metrics": metrics}
    torch.save(checkpoint, checkpoint_path)
    output = Path(args.output) if args.output else checkpoint_path.with_suffix(".calibration.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.evaluation_json:
        evaluation_path = Path(args.evaluation_json)
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        row_name = ("Trained specification fusion" if variant == "spec_fusion" else
                    f"Trained {variant}" if variant in {"attention_rich", "gated_rich", "attention_bands"} else
                    f"Trained {variant} head")
        row = evaluation.get(row_name)
        if row is None:
            raise RuntimeError(f"--evaluation-json must contain a {row_name!r} result row")
        video_rows = row.get("per_video", [])
        before_p = [float(item["fake_probability"]) for item in video_rows if item.get("status") == "ok"]
        before_y = [1 if item["label"] == "FAKE" else 0 for item in video_rows if item.get("status") == "ok"]
        after_p = []
        for item in video_rows:
            if item.get("status") != "ok": continue
            p_fake = min(max(float(item["fake_probability"]), 1e-7), 1 - 1e-7)
            log_odds = math.log(p_fake / (1 - p_fake))
            calibrated = 1.0 / (1.0 + math.exp(-log_odds / temperature))
            item["fake_probability"] = calibrated
            item["prediction"] = "FAKE" if calibrated >= 0.5 else "REAL"
            after_p.append(calibrated)
        before_t = torch.tensor(before_p); after_t = torch.tensor(after_p); target_t = torch.tensor(before_y)
        heldout_report = {"denominator": len(before_y), "temperature_fit_split": "validation",
                          "test_split": "held-out test", "temperature": temperature,
                          "brier_before": float(((before_t - target_t) ** 2).mean()),
                          "brier_after": float(((after_t - target_t) ** 2).mean()),
                          "fake_probability_ece_before_10_bins": ece(before_t, target_t),
                          "fake_probability_ece_after_10_bins": ece(after_t, target_t),
                          "note": "Predictions and ranking metrics are unchanged by positive temperature scaling; only probabilities/calibration metrics are recalculated."}
        row["brier_score"] = heldout_report["brier_after"]
        row["fake_probability_ece_10_bins"] = heldout_report["fake_probability_ece_after_10_bins"]
        calibrated_path = evaluation_path.with_name(evaluation_path.stem + "_calibrated.json")
        calibrated_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        heldout_path = evaluation_path.with_name(evaluation_path.stem + "_calibration_test.json")
        heldout_path.write_text(json.dumps(heldout_report, indent=2), encoding="utf-8")
        print(json.dumps(heldout_report, indent=2))
        print(f"Calibrated held-out predictions: {calibrated_path}\nHeld-out calibration report: {heldout_path}")
    print(json.dumps(metrics, indent=2))
    print(f"Updated checkpoint: {checkpoint_path}\nCalibration report: {output}")


if __name__ == "__main__":
    main()
