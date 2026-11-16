# Model with feature visualization

from typing import Dict, Iterable, List, Mapping, Optional, Tuple
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models
from .frequency_feature import LightweightFrequencyFeature


LABELS: Tuple[str, str] = ("REAL", "FAKE")


class Model(nn.Module):

    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False,
        use_frequency_feature=True
    ):
        super(Model, self).__init__()

        self.use_frequency_feature = use_frequency_feature
        if self.use_frequency_feature:
            self.freq_feature = LightweightFrequencyFeature(high_freq_threshold=0.5)

        # The project checkpoint supplies the pretrained weights. Avoid an
        # implicit network download at construction time so runs are repeatable.
        model = models.resnext50_32x4d(weights=None)

        self.model = nn.Sequential(*list(model.children())[:-2])

        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bias=False,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # --- Fusion classifier (temporal + frequency features → logits) ---
        self.fusion_fc1 = nn.Linear(2048 + 2, 64)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(0.4)
        self.fusion_fc2 = nn.Linear(64, num_classes)

    def extract_sequence_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_length, c, h, w = x.shape

        x_reshaped = x.view(batch_size * seq_length, c, h, w)

        fmap = self.model(x_reshaped)

        x_pooled = self.avgpool(fmap)

        x_pooled = x_pooled.view(batch_size, seq_length, 2048)

        return fmap, x_pooled

    def classify_sequence_features(self, x_pooled: Tensor) -> Tensor:
        x_lstm, _ = self.lstm(x_pooled, None)

        logits = self.linear1(self.dp(self.relu(x_lstm[:, -1, :])))

        return logits

    def classify_fused_features(
        self,
        x_pooled: Tensor,
        temporal_score: Tensor,
        frequency_score: Tensor,
    ) -> Tensor:
        """Classify using LSTM features fused with temporal and frequency scores.

        Parameters
        ----------
        x_pooled : Tensor
            Pooled sequence features, shape ``[batch, seq_length, 2048]``.
        temporal_score : Tensor
            Per-sample temporal inconsistency score, shape ``[batch]``, in [0, 1].
        frequency_score : Tensor
            Per-sample normalized frequency anomaly score, shape ``[batch]``, in [0, 1].

        Returns
        -------
        Tensor
            Classification logits, shape ``[batch, num_classes]``.
        """
        x_lstm, _ = self.lstm(x_pooled, None)
        return self.classify_fused_summary(x_lstm[:, -1, :], temporal_score, frequency_score)

    def classify_fused_summary(self, lstm_summary: Tensor, temporal_score: Tensor, frequency_score: Tensor) -> Tensor:
        """Run the fusion MLP from a cached frozen LSTM output."""
        lstm_out = self.dp(self.relu(lstm_summary))

        # Concatenate diagnostic scores as extra features
        variant = getattr(self, "inference_variant", "baseline")
        if variant == "temporal_only":
            frequency_score = torch.zeros_like(frequency_score)
        elif variant == "dct_only":
            temporal_score = torch.zeros_like(temporal_score)
        fused = torch.cat(
            [lstm_out, temporal_score.unsqueeze(-1), frequency_score.unsqueeze(-1)],
            dim=-1,
        )  # [batch, 2050]

        # Fusion MLP
        out = self.fusion_fc1(fused)
        out = self.fusion_relu(out)
        out = self.fusion_dropout(out)
        logits = self.fusion_fc2(out)
        return logits

    def enable_advanced_head(self, variant: str) -> None:
        """Attach a trainable richer-temporal head without changing legacy keys."""
        if variant not in {"attention_rich", "gated_rich", "attention_bands"}:
            raise ValueError(f"Unsupported advanced head: {variant}")
        self.advanced_variant = variant
        frequency_dims = 3 if variant == "attention_bands" else 1
        diagnostic_dims = 3 + frequency_dims
        if variant in {"attention_rich", "attention_bands"}:
            self.temporal_attention = nn.Linear(2048, 1)
            self.advanced_fc1 = nn.Linear(2048 + diagnostic_dims, 64)
            self.advanced_fc2 = nn.Linear(64, 2)
            self.advanced_dropout = nn.Dropout(0.4)
        else:
            self.diagnostic_projection = nn.Linear(diagnostic_dims, 2048)
            self.fusion_gate = nn.Linear(4096, 2048)
            self.gated_classifier = nn.Linear(2048, 2)

    def classify_advanced_features(self, x_pooled: Tensor, frequency_score: Tensor) -> Tensor:
        """Attention pooling or gated fusion with rich temporal diagnostics.

        All supplied sequences contain detected face crops only, so there is no
        padding to include in attention. The last four fusion values are mean,
        standard deviation and p90 of adjacent embedding MADs, then DCT score.
        """
        lstm_out, _ = self.lstm(x_pooled, None)
        if x_pooled.shape[1] > 1:
            deltas = (x_pooled[:, 1:] - x_pooled[:, :-1]).abs().mean(dim=-1)
            temporal_stats = torch.stack([deltas.mean(dim=1), deltas.std(dim=1, unbiased=False),
                                          torch.quantile(deltas, 0.9, dim=1)], dim=-1)
        else:
            temporal_stats = x_pooled.new_zeros((x_pooled.shape[0], 3))
        return self.classify_advanced_lstm_outputs(lstm_out, temporal_stats, frequency_score)

    def classify_advanced_lstm_outputs(self, lstm_out: Tensor, temporal_stats: Tensor, frequency_score: Tensor) -> Tensor:
        """Run advanced heads from cached frozen LSTM outputs and MAD summaries."""
        variant = getattr(self, "advanced_variant", None)
        if variant not in {"attention_rich", "gated_rich", "attention_bands"}:
            raise RuntimeError("No advanced fusion head is attached")
        attention_logits = getattr(self, "temporal_attention", None)
        if attention_logits is not None:
            weights = torch.softmax(attention_logits(lstm_out).squeeze(-1), dim=1)
            summary = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        else:
            summary = lstm_out[:, -1, :]
        if frequency_score.ndim == 1: frequency_score = frequency_score.unsqueeze(-1)
        diagnostics = torch.cat([temporal_stats, frequency_score], dim=-1)
        if variant in {"attention_rich", "attention_bands"}:
            fused = torch.cat([summary, diagnostics], dim=-1)
            hidden = F.relu(self.advanced_fc1(fused))
            return self.advanced_fc2(self.advanced_dropout(hidden))
        projected = torch.tanh(self.diagnostic_projection(diagnostics))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([summary, projected], dim=-1)))
        return self.gated_classifier(gate * summary + (1.0 - gate) * projected)

    def forward(self, x: Tensor, use_fusion_classifier: bool = True):
        freq_score = None
        temporal_score = None

        if self.use_frequency_feature:
            freq_score = self.freq_feature(x)

        fmap, x_pooled = self.extract_sequence_features(x)

        if use_fusion_classifier:
            # Compute diagnostic scores for fusion
            temporal_score = temporal_inconsistency_score(x_pooled)
            frequency_score_normalized = (
                normalize_frequency_score(freq_score)
                if freq_score is not None
                else torch.zeros(x_pooled.shape[0], device=x_pooled.device)
            )
            logits = self.classify_fused_features(
                x_pooled, temporal_score, frequency_score_normalized
            )
        else:
            # Baseline-only path (backward compatible)
            logits = self.classify_sequence_features(x_pooled)

        # Return value shapes for backward compatibility:
        #   use_fusion + use_freq  → (fmap, logits, freq_score, temporal_score)  4-tuple
        #   use_fusion + no freq   → (fmap, logits, temporal_score)              3-tuple
        #   no fusion  + use_freq  → (fmap, logits, freq_score)                  3-tuple
        #   no fusion  + no freq   → (fmap, logits)                              2-tuple
        # TODO: consider a dict return once fusion training is stable,
        # to avoid positional-unpacking bugs from these 4 different arities.
        if use_fusion_classifier:
            if self.use_frequency_feature:
                return fmap, logits, freq_score, temporal_score
            return fmap, logits, temporal_score
        else:
            if self.use_frequency_feature:
                return fmap, logits, freq_score
            return fmap, logits


def _as_batch(frames: Tensor) -> Tensor:
    if frames.dim() == 4:
        return frames.unsqueeze(0)
    if frames.dim() != 5:
        raise ValueError(
            "Expected frames with shape (seq, channels, height, width) or "
            "(batch, seq, channels, height, width)."
        )
    return frames


def _fake_probability_from_logits(logits: Tensor) -> Tensor:
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.softmax(logits, dim=-1)[:, 1]


def temporal_inconsistency_score(sequence_features: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
    """Mean absolute embedding difference over valid adjacent frame pairs.

    Returns zero where a sample has fewer than two valid adjacent frames. The
    score is intentionally unscaled; it is a diagnostic and a fusion input.
    """
    if sequence_features.shape[1] < 2:
        return sequence_features.new_zeros(sequence_features.shape[0])
    differences = (sequence_features[:, 1:] - sequence_features[:, :-1]).abs().mean(dim=-1)
    if valid_mask is None:
        return differences.mean(dim=1)
    pair_mask = valid_mask[:, 1:].bool() & valid_mask[:, :-1].bool()
    denominator = pair_mask.sum(dim=1).clamp_min(1)
    scores = (differences * pair_mask.to(differences.dtype)).sum(dim=1) / denominator
    return torch.where(pair_mask.any(dim=1), scores, torch.zeros_like(scores))


def normalize_frequency_score(frequency_score: Tensor) -> Tensor:
    """Map the DCT energy statistic to a stable 0-1 diagnostic score."""
    frequency_score = frequency_score.float().clamp_min(0.0)
    return (frequency_score / (frequency_score + 1.0)).clamp(0.0, 1.0)


def adjusted_fake_probability(
    model_fake_probability: Tensor,
    temporal_score: Optional[Tensor] = None,
    frequency_score: Optional[Tensor] = None,
    temporal_weight: float = 0.15,
    frequency_weight: float = 0.10,
) -> Tensor:
    """Blend classifier probability with optional diagnostic scores."""
    probability = model_fake_probability
    if temporal_score is not None:
        probability = probability * (1.0 - temporal_weight) + temporal_score * temporal_weight
    if frequency_score is not None:
        probability = probability * (1.0 - frequency_weight) + frequency_score * frequency_weight
    return probability.clamp(0.0, 1.0)


@torch.no_grad()
def predict_with_diagnostics(
    model: Model,
    frames: Tensor,
    device: Optional[torch.device] = None,
    use_temporal_adjustment: bool = False,
    use_frequency_adjustment: bool = False,
    use_fusion_classifier: bool = False,
    use_advanced_classifier: bool = False,
) -> Dict[str, object]:
    """Return final prediction, confidence, and temporal/frequency diagnostics.

    Frames must already be preprocessed for the checkpoint and shaped either
    (seq, channels, height, width) or (batch, seq, channels, height, width).
    This helper reports the first item when a batch is provided.
    """
    model_was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    frames = _as_batch(frames).to(device)

    _, sequence_features = model.extract_sequence_features(frames)
    temporal_score = temporal_inconsistency_score(sequence_features)
    raw_frequency_score = model.freq_feature(frames) if model.use_frequency_feature else frames.new_zeros(frames.shape[0])
    frequency_score = normalize_frequency_score(raw_frequency_score)
    if use_advanced_classifier:
        advanced_frequency = model.freq_feature.band_energies(frames) if getattr(model, "inference_variant", "baseline") == "attention_bands" else frequency_score
        logits = model.classify_advanced_features(sequence_features, advanced_frequency)
    elif use_fusion_classifier:
        logits = model.classify_fused_features(sequence_features, temporal_score, frequency_score)
    else:
        logits = model.classify_sequence_features(sequence_features)
    temperature = getattr(model, "inference_temperature", 1.0) if (use_fusion_classifier or use_advanced_classifier) else 1.0
    logits = logits / max(float(temperature), 1e-6)

    fake_probability = _fake_probability_from_logits(logits)
    # Score blending is retained only as an explicit legacy comparison path.
    reported_fake_probability = adjusted_fake_probability(
        fake_probability,
        temporal_score if use_temporal_adjustment and not use_fusion_classifier else None,
        frequency_score if use_frequency_adjustment and not use_fusion_classifier else None,
    )

    fake_probability_value = float(reported_fake_probability[0].detach().cpu().item())
    prediction = LABELS[int(fake_probability_value >= 0.5)]
    confidence = fake_probability_value if prediction == "FAKE" else 1.0 - fake_probability_value

    if model_was_training:
        model.train()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probability": round(fake_probability_value, 4),
        "temporal_score": round(float(temporal_score[0].detach().cpu().item()), 4),
        "frequency_score": round(float(frequency_score[0].detach().cpu().item()), 4),
    }


def _metrics_from_confusion_matrix(confusion_matrix: np.ndarray) -> Dict[str, object]:
    tn, fp, fn, tp = confusion_matrix.ravel()
    total = int(confusion_matrix.sum())
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall) if (real_precision + real_recall) else 0.0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float((real_f1 + f1) / 2), 4),
        "fake_precision": round(float(precision), 4),
        "fake_recall": round(float(recall), 4),
        "denominator": total,
        "confusion_matrix": confusion_matrix.astype(int).tolist(),
    }


def _calibration_metrics(y_true: List[int], y_probability: List[float], bins: int = 10) -> Dict[str, Optional[float]]:
    if not y_true:
        return {"brier_score": None, "fake_probability_ece_10_bins": None}
    brier = sum((probability - label) ** 2 for label, probability in zip(y_true, y_probability)) / len(y_true)
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [(label, probability) for label, probability in zip(y_true, y_probability)
                   if lower <= probability < upper or (index == bins - 1 and probability == 1.0)]
        if members:
            mean_probability = sum(probability for _, probability in members) / len(members)
            observed_rate = sum(label for label, _ in members) / len(members)
            ece += len(members) / len(y_true) * abs(mean_probability - observed_rate)
    return {"brier_score": round(brier, 6), "fake_probability_ece_10_bins": round(ece, 6)}


@torch.no_grad()
def evaluate_model(
    model: Model,
    dataloader: Iterable[Tuple[Tensor, Tensor]],
    device: Optional[torch.device] = None,
    use_temporal_adjustment: bool = False,
    use_frequency_adjustment: bool = False,
    use_fusion_classifier: bool = False,
    use_advanced_classifier: bool = False,
) -> Dict[str, object]:
    """Evaluate baseline or modified predictions on labeled data.

    Labels must be encoded as 0=REAL and 1=FAKE.
    """
    model_was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    y_true: List[int] = []
    y_pred: List[int] = []
    y_probability: List[float] = []
    total_count = 0
    rejected_by_class = {"REAL": 0, "FAKE": 0}
    per_video: List[Dict[str, object]] = []
    inference_times: List[float] = []

    for batch in dataloader:
        frames, labels = batch[0], batch[1]
        paths = list(batch[2]) if len(batch) > 2 else [""] * len(labels)
        usable_counts = batch[3].view(-1).tolist() if len(batch) > 3 and isinstance(batch[3], Tensor) else [int(frames.shape[1])] * len(labels)
        rejection_reasons = list(batch[4]) if len(batch) > 4 else [None] * len(labels)
        frames = _as_batch(frames).to(device)
        labels = labels.detach().cpu().view(-1).numpy().astype(int)
        total_count += len(labels)

        if frames.shape[1] < 2:
            for path, label, usable, reason in zip(paths, labels, usable_counts, rejection_reasons):
                class_name = LABELS[int(label)]
                rejected_by_class[class_name] += 1
                per_video.append({"video_path": path, "label": class_name, "prediction": None,
                                  "fake_probability": None, "status": "insufficient_evidence",
                                  "usable_frames": int(usable),
                                  "reason": reason or "fewer_than_two_detected_face_frames"})
            continue

        if device.type == "cuda": torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"): torch.mps.synchronize()
        inference_start = time.perf_counter()
        _, sequence_features = model.extract_sequence_features(frames)
        temporal_score = temporal_inconsistency_score(sequence_features)
        frequency_score = normalize_frequency_score(model.freq_feature(frames)) if model.use_frequency_feature else frames.new_zeros(frames.shape[0])
        if use_advanced_classifier:
            advanced_frequency = model.freq_feature.band_energies(frames) if getattr(model, "inference_variant", "baseline") == "attention_bands" else frequency_score
            logits = model.classify_advanced_features(sequence_features, advanced_frequency)
        elif use_fusion_classifier:
            logits = model.classify_fused_features(sequence_features, temporal_score, frequency_score)
        else:
            logits = model.classify_sequence_features(sequence_features)
        temperature = getattr(model, "inference_temperature", 1.0) if (use_fusion_classifier or use_advanced_classifier) else 1.0
        logits = logits / max(float(temperature), 1e-6)
        fake_probability = _fake_probability_from_logits(logits)

        temporal_score = None
        frequency_score = None
        if use_temporal_adjustment:
            temporal_score = temporal_inconsistency_score(sequence_features)
        if use_frequency_adjustment:
            if model.use_frequency_feature:
                raw_frequency_score = model.freq_feature(frames)
            else:
                raw_frequency_score = LightweightFrequencyFeature(high_freq_threshold=0.5).to(device)(frames)
            frequency_score = normalize_frequency_score(raw_frequency_score)

        fake_probability = adjusted_fake_probability(
            fake_probability,
            temporal_score=temporal_score,
            frequency_score=frequency_score,
        )
        if device.type == "cuda": torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"): torch.mps.synchronize()
        batch_inference_seconds = time.perf_counter() - inference_start
        inference_times.extend([batch_inference_seconds / max(len(labels), 1)] * len(labels))

        predictions = (fake_probability >= 0.5).long().detach().cpu().view(-1).numpy().astype(int)
        probabilities = fake_probability.detach().cpu().view(-1).numpy().astype(float)
        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())
        y_probability.extend(probabilities.tolist())
        for path, label, prediction, probability, usable, infer_seconds in zip(paths, labels, predictions, probabilities, usable_counts, [batch_inference_seconds / max(len(labels), 1)] * len(labels)):
            per_video.append({"video_path": path, "label": LABELS[int(label)],
                              "prediction": LABELS[int(prediction)], "fake_probability": float(probability),
                              "status": "ok", "usable_frames": int(usable),
                              "inference_seconds": infer_seconds, "preprocessing_included": False,
                              "device": device.type})

    confusion_matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual not in (0, 1):
            raise ValueError("Labels must be encoded as 0=REAL and 1=FAKE.")
        confusion_matrix[actual, predicted] += 1

    if model_was_training:
        model.train()

    metrics = _metrics_from_confusion_matrix(confusion_matrix)
    metrics["total_videos"] = total_count
    metrics["coverage"] = round(len(y_true) / total_count, 4) if total_count else 0.0
    metrics["rejected_by_class"] = rejected_by_class
    metrics["per_video"] = per_video
    metrics["mean_inference_seconds"] = round(sum(inference_times) / len(inference_times), 6) if inference_times else None
    metrics["timing_includes_preprocessing"] = False
    metrics["device"] = device.type
    metrics["auroc"] = None
    metrics["average_precision"] = None
    if 0 in y_true and 1 in y_true:
        positives = [p for y, p in zip(y_true, y_probability) if y == 1]
        negatives = [p for y, p in zip(y_true, y_probability) if y == 0]
        metrics["auroc"] = round(sum(1.0 if p > n else 0.5 if p == n else 0.0
                                     for p in positives for n in negatives) / (len(positives) * len(negatives)), 4)
        ranked = sorted(zip(y_probability, y_true), reverse=True)
        found = 0; precision_sum = 0.0
        for rank, (_, label) in enumerate(ranked, start=1):
            if label == 1:
                found += 1; precision_sum += found / rank
        metrics["average_precision"] = round(precision_sum / len(positives), 4)
    metrics.update(_calibration_metrics(y_true, y_probability))
    return metrics


def compare_model_variants(
    model: Model,
    dataloader: Iterable[Tuple[Tensor, Tensor]],
    device: Optional[torch.device] = None,
) -> Mapping[str, Dict[str, object]]:
    """Compare classifier/score paths with one shared backbone pass per video."""
    if device is None: device = next(model.parameters()).device
    model_was_training = model.training; model.eval()
    names = ["Original ResNeXt-50 + LSTM", "Modified + Temporal Score", "Modified + Temporal + Frequency"]
    trained_variant = getattr(model, "inference_variant", "baseline")
    trained_name = None
    if trained_variant == "spec_fusion": trained_name = "Trained specification fusion"
    elif trained_variant in {"temporal_only", "dct_only"}: trained_name = f"Trained {trained_variant} head"
    elif trained_variant in {"attention_rich", "gated_rich", "attention_bands"}: trained_name = f"Trained {trained_variant}"
    if trained_name: names.append(trained_name)
    records = {name: {"truth": [], "probability": [], "per_video": [],
                      "rejected": {"REAL": 0, "FAKE": 0}, "times": []} for name in names}
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            frames, labels = batch[0], batch[1]
            paths = list(batch[2]) if len(batch) > 2 else [""] * len(labels)
            usable = batch[3].view(-1).tolist() if len(batch) > 3 and isinstance(batch[3], Tensor) else [int(frames.shape[1])] * len(labels)
            rejection_reasons = list(batch[4]) if len(batch) > 4 else [None] * len(labels)
            frames = _as_batch(frames).to(device)
            labels = labels.detach().cpu().view(-1).numpy().astype(int)
            total += len(labels)
            if frames.shape[1] < 2:
                for name in names:
                    for path, label, count, reason in zip(paths, labels, usable, rejection_reasons):
                        rec = records[name]; rec["rejected"][LABELS[int(label)]] += 1
                        rec["per_video"].append({"video_path": path, "label": LABELS[int(label)],
                            "prediction": None, "fake_probability": None, "status": "insufficient_evidence",
                            "usable_frames": int(count),
                            "reason": reason or "fewer_than_two_detected_face_frames"})
                continue
            if device.type == "cuda": torch.cuda.synchronize(device)
            elif device.type == "mps" and hasattr(torch.mps, "synchronize"): torch.mps.synchronize()
            started = time.perf_counter()
            _, features = model.extract_sequence_features(frames)
            temporal = temporal_inconsistency_score(features)
            frequency = normalize_frequency_score(model.freq_feature(frames)) if model.use_frequency_feature else frames.new_zeros(frames.shape[0])
            temperature = max(float(getattr(model, "inference_temperature", 1.0)), 1e-6)
            base_probability = _fake_probability_from_logits(model.classify_sequence_features(features))
            outputs = {
                names[0]: base_probability,
                names[1]: adjusted_fake_probability(base_probability, temporal_score=temporal),
                names[2]: adjusted_fake_probability(base_probability, temporal_score=temporal, frequency_score=frequency),
            }
            if trained_name:
                if trained_variant in {"spec_fusion", "temporal_only", "dct_only"}:
                    logits = model.classify_fused_features(features, temporal, frequency) / temperature
                else:
                    advanced_frequency = model.freq_feature.band_energies(frames) if trained_variant == "attention_bands" else frequency
                    logits = model.classify_advanced_features(features, advanced_frequency) / temperature
                outputs[trained_name] = _fake_probability_from_logits(logits)
            if device.type == "cuda": torch.cuda.synchronize(device)
            elif device.type == "mps" and hasattr(torch.mps, "synchronize"): torch.mps.synchronize()
            elapsed = time.perf_counter() - started; per_item_time = elapsed / max(len(labels), 1)
            for name, probs in outputs.items():
                rec = records[name]; values = probs.detach().cpu().view(-1).tolist()
                rec["truth"].extend(labels.tolist()); rec["probability"].extend(values)
                rec["times"].extend([per_item_time] * len(labels))
                for path, label, probability, count in zip(paths, labels, values, usable):
                    prediction = int(probability >= 0.5)
                    rec["per_video"].append({"video_path": path, "label": LABELS[int(label)],
                        "prediction": LABELS[prediction], "fake_probability": float(probability),
                        "status": "ok", "usable_frames": int(count), "inference_seconds": per_item_time,
                        "preprocessing_included": False, "device": device.type})
    results = {}
    for name, rec in records.items():
        truth, probabilities = rec["truth"], rec["probability"]
        cm = np.zeros((2, 2), dtype=int)
        predictions = [int(p >= 0.5) for p in probabilities]
        for y, pred in zip(truth, predictions): cm[y, pred] += 1
        metric = _metrics_from_confusion_matrix(cm)
        metric.update({"denominator": len(truth), "total_videos": total,
            "coverage": round(len(truth) / total, 4) if total else 0.0,
            "rejected_by_class": rec["rejected"], "per_video": rec["per_video"],
            "mean_inference_seconds": round(sum(rec["times"]) / len(rec["times"]), 6) if rec["times"] else None,
            "timing_includes_preprocessing": False, "timing_note": "Shared feature extraction and comparison heads; preprocessing excluded.",
            "device": device.type, "auroc": None, "average_precision": None})
        if 0 in truth and 1 in truth:
            positives=[p for y,p in zip(truth,probabilities) if y==1]; negatives=[p for y,p in zip(truth,probabilities) if y==0]
            metric["auroc"]=round(sum(1.0 if p>n else 0.5 if p==n else 0.0 for p in positives for n in negatives)/(len(positives)*len(negatives)),4)
            ranked=sorted(zip(probabilities,truth),reverse=True);found=0;ap=0.0
            for rank,(_,label) in enumerate(ranked,1):
                if label==1:found+=1;ap+=found/rank
            metric["average_precision"]=round(ap/len(positives),4)
        metric.update(_calibration_metrics(truth, probabilities))
        results[name]=metric
    if model_was_training: model.train()
    return results
