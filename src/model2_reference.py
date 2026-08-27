# Model with feature visualization, diagnostic prediction, and evaluation.

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models


LABELS: Tuple[str, str] = ("REAL", "FAKE")


class Model(nn.Module):

    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False
    ):
        super(Model, self).__init__()

        model = models.resnext50_32x4d(pretrained=True)

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

    def extract_sequence_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_length, c, h, w = x.shape

        x = x.view(batch_size * seq_length, c, h, w)

        fmap = self.model(x)

        x = self.avgpool(fmap)

        x = x.view(batch_size, seq_length, 2048)

        return fmap, x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        fmap, x = self.extract_sequence_features(x)

        x_lstm, _ = self.lstm(x, None)

        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def _as_batch(frames: Tensor) -> Tensor:
    if frames.dim() == 4:
        return frames.unsqueeze(0)
    if frames.dim() != 5:
        raise ValueError(
            "Expected frames with shape (seq, channels, height, width) or "
            "(batch, seq, channels, height, width)."
        )
    return frames


def _to_probability(logits: Tensor) -> Tensor:
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.softmax(logits, dim=-1)[:, 1]


def temporal_inconsistency_score(sequence_features: Tensor) -> Tensor:
    """Return a 0-1 score for frame-to-frame feature instability."""
    if sequence_features.shape[1] < 2:
        return torch.zeros(sequence_features.shape[0], device=sequence_features.device)

    normalized = F.normalize(sequence_features, p=2, dim=-1)
    cosine_similarity = (normalized[:, 1:] * normalized[:, :-1]).sum(dim=-1)
    cosine_distance = (1.0 - cosine_similarity).clamp(min=0.0, max=2.0) / 2.0
    return cosine_distance.mean(dim=1).clamp(0.0, 1.0)


def frequency_domain_score(frames: Tensor, high_frequency_ratio: float = 0.35) -> Tensor:
    """Return a 0-1 score for high-frequency energy in each video sequence."""
    frames = _as_batch(frames).float()
    if not 0.0 < high_frequency_ratio < 1.0:
        raise ValueError("high_frequency_ratio must be between 0 and 1.")

    gray_frames = frames.mean(dim=2)
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray_frames, dim=(-2, -1)), dim=(-2, -1))
    energy = spectrum.abs().pow(2)

    height, width = gray_frames.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, device=frames.device)
    x = torch.linspace(-1.0, 1.0, width, device=frames.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx.pow(2) + yy.pow(2))
    high_frequency_mask = radius >= high_frequency_ratio

    high_energy = energy[..., high_frequency_mask].sum(dim=-1)
    total_energy = energy.sum(dim=(-2, -1)).clamp_min(1e-8)
    return (high_energy / total_energy).mean(dim=1).clamp(0.0, 1.0)


def adjusted_fake_probability(
    model_fake_probability: Tensor,
    temporal_score: Tensor,
    frequency_score: Optional[Tensor] = None,
    temporal_weight: float = 0.15,
    frequency_weight: float = 0.10,
) -> Tensor:
    """Blend model probability with diagnostic evidence without retraining."""
    probability = model_fake_probability * (1.0 - temporal_weight) + temporal_score * temporal_weight
    if frequency_score is not None:
        probability = probability * (1.0 - frequency_weight) + frequency_score * frequency_weight
    return probability.clamp(0.0, 1.0)


@torch.no_grad()
def predict_with_diagnostics(
    model: Model,
    frames: Tensor,
    device: Optional[torch.device] = None,
    use_temporal_adjustment: bool = True,
    use_frequency_adjustment: bool = True,
) -> Dict[str, object]:
    """Predict REAL/FAKE and expose confidence plus diagnostic scores.

    Args:
        model: Loaded FinShield Model 2 network.
        frames: Tensor shaped (seq, channels, height, width) or
            (1, seq, channels, height, width), already preprocessed for the
            checkpoint.
        device: Optional device override. Defaults to the model's device.
        use_temporal_adjustment: Include temporal score in the reported
            confidence.
        use_frequency_adjustment: Include frequency score in the reported
            confidence.
    """
    model_was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    frames = _as_batch(frames).to(device)
    _, sequence_features = model.extract_sequence_features(frames)
    _, logits = model(frames)

    fake_probability = _to_probability(logits)
    temporal_score = temporal_inconsistency_score(sequence_features)
    frequency_score = frequency_domain_score(frames)

    reported_probability = fake_probability
    if use_temporal_adjustment:
        reported_probability = adjusted_fake_probability(
            reported_probability,
            temporal_score,
            frequency_score if use_frequency_adjustment else None,
        )

    fake_probability_value = float(reported_probability[0].detach().cpu().item())
    prediction = LABELS[int(fake_probability_value >= 0.5)]
    confidence = fake_probability_value if prediction == "FAKE" else 1.0 - fake_probability_value

    if model_was_training:
        model.train()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "temporal_score": round(float(temporal_score[0].detach().cpu().item()), 4),
        "frequency_score": round(float(frequency_score[0].detach().cpu().item()), 4),
        "fake_probability": round(fake_probability_value, 4),
    }


def _metrics_from_confusion_matrix(confusion_matrix: np.ndarray) -> Dict[str, object]:
    tn, fp, fn, tp = confusion_matrix.ravel()
    total = int(confusion_matrix.sum())
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix.astype(int).tolist(),
    }


@torch.no_grad()
def evaluate_model(
    model: Model,
    dataloader: Iterable[Tuple[Tensor, Tensor]],
    device: Optional[torch.device] = None,
    use_temporal_adjustment: bool = False,
    use_frequency_adjustment: bool = False,
) -> Dict[str, object]:
    """Evaluate baseline or diagnostic-adjusted predictions on a dataloader.

    Labels are expected as 0=REAL and 1=FAKE.
    """
    model_was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    y_true: List[int] = []
    y_pred: List[int] = []

    for frames, labels in dataloader:
        frames = _as_batch(frames).to(device)
        labels = labels.detach().cpu().view(-1).numpy().astype(int)

        _, sequence_features = model.extract_sequence_features(frames)
        _, logits = model(frames)
        fake_probability = _to_probability(logits)

        if use_temporal_adjustment:
            temporal_score = temporal_inconsistency_score(sequence_features)
            frequency_score = frequency_domain_score(frames) if use_frequency_adjustment else None
            fake_probability = adjusted_fake_probability(
                fake_probability,
                temporal_score,
                frequency_score,
            )

        predictions = (fake_probability >= 0.5).long().detach().cpu().view(-1).numpy().astype(int)
        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

    confusion_matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual not in (0, 1):
            raise ValueError("Labels must be encoded as 0=REAL and 1=FAKE.")
        confusion_matrix[actual, predicted] += 1

    if model_was_training:
        model.train()

    return _metrics_from_confusion_matrix(confusion_matrix)


def compare_model_variants(
    model: Model,
    dataloader: Iterable[Tuple[Tensor, Tensor]],
    device: Optional[torch.device] = None,
) -> Mapping[str, Dict[str, object]]:
    """Evaluate the original model and the two requested modified variants."""
    return {
        "Original ResNeXt-50 + LSTM": evaluate_model(
            model,
            dataloader,
            device=device,
            use_temporal_adjustment=False,
            use_frequency_adjustment=False,
        ),
        "Modified + Temporal Score": evaluate_model(
            model,
            dataloader,
            device=device,
            use_temporal_adjustment=True,
            use_frequency_adjustment=False,
        ),
        "Modified + Temporal + Frequency": evaluate_model(
            model,
            dataloader,
            device=device,
            use_temporal_adjustment=True,
            use_frequency_adjustment=True,
        ),
    }