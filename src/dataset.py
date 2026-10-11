"""Dataset and DataLoader utilities for FinShield AI Model 2 evaluation.

Provides PyTorch Datasets for evaluating deepfake detection models on:
1. Video directories organized by class folders (e.g., ``REAL/``, ``FAKE/``).
2. CSV manifest files with ``video_path`` and ``label`` columns.
3. In-memory synthetic tensors for offline benchmarking and pipeline validation.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .video_preprocessing import preprocess_video

logger = logging.getLogger(__name__)

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "real": 0,
    "fake": 1,
    "REAL": 0,
    "FAKE": 1,
}


class VideoEvaluationDataset(Dataset):
    """Dataset for video-based evaluation.

    Extracts face frames on-the-fly using ``preprocess_video()``, with optional
    in-memory caching.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        num_frames: int = 20,
        face_size: int = 112,
        min_detection_confidence: float = 0.3,
        cache: bool = True,
    ):
        """
        Parameters
        ----------
        samples : list of (video_path, label_int)
            Pairs of video path and ground truth label (0=REAL, 1=FAKE).
        num_frames : int
            Number of frames sampled per video.
        face_size : int
            Size of face crop.
        min_detection_confidence : float
            Confidence threshold for face detector.
        cache : bool
            Whether to cache preprocessed tensors in memory after first read.
        """
        self.samples = samples
        self.num_frames = num_frames
        self.face_size = face_size
        self.min_detection_confidence = min_detection_confidence
        self.cache = cache
        self._cached_tensors: dict[int, Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        video_path, label = self.samples[idx]

        if self.cache and idx in self._cached_tensors:
            frames = self._cached_tensors[idx]
        else:
            try:
                res = preprocess_video(
                    video_path=video_path,
                    num_frames=self.num_frames,
                    face_size=self.face_size,
                    min_detection_confidence=self.min_detection_confidence,
                )
                frames = res.tensor
            except Exception as e:
                logger.warning("Error preprocessing %s (%s). Using fallback blank frames.", video_path, e)
                frames = torch.zeros(self.num_frames, 3, self.face_size, self.face_size, dtype=torch.float32)

            if self.cache:
                self._cached_tensors[idx] = frames

        return frames, torch.tensor(label, dtype=torch.long)

    @classmethod
    def from_directory(
        cls,
        data_dir: str,
        num_frames: int = 20,
        face_size: int = 112,
        cache: bool = True,
    ) -> VideoEvaluationDataset:
        """Create dataset from a directory containing REAL/ and FAKE/ folders."""
        samples: List[Tuple[str, int]] = []
        valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

        for class_name, label in [("REAL", 0), ("real", 0), ("FAKE", 1), ("fake", 1)]:
            sub_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(sub_dir):
                continue
            for fname in sorted(os.listdir(sub_dir)):
                if fname.lower().endswith(valid_exts):
                    samples.append((os.path.join(sub_dir, fname), label))

        if not samples:
            raise ValueError(f"No video files found under REAL/ or FAKE/ subdirectories in: {data_dir}")

        return cls(samples=samples, num_frames=num_frames, face_size=face_size, cache=cache)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        num_frames: int = 20,
        face_size: int = 112,
        base_dir: Optional[str] = None,
        cache: bool = True,
    ) -> VideoEvaluationDataset:
        """Create dataset from a CSV manifest with columns: `video_path,label`."""
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Evaluation CSV manifest not found: {csv_path}")

        samples: List[Tuple[str, int]] = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                v_path, lbl_str = row[0].strip(), row[1].strip()
                if base_dir and not os.path.isabs(v_path):
                    v_path = os.path.join(base_dir, v_path)
                if lbl_str in LABEL_MAP:
                    samples.append((v_path, LABEL_MAP[lbl_str]))

        if not samples:
            raise ValueError(f"No valid labeled video entries parsed from: {csv_path}")

        return cls(samples=samples, num_frames=num_frames, face_size=face_size, cache=cache)


class SyntheticEvaluationDataset(Dataset):
    """Synthetic dataset for testing and benchmarking evaluation pipelines."""

    def __init__(self, num_samples: int = 10, num_frames: int = 20, face_size: int = 112, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.samples: List[Tuple[Tensor, int]] = []
        for i in range(num_samples):
            label = i % 2  # Alternate 0 (REAL) and 1 (FAKE)
            # Add slight artificial shift for fake frames
            frames = torch.rand(num_frames, 3, face_size, face_size)
            if label == 1:
                frames = frames + 0.1 * torch.randn_like(frames)
            self.samples.append((frames, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        frames, label = self.samples[idx]
        return frames, torch.tensor(label, dtype=torch.long)


def create_evaluation_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap dataset into a DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
