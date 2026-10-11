"""FinShield AI — Model 2: Deepfake Detection Package."""

from .dataset import (
    SyntheticEvaluationDataset,
    VideoEvaluationDataset,
    create_evaluation_dataloader,
)
from .frequency_feature import LightweightFrequencyFeature
from .model2_reference import (
    LABELS,
    Model,
    adjusted_fake_probability,
    compare_model_variants,
    evaluate_model,
    normalize_frequency_score,
    normalize_temporal_score,
    predict_with_diagnostics,
    temporal_cosine_score,
    temporal_inconsistency_score,
    temporal_mad_score,
)
from .pipeline import (
    DEFAULT_CHECKPOINT_PATH,
    get_default_device,
    load_model,
    predict_batch,
    predict_video,
)
from .video_preprocessing import PreprocessResult, preprocess_video

__all__ = [
    "LABELS",
    "Model",
    "LightweightFrequencyFeature",
    "PreprocessResult",
    "preprocess_video",
    "predict_with_diagnostics",
    "evaluate_model",
    "compare_model_variants",
    "temporal_inconsistency_score",
    "normalize_frequency_score",
    "adjusted_fake_probability",
    "DEFAULT_CHECKPOINT_PATH",
    "get_default_device",
    "load_model",
    "predict_video",
    "predict_batch",
    "VideoEvaluationDataset",
    "SyntheticEvaluationDataset",
    "create_evaluation_dataloader",
]
