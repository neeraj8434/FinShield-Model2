# FinShield AI — Model 2: Deepfake Detection

Video-based deepfake detection module for the **FinShield AI** project.

Model 2 analyzes a video by extracting facial frames, generating spatial features using **ResNeXt-50 32×4d**, modeling temporal information using an **LSTM**, and producing a Real/Fake prediction.

---

## Architecture

```text
                    INPUT VIDEO
                         │
                         ▼
                  Frame Extraction
                         │
                         ▼
                    Face Detection
                         │
                         ▼
                20 Face Frames
                  112 × 112 RGB
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 ResNeXt-50 32×4d                Lightweight Frequency
 (Spatial-Temporal Branch)       Analysis (2D DCT)
        │                                 │
        ▼                                 ▼
 2048-D Feature / Frame         High-Freq Energy Score
        │                                 │
        ▼                                 │
       LSTM                               │
 Hidden Size 2048                         │
        │                                 │
        ▼                                 │
      ReLU                                │
        │                                 │
        ▼                                 │
    Dropout 0.4                           │
        │                                 │
        ▼                                 │
   Linear Layer                           │
        │                                 │
        ▼                                 ▼
   REAL / FAKE                     ANOMALY SCORE

Current Status
Completed
 Reference deepfake detection repository cloned
 Repository Model Creation notebooks inspected
 Python virtual environment created
 Jupyter installed
 PyTorch 2.8.0 installed
 TorchVision 0.23.0 installed
 Apple Silicon MPS and CUDA verified
 20-frame pretrained checkpoint downloaded
 Checkpoint integrity verified
 Exact checkpoint-compatible model architecture recreated
 Checkpoint loaded successfully with strict=True
 Model forward pass verified on MPS
 Test video added
 OpenCV installed
 Test video successfully read with OpenCV
 Integrated lightweight 2D DCT frequency analysis module
 Verified forward pass of the combined dual-branch architecture
 Integrated MediaPipe BlazeFace face detection & cropping pipeline (resolving dlib compilation blocker)
 Built end-to-end inference pipeline (`src/pipeline.py`) and CLI (`predict.py`)

Current Remaining Work
 Model evaluation benchmark experiments on labeled validation/test dataset to populate the Feature E comparison table.

Project Structure

FinShield-Model2/
│
├── checkpoints/
│   └── README.md
│
├── docs/
│   └── FinShield_Model2_Progress_Document.docx
│
├── models/
│   └── blaze_face_short_range.tflite
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── frequency_feature.py
│   ├── model2_reference.py
│   ├── pipeline.py
│   └── video_preprocessing.py
│
├── test_videos/
│   └── README.md
│
├── .gitignore
├── README.md
├── evaluate.py
├── predict.py
├── requirements.txt
├── run_test.py
├── test_end_to_end.py
├── test_fusion_paths.py
├── test_preprocessing.py
└── verify_checkpoint.py

Model

The current pretrained model is:
model_87_acc_20_frames_final_data.pt

It expects:
20 frames per video
RGB face crops
112 × 112 resolution (or 224 × 224 for testing)
ResNeXt-50 32×4d feature extraction
2048-dimensional features
LSTM temporal modeling
Parallel Lightweight Frequency Feature extraction (DCT-based)
Output: 2-class logits (Real / Fake) and Frequency Anomaly Score

LSTM Configuration
input_size  = 2048
hidden_size = 2048
num_layers  = 1
batch_first = True
bias        = False
bidirectional = False

Classifier
ReLU
↓
Dropout(0.4)
↓
Linear(2048 → 2)
---

## End-to-End Inference Pipeline

The pipeline connects raw video decoding, face detection/cropping via BlazeFace, feature extraction, temporal LSTM modeling, lightweight frequency analysis, and diagnostic confidence estimation.

### 1. Command-Line Interface (`predict.py`)

Run deepfake detection directly on any video file:

```bash
# Basic inference (auto-loads default checkpoint if available)
python predict.py path/to/video.mp4

# Inference specifying custom checkpoint and hardware acceleration
python predict.py path/to/video.mp4 --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt --device cuda

# Machine-readable JSON output (for API/backend integrations)
python predict.py path/to/video.mp4 --json
```

Example formatted terminal output:
```text
================================================================
        FinShield AI — Model 2: Deepfake Detection Result
================================================================
Video File:        data/sample.mp4
Checkpoint:        checkpoints/model_87_acc_20_frames_final_data.pt
Inference Device:  cpu
----------------------------------------------------------------
PREDICTION:        [FAKE] (FAKE)
CONFIDENCE:        94.20%
FAKE PROBABILITY:  0.9420
----------------------------------------------------------------
DIAGNOSTIC SCORES:
  • Temporal Inconsistency:  0.8120 (frame feature jitter)
  • Frequency Anomaly Score: 0.7630 (DCT energy ratio)
----------------------------------------------------------------
FACE PREPROCESSING DIAGNOSTICS:
  • Frames Sampled:          20
  • Faces Detected:          19 / 20
  • Fallback Crops Used:     1 / 20
================================================================
```

### 2. Python API

```python
from src import predict_video

# Run full pipeline directly from video file
result = predict_video(
    video_path="path/to/video.mp4",
    checkpoint_path="checkpoints/model_87_acc_20_frames_final_data.pt",
    device="cpu",  # or "cuda" / "mps"
    num_frames=20,
)
print(result)
```

---

## Feature D - Confidence and Diagnostic Output

`src/model2_reference.py` exposes `predict_with_diagnostics(model, frames)` for tensor-level reporting. The input `frames` tensor should already be preprocessed for the checkpoint and shaped as `(20, 3, 112, 112)` or batched as `(batch, 20, 3, 112, 112)`.

Example:

```python
from src.model2_reference import Model, predict_with_diagnostics

model = Model(num_classes=2, use_frequency_feature=True)
# Load checkpoint weights before inference.
result = predict_with_diagnostics(model, frames)
print(result)
```

Output format:

```json
{
  "prediction": "FAKE",
  "confidence": 0.94,
  "probability": 0.94,
  "temporal_score": 0.81,
  "frequency_score": 0.76
}
```

Implementation notes:

- `prediction` is reported as `REAL` or `FAKE` using label encoding `0=REAL`, `1=FAKE`.
- `confidence` is the probability of the selected final class.
- `probability` is the final FAKE probability after enabled diagnostic adjustments.
- `temporal_score` measures frame-to-frame inconsistency using cosine distance across ResNeXt sequence features.
- `frequency_score` reuses the lightweight DCT frequency module and normalizes its energy statistic to a stable 0-1 diagnostic range.

---

## Feature E - Evaluation & Benchmarking

The evaluation tool in `evaluate.py` benchmarks and compares the 3 model variants on labeled datasets using `compare_model_variants()`:
1. **Original ResNeXt-50 + LSTM**: Baseline checkpoint classification logits.
2. **Modified + Temporal Score**: Probability blended with frame-to-frame temporal inconsistency.
3. **Modified + Temporal + Frequency**: Probability blended with temporal and frequency-domain DCT scores.

### Running Evaluation (`evaluate.py`)

```bash
# Evaluate a directory containing REAL/ and FAKE/ video folders:
python evaluate.py --data-dir path/to/dataset/

# Evaluate using a CSV manifest (video_path, label):
python evaluate.py --csv manifest.csv

# Run pipeline validation on synthetic dataset without downloading external videos:
python evaluate.py --demo
```

### Submission Comparison Table

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| Original ResNeXt-50 + LSTM | Fill after running experiments | Fill after running experiments | Fill after running experiments | Fill after running experiments | Baseline checkpoint logits only |
| Modified + Temporal Score | Fill after running experiments | Fill after running experiments | Fill after running experiments | Fill after running experiments | Baseline probability blended with temporal inconsistency score |
| Modified + Temporal + Frequency | Fill after running experiments | Fill after running experiments | Fill after running experiments | Fill after running experiments | Baseline probability blended with temporal and frequency-domain scores |

> [!NOTE]
> To populate the table above with final numbers for your submission, run `python evaluate.py --data-dir <path_to_test_dataset> --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt`.

