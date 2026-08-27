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
 Apple Silicon MPS verified
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
Current Blocker
Face preprocessing is the only incomplete part of the pipeline.
The original repository uses face-recognition/dlib, but dlib failed to compile on the current Mac environment.
We will therefore use a Mac-compatible face detector rather than modifying the working model/checkpoint.

Project Structure

FinShield-Model2/
│
├── checkpoints/
│   └── README.md
│
├── docs/
│   └── FinShield_Model2_Progress_Document.docx
│
├── src/
│   ├── model2_reference.py
│   └── frequency_feature.py
│
├── test_videos/
│   └── README.md
│
├── reference-repo/
│   └── Original reference repository
│
├── .gitignore
├── README.md
├── requirements.txt
└── run_test.py

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

## Feature D - Confidence and Diagnostic Output

`src/model2_reference.py` exposes `predict_with_diagnostics(model, frames)` for inference-time reporting. The input `frames` tensor should already be preprocessed for the checkpoint and shaped as `(20, 3, 112, 112)` or batched as `(batch, 20, 3, 112, 112)`.

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

## Feature E - Evaluation

`src/model2_reference.py` also exposes `evaluate_model(...)` and `compare_model_variants(...)` for labeled validation/test data. Labels must be encoded as `0=REAL` and `1=FAKE`.

Example:

```python
from src.model2_reference import compare_model_variants

results = compare_model_variants(model, dataloader)
for model_name, metrics in results.items():
    print(model_name, metrics)
```

Each evaluation result includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Submission comparison table:

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| Original ResNeXt-50 + LSTM | Fill after running experiments | Fill after running experiments | Fill after running experiments | Fill after running experiments | Baseline checkpoint logits only |
| Modified + Temporal Score | Fill after running experiments | Fill after running experiments | Fill after running experiments | Fill after running experiments | Baseline probability blended with temporal inconsistency score |
| Modified + Temporal + Frequency | Fill after running experiments | Fill after running experiments | Fill after running experiments | Fill after running experiments | Baseline probability blended with temporal and frequency-domain scores |

Actual metric values should be filled after running `compare_model_variants(...)` on the labeled evaluation set. The current repository does not include a labeled dataloader or test dataset, so the code provides the evaluation path without inventing unsupported results.
