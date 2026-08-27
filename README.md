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
                         ▼
                 ResNeXt-50 32×4d
                         │
                         ▼
             2048-D Feature / Frame
                         │
                         ▼
                       LSTM
                  Hidden Size 2048
                         │
                         ▼
                      ReLU
                         │
                         ▼
                   Dropout 0.4
                         │
                         ▼
                  Linear Layer
                         │
                         ▼
                    REAL / FAKE

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
│   └── model2_reference.py
│
├── test_videos/
│   └── README.md
│
├── reference-repo/
│   └── Original reference repository
│
├── .gitignore
├── README.md
└── requirements.txt

Model

The current pretrained model is:
model_87_acc_20_frames_final_data.pt

It expects:
20 frames per video
RGB face crops
112 × 112 resolution
ResNeXt-50 32×4d feature extraction
2048-dimensional features
LSTM temporal modeling
2-class output: Real / Fake

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

## Feature D — Confidence and Diagnostic Output

`src/model2_reference.py` now includes `predict_with_diagnostics(model, frames)`. It returns the final label, confidence, temporal inconsistency score, and frequency-domain score.

Example:

```python
from src.model2_reference import Model, predict_with_diagnostics

model = Model(num_classes=2)
# load checkpoint here, then pass a preprocessed tensor shaped (20, 3, 112, 112)
result = predict_with_diagnostics(model, frames)
print(result)
```

Example output shape:

```json
{
  "prediction": "FAKE",
  "confidence": 0.94,
  "temporal_score": 0.81,
  "frequency_score": 0.76,
  "fake_probability": 0.94
}
```

Diagnostic behavior:

- `temporal_score` measures frame-to-frame feature instability using cosine distance over the extracted ResNeXt feature sequence.
- `frequency_score` measures high-frequency energy from the frame FFT spectrum.
- The baseline model output is preserved; diagnostic-adjusted variants can be enabled through the prediction/evaluation flags.

## Feature E — Evaluation

`src/model2_reference.py` now includes `evaluate_model(...)` and `compare_model_variants(...)` for labeled datasets where labels are encoded as `0=REAL` and `1=FAKE`.

```python
from src.model2_reference import compare_model_variants

results = compare_model_variants(model, dataloader)
for model_name, metrics in results.items():
    print(model_name, metrics)
```

The evaluator reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Comparison table for submission:

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| Original ResNeXt-50 + LSTM | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Baseline checkpoint logits only |
| Modified + Temporal Score | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Blends baseline fake probability with temporal inconsistency score |
| Modified + Temporal + Frequency | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Run `compare_model_variants` with labeled data | Blends baseline fake probability with temporal and frequency scores |

Actual metric values are not filled in this repository because no labeled evaluation dataset or dataloader is included in the current project files.
