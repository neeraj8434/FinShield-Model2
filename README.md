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

Current Status
 The specification fusion model, advanced gated variant, full held-out benchmark, calibrated inference, and three-input CLI demo have been run. The final audit and detailed results are in `../outputs/FinShield_implementation_audit.md` and `../outputs/benchmark_full_test.md`. Dataset expansion and cross-dataset evaluation still require separate licensed data and access approval.

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
- `temporal_score` is the mean absolute difference between adjacent ResNeXt frame embeddings. It is a diagnostic and is not a standalone manipulation threshold.
- `frequency_score` reuses the lightweight DCT frequency module and normalizes its energy statistic to a stable 0-1 diagnostic range.

---

## Feature E - Evaluation & Benchmarking

The evaluation tool in `evaluate.py` compares the baseline, inference-time score blends, and a trained head in one shared feature pass:
1. **Original ResNeXt-50 + LSTM**: Baseline checkpoint classification logits.
2. **Modified + Temporal Score**: Probability blended with frame-to-frame temporal inconsistency.
3. **Modified + Temporal + Frequency**: Probability blended with temporal and frequency-domain DCT scores.

### Running Evaluation (`evaluate.py`)

```bash
# Evaluate a directory containing REAL/ and FAKE/ video folders:
python evaluate.py --data-dir path/to/dataset/

# Evaluate using a CSV manifest (video_path, label, split):
python evaluate.py --csv manifest.csv --split test --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt

# Evaluate one trained path without rerunning comparison variants:
python evaluate.py --csv manifest.csv --split val --variant spec_fusion --checkpoint checkpoints/spec_fusion.pt

# Run pipeline validation on synthetic dataset without downloading external videos:
python evaluate.py --demo
```

The full FF++ C23 held-out test comparison has 298 scorable videos out of 300; one REAL and one FAKE clip were rejected for insufficient face evidence. FAKE is the positive class, and macro F1 is the primary metric.

| Model | Accuracy | FAKE precision | FAKE recall | Macro F1 | Notes |
|---|---:|---:|---:|---:|---|
| Original ResNeXt-50 + LSTM | 0.2550 | 0.2517 | 0.2483 | 0.2550 | Pretrained checkpoint logits |
| Trained temporal-only head | 0.7584 | 0.7584 | 0.7584 | 0.7584 | Independently trained head |
| Trained DCT-only head | 0.7450 | 0.7450 | 0.7450 | 0.7450 | Independently trained head |
| Specification concatenation fusion | 0.7584 | 0.7584 | 0.7584 | 0.7584 | LSTM summary + temporal score + DCT score |
| Advanced gated temporal + frequency | 0.7685 | 0.7985 | 0.7181 | 0.7679 | Selected using validation macro F1 |

> [!NOTE]
The full machine-readable comparison (per-video results, probabilities, coverage, rejection reasons, timing, and checkpoint sizes) is in `../outputs/benchmark_full_test.json`; `../outputs/benchmark_full_test.md` includes calibration and transformation results. Score blending remains an inference-time ablation; `--variant` can select a trained classifier head.

## Specification fusion training

Inference and training require the original ResNeXt-50 + LSTM checkpoint;
the CLI refuses to present randomly initialized weights as a detector. Create a
CSV manifest with `video_path,label,split` columns (`label` is `REAL`/`FAKE` or
`0`/`1`; `split` is `train` or `val`) and pass a data root for relative paths.
Keep every source group in a single split and reserve the test set from tuning.

```bash
python -m pip install -r requirements.txt
python train.py --manifest path/to/manifest.csv --data-root path/to/videos \
  --init-checkpoint checkpoints/model_87_acc_20_frames_final_data.pt \
  --output checkpoints/spec_fusion.pt --epochs 5 --seed 42
python predict.py path/to/video.mp4 --checkpoint checkpoints/spec_fusion.pt --json
python evaluate.py --csv path/to/heldout_test.csv \
  --checkpoint checkpoints/spec_fusion.pt --output outputs/benchmark.json
```

Training freezes the ResNeXt backbone and LSTM, trains the concatenation fusion
head, caches preprocessed clips under `work/preprocessed_face_cache`, writes
epoch and validation confusion metrics to JSONL, and selects the checkpoint by
validation macro F1 (validation loss breaks ties). `predict.py` uses the fused head for a checkpoint marked
`spec_fusion`; a legacy state-dict checkpoint uses the baseline head. Optional
`--blend-temporal` and `--blend-frequency` flags are inference-time score
blending ablations and are separate from learned feature fusion. The temporal
statistic is mean absolute difference between adjacent ResNeXt embeddings. The
lightweight DCT feature computes mean absolute coefficients outside the
top-left 50% × 50% low-frequency block after grayscale conversion.

On Windows with Python 3.13 CPU, use the validated pins in
`requirements-python313-cpu.txt`; `requirements.txt` pins a NumPy version without
a Python 3.13 wheel. Example commands for the local FF++ manifest:

```powershell
python -m pip install -r requirements-python313-cpu.txt
python train.py --manifest ..\outputs\ffpp_c23_faceswap_manifest.csv --data-root C:\path\to\FaceForensics++_C23 --init-checkpoint checkpoints\model_87_acc_20_frames_final_data.pt --output checkpoints\spec_fusion.pt --epochs 5 --seed 42
python evaluate.py --csv ..\outputs\ffpp_c23_faceswap_manifest.csv --data-root C:\path\to\FaceForensics++_C23 --split test --checkpoint checkpoints\spec_fusion.pt --output ..\outputs\benchmark.json
```

Insufficient face evidence returns `status=insufficient_evidence` with frame
counts rather than a REAL/FAKE label. The local checkout audit, requirement
checklist, and remaining blockers are in `outputs/FinShield_implementation_audit.md`.



## Importing another approved dataset

Celeb-DF v2 is a candidate for cross-dataset evaluation. The authors provide access only after a request is accepted and the dataset is governed by its Terms to Use; check those terms before use. Its metadata CSV should use the official labels and test list, with original/fake source groups kept in one split. `prepare_manifest.py` validates safe relative paths, labels, split assignments, required provenance, group leakage, and exact duplicate hashes before writing the normalized manifest and audit.

```powershell
python prepare_manifest.py --input C:\data\Celeb-DF-v2\metadata.csv --data-root C:\data\Celeb-DF-v2 --dataset-name Celeb-DF-v2 --output ..\outputs\celebdf_v2_manifest.csv --audit-output ..\outputs\celebdf_v2_dataset_audit.json
```

The required input columns are `video_path,label,split,source_id,group_id,manipulation_type`; optional fields are `dataset,identity_id,license,source_url`. Do not create these labels or group IDs by guesswork. [Official dataset and access instructions](https://github.com/yuezunli/celeb-deepfakeforensics).

## Advanced experiments and final checkpoint

`train_advanced.py` implements richer temporal summaries, attention pooling,
three-band DCT features, and gated fusion. The full-data gated checkpoint
`../outputs/advanced/gated_rich.pt` was selected because it reached 0.7917
validation macro F1. It achieved 0.7679 macro F1 and 0.8583 AUROC on the fixed
test split. Validation-only temperature scaling is saved in the checkpoint.

```powershell
python train_advanced.py --variant gated_rich --manifest ..\outputs\ffpp_c23_faceswap_manifest.csv --data-root C:\path\to\FaceForensics++_C23 --init-checkpoint checkpoints\model_87_acc_20_frames_final_data.pt --output ..\outputs\advanced\gated_rich.pt --cache-dir ..\work\preprocessed_face_cache --epochs 5 --frames 20 --seed 42 --num-threads 1
python evaluate.py --csv ..\outputs\ffpp_c23_faceswap_manifest.csv --data-root C:\path\to\FaceForensics++_C23 --split test --checkpoint ..\outputs\advanced\gated_rich.pt --variant all --cache-dir ..\work\evaluation_face_cache --output ..\outputs\gated_rich_test.json
python predict.py path\to\video.mp4 --checkpoint ..\outputs\advanced\gated_rich.pt --device cpu --json --output ..\outputs\demo.json
```
