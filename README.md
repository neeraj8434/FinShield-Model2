# FinShield AI — Model 2: Deepfake Detection

Video-based deepfake detection module for the **FinShield AI** project.

Model 2 analyzes video by extracting facial frames, generating spatial features using **ResNeXt-50 32×4d**, modeling temporal sequence dynamics using an **LSTM**, extracting lightweight frequency-domain features via 2D DCT, and producing a Real/Fake prediction.

By default, the pipeline uses **post-hoc score blending** between the baseline LSTM logits, the normalized temporal inconsistency score, and the normalized frequency anomaly score. When a trained fusion checkpoint is supplied, feature-level fusion via a lightweight MLP (`classify_fused_features`) is used instead.

---

## Architecture

```text
                    INPUT VIDEO
                         │
                         ▼
                  Frame Extraction
                         │
                         ▼
              Face Detection (BlazeFace)
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
        ▼                                 ▼
       LSTM                      Normalized Score [0, 1]
 Hidden Size 2048                         │
        │                                 │
        ▼                                 │
  Linear (2048 → 2)                       │
 (Baseline Logits)                        │
        │                                 │
        ├─────────────────────────────────┘
        │
   [Inference Mode]
   ├── Baseline + Blend: Post-hoc weighted probability adjustment
   └── Trained Fusion:   MLP head (LSTM features + Temporal MAD + DCT Frequency)
```

---

## Face Detection Model Setup

The face detection step uses the **MediaPipe Tasks Face Detector** (BlazeFace short-range TFLite model):

- **Expected Path**: `models/blaze_face_short_range.tflite`
- **Model Description**: Google MediaPipe BlazeFace Short-Range model (designed for faces within ~2 meters / selfie-to-medium shots).
- **Custom / Full-Range Models**: To use a full-range face model (e.g. for medium/wide shots), pass `--face-model /path/to/blaze_face_full_range.tflite` to `predict.py` or `preprocess_video(face_model_path=...)`.
- *Note*: Models must not be downloaded automatically; please place your `.tflite` model at `models/blaze_face_short_range.tflite`.

---

## Pretrained Baseline Checkpoint

The pretrained baseline checkpoint is:
`checkpoints/model_87_acc_20_frames_final_data.pt`

- Backbone: ResNeXt-50 32×4d
- Sequence model: 1-layer LSTM (input_size=2048, hidden_size=2048)
- Input: 20 RGB face crops at 112×112 resolution, normalized with ImageNet mean/std
- Output: 2 classes (0: REAL, 1: FAKE)

---

## End-to-End Inference Pipeline

The pipeline connects video frame extraction, face detection & cropping with sanity checks, spatial-temporal modeling, frequency analysis, and diagnostic scoring into a single API.

### 1. Command-Line Interface (`predict.py`)

Run deepfake detection directly on any video file:

```bash
# Basic inference (auto-loads default checkpoint if available)
python predict.py path/to/video.mp4

# Inference using custom checkpoint and specific compute device
python predict.py path/to/video.mp4 --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt --device mps

# Inference using trained feature fusion head
python predict.py path/to/video.mp4 --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt --fusion-checkpoint checkpoints/model_fusion_v1.pt

# Debug mode: save per-frame overlays of detected/rejected face boxes and crops
python predict.py path/to/video.mp4 --debug-dir ./debug_output/

# Machine-readable JSON output
python predict.py path/to/video.mp4 --json
```

### Actual Terminal Output (Real Run on `test_videos/01_02__talking_against_wall__YVGY8LOK.mp4`):
```text
================================================================
        FinShield AI — Model 2: Deepfake Detection Result
================================================================
Video File:        /Users/neerajapraj/Desktop/FinShield-Model2/test_videos/01_02__talking_against_wall__YVGY8LOK.mp4
Checkpoint:        /Users/neerajapraj/Desktop/FinShield-Model2/checkpoints/model_87_acc_20_frames_final_data.pt
Inference Device:  mps
Mode:              baseline+blend
----------------------------------------------------------------
PREDICTION:        [FAKE] (FAKE)
CONFIDENCE:        75.20%
FAKE PROBABILITY:  0.7520
----------------------------------------------------------------
DIAGNOSTIC SCORES:
  • Temporal Score (MAD):    0.2062 (frame feature jitter)
  • Temporal Cosine Score:   0.0285 (cosine distance)
  • Frequency Anomaly Score: 0.0402 (DCT energy ratio)
----------------------------------------------------------------
FACE PREPROCESSING DIAGNOSTICS:
  • Frames Sampled:          20
  • Faces Detected:          20 / 20
  • Fallback Crops Used:     0 / 20
================================================================
```

*(Note: The terminal block above reflects an actual run on the repository's test video. Predictions on external videos depend on video content and model training data).*

### 2. Python API

```python
from src import predict_video

# Run full pipeline directly from video file
result = predict_video(
    video_path="test_videos/test.mp4",
    checkpoint_path="checkpoints/model_87_acc_20_frames_final_data.pt",
    device="cpu",  # or "cuda" / "mps"
    num_frames=20,
)
print(result)
```

Format of result dictionary:
```json
{
  "video_path": "/path/to/video.mp4",
  "prediction": "FAKE",
  "confidence": 0.7729,
  "probability": 0.7729,
  "temporal_score": 0.0546,
  "temporal_cosine_score": 0.0122,
  "frequency_score": 0.0711,
  "mode": "baseline+blend",
  "preprocessing": {
    "frames_processed": 20,
    "faces_detected": 6,
    "fallbacks_used": 14
  }
}
```

---

## Training the Fusion Head (`train_fusion.py`)

`train_fusion.py` freezes the pretrained ResNeXt-50 backbone and LSTM, caches extracted features (`lstm_out`, `temporal_score`, `frequency_score`, `label`) to disk once per video, and trains the lightweight fusion MLP head (`fusion_fc1` and `fusion_fc2`). Only the fusion head parameters and training metadata are saved.

### Usage Commands:

```bash
# Train using a directory containing REAL/ and FAKE/ subdirectories:
python train_fusion.py \
    --data-dir path/to/dataset/ \
    --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt \
    --out checkpoints/model_fusion_v1.pt \
    --epochs 30 \
    --lr 0.001 \
    --batch-size 32 \
    --val-split 0.2 \
    --seed 42

# Train using a CSV manifest (columns: video_path, label):
python train_fusion.py \
    --csv dataset_manifest.csv \
    --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt \
    --out checkpoints/model_fusion_v1.pt \
    --cache-dir .fusion_cache/
```

Outputs:
- Checkpoint: `checkpoints/model_fusion_v1.pt` (contains only `fusion_fc1.*` and `fusion_fc2.*` weights plus training metadata)
- Metrics JSON: `checkpoints/model_fusion_v1_metrics.json` (confusion matrix, precision, recall, F1, accuracy)

---

## Evaluation & Benchmarking (`evaluate.py`)

`evaluate.py` benchmarks model variants on labeled validation/test data and produces the comparison table:
1. **Original ResNeXt-50 + LSTM**: Baseline checkpoint classification logits.
2. **Modified + Temporal Score**: Probability post-hoc blended with normalized temporal inconsistency (MAD).
3. **Modified + Temporal + Frequency**: Probability post-hoc blended with temporal and DCT frequency scores.
4. **Fusion head (trained)**: Evaluated when `--fusion-checkpoint` is supplied.

### Usage Commands:

```bash
# Evaluate baseline variants on video directory (REAL/ and FAKE/ folders):
python evaluate.py \
    --data-dir path/to/dataset/ \
    --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt \
    --output evaluation_results.md

# Evaluate all 4 variants including trained fusion head:
python evaluate.py \
    --data-dir path/to/dataset/ \
    --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt \
    --fusion-checkpoint checkpoints/model_fusion_v1.pt \
    --output evaluation_results.md

# Evaluate using CSV manifest:
python evaluate.py \
    --csv manifest.csv \
    --checkpoint checkpoints/model_87_acc_20_frames_final_data.pt \
    --output evaluation_results.md

# Validate evaluation logic using synthetic data without external video files:
python evaluate.py --demo --demo-samples 10
```

When `--output` is specified, the markdown table is saved to the given path and all confusion matrices are exported as JSON alongside it (e.g. `evaluation_results_confusion.json`). Unevaluated variants are marked as `Not run`.

---

## Submission Comparison Table Format

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| Original ResNeXt-50 + LSTM | *Run on dataset* | *Run on dataset* | *Run on dataset* | *Run on dataset* | Baseline checkpoint logits only |
| Modified + Temporal Score | *Run on dataset* | *Run on dataset* | *Run on dataset* | *Run on dataset* | Baseline probability blended with temporal inconsistency score (MAD, normalised) |
| Modified + Temporal + Frequency | *Run on dataset* | *Run on dataset* | *Run on dataset* | *Run on dataset* | Baseline probability blended with temporal and frequency-domain scores |
| Fusion head (trained) | *Run on dataset* | *Run on dataset* | *Run on dataset* | *Run on dataset* | Trained fusion MLP (LSTM features + temporal + frequency) |

*(Note: Table cells should be populated by running `evaluate.py` on your labeled dataset; numbers are never hand-fabricated).*
