# FinShield AI — Robust Multimodal KYC Fraud Detection

A multimodal KYC fraud detection system that fuses **document forgery detection**, **video deepfake analysis**, **semantic cross-validation**, and **biometric face matching** through a reliability-weighted noisy-OR fusion engine.

---

## System Architecture

```text
                        ┌─────────────────┐     ┌──────────────────┐
                        │  Document Image  │     │   Video Selfie   │
                        └────────┬────────┘     └────────┬─────────┘
                                 │                       │
              ┌──────────────────┼───────────┐           │
              ▼                  ▼           ▼           ▼
     ┌────────────────┐  ┌────────────┐  ┌────────┐  ┌──────────────────┐
     │  Model 1:      │  │    OCR     │  │  Face  │  │  Model 2:        │
     │  Document      │  │  EasyOCR   │  │ Match  │  │  Video Deepfake  │
     │  Forgery       │  │  + Regex   │  │ YuNet  │  │  ResNeXt-50      │
     │  Dual-Branch   │  │            │  │ SFace  │  │  + LSTM          │
     │  CNN           │  │            │  │        │  │  + Gated Rich    │
     └───────┬────────┘  └─────┬──────┘  └───┬────┘  └────────┬─────────┘
             │                 │             │                 │
             │           ┌─────▼──────┐      │                 │
             │           │  Semantic  │      │                 │
             │           │ Validation │      │                 │
             │           └─────┬──────┘      │                 │
             │                 │             │                 │
             ▼                 ▼             ▼                 ▼
        ┌──────────────────────────────────────────────────────────┐
        │          Reliability-Weighted Noisy-OR Fusion            │
        │   p' = 0.5 + r·(p - 0.5)  →  risk = 1 - Π(1 - p')     │
        └──────────────────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
               APPROVE        REVIEW         REJECT
              (< 0.35)     (0.35–0.65)      (≥ 0.65)
```

---

## Two Primary Models

### Model 1: Document Forgery Detection (DualBranchForgeryDetector)

Dual-branch CNN that analyzes identity documents in both spatial and frequency domains.

| Component | Details |
|---|---|
| Spatial Branch | EfficientNet-B0 (pretrained ImageNet) → 1,280-d features |
| Frequency Branch | Per-channel 2D DCT → 3-layer CNN (32→64→128) → 128-d features |
| Fusion Head | Concat 1,408-d → FC + Dropout(0.35) + ReLU → 2-class output |
| Loss | Focal Loss (α=0.25, γ=2.0) |
| Input | 224×224 RGB document image |
| Training | Two-phase: frozen backbone (5 epochs, LR=1e-3) → full fine-tune (up to 50 epochs, differential LR) |
| Dataset | 8,000+ Indian identity document images (Aadhaar + PAN) with 7 tampering categories |
| **Test Accuracy** | **97.92%** |
| **AUC-ROC** | **0.9982** |

**Sub-components of document verification:**
- **OCR** (EasyOCR): Extracts name, DOB, ID number with regex patterns for Aadhaar/PAN
- **Semantic Validation**: Cross-references OCR text against user form data (name fuzzy match, DOB check, ID format + Verhoeff checksum)
- **Face Matching** (YuNet + SFace ONNX): Cosine similarity between document face and video selfie (threshold: 0.363)

### Model 2: Video Deepfake Detection

CNN+LSTM temporal model with gated rich attention for deepfake video detection.

| Component | Details |
|---|---|
| Frame Encoder | ResNeXt-50 32×4d (pretrained ImageNet) |
| Face Detection | MediaPipe BlazeFace → 112×112 face crops |
| Temporal Model | LSTM (hidden=2048, 1 layer) over 32 frames |
| Classification | Gated Rich Attention Head with learnable fusion gate |
| Temperature | T=1.3122 (calibrated on validation set) |
| Inference | Threshold=0.575, TTA horizontal flip |
| Dataset | 2,000+ video sequences |
| **Accuracy** | **~80%** |

---

## Fusion Engine

The fusion engine combines all four modality scores into a single risk assessment:

1. **Calibration**: `p' = 0.5 + r·(p - 0.5)` — reliability weight shrinks uncertain predictions toward 0.5
2. **Noisy-OR**: `risk = 1 - Π(1 - p'_m)` — any high-risk modality elevates overall risk
3. **Decision**: APPROVE (< 0.35) · REVIEW (0.35–0.65) · REJECT (≥ 0.65)
4. **Hard Vetoes**: Face mismatch → REJECT · Insufficient evidence → REVIEW

---

## Adversarial Robustness

Evaluated under FGSM attacks on the Document Forgery model:

| Model | Clean | ε=0.01 | ε=0.05 | ε=0.10 |
|---|---:|---:|---:|---:|
| Standard | 97.92% | 43.06% | 43.06% | 43.06% |
| Adversarial Fine-Tuned | 90.28% | 86.11% | 77.08% | 71.53% |

---

## Explainability

- **Grad-CAM / Grad-CAM++**: Spatial heatmaps highlighting forgery regions
- **Integrated Gradients**: Pixel-level attribution scores
- **Branch Ablation**: Spatial-only 43.06% → Frequency-only 84.03% → Dual-branch 85.42%
- **Fusion Weights**: Per-modality attention weights displayed in the UI

---

## Project Structure

```
FinShield-Model2/
├── src/                          # Model 2: Deepfake detection
│   ├── model2_reference.py       #   ResNeXt-50 + LSTM + Gated Rich Head
│   ├── video_preprocessing.py    #   BlazeFace face detection pipeline
│   ├── frequency_feature.py      #   DCT frequency analysis
│   ├── pipeline.py               #   Standalone inference pipeline
│   └── dataset.py                #   Dataset utilities
│
├── fusion/                       # Multimodal fusion system
│   ├── app.py                    #   Flask web application
│   ├── pipeline.py               #   KYC orchestration (run_kyc)
│   ├── engine.py                 #   Noisy-OR fusion engine
│   ├── adapters.py               #   Model 1 & 2 prediction adapters
│   ├── config.py                 #   Paths, thresholds, parameters
│   ├── ocr.py                    #   EasyOCR text extraction
│   ├── semantic.py               #   Semantic cross-validation
│   ├── face_match.py             #   YuNet + SFace face matching
│   ├── schemas.py                #   Data classes for all results
│   ├── history.py                #   Verification history storage
│   └── frontend/
│       ├── templates/index.html  #   Main UI template
│       └── static/
│           ├── css/style.css     #   Dark navy + cream theme
│           └── js/app.js         #   3D face scan, form handling, results
│
├── models/                       # ONNX models (not in Git)
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── blaze_face_short_range.tflite
│
├── checkpoints/                  # Model weights (not in Git)
├── predict.py                    # Model 2 standalone CLI
├── evaluate.py                   # Benchmark evaluation
├── train.py                      # Spec fusion training
├── train_advanced.py             # Gated rich head training
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- Model checkpoints: `best_model.pt` (Model 1), `gated_rich.pt` (Model 2)
- ONNX models: `face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx`

### Installation

```bash
pip install -r requirements.txt
pip install -r fusion/requirements-fusion.txt
```

### Run the Web Application

```bash
# Full mode (requires model checkpoints + ONNX models)
python -m fusion.app

# Demo mode (mock predictions, no models needed)
FINSHIELD_DEMO_MODE=1 python -m fusion.app
```

Then open **http://localhost:5000**

### Standalone Deepfake Prediction (Model 2 only)

```bash
python predict.py path/to/video.mp4 --checkpoint checkpoints/gated_rich.pt --json
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Deep Learning | PyTorch, EfficientNet-B0, ResNeXt-50, LSTM |
| Computer Vision | OpenCV, Pillow, MediaPipe, EasyOCR |
| Face Detection/Recognition | YuNet (ONNX), SFace (ONNX) |
| Adversarial Testing | FGSM, PGD |
| Explainability | Grad-CAM, Grad-CAM++, Integrated Gradients |
| Backend | Flask, Flask-CORS |
| Frontend | HTML/CSS/JS, Three.js |

---

## Team

- Naman Agarwal (16010123209)
- Neelutpal Dutta (16010123211)
- Neeraj Apraj (16010123212)
- Shloka Pol (16010123316)

**Guide:** Prof. Supriya Agrawal
K. J. Somaiya College of Engineering — Department of Computer Engineering
