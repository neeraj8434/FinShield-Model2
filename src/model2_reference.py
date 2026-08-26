Perfect. Paste this entire README into the blank `README.md`:

````markdown
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
````

---

## Current Status

### Completed

* [x] Reference deepfake detection repository cloned
* [x] Repository Model Creation notebooks inspected
* [x] Python virtual environment created
* [x] Jupyter installed
* [x] PyTorch 2.8.0 installed
* [x] TorchVision 0.23.0 installed
* [x] Apple Silicon MPS verified
* [x] 20-frame pretrained checkpoint downloaded
* [x] Checkpoint integrity verified
* [x] Exact checkpoint-compatible model architecture recreated
* [x] Checkpoint loaded successfully with `strict=True`
* [x] Model forward pass verified on MPS
* [x] Test video added
* [x] OpenCV installed
* [x] Test video successfully read with OpenCV

### Current Blocker

Face preprocessing is the only incomplete part of the pipeline.

The original repository uses `face-recognition`/`dlib`, but `dlib` failed to compile on the current Mac environment.

We will therefore use a Mac-compatible face detector rather than modifying the working model/checkpoint.

---

## Project Structure

```text
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
```

> The pretrained model checkpoint and video datasets are intentionally excluded from Git.

---

## Model

The current pretrained model is:

```text
model_87_acc_20_frames_final_data.pt
```

It expects:

* 20 frames per video
* RGB face crops
* 112 × 112 resolution
* ResNeXt-50 32×4d feature extraction
* 2048-dimensional features
* LSTM temporal modeling
* 2-class output: Real / Fake

### LSTM Configuration

```text
input_size  = 2048
hidden_size = 2048
num_layers  = 1
batch_first = True
bias        = False
bidirectional = False
```

### Classifier

```text
ReLU
↓
Dropout(0.4)
↓
Linear(2048 → 2)
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/neeraj8434/FinShield-Model2.git
cd FinShield-Model2
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Hardware

The current development environment uses an Apple Silicon Mac.

PyTorch MPS was successfully detected:

```text
PyTorch: 2.8.0
MPS available: True
```

For NVIDIA GPU systems, the implementation may be adapted to CUDA.

---

## Pretrained Checkpoint

The pretrained checkpoint is **not stored in this Git repository** because it is approximately 216 MB.

Place it locally at:

```text
checkpoints/model_87_acc_20_frames_final_data.pt
```

The checkpoint was obtained from the trained-model link provided by the original reference repository.

Do not commit `.pt` or `.pth` files to this repository.

---

## Verification Performed

### Checkpoint verification

The checkpoint successfully loaded into the recreated model:

```text
CHECKPOINT MATCHED SUCCESSFULLY
<All keys matched successfully>
```

### Forward-pass verification

A dummy input was successfully processed:

```text
Device: mps

Input:
torch.Size([1, 20, 3, 112, 112])

Output:
torch.Size([1, 2])
```

This confirms that the ResNeXt + LSTM model can execute successfully on the development machine.

---

## Test Video

A local test video was used during development.

Video properties:

```text
Frames:      1044
FPS:         24
Resolution:  1920 × 1080
Duration:    ~43.5 seconds
Size:        ~6.4 MB
```

Test videos are excluded from Git and should not be committed to the repository.

---

## Development Workflow

The intended development pipeline is:

```text
Video
  ↓
Face Detection
  ↓
Face Cropping
  ↓
Frame Sampling
  ↓
20 × 112×112 Face Frames
  ↓
ResNeXt-50
  ↓
2048-D Features
  ↓
LSTM
  ↓
Real/Fake Probability
```

---

## Roadmap

### Phase 1 — Pipeline

* [x] Clone reference repository
* [x] Verify pretrained checkpoint
* [x] Recreate model architecture
* [x] Verify MPS inference
* [ ] Implement Mac-compatible face detection
* [ ] Extract 20 face frames
* [ ] Run first real-video prediction

### Phase 2 — Evaluation

* [ ] Test multiple real videos
* [ ] Test multiple deepfake videos
* [ ] Build independent test set
* [ ] Calculate accuracy
* [ ] Calculate precision
* [ ] Calculate recall
* [ ] Calculate F1-score
* [ ] Calculate ROC-AUC
* [ ] Generate confusion matrix

### Phase 3 — FinShield Model 2

* [ ] Clean preprocessing pipeline
* [ ] Create reusable inference module
* [ ] Create structured prediction output
* [ ] Add confidence score
* [ ] Add temporal sampling across the full video
* [ ] Add explainability/visualization
* [ ] Package Model 2 for integration

### Phase 4 — FinShield Integration

```text
                KYC SUBMISSION
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
         DOCUMENT              VIDEO
             │                   │
             ▼                   ▼
          MODEL 1              MODEL 2
             │                   │
             ▼                   ▼
     Document Fraud Score   Deepfake Score
             │                   │
             └─────────┬─────────┘
                       ▼
                    MODEL 3
               Risk-Aware Fusion
                       │
                       ▼
                 FINAL KYC RISK
```

---

## Important Notes

### Reference implementation

This project uses the original repository as a reference for the ResNeXt + LSTM architecture and pretrained checkpoint.

Reference repository:

```text
https://github.com/abhijithjadhav/Deepfake_detection_using_deep_learning
```

The original repository should be treated as third-party reference material.

### Results

The accuracy reported by the original repository must **not** be presented as FinShield's own model performance.

FinShield will perform its own evaluation and report independently reproduced results.

### Data

Datasets and pretrained models must be used according to their applicable licenses and terms.

Large datasets, test videos, and model checkpoints should remain outside this Git repository.

---

## Team Contribution Guidelines

### Before starting work

Pull the latest changes:

```bash
git pull
```

### Create a branch

Use a descriptive branch name:

```bash
git checkout -b feature/face-detection
```

Examples:

```text
feature/face-detection
feature/video-preprocessing
feature/model-inference
feature/explainability
feature/evaluation
fix/checkpoint-loading
```

### Commit changes

```bash
git add .
git commit -m "Add face detection preprocessing"
```

### Push your branch

```bash
git push -u origin feature/face-detection
```

Then open a Pull Request on GitHub.

### Do not commit

```text
.venv/
*.pt
*.pth
*.mp4
datasets/
data/
.env
.DS_Store
```

---

## Current Team Task

**Next immediate task:**

Implement a Mac-compatible face detection and preprocessing pipeline that converts:

```text
test.mp4
```

into:

```text
20 × 112 × 112 RGB face frames
```

These frames will then be passed into the verified ResNeXt + LSTM checkpoint.

---

## Project Status

**Model architecture:** ✅ Verified
**Pretrained checkpoint:** ✅ Verified
**MPS inference:** ✅ Verified
**Video loading:** ✅ Verified
**Face preprocessing:** 🔨 In progress
**Real video inference:** ⏳ Pending
**Independent evaluation:** ⏳ Pending
**Model 1 + Model 2 integration:** ⏳ Pending
**Model 3 fusion:** ⏳ Pending

```

Then press **⌘ + S** and close TextEdit.

**Don't commit or push yet.** After saving, tell me `saved`, and we'll do the next step: create `requirements.txt` so your friends can reproduce the same environment.
```
