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