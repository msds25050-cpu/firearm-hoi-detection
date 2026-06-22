# 🔫 Firearm Carrier Identification — Deep Learning Project

A deep learning system for detecting and classifying human-firearm interactions (HOI) in images, using a multi-model pipeline that combines **YOLOv8 detection**, a **3-Stream ResNet-50 CNN**, a **DP-HOI transformer model**, and **BLIP/BLIP-2 scene captioning** for threat assessment.

---

## 📋 Project Overview

This project addresses **Human-Object Interaction (HOI)** detection in the context of firearms — specifically classifying whether a person is:

| Class | Description |
|-------|-------------|
| `holding` | Person actively holding a firearm |
| `aiming` | Person aiming a firearm (highest threat) |
| `carrying` | Person passively carrying/holstering a firearm |

The system is built on the **LFC (Localizing Firearm Carriers)** dataset, which contains 2,418 annotated images with bounding boxes for both persons and firearms.

---

## 📁 Project Structure

```
Project/
├── 01_CNN_Baseline_(4) (2).ipynb   # Main training & evaluation notebook
│                                    # Includes: CNN Baseline, DP-HOI, YOLO, BLIP-2
└── DL project.ipynb                 # Standalone inference & threat scoring demo
```

---

## 🧠 Models & Architecture

### 1. YOLOv8 — Detection Stage
- **Base model:** `yolov8s.pt` (pretrained on COCO, fine-tuned on LFC dataset)
- **Task:** Detect persons (class 0) and firearms (class 1) in input images
- **Training:** 50 epochs, 640px images, batch=32, early stopping (patience=10)
- **Output:** Bounding boxes (xyxy format) + confidence scores

### 2. 3-Stream ResNet-50 CNN — Classification Stage
- **Architecture:** Three parallel ResNet-50 backbones (ImageNet pretrained), each receiving a different crop:
  - **Person stream** — crops the detected person bounding box
  - **Object stream** — crops the detected firearm bounding box
  - **Union stream** — crops the union bounding box (person + firearm)
- **Feature fusion:** 2048-dim features from each stream are concatenated → 6144-dim vector
- **Classifier head:** `6144 → 1024 → 256 → 3 (classes)` with ReLU + Dropout(0.5)
- **Total parameters:** ~71M
- **Best val accuracy:** ~92.61%

### 3. DP-HOI — Transformer-Based HOI Model
- **Backbone:** DETR ResNet-50 (from `facebook/detr-resnet-50`, HuggingFace)
- **Visual feature projection:** `AdaptiveAvgPool2d(4×4) → Flatten → Linear(32768, 256) → LayerNorm → GELU`
- **Spatial head:** Encodes 14-dimensional geometric features (union/person/object boxes + aspect ratios) → 256-dim
- **Classifier:** Concatenated visual+spatial features (512-dim) → 256 → 3 classes
- **Best val accuracy:** ~91.44%

### 4. BLIP / BLIP-2 — Scene Captioning
- **DL project.ipynb:** Uses `Salesforce/blip-image-captioning-base` for single-image captioning
- **CNN Baseline notebook:** Uses `Salesforce/blip2-opt-2.7b` (BLIP-2) for VQA on the union crop
- **Purpose:** Generates natural language scene description to augment threat scoring with caption-level keywords (gun, weapon, aiming, pointing, etc.)

---

## 📊 Threat Scoring System (DL project.ipynb)

The `DL project.ipynb` notebook implements a rule-based threat reasoner:

```
Threat Score = 0.50 × action_score + 0.30 × YOLO_confidence + 0.20 × caption_score
```

| Action | Score |
|--------|-------|
| `carrying` | 0.30 |
| `holding` | 0.60 |
| `aiming` | 0.95 |
| unknown | 0.20 |

| Threat Level | Score Range |
|--------------|-------------|
| HIGH | ≥ 0.80 |
| MEDIUM | 0.50 – 0.79 |
| LOW | < 0.50 |

---

## ⚙️ Setup & Requirements

### Environment
- **Platform:** Google Colab (T4 GPU recommended)
- **Python:** 3.9+

### Installation
```bash
pip install transformers pillow torch torchvision
pip install ultralytics
pip install timm scikit-learn matplotlib seaborn
pip install transformers accelerate  # for BLIP-2
```

### Dataset
- **LFC Dataset** — stored on Google Drive
- Path: `Imaan Deep Learning/SHARE TO IMAAN/LFC-LocalizingFirearmCarriers-Dataset/`
- Annotations: `Copy of labeled_interactions_augmented.json`
- Augmented carrying images: `augmented_carrying_images/`

### Data Splits
| Split | Fraction |
|-------|----------|
| Train | 70% |
| Val | 15% |
| Test | 15% |

---

## 🏋️ Training Configuration

| Hyperparameter | CNN Baseline | DP-HOI |
|----------------|-------------|--------|
| Batch size | 64 | (variable) |
| Epochs | 60 | 60 |
| Base LR | 1e-4 | 1e-4 |
| Backbone LR | 3e-5 | 1e-5 |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Loss | CrossEntropy (class-weighted) | CrossEntropy (class-weighted) |
| Grad clip | 1.0 | 0.1 |
| Image size | 224×224 | 224×224 |
| Dropout | 0.5 | 0.3 |
| Seed | 42 | 42 |

---

## 📈 Results

| Model | Best Val Acc | Test Acc |
|-------|-------------|----------|
| CNN Baseline (3-Stream ResNet-50) | ~92.61% | — |
| DP-HOI (DETR ResNet-50 + Spatial Head) | ~91.44% | ~91.44% |

Results are saved to:
- `CNN_baseline/cnn_baseline_results.json`
- `CNN_baseline/dphoi_results.json`

---

## 🔄 Inference Pipeline (End-to-End)

```
Input Image
    ↓
YOLOv8 Detection → Person Box + Firearm Box
    ↓
Crop: Person | Firearm | Union
    ↓
CNN / DP-HOI Classification → Action Label + Confidence
    ↓
BLIP-2 VQA on Union Crop → Scene Description
    ↓
Threat Score Calculation
    ↓
Threat Report (LOW / MEDIUM / HIGH)
```

---

## 💾 Checkpoints

All model weights are saved to Google Drive at `Imaan Deep Learning/CNN_baseline/`:

| File | Description |
|------|-------------|
| `cnn_baseline_best.pth` | Best CNN baseline weights |
| `dphoi_best.pth` | Best DP-HOI weights |
| `yolo_lfc/weights/best.pt` | Best YOLOv8 weights |
| `cnn_baseline_curves.png` | CNN training curves |
| `dphoi_curves.png` | DP-HOI training curves |
| `cnn_baseline_confusion.png` | CNN confusion matrix |
| `dphoi_confusion.png` | DP-HOI confusion matrix |

---

## ⚠️ Known Issues & Limitations

See the full [Code Analysis Report](./CODE_ANALYSIS_REPORT.md) for a detailed list of issues identified in the code.

---

## 👤 Author

Maria Imran, Imaan Mufti, Zunaira Zaheer, Bilal Bushra, Neeka Javed
