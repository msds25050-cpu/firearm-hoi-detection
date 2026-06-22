# 🔫 Firearm Carrier Identification — Deep Learning Project

A deep learning system for detecting and classifying human-firearm interactions (HOI) in images, using a multi-model pipeline that combines **YOLOv8 detection**, a **3-Stream ResNet-50 CNN**, and **BLIP scene captioning** for threat assessment.

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
firearm-hoi-detection/
├── 01_CNN_Training.ipynb        # Model training & evaluation
│                                # Includes: CNN Baseline, YOLO fine-tuning, CNN v2
└── 02_Threat_Detection.ipynb    # Standalone inference & threat scoring demo
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
  - **Union stream** — crops the union bounding box (person + firearm combined)
- **Feature fusion:** 2048-dim features from each stream concatenated → 6144-dim vector
- **Classifier head:** `6144 → 1024 → 256 → 3 classes` with ReLU + Dropout(0.5)
- **Total parameters:** ~71M
- **Best val accuracy:** ~97.5%

### 3. BLIP — Scene Captioning
- **Model:** `Salesforce/blip-image-captioning-base`
- **Purpose:** Generates a natural language description of the scene (union crop) to augment threat scoring with caption-level keywords (e.g. gun, weapon, aiming, pointing)
- **Usage:** Inference only — no fine-tuning

---

## 📊 Threat Scoring System

The `02_Threat_Detection.ipynb` notebook implements a rule-based threat reasoner:

```
Threat Score = 0.50 × action_score + 0.30 × YOLO_confidence + 0.20 × caption_score
```

| Action | Score |
|--------|-------|
| `carrying` | 0.45 |
| `holding` | 0.65 |
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
```

### Dataset
- **LFC Dataset** — [Download from Google Drive](https://drive.google.com/drive/folders/1X0wF1EfShQzQRJILPQEHGRAH53h2Qrtu?usp=sharing)
- Update the dataset path in the notebook to match your own Drive location
- Annotations file: `labeled_interactions_augmented.json`

### Data Splits
| Split | Fraction |
|-------|----------|
| Train | 70% |
| Val | 15% |
| Test | 15% |

---

## 🏋️ Training Configuration

| Hyperparameter | CNN Baseline | CNN v2 (Fine-tuned) |
|----------------|-------------|----------------------|
| Batch size | 64 | 32 |
| Epochs | 60 | 50 |
| Base LR | 1e-4 | 1e-4 |
| Classifier LR | 1e-3 | 1e-3 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Loss | CrossEntropy (class-weighted) | CrossEntropy (class-weighted) |
| Grad clip | 1.0 | 1.0 |
| Image size | 224×224 | 224×224 |
| Dropout | 0.5 | 0.5 |
| Seed | 42 | 42 |

---

## 📈 Results

| Model | Best Val Acc |
|-------|-------------|
| CNN Baseline (3-Stream ResNet-50) | ~92.61% |
| CNN v2 (Fine-tuned on expanded data) | ~97.5% |

---

## 🔄 Inference Pipeline (End-to-End)

```
Input Image
    ↓
YOLOv8 Detection → Person Box + Firearm Box
    ↓
Crop: Person | Firearm | Union
    ↓
CNN v2 Classification → Action Label + Confidence
    ↓
BLIP Captioning on Union Crop → Scene Description
    ↓
Threat Score Calculation
    ↓
Threat Report (LOW / MEDIUM / HIGH)
```

---

## 💾 Checkpoints
 
> 📁 [Access all checkpoints on Google Drive](https://drive.google.com/drive/folders/1lSHswuMxz1x-raHuYBxy2kAqz5Hw63Ee?usp=sharing)

Model weights are saved to Google Drive. Update paths in the notebooks to match your Drive:

| File | Description |
|------|-------------|
| `cnn_baseline_best.pth` | Best CNN baseline weights |
| `cnn_v2_best.pth` | Best CNN v2 weights |
| `yolo_weights/best.pt` | Best YOLOv8 weights |

---

## 👤 Authors

Bilal Bushra   (MSDS25051)
Maria Imran    (MSDS25012)
Neeka Javed    (MSDS25008)
Imaan Mufti    (MSDS25050)
Zunaira Zaheer (MSDS25049)
