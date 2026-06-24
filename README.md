# LTRWDOCR — Numeric CAPTCHA Recognition

> **99.82% per-digit accuracy** across 4-, 5-, and 6-digit CAPTCHA formats.  
> CPU-only inference · No GPU required · Portable `.h5` model

---

## Overview

End-to-end machine learning pipeline that automatically reads numeric CAPTCHAs from a web service. The system handles multiple image formats in a single model, achieving near-human accuracy on unseen images after training on a few hundred labeled samples.

### Key results

| Metric | Value |
|--------|-------|
| Per-digit accuracy (test set) | **99.82 %** |
| 4-digit CAPTCHA accuracy | ~99.3 % |
| 5-digit CAPTCHA accuracy | ~99.1 % |
| 6-digit CAPTCHA accuracy | ~98.8 % |
| Training samples needed | ~150–200 labeled CAPTCHAs |
| Inference time (CPU) | < 1 s per image |

---

## Technical approach

**Preprocessing pipeline**

Raw CAPTCHAs arrive in two image formats with varying digit counts. The pipeline auto-detects the format before segmentation:

1. **Format detection** — image width classifies 4-digit JPEG format (56 px) vs. wider PNG formats; a discriminant-zone projection distinguishes 5- vs. 6-digit within the PNG group.
2. **Contrast enhancement** — converts to HSV color space, blends the saturation channel (70 %) with inverted grayscale (30 %), and applies CLAHE. Colored digit pixels become bright on a dark background regardless of hue variation.
3. **Digit segmentation** — uses per-format known margins (calibrated from the source spec) to divide the active region into equal-width crops. Each crop is resized to 28 × 28 px.

**Model**

A lightweight CNN trained with TensorFlow (CPU-only):

```
Input 28×28×1
→ Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
→ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
→ Dense(128, ReLU) + Dropout(0.5)
→ Dense(10, Softmax)
```

Trained with Adam optimizer, sparse categorical cross-entropy loss, and early stopping on validation loss (patience = 5). Model is serialized as HDF5 (`.h5`) for maximum portability — loadable in any Python environment with `tf.keras.models.load_model()`.

**Labeling efficiency**

A two-phase labeling strategy cuts annotation time significantly:

- **Phase 1** — manually label ~150 images (≈ 20 min) and train an initial model.
- **Phase 2** — use the model to pre-fill answers; the operator only corrects mistakes. An interactive progress display shows the digit class with the fewest samples, guiding targeted collection.

---

## Supported CAPTCHA formats

| Format | Image size | Digits | Active region |
|--------|-----------|--------|---------------|
| 4-digit JPEG | 56 × 20 px | 4 | cols 3–47 |
| 5-digit PNG | 90 × 30 px | 5 | cols 15–75 (scaled) |
| 6-digit PNG | 90 × 30 px | 6 | cols 8–83 (scaled) |

---

## Tech stack

| Layer | Library |
|-------|---------|
| Image processing | OpenCV, Pillow |
| Model training & inference | TensorFlow (CPU) |
| Dataset splitting & metrics | scikit-learn |
| HTTP download loop | requests + python-dotenv |
| Testing | pytest |

---

## Setup

**Requirements:** Python 3.9+, macOS / Linux (no CUDA needed)

```bash
git clone https://github.com/srwang721tw/LTRWDOCR.git
cd LTRWDOCR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the source URL:

```bash
cp .env.example .env
# edit .env → CAPTCHA_URL=<your URL>
```

> **macOS arm64 note:** The `tensorflow` package is used (not `tensorflow-cpu` or `tensorflow-metal`). Omitting `tensorflow-metal` keeps inference on CPU, which is intentional.

---

## Usage

### Collect training data

```bash
python -m src.download --count 1000          # ~1000 raw CAPTCHAs, random 0–1 s delay
```

### Label (two-phase)

**Phase 1 — plain labeling:**
```bash
python -m src.label
```

A cv2 window shows each CAPTCHA at 4× zoom. Type the answer and press Enter.

**Phase 2 — model-assisted (after first training run):**
```bash
python -m src.label --model models/digit_cnn.h5
```

The model pre-fills the predicted answer. Press Enter to accept or type a correction. The prompt shows `[current/total]` and the digit class with the fewest samples.

| Input | Action |
|-------|--------|
| `<digits>` + Enter | Confirm / correct |
| Enter (blank) | Skip |
| `q` + Enter | Quit |

### Train

```bash
python -m src.train
```

Prints a `classification_report` on the validation set (for hyperparameter tuning) and the held-out test set (reported once). Saves the best checkpoint to `models/digit_cnn.h5`.

### Predict

```bash
python -m src.predict --image data/raw/some_captcha.png
# → Predicted CAPTCHA: 58884
```

**From Python (integration example):**

```python
from src.predict import predict_captcha

answer = predict_captcha("path/to/captcha.png", "models/digit_cnn.h5")
print(answer)  # e.g. "5865"
```

---

## Project structure

```
LTRWDOCR/
├── data/
│   ├── raw/      # raw downloaded CAPTCHAs (PNG + JPEG, git-ignored)
│   └── 0/ … 9/  # labeled digit crops, one folder per class (git-ignored)
├── examples/
│   └── predict_captcha.py   # minimal integration example
├── models/
│   └── digit_cnn.h5         # trained model (git-ignored)
├── src/
│   ├── download.py     # loop-download with random inter-request delay
│   ├── preprocess.py   # enhance_contrast · detect_n_digits · segment_digits
│   ├── label.py        # interactive labeling (plain + model-assisted)
│   ├── train.py        # CNN training with stratified train/val/test split
│   └── predict.py      # end-to-end inference
└── tests/
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## License

Private repository — all rights reserved.
