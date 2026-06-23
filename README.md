# LTRWDOCR

An end-to-end pipeline for recognizing 5–6 digit numeric CAPTCHA images.  
**No GPU required** — uses a small CNN (TensorFlow, CPU-only).

---

## Workflow overview

```
Download → Label → Train → Predict
```

1. **Download** raw CAPTCHA PNGs into `data/raw/`
2. **Label** — interactive cv2 tool; segments digits on the fly and saves crops to `data/0/`–`data/9/`
3. **Train** — CNN with 70/15/15 train/val/test split; model saved as `models/digit_cnn.h5`
4. **Predict** — load model and infer the digit string from any CAPTCHA image

> No separate preprocessing step is needed before labeling.  
> Digit count (5 or 6) and segmentation are handled automatically.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set the source URL:

```bash
cp .env.example .env
# edit .env → CAPTCHA_URL=<your URL>
```

---

## Usage

### 1. Download images

```bash
python -m src.download --count 1000
```

Inter-request delay is a random float in `[0, delay_max]` seconds (default max = 1.0 s):

```bash
python -m src.download --count 1000 --delay-max 0.8
```

### 2. Label

A cv2 window opens showing each CAPTCHA (scaled 4×). Type the answer in the terminal and press Enter.

```bash
python -m src.label
```

**Assisted mode** (after a first training run — much faster):

```bash
python -m src.label --model models/digit_cnn.h5
```

The model pre-fills the predicted answer; press Enter to confirm or type a correction.  
The prompt also shows the digit class with the fewest samples so far.

| Key | Action |
|-----|--------|
| `<digits>` + Enter | Accept / correct answer |
| Enter (blank) | Skip this image |
| `q` + Enter | Quit |

### 3. Train

```bash
python -m src.train
```

Prints a `classification_report` for the validation set (for tuning) and the test set (final, once).  
Saves the best checkpoint to `models/digit_cnn.h5`.

### 4. Predict

```bash
python -m src.predict --image data/raw/<filename>.png
```

Or from Python:

```python
from src.predict import predict_captcha
print(predict_captcha("data/raw/some.png"))   # e.g. "58884"
```

---

## Project structure

```
LTRWDOCR/
├── data/
│   ├── raw/      # downloaded CAPTCHA PNGs
│   └── 0/ … 9/  # labeled digit crops, one folder per class
├── examples/
│   └── predict_captcha.py   # minimal usage example for external programs
├── models/
│   └── digit_cnn.h5         # trained model (HDF5, load with tf.keras)
├── src/
│   ├── download.py     # loop-download with random delay
│   ├── preprocess.py   # enhance_contrast, detect_n_digits, segment_digits
│   ├── label.py        # interactive labeling (plain + model-assisted)
│   ├── train.py        # CNN training with train/val/test split
│   └── predict.py      # load model → predict CAPTCHA string
├── tests/
├── .env.example
├── requirements.txt
└── CLAUDE.md
```

---

## Image format & segmentation

Raw CAPTCHAs are **90 × 30 px** (120 × 40 px reference spec).

| Type | Left/right margin | Active columns (120 px ref) | Digits |
|------|-------------------|-----------------------------|--------|
| 5-digit | 20 px | 20–100 (80 px, 16 px each) | 5 |
| 6-digit | 10 px | 10–110 (100 px, 16.7 px each) | 6 |

`detect_n_digits` classifies the image by computing the mean binary projection in the discriminant zone (cols 10–20 in the 120 px reference). A high mean indicates digit content → 6-digit; a low mean indicates margin → 5-digit.

Each digit crop is resized to **28 × 28 px** before being fed to the model.

---

## Model

| Layer | Output |
|-------|--------|
| Conv2D(32, 3×3, relu) | 28×28×32 |
| MaxPool(2×2) | 14×14×32 |
| Conv2D(64, 3×3, relu) | 14×14×64 |
| MaxPool(2×2) | 7×7×64 |
| Dense(128, relu) + Dropout(0.5) | 128 |
| Dense(10, softmax) | 10 |

- Optimizer: Adam  
- Loss: sparse categorical cross-entropy  
- Early stopping: patience=5 on `val_loss`  
- Saved as `.h5` (HDF5) — portable, loadable with `tf.keras.models.load_model()`

Achieved **~99.7% per-digit accuracy** (test set) with 150 labeled CAPTCHAs.

---

## Running tests

```bash
pytest tests/
```
