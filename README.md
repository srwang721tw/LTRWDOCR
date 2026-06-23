# LTRWDOCR

An end-to-end pipeline for recognizing 5–6 digit numeric CAPTCHA images.  
**No GPU required** — uses a small CNN (TensorFlow CPU).

---

## Workflow overview

```
Download → Preprocess → Label → Train → Predict
```

1. **Download** raw CAPTCHA PNGs into `data/raw/`
2. **Preprocess** — contrast enhancement + digit segmentation → `data/segments/`
3. **Label** — interactive tool maps each segment to `data/0/` … `data/9/`
4. **Train** — CNN with train / val / test split; model saved as `models/digit_cnn.h5`
5. **Predict** — load model and infer the digit string from any CAPTCHA image

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the CAPTCHA source URL:

```bash
cp .env.example .env
# edit .env and set CAPTCHA_URL=<your URL>
```

---

## Usage

### 1. Download images

```bash
python -m src.download --count 1000 --out data/raw --delay 1.5
```

### 2. Segment digits

```bash
python -m src.preprocess --raw data/raw --out data/segments
```

### 3. Label segments

A **cv2 window** opens showing each full CAPTCHA. Type the digit answer in the
terminal and press Enter. Type `q` to quit early.

```bash
python -m src.label --raw data/raw --seg data/segments --out data
```

### 4. Train

```bash
python -m src.train --data data --model models/digit_cnn.h5
```

Outputs train / val / test accuracy and a per-class `classification_report`.

### 5. Predict

```bash
python -m src.predict --image data/raw/<filename>.png --model models/digit_cnn.h5
```

Or use the example script:

```bash
python examples/predict_captcha.py
```

---

## Project structure

```
LTRWDOCR/
├── data/
│   ├── raw/          # downloaded CAPTCHA images
│   ├── segments/     # unlabeled digit crops
│   └── 0/ … 9/      # labeled digit crops (one folder per class)
├── examples/
│   └── predict_captcha.py   # minimal usage example
├── models/
│   └── digit_cnn.h5         # trained model (after training)
├── src/
│   ├── download.py
│   ├── preprocess.py
│   ├── label.py
│   ├── train.py
│   └── predict.py
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Model

| Layer | Output shape |
|-------|-------------|
| Conv2D(32, 3, relu) | 28×28×32 |
| MaxPool(2) | 14×14×32 |
| Conv2D(64, 3, relu) | 14×14×64 |
| MaxPool(2) | 7×7×64 |
| Dense(128, relu) + Dropout(0.5) | 128 |
| Dense(10, softmax) | 10 |

Input: 28×28 grayscale digit crop, values in [0, 1].  
Training: Adam optimizer, EarlyStopping (patience=5) on val_loss.  
Serialized as `.h5` (HDF5) — load with `tf.keras.models.load_model()`.

---

## Running tests

```bash
pytest tests/
```
