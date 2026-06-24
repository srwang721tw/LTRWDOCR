# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the project-local venv — **never** the system conda environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS arm64 (Apple Silicon) the correct TensorFlow package is `tensorflow` (not `tensorflow-cpu` or `tensorflow-metal`). Without `tensorflow-metal` installed, it runs CPU-only, which is intentional.

## Common commands

| Task | Command |
|------|---------|
| Download CAPTCHAs | `python -m src.download --count 1000` |
| Label (plain) | `python -m src.label` |
| Label (model-assisted) | `python -m src.label --model models/digit_cnn.h5` |
| Train model | `python -m src.train` |
| Predict one image | `python -m src.predict --image data/raw/<file>.png` |
| Run tests | `pytest tests/` |
| Run one test file | `pytest tests/test_preprocess.py -v` |

All `python -m src.*` commands must be run from the project root.

## Architecture

The pipeline has four stages:

1. **`download.py`** — fetches raw CAPTCHA PNGs from `CAPTCHA_URL` (loaded from `.env`). Inter-request sleep is `random.uniform(0, delay_max)`.

2. **`preprocess.py`** — three public functions used by all other modules:
   - `enhance_contrast(img)` — converts to HSV, blends saturation channel (70%) with inverted grayscale (30%), applies CLAHE. Digit pixels become bright.
   - `detect_n_digits(img)` — classifies image as 4-, 5-, or 6-digit. Width ≤ 65 px → 4 digits. Otherwise uses discriminant-zone mean (cols 10–20 in 120 px reference): zone mean ≥ 15% of overall max → 6 digits.
   - `segment_digits(enhanced, n)` — splits the image into exactly `n` equal crops using the known fixed margins (`_MARGINS` dict), scaled by `actual_width / _REF_WIDTHS[n]`. No projection-based boundary detection.

3. **`label.py`** — interactive cv2 labeling tool. Handles PNG, JPG, and JPEG files. Segments on the fly during labeling (no pre-processing step required). Supports `--model` flag for assisted mode where the model pre-fills the predicted answer. Uses `len(answer)` (not `detect_n_digits`) for segmentation to keep labeling authoritative.

4. **`train.py`** — loads `data/0`–`data/9`, 70/15/15 stratified split, trains CNN with `EarlyStopping`, saves `models/digit_cnn.h5`.

5. **`predict.py`** — calls `detect_n_digits` → `enhance_contrast` → `segment_digits` → model inference → digit string.

`data/segments/` is **not** part of the main workflow. `process_batch` in `preprocess.py` exists only for ad-hoc inspection/testing.

## Image spec

Three formats are supported simultaneously:

| Format | Actual size | Digits | Active cols | Reference width |
|--------|-------------|--------|-------------|-----------------|
| 4-digit JPEG | 56 × 20 px | 4 | 3–47 (actual px) | 56 px |
| 5-digit PNG  | 90 × 30 px | 5 | 20–100 (ref) | 120 px |
| 6-digit PNG  | 90 × 30 px | 6 | 10–110 (ref) | 120 px |

Each digit crop is resized to **28 × 28 px**.

## Key constraints

- **No GPU / CUDA** — `tensorflow` without `tensorflow-metal`; CPU-only by design.
- **No company / service names** in code, comments, or docs. Use "LTRWD service" or "CAPTCHA source".
- **Source URL is secret** — lives in `.env` only; git-ignored. Key name: `CAPTCHA_URL`.
- **Model format** — always `.h5` (HDF5). Do not use `.keras` or `.pkl`.
- All source files follow **Google-style docstrings**, **KISS / DRY / SOLID** principles.

## Adding a new module

- Place it in `src/` with an `if __name__ == "__main__"` CLI block using `argparse`.
- Add corresponding tests in `tests/test_<module>.py`.
- `predict.py` may import from `preprocess.py`. No other cross-stage imports.
