# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the project-local venv — **never** the system conda environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Common commands

| Task | Command |
|------|---------|
| Download CAPTCHAs | `python -m src.download --count 1000 --delay 1.5` |
| Segment digits | `python -m src.preprocess` |
| Label segments | `python -m src.label` |
| Train model | `python -m src.train` |
| Predict one image | `python -m src.predict --image data/raw/<file>.png` |
| Run tests | `pytest tests/` |
| Run one test file | `pytest tests/test_preprocess.py -v` |

All `python -m src.*` commands must be run from the project root.

## Architecture

The pipeline has five independent stages, each in its own module under `src/`:

1. **`download.py`** — fetches raw CAPTCHA PNGs from `CAPTCHA_URL` (loaded from `.env`).
2. **`preprocess.py`** — CLAHE contrast enhancement → vertical-projection digit segmentation → 28×28 crops in `data/segments/`.
3. **`label.py`** — cv2 window shows each full CAPTCHA; user types the full answer; script maps characters to segment files and copies them to `data/0/`–`data/9/`.
4. **`train.py`** — loads `data/0`–`data/9`, 70/15/15 stratified split, trains a small CNN with `EarlyStopping`, saves to `models/digit_cnn.h5`.
5. **`predict.py`** — loads `.h5`, runs stages 2+4 on a new image, returns digit string.

Data flows strictly forward: no stage reads output of a later stage.

## Key constraints

- **No GPU / CUDA** — model is CPU-only (`tensorflow-cpu`).
- **No company / service names** in code, comments, or docs. Use "LTRWD service" or "CAPTCHA source" when referring to the data origin.
- **Source URL is secret** — lives in `.env` only; `.env` is git-ignored. Check `.env.example` for the key name.
- **Model format** — always `.h5` (HDF5) for cross-version portability. Do not use `.keras` or `.pkl`.
- All source files follow **Google-style docstrings**, **KISS / DRY / SOLID** principles.

## Adding a new module

- Place it in `src/` with an `if __name__ == "__main__"` CLI block using `argparse`.
- Add corresponding tests in `tests/test_<module>.py`.
- Do not import between stages (download ↔ preprocess ↔ label ↔ train ↔ predict) except `predict.py` importing from `preprocess.py`.
