"""Standalone CAPTCHA predictor.

Drop this file and a trained ``digit_cnn.h5`` into any project and call
:func:`predict_captcha` directly — no other project files are required.

Dependencies (pip install):
    opencv-python, Pillow, tensorflow, numpy

Usage (CLI)::

    python captcha_predictor.py --image captcha.png --model digit_cnn.h5

Usage (Python)::

    from captcha_predictor import predict_captcha
    print(predict_captcha("captcha.png", "digit_cnn.h5"))  # e.g. "5865"
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Image format constants
# ---------------------------------------------------------------------------

_DIGIT_SIZE = 28  # each crop is resized to this square size (px)

# Per-format reference widths and active-region column ranges.
#   n=4 : 56 × 20 px JPEG — digits at actual cols 3–47
#   n=5 : 90 × 30 px PNG  — 120 px reference, digits at ref cols 20–100
#   n=6 : 90 × 30 px PNG  — 120 px reference, digits at ref cols 10–110
_REF_WIDTHS: dict[int, int] = {4: 56, 5: 120, 6: 120}
_MARGINS: dict[int, tuple[int, int]] = {
    4: (3, 47),
    5: (20, 100),
    6: (10, 110),
}

_FOUR_DIGIT_MAX_WIDTH = 65  # images narrower than this are the 4-digit JPEG format
_MARGIN_MID = 15            # midpoint between 5-digit (20 px) and 6-digit (10 px) margins


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def enhance_contrast(img: Image.Image) -> Image.Image:
    """Isolate colored digit pixels using HSV saturation.

    Colored digits (red/blue) have high HSV saturation; the white/gray
    background and noise have low saturation.  The saturation channel is
    blended with inverted grayscale to catch achromatic strokes, then
    sharpened with CLAHE.

    Args:
        img: Input PIL image (any mode).

    Returns:
        Grayscale PIL image where digit pixels are bright.
    """
    rgb = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blended = cv2.addWeighted(saturation, 0.7, 255 - gray, 0.3, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return Image.fromarray(clahe.apply(blended))


def detect_n_digits(img: Image.Image) -> int:
    """Classify a CAPTCHA as 4-, 5-, or 6-digit.

    The 4-digit format is identified by image width (≤ 65 px).
    For wider images the mean binary projection in the discriminant zone
    (cols 10–20 in the 120 px reference) separates 5- from 6-digit images.

    Args:
        img: Original PIL CAPTCHA image (any mode).

    Returns:
        4, 5, or 6.
    """
    if img.width <= _FOUR_DIGIT_MAX_WIDTH:
        return 4

    enhanced = enhance_contrast(img)
    gray = np.array(enhanced.convert("L"))
    w = gray.shape[1]

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proj = binary.sum(axis=0).astype(float)

    if proj.max() == 0:
        return 6

    ref = _REF_WIDTHS[5]
    zone_start = round(_MARGINS[6][0] * w / ref)
    zone_end = round(_MARGINS[5][0] * w / ref)
    zone_mean = proj[zone_start:zone_end].mean()

    return 6 if zone_mean >= proj.max() * 0.15 else 5


def segment_digits(enhanced: Image.Image, n: int) -> list[Image.Image]:
    """Split a CAPTCHA into exactly n equal digit crops.

    Uses fixed per-format margins scaled to the actual image width.
    Each crop is resized to ``_DIGIT_SIZE × _DIGIT_SIZE`` px.

    Args:
        enhanced: Grayscale output of :func:`enhance_contrast`.
        n: Number of digits (4, 5, or 6).

    Returns:
        List of exactly n square PIL images.
    """
    gray = np.array(enhanced.convert("L"))
    w = gray.shape[1]
    scale = w / _REF_WIDTHS[n]

    ref_left, ref_right = _MARGINS[n]
    left = round(ref_left * scale)
    right = round(ref_right * scale)
    width = right - left

    crops: list[Image.Image] = []
    for i in range(n):
        x0 = left + round(i * width / n)
        x1 = left + round((i + 1) * width / n)
        crop = Image.fromarray(gray[:, x0:x1])
        crops.append(crop.resize((_DIGIT_SIZE, _DIGIT_SIZE), Image.LANCZOS))
    return crops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_captcha(
    img_path: str | Path,
    model_path: str | Path = "digit_cnn.h5",
) -> str:
    """Predict the digit string encoded in a CAPTCHA image.

    Args:
        img_path: Path to a CAPTCHA image (PNG, JPG, or JPEG).
        model_path: Path to the trained ``.h5`` model file.

    Returns:
        Predicted digit string, e.g. ``"5865"`` or ``"912599"``.

    Raises:
        FileNotFoundError: If either path does not exist.

    Example::

        from captcha_predictor import predict_captcha
        print(predict_captcha("captcha.png", "digit_cnn.h5"))
    """
    img_path = Path(img_path)
    model_path = Path(model_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path))
    img = Image.open(img_path)
    n = detect_n_digits(img)
    segs = segment_digits(enhance_contrast(img), n)

    arrays = np.stack(
        [np.array(s, dtype=np.float32) / 255.0 for s in segs]
    )[..., np.newaxis]  # (n, 28, 28, 1)

    predictions = model.predict(arrays, verbose=0).argmax(axis=1)
    return "".join(str(d) for d in predictions)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CAPTCHA digit string.")
    parser.add_argument("--image", required=True, help="Path to CAPTCHA image")
    parser.add_argument("--model", default="digit_cnn.h5", help="Path to .h5 model")
    args = parser.parse_args()
    print(predict_captcha(args.image, args.model))
