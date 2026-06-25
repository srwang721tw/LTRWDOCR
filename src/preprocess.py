"""Contrast enhancement and digit segmentation for CAPTCHA images."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_DIGIT_SIZE = 28
_EXPECTED_DIGITS = (4, 5, 6)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# Per-format reference widths and active-region column ranges.
#
# n=4 : 56 × 30 px JPEG, digits at actual cols  3–47  (no scaling needed)
# n=5 : 90 × 30 px PNG,  120 px reference, digits at ref cols 20–100
# n=6 : 90 × 30 px PNG,  120 px reference, digits at ref cols 10–110
#
# segment_digits scales (left, right) by actual_width / ref_width before cutting.
_REF_WIDTHS: dict[int, int] = {4: 56, 5: 120, 6: 120}
_MARGINS: dict[int, tuple[int, int]] = {
    4: (3, 47),
    5: (20, 100),
    6: (10, 110),
}

# Images narrower than this (px) are the 4-digit JPEG format.
_FOUR_DIGIT_MAX_WIDTH = 65
# Midpoint between the 5-digit and 6-digit left margins (in the 120 px reference).
_MARGIN_MID = 15


def enhance_contrast(img: Image.Image) -> Image.Image:
    """Isolate colored digit pixels using HSV saturation.

    Colored digits (red/blue) have high HSV saturation; the white/gray
    background and light noise have low saturation.  The saturation channel
    is blended with inverted grayscale to catch any achromatic strokes,
    then sharpened with CLAHE.

    Args:
        img: Input PIL image (any mode).

    Returns:
        Grayscale PIL image where digit pixels are bright.
    """
    rgb = np.array(img.convert("RGB"))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray

    blended = cv2.addWeighted(saturation, 0.7, inverted, 0.3, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(blended)
    return Image.fromarray(enhanced)


def detect_n_digits(img: Image.Image) -> int:
    """Classify a CAPTCHA as 4-, 5-, or 6-digit.

    The 4-digit format is identified purely by image width (56 px vs 90 px).
    For the 5- vs 6-digit distinction the mean binary projection in the
    discriminant zone (cols 10–20 in the 120 px reference) is used: a high
    mean indicates digit content is already present → 6-digit; a low mean
    means the zone is still background margin → 5-digit.

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
        return 6  # fallback for blank image

    ref = _REF_WIDTHS[5]  # both 5 and 6 share the same 120 px reference
    zone_start = round(_MARGINS[6][0] * w / ref)  # ≈ col 7 in 90 px
    zone_end = round(_MARGINS[5][0] * w / ref)    # ≈ col 15 in 90 px
    zone_mean = proj[zone_start:zone_end].mean()

    return 6 if zone_mean >= proj.max() * 0.15 else 5


def segment_digits(enhanced: Image.Image, n: int) -> list[Image.Image]:
    """Split a CAPTCHA into exactly n equal digit crops using fixed margins.

    The active region boundaries are looked up from ``_MARGINS[n]`` and scaled
    from the reference width (``_REF_WIDTHS[n]``) to the actual image width.
    Each crop is resized to ``_DIGIT_SIZE × _DIGIT_SIZE``.

    Args:
        enhanced: Grayscale PIL image output of :func:`enhance_contrast`.
        n: Number of digits to extract (4, 5, or 6).

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
        crop = crop.resize((_DIGIT_SIZE, _DIGIT_SIZE), Image.LANCZOS)
        crops.append(crop)

    return crops


