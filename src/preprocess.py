"""Contrast enhancement and digit segmentation for CAPTCHA images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_DIGIT_SIZE = 28
_PROJECTION_SMOOTH = 3
_PROJECTION_THRESH_RATIO = 0.05
_EXPECTED_DIGITS = (5, 6)
# Half-width of the valley-search window around each ideal cut position.
_CUT_SEARCH_WINDOW = 3


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


def _equal_split_refined(
    left: int,
    right: int,
    n: int,
    projection: np.ndarray,
) -> list[tuple[int, int]]:
    """Divide [left, right] into n equal parts, nudging each cut to the deepest
    projection valley within ``_CUT_SEARCH_WINDOW`` pixels of the ideal position.

    The valley snap is only applied when the found minimum is clearly lower than
    the ideal-position value (i.e. a genuine inter-digit gap exists nearby).
    When digits are touching and no real gap is present, the ideal equal-width
    position is kept so cuts stay inside the correct digit rather than drifting
    to a thin stroke of the wrong character.

    Args:
        left: First active column (inclusive).
        right: Last active column (exclusive).
        n: Number of segments to produce.
        projection: Smoothed 1-D vertical projection of the binary image.

    Returns:
        List of n (x0, x1) column pairs.
    """
    width = right - left
    cut_positions = [left]

    for i in range(1, n):
        ideal = left + round(i * width / n)
        lo = max(left, ideal - _CUT_SEARCH_WINDOW)
        hi = min(right, ideal + _CUT_SEARCH_WINDOW + 1)
        sub = projection[lo:hi]

        if len(sub) > 0:
            valley_offset = int(np.argmin(sub))
            valley = lo + valley_offset
            # Only snap when the valley is meaningfully lower than the ideal
            # position (genuine gap), not just a thin stroke within a digit.
            ideal_val = projection[ideal] if ideal < len(projection) else sub[valley_offset]
            valley_val = sub[valley_offset]
            cut = valley if valley_val < ideal_val * 0.6 else ideal
        else:
            cut = ideal

        cut_positions.append(cut)

    cut_positions.append(right)
    return [(cut_positions[i], cut_positions[i + 1]) for i in range(n)]


def segment_digits(img: Image.Image, n: int) -> list[Image.Image]:
    """Split a CAPTCHA image into exactly n digit crops.

    Pipeline:
      1. Otsu threshold (bright digit pixels → white).
      2. Morphological closing to reconnect fragmented strokes.
      3. Smoothed vertical projection to locate the active region.
      4. Equal-width split of the active region into n parts, with each
         cut nudged to the nearest local projection valley.
      5. Crop and resize each part to ``_DIGIT_SIZE × _DIGIT_SIZE``.

    The caller is responsible for supplying the correct ``n`` (5 or 6).
    During labeling this is the length of the typed answer; during inference
    both n=5 and n=6 are tried and the model picks the better prediction.

    Args:
        img: PIL image output of :func:`enhance_contrast`.
        n: Number of digits to extract (typically 5 or 6).

    Returns:
        List of exactly n square PIL images, or an empty list when the
        image appears blank.
    """
    gray = np.array(img.convert("L"))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    projection = binary.sum(axis=0).astype(float)
    smooth_k = np.ones(_PROJECTION_SMOOTH) / _PROJECTION_SMOOTH
    projection = np.convolve(projection, smooth_k, mode="same")

    if projection.max() == 0:
        return []

    col_thresh = projection.max() * _PROJECTION_THRESH_RATIO
    active_cols = projection > col_thresh

    if not active_cols.any():
        return []

    left_bound = int(np.argmax(active_cols))
    right_bound = int(len(active_cols) - 1 - np.argmax(active_cols[::-1])) + 1

    segments = _equal_split_refined(left_bound, right_bound, n, projection)

    crops: list[Image.Image] = []
    for x0, x1 in segments:
        crop = Image.fromarray(gray[:, max(0, x0):min(gray.shape[1], x1)])
        crop = crop.resize((_DIGIT_SIZE, _DIGIT_SIZE), Image.LANCZOS)
        crops.append(crop)

    return crops


def process_batch(raw_dir: str | Path, seg_dir: str | Path, n: int) -> int:
    """Enhance and segment all raw CAPTCHA images in a directory.

    Skips images that are already segmented or that return a blank result.
    The caller must supply the correct digit count ``n``; this function is
    mainly used for testing the segmentation visually before labeling.

    Args:
        raw_dir: Directory containing raw CAPTCHA PNG files.
        seg_dir: Directory in which to write ``{stem}_d{i}.png`` crops.
        n: Number of digits per CAPTCHA (5 or 6).

    Returns:
        Total number of segment files written.
    """
    raw_path = Path(raw_dir)
    seg_path = Path(seg_dir)
    seg_path.mkdir(parents=True, exist_ok=True)

    written = 0
    images = sorted(raw_path.glob("*.png"))
    print(f"Processing {len(images)} images with n={n} …")

    for img_file in images:
        if list(seg_path.glob(f"{img_file.stem}_d*.png")):
            continue

        try:
            img = Image.open(img_file)
            enhanced = enhance_contrast(img)
            digits = segment_digits(enhanced, n)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {img_file.name}: {exc}")
            continue

        if not digits:
            print(f"  [skip] {img_file.name}: blank image")
            continue

        for i, digit_img in enumerate(digits):
            out_file = seg_path / f"{img_file.stem}_d{i}.png"
            digit_img.save(out_file)
            written += 1

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment CAPTCHA images into digits.")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/segments")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    count = process_batch(args.raw, args.out)
    print(f"\nDone. {count} segment files written to {args.out}")
