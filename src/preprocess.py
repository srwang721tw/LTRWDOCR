"""Contrast enhancement and digit segmentation for CAPTCHA images."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_DIGIT_SIZE = 28  # output square size in pixels
_MIN_SEGMENT_WIDTH = 6  # pixels; narrower gaps are ignored


def enhance_contrast(img: Image.Image) -> Image.Image:
    """Convert to grayscale and apply CLAHE to improve digit visibility.

    Args:
        img: Input PIL image (any mode).

    Returns:
        Grayscale PIL image with enhanced local contrast.
    """
    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)


def segment_digits(img: Image.Image) -> list[Image.Image]:
    """Split a CAPTCHA image into individual digit crops.

    Uses vertical projection (column-wise pixel sums on a binarised image) to
    locate the valleys between digits, then crops and resizes each region to
    ``_DIGIT_SIZE x _DIGIT_SIZE``.

    Args:
        img: Grayscale PIL image of a full CAPTCHA.

    Returns:
        List of square PIL images, one per detected digit.  May be empty if
        segmentation produces fewer than 4 regions (malformed image).
    """
    gray = np.array(img.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Vertical projection: sum of white pixels per column
    projection = binary.sum(axis=0)

    # Find column ranges where projection > 0 (digit content)
    in_digit = False
    segments: list[tuple[int, int]] = []
    start = 0

    for col, val in enumerate(projection):
        if not in_digit and val > 0:
            in_digit = True
            start = col
        elif in_digit and val == 0:
            in_digit = False
            if col - start >= _MIN_SEGMENT_WIDTH:
                segments.append((start, col))

    if in_digit and gray.shape[1] - start >= _MIN_SEGMENT_WIDTH:
        segments.append((start, gray.shape[1]))

    crops: list[Image.Image] = []
    for x0, x1 in segments:
        crop = Image.fromarray(gray[:, x0:x1])
        crop = crop.resize((_DIGIT_SIZE, _DIGIT_SIZE), Image.LANCZOS)
        crops.append(crop)

    return crops


def process_batch(raw_dir: str | Path, seg_dir: str | Path) -> int:
    """Enhance and segment all raw CAPTCHA images in a directory.

    Skips any image that already has corresponding segment files.

    Args:
        raw_dir: Directory containing raw CAPTCHA PNG files.
        seg_dir: Directory in which to write ``{stem}_d{i}.png`` crops.

    Returns:
        Total number of segment files written.
    """
    raw_path = Path(raw_dir)
    seg_path = Path(seg_dir)
    seg_path.mkdir(parents=True, exist_ok=True)

    written = 0
    images = sorted(raw_path.glob("*.png"))
    print(f"Processing {len(images)} images …")

    for img_file in images:
        existing = list(seg_path.glob(f"{img_file.stem}_d*.png"))
        if existing:
            continue  # already segmented

        try:
            img = Image.open(img_file)
            enhanced = enhance_contrast(img)
            digits = segment_digits(enhanced)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {img_file.name}: {exc}")
            continue

        if len(digits) < 4:
            print(f"  [skip] {img_file.name}: only {len(digits)} segments found")
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
