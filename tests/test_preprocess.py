"""Unit tests for preprocess.py."""

import numpy as np
import pytest
from PIL import Image

from src.preprocess import enhance_contrast, segment_digits, detect_n_digits


def _make_captcha(width: int = 90, height: int = 30, n_digits: int = 5) -> Image.Image:
    """Create a synthetic RGB CAPTCHA-like image with colored digit columns."""
    arr = np.ones((height, width, 3), dtype=np.uint8) * 255  # white background
    col_width = width // n_digits
    for i in range(n_digits):
        x0 = i * col_width + 2
        x1 = x0 + col_width - 4
        arr[4:height - 4, x0:x1] = [200, 50, 50]  # red rectangles as fake digits
    return Image.fromarray(arr, mode="RGB")


def test_enhance_contrast_returns_grayscale():
    img = Image.new("RGB", (80, 30), color=(100, 100, 100))
    result = enhance_contrast(img)
    assert result.mode == "L"


def test_enhance_contrast_same_size():
    img = Image.new("RGB", (120, 45))
    result = enhance_contrast(img)
    assert result.size == img.size


def test_segment_digits_returns_n_crops():
    for n in (4, 5, 6):
        captcha = _make_captcha(n_digits=n)
        enhanced = enhance_contrast(captcha)
        segments = segment_digits(enhanced, n)
        assert len(segments) == n, f"Expected {n} crops, got {len(segments)}"


def test_segment_digits_size():
    captcha = _make_captcha(n_digits=5)
    enhanced = enhance_contrast(captcha)
    for seg in segment_digits(enhanced, 5):
        assert seg.size == (28, 28)


def test_detect_n_digits_four_by_width():
    """Images narrower than 65 px should always be classified as 4-digit."""
    img = Image.new("RGB", (56, 30), color=(255, 255, 255))
    assert detect_n_digits(img) == 4


def test_detect_n_digits_four_jpeg_size():
    """56 × 30 px is the 4-digit JPEG format regardless of content."""
    img = _make_captcha(width=56, height=30, n_digits=4)
    assert detect_n_digits(img) == 4
