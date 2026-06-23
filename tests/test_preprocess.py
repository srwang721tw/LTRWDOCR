"""Unit tests for preprocess.py."""

import numpy as np
import pytest
from PIL import Image

from src.preprocess import enhance_contrast, segment_digits


def _make_captcha(width: int = 160, height: int = 60) -> Image.Image:
    """Create a synthetic grayscale CAPTCHA-like image with 5 digit columns."""
    arr = np.zeros((height, width), dtype=np.uint8)
    col_width = width // 5
    for i in range(5):
        x0 = i * col_width + 4
        x1 = x0 + col_width - 8
        arr[10:50, x0:x1] = 200  # white rectangles as fake digits
    return Image.fromarray(arr, mode="L")


def test_enhance_contrast_returns_grayscale():
    img = Image.new("RGB", (80, 30), color=(100, 100, 100))
    result = enhance_contrast(img)
    assert result.mode == "L"


def test_enhance_contrast_same_size():
    img = Image.new("RGB", (120, 45))
    result = enhance_contrast(img)
    assert result.size == img.size


def test_segment_digits_count():
    captcha = _make_captcha()
    enhanced = enhance_contrast(captcha)
    segments = segment_digits(enhanced)
    # Should detect ~5 regions; allow ±1 due to merge/split
    assert 4 <= len(segments) <= 6


def test_segment_digits_size():
    captcha = _make_captcha()
    enhanced = enhance_contrast(captcha)
    segments = segment_digits(enhanced)
    for seg in segments:
        assert seg.size == (28, 28)


def test_segment_digits_blank_image():
    blank = Image.new("L", (160, 60), color=0)
    segments = segment_digits(blank)
    assert segments == []
