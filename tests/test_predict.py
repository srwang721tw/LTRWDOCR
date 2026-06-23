"""Unit tests for predict.py (no model file required)."""

import pytest
from pathlib import Path


def test_predict_captcha_missing_image(tmp_path):
    """Should raise FileNotFoundError when image path does not exist."""
    from src.predict import predict_captcha

    with pytest.raises(FileNotFoundError, match="Image not found"):
        predict_captcha(tmp_path / "nonexistent.png", "models/digit_cnn.h5")


def test_predict_captcha_missing_model(tmp_path):
    """Should raise FileNotFoundError when model file does not exist."""
    from src.predict import predict_captcha

    img = tmp_path / "captcha.png"
    img.write_bytes(b"\x89PNG\r\n")  # dummy file — existence check only

    with pytest.raises(FileNotFoundError, match="Model not found"):
        predict_captcha(img, tmp_path / "no_model.h5")
