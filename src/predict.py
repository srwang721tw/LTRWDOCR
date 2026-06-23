"""Load a trained model and predict the digit string in a CAPTCHA image."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.preprocess import enhance_contrast, segment_digits

_DEFAULT_MODEL = "models/digit_cnn.h5"
_IMG_SIZE = 28


def predict_captcha(img_path: str | Path, model_path: str | Path = _DEFAULT_MODEL) -> str:
    """Predict the digit string encoded in a CAPTCHA image.

    Args:
        img_path: Path to a CAPTCHA PNG file.
        model_path: Path to the trained ``.h5`` model file.

    Returns:
        Predicted digit string (e.g. ``"38271"``).

    Raises:
        FileNotFoundError: If ``img_path`` or ``model_path`` does not exist.
        ValueError: If no digit segments can be extracted from the image.
    """
    img_path = Path(img_path)
    model_path = Path(model_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path))

    img = Image.open(img_path)
    enhanced = enhance_contrast(img)
    segments = segment_digits(enhanced)

    if not segments:
        raise ValueError(f"No digit segments extracted from {img_path}")

    arrays = np.stack(
        [
            np.array(seg.resize((_IMG_SIZE, _IMG_SIZE)), dtype=np.float32) / 255.0
            for seg in segments
        ]
    )[..., np.newaxis]  # (N, 28, 28, 1)

    predictions = model.predict(arrays, verbose=0).argmax(axis=1)
    return "".join(str(d) for d in predictions)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CAPTCHA digit string.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = predict_captcha(args.image, args.model)
    print(f"Predicted CAPTCHA: {result}")
