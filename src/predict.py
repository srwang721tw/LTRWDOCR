"""Load a trained model and predict the digit string in a CAPTCHA image."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.preprocess import enhance_contrast, detect_n_digits, segment_digits

_DEFAULT_MODEL = "models/digit_cnn.h5"
_IMG_SIZE = 28


def predict_captcha(
    img_path: str | Path,
    model_path: str | Path = _DEFAULT_MODEL,
) -> str:
    """Predict the digit string encoded in a CAPTCHA image.

    Digit count (5 or 6) is determined automatically from the image whitespace
    using :func:`~src.preprocess.detect_n_digits`.

    Args:
        img_path: Path to a CAPTCHA PNG file.
        model_path: Path to the trained ``.h5`` model file.

    Returns:
        Predicted digit string (e.g. ``"38271"`` or ``"914182"``).

    Raises:
        FileNotFoundError: If ``img_path`` or ``model_path`` does not exist.
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
    enhanced = enhance_contrast(img)
    segs = segment_digits(enhanced, n)

    arrays = np.stack(
        [
            np.array(seg.resize((_IMG_SIZE, _IMG_SIZE)), dtype=np.float32) / 255.0
            for seg in segs
        ]
    )[..., np.newaxis]

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
