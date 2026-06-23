"""Load a trained model and predict the digit string in a CAPTCHA image."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.preprocess import enhance_contrast, segment_digits, _EXPECTED_DIGITS

_DEFAULT_MODEL = "models/digit_cnn.h5"
_IMG_SIZE = 28


def _predict_n(
    enhanced: Image.Image,
    model: tf.keras.Model,
    n: int,
) -> tuple[str, float]:
    """Segment into n digits, run inference, and return (prediction, avg_confidence).

    Args:
        enhanced: Output of :func:`~src.preprocess.enhance_contrast`.
        model: Loaded Keras model.
        n: Number of digits to segment into.

    Returns:
        Tuple of the predicted digit string and the mean max-class probability.
    """
    segs = segment_digits(enhanced, n)
    if not segs:
        return ("", 0.0)

    arrays = np.stack(
        [
            np.array(seg.resize((_IMG_SIZE, _IMG_SIZE)), dtype=np.float32) / 255.0
            for seg in segs
        ]
    )[..., np.newaxis]  # (n, 28, 28, 1)

    probs = model.predict(arrays, verbose=0)        # (n, 10)
    digits = probs.argmax(axis=1)
    confidence = float(probs.max(axis=1).mean())    # average of per-digit max prob

    return ("".join(str(d) for d in digits), confidence)


def predict_captcha(
    img_path: str | Path,
    model_path: str | Path = _DEFAULT_MODEL,
) -> str:
    """Predict the digit string encoded in a CAPTCHA image.

    Both 5-digit and 6-digit segmentations are attempted; the one with the
    higher average model confidence is returned.

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
    enhanced = enhance_contrast(Image.open(img_path))

    results = {n: _predict_n(enhanced, model, n) for n in _EXPECTED_DIGITS}
    best_n = max(results, key=lambda n: results[n][1])

    prediction, confidence = results[best_n]
    return prediction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CAPTCHA digit string.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = predict_captcha(args.image, args.model)
    print(f"Predicted CAPTCHA: {result}")
