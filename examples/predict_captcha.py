"""Example: use captcha_predictor.py as a standalone module.

Copy ``captcha_predictor.py`` and ``digit_cnn.h5`` into your project,
then import and call as shown below — no other files needed.

Run from the project root:
    python examples/predict_captcha.py
"""

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from captcha_predictor import predict_captcha

MODEL_PATH = "models/digit_cnn.h5"
IMAGE_PATH = "data/raw/0001.jpeg"  # replace with any raw CAPTCHA file


def main() -> None:
    """Predict and print the CAPTCHA answer."""
    result = predict_captcha(IMAGE_PATH, MODEL_PATH)
    print(f"Predicted CAPTCHA: {result}")


if __name__ == "__main__":
    main()
