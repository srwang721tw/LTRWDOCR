"""Example: load the trained model and predict a CAPTCHA image.

Run from the project root with the virtual environment activated:

    python examples/predict_captcha.py

Adjust ``MODEL_PATH`` and ``IMAGE_PATH`` to match your local files.
"""

import sys
from pathlib import Path

# Allow importing src when running directly from examples/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import predict_captcha

MODEL_PATH = "models/digit_cnn.h5"
IMAGE_PATH = "data/raw/sample.png"  # replace with an actual downloaded image


def main() -> None:
    """Run prediction and print the result."""
    result = predict_captcha(IMAGE_PATH, MODEL_PATH)
    print(f"Predicted CAPTCHA: {result}")


if __name__ == "__main__":
    main()
