"""Interactive labeling tool: show CAPTCHA → user types answer → segment on the fly.

Normal mode:  type the full answer (e.g. 51607).
Assisted mode (--model): model pre-fills the answer; press Enter to confirm or
                          type a correction.  Much faster after a first training run.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.preprocess import detect_n_digits, enhance_contrast, segment_digits, _EXPECTED_DIGITS, _IMAGE_EXTS


def _to_cv2(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a BGR numpy array for cv2.imshow."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _already_labeled(raw_stem: str, out_base: Path) -> bool:
    """Return True if any digit class folder contains a file for this stem."""
    return any(
        (out_base / str(d) / f"{raw_stem}_0.png").exists() for d in range(10)
    )


def _count_per_class(out_base: Path) -> dict[str, int]:
    """Return the number of labeled crops for each digit class."""
    return {
        str(d): len(list((out_base / str(d)).glob("*.png"))) for d in range(10)
    }


def _load_model(model_path: str | Path):
    """Load a Keras model, or return None if the path does not exist."""
    path = Path(model_path)
    if not path.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(path))


def _predict(model, img: Image.Image, n: int) -> str:
    """Run the model on ``n`` segments of ``img`` and return the digit string."""
    enhanced = enhance_contrast(img)
    segs = segment_digits(enhanced, n)
    arrays = np.stack(
        [np.array(s.resize((28, 28)), dtype=np.float32) / 255.0 for s in segs]
    )[..., np.newaxis]
    preds = model.predict(arrays, verbose=0).argmax(axis=1)
    return "".join(str(d) for d in preds)


def label_batch(
    raw_dir: str | Path,
    out_base: str | Path,
    model_path: str | Path | None = None,
) -> int:
    """Interactively label CAPTCHA images.

    For each unlabeled raw CAPTCHA:

    1. Opens a cv2 window showing the full image (scaled 4× for readability).
    2. If a model is supplied, pre-fills the predicted answer in the prompt.
       Press Enter to accept, or type a correction.
    3. Segments the image using the answer length as digit count.
    4. Saves each segment crop to ``out_base/{digit}/``.

    Controls: Enter on blank input → skip.  Type ``q`` → quit.

    Args:
        raw_dir: Directory containing raw CAPTCHA PNG files.
        out_base: Root directory whose subdirectories ``0``–``9`` receive crops.
        model_path: Optional path to a trained ``.h5`` model for assisted labeling.

    Returns:
        Number of digit images successfully labeled and saved.
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_base)

    for d in range(10):
        (out_path / str(d)).mkdir(parents=True, exist_ok=True)

    model = _load_model(model_path) if model_path else None
    if model:
        print("Assisted mode: model will pre-fill answers. Press Enter to confirm.")

    labeled = 0
    raw_images = sorted(f for f in raw_path.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    pending = [f for f in raw_images if not _already_labeled(f.stem, out_path)]
    total = len(pending)
    print(f"{total} images to label. Type 'q' to quit, Enter to skip.\n")

    for idx, raw_file in enumerate(pending, 1):
        img = Image.open(raw_file)
        n = detect_n_digits(img)

        # Show CAPTCHA scaled 4× in cv2 window
        display = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
        cv2.imshow("CAPTCHA", _to_cv2(display))
        cv2.waitKey(1)

        # Per-class counts for progress display
        counts = _count_per_class(out_path)
        min_class = min(counts, key=counts.get)
        min_count = counts[min_class]
        prompt = f"[{idx}/{total}] min={min_count}(digit {min_class})"

        if model:
            suggestion = _predict(model, img, n)
            raw_answer = input(f"{prompt} > [{suggestion}] ").strip()
            answer = raw_answer if raw_answer else suggestion
        else:
            answer = input(f"{prompt} > ").strip()

        if answer.lower() in ("q", "quit"):
            break
        if not answer:
            continue
        if not answer.isdigit() or len(answer) not in _EXPECTED_DIGITS:
            print(f"  [skip] expected {_EXPECTED_DIGITS}-digit answer, got '{answer}'")
            continue

        enhanced = enhance_contrast(img)
        segs = segment_digits(enhanced, len(answer))

        if len(segs) != len(answer):
            print(f"  [skip] segmentation returned {len(segs)} crops")
            continue

        for i, (seg, char) in enumerate(zip(segs, answer)):
            seg.save(out_path / char / f"{raw_file.stem}_{i}.png")
            labeled += 1

    cv2.destroyAllWindows()
    return labeled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively label digit crops.")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to trained .h5 model for assisted labeling (optional).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    count = label_batch(args.raw, args.out, args.model)
    print(f"\nDone. {count} digit images labeled.")
