"""Interactive labeling tool: show CAPTCHA → user types answer → segment on the fly."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.preprocess import enhance_contrast, segment_digits, _EXPECTED_DIGITS


def _to_cv2(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a BGR numpy array for cv2.imshow."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _already_labeled(raw_stem: str, out_base: Path) -> bool:
    """Return True if any digit class folder contains a file for this stem."""
    return any(
        (out_base / str(d) / f"{raw_stem}_d0.png").exists() for d in range(10)
    )


def label_batch(raw_dir: str | Path, out_base: str | Path) -> int:
    """Interactively label CAPTCHA images.

    For each unlabeled raw CAPTCHA:

    1. Opens a cv2 window showing the full image (scaled 4× for readability).
    2. Prompts the user to type the digit string in the terminal.
    3. Segments the image on the fly using the answer length as digit count.
    4. Copies each segment crop to ``out_base/{digit}/``.

    Type ``q`` or ``quit`` to stop early.  Invalid answers (wrong length or
    non-digit characters) are skipped with a warning.

    Args:
        raw_dir: Directory containing raw CAPTCHA PNG files.
        out_base: Root directory whose subdirectories ``0``–``9`` receive crops.

    Returns:
        Number of digit images successfully labeled and saved.
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_base)

    for d in range(10):
        (out_path / str(d)).mkdir(parents=True, exist_ok=True)

    labeled = 0
    raw_images = sorted(raw_path.glob("*.png"))
    pending = [f for f in raw_images if not _already_labeled(f.stem, out_path)]
    print(f"{len(pending)} images to label. Type 'q' to quit, Enter to skip.")

    for raw_file in pending:
        img = Image.open(raw_file)

        # Scale up for visibility (4×) and show in cv2 window
        display = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
        frame = _to_cv2(display)
        cv2.imshow("CAPTCHA — type answer in terminal", frame)
        cv2.waitKey(1)

        answer = input(f"\n{raw_file.name} > ").strip()

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
            print(f"  [skip] segmentation returned {len(segs)} crops for {len(answer)}-char answer")
            continue

        for i, (seg, char) in enumerate(zip(segs, answer)):
            dest = out_path / char / f"{raw_file.stem}_{i}.png"
            seg.save(dest)
            labeled += 1

    cv2.destroyAllWindows()
    return labeled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively label digit crops.")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    count = label_batch(args.raw, args.out)
    print(f"\nDone. {count} digit images labeled.")
