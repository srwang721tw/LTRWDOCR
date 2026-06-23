"""Interactive labeling tool: display CAPTCHA → user types answer → sort segments."""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _load_cv2(path: Path) -> np.ndarray:
    """Load an image as a BGR numpy array suitable for cv2.imshow."""
    img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _segments_for(raw_stem: str, seg_dir: Path) -> list[Path]:
    """Return sorted segment paths for a given raw image stem."""
    return sorted(seg_dir.glob(f"{raw_stem}_d*.png"))


def _already_labeled(raw_stem: str, seg_dir: Path, out_base: Path) -> bool:
    """Return True if every segment for this raw image has been distributed."""
    segs = _segments_for(raw_stem, seg_dir)
    if not segs:
        return False
    return all(
        any((out_base / str(d) / seg.name).exists() for d in range(10))
        for seg in segs
    )


def label_batch(
    raw_dir: str | Path,
    seg_dir: str | Path,
    out_base: str | Path,
) -> int:
    """Interactively label segmented digit crops using their parent CAPTCHA image.

    For each unlabeled CAPTCHA in ``raw_dir``:

    1. Opens a cv2 window showing the full CAPTCHA.
    2. Prompts the user to type the digit string (e.g. ``38271``).
    3. Maps each typed character to the corresponding segment file.
    4. Copies each segment to ``out_base/{digit}/``.

    Args:
        raw_dir: Directory containing raw CAPTCHA PNG files.
        seg_dir: Directory containing segmented digit PNG files.
        out_base: Root directory whose subdirectories ``0``–``9`` receive crops.

    Returns:
        Number of digit images successfully labeled and copied.
    """
    raw_path = Path(raw_dir)
    seg_path = Path(seg_dir)
    out_path = Path(out_base)

    for d in range(10):
        (out_path / str(d)).mkdir(parents=True, exist_ok=True)

    labeled = 0
    raw_images = sorted(raw_path.glob("*.png"))
    pending = [
        f for f in raw_images if not _already_labeled(f.stem, seg_path, out_path)
    ]

    print(f"{len(pending)} images to label (press 'q' in window to quit early).")

    for raw_file in pending:
        segs = _segments_for(raw_file.stem, seg_path)
        if not segs:
            continue

        frame = _load_cv2(raw_file)
        cv2.imshow("CAPTCHA — type answer in terminal, then press Enter", frame)
        cv2.waitKey(1)  # pump event loop so window appears

        answer = input(f"\n{raw_file.name} ({len(segs)} digits) > ").strip()

        if answer.lower() == "q":
            break

        if len(answer) != len(segs) or not answer.isdigit():
            print(f"  [skip] expected {len(segs)} digits, got '{answer}'")
            continue

        for seg_file, char in zip(segs, answer):
            dest = out_path / char / seg_file.name
            shutil.copy2(seg_file, dest)
            labeled += 1

    cv2.destroyAllWindows()
    return labeled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively label digit crops.")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--seg", default="data/segments")
    parser.add_argument("--out", default="data")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    count = label_batch(args.raw, args.seg, args.out)
    print(f"\nDone. {count} digit images labeled.")
