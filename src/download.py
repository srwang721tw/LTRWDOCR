"""Download CAPTCHA images from the configured source URL."""

import argparse
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_URL = os.getenv("CAPTCHA_URL", "")
_DEFAULT_DELAY_MAX = 1.0
_DEFAULT_COUNT = 1000


def download_captchas(
    n: int,
    out_dir: str | Path,
    delay_max: float = _DEFAULT_DELAY_MAX,
    url: str = _DEFAULT_URL,
) -> list[Path]:
    """Download CAPTCHA images in a loop and save them to disk.

    Args:
        n: Number of images to download.
        out_dir: Directory in which to save images.
        delay_max: Upper bound (seconds) for the random inter-request sleep;
            actual delay is drawn from uniform(0, delay_max).
        url: CAPTCHA endpoint URL (defaults to CAPTCHA_URL env var).

    Returns:
        List of paths to the saved image files.

    Raises:
        ValueError: If ``url`` is empty or ``n`` is non-positive.
        requests.HTTPError: If any HTTP response has a non-2xx status code.
    """
    if not url:
        raise ValueError("CAPTCHA_URL is not set. Create a .env file or pass url=.")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    session = requests.Session()

    for i in range(n):
        response = session.get(url, timeout=10)
        response.raise_for_status()

        filename = out_path / f"{int(time.time() * 1000)}_{i:05d}.png"
        filename.write_bytes(response.content)
        saved.append(filename)

        print(f"[{i + 1}/{n}] saved {filename.name}")
        if i < n - 1:
            time.sleep(random.uniform(0, delay_max))

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CAPTCHA images.")
    parser.add_argument("--count", type=int, default=_DEFAULT_COUNT)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--delay-max", type=float, default=_DEFAULT_DELAY_MAX)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    paths = download_captchas(args.count, args.out, args.delay_max)
    print(f"\nDone. {len(paths)} images saved to {args.out}")
