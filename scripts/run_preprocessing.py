"""
run_preprocessing.py
--------------------
Process all images in data/raw/ through the preprocessing pipeline
and save results to data/processed/.

Usage:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --input data/raw --output data/processed
    python scripts/run_preprocessing.py --input path/to/single_image.jpg
"""

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.preprocessing.pipeline import run_pipeline

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess pressure meter images.")
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Input directory (or single image file). Default: data/raw",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed",
        help="Output directory. Default: data/processed",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    images = collect_images(input_path)
    if not images:
        print(f"[WARN] No supported images found in: {input_path}")
        sys.exit(0)

    print(f"Found {len(images)} image(s). Processing...")
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for img_path in tqdm(images, unit="img"):
        out_path = output_dir / img_path.name
        result = run_pipeline(img_path, out_path)

        if result["success"]:
            success_count += 1
        else:
            fail_count += 1
            tqdm.write(f"[FAIL] {img_path.name}: {result['message']}")

    print(f"\nDone. {success_count} succeeded, {fail_count} failed.")
    print(f"Processed images saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
