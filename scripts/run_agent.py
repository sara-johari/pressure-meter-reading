"""
run_agent.py
------------
Run the VLM agent on all processed images and save predictions to
outputs/predictions.csv.

Each image produces one row per detected scale (outer, inner).
A "position" column identifies which scale each row represents.

Usage:
    python scripts/run_agent.py
    python scripts/run_agent.py --input data/processed --output outputs/predictions.csv
    python scripts/run_agent.py --prompt v2
    python scripts/run_agent.py --input path/to/single_image.jpg
"""
from __future__ import annotations


import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.agent.reader import read_meter
from src.agent.prompt import DEFAULT_PROMPT_VERSION, PROMPTS

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

FIELDNAMES = [
    "filename",
    "position",
    "predicted_value",
    "unit",
    "scale_min",
    "scale_max",
    "confidence",
    "primary_scale",
    "prompt_version",
    "model",
    "agent_notes",
    "parse_error",
    "raw_response",
]


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(description="Run VLM agent on processed meter images.")
    parser.add_argument(
        "--input", "-i",
        default="data/processed",
        help="Input directory (or single image). Default: data/processed",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/predictions.csv",
        help="Output CSV path. Default: outputs/predictions.csv",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=DEFAULT_PROMPT_VERSION,
        choices=list(PROMPTS.keys()),
        help=f"Prompt version to use. Default: {DEFAULT_PROMPT_VERSION}",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (rate limit buffer). Default: 0.5",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    images = collect_images(input_path)
    if not images:
        print(f"[WARN] No supported images found in: {input_path}")
        sys.exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Found {len(images)} image(s). Running agent with prompt "{args.prompt}"...')

    success_count = 0
    fail_count = 0

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for img_path in tqdm(images, unit="img"):
            scale_results = read_meter(img_path, prompt_version=args.prompt)

            for scale_entry in scale_results:
                scale_entry["agent_notes"] = scale_entry.pop("notes", "")
                scale_entry["filename"] = img_path.name

                writer.writerow(scale_entry)

                if scale_entry.get("parse_error"):
                    fail_count += 1
                    pos = scale_entry.get("position", "?")
                    tqdm.write(
                        f"[WARN] {img_path.name} ({pos}): "
                        f'parse error - {scale_entry["parse_error"]}'
                    )
                else:
                    success_count += 1

            csvfile.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\nDone. {success_count} scale readings parsed, {fail_count} with parse errors.")
    print(f"Predictions saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
