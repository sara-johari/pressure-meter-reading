import cv2
import numpy as np
from pathlib import Path

from .denoise import denoise
from .perspective import fix_perspective
from .crop import crop_to_dial
from .contrast import enhance_contrast
from .pointer import enhance_pointer
from .scale import enhance_scale


PIPELINE_VERSION = "v1.0"

# Target size passed to VLM (square)
OUTPUT_SIZE = 768


def run_pipeline(image_path: str | Path, output_path: str | Path) -> dict:
    """
    Run the full preprocessing pipeline on a single image and save the result.

    Steps (in order):
        1. Denoise       — bilateral filter
        2. Perspective   — ellipse → circle warp
        3. Crop          — Hough circle detection + tight crop
        4. Contrast      — CLAHE + gamma correction
        5. Pointer       — Canny overlay + HSV channel boost
        6. Scale         — adaptive threshold blend
        7. Resize        — to OUTPUT_SIZE × OUTPUT_SIZE

    Args:
        image_path:  path to the raw input image
        output_path: path where the processed image will be saved

    Returns:
        dict with keys: success (bool), message (str), version (str)
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "success": False,
            "message": f"Could not read image: {image_path}",
            "version": PIPELINE_VERSION,
        }

    try:
        image = denoise(image)
        image = fix_perspective(image)
        image = crop_to_dial(image)
        image = enhance_contrast(image)
        image = enhance_pointer(image)
        image = enhance_scale(image)
        image = cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

        return {
            "success": True,
            "message": f"Saved to {output_path}",
            "version": PIPELINE_VERSION,
        }

    except Exception as exc:
        return {
            "success": False,
            "message": str(exc),
            "version": PIPELINE_VERSION,
        }
