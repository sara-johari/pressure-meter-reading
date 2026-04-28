from __future__ import annotations
import base64
import os
from pathlib import Path

import zhipuai
from dotenv import load_dotenv

from .prompt import get_prompt, DEFAULT_PROMPT_VERSION
from .parser import parse_response

load_dotenv()

MODEL = "glm-4.6v"


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    suffix = image_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def read_meter(
    image_path: str | Path,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict]:
    """
    Send a processed meter image to the ZhipuAI GLM-4.6V API and return
    structured readings for each scale detected.

    Args:
        image_path:     path to the processed image
        prompt_version: which prompt template to use (see prompt.py)

    Returns:
        list of dicts, one per scale, each containing keys from
        parser.parse_response() plus:
            raw_response     (str)
            prompt_version   (str)
            model            (str)
            image_path       (str)
    """
    image_path = Path(image_path)
    prompt_text = get_prompt(prompt_version)

    b64_data, media_type = _encode_image(image_path)

    client = zhipuai.ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY"))

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_data}",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )

    raw_response = response.choices[0].message.content
    parsed_scales = parse_response(raw_response)

    results = []
    for scale_entry in parsed_scales:
        results.append({
            **scale_entry,
            "raw_response": raw_response,
            "prompt_version": prompt_version,
            "model": MODEL,
            "image_path": str(image_path),
        })

    return results
