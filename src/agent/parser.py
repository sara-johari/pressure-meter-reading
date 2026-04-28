from __future__ import annotations
import json
import re


def parse_response(raw_response: str) -> list[dict]:
    """
    Extract structured data from the VLM's text response.

    Handles two response formats:
      - v1 (dual-scale): top-level "scales" array -> list of scale dicts
      - v2 (flat/single): flat dict -> wrapped as single "outer" scale

    Args:
        raw_response: raw text string from the VLM

    Returns:
        list of dicts, each with keys:
            position        (str)            — "outer" | "inner"
            predicted_value (float | None)
            unit            (str | None)
            scale_min       (float | None)
            scale_max       (float | None)
            confidence      (str)            — "high" | "medium" | "low" | "unknown"
            notes           (str)
            parse_error     (str | None)     — set if JSON parsing failed
            primary_scale   (str | None)     — from v1 responses
    """
    error_defaults = {
        "position": "outer",
        "predicted_value": None,
        "unit": None,
        "scale_min": None,
        "scale_max": None,
        "confidence": "unknown",
        "notes": "",
        "parse_error": None,
        "primary_scale": None,
    }

    cleaned = _strip_code_fences(raw_response)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if not match:
        return [_with_error(error_defaults, "No JSON object found in response", raw_response[:500])]

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return [_with_error(error_defaults, f"JSON decode error: {exc}", match.group()[:500])]

    if "scales" in data:
        return _parse_v1_dual_scale(data)

    return _parse_v2_flat(data)


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text).strip().strip("`")


def _float_or_none(val):
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _with_error(template: dict, error_msg: str, notes: str) -> dict:
    result = template.copy()
    result["parse_error"] = error_msg
    result["notes"] = notes
    return result


def _parse_v1_dual_scale(data: dict) -> list[dict]:
    """Parse a v1 response containing a "scales" array."""
    scales_raw = data.get("scales", [])
    primary = data.get("primary_scale")
    global_notes = data.get("notes", "")

    results = []
    for entry in scales_raw:
        results.append({
            "position": entry.get("position", "outer"),
            "predicted_value": _float_or_none(entry.get("predicted_value")),
            "unit": entry.get("unit"),
            "scale_min": _float_or_none(entry.get("scale_min")),
            "scale_max": _float_or_none(entry.get("scale_max")),
            "confidence": entry.get("confidence", "unknown"),
            "notes": entry.get("notes", global_notes),
            "parse_error": None,
            "primary_scale": primary,
        })

    if not results:
        fallback = {
            "position": "outer",
            "predicted_value": None,
            "unit": None,
            "scale_min": None,
            "scale_max": None,
            "confidence": "unknown",
            "notes": global_notes,
            "parse_error": "v1 response had empty scales array",
            "primary_scale": primary,
        }
        return [fallback]

    return results


def _parse_v2_flat(data: dict) -> list[dict]:
    """Parse a v2 flat/single-scale response as a single outer scale."""
    return [{
        "position": "outer",
        "predicted_value": _float_or_none(data.get("predicted_value")),
        "unit": data.get("unit"),
        "scale_min": _float_or_none(data.get("scale_min")),
        "scale_max": _float_or_none(data.get("scale_max")),
        "confidence": data.get("confidence", "unknown"),
        "notes": data.get("notes", ""),
        "parse_error": None,
        "primary_scale": None,
    }]
