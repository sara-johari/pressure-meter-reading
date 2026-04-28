"""
Prompt templates for the pressure meter reading agent.

To A/B test prompts, add a new key to PROMPTS and pass prompt_version=
to reader.py. The version string is recorded in predictions.csv so you
can compare results across runs.
"""

PROMPTS: dict[str, str] = {

    # ---------------------------------------------------------------
    # v1 — structured, step-by-step reasoning
    # ---------------------------------------------------------------
    "v1": """You are an expert instrument reader specialising in pressure gauges.

This gauge may have two concentric scales with different units.

Reply ONLY with this JSON:

{
  "scales": [
    {
      "position": "outer" | "inner",
      "unit": "<string>",
      "scale_min": <number>,
      "scale_max": <number>,
      "predicted_value": <number>,
      "confidence": "high" | "medium" | "low"
    }
  ],
  "primary_scale": "outer" | "inner",
  "notes": "<optional>"
}

Steps to follow before answering:
1. Identify the full scale range from the printed numbers on the dial.
2. Identify the unit of measurement (bar, psi, kPa, MPa, etc.).
3. Describe where the needle is pointing using clock positions
   (e.g. "pointing at 4 o'clock").
4. Calculate the reading by interpolating the needle position between
   the nearest tick marks.
5. Report your confidence level.

If you cannot determine a value with any confidence, set predicted_value
to null and explain in notes.""",

    # ---------------------------------------------------------------
    # v2 — shorter, more direct
    # ---------------------------------------------------------------
    "v2": """Read the pressure gauge in this image.

Reply ONLY with this JSON (no other text):

{
  "scale_min": <number or null>,
  "scale_max": <number or null>,
  "unit": "<string or null>",
  "predicted_value": <number or null>,
  "confidence": "high" | "medium" | "low",
  "notes": "<optional>"
}""",

}

DEFAULT_PROMPT_VERSION = "v1"


def get_prompt(version: str = DEFAULT_PROMPT_VERSION) -> str:
    if version not in PROMPTS:
        raise ValueError(
            f"Unknown prompt version '{version}'. "
            f"Available: {list(PROMPTS.keys())}"
        )
    return PROMPTS[version]
