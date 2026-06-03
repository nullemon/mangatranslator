"""Shared translation prompts and JSON parsing, used by every translation
backend (Claude, Gemini) so they behave identically."""

import json
import re


def region_translate_prompt(target_lang: str, num_regions: int) -> str:
    return f"""You are an expert manga/comic translator. You are shown two images:
1. An original manga page
2. The SAME page with detected text regions marked by red numbered boxes

There are {num_regions} numbered regions. For each one that contains Japanese
(or other non-{target_lang}) text, translate it into natural {target_lang}.

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{"id": 1, "original": "Japanese text here", "translation": "{target_lang.upper()} TEXT HERE", "type": "dialogue"}},
  {{"id": 2, "original": "ナレーション", "translation": "NARRATION", "type": "narration"}}
]

Rules:
- Translate every region containing non-{target_lang} text.
- Use UPPERCASE for dialogue and narration (standard manga typesetting).
- Keep translations concise — they must fit inside small speech bubbles.
- "type" must be one of: "dialogue", "narration", "sfx", "title".
- Skip regions with no readable text or already in {target_lang}.
- Return ONLY the JSON array."""


def smart_detect_prompt(target_lang: str) -> str:
    return f"""You are an expert manga page analyzer and translator.

Carefully examine this manga page. Find EVERY piece of Japanese text — speech
bubbles, narration boxes, sound effects, titles, and small annotations.

For each text region, return its bounding box as PERCENTAGE coordinates of the
image (so they are resolution independent):
- x_pct, y_pct: top-left corner as a percentage (0-100)
- width_pct, height_pct: size as a percentage of the image dimensions

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{
    "x_pct": 50.0,
    "y_pct": 10.0,
    "width_pct": 20.0,
    "height_pct": 8.0,
    "original": "日本語テキスト",
    "translation": "{target_lang.upper()} TEXT",
    "type": "speech_bubble",
    "in_bubble": true
  }}
]

Rules:
- Find ALL text, including small annotations and sound effects.
- Be precise with bounding box coordinates — they are used for text replacement.
- Use UPPERCASE for dialogue and narration.
- Keep translations concise so they fit inside speech bubbles.
- type: "speech_bubble", "narration_box", "sfx", or "title".
- "in_bubble": true ONLY if the text is enclosed in a speech bubble or a
  drawn narration box. Set it to false for sound effects (onomatopoeia) and
  any text drawn directly over the artwork — these must NOT be replaced.
- Set type to "sfx" for sound effects / onomatopoeia (e.g. ドーン, バァン, わぁぁ).
- Return ONLY the JSON array."""


def extract_json_array(text: str) -> list:
    """Robustly pull a JSON array out of a model response that may include
    markdown fences or stray prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Some models wrap the array, e.g. {"regions": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response: {text[:300]}")
