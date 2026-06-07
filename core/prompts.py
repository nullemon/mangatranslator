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
bubbles, narration boxes, titles, credits, captions, and annotations.

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
    "type": "dialogue",
    "in_bubble": true
  }}
]

Rules:
- Find ALL text, including chapter titles, author credits, narration, and captions.
- Be precise with bounding box coordinates — they are used for text replacement.
- Use UPPERCASE for dialogue and narration.
- Keep translations concise so they fit inside their regions.
- type must be one of: "dialogue", "narration", "sfx", "title", "credit", "caption".
- "in_bubble": true if the text is enclosed in a speech bubble or drawn box.
  Set to false for titles, credits, captions, and any text NOT enclosed.
- Set type to "sfx" for sound effects / onomatopoeia (e.g. ドーン, バァン, わぁぁ).
  SFX will NOT be replaced — everything else WILL be translated and placed.
- Do NOT include tiny furigana readings above kanji.
- Return ONLY the JSON array."""


def free_text_detect_prompt(target_lang: str, bubble_ids: list) -> str:
    skip = f"Already-handled bubble IDs: {bubble_ids}. " if bubble_ids else ""
    return f"""You are an expert manga page analyzer and translator.

The speech bubbles on this page have already been translated. {skip}Now find any
REMAINING Japanese text that is NOT inside a speech bubble and should be translated:

- Chapter titles (e.g. 第1163話 "約束")
- Author names / credits
- Narration text outside bubbles
- Editorial notes, page captions, magazine announcements
- Location/time labels

Do NOT include:
- Sound effects / onomatopoeia (ドーン, バァン, etc.)
- Text already in {target_lang}
- Tiny furigana readings above kanji

Find EVERY such region — do not stop at the obvious ones. Include short,
narrow, or vertical columns, faint text, and small boxes tucked next to other
boxes. Each separate framed box (caption / narration panel) is its OWN region.

For each text region, return its bounding box as PERCENTAGE coordinates:
- x_pct, y_pct: top-left corner (0-100)
- width_pct, height_pct: size as percentage of image dimensions

Make each box TIGHT — wrap it snugly around just that text or its framed panel.
Never let one box overlap, contain, or spill into a neighbouring box; if two
texts sit in separate boxes, return two separate, non-overlapping regions.

Return ONLY a JSON array — no markdown fences:
[
  {{
    "x_pct": 30.0,
    "y_pct": 2.0,
    "width_pct": 40.0,
    "height_pct": 5.0,
    "original": "第1163話 \\"約束\\"",
    "translation": "CHAPTER 1163: \\"THE PROMISE\\"",
    "type": "title"
  }}
]

type must be one of: "title", "credit", "narration", "caption".
Keep translations concise. Return an empty array [] if no free text is found.
Return ONLY the JSON array."""


def text_translate_prompt(target_lang: str) -> str:
    return f"""You are an expert manga translator. Below is a JSON object mapping
each speech-bubble id to the Japanese text that was read from that bubble (by
OCR). Translate every entry into natural, concise {target_lang}.

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{"id": 1, "original": "<the Japanese>", "translation": "{target_lang.upper()} TEXT", "type": "dialogue"}}
]

Rules:
- Keep each translation matched to the SAME id — never move text between ids.
- Use UPPERCASE (standard manga lettering). Keep it concise to fit the bubble.
- "type" is one of: "dialogue", "narration", "sfx". Mark pure sound effects
  (onomatopoeia) as "sfx".
- If an entry's text is empty or unreadable, return an empty translation for it.
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
