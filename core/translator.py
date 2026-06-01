import anthropic
import base64
import json
import re
import cv2
import numpy as np
from typing import Dict, List, Optional


class ClaudeTranslator:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def translate_regions(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
        num_regions: int,
        target_lang: str = "English",
    ) -> Dict[int, dict]:
        orig_b64 = self._encode(original)
        ann_b64 = self._encode(annotated)

        prompt = f"""You are an expert manga/comic translator. I'm showing you:
1. An original manga page
2. The same page with detected text regions marked by red numbered boxes

There are {num_regions} numbered regions. For each one that contains Japanese (or other non-{target_lang}) text, translate it.

Return ONLY a JSON array — no markdown fences, no commentary:
[
  {{"id": 1, "original": "Japanese text here", "translation": "{target_lang.upper()} TEXT HERE", "type": "dialogue"}},
  {{"id": 2, "original": "ナレーション", "translation": "NARRATION", "type": "narration"}}
]

Rules:
- Translate every region containing non-{target_lang} text
- Use UPPERCASE for dialogue and narration (standard manga typesetting)
- Keep translations concise — they must fit in small speech bubbles
- "type" must be one of: "dialogue", "narration", "sfx", "title"
- Skip regions with no readable text or already in {target_lang}
- Return ONLY the JSON array"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": orig_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": ann_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        return self._parse(response.content[0].text)

    def smart_detect_and_translate(
        self,
        image: np.ndarray,
        target_lang: str = "English",
    ) -> List[dict]:
        b64 = self._encode(image)

        prompt = f"""You are an expert manga page analyzer and translator.

Carefully examine this manga page. Find EVERY piece of Japanese text — speech bubbles, narration boxes, sound effects, titles, small text, everything.

For each text region, return its bounding box as percentage coordinates of the image:
- x_pct, y_pct: top-left corner as percentage (0-100)
- width_pct, height_pct: size as percentage of image dimensions

Return ONLY a JSON array — no markdown fences:
[
  {{
    "x_pct": 50.0,
    "y_pct": 10.0,
    "width_pct": 20.0,
    "height_pct": 8.0,
    "original": "日本語テキスト",
    "translation": "{target_lang.upper()} TEXT",
    "type": "speech_bubble"
  }}
]

Rules:
- Find ALL text, including small annotations and sound effects
- Be precise with bounding box coordinates — they're used for text replacement
- Use UPPERCASE for dialogue and narration
- Keep translations concise for speech bubbles
- type: "speech_bubble", "narration_box", "sfx", or "title"
- Return ONLY the JSON array"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        return self._parse_list(response.content[0].text)

    def _encode(self, image: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", image)
        return base64.b64encode(buf).decode("utf-8")

    def _parse(self, text: str) -> Dict[int, dict]:
        data = self._extract_json_array(text)
        result = {}
        for item in data:
            rid = item.get("id")
            if rid is not None:
                result[int(rid)] = item
        return result

    def _parse_list(self, text: str) -> List[dict]:
        return self._extract_json_array(text)

    def _extract_json_array(self, text: str) -> list:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {text[:300]}")
