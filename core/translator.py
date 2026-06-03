import anthropic
import base64
import cv2
import json
import numpy as np
import httpx
from typing import Dict, List

from . import prompts


def _encode_png_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf).decode("utf-8")


def _to_region_dict(items: list) -> Dict[int, dict]:
    result = {}
    for item in items:
        rid = item.get("id")
        if rid is not None:
            try:
                result[int(rid)] = item
            except (ValueError, TypeError):
                continue
    return result


class ClaudeTranslator:
    """Translation backend powered by Anthropic Claude (vision)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _image_block(self, image: np.ndarray) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_png_b64(image),
            },
        }

    def _ask(self, content: list) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    def translate_regions(
        self, original, annotated, num_regions, target_lang="English"
    ) -> Dict[int, dict]:
        prompt = prompts.region_translate_prompt(target_lang, num_regions)
        text = self._ask(
            [self._image_block(original), self._image_block(annotated), {"type": "text", "text": prompt}]
        )
        return _to_region_dict(prompts.extract_json_array(text))

    def smart_detect_and_translate(self, image, target_lang="English") -> List[dict]:
        prompt = prompts.smart_detect_prompt(target_lang)
        text = self._ask([self._image_block(image), {"type": "text", "text": prompt}])
        return prompts.extract_json_array(text)

    def translate_texts(self, id_to_text: dict, target_lang="English") -> Dict[int, dict]:
        prompt = prompts.text_translate_prompt(target_lang)
        payload = json.dumps(id_to_text, ensure_ascii=False)
        text = self._ask([{"type": "text", "text": prompt + "\n\n" + payload}])
        return _to_region_dict(prompts.extract_json_array(text))

    def detect_free_text(self, image, target_lang="English", bubble_ids=None) -> List[dict]:
        prompt = prompts.free_text_detect_prompt(target_lang, bubble_ids or [])
        text = self._ask([self._image_block(image), {"type": "text", "text": prompt}])
        try:
            return prompts.extract_json_array(text)
        except ValueError:
            return []


class GeminiTranslator:
    """Translation backend powered by Google Gemini (vision), via REST."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", timeout: float = 180.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _image_part(self, image: np.ndarray) -> dict:
        return {"inlineData": {"mimeType": "image/png", "data": _encode_png_b64(image)}}

    def _ask(self, parts: list) -> str:
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
        }
        url = self.URL.format(model=self.model)
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=body)

        if resp.status_code != 200:
            raise RuntimeError(self._err(resp))

        data = resp.json()
        cands = data.get("candidates", [])
        if not cands:
            fb = data.get("promptFeedback")
            raise RuntimeError(f"Gemini returned no candidates: {fb or str(data)[:300]}")

        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip():
            reason = cands[0].get("finishReason")
            raise RuntimeError(f"Gemini returned no text (finishReason={reason})")
        return text

    def translate_regions(
        self, original, annotated, num_regions, target_lang="English"
    ) -> Dict[int, dict]:
        prompt = prompts.region_translate_prompt(target_lang, num_regions)
        text = self._ask(
            [{"text": prompt}, self._image_part(original), self._image_part(annotated)]
        )
        return _to_region_dict(prompts.extract_json_array(text))

    def smart_detect_and_translate(self, image, target_lang="English") -> List[dict]:
        prompt = prompts.smart_detect_prompt(target_lang)
        text = self._ask([{"text": prompt}, self._image_part(image)])
        return prompts.extract_json_array(text)

    def translate_texts(self, id_to_text: dict, target_lang="English") -> Dict[int, dict]:
        prompt = prompts.text_translate_prompt(target_lang)
        payload = json.dumps(id_to_text, ensure_ascii=False)
        text = self._ask([{"text": prompt + "\n\n" + payload}])
        return _to_region_dict(prompts.extract_json_array(text))

    def detect_free_text(self, image, target_lang="English", bubble_ids=None) -> List[dict]:
        prompt = prompts.free_text_detect_prompt(target_lang, bubble_ids or [])
        text = self._ask([{"text": prompt}, self._image_part(image)])
        try:
            return prompts.extract_json_array(text)
        except ValueError:
            return []

    def _err(self, resp: httpx.Response) -> str:
        try:
            msg = resp.json().get("error", {})
            msg = msg.get("message") if isinstance(msg, dict) else msg
            if msg:
                return f"Gemini error {resp.status_code}: {msg}"
        except Exception:
            pass
        return f"Gemini error {resp.status_code}: {resp.text[:300]}"


DEFAULT_MODELS = {"claude": "claude-sonnet-4-6", "gemini": "gemini-2.5-flash"}


def make_translator(provider: str, api_key: str, model: str = ""):
    provider = (provider or "claude").lower().strip()
    model = (model or "").strip()

    if provider in ("claude", "anthropic"):
        # Guard against a stale Gemini model id being sent for Claude.
        if not model or model.startswith("gemini"):
            model = DEFAULT_MODELS["claude"]
        return ClaudeTranslator(api_key, model)

    if provider in ("gemini", "google"):
        if not model or model.startswith("claude"):
            model = DEFAULT_MODELS["gemini"]
        return GeminiTranslator(api_key, model)

    raise ValueError(f"Unknown translation provider: {provider!r}")
