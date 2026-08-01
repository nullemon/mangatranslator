import anthropic
import base64
import cv2
import json
import numpy as np
import httpx
from typing import Dict, List

from . import prompts


# Vision APIs cap an inline image at ~10 MB of base64 and downsample anything
# larger than ~1568 px on the long edge anyway. Shrink to that and send JPEG so
# a big lossless scan can't blow the limit. Detections use PERCENTAGE coords, so
# this never shifts where text lands back on the full-res page.
MAX_IMAGE_EDGE = 1568
API_IMAGE_MEDIA_TYPE = "image/jpeg"


def _prep_for_api(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge > MAX_IMAGE_EDGE:
        s = MAX_IMAGE_EDGE / float(long_edge)
        image = cv2.resize(
            image, (max(1, round(w * s)), max(1, round(h * s))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _encode_image_b64(image: np.ndarray, quality: int = 90) -> str:
    """Downscale to the API's working size and encode as JPEG, stepping quality
    down if the result would still exceed the inline limit."""
    image = _prep_for_api(image)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode image")
    q = quality
    while len(buf) * 4 / 3 > 9_500_000 and q > 40:
        q -= 15
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
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

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6", style: str = "",
                 source_lang: str = "Japanese", translate_sfx: bool = False):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.style = (style or "").strip()
        self.source_lang = source_lang or "Japanese"
        self.translate_sfx = bool(translate_sfx)

    def _image_block(self, image: np.ndarray) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": API_IMAGE_MEDIA_TYPE,
                "data": _encode_image_b64(image),
            },
        }

    def _ask(self, content: list) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,   # a full page of dialogue can overrun 4096 and truncate the JSON
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    def translate_regions(
        self, original, annotated, num_regions, target_lang="English"
    ) -> Dict[int, dict]:
        prompt = prompts.region_translate_prompt(
            target_lang, num_regions, self.style, self.source_lang, self.translate_sfx)
        text = self._ask(
            [self._image_block(original), self._image_block(annotated), {"type": "text", "text": prompt}]
        )
        return _to_region_dict(prompts.extract_json_array(text))

    def smart_detect_and_translate(self, image, target_lang="English") -> List[dict]:
        prompt = prompts.smart_detect_prompt(
            target_lang, self.style, self.source_lang, self.translate_sfx)
        text = self._ask([self._image_block(image), {"type": "text", "text": prompt}])
        return prompts.extract_json_array(text)

    def translate_texts(self, id_to_text: dict, target_lang="English",
                        image=None) -> Dict[int, dict]:
        prompt = prompts.text_translate_prompt(
            target_lang, self.style, self.source_lang, self.translate_sfx,
            with_image=image is not None)
        payload = json.dumps(id_to_text, ensure_ascii=False)
        content = []
        if image is not None:
            content.append(self._image_block(image))  # panel context for the AI
        content.append({"type": "text", "text": prompt + "\n\n" + payload})
        text = self._ask(content)
        return _to_region_dict(prompts.extract_json_array(text))

    def detect_free_text(self, image, target_lang="English", bubble_ids=None) -> List[dict]:
        prompt = prompts.free_text_detect_prompt(
            target_lang, bubble_ids or [], self.style, self.source_lang, self.translate_sfx)
        text = self._ask([self._image_block(image), {"type": "text", "text": prompt}])
        try:
            return prompts.extract_json_array(text)
        except ValueError:
            return []

    def translate_crop(self, image, target_lang="English") -> dict:
        """Vision read + translate one cropped region (any language)."""
        prompt = prompts.crop_translate_prompt(target_lang, self.source_lang, self.style)
        text = self._ask([self._image_block(image), {"type": "text", "text": prompt}])
        try:
            arr = prompts.extract_json_array(text)
        except ValueError:
            arr = []
        return arr[0] if arr else {"original": "", "translation": ""}

    def analyze_pages(self, images, target_lang="English") -> dict:
        """Study several already-translated pages and distill a style profile."""
        prompt = prompts.learn_profile_prompt(target_lang, self.source_lang)
        content = [self._image_block(im) for im in images]
        content.append({"type": "text", "text": prompt})
        return prompts.extract_json_object(self._ask(content))


class GeminiTranslator:
    """Translation backend powered by Google Gemini (vision), via REST."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite",
                 timeout: float = 180.0, style: str = "",
                 source_lang: str = "Japanese", translate_sfx: bool = False):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.style = (style or "").strip()
        self.source_lang = source_lang or "Japanese"
        self.translate_sfx = bool(translate_sfx)

    def _image_part(self, image: np.ndarray) -> dict:
        return {"inlineData": {"mimeType": API_IMAGE_MEDIA_TYPE, "data": _encode_image_b64(image)}}

    def _ask(self, parts: list) -> str:
        def _body(disable_thinking: bool) -> dict:
            gc = {"temperature": 0.2, "maxOutputTokens": 16384}
            if not disable_thinking:
                # Thinking-only models spend reasoning tokens from the SAME
                # output budget. 16k gets eaten by the thinking and the JSON
                # comes back TRUNCATED — garbage half-translations like
                # "THE PSYCHIC CLU..." — so give these models real headroom.
                gc["maxOutputTokens"] = 65536
            if disable_thinking:
                # Gemini 2.5 flash/flash-lite spend "thinking" tokens from the
                # SAME output budget — a busy page can burn it all and return no
                # text (finishReason=MAX_TOKENS). Turning thinking off (cheapest)
                # gives the whole budget to the translation JSON.
                gc["thinkingConfig"] = {"thinkingBudget": 0}
            return {"contents": [{"parts": parts}], "generationConfig": gc}

        url = self.URL.format(model=self.model)
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=_body(True))
            # Thinking-only models (gemini-2.5-pro, Gemini 3, and the -latest
            # aliases that point at them) reject thinkingBudget:0. The rejection
            # is sometimes a specific "thinking budget" message, but on newer
            # models it's just a generic 400 "invalid argument" — so retry once
            # with thinking left on for ANY 400, not only ones that name it.
            if resp.status_code == 400:
                resp = client.post(url, headers=headers, json=_body(False))

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
        prompt = prompts.region_translate_prompt(
            target_lang, num_regions, self.style, self.source_lang, self.translate_sfx)
        text = self._ask(
            [{"text": prompt}, self._image_part(original), self._image_part(annotated)]
        )
        return _to_region_dict(prompts.extract_json_array(text))

    def smart_detect_and_translate(self, image, target_lang="English") -> List[dict]:
        prompt = prompts.smart_detect_prompt(
            target_lang, self.style, self.source_lang, self.translate_sfx)
        text = self._ask([{"text": prompt}, self._image_part(image)])
        return prompts.extract_json_array(text)

    def translate_texts(self, id_to_text: dict, target_lang="English",
                        image=None) -> Dict[int, dict]:
        prompt = prompts.text_translate_prompt(
            target_lang, self.style, self.source_lang, self.translate_sfx,
            with_image=image is not None)
        payload = json.dumps(id_to_text, ensure_ascii=False)
        parts = []
        if image is not None:
            parts.append(self._image_part(image))  # panel context for the AI
        parts.append({"text": prompt + "\n\n" + payload})
        text = self._ask(parts)
        return _to_region_dict(prompts.extract_json_array(text))

    def detect_free_text(self, image, target_lang="English", bubble_ids=None) -> List[dict]:
        prompt = prompts.free_text_detect_prompt(
            target_lang, bubble_ids or [], self.style, self.source_lang, self.translate_sfx)
        text = self._ask([{"text": prompt}, self._image_part(image)])
        try:
            return prompts.extract_json_array(text)
        except ValueError:
            return []

    def translate_crop(self, image, target_lang="English") -> dict:
        """Vision read + translate one cropped region (any language)."""
        prompt = prompts.crop_translate_prompt(target_lang, self.source_lang, self.style)
        text = self._ask([{"text": prompt}, self._image_part(image)])
        try:
            arr = prompts.extract_json_array(text)
        except ValueError:
            arr = []
        return arr[0] if arr else {"original": "", "translation": ""}

    def analyze_pages(self, images, target_lang="English") -> dict:
        """Study several already-translated pages and distill a style profile."""
        prompt = prompts.learn_profile_prompt(target_lang, self.source_lang)
        parts = [{"text": prompt}] + [self._image_part(im) for im in images]
        return prompts.extract_json_object(self._ask(parts))

    def _err(self, resp: httpx.Response) -> str:
        try:
            msg = resp.json().get("error", {})
            msg = msg.get("message") if isinstance(msg, dict) else msg
            if msg:
                return f"Gemini error {resp.status_code}: {msg}"
        except Exception:
            pass
        return f"Gemini error {resp.status_code}: {resp.text[:300]}"


# Gemini default is a rolling "latest" alias so a fresh install never lands on
# a model Google has retired ("no longer available"). Retired ids that a saved
# setting might still send are remapped below.
DEFAULT_MODELS = {"claude": "claude-sonnet-4-6", "gemini": "gemini-flash-latest"}

# Gemini ids Google has retired for new accounts → the current equivalent, so an
# old saved/hardcoded choice keeps working instead of 404-ing.
_GEMINI_RETIRED = {
    "gemini-2.0-flash": "gemini-flash-latest",
    "gemini-1.5-flash": "gemini-flash-latest",
    "gemini-1.5-pro": "gemini-pro-latest",
}


def make_translator(provider: str, api_key: str, model: str = "", style: str = "",
                    source_lang: str = "Japanese", translate_sfx: bool = False):
    provider = (provider or "claude").lower().strip()
    model = (model or "").strip()

    if provider in ("claude", "anthropic"):
        # Guard against a stale Gemini model id being sent for Claude.
        if not model or model.startswith("gemini"):
            model = DEFAULT_MODELS["claude"]
        return ClaudeTranslator(api_key, model, style=style,
                                source_lang=source_lang, translate_sfx=translate_sfx)

    if provider in ("gemini", "google"):
        if not model or model.startswith("claude"):
            model = DEFAULT_MODELS["gemini"]
        # A saved setting may still point at a fully-retired model — remap it to
        # the current equivalent so it works instead of 404-ing. (Models that
        # are only "restricted to existing accounts", like gemini-2.5-pro, are
        # left as-is: they still work for the accounts that have them.)
        if model in _GEMINI_RETIRED:
            new = _GEMINI_RETIRED[model]
            print(f"[translator] '{model}' is retired — using '{new}' instead")
            model = new
        return GeminiTranslator(api_key, model, style=style,
                                source_lang=source_lang, translate_sfx=translate_sfx)

    raise ValueError(f"Unknown translation provider: {provider!r}")
