import anthropic
import base64
import contextlib
import cv2
import json
import threading
import time
import numpy as np
import httpx
from typing import Dict, List

from . import prompts


@contextlib.contextmanager
def _heartbeat(label: str, on_wait=None, every: float = 10.0):
    """Report progress WHILE a long API call is in flight.

    A vision call on a dense page can take a minute or more; without this the
    server log and the UI both sit silent and the app looks frozen. A daemon
    thread ticks every `every` seconds with the elapsed time — printed to the
    console and, when `on_wait` is set, pushed to the UI progress message."""
    stop = threading.Event()
    t0 = time.time()

    def tick():
        while not stop.wait(every):
            secs = int(time.time() - t0)
            print(f"[api] {label}: still waiting… {secs}s", flush=True)
            if on_wait:
                try:
                    on_wait(secs)
                except Exception:
                    pass

    th = threading.Thread(target=tick, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()


# Vision APIs cap an inline image at ~10 MB of base64 and downsample anything
# larger than ~1568 px on the long edge anyway. Shrink to that and send JPEG so
# a big lossless scan can't blow the limit. Detections use PERCENTAGE coords, so
# this never shifts where text lands back on the full-res page.
# How large an image each backend actually benefits from.
#
# 1568 is Anthropic's optimal tile size — sending Claude more just costs
# tokens. Gemini is different: it tiles internally and reads much larger
# images happily, and manga NEEDS that. A 2833x4000 page squeezed to 1568
# leaves the text in a speech bubble only a few pixels tall, so the model
# can no longer read the page itself and falls back on whatever the OCR
# handed it — which is exactly why pasting a page into Gemini by hand beat
# the app's own output.
MAX_IMAGE_EDGE = 1568                  # Claude / default
GEMINI_MAX_IMAGE_EDGE = 3072
API_IMAGE_MEDIA_TYPE = "image/jpeg"


def _prep_for_api(image: np.ndarray, max_edge: int = MAX_IMAGE_EDGE) -> np.ndarray:
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge > max_edge:
        s = max_edge / float(long_edge)
        image = cv2.resize(
            image, (max(1, round(w * s)), max(1, round(h * s))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _encode_image_b64(image: np.ndarray, quality: int = 90,
                      max_edge: int = MAX_IMAGE_EDGE) -> str:
    """Downscale to the API's working size and encode as JPEG, stepping quality
    down if the result would still exceed the inline limit."""
    image = _prep_for_api(image, max_edge)
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
                 source_lang: str = "Japanese", translate_sfx: bool = False,
                 webtoon: bool = False):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.style = (style or "").strip()
        self.source_lang = source_lang or "Japanese"
        self.translate_sfx = bool(translate_sfx)
        # Vertical-scroll webtoon/manhwa: changes the reading-order guidance
        # in every prompt (down the strip, left-to-right — not manga order).
        self.webtoon = bool(webtoon)
        # Optional callback(seconds) fired while a call is in flight,
        # so the UI can show live elapsed time instead of freezing.
        self.on_wait = None

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
        t0 = time.time()
        try:
            with _heartbeat(self.model, getattr(self, "on_wait", None)):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,   # a full page of dialogue can overrun 4096 and truncate the JSON
                    messages=[{"role": "user", "content": content}],
                )
        except anthropic.APIConnectionError as e:
            print(f"[claude] network error after {time.time() - t0:.0f}s: {e}",
                  flush=True)
            raise RuntimeError(
                "Can't reach the Claude API — check your internet connection. "
                "Nothing was charged; retry when you're back online, or use "
                "the Offline engine.")
        el = time.time() - t0
        if el >= 5:
            print(f"[claude] {self.model}: {el:.1f}s", flush=True)
        return response.content[0].text

    def translate_regions(
        self, original, annotated, num_regions, target_lang="English"
    ) -> Dict[int, dict]:
        prompt = prompts.region_translate_prompt(
            target_lang, num_regions, self.style, self.source_lang,
            self.translate_sfx, self.webtoon)
        text = self._ask(
            [self._image_block(original), self._image_block(annotated), {"type": "text", "text": prompt}]
        )
        return _to_region_dict(prompts.extract_json_array(text))

    def smart_detect_and_translate(self, image, target_lang="English") -> List[dict]:
        prompt = prompts.smart_detect_prompt(
            target_lang, self.style, self.source_lang, self.translate_sfx,
            self.webtoon)
        text = self._ask([self._image_block(image), {"type": "text", "text": prompt}])
        return prompts.extract_json_array(text)

    def translate_texts(self, id_to_text: dict, target_lang="English",
                        image=None) -> Dict[int, dict]:
        prompt = prompts.text_translate_prompt(
            target_lang, self.style, self.source_lang, self.translate_sfx,
            with_image=image is not None, webtoon=self.webtoon)
        payload = json.dumps(id_to_text, ensure_ascii=False)
        content = []
        if image is not None:
            content.append(self._image_block(image))  # panel context for the AI
        content.append({"type": "text", "text": prompt + "\n\n" + payload})
        text = self._ask(content)
        return _to_region_dict(prompts.extract_json_array(text))

    def detect_free_text(self, image, target_lang="English", bubble_ids=None) -> List[dict]:
        prompt = prompts.free_text_detect_prompt(
            target_lang, bubble_ids or [], self.style, self.source_lang,
            self.translate_sfx, self.webtoon)
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
                 source_lang: str = "Japanese", translate_sfx: bool = False,
                 webtoon: bool = False):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.style = (style or "").strip()
        self.source_lang = source_lang or "Japanese"
        self.translate_sfx = bool(translate_sfx)
        # Vertical-scroll webtoon/manhwa: changes the reading-order guidance
        # in every prompt (down the strip, left-to-right — not manga order).
        self.webtoon = bool(webtoon)
        # Optional callback(seconds) fired while a call is in flight,
        # so the UI can show live elapsed time instead of freezing.
        self.on_wait = None

    def _image_part(self, image: np.ndarray) -> dict:
        # Send the page big enough that Gemini can READ it rather than relying
        # on the OCR text we also pass.
        return {"inlineData": {"mimeType": API_IMAGE_MEDIA_TYPE,
                               "data": _encode_image_b64(
                                   image, max_edge=GEMINI_MAX_IMAGE_EDGE)}}

    def _ask(self, parts: list) -> str:
        def _body(budget) -> dict:
            # Full output headroom on EVERY attempt: a busy page's detection
            # JSON alone can overrun 16k and arrive truncated mid-array (and
            # thinking models additionally spend reasoning tokens from this
            # same budget). Tokens are billed as used, so the high cap only
            # costs anything when a page genuinely needs it.
            gc = {"temperature": 0.2, "maxOutputTokens": 65536}
            if budget is not None:
                # Gemini 2.5 flash/flash-lite spend "thinking" tokens from the
                # SAME output budget — a busy page can burn it all and return no
                # text (finishReason=MAX_TOKENS). Budget 0 turns thinking off
                # (cheapest); a small positive budget caps it on thinking-only
                # models that reject 0 — WITHOUT a cap they can grind for
                # minutes on a full-page smart-detect call, which reads as the
                # whole app being frozen.
                gc["thinkingConfig"] = {"thinkingBudget": int(budget)}
            return {
                "contents": [{"parts": parts}],
                "generationConfig": gc,
                # Manga action pages trip Gemini's default safety filters
                # (finishReason=PROHIBITED_CONTENT on a fight scene, which
                # silently killed the free-text pass). This is published
                # commercial fiction being translated — turn the filters off.
                "safetySettings": [
                    {"category": c, "threshold": "BLOCK_NONE"}
                    for c in (
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    )
                ],
            }

        url = self.URL.format(model=self.model)
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        t0 = time.time()
        try:
            with _heartbeat(self.model, getattr(self, "on_wait", None)), \
                    httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=_body(0))
                # Thinking-only models (gemini-2.5-pro, Gemini 3, and the
                # -latest aliases that point at them) reject thinkingBudget:0
                # — sometimes with a specific "thinking budget" message, on
                # newer models just a generic 400 "invalid argument". Retry
                # with a SMALL budget first (keeps the answer fast); only if
                # that is also rejected fall back to no thinkingConfig at all
                # (model default — unbounded thinking, the slowest path).
                if resp.status_code == 400:
                    resp = client.post(url, headers=headers, json=_body(1024))
                if resp.status_code == 400:
                    resp = client.post(url, headers=headers, json=_body(None))
        except httpx.TimeoutException:
            print(f"[gemini] {self.model}: TIMED OUT after "
                  f"{time.time() - t0:.0f}s", flush=True)
            raise RuntimeError(
                f"Gemini ({self.model}) timed out after {int(self.timeout)}s — "
                "the model may be overloaded; try again or pick a faster model")
        except httpx.TransportError as e:
            # DNS failure, dropped Wi-Fi, VPN flap. Used to escape as a raw
            # httpx traceback ("Temporary failure in name resolution"); say
            # plainly that it's the connection, not the page or the key.
            print(f"[gemini] network error after {time.time() - t0:.0f}s: {e}",
                  flush=True)
            raise RuntimeError(
                "Can't reach the Gemini API — check your internet connection "
                f"({type(e).__name__}). Nothing was charged; retry when you're "
                "back online, or use the Offline engine.")
        el = time.time() - t0
        if el >= 5 or resp.status_code != 200:
            print(f"[gemini] {self.model}: {el:.1f}s "
                  f"(HTTP {resp.status_code})", flush=True)

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
            target_lang, num_regions, self.style, self.source_lang,
            self.translate_sfx, self.webtoon)
        text = self._ask(
            [{"text": prompt}, self._image_part(original), self._image_part(annotated)]
        )
        return _to_region_dict(prompts.extract_json_array(text))

    def smart_detect_and_translate(self, image, target_lang="English") -> List[dict]:
        prompt = prompts.smart_detect_prompt(
            target_lang, self.style, self.source_lang, self.translate_sfx,
            self.webtoon)
        text = self._ask([{"text": prompt}, self._image_part(image)])
        return prompts.extract_json_array(text)

    def translate_texts(self, id_to_text: dict, target_lang="English",
                        image=None) -> Dict[int, dict]:
        prompt = prompts.text_translate_prompt(
            target_lang, self.style, self.source_lang, self.translate_sfx,
            with_image=image is not None, webtoon=self.webtoon)
        payload = json.dumps(id_to_text, ensure_ascii=False)
        parts = []
        if image is not None:
            parts.append(self._image_part(image))  # panel context for the AI
        parts.append({"text": prompt + "\n\n" + payload})
        text = self._ask(parts)
        return _to_region_dict(prompts.extract_json_array(text))

    def detect_free_text(self, image, target_lang="English", bubble_ids=None) -> List[dict]:
        prompt = prompts.free_text_detect_prompt(
            target_lang, bubble_ids or [], self.style, self.source_lang,
            self.translate_sfx, self.webtoon)
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


class LocalTranslator:
    """Offline backend: the downloaded model translates text on your own GPU.

    It handles the text paths (which is all the standard pipeline needs:
    balloons are found by the segmentation model and read by manga-ocr, so
    only the READING has to be translated). The vision-only paths — Smart
    Detection and the AI free-text finder — have no local equivalent, so they
    report as unavailable and the pipeline uses its local detectors instead.
    """

    def __init__(self, api_key: str = "", model: str = "", style: str = "",
                 source_lang: str = "Japanese", translate_sfx: bool = False,
                 webtoon: bool = False):
        self.model = model or "offline"
        self.style = (style or "").strip()
        self.source_lang = source_lang or "Japanese"
        self.translate_sfx = bool(translate_sfx)
        self.webtoon = bool(webtoon)
        self.on_wait = None

    # -- the one that matters: bubble readings -> English ----------------
    def translate_texts(self, id_to_text: dict, target_lang="English",
                        image=None) -> Dict[int, dict]:
        from . import local_mt
        mt = local_mt.get(self.source_lang)
        if mt is None:
            why = getattr(local_mt.LocalMT, "last_error", "") or (
                "the model isn't downloaded yet")
            raise RuntimeError(f"Offline translation is unavailable — {why}")
        keys = list(id_to_text.keys())
        outs = mt.translate_many([str(id_to_text[k]) for k in keys])
        result = {}
        for k, tr in zip(keys, outs):
            try:
                rid = int(k)
            except (TypeError, ValueError):
                continue
            result[rid] = {"original": str(id_to_text[k]),
                           "translation": tr, "type": "dialogue"}
        return result

    def translate_crop(self, image, target_lang="English") -> dict:
        """Manual add / point-select: read the crop locally, then translate.
        Keeps the editor's add tools working with no network."""
        from .ocr import MangaOCR, _has_source_text
        from . import local_mt
        original = ""
        try:
            ocr = MangaOCR()
            if ocr.ok:
                import cv2
                padded = cv2.copyMakeBorder(image, 12, 12, 12, 12,
                                            cv2.BORDER_CONSTANT,
                                            value=(255, 255, 255))
                original = ocr.read(padded) or ""
        except Exception as e:
            print(f"[local-mt] crop OCR failed: {e}")
        if not original or not _has_source_text(original, self.source_lang):
            return {"original": original, "translation": ""}
        mt = local_mt.get(self.source_lang)
        if mt is None:
            return {"original": original, "translation": ""}
        return {"original": original, "translation": mt.translate_one(original)}

    #: the pipeline checks this before falling back to a vision path
    has_vision = False

    # -- vision-only paths: no local equivalent --------------------------
    def _no_vision(self, what):
        raise RuntimeError(
            f"{what} needs a vision model. Offline mode uses the local "
            f"balloon detector and manga-ocr instead — leave Smart Detection "
            f"off, or switch the engine to Gemini/Claude.")

    def smart_detect_and_translate(self, image, target_lang="English") -> List[dict]:
        self._no_vision("Smart Detection")

    def detect_free_text(self, image, target_lang="English", bubble_ids=None) -> List[dict]:
        return []            # CRAFT + manga-ocr cover this locally

    def translate_regions(self, original, annotated, num_regions,
                          target_lang="English") -> Dict[int, dict]:
        self._no_vision("Annotated-page translation")

    def analyze_pages(self, images, target_lang="English") -> dict:
        self._no_vision("Style training")


def make_translator(provider: str, api_key: str, model: str = "", style: str = "",
                    source_lang: str = "Japanese", translate_sfx: bool = False,
                    webtoon: bool = False):
    provider = (provider or "claude").lower().strip()
    model = (model or "").strip()

    if provider in ("claude", "anthropic"):
        # Guard against a stale Gemini model id being sent for Claude.
        if not model or model.startswith("gemini"):
            model = DEFAULT_MODELS["claude"]
        return ClaudeTranslator(api_key, model, style=style,
                                source_lang=source_lang, translate_sfx=translate_sfx,
                                webtoon=webtoon)

    if provider in ("local", "offline"):
        return LocalTranslator(api_key, model, style=style,
                               source_lang=source_lang,
                               translate_sfx=translate_sfx, webtoon=webtoon)

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
                                source_lang=source_lang, translate_sfx=translate_sfx,
                                webtoon=webtoon)

    raise ValueError(f"Unknown translation provider: {provider!r}")
