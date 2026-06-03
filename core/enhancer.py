import base64
import time
import cv2
import numpy as np
import httpx

# Status codes worth retrying: rate limits and transient server errors.
_RETRY_CODES = {429, 500, 502, 503, 504}


def _post_with_retry(client, url, *, headers, json=None, data=None, files=None,
                     attempts=3):
    """POST with exponential backoff on rate-limits / transient 5xx errors, so
    one throttled page in a bulk run doesn't silently drop to the local
    fallback. Returns the final response (caller checks status_code)."""
    resp = None
    for i in range(attempts):
        resp = client.post(url, headers=headers, json=json, data=data, files=files)
        if resp.status_code not in _RETRY_CODES:
            return resp
        if i < attempts - 1:
            wait = 2 ** i  # 1s, 2s, 4s
            print(f"[enhance] {resp.status_code} from API, retrying in {wait}s "
                  f"({i + 1}/{attempts})")
            time.sleep(wait)
    return resp


class ImageEnhancer:
    """Convert a rough/sketch manga page into a clean 'scanned' page using
    an external image-to-image model (OpenAI gpt-image-1 or Google Gemini)."""

    DEFAULT_PROMPT = (
        "Convert this manga page photo into a clean, professional "
        "black-and-white printed manga scan, as if scanned flat on a "
        "high-quality scanner. Make the paper pure white with no shadows, "
        "no page curvature, no photo lighting, and no background around the "
        "page. Crisp solid-black inked line art, smooth even screentones, "
        "deep solid blacks, sharp high-contrast print quality. Straighten "
        "the page so it is perfectly rectangular and front-on. Keep the exact "
        "same composition, panel layout, characters, poses, perspective, and "
        "all existing text and Japanese lettering unchanged. Do not add, "
        "remove, redraw, or rearrange panels, characters, or text."
    )

    OPENAI_URL = "https://api.openai.com/v1/images/edits"
    GEMINI_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    DEFAULT_MODELS = {
        "openai": "gpt-image-1",
        "gemini": "gemini-2.5-flash-image",
    }

    def __init__(self, timeout: float = 240.0):
        self.timeout = timeout

    def enhance(
        self,
        image: np.ndarray,
        prompt: str,
        provider: str,
        api_key: str,
        model: str = "",
    ) -> np.ndarray:
        if not api_key:
            raise ValueError("An API key is required for image enhancement")

        prompt = (prompt or "").strip() or self.DEFAULT_PROMPT
        provider = (provider or "").lower().strip()
        model = (model or "").strip() or self.DEFAULT_MODELS.get(provider, "")

        if provider == "openai":
            return self._openai(image, prompt, api_key, model)
        if provider == "gemini":
            return self._gemini(image, prompt, api_key, model)
        raise ValueError(f"Unknown enhancement provider: {provider!r}")

    # ── Encoding helpers ──
    def _encode_png(self, image: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Failed to encode source image")
        return buf.tobytes()

    def _decode(self, data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("The model returned data that is not a valid image")
        return img

    # ── OpenAI (ChatGPT) gpt-image-1 ──
    def _openai(self, image, prompt, api_key, model) -> np.ndarray:
        png = self._encode_png(image)
        files = {"image": ("page.png", png, "image/png")}
        data = {"model": model, "prompt": prompt, "n": "1", "size": "auto"}
        headers = {"Authorization": f"Bearer {api_key}"}

        with httpx.Client(timeout=self.timeout) as client:
            resp = _post_with_retry(client, self.OPENAI_URL, headers=headers,
                                    data=data, files=files)

        if resp.status_code != 200:
            raise RuntimeError(self._err("OpenAI", resp))

        payload = resp.json()
        items = payload.get("data") or []
        if not items or "b64_json" not in items[0]:
            raise RuntimeError(f"OpenAI returned no image: {str(payload)[:300]}")
        return self._decode(base64.b64decode(items[0]["b64_json"]))

    # ── Google Gemini (2.5 Flash Image / "Nano Banana") ──
    def _gemini(self, image, prompt, api_key, model) -> np.ndarray:
        h, w = image.shape[:2]
        max_dim = 2048
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Failed to encode image for Gemini")
        b64 = base64.b64encode(buf.tobytes()).decode()
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                    ]
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        url = self.GEMINI_URL.format(model=model)
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

        img_kb = len(b64) * 3 // 4 // 1024
        print(f"[enhance] Gemini request: model={model}, image={img_kb}KB")

        with httpx.Client(timeout=self.timeout) as client:
            resp = _post_with_retry(client, url, headers=headers, json=body)

        print(f"[enhance] Gemini response: {resp.status_code}")
        if resp.status_code != 200:
            raise RuntimeError(self._err("Gemini", resp))

        payload = resp.json()
        for cand in payload.get("candidates", []):
            parts = cand.get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return self._decode(base64.b64decode(inline["data"]))
        raise RuntimeError(f"Gemini returned no image: {str(payload)[:300]}")

    def _err(self, name: str, resp: httpx.Response) -> str:
        try:
            data = resp.json()
            msg = data.get("error", {})
            msg = msg.get("message") if isinstance(msg, dict) else msg
            if msg:
                return f"{name} error {resp.status_code}: {msg}"
        except Exception:
            pass
        return f"{name} error {resp.status_code}: {resp.text[:300]}"
