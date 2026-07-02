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
        "Restore this into a clean TCB-style black-and-white manga scan: pure "
        "white paper, solid black ink, sharp crisp lines, flattened and "
        "straightened, no creases or shadows. Keep all artwork, screentones and "
        "Japanese text exactly as drawn."
    )

    OPENAI_URL = "https://api.openai.com/v1/images/edits"
    GEMINI_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    XAI_URL = "https://api.x.ai/v1/images/edits"

    DEFAULT_MODELS = {
        "openai": "gpt-image-1",
        "gemini": "gemini-2.5-flash-image",
        "xai": "grok-imagine-image-quality",
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
        if provider == "grok":
            provider = "xai"
        model = (model or "").strip() or self.DEFAULT_MODELS.get(provider, "")

        if provider == "openai":
            return self._openai(image, prompt, api_key, model)
        if provider == "gemini":
            return self._gemini(image, prompt, api_key, model)
        if provider == "xai":
            return self._xai(image, prompt, api_key, model)
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

    @staticmethod
    def _feather(h: int, w: int, ramp: int) -> np.ndarray:
        """Blend weight: ~1 in the centre, ramps down toward every edge over
        `ramp` px, so overlapping tiles cross-fade instead of showing a seam."""
        ramp = max(1, min(ramp, h // 2, w // 2))
        ry = np.ones(h, np.float32); rx = np.ones(w, np.float32)
        r = np.linspace(0.02, 1.0, ramp, dtype=np.float32)
        ry[:ramp] = r; ry[-ramp:] = r[::-1]
        rx[:ramp] = r; rx[-ramp:] = r[::-1]
        return np.outer(ry, rx)

    def enhance_tiled(self, image, prompt, provider, api_key, model,
                      tiles: int = 2, out_scale: float = 2.0,
                      progress=None) -> np.ndarray:
        """Beat the model's ~2K cap: split the page into `tiles` pieces, AI-scan
        EACH at full quality (so each gets the model's whole resolution budget),
        then merge into one high-res page. Each enhanced tile is forced back to
        its exact tile shape (1:1) so the pieces line up; overlaps are feathered
        so seams disappear. tiles=2 splits along the long axis; tiles=4 is 2x2."""
        h, w = image.shape[:2]
        if tiles >= 4:
            rows, cols = 2, 2
        elif tiles == 2:
            rows, cols = (2, 1) if h >= w else (1, 2)
        else:
            return self.enhance(image, prompt, provider, api_key, model)

        tile_prompt = (prompt or self.DEFAULT_PROMPT).strip() + (
            " This is ONE tile of a larger page — keep the EXACT same framing, "
            "crop and proportions, edge to edge; do not add borders, zoom, or "
            "shift anything, so tiles line up seamlessly.")
        ov = max(8, int(min(h, w) * 0.05))          # overlap between tiles
        S = max(1.0, float(out_scale))
        OH, OW = int(round(h * S)), int(round(w * S))
        acc = np.zeros((OH, OW, 3), np.float32)
        wsum = np.zeros((OH, OW), np.float32)

        n, total = 0, rows * cols
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r * h // rows, (r + 1) * h // rows
                x0, x1 = c * w // cols, (c + 1) * w // cols
                ey0, ey1 = max(0, y0 - ov), min(h, y1 + ov)
                ex0, ex1 = max(0, x0 - ov), min(w, x1 + ov)
                if progress:
                    progress(n, total)
                enh = self.enhance(image[ey0:ey1, ex0:ex1], tile_prompt,
                                   provider, api_key, model)
                th, tw = int(round((ey1 - ey0) * S)), int(round((ex1 - ex0) * S))
                # Force 1:1 back to the tile's shape so it aligns on merge.
                interp = cv2.INTER_AREA if enh.shape[0] > th else cv2.INTER_CUBIC
                enh = cv2.resize(enh, (tw, th), interpolation=interp)
                mask = self._feather(th, tw, int(ov * S))
                oy0, ox0 = int(round(ey0 * S)), int(round(ex0 * S))
                acc[oy0:oy0 + th, ox0:ox0 + tw] += enh.astype(np.float32) * mask[..., None]
                wsum[oy0:oy0 + th, ox0:ox0 + tw] += mask
                n += 1
        wsum[wsum == 0] = 1.0
        return np.clip(acc / wsum[..., None], 0, 255).astype(np.uint8)

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

    # ── xAI (Grok Imagine) image-to-image edit ──
    def _xai(self, image, prompt, api_key, model) -> np.ndarray:
        h, w = image.shape[:2]
        max_dim = 2048
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # PNG (lossless) so line art / screentones aren't softened by JPEG before
        # Grok even sees them — pasting a crisp page in gives a crisp remake.
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Failed to encode image for xAI")
        data_uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
        body = {
            "model": model,
            "prompt": prompt,
            "image": {"url": data_uri, "type": "image_url"},
            "response_format": "b64_json",
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        print(f"[enhance] xAI request: POST {self.XAI_URL} | model={model} | "
              f"input={w}x{h} | prompt[:80]={prompt[:80]!r}")
        with httpx.Client(timeout=self.timeout) as client:
            resp = _post_with_retry(client, self.XAI_URL, headers=headers, json=body)
        print(f"[enhance] xAI response: {resp.status_code}")

        if resp.status_code != 200:
            raise RuntimeError(self._err("xAI", resp))

        payload = resp.json()
        # Diagnostics: what did xAI actually send back? (everything except the
        # huge image blob). 'revised_prompt' tells us whether it used our prompt.
        meta = {k: v for k, v in payload.items() if k != "data"}
        print(f"[enhance] xAI payload meta: {meta}")
        items = payload.get("data") or []
        if items:
            first = items[0]
            print(f"[enhance] xAI data[0] keys: {list(first.keys())}")
            if first.get("revised_prompt"):
                print(f"[enhance] xAI revised_prompt: {str(first['revised_prompt'])[:200]}")
            if first.get("b64_json"):
                outimg = self._decode(base64.b64decode(first["b64_json"]))
                print(f"[enhance] xAI returned image {outimg.shape[1]}x{outimg.shape[0]}")
                return outimg
            if first.get("url"):
                with httpx.Client(timeout=self.timeout) as client:
                    img_resp = client.get(first["url"])
                if img_resp.status_code == 200:
                    outimg = self._decode(img_resp.content)
                    print(f"[enhance] xAI returned image {outimg.shape[1]}x{outimg.shape[0]} (via url)")
                    return outimg
        raise RuntimeError(f"xAI returned no image: {str(payload)[:300]}")

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
