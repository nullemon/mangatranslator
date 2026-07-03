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
    def _low_ink_line(profile: np.ndarray, center: int, band: int) -> int:
        """Index near `center` (within ±band) with the least ink — a panel gutter
        or plain area — so a tile seam placed there is invisible."""
        lo = max(1, center - band)
        hi = min(len(profile) - 1, center + band)
        if hi <= lo:
            return center
        return lo + int(np.argmin(profile[lo:hi]))

    @staticmethod
    def _dp_seam(cost: np.ndarray) -> np.ndarray:
        """Least-cost top→bottom path through a cost map (H x W): one x per row,
        moving at most 1 px sideways per row (classic seam-carving DP)."""
        H, W = cost.shape
        dp = cost.astype(np.float32).copy()
        INF = np.float32(1e9)
        for y in range(1, H):
            prev = dp[y - 1]
            left = np.concatenate(([INF], prev[:-1]))
            right = np.concatenate((prev[1:], [INF]))
            dp[y] += np.minimum(prev, np.minimum(left, right))
        xs = np.empty(H, np.int32)
        xs[-1] = int(np.argmin(dp[-1]))
        for y in range(H - 2, -1, -1):
            x = xs[y + 1]
            lo, hi = max(0, x - 1), min(W, x + 2)
            xs[y] = lo + int(np.argmin(dp[y, lo:hi]))
        return xs

    def _stitch(self, A: np.ndarray, B: np.ndarray, band: int, axis: int) -> np.ndarray:
        """Join two enhanced pieces whose facing edges both contain a rendering
        of the same `band`-px strip. Instead of a straight cut, trace the
        LEAST-VISIBLE seam through the strip — a path that prefers pixels where
        the two renderings agree (white gaps, matching fills) and avoids ink —
        so an eye or a text glyph the tiles drew differently comes wholly from
        ONE tile rather than being sliced."""
        if axis == 0:   # horizontal seam = transpose, do the vertical case, undo
            return np.transpose(
                self._stitch(np.ascontiguousarray(np.transpose(A, (1, 0, 2))),
                             np.ascontiguousarray(np.transpose(B, (1, 0, 2))),
                             band, 1), (1, 0, 2)).copy()
        band = int(min(band, A.shape[1] - 1, B.shape[1] - 1))
        if band < 4:
            return np.concatenate([A, B[:, band:]], axis=1)
        bandA = np.ascontiguousarray(A[:, A.shape[1] - band:])
        bandB = np.ascontiguousarray(B[:, :band])
        gA = cv2.cvtColor(bandA, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gB = cv2.cvtColor(bandB, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(gA - gB)                    # disagreement between the tiles
        ink = 255.0 - np.minimum(gA, gB)          # darkness (prefer paper gaps)
        seam = self._dp_seam(diff + 0.15 * ink)
        take_a = np.arange(band)[None, :] < seam[:, None]
        merged = np.where(take_a[..., None], bandA, bandB)
        return np.concatenate([A[:, :A.shape[1] - band], merged, B[:, band:]], axis=1)

    def enhance_tiled(self, image, prompt, provider, api_key, model,
                      tiles: int = 2, out_scale: float = 2.0,
                      progress=None) -> np.ndarray:
        """Beat the model's ~2K cap: split the page into `tiles` pieces, AI-scan
        EACH at full quality (so each gets the model's whole resolution budget),
        then merge into one high-res page. Splits are snapped to panel gutters,
        every tile is sent WITH an overlap band, and neighbouring tiles are
        joined along a least-visible seam traced through that band — so nothing
        gets cut through a face, an eye, or a line of text.
        tiles=2 splits along the long axis; tiles=4 is 2x2."""
        h, w = image.shape[:2]
        if tiles < 2:
            return self.enhance(image, prompt, provider, api_key, model)
        n = max(2, min(int(tiles), 4))
        # Beta 2 and Beta 4 use the layouts of the build that generated best
        # (hard-boundary era): 2 = halves along the long axis, 4 = 2x2 quarters
        # (square-ish crops the model handles well). Beta 3 = 3 strips across
        # the long axis (no mid-panel vertical seam), per user request.
        if n == 4:
            rows, cols = 2, 2
        elif n == 3:
            rows, cols = (3, 1) if h >= w else (1, 3)
        else:
            rows, cols = (2, 1) if h >= w else (1, 2)

        # Each tile generates SOLO — same prompt wording as the build whose
        # content was right. The only post-work is the seam join.
        tile_prompt = (prompt or self.DEFAULT_PROMPT).strip() + (
            " This is ONE tile of a larger page — keep the EXACT same framing, "
            "crop and proportions, edge to edge; do not add borders, zoom, or "
            "shift anything, so tiles line up seamlessly.")
        ov = max(16, int(min(h, w) * 0.08))     # shared band the seam can roam in
        S = int(round(max(1.0, out_scale)))     # integer scale → exact geometry
        if max(h, w) >= 2600:
            # A very high-res upload already exceeds the model's ~2K output tier;
            # inflating its returns 2x just magnifies softness. Merge at 1:1 and
            # let MangaJaNai (HD toggle) do the true upscaling afterwards.
            S = 1

        # Prior: put each split on the lowest-ink line near its midpoint (a panel
        # gutter) so the seam usually has an easy home to begin with.
        g = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2GRAY)
        dark = (g < 110).astype(np.int32)
        band = max(6, int(min(h, w) * 0.09))
        ys = [0] + [self._low_ink_line(dark.sum(1), r * h // rows, band)
                    for r in range(1, rows)] + [h]
        xs = [0] + [self._low_ink_line(dark.sum(0), c * w // cols, band)
                    for c in range(1, cols)] + [w]

        n, total = 0, rows * cols
        strips = []
        for r in range(rows):
            ey0 = max(0, ys[r] - ov) if r > 0 else 0
            ey1 = min(h, ys[r + 1] + ov) if r < rows - 1 else h
            pieces = []
            for c in range(cols):
                ex0 = max(0, xs[c] - ov) if c > 0 else 0
                ex1 = min(w, xs[c + 1] + ov) if c < cols - 1 else w
                if progress:
                    progress(n, total)
                n += 1
                crop = image[ey0:ey1, ex0:ex1]
                ch, cw = crop.shape[:2]
                # The model only supports aspect ratios up to 2:1 — a thinner
                # strip gets cropped/squashed. Pad the short side with white
                # paper to 1.9:1 before sending, cut the padding back off after.
                if cw >= ch and cw > 1.9 * ch:
                    crop = cv2.copyMakeBorder(crop, 0, int(np.ceil(cw / 1.9)) - ch,
                                              0, 0, cv2.BORDER_CONSTANT,
                                              value=(255, 255, 255))
                elif ch > cw and ch > 1.9 * cw:
                    crop = cv2.copyMakeBorder(crop, 0, 0, 0,
                                              int(np.ceil(ch / 1.9)) - cw,
                                              cv2.BORDER_CONSTANT,
                                              value=(255, 255, 255))
                enh = self.enhance(crop, tile_prompt, provider, api_key, model)
                pw, ph = crop.shape[1] * S, crop.shape[0] * S
                interp = cv2.INTER_AREA if enh.shape[0] > ph else cv2.INTER_CUBIC
                enh = cv2.resize(enh, (pw, ph), interpolation=interp)
                # Drop the padding → exactly the strip's true region, 1:1.
                pieces.append(np.ascontiguousarray(enh[:(ey1 - ey0) * S,
                                                       :(ex1 - ex0) * S]))
            strip = pieces[0]
            for c in range(1, cols):
                shared = (min(w, xs[c] + ov) - max(0, xs[c] - ov)) * S
                strip = self._stitch(strip, pieces[c], shared, axis=1)
            strips.append(strip)
        out = strips[0]
        for r in range(1, rows):
            shared = (min(h, ys[r] + ov) - max(0, ys[r] - ov)) * S
            out = self._stitch(out, strips[r], shared, axis=0)
        return np.ascontiguousarray(out)

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
            # The API defaults to the 1K tier (we measured a 1424x720 return —
            # that was the mush). Always request the 2K tier.
            "resolution": "2k",
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        print(f"[enhance] xAI request: POST {self.XAI_URL} | model={model} | "
              f"input={w}x{h} | resolution=2k | prompt[:80]={prompt[:80]!r}")
        with httpx.Client(timeout=self.timeout) as client:
            resp = _post_with_retry(client, self.XAI_URL, headers=headers, json=body)
            if resp.status_code == 422 and "resolution" in resp.text.lower():
                # Account/tier that doesn't accept the param — retry without.
                print("[enhance] xAI rejected 'resolution'; retrying without")
                body.pop("resolution", None)
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
