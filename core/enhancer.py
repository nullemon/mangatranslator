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
        refs=None,
    ) -> np.ndarray:
        if not api_key:
            raise ValueError("An API key is required for image enhancement")

        prompt = (prompt or "").strip() or self.DEFAULT_PROMPT
        provider = (provider or "").lower().strip()
        if provider == "grok":
            provider = "xai"
        model = (model or "").strip() or self.DEFAULT_MODELS.get(provider, "")

        if provider == "openai":
            return self._openai(image, prompt, api_key, model)   # refs unsupported
        if provider == "gemini":
            return self._gemini(image, prompt, api_key, model, refs=refs)
        if provider == "xai":
            return self._xai(image, prompt, api_key, model, refs=refs)
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

    def _match_tone(self, A: np.ndarray, B: np.ndarray, band: int, axis: int) -> np.ndarray:
        """Histogram-match B's tone to A's over their shared overlap band — both
        contain a rendering of the SAME source strip, so any level difference
        there is pure model drift. Applying the matched LUT to all of B pulls a
        strip the model rendered lighter/darker (bleached screentone, lifted
        blacks) onto its neighbour's levels BEFORE stitching — the strip-wide
        cure for one half of a panel coming out white and the other grey."""
        band = int(min(band, (A.shape[1] if axis == 1 else A.shape[0]) - 1,
                       (B.shape[1] if axis == 1 else B.shape[0]) - 1))
        if band < 4:
            return B
        bandA = A[:, A.shape[1] - band:] if axis == 1 else A[A.shape[0] - band:]
        bandB = B[:, :band] if axis == 1 else B[:band]
        gA = cv2.cvtColor(np.ascontiguousarray(bandA), cv2.COLOR_BGR2GRAY)
        gB = cv2.cvtColor(np.ascontiguousarray(bandB), cv2.COLOR_BGR2GRAY)
        ca = np.cumsum(np.bincount(gA.ravel(), minlength=256).astype(np.float64))
        cb = np.cumsum(np.bincount(gB.ravel(), minlength=256).astype(np.float64))
        ca /= max(ca[-1], 1.0)
        cb /= max(cb[-1], 1.0)
        lut = np.clip(np.interp(cb, ca, np.arange(256.0)), 0, 255).astype(np.uint8)
        return cv2.LUT(np.ascontiguousarray(B), lut)

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
        n = 4 if tiles >= 4 else 2
        # STRIPS, not a grid: cut only across the long axis, so every seam can
        # sit in a panel-row gutter and a wide panel is never split down the
        # middle (that mid-panel vertical seam was the killer). Portrait page →
        # horizontal strips; landscape double-spread → vertical strips, whose
        # first cut lands on the spine.
        rows, cols = (n, 1) if h >= w else (1, n)

        tile_prompt = (prompt or self.DEFAULT_PROMPT).strip() + (
            " This is ONE tile of a larger page — keep the EXACT same framing, "
            "crop and proportions, edge to edge; do not add borders, zoom, or "
            "shift anything, so tiles line up seamlessly. The FIRST extra image "
            "is the FULL page this tile comes from, and any second extra image "
            "is a neighbouring tile that was already cleaned: match their style, "
            "line weight, tone, contrast and level of detail EXACTLY, so every "
            "tile looks like one consistent scan of the same page.")
        ov = max(16, int(min(h, w) * 0.08))     # shared band the seam can roam in
        S = int(round(max(1.0, out_scale)))     # integer scale → exact geometry
        page_ref = self._shrink(image, 768)     # whole-page style reference

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
        prev_tile = None    # each tile follows the one before it → consistent style
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
                refs = [page_ref] + ([prev_tile] if prev_tile is not None else [])
                try:
                    enh = self.enhance(crop, tile_prompt, provider, api_key, model,
                                       refs=refs)
                except Exception as e:
                    # A provider that rejects multi-image input still works solo.
                    print(f"[enhance] tile refs rejected ({e}); retrying without")
                    enh = self.enhance(crop, tile_prompt, provider, api_key, model)
                prev_tile = self._shrink(enh, 768)
                tw, th = (ex1 - ex0) * S, (ey1 - ey0) * S
                interp = cv2.INTER_AREA if enh.shape[0] > th else cv2.INTER_CUBIC
                pieces.append(cv2.resize(enh, (tw, th), interpolation=interp))
            strip = pieces[0]
            for c in range(1, cols):
                shared = (min(w, xs[c] + ov) - max(0, xs[c] - ov)) * S
                nxt = self._match_tone(strip, pieces[c], shared, axis=1)
                strip = self._stitch(strip, nxt, shared, axis=1)
            strips.append(strip)
        out = strips[0]
        for r in range(1, rows):
            shared = (min(h, ys[r] + ov) - max(0, ys[r] - ov)) * S
            nxt = self._match_tone(out, strips[r], shared, axis=0)
            out = self._stitch(out, nxt, shared, axis=0)
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

    @staticmethod
    def _shrink(img: np.ndarray, target: int = 768) -> np.ndarray:
        """Downscale a reference image so it adds context, not payload."""
        if max(img.shape[:2]) <= target:
            return img
        s = target / max(img.shape[:2])
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # ── Google Gemini (2.5 Flash Image / "Nano Banana") ──
    def _gemini(self, image, prompt, api_key, model, refs=None) -> np.ndarray:
        h, w = image.shape[:2]
        max_dim = 2048
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Failed to encode image for Gemini")
        b64 = base64.b64encode(buf.tobytes()).decode()
        parts = [{"text": prompt},
                 {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]
        # Style/context references (e.g. the full page, the previous tile) so
        # every tile is rendered consistently with the others.
        for ref in (refs or []):
            okr, rbuf = cv2.imencode(".jpg", self._shrink(ref),
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            if okr:
                parts.append({"inlineData": {"mimeType": "image/jpeg",
                                             "data": base64.b64encode(rbuf.tobytes()).decode()}})
        body = {
            "contents": [{"parts": parts}],
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
    def _xai(self, image, prompt, api_key, model, refs=None) -> np.ndarray:
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
        main = {"url": data_uri, "type": "image_url"}
        # xAI multi-image edit takes up to 3 source images — main + 2 references
        # (the full page / the previous tile) so tiles come out style-consistent.
        images = [main]
        for ref in (refs or [])[:2]:
            okr, rbuf = cv2.imencode(".jpg", self._shrink(ref),
                                     [cv2.IMWRITE_JPEG_QUALITY, 88])
            if okr:
                images.append({"url": "data:image/jpeg;base64,"
                                      + base64.b64encode(rbuf.tobytes()).decode(),
                               "type": "image_url"})
        body = {
            "model": model,
            "prompt": prompt,
            "image": images if len(images) > 1 else main,
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
