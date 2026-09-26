"""A stand-in for the image-generation APIs the Scan workflows call, so
Raw -> Scan, AI Scan -> HD and the scan half of Raw -> Scan -> Translate can
be driven end to end without a key.

Run it, then start the app pointing both providers at it:

    python tests/ui/stub_image_api.py 8141 &
    GEMINI_BASE_URL=http://127.0.0.1:8141 XAI_BASE_URL=http://127.0.0.1:8141 \\
        OPENAI_BASE_URL=http://127.0.0.1:8141 PORT=8041 python app.py

It answers in each provider's real response shape:

  POST /v1beta/models/<model>:generateContent   Gemini: candidates[].content
       .parts[].inlineData (PNG, base64), snapped to Gemini's aspect buckets
       (~1 megapixel: 832x1248 for a 2:3 page, ...) the way the real model does.
  POST /v1/images/edits  (JSON body)             xAI Grok: data[].b64_json,
       snapped to a 2048px grid size of the nearest standard ratio — so a
       page comes back with the wrong aspect, as it really does, and the
       app's aspect restore is exercised.
  POST /v1/images/edits  (multipart)             OpenAI: data[].b64_json.

The returned picture is the input, slightly LIGHTENED (ink 0 -> 40, paper
stays 255), never rotated or mirrored, so a test can check orientation and
that the page is not blank.

The MODEL name picks a behaviour, which the UI can set through the Model box:

  (anything else)   a normal answer
  stub-refuse       Gemini: finishReason IMAGE_SAFETY, no image.
                    xAI: 400 "Generated image rejected by content moderation."
  stub-block        Gemini: promptFeedback.blockReason PROHIBITED_CONTENT.
  stub-badkey       400/401 "API key not valid" in the provider's error shape.
  stub-noimage      200 with a text-only reply.
  stub-oversize     an image 3x the size the real API would return.
  stub-url          xAI only: data[].url pointing back at this server.
  stub-429          429 on the first call of each run, then a normal answer.

GET /__log returns every request seen as JSON (url, model, prompt, key,
decoded input size, returned size); GET /__reset clears it.
"""
import base64
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

LOG = []
_LOCK = threading.Lock()
_BLOBS = {}
_SEEN_429 = set()

GEMINI_BUCKETS = [(1024, 1024), (832, 1248), (1248, 832), (864, 1184),
                  (1184, 864), (896, 1152), (1152, 896), (768, 1344),
                  (1344, 768), (1536, 672)]
GROK_RATIOS = [(1, 1), (3, 4), (4, 3), (9, 16), (16, 9), (2, 3), (3, 2),
               (1, 2), (2, 1)]


def lighten(img):
    out = img.astype(np.float32) * (215.0 / 255.0) + 40.0
    return out.clip(0, 255).astype(np.uint8)


def gemini_size(w, h):
    a = w / h
    return min(GEMINI_BUCKETS, key=lambda s: abs(np.log((s[0] / s[1]) / a)))


def grok_size(w, h, long_edge=2048):
    a = w / h
    rw, rh = min(GROK_RATIOS, key=lambda r: abs(np.log((r[0] / r[1]) / a)))
    if rw >= rh:
        W, H = long_edge, long_edge * rh / rw
    else:
        W, H = long_edge * rw / rh, long_edge
    return int(W) // 16 * 16, int(H) // 16 * 16


def png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def decode_b64(data):
    arr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/__log"):
            with _LOCK:
                return self._send(200, LOG)
        if self.path.startswith("/__reset"):
            with _LOCK:
                LOG.clear()
                _SEEN_429.clear()
            return self._send(200, {"ok": True})
        m = re.match(r"/blob/(\w+)\.png", self.path)
        if m and m.group(1) in _BLOBS:
            return self._send(200, _BLOBS[m.group(1)], "image/png")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        m = re.match(r"/v1beta/models/([^:]+):generateContent", self.path)
        if m:
            return self._gemini(m.group(1), raw)
        if self.path.startswith("/v1/images/edits"):
            ctype = self.headers.get("Content-Type", "")
            if ctype.startswith("application/json"):
                return self._xai(raw)
            return self._openai(raw, ctype)
        self._send(404, {"error": {"message": f"no route {self.path}"}})

    # ── helpers ──
    def _log(self, **kw):
        with _LOCK:
            LOG.append(kw)

    def _first_429(self, model, key):
        if model != "stub-429":
            return False
        with _LOCK:
            if key in _SEEN_429:
                return False
            _SEEN_429.add(key)
            return True

    # ── Gemini generateContent ──
    def _gemini(self, model, raw):
        body = json.loads(raw or b"{}")
        parts = (body.get("contents") or [{}])[0].get("parts", [])
        prompt = "".join(p.get("text", "") for p in parts)
        img = None
        for p in parts:
            inline = p.get("inlineData") or p.get("inline_data")
            if inline:
                img = decode_b64(inline["data"])
        key = self.headers.get("x-goog-api-key", "")
        entry = dict(api="gemini", model=model, prompt=prompt, key=key,
                     sent=[int(img.shape[1]), int(img.shape[0])] if img is not None else None,
                     modalities=body.get("generationConfig", {}).get("responseModalities"))
        if self._first_429(model, "gemini"):
            entry["status"] = 429
            self._log(**entry)
            return self._send(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                              "message": "Resource has been exhausted (e.g. check quota)."}})
        if model == "stub-badkey":
            entry["status"] = 400
            self._log(**entry)
            return self._send(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                              "message": "API key not valid. Please pass a valid API key."}})
        if model == "stub-refuse":
            entry["status"] = 200
            self._log(**entry)
            return self._send(200, {"candidates": [{"finishReason": "IMAGE_SAFETY", "index": 0}],
                                    "modelVersion": model})
        if model == "stub-block":
            entry["status"] = 200
            self._log(**entry)
            return self._send(200, {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
                                    "modelVersion": model})
        if model == "stub-noimage" or img is None:
            entry["status"] = 200
            self._log(**entry)
            return self._send(200, {"candidates": [{"content": {"role": "model", "parts": [
                {"text": "I can describe the page, but I can't produce an image of it."}]},
                "finishReason": "STOP", "index": 0}], "modelVersion": model})
        w, h = gemini_size(img.shape[1], img.shape[0])
        if model == "stub-oversize":
            w, h = w * 3, h * 3
        out = cv2.resize(lighten(img), (w, h), interpolation=cv2.INTER_AREA)
        entry.update(status=200, returned=[w, h])
        self._log(**entry)
        self._send(200, {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "Here is the cleaned scan."},
            {"inlineData": {"mimeType": "image/png", "data": png_b64(out)}}]},
            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1290, "candidatesTokenCount": 1290},
            "modelVersion": model})

    # ── xAI images/edits ──
    def _xai(self, raw):
        body = json.loads(raw or b"{}")
        model = body.get("model", "")
        url = ((body.get("image") or {}).get("url") or "")
        img = decode_b64(url.split(",", 1)[1]) if url.startswith("data:") else None
        key = self.headers.get("Authorization", "")
        entry = dict(api="xai", model=model, prompt=body.get("prompt", ""), key=key,
                     sent=[int(img.shape[1]), int(img.shape[0])] if img is not None else None,
                     resolution=body.get("resolution"),
                     response_format=body.get("response_format"))
        if self._first_429(model, "xai"):
            entry["status"] = 429
            self._log(**entry)
            return self._send(429, {"code": "Too many requests", "error": "Rate limit exceeded"})
        if model == "stub-badkey":
            entry["status"] = 401
            self._log(**entry)
            return self._send(401, {"code": "Client specified an invalid argument",
                                    "error": "Incorrect API key provided: xa***ey."})
        if model == "stub-refuse":
            entry["status"] = 400
            self._log(**entry)
            return self._send(400, {"code": "Client specified an invalid argument",
                                    "error": "Generated image rejected by content moderation."})
        if model == "stub-noimage" or img is None:
            entry["status"] = 200
            self._log(**entry)
            return self._send(200, {"data": []})
        w, h = grok_size(img.shape[1], img.shape[0])
        if model == "stub-oversize":
            w, h = w * 3, h * 3
        out = cv2.resize(lighten(img), (w, h), interpolation=cv2.INTER_CUBIC)
        entry.update(status=200, returned=[w, h])
        self._log(**entry)
        if model == "stub-url":
            bid = f"b{len(_BLOBS)}"
            ok, buf = cv2.imencode(".png", out)
            _BLOBS[bid] = buf.tobytes()
            host = self.headers.get("Host", "127.0.0.1")
            item = {"url": f"http://{host}/blob/{bid}.png", "revised_prompt": ""}
        else:
            item = {"b64_json": png_b64(out), "revised_prompt": ""}
        self._send(200, {"data": [item]})

    # ── OpenAI images/edits (multipart) ──
    def _openai(self, raw, ctype):
        m = re.search(r"boundary=(.+)", ctype)
        fields, img = {}, None
        if m:
            bnd = m.group(1).strip('"').encode()
            for part in raw.split(b"--" + bnd):
                head, _, data = part.partition(b"\r\n\r\n")
                name = re.search(rb'name="([^"]+)"', head)
                if not name:
                    continue
                data = data[:-2] if data.endswith(b"\r\n") else data
                if b"filename=" in head:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    fields[name.group(1).decode()] = data.decode(errors="replace")
        model = fields.get("model", "")
        entry = dict(api="openai", model=model, prompt=fields.get("prompt", ""),
                     key=self.headers.get("Authorization", ""),
                     sent=[int(img.shape[1]), int(img.shape[0])] if img is not None else None)
        if model == "stub-badkey":
            entry["status"] = 401
            self._log(**entry)
            return self._send(401, {"error": {"message": "Incorrect API key provided.",
                                              "type": "invalid_request_error"}})
        if model == "stub-refuse":
            entry["status"] = 400
            self._log(**entry)
            return self._send(400, {"error": {"message": "Your request was rejected by the safety system.",
                                              "type": "image_generation_user_error",
                                              "code": "moderation_blocked"}})
        if img is None:
            entry["status"] = 400
            self._log(**entry)
            return self._send(400, {"error": {"message": "image is required"}})
        h0, w0 = img.shape[:2]
        w, h = (1024, 1536) if h0 > w0 else (1536, 1024) if w0 > h0 else (1024, 1024)
        out = cv2.resize(lighten(img), (w, h), interpolation=cv2.INTER_AREA)
        entry.update(status=200, returned=[w, h])
        self._log(**entry)
        self._send(200, {"created": 0, "data": [{"b64_json": png_b64(out)}]})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8141
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
