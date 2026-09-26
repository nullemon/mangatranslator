"""A stand-in for the Claude Messages API, for driving the editor without a key.

Run it, then start the app with ANTHROPIC_BASE_URL pointing at it:

    python tests/ui/stub_llm.py 8022 &
    ANTHROPIC_BASE_URL=http://127.0.0.1:8022 PORT=8021 python app.py

It answers every prompt the pipeline sends with canned, deterministic JSON,
so a page run through the real detector + cleaner comes back with real
translation ITEMS (numbered regions with bboxes) that the on-image tools can
move, resize and edit. Nothing here is a test of translation quality — the
lines are "LINE 1", "LINE 2", ... so a test can tell regions apart in the
rendered image.

Every prompt kind gets an answer of the shape it asks for:

    region translate   (numbered boxes)      -> [{id, original, translation, type, tone}]
    text translate     (id -> OCR text JSON) -> the same, one per id sent
    crop translate     (Add / Point tools)   -> [{original, translation}]
    smart detection    (whole-page vision)   -> [{x_pct, y_pct, width_pct, ...}] placed
                                                on the balloons the page really has
    free-text detect                         -> [] (set STUB_FREE_TEXT=1 for one
                                                narration box at the top of the page)
    profile learning   (STYLE GUIDE)         -> {glossary, honorifics, sfx_policy,
                                                style_guide}

Every line's "tone" is "dialogue"; put STUB-TONES in the Translation Style
box and every third line becomes a "shout", every fourth a "thought".

It also keeps a log of what it was asked, for the tests to read:

    GET  /__log     -> [{"kind", "model", "prompt", "images", "config"}, ...]
    POST /__reset   -> clears the log

`tests/ui/stub_gemini.py` serves the same answers in Gemini's REST shape.
"""
import base64
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = []
_LOG_LOCK = threading.Lock()
LOG_MAX = 400

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def kind_of(prompt_text: str) -> str:
    if re.search(r"There are (\d+) numbered regions", prompt_text):
        return "regions"
    if "mapping each speech-bubble id" in prompt_text:
        return "texts"
    if "CROPPED region" in prompt_text:
        return "crop"
    if "building a STYLE GUIDE" in prompt_text:
        return "profile"
    if "have already been translated" in prompt_text:
        return "free_text"
    if "Carefully examine this manga page" in prompt_text:
        return "smart"
    return "unknown"


def _balloons(image_b64: str):
    """Where the page's balloons really are, as percentage boxes — found with
    the app's own CV balloon detector, so a smart-detection answer lands on
    the bubbles instead of on the art."""
    try:
        import cv2
        import numpy as np
        sys.path.insert(0, ROOT)
        from core.detector import BubbleDetector
        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            return []
        h, w = img.shape[:2]
        out = []
        for r in BubbleDetector().detect(img):
            x, y, bw, bh = r.bbox
            out.append((100.0 * x / w, 100.0 * y / h,
                        100.0 * bw / w, 100.0 * bh / h))
        return out
    except Exception as e:                  # never crash the stub
        print(f"[stub] balloon finder failed: {e}", flush=True)
        return []


def answer(prompt_text: str, images=()):
    """The reply for one prompt: a list (JSON array) or a dict (JSON object)."""
    kind = kind_of(prompt_text)
    # A test can ask for varied voices by putting STUB-TONES in the style box:
    # every third line is then a shout and every fourth a thought.
    varied = "STUB-TONES" in prompt_text

    def tone(i):
        if varied and i % 3 == 0:
            return "shout"
        if varied and i % 4 == 0:
            return "thought"
        return "dialogue"
    if kind == "regions":                   # region_translate_prompt
        n = int(re.search(r"There are (\d+) numbered regions", prompt_text).group(1))
        return [{"id": i, "original": f"原文{i}", "translation": f"LINE {i}",
                 "type": "dialogue", "tone": tone(i)} for i in range(1, n + 1)]
    if kind == "texts":                     # text_translate_prompt
        tail = prompt_text.rsplit("\n\n", 1)[-1]
        try:
            ids = json.loads(tail)
        except json.JSONDecodeError:
            ids = {}
        return [{"id": int(k), "original": v, "translation": f"LINE {k}",
                 "type": "dialogue", "tone": tone(int(k))} for k, v in ids.items()]
    if kind == "crop":                      # crop_translate_prompt (Add tool)
        return [{"original": "テスト", "translation": "CROP TEXT"}]
    if kind == "profile":                   # learn_profile_prompt
        return {
            "glossary": [
                {"term": "ルフィ", "translation": "Luffy", "notes": "captain"},
                {"term": "ゾロ", "translation": "Zoro", "notes": "swordsman"},
            ],
            "honorifics": "Keep -san and -sama as written.",
            "sfx_policy": "Leave sound effects in the art.",
            "style_guide": "Punchy shounen voice. Short lines, lots of !!.",
        }
    if kind == "smart":                     # smart_detect_prompt
        boxes = _balloons(images[0]) if images else []
        return [{"x_pct": round(x, 2), "y_pct": round(y, 2),
                 "width_pct": round(bw, 2), "height_pct": round(bh, 2),
                 "rotation_deg": 0, "original": f"原文{i}",
                 "translation": f"LINE {i}", "type": "dialogue",
                 "tone": tone(i), "in_bubble": True}
                for i, (x, y, bw, bh) in enumerate(boxes, start=1)]
    if kind == "free_text" and os.environ.get("STUB_FREE_TEXT") == "1":
        return [{"x_pct": 30.0, "y_pct": 1.0, "width_pct": 40.0,
                 "height_pct": 4.0, "rotation_deg": 0,
                 "original": "第1話「約束」", "translation": "CHAPTER 1: THE PROMISE",
                 "type": "title"}]
    return []                               # free-text detect, unknown


def remember(kind, model, prompt, images, config=None):
    with _LOG_LOCK:
        LOG.append({"kind": kind, "model": model, "prompt": prompt,
                    "images": images, "config": config})
        del LOG[:-LOG_MAX]


class LogHandler(BaseHTTPRequestHandler):
    """The /__log and /__reset endpoints shared by both stubs."""

    def log_message(self, *a):              # keep the test output quiet
        pass

    def _json(self, status, obj):
        out = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.startswith("/__log"):
            with _LOG_LOCK:
                return self._json(200, list(LOG))
        self._json(404, {"error": {"message": "not found"}})

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def handle_reset(self):
        if self.path.startswith("/__reset"):
            self._read_body()
            with _LOG_LOCK:
                LOG.clear()
            self._json(200, {"ok": True})
            return True
        return False


class Handler(LogHandler):
    def do_POST(self):
        if self.handle_reset():
            return
        body = self._read_body()
        text, images = "", []
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                text += content
            else:
                for block in content or []:
                    if block.get("type") == "text":
                        text += block.get("text", "")
                    elif block.get("type") == "image":
                        images.append((block.get("source") or {}).get("data", ""))
        kind = kind_of(text)
        remember(kind, body.get("model", ""), text, len(images))
        reply = json.dumps(answer(text, images), ensure_ascii=False)
        self._json(200, {
            "id": "msg_stub", "type": "message", "role": "assistant",
            "model": body.get("model", "stub"),
            "content": [{"type": "text", "text": reply}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 50},
        })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8022
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
