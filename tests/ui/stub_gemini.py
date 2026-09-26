"""A stand-in for Gemini's REST generateContent, for running the real pipeline
with the Gemini engine and no key.

Run it, then start the app with GEMINI_BASE_URL pointing at it:

    python tests/ui/stub_gemini.py 8132 &
    GEMINI_BASE_URL=http://127.0.0.1:8132 PORT=8031 python app.py

It answers   POST /v1beta/models/<model>:generateContent   in Gemini's own
response shape (candidates[0].content.parts[0].text, finishReason,
usageMetadata with promptTokenCount / candidatesTokenCount /
thoughtsTokenCount), with the same deterministic JSON the Claude stub gives
("LINE 1", "LINE 2", ... — see stub_llm.py).

The thinking-config rules of the real model families are imitated, so the
translator's THINK_LADDER is exercised the way it is in production:

    *flash* / *flash-lite*        accept {"thinkingBudget": 0}
    gemini-3-* and gemini-pro-latest (Gemini 3 Pro)
                                  reject thinkingBudget ("Budget 0 is invalid.
                                  This model only works in thinking mode."),
                                  accept {"thinkingLevel": ...}
    gemini-2.5-pro                rejects budget 0 and thinkingLevel
                                  ("Invalid argument"), accepts budget >= 128
    gemini-stub-uncapped          rejects every thinking config

Keys and models that make it fail the way Google does:

    key "bad-key"                 400 "API key not valid. Please pass a valid API key."
    key "quota-key"               429 RESOURCE_EXHAUSTED
    key "empty-key"               200 with no text (finishReason MAX_TOKENS)
    model "gemini-retired"        404 "... is not found for API version v1beta"

GET /__log returns every request (kind, model, prompt, image count and the
generationConfig it was sent with); POST /__reset clears it.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stub_llm import LogHandler, answer, kind_of, remember  # noqa: E402

from http.server import ThreadingHTTPServer  # noqa: E402


def thinking_error(model: str, think):
    """The 400 message this model gives for this thinking config, or None."""
    m = model.lower()
    if think is None:
        return None
    if m == "gemini-stub-uncapped":
        return "Invalid argument: thinking is not supported by this model."
    if "flash" in m:
        return None
    if m.startswith("gemini-3") or m == "gemini-pro-latest":
        if "thinkingBudget" in think:
            return ("Budget 0 is invalid. This model only works in thinking "
                    "mode.")
        return None
    if "pro" in m:                               # gemini-2.5-pro
        if "thinkingLevel" in think:
            return "Request contains an invalid argument."
        if int(think.get("thinkingBudget", 0)) < 128:
            return ("Budget 0 is invalid. This model only works in thinking "
                    "mode.")
        return None
    return None


class Handler(LogHandler):
    def _err(self, status, msg, reason="INVALID_ARGUMENT"):
        self._json(status, {"error": {"code": status, "message": msg,
                                      "status": reason}})

    def do_POST(self):
        if self.handle_reset():
            return
        m = re.match(r"^/v1beta/models/([^/:]+):generateContent", self.path)
        body = self._read_body()
        if not m:
            return self._err(404, f"unknown path {self.path}", "NOT_FOUND")
        model = m.group(1)
        key = self.headers.get("x-goog-api-key", "")
        gc = body.get("generationConfig") or {}
        think = gc.get("thinkingConfig")
        text, images = "", []
        for c in body.get("contents", []):
            for part in c.get("parts", []):
                if "text" in part:
                    text += part["text"]
                elif "inlineData" in part:
                    images.append(part["inlineData"].get("data", ""))
        kind = kind_of(text)
        remember(kind, model, text, len(images), gc)

        if not key or key == "bad-key":
            return self._err(400, "API key not valid. Please pass a valid API key.")
        if key == "quota-key":
            return self._err(429, "You exceeded your current quota.",
                             "RESOURCE_EXHAUSTED")
        if model == "gemini-retired":
            return self._err(404, f"models/{model} is not found for API version "
                                  "v1beta, or is not supported for "
                                  "generateContent.", "NOT_FOUND")
        why = thinking_error(model, think)
        if why:
            return self._err(400, why)

        thoughts = 0 if (think or {}).get("thinkingBudget") == 0 else 64
        if key == "empty-key":
            return self._json(200, {
                "candidates": [{"content": {"parts": [], "role": "model"},
                                "finishReason": "MAX_TOKENS", "index": 0}],
                "usageMetadata": {"promptTokenCount": 1200,
                                  "thoughtsTokenCount": 65536,
                                  "totalTokenCount": 66736},
                "modelVersion": model})
        reply = json.dumps(answer(text, images), ensure_ascii=False)
        self._json(200, {
            "candidates": [{
                "content": {"parts": [{"text": reply}], "role": "model"},
                "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1200,
                              "candidatesTokenCount": 80,
                              "thoughtsTokenCount": thoughts,
                              "totalTokenCount": 1280 + thoughts},
            "modelVersion": model,
        })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8132
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
