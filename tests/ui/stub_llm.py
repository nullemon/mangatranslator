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
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def answer(prompt_text: str) -> list:
    m = re.search(r"There are (\d+) numbered regions", prompt_text)
    if m:                                   # region_translate_prompt
        n = int(m.group(1))
        return [{"id": i, "original": f"原文{i}", "translation": f"LINE {i}",
                 "type": "dialogue", "tone": "dialogue"} for i in range(1, n + 1)]
    if "mapping each speech-bubble id" in prompt_text:   # text_translate_prompt
        tail = prompt_text.rsplit("\n\n", 1)[-1]
        try:
            ids = json.loads(tail)
        except json.JSONDecodeError:
            ids = {}
        return [{"id": int(k), "original": v, "translation": f"LINE {k}",
                 "type": "dialogue", "tone": "dialogue"} for k, v in ids.items()]
    if "CROPPED region" in prompt_text:     # crop_translate_prompt (Add tool)
        return [{"original": "テスト", "translation": "CROP TEXT"}]
    return []                               # free-text detect, smart detect, ...


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):              # keep the test output quiet
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        text = ""
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                text += content
            else:
                for block in content or []:
                    if block.get("type") == "text":
                        text += block.get("text", "")
        reply = json.dumps(answer(text), ensure_ascii=False)
        out = json.dumps({
            "id": "msg_stub", "type": "message", "role": "assistant",
            "model": body.get("model", "stub"),
            "content": [{"type": "text", "text": reply}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8022
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
