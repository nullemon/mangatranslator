"""HTTP-level test of the editor endpoints the add tools call, with no API
key in play — the answers the browser relies on to explain a failed read.

Run:  /path/to/python tests/ui/test_editor_endpoints.py   (skips if no server)
"""
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_harness import BASE, PAGE_IMG, Results, skip_unless_server


def clean_page(path):
    r = requests.post(f"{BASE}/api/translate",
                      files={"file": (os.path.basename(path), open(path, "rb"), "image/jpeg")},
                      data={"clean_only": "true", "finish": "clean"}, timeout=60)
    r.raise_for_status()
    tid = r.json()["task_id"]
    t0 = time.time()
    while time.time() - t0 < 400:
        s = requests.get(f"{BASE}/api/status/{tid}", timeout=30).json()
        if s["status"] in ("done", "error"):
            return tid, s
        time.sleep(2)
    raise TimeoutError("clean did not finish")


def main():
    skip_unless_server()
    R = Results()
    tid, s = clean_page(PAGE_IMG)
    R.check("clean-only page finished", s["status"] == "done", s.get("message"))

    def post(path, body):
        r = requests.post(f"{BASE}{path}", json=body, timeout=300)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"raw": r.text}

    # Claude engine, no key: a message a person can act on, not "api_key is required".
    code, j = post(f"/api/ocr-translate/{tid}", {"bbox": [100, 100, 200, 200], "provider": "claude", "api_key": ""})
    R.check("ocr-translate no key -> 400 with a human message", code == 400 and "key" in j.get("detail", "").lower()
            and j.get("detail") != "api_key is required", str(j))
    code, j = post("/api/translate-text", {"text": "こんにちは", "provider": "claude", "api_key": ""})
    R.check("translate-text no key -> 400 with a human message", code == 400 and "Settings" in j.get("detail", ""), str(j))
    code, j = post(f"/api/rescan/{tid}", {"provider": "claude", "api_key": ""})
    R.check("rescan no key -> 400 with a human message", code == 400 and "Settings" in j.get("detail", ""), str(j))

    # Offline engine on a box without manga-ocr: say so.
    code, j = post(f"/api/ocr-translate/{tid}", {"bbox": [100, 100, 200, 200], "provider": "local", "api_key": ""})
    R.check("ocr-translate offline -> empty read carries a reason", code == 200 and not j.get("translation")
            and "manga-ocr" in j.get("reason", ""), str(j))
    code, j = post(f"/api/ocr-translate/{tid}", {"bbox": [100, 100, 200, 200], "provider": "local", "api_key": "",
                                                 "poly": [[100, 100], [300, 100], [300, 300]]})
    R.check("ocr-translate offline with poly -> same, no crash", code == 200 and "reason" in j, str(j))
    code, j = post(f"/api/rescan/{tid}", {"provider": "local", "api_key": "", "model": "auto"})
    R.check("rescan offline -> detector ran, notice explains OCR is missing",
            code == 200 and j.get("added_count") == 0 and "manga-ocr" in j.get("notice", "")
            and "Smart Detection" not in j.get("notice", ""), str(j)[:300])

    # Re-render with a typed line: the text lands, the rest of the page doesn't move.
    import cv2
    import numpy as np
    from add_harness import changed_fraction, fetch_result, ink_fraction
    before = fetch_result(tid, 1)
    body = {"excluded": [], "erased": [], "edits": {}, "font_scale": 1.0, "fonts": {"m1": "Bangers-Regular.ttf"},
            "added": [{"id": "m1", "bbox": [70, 700, 120, 120], "poly": None, "original": "テスト",
                       "translation": "HELLO"}]}
    code, j = post(f"/api/rerender/{tid}", body)
    R.check("rerender with a typed line -> 200, echoed as placed",
            code == 200 and j.get("added") and j["added"][0].get("placed") and j["added"][0].get("original") == "テスト",
            str(j)[:300])
    after = fetch_result(tid, 2)
    R.check("typed line drawn in its box", ink_fraction(after, 70, 700, 120, 120) > 0.02)
    R.check("far region untouched", changed_fraction(before, after, 620, 1300, 300, 150) < 0.01)
    # A font that isn't installed must be ignored, never a path traversal.
    body["fonts"] = {"m1": "../app.py"}
    code, j = post(f"/api/rerender/{tid}", body)
    R.check("bogus font name is ignored", code == 200, str(j)[:200])
    sys.exit(R.summary())


if __name__ == "__main__":
    main()
