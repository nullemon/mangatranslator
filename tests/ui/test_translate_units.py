"""No-server checks for the translate-side helpers the browser suite leans on.

    python tests/ui/test_translate_units.py

Covers: where a long watermark lands (never off the page), the Gemini
endpoint override, the trained-profile fold-in, profile slugs for names with
no latin letters, and the fields a finished page's items carry into the
editor (tone / title caption / source rect / credit).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)              # app.py mounts ./static, ./fonts at import

from _harness import Check  # noqa: E402

c = Check("translate units")

import app  # noqa: E402
from core import profiles, translator  # noqa: E402
from core.pipeline import TranslationPipeline  # noqa: E402

# ── _clear_spot: a mark wider than half the page, corner busy ──
H, W = 2048, 1403
img = np.full((H, W, 3), 255, np.uint8)
keep = np.zeros((H, W), np.uint8)
tw, th = 900, 42
m = max(12, int(W * 0.015))
# Busy along every corner and edge band so the corner candidates all fail and
# the quarter sweep has to decide.
keep[:, W - 200:] = 255
keep[H - 200:, :] = 255
keep[:200, :] = 255
keep[:, :200] = 255
for place in ("br", "bl", "tr", "tl"):
    x, y = app._clear_spot(img, tw, th, place, keep)
    c.ok(0 <= x and x + tw <= W and 0 <= y and y + th <= H,
         f"{place}: a {tw}px mark on a {W}px page stays on the page ({x},{y})")

# Free bottom-right corner: the mark goes exactly there.
keep2 = np.zeros((H, W), np.uint8)
x, y = app._clear_spot(img, 300, 40, "br", keep2)
c.eq((x, y), (W - 300 - m, H - 40 - m), "free corner is used as picked")

# Corner blocked only near the corner itself: the nearest clear spot in the
# bottom-right quarter is chosen, not the top of that quarter.
keep3 = np.zeros((H, W), np.uint8)
keep3[H - 60:, :] = 255                 # a strip of text along the bottom
keep3[:, W - 40:] = 255
keep3[:H - 400, :] = 255                # everything above the last 400px busy
keep3[:H // 2, :W // 2] = 0
keep3[:H // 2, W // 2:] = 0             # ...except the top half (wrong half)
x, y = app._clear_spot(img, 300, 40, "br", keep3)
c.ok(y >= H - 400 and x >= W // 2,
     f"a busy corner moves to the nearest clear spot in its quarter ({x},{y})")

# ── Gemini endpoint override ──
old = os.environ.pop("GEMINI_BASE_URL", None)
c.eq(translator.GeminiTranslator.endpoint("gemini-pro-latest"),
     "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent",
     "default Gemini endpoint unchanged")
os.environ["GEMINI_BASE_URL"] = "http://127.0.0.1:9/"
c.eq(translator.GeminiTranslator.endpoint("m"),
     "http://127.0.0.1:9/v1beta/models/m:generateContent", "GEMINI_BASE_URL override")
if old is None:
    os.environ.pop("GEMINI_BASE_URL", None)
else:
    os.environ["GEMINI_BASE_URL"] = old

# ── profile slugs + fold-in ──
c.eq(profiles.slugify("One PIECE"), "one-piece", "latin slug unchanged")
c.ok(profiles.slugify("ワンピース") != profiles.slugify("呪術廻戦"),
     "two series with non-latin names get different profiles")
c.ok("/" not in profiles.slugify("../../x") and "." not in profiles.slugify("a.b/c"),
     "slug can't escape the profiles folder")
saved = profiles.save(profiles.normalize({
    "name": "QA Units Series", "glossary": [{"term": "ルフィ", "translation": "Luffy"}],
    "style_guide": "Punchy."}))
try:
    got = app._with_profile("Keep -san.", saved["slug"])
    c.ok(got.startswith("SERIES STYLE PROFILE") and "ルフィ → Luffy" in got
         and got.endswith("Keep -san."), "profile folded in ahead of the style box")
    c.eq(app._with_profile("x", ""), "x", "no profile → style box as is")
    c.eq(app._with_profile("x", "no-such-profile"), "x", "unknown profile → style box as is")
finally:
    profiles.delete(saved["slug"])

# ── "Learn from these pages" reaches the learn route, not the save route ──
from starlette.routing import Match  # noqa: E402
scope = {"type": "http", "path": "/api/profile/learn", "method": "POST"}
first = next(r for r in app.app.routes if r.matches(scope)[0] == Match.FULL)
c.eq(getattr(first, "endpoint").__name__, "profile_learn",
     "POST /api/profile/learn is routed to the learner")

# ── Replace With My Watermark ──
site = [{"id": 1, "type": "watermark", "translation": ""}, {"id": 2, "type": "dialogue"}]
plain = [{"id": 2, "type": "dialogue"}]
c.eq(app._mark_to_stamp("@me", False, site), "@me", "replace off: always stamped")
c.eq(app._mark_to_stamp("@me", True, site), "", "replace on + a site mark: placed there, not stamped")
c.eq(app._mark_to_stamp("@me", True, plain), "@me",
     "replace on, no site mark on the page: stamped as usual, not lost")
c.eq(app._mark_to_stamp("", True, site), "", "no watermark typed: nothing")

# ── what a finished page's items carry ──
pipe = TranslationPipeline.__new__(TranslationPipeline)
pipe.detector_name = "CV detector"
res = pipe._result("o.png", "b.png", [
    {"id": 1, "bbox": [0, 0, 9, 9], "translation": "HEY!!", "tone": "shout"},
    {"id": 2, "bbox": [0, 0, 9, 9], "translation": "T", "type": "title",
     "title_caption": True, "src_rect": [1, 2, 3, 4]},
    {"id": 3, "bbox": [0, 0, 9, 9], "translation": "TL: x", "type": "credit",
     "credit": True},
    {"id": 4, "bbox": [0, 0, 9, 9], "translation": "plain"},
])
items = {it["id"]: it for it in res["items"]}
c.eq(items[1].get("tone"), "shout", "the translator's tone reaches the editor")
c.ok(items[2].get("title_caption") and items[2].get("src_rect") == [1, 2, 3, 4],
     "a title caption keeps its treatment for the next re-render")
c.ok(items[3].get("credit"), "the credit stays a credit")
c.ok("tone" not in items[4] and "credit" not in items[4], "no empty extras on plain items")

c.finish()
